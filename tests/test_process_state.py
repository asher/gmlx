"""Each test hands the process back as it found it.

The first test runs CLI entry points that change the process for their own
run, as ``gmlx distill`` does when it turns TF32 off, and the MLX setters
that library entry points call. The second, which runs next, must find
everything as the first found it. The last check runs at the end of the
session (tests/conftest.py moves it there): the environment still holds
the session's MLX_ENABLE_TF32, and float32 matmuls in this process round as
in a fresh process started from the session environment. The tests
before it run the harness itself in a fresh pytest process.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx_kquant as kq
import pytest

import process_state

_BEFORE: dict = {}
# Set for each test by pytest and by the conftest's own fixtures.
_PER_TEST = ("PYTEST_CURRENT_TEST", "XDG_DATA_HOME")


def _observe() -> dict:
    return {
        "env": {k: v for k, v in os.environ.items() if k not in _PER_TEST},
        "formatwarning": warnings.formatwarning,
        "device": mx.default_device(),
        "stream": mx.default_stream(mx.default_device()),
        "rng": mx.random.state[0].tolist(),
        "set_wired_limit": mx.set_wired_limit,
        "cb_caps": process_state.cb_caps(),
    }


def _run(argv) -> int:
    from gmlx.commands import cli

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return cli.umbrella_main(argv)


def test_a_cli_run_changes_the_process():
    _BEFORE.clear()
    _BEFORE.update(_observe())
    # gmlx distill turns TF32 off in its own process; --cpu moves it to the
    # CPU device. Without --teacher it refuses after both.
    assert _run(["distill", "cache", "--cpu"]) == 2
    assert os.environ["MLX_ENABLE_TF32"] == "0"
    assert mx.default_device() == mx.cpu
    assert warnings.formatwarning is not _BEFORE["formatwarning"]
    # A streamed run lifts the command buffer caps before it parses.
    _run(["run", "--stream-cpu", "--no-such-flag"])
    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == "400"
    # The setters library entry points call.
    mx.random.seed(5)
    _BEFORE["cache_limit"] = mx.set_cache_limit(123 << 20)
    _BEFORE["memory_limit"] = mx.set_memory_limit(1 << 40)
    mx.set_memory_limit(_BEFORE["memory_limit"] - (1 << 20))
    mx.set_default_stream(mx.new_stream(mx.default_device()))
    mx.set_wired_limit = lambda *_a, **_k: 0
    if _BEFORE["cb_caps"] is not None:
        kq.set_cb_caps(7, 7)


def test_b_the_next_test_finds_the_process_as_it_was():
    if not _BEFORE:
        pytest.skip("runs after test_a_cli_run_changes_the_process")
    now = _observe()
    for key in ("env", "device", "stream", "rng", "cb_caps"):
        assert now[key] == _BEFORE[key], key
    assert now["formatwarning"] is _BEFORE["formatwarning"]
    assert now["set_wired_limit"] is _BEFORE["set_wired_limit"]
    assert mx.set_cache_limit(_BEFORE["cache_limit"]) == _BEFORE["cache_limit"]
    assert mx.set_memory_limit(_BEFORE["memory_limit"]) == _BEFORE["memory_limit"]


def _phase_caps():
    from gmlx.serve import cb_phase

    caps = process_state.cb_caps()
    if caps is None or caps == cb_phase.COARSE:
        pytest.skip("needs command buffer caps apart from the decode caps")
    return cb_phase


def test_c_a_serve_step_flips_the_phase():
    cb_phase = _phase_caps()
    cb_phase.flip("decode")
    assert process_state.cb_caps() == cb_phase.COARSE


def test_d_the_next_decode_step_sets_the_decode_caps():
    # The caps went back after the last test. cb_phase must not still name
    # the decode phase, or this flip returns early and decode runs with the
    # caps the last test found.
    cb_phase = _phase_caps()
    cb_phase.flip("decode")
    assert process_state.cb_caps() == cb_phase.COARSE


# The Python-side copies of the MLX settings. Each pair changes one group
# the way a library entry point does and checks that the next test finds
# it as the first found it.
_MIRRORS: dict = {}
# A batch generator that outlives its test, as a module-scoped server's
# does.
_GEN = SimpleNamespace()


def _wired_limit_contexts() -> list:
    # importlib, since import mlx_lm.generate binds the function mlx_lm
    # re-exports under that name.
    return [getattr(importlib.import_module(m), "wired_limit", None)
            for m in ("gmlx._vendor.mlx_lm.generate", "gmlx._vendor.mlx_lm.utils")]


def test_e_a_cpu_device_setup_rebinds_the_wired_limit_context():
    from gmlx.stream.wired_limit import configure_cpu_device

    _MIRRORS["wired_limit"] = _wired_limit_contexts()
    configure_cpu_device()
    assert _wired_limit_contexts() != _MIRRORS["wired_limit"]


def test_f_the_next_test_finds_the_stock_wired_limit_context():
    if "wired_limit" not in _MIRRORS:
        pytest.skip("runs after test_e")
    assert _wired_limit_contexts() == _MIRRORS["wired_limit"]


def test_g_a_stream_cpu_setup_rebinds_device_info():
    _MIRRORS["device_info"] = mx.device_info
    # configure_stream_cpu keeps the GPU's answer for the CPU device.
    gpu_info = {"max_recommended_working_set_size": 1 << 30}
    mx.device_info = lambda: gpu_info


def test_h_the_next_test_finds_the_stock_device_info():
    if "device_info" not in _MIRRORS:
        pytest.skip("runs after test_g")
    assert mx.device_info is _MIRRORS["device_info"]


def test_i_a_server_notes_its_cache_limit_and_arms_the_governor(monkeypatch):
    from gmlx.gen import prefill_decay
    from gmlx.serve import governor, memory

    monkeypatch.delenv("GMLX_CACHE_LIMIT_GB", raising=False)
    _MIRRORS["noted_cache_limit"] = prefill_decay._NOTED_CACHE_LIMIT
    memory.apply_cache_limit(SimpleNamespace(cache_limit_gb=3, models={}))
    st = governor._state(_GEN)
    governor._arm_throttle(_GEN, st, 64e9, 0.1)
    assert st.saved_mem_limit is not None
    assert st.saved_cache_limit is not None
    assert prefill_decay._NOTED_CACHE_LIMIT == 3 * (1 << 30)


def test_j_the_next_test_finds_the_governor_unarmed():
    if "noted_cache_limit" not in _MIRRORS:
        pytest.skip("runs after test_i")
    from gmlx.gen import prefill_decay
    from gmlx.serve import governor

    # The MLX limits went back, so a governor that still held the limits
    # it saved would take its throttle for armed and never arm it again.
    st = governor._state(_GEN)
    assert (st.saved_mem_limit, st.saved_cache_limit) == (None, None)
    assert prefill_decay._NOTED_CACHE_LIMIT == _MIRRORS["noted_cache_limit"]



# The harness itself, run in a fresh pytest process on the CPU device over
# copies of the conftest, this file and the modules of this folder they
# import.
_TESTS = Path(__file__).resolve().parent
_END_OF_RUN = "test_process_state.py::test_the_run_ends_on_production_numerics"


def _inner_files() -> list[str]:
    """The conftest, this file, and each module of this folder that one of
    them imports, at any depth, by file name."""
    import ast

    todo, found = ["conftest.py", Path(__file__).name], set()
    while todo:
        name = todo.pop()
        if name in found:
            continue
        found.add(name)
        for node in ast.walk(ast.parse((_TESTS / name).read_text())):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                mods = [node.module or ""]
            else:
                continue
            todo += [f for f in (m.split(".")[0] + ".py" for m in mods)
                     if (_TESTS / f).is_file()]
    return sorted(found)


def _inner_tree(tmp_path) -> Path:
    root = tmp_path / "inner"
    root.mkdir()
    for name in _inner_files():
        shutil.copyfile(_TESTS / name, root / name)
    return root


def test_the_inner_tree_holds_the_modules_its_files_import():
    assert {"conftest.py", "hf_cache_guard.py", "process_state.py",
            "test_process_state.py"} <= set(_inner_files())


def _inner_pytest(root: Path, *args, **env):
    run_env = {k: v for k, v in process_state.SESSION["env"].items()
               if not k.startswith("PYTEST_")
               and k != "GMLX_TEST_STATE_REPORT"}
    run_env.update(KQUANT_FORCE_CPU="1", PYTHONPATH=str(_TESTS.parent), **env)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short",
         "-o", f"cache_dir={root / '.cache'}", *args],
        cwd=root, env=run_env, capture_output=True, text=True, timeout=600)


_EARLY_FIXTURE = '''

@pytest.fixture(autouse=True)
def _a_fixture_named_before_the_others(request):
    first = request.node.name == "test_first"
    if first:
        os.environ["GMLX_INNER_SETUP_LEAK"] = "1"
    yield
    if first:
        os.environ["GMLX_INNER_TEARDOWN_LEAK"] = "1"
'''

_ORDER_TESTS = '''
import os


def test_first():
    pass


def test_second():
    assert "GMLX_INNER_SETUP_LEAK" not in os.environ
    assert "GMLX_INNER_TEARDOWN_LEAK" not in os.environ
'''

_SCOPED_TESTS = '''
import os

import pytest


@pytest.fixture(scope="module")
def module_env():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GMLX_INNER_MODULE_ENV", "1")
        yield


@pytest.mark.parametrize("i", [1, 2, 3])
def test_module_env_stays_for_the_module(module_env, i):
    assert os.environ.get("GMLX_INNER_MODULE_ENV") == "1"
'''

_AFTER_SCOPED = '''
import os


def test_module_env_goes_with_its_module():
    assert "GMLX_INNER_MODULE_ENV" not in os.environ
'''


def test_restore_wraps_every_function_fixture_whatever_its_name(tmp_path):
    # A conftest fixture named before the others leaks at its setup and its
    # teardown, and the next test must find neither. A module-scoped
    # fixture's environment stays for its module and leaves with it.
    root = _inner_tree(tmp_path)
    with open(root / "conftest.py", "a") as f:
        f.write(_EARLY_FIXTURE)
    (root / "test_order.py").write_text(_ORDER_TESTS)
    (root / "test_scoped.py").write_text(_SCOPED_TESTS)
    (root / "test_scoped_after.py").write_text(_AFTER_SCOPED)
    r = _inner_pytest(root, "test_order.py", "test_scoped.py",
                      "test_scoped_after.py")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]


_KVARN_TESTS = '''
import pytest


@pytest.mark.needs_kvarn_ops
def test_needs_kvarn_ops():
    pass
'''


def test_a_test_skipped_in_setup_tears_down_clean(tmp_path):
    # The CPU device has no kvarn kernels, so the conftest skips this test
    # in its setup hook. The other plugins' setup wrappers run before the
    # skip, so their teardown finds what they set.
    root = _inner_tree(tmp_path)
    (root / "test_kvarn.py").write_text(_KVARN_TESTS)
    r = _inner_pytest(root, "-p", "no:warnings", "test_kvarn.py")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "1 skipped" in r.stdout, r.stdout[-4000:]


def test_the_end_of_run_check_stays_last_under_ff_and_nf(tmp_path):
    # The check failed last time and its file is the newest, so --ff and
    # --nf would each move it to the front.
    root = _inner_tree(tmp_path)
    (root / "test_other.py").write_text("def test_other():\n    pass\n")
    cache = root / ".cache" / "v" / "cache"
    cache.mkdir(parents=True)
    (cache / "lastfailed").write_text(json.dumps({_END_OF_RUN: True}))
    (cache / "nodeids").write_text(
        json.dumps(["test_other.py::test_other", _END_OF_RUN]))
    (root / "test_new.py").write_text("def test_new():\n    pass\n")
    os.utime(root / "test_other.py", (1, 1))
    for flag in ("--ff", "--nf"):
        r = _inner_pytest(root, flag, "--collect-only", "test_other.py",
                          "test_new.py", _END_OF_RUN)
        ids = [line for line in r.stdout.splitlines() if "::" in line]
        assert len(ids) == 3 and ids[-1] == _END_OF_RUN, (flag, r.stdout)


_KQ_TESTS = '''
def test_sets_a_key_kq_keeps(monkeypatch):
    monkeypatch.setenv("KQ_VERIFY_EXT", "0")


def test_sets_a_key_kq_reads_at_each_call(monkeypatch):
    monkeypatch.setenv("KQ_GQA_NAX", "0")
'''


def test_the_end_of_run_check_names_a_kq_key_a_test_set(tmp_path):
    root = _inner_tree(tmp_path)
    (root / "test_kq.py").write_text(_KQ_TESTS)
    r = _inner_pytest(root, "test_kq.py", _END_OF_RUN)
    assert r.returncode == 1, r.stdout[-4000:] + r.stderr[-2000:]
    assert "1 failed, 2 passed" in r.stdout, r.stdout[-4000:]
    touched = "[('test_kq.py::test_sets_a_key_kq_keeps', 'KQ_VERIFY_EXT')]"
    assert touched in r.stdout, r.stdout[-4000:]


_KQ_FIXTURE_TESTS = '''
import pytest


@pytest.fixture(scope="module")
def kq_module_env():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KQ_VERIFY_EXT", "0")
        yield


def test_one(kq_module_env):
    pass


def test_two(kq_module_env):
    pass
'''


def test_the_end_of_run_check_names_a_kq_key_a_wider_fixture_set(tmp_path):
    # The key is set before either test's capture, and its teardown
    # removes it before the check.
    root = _inner_tree(tmp_path)
    (root / "test_kq_fixture.py").write_text(_KQ_FIXTURE_TESTS)
    r = _inner_pytest(root, "test_kq_fixture.py", _END_OF_RUN)
    assert r.returncode == 1, r.stdout[-4000:] + r.stderr[-2000:]
    assert "1 failed, 2 passed" in r.stdout, r.stdout[-4000:]
    touched = ("[('test_kq_fixture.py::kq_module_env (module fixture)', "
               "'KQ_VERIFY_EXT')]")
    assert touched in r.stdout, r.stdout[-4000:]


def test_the_end_of_run_check_names_a_kq_key_set_outside_any_test(tmp_path):
    # Set as the module is collected, so every capture starts with it.
    root = _inner_tree(tmp_path)
    (root / "test_kq_import.py").write_text(
        'import os\n\nos.environ["KQ_VERIFY_EXT"] = "0"\n\n\n'
        "def test_one():\n    pass\n")
    r = _inner_pytest(root, "test_kq_import.py", _END_OF_RUN)
    assert r.returncode == 1, r.stdout[-4000:] + r.stderr[-2000:]
    assert "1 failed, 1 passed" in r.stdout, r.stdout[-4000:]
    assert "[('the end of the run', 'KQ_VERIFY_EXT')]" in r.stdout, (
        r.stdout[-4000:])


_CHANGE_TESTS = '''
import os
import warnings

import mlx.core as mx


def test_changes_the_process():
    os.environ["GMLX_INNER_CHANGE"] = "1"
    warnings.formatwarning = lambda *a, **k: ""
    mx.set_cache_limit(123 << 20)


def test_leaves_the_process_alone():
    pass
'''


def test_the_state_report_names_each_field_restore_changed(tmp_path):
    root = _inner_tree(tmp_path)
    (root / "test_change.py").write_text(_CHANGE_TESTS)
    r = _inner_pytest(root, "test_change.py", GMLX_TEST_STATE_REPORT="1")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "process state restored" in r.stdout, r.stdout[-4000:]
    assert ("test_change.py::test_changes_the_process: env GMLX_INNER_CHANGE, "
            "warnings.formatwarning, cache limit") in r.stdout, r.stdout[-4000:]
    assert "test_leaves_the_process_alone" not in r.stdout, r.stdout[-4000:]
    quiet = _inner_pytest(root, "test_change.py")
    assert quiet.returncode == 0, quiet.stdout[-4000:]
    assert "process state restored" not in quiet.stdout, quiet.stdout[-4000:]


_GEMM = r"""
import os

import mlx.core as mx


def gemm_error():
    a = mx.random.normal((256, 256), key=mx.random.key(0))
    b = mx.random.normal((256, 256), key=mx.random.key(1))
    ref = mx.matmul(a, b, stream=mx.cpu)
    return mx.abs(mx.matmul(a, b) - ref).max().item()


if __name__ == "__main__":
    if os.environ.get("KQUANT_FORCE_CPU"):
        mx.set_default_device(mx.cpu)
    print(gemm_error())
"""


def test_the_run_ends_on_production_numerics():
    process_state.note_kq_env_left()
    assert not process_state.KQ_TOUCHED, (
        "mlx-kquant keeps these KQ_* keys from its first read, so each test "
        f"or fixture decided one for the rest of the run: "
        f"{process_state.KQ_TOUCHED}")
    start = process_state.SESSION
    assert os.environ.get("MLX_ENABLE_TF32") == start["env"].get(
        "MLX_ENABLE_TF32")
    assert str(mx.default_device()) == start["device"]
    ns: dict = {}
    exec(_GEMM, ns)
    here = ns["gemm_error"]()
    r = subprocess.run([sys.executable, "-c", _GEMM], env=start["env"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    fresh = float(r.stdout.strip().splitlines()[-1])
    # TF32 rounding leaves about 5e-2 on these products, float32 about 5e-5.
    assert (here > 1e-3) == (fresh > 1e-3), (here, fresh)
