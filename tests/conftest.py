"""Shared pytest fixtures.

The logic tests (remap / config-synth / arch-gate / transforms / preflight) are
pure Python and need no models - they run anywhere, including CI.

The ``integration`` tests need real GGUFs, and the parity tests additionally
need a ``llama.cpp`` completion binary. To keep this package free of any machine-
specific paths, those locations come from the environment and the tests skip when
unset:

  KQUANT_TEST_GGUF_DIR   directory searched recursively for ``*.gguf`` test models
  KQUANT_LLAMACPP_BIN    path to a llama.cpp ``llama-completion`` (or ``llama-cli``)
                         binary, used as the reference for greedy/long-context parity
"""

from __future__ import annotations

import os
from pathlib import Path

import sys

import pytest

# The suite is split into per-subsystem directories, but tests import shared
# helpers from sibling directories by module basename (pytest prepend mode).
# Make every test directory importable up front so collection order and
# single-file runs cannot break those imports.
_TESTS_DIR = Path(__file__).resolve().parent
for _sub in sorted(_TESTS_DIR.iterdir()):
    if _sub.is_dir() and any(_sub.glob("test_*.py")):
        _p = str(_sub)
        if _p not in sys.path:
            sys.path.insert(0, _p)
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# gmlx carries mlx-vlm and mlx-lm under gmlx._vendor. Hide any installed
# stock copy, so a stray import fails here as it does without one.
import stock_blocker  # noqa: E402

stock_blocker.install()


def pytest_configure(config):
    # It lists the blocked imports and fails the run on any.
    config.pluginmanager.register(stock_blocker, "stock_blocker")

    # gmlx first, as the console entry point does. The import loads no MLX.
    import gmlx  # noqa: F401

    # The logic tests touch only small mx array ops (transforms,
    # config-synth instantiation) and never dispatch a kquant kernel, so they
    # can run on the CPU device. Set KQUANT_FORCE_CPU=1 in environments without
    # a usable Metal GPU (e.g. CI) to keep them off the GPU path.
    if os.environ.get("KQUANT_FORCE_CPU"):
        import mlx.core as mx

        mx.set_default_device(mx.cpu)

    # Every test hands the process back as it found it (process_state.py).
    import process_state

    process_state.start_session()

    # Track every daemon read pool so the autouse fixture below can shut
    # down the ones a test leaks. Leaked pools park their workers in
    # queue.get forever; past ~100 live threads faulthandler truncates
    # fatal-error dumps, cutting off the main thread's stack (the one that
    # names the crashing test).
    import gmlx.stream.feeder_common as feeder_common

    orig_init = feeder_common.DaemonPool.__init__

    def tracking_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _read_pools.append(self)

    feeder_common.DaemonPool.__init__ = tracking_init


_read_pools: list = []


# Every test hands the process back as it found it (process_state.py).
# pytest sets up a test's fixtures in scope order, the wider scopes first.
# The capture runs as the first function-scoped fixture sets up, whatever
# the fixture names, and the restore runs as an item finalizer registered
# then: after the teardown of every function-scoped fixture, and before
# the teardown of any wider one. Restore covers the test and its
# function-scoped fixtures. Class-, module- and session-scoped fixtures
# set up outside the capture, so what they change stays for the tests
# that share them, and what their own teardown does not undo persists
# until the end-of-run check.
_awaiting_capture = None


_CAPTURED = pytest.StashKey[dict]()
# (test node id, fields restore changed), kept under GMLX_TEST_STATE_REPORT=1.
_restored: list = []


def _hand_back_process_state(item) -> None:
    import process_state

    state = item.stash[_CAPTURED] = process_state.capture()

    def restore():
        process_state.note_kq_env(state, item.nodeid)
        changed = process_state.restore(state)
        if changed and _state_report():
            _restored.append((item.nodeid, changed))

    item.addfinalizer(restore)


def _state_report() -> bool:
    import process_state

    return process_state.SESSION["env"].get("GMLX_TEST_STATE_REPORT") == "1"


def pytest_terminal_summary(terminalreporter):
    if not _state_report():
        return
    terminalreporter.section("process state restored")
    if not _restored:
        terminalreporter.write_line("no test changed the process state")
    for nodeid, changed in _restored:
        terminalreporter.write_line(f"{nodeid}: {', '.join(changed)}")


# Its own name, so the plain hook that skips the kvarn tests stays a plain
# hook: a skip raised there runs inside every plugin's setup wrapper, whose
# teardown then finds what its setup left.
@pytest.hookimpl(wrapper=True, tryfirst=True, specname="pytest_runtest_setup")
def pytest_runtest_setup_capture(item):
    global _awaiting_capture
    _awaiting_capture = item
    try:
        result = yield
    except BaseException:
        # Setup stopped. A capture it made still restores at teardown.
        _awaiting_capture = None
        raise
    if _awaiting_capture is item:
        # A test with no function-scoped fixture.
        _awaiting_capture = None
        _hand_back_process_state(item)
    return result


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    global _awaiting_capture
    item = _awaiting_capture
    if item is not None and fixturedef.scope == "function":
        _awaiting_capture = None
        _hand_back_process_state(item)
    if fixturedef.scope == "function":
        return (yield)
    # A wider fixture sets up and tears down outside the capture of every
    # test, so its KQ_* keys are recorded under its own name, around its
    # setup here and around its teardown below.
    import process_state

    before = {"env": dict(os.environ)}
    try:
        return (yield)
    finally:
        process_state.note_kq_env(before, _fixture_name(fixturedef))
        # Added after the fixture's own teardown, so it runs first.
        fixturedef.addfinalizer(
            lambda: _before_teardown.__setitem__(
                fixturedef, {"env": dict(os.environ)}))


# The environment as each wider fixture's teardown starts.
_before_teardown: dict = {}


def _fixture_name(fixturedef) -> str:
    where = f"{fixturedef.baseid}::" if fixturedef.baseid else ""
    return f"{where}{fixturedef.argname} ({fixturedef.scope} fixture)"


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_post_finalizer(fixturedef, request):
    before = _before_teardown.pop(fixturedef, None)
    if before is not None:
        import process_state

        process_state.note_kq_env(before, _fixture_name(fixturedef))


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    # A KQ_* key set by a fixture shows before the call, and one the test
    # sets through monkeypatch until its teardown.
    import process_state

    state = item.stash.get(_CAPTURED, None)
    if state is not None:
        process_state.note_kq_env(state, item.nodeid)
    try:
        return (yield)
    finally:
        if state is not None:
            process_state.note_kq_env(state, item.nodeid)


_END_OF_RUN = "test_process_state.py::test_the_run_ends_on_production_numerics"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_finish(session):
    # The production-numerics check reads the state the whole run left, so
    # it runs last. The --ff, --nf and --lf reorders are wrappers around
    # pytest_collection_modifyitems that run after every plain hook, a
    # trylast one included, so the move waits until that hook returns.
    items = session.items
    last = [i for i in items if i.nodeid.endswith(_END_OF_RUN)]
    if last:
        items[:] = [i for i in items if i not in last] + last


@pytest.fixture(autouse=True)
def _shutdown_read_pools():
    n = len(_read_pools)
    yield
    # Only the pools this test created. A module or session fixture is set
    # up before this one, so the pools of a model it shares stay open for
    # the next test, and its own teardown closes them.
    # wait=False: a deliberately wedged worker (feeder wedge tests) never
    # takes its poison pill; joining it would hang the suite.
    while len(_read_pools) > n:
        _read_pools.pop().shutdown(wait=False)


@pytest.fixture(scope="session")
def gguf_dir() -> Path:
    d = os.environ.get("KQUANT_TEST_GGUF_DIR")
    if not d:
        pytest.skip("set KQUANT_TEST_GGUF_DIR to a dir of *.gguf to run integration tests")
    p = Path(d).expanduser()
    if not p.is_dir():
        pytest.skip(f"KQUANT_TEST_GGUF_DIR={d!r} is not a directory")
    return p


@pytest.fixture(scope="session")
def llamacpp_bin() -> str:
    b = os.environ.get("KQUANT_LLAMACPP_BIN")
    if not b:
        pytest.skip("set KQUANT_LLAMACPP_BIN to a llama.cpp completion binary for parity tests")
    if not Path(b).expanduser().is_file():
        pytest.skip(f"KQUANT_LLAMACPP_BIN={b!r} not found")
    b = str(Path(b).expanduser())
    # Parity tests drive one-shot completions via -no-cnv. Newer llama.cpp
    # builds split that out of llama-cli (interactive-only) into
    # llama-completion; catch the wrong binary here, not 20 minutes into a run.
    import subprocess
    try:
        help_text = subprocess.run(
            [b, "--help"], capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception as e:
        pytest.skip(f"KQUANT_LLAMACPP_BIN={b!r} --help failed: {e}")
    if "-no-cnv" not in help_text:
        pytest.fail(
            f"KQUANT_LLAMACPP_BIN={b!r} does not support -no-cnv (interactive-"
            f"only llama-cli?) - point it at llama-completion instead")
    return b


@pytest.fixture(scope="session")
def gguf_index(gguf_dir) -> dict:
    """Map ``general.architecture`` -> list of GGUF paths found under the test dir.

    Reads only the header of each shard-0 GGUF (cheap). Split shards collapse to
    their first shard. Lets a parity test ask for "a gemma2 model" without hard-
    coding filenames.
    """
    from gmlx.load.headerscan import scan_gguf

    index: dict[str, list[str]] = {}
    for path in sorted(gguf_dir.rglob("*.gguf")):
        name = path.name
        # skip non-first split shards and multimodal projectors
        if "-of-" in name and not ("00001-of-" in name or "-00001-of" in name):
            continue
        if "mmproj" in name:
            continue
        try:
            kv = scan_gguf(str(path), include_tensors=False).kv
            arch = kv.get("general.architecture")
            if arch is None:
                continue
            # Mixtral ships under general.architecture='llama' + an expert count;
            # index it as 'mixtral' so a parity test can select it distinctly from
            # a dense Llama (the loader still routes model_type=mixtral via the
            # same metadata signal).
            if arch == "llama" and int(kv.get("llama.expert_count") or 0) > 0:
                arch = "mixtral"
        except Exception:
            continue
        index.setdefault(arch, []).append(str(path))
    if not index:
        pytest.skip(f"no readable *.gguf found under {gguf_dir}")
    # Deterministic pick order: non-embedding/rerank names first, then smallest.
    # Generation-parity tests asking for e.g. "qwen3" must not land on an
    # embedding checkpoint that happens to sort or size first.
    for paths in index.values():
        paths.sort(key=lambda s: (
            ("embed" in Path(s).name.lower() or "rerank" in Path(s).name.lower()),
            Path(s).stat().st_size,
        ))
    return index


def require_arch(gguf_index: dict, arch: str) -> str:
    """Return one GGUF path for ``arch`` from the index, or skip the test."""
    paths = gguf_index.get(arch)
    if not paths:
        pytest.skip(f"no {arch!r} GGUF under KQUANT_TEST_GGUF_DIR (have: {sorted(gguf_index)})")
    return paths[0]


@pytest.fixture(autouse=True)
def _isolated_xdg_data(tmp_path_factory, monkeypatch):
    # Chat autosave writes sessions under $XDG_DATA_HOME/gmlx/chats; keep
    # every test away from the user's real data dir.
    monkeypatch.setenv(
        "XDG_DATA_HOME", str(tmp_path_factory.mktemp("xdg-data"))
    )


@pytest.fixture(autouse=True)
def _isolated_server_max_tokens(monkeypatch):
    # `_serve` sets MLX_VLM_MAX_TOKENS in the process environment; restore it
    # after each test so one server test cannot change another's default.
    monkeypatch.setenv("MLX_VLM_MAX_TOKENS", "0")
    monkeypatch.delenv("MLX_VLM_MAX_TOKENS")


@pytest.fixture(autouse=True)
def _no_live_server(monkeypatch):
    # chat's automatic --server gate probes the box's managed server; a
    # live server on the dev machine must not flip test behavior. Tests
    # that want the probe up re-patch it locally.
    import gmlx.commands.launch as _launch
    monkeypatch.setattr(_launch, "_server_ready",
                        lambda base_url, api_key=None: False)


def pytest_runtest_setup(item):
    ops = item.get_closest_marker("needs_kvarn_ops") is not None
    row_ends = item.get_closest_marker("needs_kvarn_row_ends") is not None
    if not (ops or row_ends):
        return
    from gmlx.cache import kvarn_sdpa

    reason = kvarn_sdpa._probe()
    if reason:
        pytest.skip(f"kvarn ops unavailable: {reason}")
    if row_ends and not kvarn_sdpa._probe_row_ends():
        pytest.skip("kvarn per-row ends need mlx-kquant 0.4.9 or later")


@pytest.fixture
def kvarn_ops_ok(monkeypatch):
    """kvarn eligibility without the Metal kernels: the resolver reads the
    ops probe (and the per-row ends version gate), and the env knobs start
    clear."""
    from gmlx.cache import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    monkeypatch.setattr(kvarn_sdpa, "_row_ends_result", (True,))
    for k in ("GMLX_KVARN", "GMLX_KVARN_BITS", "KV_BITS", "KV_TAIL_TOKENS",
              "KV_QUANT_SCHEME"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def kvarn_apc_arms(monkeypatch):
    """install_kvarn_apc for one test: the APC arms it rebinds are pinned
    first, so the wrap does not outlive the test."""
    from gmlx.cache import kvarn_apc
    from helpers import pin_apc_arms

    apc = pin_apc_arms(monkeypatch)
    monkeypatch.setattr(apc, kvarn_apc._FLAG, False, raising=False)
    kvarn_apc.install_kvarn_apc()
