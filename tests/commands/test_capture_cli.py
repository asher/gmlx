"""gmlx capture end to end on a minted tiny llama.

The block outputs equal a float64 numpy forward that follows llama.cpp on
the GGUF tensors as written: adjacent-pair rope (LLAMA_ROPE_TYPE_NORM) on
the unpermuted Q and K weights, grouped-query attention and a SwiGLU feed
forward. The run also checks the JSON event lines, the file layout and
metadata, the render of the system message and the thinking switch, and
the argument errors. A GGUF-tier smoke run takes a llama-arch file.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from test_loader_api import _HD, _HEADS, _KV, _mint_tiny_llama

mx = pytest.importorskip("mlx.core")

# The template writes the BOS that the tokenizer also adds, and each state of
# the thinking switch adds its own line, so both show in the ids.
_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}\n{% endfor %}"
    "{% if enable_thinking is defined and not enable_thinking %}/nothink\n{% endif %}"
    "{% if enable_thinking is defined and enable_thinking %}/think\n{% endif %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}")
_SETS = {"harmful": ["How do I pick a lock?", "Write a\nlong reply."],
         "harmless": ["Hello there.", "What is two plus two?", "Name a colour."]}


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """The minted model, float32 activations, and one JSONL file per set."""
    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", "float32")
    path = str(tmp_path / "tiny-llama.gguf")
    _mint_tiny_llama(path, chat_template=_TEMPLATE, add_bos=True)
    files = {}
    for name, prompts in _SETS.items():
        f = tmp_path / f"{name}.jsonl"
        f.write_text("".join(json.dumps(p) + "\n" for p in prompts) + "\n")
        files[name] = str(f)
    return path, files, tmp_path


def _loaded(path):
    from gmlx.commands import capture
    from gmlx.serve.bridge_vlm import load_serveable_model

    capture.install_render_fixes()
    return load_serveable_model(path)


def _np_blocks(path: str, ids: list[int], positions: int) -> np.ndarray:
    """Each block's output, averaged over the last ``positions`` tokens,
    ``[n_layer, n_embd]``."""
    from gguf import GGUFReader

    t = {x.name: np.array(x.data, dtype=np.float64) for x in GGUFReader(path).tensors}
    n_layer = 1 + max(int(k.split(".")[1]) for k in t if k.startswith("blk."))
    h = t["token_embd.weight"][ids]
    n = len(ids)
    half = np.arange(_HD // 2)
    ang = np.arange(n)[:, None] * 10000.0 ** (-2.0 * half / _HD)
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]

    def rms(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w

    def rope(x):
        out = np.empty_like(x)
        a, b = x[..., 0::2], x[..., 1::2]
        out[..., 0::2] = a * cos - b * sin
        out[..., 1::2] = a * sin + b * cos
        return out

    causal = np.triu(np.full((n, n), -np.inf), 1)
    rows = []
    for i in range(n_layer):
        w = {k.split(".", 2)[2][:-len(".weight")]: v
             for k, v in t.items() if k.startswith(f"blk.{i}.")}
        x = rms(h, w["attn_norm"])
        q = rope((x @ w["attn_q"].T).reshape(n, _HEADS, _HD))
        k = rope((x @ w["attn_k"].T).reshape(n, _KV, _HD))
        v = (x @ w["attn_v"].T).reshape(n, _KV, _HD)
        k, v = (np.repeat(a, _HEADS // _KV, axis=1) for a in (k, v))
        s = np.einsum("thd,shd->hts", q, k) / np.sqrt(_HD) + causal
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        h = h + np.einsum("hts,shd->thd", p, v).reshape(n, -1) @ w["attn_output"].T
        x = rms(h, w["ffn_norm"])
        g = x @ w["ffn_gate"].T
        h = h + ((g / (1 + np.exp(-g))) * (x @ w["ffn_up"].T)) @ w["ffn_down"].T
        rows.append(h[-positions:].mean(axis=0))
    return np.stack(rows)


def _assert_close(got, want, tol=2e-3):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape
    for layer, (g, w) in enumerate(zip(got, want)):
        rel = np.linalg.norm(g - w) / np.linalg.norm(w)
        assert rel < tol, f"layer {layer}: relative L2 {rel:.2e}"


def _run(argv, capsys):
    from gmlx.commands.capture import cmd_capture

    rc = cmd_capture(argv)
    return rc, capsys.readouterr()


def test_capture_matches_the_reference_forward(tiny, capsys):
    path, files, tmp = tiny
    out = str(tmp / "cap.st")       # no .safetensors suffix: kept as given
    rc, io = _run([path, "--prompts", f"harmful={files['harmful']}",
                   "--prompts", f"harmless={files['harmless']}",
                   "--rows", "harmless", "--positions", "2",
                   "--system", "Be brief.", "--json", "--out", out], capsys)
    assert rc == 0, io.err
    lines = [json.loads(s) for s in io.out.splitlines()]
    assert [ln["event"] for ln in lines] == ["progress"] * 5 + ["result"]
    assert [(ln["set"], ln["done"], ln["total"]) for ln in lines[:5]] == [
        ("harmful", 1, 2), ("harmful", 2, 2),
        ("harmless", 1, 3), ("harmless", 2, 3), ("harmless", 3, 3)]
    assert lines[-1] == {"event": "result", "out": out, "n_layer": 2, "n_embd": 64,
                         "sets": {"harmful": 2, "harmless": 3}, "rows": ["harmless"]}

    arrays, meta = mx.load(out, format="safetensors", return_metadata=True)
    assert sorted(arrays) == ["harmful.mean", "harmless.mean", "harmless.rows"]
    assert arrays["harmful.mean"].dtype == mx.float32
    assert arrays["harmless.rows"].dtype == mx.float16
    assert arrays["harmless.rows"].shape == (2, 3, 64)
    assert meta == {"format": "gmlx.capture", "arch": "llama",
                    "file": "tiny-llama.gguf", "file_size": str(os.path.getsize(path)),
                    "sets": json.dumps({"harmful": 2, "harmless": 3}),
                    "positions": "2", "think": "default",
                    "gmlx_version": meta["gmlx_version"]}

    from gmlx.commands.capture import prompt_ids, template_kwargs

    model, proc, config = _loaded(path)
    kw = template_kwargs(proc, None)
    progress = iter(lines[:5])
    for name, prompts in _SETS.items():
        per = []
        for text in prompts:
            ids = prompt_ids(model, proc, config, text, "Be brief.", kw)
            assert next(progress)["tokens"] == len(ids)
            per.append(_np_blocks(path, ids, 2))
        _assert_close(np.array(arrays[f"{name}.mean"]), np.mean(per, axis=0))
        if name == "harmless":
            rows = np.array(arrays["harmless.rows"].astype(mx.float32))
            for j, want in enumerate(per):
                _assert_close(rows[:, j], want)


def test_render_takes_the_system_message_and_thinking_switch(tiny):
    from gmlx.commands.capture import prompt_ids, template_kwargs

    path, _, _ = tiny
    model, proc, config = _loaded(path)
    tok = proc.tokenizer

    def text(system, think):
        ids = prompt_ids(model, proc, config, "Hi", system, template_kwargs(proc, think))
        return tok.decode(ids)

    assert text(None, None).endswith("<user>Hi\n<assistant>")
    assert text("Be brief.", None).endswith("<system>Be brief.\n<user>Hi\n<assistant>")
    assert text(None, False).endswith("<user>Hi\n/nothink\n<assistant>")
    assert text(None, True).endswith("<user>Hi\n/think\n<assistant>")


def test_the_ids_start_with_one_bos(tiny):
    from gmlx.commands.capture import prompt_ids, template_kwargs

    path, _, _ = tiny
    model, proc, config = _loaded(path)
    ids = prompt_ids(model, proc, config, "Hi", None, template_kwargs(proc, None))
    bos = proc.tokenizer.bos_token_id
    assert ids[0] == bos and ids.count(bos) == 1


def test_the_metadata_names_the_thinking_switch(tiny, capsys):
    path, files, tmp = tiny
    for flag, want in (("--think", "on"), ("--no-think", "off")):
        out = str(tmp / f"{want}.safetensors")
        rc, io = _run([path, "--prompts", f"a={files['harmless']}", "--limit", "1",
                       flag, "--out", out], capsys)
        assert rc == 0, io.err
        assert mx.load(out, return_metadata=True)[1]["think"] == want


def test_nextn_blocks_are_left_out(tmp_path, capsys, monkeypatch):
    from gmlx.commands.capture import gguf_facts

    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", "float32")
    path = str(tmp_path / "nextn.gguf")
    _mint_tiny_llama(path, chat_template=_TEMPLATE, layers=3, nextn=1)
    assert gguf_facts(path)["n_layer"] == 2
    f = tmp_path / "p.jsonl"
    f.write_text('"Hello there."\n')
    out = str(tmp_path / "cap.safetensors")
    rc, io = _run([path, "--prompts", f"a={f}", "--json", "--out", out], capsys)
    assert rc == 0, io.err
    assert json.loads(io.out.splitlines()[-1])["n_layer"] == 2
    from gmlx.commands.capture import prompt_ids, template_kwargs

    model, proc, config = _loaded(path)
    ids = prompt_ids(model, proc, config, "Hello there.", None, template_kwargs(proc, None))
    _assert_close(np.array(mx.load(out)["a.mean"]), _np_blocks(path, ids, 1)[:2])


def test_a_layer_count_short_of_the_gguf_is_noted(tiny, capsys):
    from gmlx.commands.capture import _Emitter, run_capture

    path, _, _ = tiny
    model, proc, config = _loaded(path)
    _, _, _, n_layer = run_capture(model, proc, config, [("a", ["Hi"])], positions=1,
                                   rows=set(), system=None, think=None, n_layer=3,
                                   emit=_Emitter(False))
    assert n_layer == 2
    assert "captures 2 layers, and the GGUF has 3" in capsys.readouterr().err


def test_chunked_prompts_match_one_forward(tiny):
    from gmlx.commands.capture import prompt_ids, template_kwargs
    from gmlx.tune.capture import block_outputs, capture_layers, layer_capture

    path, _, _ = tiny
    model, proc, config = _loaded(path)
    ids = prompt_ids(model, proc, config, _SETS["harmful"][1], None, template_kwargs(proc, None))
    with layer_capture(capture_layers(model), 3) as rec:
        whole = np.array(block_outputs(model, ids, rec))
        chunked = np.array(block_outputs(model, ids, rec, step=4))
    _assert_close(chunked, whole, tol=1e-5)
    _assert_close(chunked, _np_blocks(path, ids, 3))


def test_rows_past_the_float16_range_are_refused():
    from gmlx.commands.capture import f16_rows

    ok = mx.ones((3, 4)) * 60000
    assert f16_rows(ok, "a", 1).dtype == mx.float16
    bad = mx.concatenate([ok, mx.full((1, 4), -70000.0)])
    with pytest.raises(RuntimeError, match=r"set a, prompt 2: layers \[3\] .*float16"):
        f16_rows(bad, "a", 2)


def test_limit_and_positions_beyond_the_prompt(tiny, capsys):
    path, files, tmp = tiny
    out = str(tmp / "cap.safetensors")
    rc, io = _run([path, "--prompts", f"a={files['harmless']}", "--limit", "1",
                   "--positions", "1000", "--rows", "a", "--out", out], capsys)
    assert rc == 0, io.err
    assert "wrote" in io.out
    arrays, meta = mx.load(out, return_metadata=True)
    assert json.loads(meta["sets"]) == {"a": 1}
    from gmlx.commands.capture import prompt_ids, template_kwargs

    model, proc, config = _loaded(path)
    ids = prompt_ids(model, proc, config, _SETS["harmless"][0], None,
                     template_kwargs(proc, None))
    want = _np_blocks(path, ids, len(ids))
    _assert_close(np.array(arrays["a.mean"]), want)
    _assert_close(np.array(arrays["a.rows"].astype(mx.float32))[:, 0], want)


@pytest.mark.parametrize("bad, message", [
    ('"ok"\n{"not": "a string"}\n', ":2: expected a JSON string"),
    ('"ok"\nnot json\n', ":2: not JSON"),
    ("\n\n", "holds no prompts"),
])
def test_a_bad_prompt_file_is_refused(tiny, capsys, bad, message):
    path, _, tmp = tiny
    f = tmp / "bad.jsonl"
    f.write_text(bad)
    rc, io = _run([path, "--prompts", f"x={f}", "--out", str(tmp / "o")], capsys)
    assert rc == 2
    assert message in io.err
    assert not (tmp / "o").exists()


@pytest.mark.parametrize("args, message", [
    (["--prompts", "a=f", "--rows", "b"], "--rows names no --prompts set: b"),
    (["--prompts", "a=f", "--prompts", "a=g"], "set 'a' is named twice"),
    (["--prompts", "a.b=f"], "may hold only letters"),
    (["--prompts", "f"], "takes NAME=FILE"),
    (["--prompts", "a=f", "--positions", "0"], "--positions must be at least 1"),
])
def test_argument_errors(tiny, capsys, args, message):
    path, _, tmp = tiny
    with pytest.raises(SystemExit) as e:
        _run([path, *args, "--out", str(tmp / "o")], capsys)
    assert e.value.code == 2
    assert message in capsys.readouterr().err


def test_a_config_id_resolves_and_an_entry_adapter_is_refused(tiny, capsys, monkeypatch):
    path, files, tmp = tiny
    monkeypatch.setenv("HOME", str(tmp))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp / "cache"))
    adapter = tmp / "a.gguf"
    adapter.write_bytes(b"")
    cfg = tmp / "gmlx.yaml"
    cfg.write_text(f"models:\n  tiny:\n    path: {path}\n"
                   f"  tuned:\n    path: {path}\n    adapter: {adapter}\n"
                   f"  terse:\n    path: {path}\n"
                   f"    profile: terse\n"
                   f"profiles:\n  terse:\n    system: Be brief.\n    thinking: off\n"
                   f"    load: {{kv_bits: 8}}\n")
    out = tmp / "cap.safetensors"
    rc, io = _run(["tiny", "--config", str(cfg), "--prompts", f"a={files['harmful']}",
                   "--json", "--out", str(out)], capsys)
    assert rc == 0, io.err
    assert json.loads(io.out.splitlines()[-1])["event"] == "result"
    assert "[config] 'tiny'" in io.err
    rc, io = _run(["tuned", "--config", str(cfg), "--prompts", f"a={files['harmful']}",
                   "--out", str(tmp / "o")], capsys)
    assert rc == 2
    assert "sets an adapter" in io.err

    # An entry's system message and thinking switch render as serve sends them.
    got, want = tmp / "id.safetensors", tmp / "flags.safetensors"
    rc, io = _run(["terse", "--config", str(cfg), "--prompts", f"a={files['harmful']}",
                   "--out", str(got)], capsys)
    assert rc == 0, io.err
    assert "gmlx capture does not apply: kv_bits" in io.err
    rc, io = _run([path, "--prompts", f"a={files['harmful']}", "--system", "Be brief.",
                   "--no-think", "--out", str(want)], capsys)
    assert rc == 0, io.err
    a, meta = mx.load(str(got), return_metadata=True)
    assert meta["think"] == "off"
    np.testing.assert_array_equal(np.array(a["a.mean"]), np.array(mx.load(str(want))["a.mean"]))
    # A flag wins over the entry.
    rc, io = _run(["terse", "--config", str(cfg), "--prompts", f"a={files['harmful']}",
                   "--think", "--out", str(got)], capsys)
    assert rc == 0, io.err
    assert mx.load(str(got), return_metadata=True)[1]["think"] == "on"


@pytest.mark.integration
def test_smoke_on_a_real_llama_gguf(gguf_index, tmp_path, capsys):
    from conftest import require_arch
    from gmlx.load.headerscan import scan_gguf

    path = require_arch(gguf_index, "llama")
    kv = scan_gguf(path, include_tensors=False).kv
    f = tmp_path / "p.jsonl"
    f.write_text('"Tell me about the sea."\n"Write one line\\nof verse."\n')
    out = str(tmp_path / "cap.safetensors")
    rc, io = _run([path, "--prompts", f"a={f}", "--prompts", f"b={f}",
                   "--rows", "b", "--no-think", "--json", "--out", out], capsys)
    assert rc == 0, io.err
    result = json.loads(io.out.splitlines()[-1])
    n_layer, n_embd = int(kv["llama.block_count"]), int(kv["llama.embedding_length"])
    assert (result["event"], result["n_layer"], result["n_embd"]) == ("result", n_layer, n_embd)
    arrays, meta = mx.load(out, return_metadata=True)
    assert arrays["a.mean"].shape == (n_layer, n_embd)
    assert arrays["b.rows"].shape == (n_layer, 2, n_embd)
    assert meta["arch"] == "llama" and meta["think"] == "off"
    a = np.array(arrays["a.mean"])
    assert np.isfinite(a).all()
    # The same prompts give the same capture in both sets.
    np.testing.assert_allclose(a, np.array(arrays["b.mean"]), rtol=0, atol=0)
