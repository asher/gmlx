"""gmlx kld: the refusals, the tokenization, and both passes on fake models,
checked against the arithmetic run directly."""

import contextlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from gmlx.commands import kld as cmd
from gmlx.gen import kld as K

V = 41
N_CTX = 16
_RELEASE = cmd._release
_SLOT_SHIFT = (0.3, -0.7)


def _fake_logits(ids: np.ndarray, shift: float) -> np.ndarray:
    """Logits that depend on every id up to each scored position."""
    first, n = K.scored_span(ids.shape[1])
    pref = np.cumsum(ids, axis=1)[:, first:first + n].astype(np.float64)
    grid = np.arange(V, dtype=np.float64)
    z = np.sin(pref[..., None] * 0.37 + grid * 0.11) * 6 + shift * np.cos(grid)
    return z.astype(np.float32)


class _Model:
    def __init__(self, path):
        self.path = path


class _Scorer:
    """MODEL's logits are the reference's shifted by 0.5, plus each slot's
    strength times the slot's own shift."""

    gap = 0.0
    forwards: list = []

    def __init__(self, model, n_ctx):
        self.shift = 0.0 if model.path.endswith("ref.gguf") else 0.5
        self.width = V
        self.first, self.n_scored = K.scored_span(n_ctx)

    def slices(self, ids, uids=None):
        from gmlx import lora_rows

        uids = [cmd.BARE] * len(ids) if uids is None else list(uids)
        self.forwards.append(uids)
        rows = []
        for row, uid in zip(ids, uids):
            scales = lora_rows.scales_for(uid) or ()
            shift = self.shift + sum(s * _SLOT_SHIFT[i] for i, s in enumerate(scales))
            rows.append(_fake_logits(row[None], shift))
        z = np.concatenate(rows).reshape(-1, V)
        step = cmd.slice_rows(V)
        for p0 in range(0, len(z), step):
            yield z[p0:p0 + step]

    def parity_gap(self, ids):
        return self.gap


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Fake GGUFs, tokenizer and models; records loads and unloads."""
    import gmlx.distill.tokens as toks

    (tmp_path / "ref.gguf").write_bytes(b"")
    (tmp_path / "model.gguf").write_bytes(b"")
    text = tmp_path / "text.txt"
    text.write_text("x" * 100 + "\n")
    tokens = (np.arange(100, dtype=np.int32) * 7) % V
    events = []
    monkeypatch.setattr(toks, "tokenizer_from_gguf", lambda p: p)
    monkeypatch.setattr(toks, "logits_width_from_gguf", lambda p: V)
    monkeypatch.setattr(cmd, "tokenize", lambda tok, t: (tokens.copy(), 3))
    monkeypatch.setattr(cmd, "tokenizer_mismatch", lambda *a: None)

    def load(src, args, slots=()):
        events.append(("load", src.path.rsplit("/", 1)[-1]))
        if slots:
            events.append(("slots", [p.rsplit("/", 1)[-1] for p, _ in slots]))
        return _Model(src.path), args

    monkeypatch.setattr(cmd, "_load", load)
    monkeypatch.setattr(cmd, "_release", lambda m, p: None)
    monkeypatch.setattr(cmd, "_unload", lambda: events.append(("unload",)))
    monkeypatch.setattr(cmd, "_resident_wire",
                        lambda m, a: contextlib.nullcontext())
    monkeypatch.setattr(cmd, "Scorer", _Scorer)
    monkeypatch.setattr(_Scorer, "forwards", [])
    monkeypatch.setattr(cmd.tempfile, "tempdir", str(tmp_path))
    argv = [str(tmp_path / "model.gguf"), "--reference",
            str(tmp_path / "ref.gguf"), "--text", str(text), "--ctx",
            str(N_CTX)]
    return {"argv": argv, "events": events, "tokens": tokens,
            "tmp": tmp_path}


def _expected(tokens, n_chunk, bos, ref_shift=0.0, shift=0.5):
    sums = K.KldSums()
    for c in range(n_chunk):
        ids = K.chunk_ids(tokens, c, N_CTX, bos)[None]
        rec = K.quantize_logprobs(_fake_logits(ids, ref_shift)[0])
        sums.add(_fake_logits(ids, shift)[0], rec)
    return sums


def _run(argv, capsys):
    rc = cmd.cmd_kld(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err


@pytest.mark.parametrize("budget", [1, 1 << 30])
def test_both_passes_match_the_arithmetic(rig, capsys, monkeypatch, budget):
    monkeypatch.setattr(cmd, "ROW_BUDGET_BYTES", budget)
    rc, out, _ = _run([*rig["argv"], "--json"], capsys)
    assert rc == 0
    lines = [json.loads(s) for s in out.splitlines()]
    assert all("event" in r for r in lines)
    progress = [(r["stage"], r["chunk"]) for r in lines[:-1]]
    n_chunk = 100 // N_CTX
    assert progress == ([("reference", c) for c in range(1, n_chunk + 1)]
                        + [("model", c) for c in range(1, n_chunk + 1)])
    result = lines[-1]
    want = _expected(rig["tokens"], n_chunk, 3)
    assert result["event"] == "result"
    assert result["positions"] == n_chunk * (N_CTX - 1 - N_CTX // 2)
    assert result["mean_kld"] == pytest.approx(want.mean_kld()[0], rel=1e-12)
    assert result["same_top_p"] == pytest.approx(want.same_top()[0])
    assert 0 <= result["same_top_p"] <= 1
    assert rig["events"] == [("load", "ref.gguf"), ("unload",),
                             ("load", "model.gguf"), ("unload",)]
    assert not list(rig["tmp"].glob("gmlx-kld-*"))
    rows = 1 if budget == 1 else n_chunk
    assert [len(f) for f in _Scorer.forwards] == [rows] * (2 * n_chunk // rows)


def test_chunks_caps_the_run_and_the_table_prints_a_percent(rig, capsys):
    rc, out, _ = _run([*rig["argv"], "--chunks", "2"], capsys)
    assert rc == 0
    want = _expected(rig["tokens"], 2, 3)
    assert f"{100 * want.same_top()[0]:.3f}" in out and "%" in out
    assert f"{want.mean_kld()[0]:.6f}" in out


def test_neither_reference_nor_adapter_refuses(rig, capsys):
    rc, _, err = _run(_no_reference(rig), capsys)
    assert rc == 2 and "--reference" in err and "--adapter" in err
    assert rig["events"] == []


def _no_reference(rig):
    return [a for a in rig["argv"]
            if a not in ("--reference", str(rig["tmp"] / "ref.gguf"))]


def _adapters(tmp_path, *names, head=False):
    """Rank-1 llama adapters on one target each, or on the output head."""
    import mlx.core as mx

    from gmlx.load.adapter import save_lora_adapter

    out = []
    for i, name in enumerate(names):
        module = "lm_head" if head else f"model.layers.{i}.mlp.down_proj"
        n_in = 8 if head else 16
        pair = (module, mx.ones((1, n_in)), mx.ones((8, 1)))
        save_lora_adapter(str(tmp_path / name), [pair], alpha=1.0,
                          base_arch="llama", n_head=2, n_head_kv=2, n_layers=2)
        out.append(str(tmp_path / name))
    return out


def _results(out):
    lines = [json.loads(s) for s in out.splitlines()]
    return ([r for r in lines if r["event"] == "progress"],
            [r for r in lines if r["event"] == "result"])


def test_no_adapter_tensors_are_held_while_the_reference_scores(
        rig, capsys, monkeypatch):
    """The adapters are checked before any load, then read again for MODEL,
    so the reference pass holds none of their tensors."""
    import gc
    import weakref

    import gmlx.load.adapter as adapter

    a, b = _adapters(rig["tmp"], "a.gguf", "b.gguf")
    plans = []
    real = adapter.load_lora_adapter

    def load_lora(path, **kw):
        plan = real(path, **kw)
        plans.append(weakref.ref(plan))
        return plan

    monkeypatch.setattr(adapter, "load_lora_adapter", load_lora)
    alive = []

    def load(src, args, slots=()):
        gc.collect()
        alive.append((src.path.rsplit("/", 1)[-1],
                      sum(r() is not None for r in plans)))
        return _Model(src.path), args

    monkeypatch.setattr(cmd, "_load", load)
    rc, _, err = _run([*rig["argv"], "--adapter", f"{a}:0.5", "--adapter", b],
                      capsys)
    assert rc == 0, err
    assert len(plans) == 2
    assert alive == [("ref.gguf", 0), ("model.gguf", 0)]


def test_model_load_reads_each_adapter_again_and_refuses_a_changed_file(
        tmp_path, monkeypatch):
    import gmlx.load.loader as loader
    import gmlx.load.adapter as adapter

    (a,) = _adapters(tmp_path, "a.gguf")
    installed = []
    monkeypatch.setattr(loader, "load_model",
                        lambda p: (_Model(p), {}, None))
    monkeypatch.setattr(adapter, "install_gguf_adapter",
                        lambda m, c, plan, slot: installed.append(
                            (plan.path, plan.sha256, slot)))
    monkeypatch.setattr("gmlx.commands.cli._apply_placement",
                        lambda ns, m: None)
    monkeypatch.setattr("gmlx.lora_rows.ensure_rows", lambda n: None)
    args = cmd.parse_args([str(tmp_path / "m.gguf"), "--text", "t.txt",
                           "--adapter", a])
    src = cmd.Source(str(tmp_path / "m.gguf"))
    slots, _points = cmd.adapter_points(args.adapter, src.path,
                                        with_bare=False)
    cmd._load(src, args, slots)
    assert installed == [(a, slots[0][1], 0)]
    _adapters(tmp_path, "a.gguf", head=True)
    with pytest.raises(cmd.Refusal, match="changed"):
        cmd._load(src, args, slots)


def test_without_a_reference_model_bare_is_the_reference(rig, capsys):
    """Only the adapter points score, against MODEL bare, in one load; each
    flag is a point and each distinct file a slot."""
    a, b = _adapters(rig["tmp"], "a.gguf", "b.gguf")
    rc, out, err = _run([*_no_reference(rig), "--json", "--adapter", f"{a}:0.5",
                         "--adapter", f"{a}:1.5", "--adapter", b], capsys)
    assert rc == 0, err
    progress, results = _results(out)
    n_chunk = 100 // N_CTX
    assert [(r["stage"], r["chunk"]) for r in progress] == (
        [("reference", c) for c in range(1, n_chunk + 1)]
        + [("model", c) for c in range(1, n_chunk + 1)])
    assert rig["events"] == [("load", "model.gguf"), ("slots", ["a.gguf", "b.gguf"]),
                             ("unload",)]
    assert [(r["adapter"], r["strength"]) for r in results] == [
        (a, 0.5), (a, 1.5), (b, 1.0)]
    for r, shift in zip(results, (0.5 * 0.3, 1.5 * 0.3, -0.7)):
        want = _expected(rig["tokens"], n_chunk, 3, 0.5, 0.5 + shift)
        assert r["reference"] == r["model"]
        assert r["mean_kld"] == pytest.approx(want.mean_kld()[0], rel=1e-12)
        assert r["same_top_p"] == pytest.approx(want.same_top()[0])
    assert not list(rig["tmp"].glob("gmlx-kld-*"))


@pytest.mark.parametrize("budget_rows", [1, 3, 64])
def test_with_a_reference_the_bare_model_and_each_point_score(
        rig, capsys, monkeypatch, budget_rows):
    """MODEL bare scores first; the rows of one forward are (chunk, point)
    pairs, chunk-major, as many as the budget holds."""
    monkeypatch.setattr(cmd, "ROW_BUDGET_BYTES",
                        budget_rows * 4 * V * (N_CTX - 1 - N_CTX // 2))
    (a,) = _adapters(rig["tmp"], "a.gguf")
    rc, out, err = _run([*rig["argv"], "--json", "--adapter", f"{a}:2"], capsys)
    assert rc == 0, err
    _, results = _results(out)
    n_chunk = 100 // N_CTX
    assert [r["adapter"] for r in results] == [None, a]
    for r, shift in zip(results, (0.5, 0.5 + 2 * 0.3)):
        want = _expected(rig["tokens"], n_chunk, 3, 0.0, shift)
        assert r["mean_kld"] == pytest.approx(want.mean_kld()[0], rel=1e-12)
    items = [uid for _c in range(n_chunk) for uid in (cmd.BARE, 1)]
    rows = min(budget_rows, len(items))
    want = [items[g:g + rows] for g in range(0, len(items), rows)]
    assert _Scorer.forwards[-len(want):] == want


def test_the_table_has_a_row_per_point(rig, capsys):
    (a,) = _adapters(rig["tmp"], "a.gguf")
    rc, out, _ = _run([*rig["argv"], "--adapter", f"{a}:0.25"], capsys)
    assert rc == 0
    rows = out.splitlines()[-2:]
    assert rows[0].startswith("MODEL ") and rows[1].startswith("a.gguf:0.25 ")


@pytest.mark.parametrize("case, message", [
    ("missing", "--adapter: no such file"),
    ("head", "output head"),
    ("not-lora", "not a LoRA adapter"),
])
def test_an_adapter_kld_cannot_score_refuses_before_any_load(
        rig, capsys, case, message):
    if case == "missing":
        path = str(rig["tmp"] / "gone.gguf")
    elif case == "head":
        (path,) = _adapters(rig["tmp"], "h.gguf", head=True)
    else:
        from gguf import GGUFWriter

        path = str(rig["tmp"] / "plain.gguf")
        w = GGUFWriter(path, "llama")
        w.add_tensor("token_embd.weight", np.zeros((4, 8), np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
    rc, _, err = _run([*_no_reference(rig), "--adapter", path], capsys)
    assert rc == 2 and message in err, err
    assert rig["events"] == []


@pytest.mark.parametrize("extra, patch", [
    (["--ctx", "2"], None),
    (["--chunks", "0"], None),
    ([], "mismatch"),
    ([], "short"),
    ([], "disk"),
])
def test_refusals_come_before_any_load(rig, capsys, monkeypatch, extra, patch):
    if patch == "mismatch":
        monkeypatch.setattr(cmd, "tokenizer_mismatch",
                            lambda *a: "the vocabularies differ")
    elif patch == "short":
        monkeypatch.setattr(cmd, "tokenize",
                            lambda tok, t: (rig["tokens"][:2 * N_CTX - 1], None))
    elif patch == "disk":
        class _Usage:
            free = 0
        monkeypatch.setattr(cmd.shutil, "disk_usage", lambda d: _Usage)
    rc, out, err = _run([*rig["argv"], *extra], capsys)
    assert rc == 2 and err.startswith("error: ")
    assert rig["events"] == []
    assert out == ""


def test_missing_model_refuses(rig, capsys):
    argv = list(rig["argv"])
    argv[0] = str(rig["tmp"] / "nope.gguf")
    rc, _, err = _run(argv, capsys)
    assert rc == 2 and "nope.gguf" in err


def test_a_head_that_misses_the_model_logits_refuses(rig, capsys, monkeypatch):
    monkeypatch.setattr(_Scorer, "gap", 0.5)
    rc, _, err = _run(rig["argv"], capsys)
    assert rc == 2 and "head" in err
    assert rig["events"] == [("load", "ref.gguf"), ("unload",)]
    assert not list(rig["tmp"].glob("gmlx-kld-*"))


class _Helper:
    def __init__(self, events, name):
        self.events, self.name = events, name

    def close(self):
        self.events.append(("close", self.name))


def test_each_streaming_install_closes_before_the_next_load(rig, capsys,
                                                           monkeypatch):
    import gmlx.stream.pagecache as pagecache

    events = rig["events"]

    def load(src, args, slots=()):
        name = src.path.rsplit("/", 1)[-1]
        events.append(("load", name))
        m = _Model(src.path)
        m._kq_feeder = _Helper(events, name)
        m._kq_weights_pin = _Helper(events, name)
        return m, args

    monkeypatch.setattr(cmd, "_load", load)
    monkeypatch.setattr(cmd, "_release", _RELEASE)
    monkeypatch.setattr(pagecache, "release_streaming_for",
                        lambda p: events.append(("sweep", p.rsplit("/", 1)[-1])))
    rc, _, _ = _run(rig["argv"], capsys)
    assert rc == 0
    assert events == [
        ("load", "ref.gguf"), ("close", "ref.gguf"), ("close", "ref.gguf"),
        ("sweep", "ref.gguf"), ("unload",),
        ("load", "model.gguf"), ("close", "model.gguf"),
        ("close", "model.gguf"), ("sweep", "model.gguf"), ("unload",)]


def test_a_vocabulary_that_adds_eos_refuses(rig, capsys, monkeypatch):
    import types

    import gmlx.distill.tokens as toks

    monkeypatch.setattr(toks, "tokenizer_from_gguf", lambda p: types.SimpleNamespace(
        _gguf_add_eos_token=p.endswith("model.gguf")))
    rc, _, err = _run(rig["argv"], capsys)
    assert rc == 2 and "MODEL" in err and "EOS" in err
    assert rig["events"] == []


def _entry_config(tmp_path, entry):
    (tmp_path / "s.gguf").write_bytes(b"GGUF")
    (tmp_path / "ad.gguf").write_bytes(b"GGUF")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"server:\n  model_dirs: [{tmp_path}]\n"
                   f"models:\n  s: {{path: s.gguf, {entry}}}\n")
    return str(cfg)


@pytest.mark.parametrize("typed, want", [
    ([], (True, False)),
    (["--stream-cpu"], (False, True)),
])
def test_a_config_entry_brings_its_placement_and_a_typed_flag_wins(
        tmp_path, typed, want):
    cfg = _entry_config(tmp_path, "stream: experts")
    args = cmd.parse_args(["s", "--reference", "s", "--text", "t", "--config",
                           cfg, *typed])
    src = cmd.resolve_model("s", cfg)
    assert src.path == str(tmp_path / "s.gguf")
    ns = cmd._model_args(src, args)
    assert (ns.stream_experts, ns.stream_cpu) == want
    assert ns.gguf == src.path
    plain = cmd._model_args(cmd.Source(src.path), args)
    assert (plain.stream_experts, plain.stream_cpu) == (False, bool(typed))


@pytest.mark.parametrize("entry, key", [
    ("adapter: ad.gguf", "adapter"),
    ("adapters: [{path: ad.gguf, strength: 0}]", "adapter"),
    ("rope_scaling: {type: yarn, factor: 2}", "rope_scaling"),
])
def test_a_config_entry_kld_cannot_apply_refuses(tmp_path, entry, key):
    cfg = _entry_config(tmp_path, entry)
    with pytest.raises(cmd.Refusal, match=key):
        cmd.resolve_model("s", cfg)


def test_rows_per_forward_keeps_the_logits_under_the_budget():
    assert cmd.rows_per_forward(255, 248320, 8, budget=1 << 30) == 4
    assert cmd.rows_per_forward(255, 248320, 2, budget=1 << 30) == 2
    assert cmd.rows_per_forward(2047, 248320, 8, budget=1 << 30) == 1


def _tiny_tokenizer(add_bos: bool):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<s>": 0, "<|x|>": 1, "a": 2, "b": 3, "<": 4, "|": 5, "x": 6,
             ">": 7, "[UNK]": 8}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>",
                                   unk_token="[UNK]")
    fast.add_special_tokens({"additional_special_tokens": ["<|x|>"]})
    fast._gguf_add_bos_token = add_bos
    return fast


def test_tokenize_keeps_control_tokens_as_text_and_adds_bos_by_the_vocab():
    tok = _tiny_tokenizer(False)
    ids, bos = cmd.tokenize(tok, "a<|x|>b")
    assert ids.tolist() == [2, 4, 5, 6, 5, 7, 3] and bos is None
    assert tok.encode("a<|x|>b", add_special_tokens=False) == [2, 1, 3]
    ids, bos = cmd.tokenize(_tiny_tokenizer(True), "a<|x|>b")
    assert ids.tolist() == [0, 2, 4, 5, 6, 5, 7, 3] and bos == 0


def test_tokenizer_mismatch_reasons():
    a, b = _tiny_tokenizer(False), _tiny_tokenizer(False)
    ids = np.array([2, 3], dtype=np.int32)
    assert cmd.tokenizer_mismatch(a, b, 9, 9, ids, ids) is None
    assert "widths" in cmd.tokenizer_mismatch(a, b, 9, 10, ids, ids)
    assert "differently" in cmd.tokenizer_mismatch(a, b, 9, 9, ids, ids[:1])
    b.add_tokens(["zz"])
    assert "sizes" in cmd.tokenizer_mismatch(a, b, 9, 9, ids, ids)


def _kq_head(d: int, codec: str, fold: bool):
    import mlx.core as mx
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantLinear

    head = KQuantLinear(d, V, False, codec)
    head.weight = kq.quantize(mx.random.normal((V, d)) * 0.2, codec)[0]
    if fold:
        from gmlx.load.hadamard import FoldTarget
        from gmlx.load.hadamard_modules import HadamardKQuantLinear, _Fold

        head.__class__ = HadamardKQuantLinear
        head._hadamard = _Fold(FoldTarget(width=d, block=d, signs=None), None)
    return head


def _tiny_model(head: str, d: int = 32):
    """A text model: a cumulative-sum trunk, so each position sees its
    prefix, and an lm_head or tied embedding of width V."""
    import types

    import mlx.core as mx
    import mlx.nn as nn

    class Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(V, d)

        def __call__(self, x):
            return mx.cumsum(self.embed_tokens(x), axis=1) * 0.3

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Trunk()
            self.args = types.SimpleNamespace(tie_word_embeddings=head == "tied")
            if head in ("q8_0", "q6_k", "hadamard"):
                self.lm_head = _kq_head(d, "q8_0" if head == "hadamard" else head,
                                        head == "hadamard")
            elif head != "tied":
                lin = nn.Linear(d, V, bias=head in ("bias", "scaled"))
                self.lm_head = (nn.QuantizedLinear.from_linear(lin, group_size=32, bits=8)
                                if head == "affine" else lin)
            if head == "scaled":
                from gmlx.load.out_scale import _scaled_class

                self.lm_head.__class__ = _scaled_class(type(self.lm_head))
                self.lm_head.post_scale = mx.array([0.5], dtype=mx.float32)

        def __call__(self, x):
            h = self.model(x)
            if self.args.tie_word_embeddings:
                return self.model.embed_tokens.as_linear(h)
            return self.lm_head(h)

    mx.random.seed(0)
    return Tiny()


def _scored_logits(scorer, ids):
    parts = list(scorer.slices(ids))
    assert all(p.dtype == np.float32 and p.shape[1] == V for p in parts)
    first, n = K.scored_span(N_CTX)
    return np.concatenate(parts).reshape(len(ids), n, V), parts


_IDS = (np.arange(2 * N_CTX, dtype=np.int32).reshape(2, N_CTX) * 5) % V


def _exact() -> bool:
    from gmlx.image.float32 import f32_products_exact

    return f32_products_exact()


def _rerun_without_tf32(request) -> None:
    """Rerun this test in a child process with TF32 off, as gmlx kld runs
    (gmlx.commands.cli sets MLX_ENABLE_TF32=0 for kld)."""
    env = dict(os.environ, MLX_ENABLE_TF32="0")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         request.node.nodeid], cwd=request.config.rootpath, env=env,
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]


@pytest.mark.parametrize("head", ["linear", "bias", "tied", "affine", "scaled"])
def test_scorer_runs_the_head_in_float32_on_the_scored_positions(head, monkeypatch,
                                                                 request):
    if not _exact():
        _rerun_without_tf32(request)
        return
    import mlx.core as mx

    model = _tiny_model(head)
    monkeypatch.setattr(cmd, "HEAD_BLOCK", 7)
    got, _ = _scored_logits(cmd.Scorer(model, N_CTX), _IDS)
    first, n = K.scored_span(N_CTX)
    full = np.array(model(mx.array(_IDS)).astype(mx.float32))
    np.testing.assert_allclose(got, full[:, first:first + n], rtol=1e-5, atol=1e-5)
    assert cmd.Scorer(model, N_CTX).parity_gap(_IDS[0]) < 1e-5


@pytest.mark.parametrize("head, d", [("q8_0", 32), ("q6_k", 256), ("hadamard", 32)])
def test_scorer_dequantizes_a_kquant_head_by_row_block(head, d, monkeypatch):
    """Each block of head rows dequantizes on its own and matches the
    whole weight dequantized in float32, after the head's input rotation."""
    import mlx.core as mx
    import mlx_kquant as kq

    from gmlx.distill.head import HEAD_PARITY_TOL
    from gmlx.distill.student import trunk_hidden

    model = _tiny_model(head, d)
    monkeypatch.setattr(cmd, "HEAD_BLOCK", 7)
    got, _ = _scored_logits(cmd.Scorer(model, N_CTX), _IDS)
    first, n = K.scored_span(N_CTX)
    lm = model.lm_head
    w = kq.dequantize(lm.weight, lm.scales, lm.kquant_type, dtype=mx.float32)
    h = trunk_hidden(model, mx.array(_IDS))[:, first:first + n].astype(mx.float32)
    if head == "hadamard":
        from gmlx.load.hadamard_modules import rotate

        h = rotate(h, lm._hadamard, kernel=False)
    want = np.array(h @ w.T)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    assert cmd.Scorer(model, N_CTX).parity_gap(_IDS[0]) < HEAD_PARITY_TOL


def test_scorer_slices_a_chunk_over_the_budget(monkeypatch):
    import mlx.core as mx

    model = _tiny_model("linear")
    monkeypatch.setattr(cmd, "ROW_BUDGET_BYTES", 4 * V * 3)
    got, parts = _scored_logits(cmd.Scorer(model, N_CTX), _IDS)
    first, n = K.scored_span(N_CTX)
    assert [len(p) for p in parts] == [3] * (2 * n // 3) + [2 * n % 3] * bool(2 * n % 3)
    full = np.array(model(mx.array(_IDS)).astype(mx.float32))
    np.testing.assert_allclose(got, full[:, first:first + n], rtol=1e-5, atol=1e-5)


def test_adapter_points_on_a_tiny_llama_match_each_adapter_folded(
        tmp_path, monkeypatch):
    """Real loads: three points on two slots score in shared forwards, and
    each matches the arithmetic on a model with that one adapter folded
    into its static scale, scored alone."""
    import mlx.core as mx

    import gmlx
    from gmlx import lora_rows
    from gmlx.load.adapter import apply_gguf_adapter, save_lora_adapter
    from test_loader_api import _FFN, _HEADS, _HID, _KV, _mint_tiny_llama

    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", "float32")
    path = str(tmp_path / "tiny.gguf")
    _mint_tiny_llama(path)
    from gguf import GGUFReader

    # The minted weights give near-flat logits; widen them in the file.
    (norm,) = [t for t in GGUFReader(path, "r+").tensors
               if t.name == "output_norm.weight"]
    norm.data[:] = norm.data * 400
    rng = np.random.default_rng(5)
    lora = {}
    for name, module, n_in in (("a.gguf", "model.layers.0.mlp.down_proj", _FFN),
                               ("b.gguf", "model.layers.1.self_attn.o_proj",
                                _HEADS * 16)):
        pair = (module, mx.array(rng.standard_normal((1, n_in)) * 30, mx.float32),
                mx.array(rng.standard_normal((_HID, 1)) * 30, mx.float32))
        lora[name] = str(tmp_path / name)
        save_lora_adapter(lora[name], [pair], alpha=1.0, base_arch="llama",
                          n_head=_HEADS, n_head_kv=_KV, n_layers=2)
    text = tmp_path / "t.txt"
    text.write_text("Hello world, we go on. " * 6)
    points = [("a.gguf", 0.5), ("a.gguf", 2.0), ("b.gguf", 1.0)]
    args = cmd.parse_args([path, "--text", str(text), "--ctx", str(N_CTX),
                           *(f for n, s in points
                             for f in ("--adapter", f"{lora[n]}:{s}"))])
    try:
        results = cmd.run(args, lambda rec: None, lambda msg: None)
    finally:
        lora_rows.configure("static", 1)
    assert lora_rows.mode() == "static"

    from gmlx.distill.tokens import tokenizer_from_gguf

    tokens, bos = cmd.tokenize(tokenizer_from_gguf(path), text.read_text())
    n_chunk = len(tokens) // N_CTX
    first, n = K.scored_span(N_CTX)

    def logits(model):
        out = []
        for c in range(n_chunk):
            ids = mx.array(K.chunk_ids(tokens, c, N_CTX, bos)[None])
            out.append(np.array(model(ids)[0, first:first + n].astype(mx.float32)))
        return out

    base, cfg, _ = gmlx.load_model(path)
    records = [K.quantize_logprobs(z) for z in logits(base)]
    assert [r["adapter"] for r in results] == [lora[n] for n, _ in points]
    for r, (name, strength) in zip(results, points):
        m, cfg, _ = gmlx.load_model(path)
        apply_gguf_adapter(m, cfg, lora[name], base_arch="llama",
                           strength=strength)
        sums = K.KldSums()
        for z, rec in zip(logits(m), records):
            sums.add(z, rec)
        assert r["positions"] == sums.count == n_chunk * n
        assert sums.mean_kld()[0] > 1e-3
        assert r["mean_kld"] == pytest.approx(sums.mean_kld()[0], rel=2e-3)
        assert abs(r["same_top_p"] - sums.same_top()[0]) <= 1 / sums.count
