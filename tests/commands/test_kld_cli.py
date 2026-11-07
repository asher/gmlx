"""gmlx kld: the refusals, the tokenization, and both passes on fake models,
checked against the arithmetic run directly."""

import contextlib
import json

import numpy as np
import pytest

from gmlx.commands import kld as cmd
from gmlx.gen import kld as K

V = 41
N_CTX = 16


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
    gap = 0.0
    forwards: list = []

    def __init__(self, model, n_ctx):
        self.shift = 0.0 if model.path.endswith("ref.gguf") else 0.5
        self.width = V
        self.first, self.n_scored = K.scored_span(n_ctx)

    def logits(self, ids):
        self.forwards.append(ids.shape[0])
        return _fake_logits(ids, self.shift)

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

    def load(path, args):
        events.append(("load", path.rsplit("/", 1)[-1]))
        return _Model(path)

    monkeypatch.setattr(cmd, "_load", load)
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


def _expected(tokens, n_chunk, bos):
    sums = K.KldSums()
    for c in range(n_chunk):
        ids = K.chunk_ids(tokens, c, N_CTX, bos)[None]
        rec = K.quantize_logprobs(_fake_logits(ids, 0.0)[0])
        sums.add(_fake_logits(ids, 0.5)[0], rec)
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
    assert _Scorer.forwards == [rows] * (2 * n_chunk // rows)


def test_chunks_caps_the_run_and_the_table_prints_a_percent(rig, capsys):
    rc, out, _ = _run([*rig["argv"], "--chunks", "2"], capsys)
    assert rc == 0
    want = _expected(rig["tokens"], 2, 3)
    assert f"{100 * want.same_top()[0]:.3f}" in out and "%" in out
    assert f"{want.mean_kld()[0]:.6f}" in out


def test_no_reference_refuses(rig, capsys):
    rc, _, err = _run([a for a in rig["argv"]
                       if a not in ("--reference", str(rig["tmp"] / "ref.gguf"))],
                      capsys)
    assert rc == 2 and "--reference" in err
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


def _tiny_model(head: str):
    """A text model: a cumulative-sum trunk, so each position sees its
    prefix, and an lm_head or tied embedding of width V."""
    import types

    import mlx.core as mx
    import mlx.nn as nn

    d = 32

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
            if head != "tied":
                lin = nn.Linear(d, V, bias=head == "bias")
                self.lm_head = (nn.QuantizedLinear.from_linear(lin, group_size=32, bits=8)
                                if head == "affine" else lin)

        def __call__(self, x):
            h = self.model(x)
            if self.args.tie_word_embeddings:
                return self.model.embed_tokens.as_linear(h)
            return self.lm_head(h)

    mx.random.seed(0)
    return Tiny()


@pytest.mark.parametrize("head", ["linear", "bias", "tied", "affine"])
def test_scorer_runs_the_head_in_float32_on_the_scored_positions(head, monkeypatch):
    import mlx.core as mx

    model = _tiny_model(head)
    monkeypatch.setattr(cmd, "HEAD_BLOCK", 7)
    ids = (np.arange(2 * N_CTX, dtype=np.int32).reshape(2, N_CTX) * 5) % V
    got = cmd.Scorer(model, N_CTX).logits(ids)
    first, n = K.scored_span(N_CTX)
    full = np.array(model(mx.array(ids)).astype(mx.float32))
    assert got.shape == (2, n, V) and got.dtype == np.float32
    np.testing.assert_allclose(got, full[:, first:first + n], rtol=1e-5, atol=1e-5)
    assert cmd.Scorer(model, N_CTX).parity_gap(ids[0]) < 1e-5
