"""Orchestration tests for generation._generate_over with stream_generate and the
prompt cache mocked, so the two-phase seam logic runs without a model."""
import importlib
import json
import types

from gmlx.generation import _generate_over

# mlx_lm re-exports the `generate` function over the submodule name, so attribute
# access returns the function; import the real modules explicitly.
mlg = importlib.import_module("mlx_lm.generate")
mlc = importlib.import_module("mlx_lm.models.cache")


def _resp(token, text, n):
    return types.SimpleNamespace(
        token=token, text=text, generation_tokens=n, generation_tps=10.0
    )


class _Tok:
    """EOG = {2}; special ids {2, 7}; template gen prompt appends [60, 70]."""

    all_special_ids = [2, 7]

    def __init__(self):
        self.eos_token_ids = {2}

    def decode(self, ids):
        return {2: "<|user|>"}.get(ids[0], f"<{ids[0]}>")

    def encode(self, text, add_special_tokens=True):
        return [101, 102]

    def apply_chat_template(self, messages, add_generation_prompt=False, **kw):
        base = [9, 50, 120]
        return base + [60, 70] if add_generation_prompt else list(base)


def _patch_stream(monkeypatch, phase1, phase2):
    """phase1/phase2 are [(token_id, text)]; phase is chosen by prompt type
    (str = phase 1 templated prompt, list = phase 2 bridge)."""

    def fake(model, tokenizer, prompt, *, max_tokens, sampler,
             logits_processors, prompt_cache, **base):
        seq = phase1 if isinstance(prompt, str) else phase2
        for i, (tid, txt) in enumerate(seq):
            if i >= max_tokens:
                break
            yield _resp(tid, txt, i + 1)

    monkeypatch.setattr(mlg, "stream_generate", fake)
    monkeypatch.setattr(mlc, "make_prompt_cache", lambda model, max_kv_size=None: [])


def _call(**kw):
    base = dict(
        main_sampler=None, over_sampler=None, logits_processors=[],
        base_kwargs={}, max_kv_size=None, max_tokens=64, window=0,
        inject_critique=None, template_kwargs=None, log_path=None,
        params={}, verbose=False,
    )
    base.update(kw)
    return _generate_over(object(), _Tok(), "PROMPT", **base)


def test_free_mode_splits_at_seam_and_forces_window(monkeypatch, tmp_path):
    _patch_stream(
        monkeypatch,
        phase1=[(11, "def "), (12, "foo"), (2, "<|user|>")],
        phase2=[(20, "crit"), (2, "<|user|>"), (22, "x")],
    )
    log = tmp_path / "probe.jsonl"
    text = _call(window=3, log_path=str(log))
    # pre-text excludes the seam token; over-text is the forced window.
    assert text == "def foocrit<|user|>x"
    rec = json.loads(log.read_text().strip())
    assert rec["mode"] == "free"
    assert rec["seam"]["token_id"] == 2
    assert rec["pre_text"] == "def foo"
    assert rec["over_text"] == "crit<|user|>x"
    assert rec["over_tokens"] == 3
    # the replayed EOG inside the window is recorded as an interim stop.
    assert rec["interim_stops"] == [{"index": 1, "token_id": 2}]


def test_free_mode_window_caps_token_count(monkeypatch):
    _patch_stream(
        monkeypatch,
        phase1=[(11, "a"), (2, "<|user|>")],
        phase2=[(20, "one"), (21, "two"), (22, "three")],
    )
    text = _call(window=2)
    assert text == "aonetwo"  # window=2 stops before "three"


def test_free_mode_cuts_short_on_special_token_run(monkeypatch, tmp_path):
    # Coherent token, then the model only re-emits a special token (id 7): the
    # window must cut short instead of forcing the full budget of noise.
    _patch_stream(
        monkeypatch,
        phase1=[(11, "code"), (2, "<|user|>")],
        phase2=[(20, "a"), (7, "x"), (7, "x"), (7, "x"), (7, "x"), (7, "x"),
                (30, "never")],
    )
    log = tmp_path / "probe.jsonl"
    _call(window=50, log_path=str(log))
    rec = json.loads(log.read_text().strip())
    assert rec["early_stop"] is True
    assert "never" not in rec["over_text"]   # cut before the post-collapse token
    assert rec["over_tokens"] == 6           # "a" + 5 special tokens, then stop


def test_inject_mode_builds_bridge_and_answers(monkeypatch, tmp_path):
    _patch_stream(
        monkeypatch,
        phase1=[(11, "code"), (2, "<|user|>")],
        phase2=[(30, "the "), (31, "dt "), (32, "bug")],
    )
    log = tmp_path / "probe.jsonl"
    text = _call(inject_critique="any bugs?", log_path=str(log))
    assert text == "codethe dt bug"
    rec = json.loads(log.read_text().strip())
    assert rec["mode"] == "inject"
    assert rec["inject_critique"] == "any bugs?"
    assert rec["pre_text"] == "code"
    assert rec["over_text"] == "the dt bug"


def test_inject_window_caps_reply(monkeypatch):
    _patch_stream(
        monkeypatch,
        phase1=[(11, "code"), (2, "<|user|>")],
        phase2=[(30, "a"), (31, "b"), (32, "c")],
    )
    text = _call(inject_critique="bugs?", window=1)
    assert text == "codea"  # window caps the injected reply at 1 token


def test_no_seam_returns_pre_text_and_does_not_log(monkeypatch, tmp_path):
    _patch_stream(
        monkeypatch,
        phase1=[(11, "a"), (12, "b")],  # no EOG within max_tokens
        phase2=[(20, "x")],
    )
    log = tmp_path / "probe.jsonl"
    text = _call(window=5, log_path=str(log))
    assert text == "ab"
    assert not log.exists()  # no seam reached, nothing recorded


def test_record_carries_prompt_and_label(monkeypatch, tmp_path):
    _patch_stream(
        monkeypatch,
        phase1=[(11, "x"), (2, "<|user|>")],
        phase2=[(20, "y")],
    )
    log = tmp_path / "probe.jsonl"
    _call(window=1, log_path=str(log), orig_prompt="write code", label="exp1")
    rec = json.loads(log.read_text().strip())
    assert rec["prompt"] == "write code"
    assert rec["label"] == "exp1"


def test_seam_text_filled_from_decode_when_blank(monkeypatch, tmp_path):
    _patch_stream(
        monkeypatch,
        phase1=[(11, "a"), (2, "")],  # EOG streamed as empty text
        phase2=[(20, "y")],
    )
    log = tmp_path / "probe.jsonl"
    _call(window=1, log_path=str(log))
    rec = json.loads(log.read_text().strip())
    assert rec["seam"]["text"] == "<|user|>"  # filled via tokenizer.decode


def test_over_generation_with_prompt_cache_kwarg(monkeypatch):
    """--kv-bits on a rotating-cache model puts prompt_cache into gen_kwargs;
    the over-generation branch must not re-pass it to stream_generate."""
    import gmlx.generation as generation

    _patch_stream(
        monkeypatch,
        phase1=[(11, "a"), (2, "<|user|>")],
        phase2=[(20, "b")],
    )
    monkeypatch.setattr(generation, "kv_quantization_unsupported", lambda m: "rotating")
    monkeypatch.setattr(generation, "quantize_pooled_caches", lambda c, b, g: 1)
    text = generation.generate(
        object(), _Tok(), "PROMPT", kv_bits=8, over_generation=2,
        apply_chat_template=False,
    )
    assert text == "ab"
