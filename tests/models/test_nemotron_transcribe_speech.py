"""The Nemotron 3.5 Transcribe speech encoder (``pee`` GGUF): features,
the ASR and speaker branches, fusion and the CTC timestamp head, against
transformers' reference on a tiny random encoder.

The goldens come from scripts/speech_oracle/tiny_encoder.py (transformers
9b283429, float32 on the CPU); the GGUF is rebuilt here from the committed
weights by the same scripts/speech_oracle/tiny_pee.py."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

from gmlx.models.nemotron_h import speech, transcription

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "nemotron_transcribe"
NAME = "selftest_pee_tiny"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx = _module(ROOT / "scripts" / "image_oracle" / "fixtures.py", "_pee_fixtures")
tiny_pee = _module(ROOT / "scripts" / "speech_oracle" / "tiny_pee.py", "_tiny_pee")


@pytest.fixture()
def _cpu():
    with mx.stream(mx.cpu):
        prev = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            yield
        finally:
            mx.set_default_device(prev)


@pytest.fixture(scope="module")
def golden():
    arrays, manifest = fx.load(NAME, root=FIXTURES)
    return arrays, manifest


@pytest.fixture(scope="module")
def gguf_path(golden, tmp_path_factory):
    arrays, _ = golden
    weights = {k[2:]: v for k, v in arrays.items() if k.startswith("w/")}
    return tiny_pee.write_gguf(tmp_path_factory.mktemp("pee") / "tiny.gguf",
                               weights)


@pytest.fixture()
def encoder(gguf_path, _cpu):
    return speech.load(gguf_path)


def _audio(arrays, case):
    return arrays[f"{case}/pcm"].astype(np.float32) / 32768.0


def _close(manifest, key, got, ref):
    tol = manifest["tolerances"][key]["rel_l2"]
    assert got.shape == ref.shape, key
    err = fx.rel_l2(got, ref)
    assert err <= tol, f"{key}: relative L2 {err:.3g} over {tol:.3g}"


def test_audio_tokens_are_one_per_8_frames():
    assert speech.num_audio_tokens(0) == 0
    assert speech.num_audio_tokens(159) == 0
    assert speech.num_audio_tokens(160) == 1
    assert speech.num_audio_tokens(160 * 8) == 1
    assert speech.num_audio_tokens(160 * 9) == 2


def test_features_match_the_reference_extractor(golden, encoder):
    arrays, manifest = golden
    _, mel, _ = encoder
    audio = _audio(arrays, "short")
    valid = speech.num_frames(audio.size)
    feats = speech.features(audio, mel)
    assert feats.shape == (speech.num_audio_tokens(audio.size) * 8, 128)
    assert not feats[valid:].any()
    ref = arrays["short/features"]
    np.testing.assert_allclose(feats[:valid], ref[:valid], atol=1e-3)
    assert fx.rel_l2(feats[:valid], ref[:valid]) < 1e-5


@pytest.mark.parametrize("case", ["long", "short", "whole"])
def test_encoder_matches_the_reference(golden, encoder, case):
    arrays, manifest = golden
    enc, mel, _ = encoder
    audio = _audio(arrays, case)
    valid = speech.num_frames(audio.size)
    tokens = speech.num_audio_tokens(audio.size)
    feats = speech.features(audio, mel)
    out = enc(feats, valid, tokens)
    assert out.embeds.shape == (tokens, 24)
    # A clip of whole 8-frame groups: the reference's speaker head reads a
    # masked group past the audio at the last frame, where NeMo reads zeros.
    keep = tokens - 1 if case == "whole" else tokens
    _close(manifest, f"{case}/embeds", np.array(out.embeds)[:keep],
           arrays[f"{case}/embeds"][:keep])
    _close(manifest, f"{case}/asr_states", np.array(out.asr_states),
           arrays[f"{case}/asr_states"][:tokens])
    _close(manifest, f"{case}/speaker_probs", out.speaker_probs[:keep],
           arrays[f"{case}/speaker_probs"][:keep])
    logits = enc.diar.logits(np.asarray(feats))[: min(valid, keep * 8)]
    _close(manifest, f"{case}/speaker_logits", logits,
           arrays[f"{case}/speaker_logits"][: len(logits)])
    rows = []
    for _, lg in enc.ctc_logits(out.asr_states):
        lg = lg.astype(mx.float32)
        rows.append(np.array(lg - mx.logsumexp(lg, -1, keepdims=True)))
    _close(manifest, f"{case}/ctc_logp", np.concatenate(rows),
           arrays[f"{case}/ctc_logp"][: tokens * 2])


def test_long_case_overflows_the_speaker_cache(golden):
    _, manifest = golden
    assert manifest["results"]["long_compressions"] >= 1
    assert manifest["results"]["long_threshold_margin"] > 1e-3


def test_steps_yield_per_window_and_chunk(golden, encoder):
    arrays, _ = golden
    enc, mel, _ = encoder
    audio = _audio(arrays, "long")
    valid = speech.num_frames(audio.size)
    tokens = speech.num_audio_tokens(audio.size)
    feats = speech.features(audio, mel)
    wins = enc.windows(tokens)
    lengths = [b - a for a, b, _, _ in wins]
    fusions = -(-tokens // speech.FUSE_ROWS)

    def run(**kw):
        gen = enc.steps(feats, valid, tokens, **kw)
        yields = 0
        while True:
            try:
                next(gen)
                yields += 1
            except StopIteration as done:
                return yields, done.value

    batches = sum(-(-lengths.count(n) // speech.MAX_WINDOW_BATCH)
                  for n in set(lengths))
    yields, result = run()
    assert yields == batches + len(wins) + fusions
    whole = enc(feats, valid, tokens)
    np.testing.assert_array_equal(np.array(result.embeds), np.array(whole.embeds))
    yields, one = run(batch=1, dtype=mx.bfloat16, keep_states=False)
    assert yields == 2 * len(wins) + fusions
    assert one.asr_states is None and one.embeds.dtype == mx.bfloat16
    np.testing.assert_allclose(np.array(one.embeds.astype(mx.float32)),
                               np.array(whole.embeds), rtol=1e-2, atol=1e-2)


def test_ctc_columns_pick_from_the_log_softmax(golden, encoder):
    arrays, _ = golden
    enc, mel, _ = encoder
    audio = _audio(arrays, "short")
    feats = speech.features(audio, mel)
    tokens = speech.num_audio_tokens(audio.size)
    out = enc(feats, speech.num_frames(audio.size), tokens)
    full = []
    for _, lg in enc.ctc_logits(out.asr_states):
        lg = lg.astype(mx.float32)
        full.append(np.array(lg - mx.logsumexp(lg, -1, keepdims=True)))
    full = np.concatenate(full)
    cols = [3, 17, 39]
    got = enc.ctc_columns(out.asr_states, cols)
    np.testing.assert_array_equal(got["argmax"], full.argmax(-1))
    np.testing.assert_allclose(got["blank"], full[:, 39], atol=1e-5)
    np.testing.assert_allclose(got["columns"], full[:, cols], atol=1e-5)


def test_ctc_log_probs_read_their_columns_a_slice_at_a_time(golden, encoder,
                                                          monkeypatch):
    arrays, _ = golden
    enc, mel, _ = encoder
    audio = _audio(arrays, "short")
    out = enc(speech.features(audio, mel), speech.num_frames(audio.size),
              speech.num_audio_tokens(audio.size))
    whole = enc.ctc_columns(out.asr_states, [39, 3, 17])
    # Slices of 10 frames: the vocabulary is 40 classes.
    monkeypatch.setattr(speech, "CTC_MAX_ELEMENTS", 400)
    steps = enc.ctc_log_prob_steps(out.asr_states, [39, 3, 17])
    n = 0
    while True:
        try:
            next(steps)
            n += 1
        except StopIteration as done:
            got = done.value
            break
    assert n == -(-got.shape[0] // 10)
    np.testing.assert_allclose(got, whole["columns"], atol=1e-5)


def test_diarization_labels_are_the_10ms_activity_above_the_threshold(
        golden, encoder):
    arrays, _ = golden
    enc, mel, _ = encoder
    audio = _audio(arrays, "short")
    feats = speech.features(audio, mel)
    valid = speech.num_frames(audio.size)
    out = enc.steps(feats, valid, speech.num_audio_tokens(audio.size),
                    label_threshold=0.4641)
    while True:
        try:
            next(out)
        except StopIteration as done:
            labels = done.value.diarization_labels
            break
    probs = 1 / (1 + np.exp(-enc.diar.logits(np.asarray(feats))[:valid]))
    np.testing.assert_array_equal(labels, (probs > 0.4641).T)


def test_load_refuses_a_gguf_of_another_arch(tmp_path):
    import gguf

    w = gguf.GGUFWriter(str(tmp_path / "x.gguf"), "llama")
    w.add_tensor("x", np.zeros(4, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(ValueError, match="pee"):
        speech.load(tmp_path / "x.gguf")


# -- the real encoder against the transformers oracle -------------------------

REAL_ENV = "GMLX_TEST_NEMOTRON_TRANSCRIBE"
ARRAYS_ENV = "GMLX_SPEECH_FIXTURES"
ORACLE = FIXTURES / "oracle"
REAL_CLIPS = ["enc_short", "enc_two_speakers", "enc_six_speakers_stable"]


@pytest.fixture(scope="module")
def real_encoder():
    import host_box

    folder = os.environ.get(REAL_ENV)
    if not folder:
        pytest.skip(f"set {REAL_ENV} to the folder of the encoder GGUF")
    want = fx.read_manifest(REAL_CLIPS[0], root=ORACLE)["files"]["encoder"]
    path = Path(folder).expanduser() / want["name"]
    if not path.is_file() or path.stat().st_size != want["size"]:
        pytest.skip(f"{want['name']} ({want['size']} bytes) is not in {folder}")
    host_box.real_files()
    return speech.load(path)


@pytest.mark.parametrize("name", REAL_CLIPS)
def test_real_encoder_matches_the_oracle(real_encoder, name):
    manifest = fx.read_manifest(name, root=ORACLE)
    root = os.environ.get(ARRAYS_ENV)
    if not root or not (Path(root) / manifest["npz"]).is_file():
        pytest.skip(f"{manifest['npz']} is not under {ARRAYS_ENV}; run "
                    f"scripts/speech_oracle/{manifest['script']}")
    arrays, _ = fx.load(name, root=root, manifest=manifest)
    tol = manifest["tolerances"]
    enc, mel, _ = real_encoder
    audio = arrays["audio"]
    valid = speech.num_frames(audio.size)
    tokens = speech.num_audio_tokens(audio.size)
    out = enc(speech.features(audio, mel), valid, tokens)

    def close(key, got, ref):
        err = fx.rel_l2(got, ref[: len(got)])
        assert err <= tol[key]["rel_l2"], f"{key}: {err:.3g} over {tol[key]['rel_l2']:.3g}"

    close("embeds", np.array(out.embeds), arrays["fp32_embeds"])
    close("asr_states", np.array(out.asr_states), arrays["fp32_asr_states"])
    close("speaker_probs", out.speaker_probs, arrays["fp32_speaker_probs"])
    if "fp32_ctc_logp" in arrays:
        rows = []
        for _, lg in enc.ctc_logits(out.asr_states):
            lg = lg.astype(mx.float32)
            rows.append(np.array(lg - mx.logsumexp(lg, -1, keepdims=True)))
        close("ctc_logp", np.concatenate(rows), arrays["fp32_ctc_logp"])
    else:
        got = enc.ctc_columns(out.asr_states, arrays["ctc_columns"])
        agree = float(np.mean(got["argmax"] == arrays["fp32_ctc_argmax"]))
        assert agree >= tol["ctc_argmax"]["agree"]
        close("ctc_columns_logp", got["columns"], arrays["fp32_ctc_columns_logp"])


# -- transcription requests ---------------------------------------------------


@pytest.mark.parametrize("code, name", [
    ("en", "English"), ("pt-BR", "Portuguese"), ("zh_Hant", "Chinese"),
    (" FIL ", "Filipino"), ("en-US-x", None), ("xx", None), ("", None),
    (None, None),
])
def test_language_names(code, name):
    assert transcription.language_name(code) == name


def test_instructions_follow_the_card():
    assert transcription.transcription_instruction("de") == (
        transcription.TRANSCRIBE + " The speech is in German.")
    assert transcription.transcription_instruction("xx") == transcription.TRANSCRIBE
    assert transcription.transcription_instruction("de", speakers=True) == transcription.SPEAKERS
    assert transcription.translation_instruction() == (
        "Please translate the given speech to English.")


@pytest.mark.parametrize("seconds, prompt, want", [
    (0.0, 100, 24), (10.0, 300, 84), (3600.0, 100, 21_624), (3700.0, 100, 21_664),
    (3600.0, 45_200, 65_536 - 45_200), (3600.0, 70_000, 0),
])
def test_output_budget(seconds, prompt, want):
    assert transcription.output_budget(seconds, prompt) == want


def test_near_silence():
    assert transcription.near_silent(np.zeros(0, np.float32))
    assert transcription.near_silent(np.full(16000, 2e-5, np.float32))
    assert not transcription.near_silent(np.full(16000, 5e-5, np.float32))
    quiet = np.zeros(16000, np.float32)
    quiet[5] = 2e-3
    assert not transcription.near_silent(quiet)


def test_loop_cleanup_matches_the_card_regex():
    def card(text):
        text = transcription._CHAR_RUN.sub(lambda m: m.group(1), text)
        return transcription._BLOCK_RUN.sub(lambda m: m.group(1), text)

    rng = np.random.default_rng(3)
    alphabet = list("ab c\n")
    for _ in range(300):
        parts = []
        for _ in range(int(rng.integers(0, 6))):
            piece = "".join(rng.choice(alphabet, int(rng.integers(1, 40))))
            parts.append(piece * (int(rng.integers(15, 45)) if rng.random() < 0.6 else 1))
        text = "".join(parts)
        assert transcription.cleanup_generation_loops(text) == card(text)


def test_loop_cleanup_collapses_a_long_block():
    block = "".join(chr(0x4E00 + i) for i in range(3000))
    text = "start " + block * 20 + " end"
    assert transcription.cleanup_generation_loops(text) == "start " + block + " end"


def test_tags_strip_from_text():
    text = "<spk:0> Hi <ev:laugh> there. <sty:whisper>  <spk:1> Bye."
    assert transcription.strip_tags(text) == "<spk:0> Hi there. <spk:1> Bye."
    assert transcription.strip_tags(text, speakers=True) == "Hi there. Bye."
