"""Nemotron 3.5 Transcribe audio input: the chat template's user turn, the
processor's placeholder expansion and limits, the container's merge of
encoder rows, and the run defaults of a transcription. CPU only; the
gated test at the end reads the real decoder's tokenizer."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

from gmlx._vendor.mlx_vlm import apc as _apc
from gmlx._vendor.mlx_vlm.prompt_utils import get_message_json
from gmlx.cache.apc_media import install_media_token_ids
from gmlx.commands import cli
from gmlx.load import vlm
from gmlx.models.nemotron_h import speech, transcription

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "nemotron_transcribe"
TYPE = "nemotron_h_transcribe"
AUDIO_ID = 19


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx = _module(ROOT / "scripts" / "image_oracle" / "fixtures.py", "_pee_fixtures")
tiny_pee = _module(ROOT / "scripts" / "speech_oracle" / "tiny_pee.py", "_tiny_pee")


# The chat template

def _template_tokenizer():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0},
                                                    unk_token="[UNK]")),
        unk_token="[UNK]")
    tok.chat_template = (FIXTURES / "chat_template.jinja").read_text()
    return tok


@pytest.mark.parametrize("text, clips, user", [
    ("Transcribe.", 1, "Transcribe. <|audio|>"),
    ("  Two clips. ", 2, "Two clips. <|audio|> <|audio|>"),
    ("", 1, "<|audio|>"),
    ("No audio.", 0, "No audio."),
])
def test_user_turn_puts_the_locators_after_the_text(text, clips, user):
    msg = get_message_json(TYPE, text, num_audios=clips)
    assert msg == {"role": "user", "content": user}
    rendered = _template_tokenizer().apply_chat_template(
        [msg], tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    assert rendered == ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
                        f"{user}<|im_end|>\n<|im_start|>assistant\n"
                        "<think></think>")


def test_user_turn_refuses_images():
    with pytest.raises(ValueError, match="no image input"):
        get_message_json(TYPE, "hi", num_images=1)


def test_assistant_turn_is_plain_text():
    assert get_message_json(TYPE, "ok", role="assistant", num_audios=1) == {
        "role": "assistant", "content": "ok"}


# The processor

def _processor():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "hi": 1, "<SPECIAL_19>": AUDIO_ID}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["<SPECIAL_19>"])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    proc = vlm._synthesize_vlm_processor(TYPE, fast, {})
    proc.feature_extractor.mel_filters = np.full((128, 257), 1e-3, np.float32)
    return proc


def test_processor_expands_each_clip_to_one_token_per_80_ms():
    proc = _processor()
    fe = proc.feature_extractor
    assert fe.sampling_rate == 16000
    assert [fe.tokens(n) for n in (0, 159, 160, 1280, 1281, 1440)] == [
        0, 0, 1, 1, 1, 2]
    rng = np.random.default_rng(3)
    clips = [rng.normal(0, 0.1, 2000).astype(np.float32),
             rng.normal(0, 0.1, 1300).astype(np.float32)]
    out = proc(text="hi <|audio|> hi <|audio|>", audio=clips)
    ids = np.asarray(out["input_ids"])[0].tolist()
    assert ids == [1, AUDIO_ID, AUDIO_ID, 1, AUDIO_ID]
    feats = np.asarray(out["input_features"])
    mask = np.asarray(out["input_features_mask"])
    assert feats.shape == (2, 16, 128) and mask.shape == (2, 16)
    assert mask.sum(axis=1).tolist() == [12, 8]
    assert not feats[1, 8:].any()


def test_encoded_clips_reach_the_model_through_prepare_inputs():
    from gmlx._vendor.mlx_vlm.utils import prepare_inputs
    from gmlx.models.nemotron_h.transcribe_model import EncodedClip

    proc = _processor()
    proc.tokenizer.pad_token = "[UNK]"
    clips = [EncodedClip.of(n, f"d{n}", mx.zeros((proc.feature_extractor.tokens(n), 4)),
                            None) for n in (2000, 1300)]
    out = prepare_inputs(proc, audio=clips, prompts="hi <|audio|> hi <|audio|>")
    assert [id(c) for c in out["audio_clips"]] == [id(c) for c in clips]
    assert out["audio_digests"] == ["d2000", "d1300"]
    assert np.asarray(out["input_ids"])[0].tolist() == [1, AUDIO_ID, AUDIO_ID, 1,
                                                        AUDIO_ID]


@pytest.mark.parametrize("text, clips", [
    ("hi", 1), ("<|audio|> <|audio|>", 1), ("<|audio|>", 2)])
def test_processor_refuses_markers_that_miss_the_clips(text, clips):
    proc = _processor()
    with pytest.raises(ValueError, match="markers for"):
        proc(text=text, audio=[np.zeros(1600, np.float32)] * clips)


def test_processor_refuses_images():
    with pytest.raises(ValueError, match="no image input"):
        _processor()(images=[object()], text="hi")


def test_processor_limits_clips_and_seconds():
    fe = _processor().feature_extractor
    fe.check([np.zeros(16)] * 6)
    with pytest.raises(ValueError, match="7 audio clips.*limit is 6.*new "
                                         "conversation"):
        fe.check([np.zeros(16)] * 7)
    hour = 3600 * 16000
    fe.check([np.broadcast_to(np.float32(0), (hour // 2,))] * 2)
    with pytest.raises(ValueError, match="3600.1 s of audio.*limit is 3600 s"):
        fe.check([np.broadcast_to(np.float32(0), (hour + 1600,))])


def test_processor_needs_the_model_filterbank():
    proc = _processor()
    proc.feature_extractor.mel_filters = None
    with pytest.raises(RuntimeError, match="no mel filterbank"):
        proc(text="<|audio|>", audio=[np.zeros(1600, np.float32)])


# The container

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
def tiny_gguf(tmp_path_factory):
    arrays, _ = fx.load("selftest_pee_tiny", root=FIXTURES)
    weights = {k[2:]: v for k, v in arrays.items() if k.startswith("w/")}
    pcm = arrays["short/pcm"].astype(np.float32) / 32768.0
    path = tiny_pee.write_gguf(tmp_path_factory.mktemp("pee") / "tiny.gguf",
                               weights)
    return path, pcm


@pytest.fixture()
def container(tiny_gguf, _cpu):
    from gmlx.models.nemotron_h.model import ModelArgs
    from gmlx.models.nemotron_h.transcribe_model import (
        AudioConfig,
        Model,
        ModelConfig,
    )

    text = ModelArgs(
        model_type="nemotron_h", vocab_size=40, hidden_size=24,
        intermediate_size=32, num_hidden_layers=2, max_position_embeddings=64,
        num_attention_heads=4, num_key_value_heads=2, attention_bias=False,
        mamba_num_heads=4, mamba_head_dim=6, mamba_proj_bias=False,
        ssm_state_size=4, conv_kernel=4, n_groups=2, mlp_bias=False,
        layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
        hybrid_override_pattern=["M", "*"], head_dim=6)
    model = Model(ModelConfig(text_config=text,
                              audio_config=AudioConfig(fields=tiny_pee.FIELDS)))
    enc, mel, _ = speech.load(tiny_gguf[0])
    model.audio_tower = enc
    return model, mel, tiny_gguf[1]


def _inputs(mel, clips):
    feats = [speech.features(c, mel) for c in clips]
    width = max(f.shape[0] for f in feats)
    out = np.zeros((len(feats), width, mel.shape[0]), np.float32)
    mask = np.zeros((len(feats), width), bool)
    for i, (c, f) in enumerate(zip(clips, feats)):
        out[i, : f.shape[0]] = f
        mask[i, : speech.num_frames(c.size)] = True
    return out, mask


def test_container_puts_encoder_rows_on_the_audio_slots(container):
    model, mel, pcm = container
    clips = [pcm, pcm[:8000]]
    feats, mask = _inputs(mel, clips)
    n = [speech.num_audio_tokens(c.size) for c in clips]
    ids = mx.array([[5] + [AUDIO_ID] * n[0] + [6] + [AUDIO_ID] * n[1] + [7]])
    merged = model.get_input_embeddings(
        ids, input_features=feats, input_features_mask=mask).inputs_embeds
    want = [model.audio_tower(feats[i, : n[i] * 8], int(mask[i].sum()), n[i])
            .embeds for i in range(2)]
    slots = np.flatnonzero(np.asarray(ids[0]) == AUDIO_ID)
    np.testing.assert_allclose(np.asarray(merged[0, mx.array(slots)]),
                               np.asarray(mx.concatenate(want)), rtol=1e-5,
                               atol=1e-6)
    text = model.get_input_embeddings(ids).inputs_embeds
    others = mx.array([0, n[0] + 1, ids.shape[1] - 1])
    assert mx.array_equal(merged[0, others], text[0, others])
    assert model(ids, input_features=feats,
                 input_features_mask=mask).logits.shape == (1, ids.shape[1], 40)


def test_container_encodes_only_the_slots_past_a_reused_prefix(container):
    model, mel, pcm = container
    clips = [pcm, pcm[:8000]]
    feats, mask = _inputs(mel, clips)
    n = [speech.num_audio_tokens(c.size) for c in clips]
    full = model.audio_features(feats, mask, sum(n))
    calls = []
    real = model.audio_tower.steps

    class Spy:
        cfg = model.audio_tower.cfg

        def steps(self, *a, **kw):
            calls.append(a[2])
            return real(*a, **kw)

    model.audio_tower = Spy()
    tail = model.audio_features(feats, mask, n[1] + 3)
    assert calls == [n[1], n[0]]
    np.testing.assert_allclose(np.asarray(tail), np.asarray(full[-(n[1] + 3):]),
                               rtol=1e-5, atol=1e-6)
    calls.clear()
    model.audio_features(feats, mask, n[1])
    assert calls == [n[1]]


@pytest.mark.parametrize("chunk", [1, 4, 7, 1000])
@pytest.mark.parametrize("prefix", [False, True])
def test_prompt_rows_build_the_merged_rows_a_chunk_at_a_time(
        container, monkeypatch, chunk, prefix):
    model, mel, pcm = container
    # Several fusion parts per clip, so the rows free them as they pass.
    monkeypatch.setattr(speech, "FUSE_ROWS", 3)
    clips = [pcm, pcm[:8000]]
    feats, mask = _inputs(mel, clips)
    n = [speech.num_audio_tokens(c.size) for c in clips]
    ids = [5] + [AUDIO_ID] * n[0] + [6] + [AUDIO_ID] * n[1] + [7]
    if prefix:
        # A reused prefix ends inside the first clip.
        ids = ids[3:]
    ids = mx.array([ids])
    kw = dict(input_features=feats, input_features_mask=mask)
    merged = model.get_input_embeddings(ids, **kw).inputs_embeds
    rows = model.get_input_embeddings(ids, prompt_rows=True, **kw).inputs_embeds
    assert rows.shape == merged.shape and rows.dtype == merged.dtype
    got = []
    while rows.shape[1] > 0:
        k = min(chunk, rows.shape[1])
        got.append(rows[:, :k])
        rows = rows[:, k:]
    assert mx.array_equal(mx.concatenate(got, axis=1), merged)
    assert rows._parts == []


def test_generate_step_takes_prompt_rows(container):
    from gmlx._vendor.mlx_vlm.generate.ar import generate_step

    model, mel, pcm = container
    feats, mask = _inputs(mel, [pcm])
    n = speech.num_audio_tokens(pcm.size)
    ids = mx.array([[5] + [AUDIO_ID] * n + [6, 7]])
    kw = dict(input_features=feats, input_features_mask=mask)

    def first(lazy):
        model.builds_prompt_rows = lazy
        try:
            step = generate_step(ids, model, None, None, max_tokens=2,
                                 temperature=0.0, prefill_step_size=5, **kw)
            return [(int(y), np.asarray(lp)) for (y, lp), _ in zip(step, range(2))]
        finally:
            del model.builds_prompt_rows

    lazy, whole = first(True), first(False)
    assert [y for y, _ in lazy] == [y for y, _ in whole]
    for (_, a), (_, b) in zip(lazy, whole):
        np.testing.assert_array_equal(a, b)


def test_container_refuses_more_slots_than_audio(container):
    model, mel, pcm = container
    feats, mask = _inputs(mel, [pcm[:1280]])
    with pytest.raises(ValueError, match="2 audio placeholder tokens but 1"):
        model.get_input_embeddings(mx.array([[AUDIO_ID, AUDIO_ID]]),
                                   input_features=feats,
                                   input_features_mask=mask)
    with pytest.raises(ValueError, match="no image input"):
        model.get_input_embeddings(mx.array([[1]]), mx.zeros((1, 3, 4, 4)))


def test_prefix_guard_keeps_audio_slots_whole(container):
    model = container[0]
    install_media_token_ids()
    media = _apc.multimodal_token_ids_from_config(model.language_model.config)
    assert AUDIO_ID in media
    ids = [1, AUDIO_ID, AUDIO_ID, AUDIO_ID, 2]
    assert not _apc.prefix_leaves_text_only_suffix(ids, 2, media)
    assert _apc.prefix_leaves_text_only_suffix(ids, 4, media)


def test_loader_rows():
    from gmlx.load.loader import (
        _F16_KEEP_BY_MODEL_TYPE,
        _FP32_KEEP_BY_MODEL_TYPE,
    )
    from gmlx.load.mtp_target import _VLM_SPEC_MODEL_TYPE_ALIASES
    from gmlx.models.owned import OWNED_VLM_MODELS

    assert set(_FP32_KEEP_BY_MODEL_TYPE["nemotron_h"]) <= set(
        _FP32_KEEP_BY_MODEL_TYPE[TYPE])
    assert _F16_KEEP_BY_MODEL_TYPE[TYPE] == ("audio_tower.",)
    assert _VLM_SPEC_MODEL_TYPE_ALIASES[TYPE] == "nemotron_h"
    assert OWNED_VLM_MODELS[TYPE] == "gmlx.models.nemotron_h.transcribe_model"
    assert vlm.resolve_vlm_model_type(
        "nemotron_h_moe", {"general.architecture": "pee"}) == TYPE


# gmlx run

def _run_args(argv):
    parser = cli._build_parser()
    return parser, parser.parse_args(argv)


@pytest.fixture()
def speech_run(monkeypatch, tmp_path):
    """A run whose --mmproj is a speech encoder and whose clips are 2.5 s."""
    monkeypatch.setattr(cli, "is_speech_encoder", lambda p: bool(p))
    import gmlx._vendor.mlx_vlm.utils as vu

    monkeypatch.setattr(vu, "load_audio",
                        lambda path, sr: np.zeros(40_000, np.float32))
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    return str(model)


def test_run_transcribes_audio_without_a_prompt(speech_run):
    argv = [speech_run, "--mmproj", "enc.gguf", "--audio", "a.wav"]
    parser, args = _run_args(argv)
    assert cli.transcribes(args, parser, argv)
    cli.apply_transcription_defaults(args, parser, argv)
    assert args.prompt == transcription.TRANSCRIBE
    assert (args.temp, args.top_p, args.thinking) == (0.0, 1.0, "off")
    assert args.max_tokens == 15 + 24


def test_run_keeps_the_flags_the_user_passed(speech_run):
    argv = [speech_run, "--mmproj", "enc.gguf", "--audio", "a.wav",
            "--temp", "0.7", "--max-tokens", "9", "--thinking", "on"]
    parser, args = _run_args(argv)
    cli.apply_transcription_defaults(args, parser, argv)
    assert args.prompt == transcription.TRANSCRIBE
    assert (args.temp, args.top_p, args.thinking, args.max_tokens) == (
        0.7, 1.0, "on", 9)


def test_run_with_a_prompt_asks_the_model(speech_run):
    argv = [speech_run, "--mmproj", "enc.gguf", "--audio", "a.wav",
            "--prompt", "Who speaks?"]
    parser, args = _run_args(argv)
    assert not cli.transcribes(args, parser, argv)
    before = vars(args).copy()
    cli.apply_transcription_defaults(args, parser, argv)
    assert vars(args) == before


def test_room_depth_counts_the_audio_tokens(speech_run):
    argv = [speech_run, "--mmproj", "enc.gguf", "--audio", "a.wav",
            "--no-speculative"]
    parser, args = _run_args(argv)
    args._transcribes = True
    budget = transcription.output_budget(2.5, 32)
    prompt = len(args.prompt.encode()) // 2
    assert cli.run_room_depth(args) == (
        prompt + cli._TEMPLATE_TOKENS + budget + 32)
    args._transcribes = False
    assert cli.run_room_depth(args) is None
    args.max_tokens = 50
    assert cli.run_room_depth(args) == prompt + cli._TEMPLATE_TOKENS + 50 + 32


def _meta(monkeypatch, checkpoint="ck"):
    import gmlx.load.discovery as disc

    monkeypatch.setattr(disc, "header_meta", lambda p: {
        "arch": "nemotron_h_moe", "speech_checkpoint": checkpoint})
    return disc


def test_pairing_finds_the_encoder_beside_the_model(monkeypatch, capsys):
    disc = _meta(monkeypatch)
    monkeypatch.setattr(disc, "find_speech_encoder",
                        lambda p: "/m/Encoder-F16.gguf")
    _, args = _run_args(["/m/model.gguf", "--audio", "a.wav"])
    cli.pair_speech_encoder(args, always=False)
    assert args.mmproj == "/m/Encoder-F16.gguf"
    assert "[asr] speech encoder: Encoder-F16.gguf" in capsys.readouterr().err

    _, args = _run_args(["/m/model.gguf"])
    cli.pair_speech_encoder(args, always=False)
    assert args.mmproj is None
    cli.pair_speech_encoder(args, always=True)
    assert args.mmproj == "/m/Encoder-F16.gguf"

    _, args = _run_args(["/m/model.gguf", "--audio", "a.wav", "--no-mmproj"])
    cli.pair_speech_encoder(args, always=True)
    assert args.mmproj is None


def test_the_encoder_pairs_once_per_folder_change(monkeypatch, tmp_path):
    import gmlx.load.discovery as disc

    model, enc, other = (tmp_path / n for n in (
        "Transcribe-NVFP4.gguf", "Transcribe-Encoder-F16.gguf", "x.gguf"))
    for f in (model, enc, other):
        f.write_bytes(b"")
    heads = {str(model): {"arch": "nemotron_h_moe", "speech_checkpoint": "ck"},
             str(enc): {"arch": "pee", "speech_checkpoint": "ck"},
             str(other): {"arch": "pee", "speech_checkpoint": "other"}}
    reads = []

    def meta(p):
        reads.append(str(p))
        return heads.get(str(p))

    monkeypatch.setattr(disc, "header_meta", meta)
    monkeypatch.setattr(disc, "_SPEECH_PAIRS", {})
    assert disc.find_speech_encoder(str(model)) == str(enc)
    n = reads.count(str(enc))
    assert disc.find_speech_encoder(str(model)) == str(enc)
    assert reads.count(str(enc)) == n
    heads[str(model)] = {"arch": "nemotron_h_moe"}      # Lightning: no pair
    assert disc.find_speech_encoder(str(model)) is None


def test_serve_on_a_model_path_pairs_its_encoder(monkeypatch):
    import argparse

    import gmlx.load.discovery as disc
    from gmlx.serve import server

    monkeypatch.setattr(disc, "find_speech_encoder",
                        lambda p: "/m/Encoder-F16.gguf" if "Transcribe" in p else None)
    ap = argparse.ArgumentParser()
    server._add_serve_args(ap)
    for path, flag, want in (("/m/Transcribe.gguf", None, "/m/Encoder-F16.gguf"),
                             ("/m/Transcribe.gguf", "/m/mine.gguf", "/m/mine.gguf"),
                             ("/m/Lightning.gguf", None, None)):
        a = ap.parse_args([])
        a.model, a.mmproj = path, flag
        cfg = server._single_model_cfg(a)
        assert cfg.models[cfg.defaults.model].mmproj == want


def test_pairing_refuses_asr_and_a_missing_encoder(monkeypatch):
    disc = _meta(monkeypatch)
    monkeypatch.setattr(disc, "find_speech_encoder", lambda p: None)
    _, args = _run_args(["/m/model.gguf", "--audio", "a.wav"])
    with pytest.raises(ValueError, match="needs this model's speech encoder"):
        cli.pair_speech_encoder(args, always=False)
    _, args = _run_args(["/m/model.gguf"])
    cli.pair_speech_encoder(args, always=True)
    assert args.mmproj is None
    _, args = _run_args(["/m/model.gguf", "--asr", "w.gguf"])
    with pytest.raises(ValueError, match="--mmproj, not --asr"):
        cli.pair_speech_encoder(args, always=True)


def test_pairing_leaves_other_models_alone(monkeypatch):
    _meta(monkeypatch, checkpoint=None)
    _, args = _run_args(["/m/model.gguf", "--audio", "a.wav", "--asr", "w"])
    cli.pair_speech_encoder(args, always=True)
    assert args.mmproj is None


# The real tokenizer

def test_speaker_and_event_tags_survive_decoding():
    folder = os.environ.get("GMLX_TEST_NEMOTRON_TRANSCRIBE")
    if not folder:
        pytest.skip("set GMLX_TEST_NEMOTRON_TRANSCRIBE to the model folder")
    from gmlx._vendor.mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer
    from gmlx.distill.tokens import tokenizer_from_gguf
    from gmlx.load.discovery import header_meta

    models = [p for p in sorted(Path(folder).expanduser().glob("*.gguf"))
              if (header_meta(str(p)) or {}).get("arch") == "nemotron_h_moe"]
    if not models:
        pytest.skip(f"no nemotron_h_moe GGUF in {folder}")
    tok = tokenizer_from_gguf(str(models[0]))
    assert tok.convert_ids_to_tokens([AUDIO_ID, 100, 101]) == [
        "<SPECIAL_19>", "<spk:0>", "<spk:1>"]
    text = "<spk:0> Hello. <spk:1> Hi <ev:breath> there."
    ids = tok.encode(text, add_special_tokens=False)
    assert ids[0] == 100 and AUDIO_ID not in ids
    detok = NaiveStreamingDetokenizer(tok)
    detok.reset()
    for i in ids:
        detok.add_token(i)
    detok.finalize()
    assert detok.text == text


# gmlx chat

def test_run_reads_each_clip_once(speech_run, monkeypatch):
    import gmlx._vendor.mlx_vlm.utils as vu

    reads = []
    monkeypatch.setattr(vu, "load_audio", lambda path, sr: (
        reads.append(path), np.zeros(40_000, np.float32))[1])
    argv = [speech_run, "--mmproj", "enc.gguf", "--audio", "a.wav,b.wav"]
    parser, args = _run_args(argv)
    cli.apply_transcription_defaults(args, parser, argv)
    assert cli.speech_audio(args) == (5.0, 2 * 32)
    assert len(cli.speech_clips(args) or []) == 2
    assert reads == ["a.wav", "b.wav"]


def test_chat_transcription_turn_keeps_the_knobs_the_user_set():
    from gmlx.tui.chat import _transcription_sampling

    assert _transcription_sampling(set()) == {
        "temp": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
        "repetition_penalty": 1.0}
    assert _transcription_sampling({"temp", "top_k"}) == {
        "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0}


def test_chat_reads_each_clip_once_and_keeps_the_limits(monkeypatch):
    import gmlx._vendor.mlx_vlm.utils as vu
    from gmlx.tui.chat import _speech_clips

    reads = []
    monkeypatch.setattr(vu, "load_audio", lambda path, sr: (
        reads.append(path), np.zeros(1280, np.float32))[1])
    proc = _processor()
    held, new = _speech_clips(proc, ["a.wav"], ["b.wav"])
    assert reads == ["a.wav", "b.wav"]
    held, new = _speech_clips(proc, held + new, ["c.wav"])
    assert reads == ["a.wav", "b.wav", "c.wav"] and len(held) == 2
    with pytest.raises(ValueError, match="7 audio clips"):
        _speech_clips(proc, held + new, ["d"] * 4)


def test_chat_counts_the_audio_tokens_of_the_prompt():
    from gmlx.tui.chat import _speech_prompt_tokens

    proc = _processor()
    clips = [np.zeros(1280, np.float32), np.zeros(2560, np.float32)]
    assert _speech_prompt_tokens(proc, "hi <|audio|> hi <|audio|>",
                                 clips) == 2 + 1 + 2


def test_chat_context_is_the_trained_one_with_the_encoder(monkeypatch):
    from argparse import Namespace

    from gmlx.tui.chat import _speech_context

    monkeypatch.setattr(cli, "is_speech_encoder", lambda p: bool(p))
    assert _speech_context(Namespace(mmproj="enc.gguf")) == 65_536
    assert _speech_context(Namespace(mmproj=None)) is None


def test_profile_follows_the_card():
    from gmlx.gen import profiles

    family = profiles.detect_family(
        "nemotron_h_moe", "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
    assert family == "nemotron-transcribe"
    assert profiles.groups_for(family)["sampling"] == {
        "temperature": 0.6, "top_p": 0.95, "top_k": 0,
        "thinking_budget": 16384}
    instruct = profiles.groups_for(family, "instruct")
    assert instruct["sampling"]["temperature"] == 0.0
    assert instruct["chat_template_kwargs"] == {"enable_thinking": False}
    assert profiles.detect_family(
        "nemotron_h_moe", "NVIDIA-Nemotron-3-Nano-30B-A3B") == "nemotron"


# The server's encoded clips

def _encoded(model, mel, pcm, digest="d"):
    from gmlx.models.nemotron_h.transcribe_model import EncodedClip, drain

    out = drain(model.encode_steps(speech.features(pcm, mel),
                                   speech.num_frames(pcm.size),
                                   speech.num_audio_tokens(pcm.size)))
    return EncodedClip.of(pcm.size, digest, out.embeds, out.speaker_probs)


def test_encoded_clips_fill_the_slots_as_features_do(container):
    model, mel, pcm = container
    clips = [pcm, pcm[:8000]]
    feats, mask = _inputs(mel, clips)
    n = [speech.num_audio_tokens(c.size) for c in clips]
    ids = mx.array([[5] + [AUDIO_ID] * n[0] + [6] + [AUDIO_ID] * n[1] + [7]])
    want = model.get_input_embeddings(
        ids, input_features=feats, input_features_mask=mask).inputs_embeds
    encoded = tuple(_encoded(model, mel, c, str(i)) for i, c in enumerate(clips))
    got = model.get_input_embeddings(ids, audio_clips=encoded).inputs_embeds
    assert mx.array_equal(got, want)
    tail = mx.array([[AUDIO_ID] * (n[1] + 2)])
    rows = model.get_input_embeddings(tail, audio_clips=encoded).inputs_embeds
    assert mx.array_equal(rows[0], mx.concatenate(
        [encoded[0].embeds[-2:], encoded[1].embeds]))
    with pytest.raises(ValueError, match="placeholder tokens but"):
        model.get_input_embeddings(mx.array([[AUDIO_ID] * (sum(n) + 1)]),
                                   audio_clips=encoded)


def test_an_encoded_clip_keeps_its_rows_through_the_audio_loader(container):
    from gmlx._vendor.mlx_vlm.utils import load_audio
    from gmlx.models.nemotron_h.transcribe_model import EncodedClip

    model, mel, pcm = container
    clip = _encoded(model, mel, pcm, "abc")
    loaded = load_audio(clip, sr=16000)
    assert isinstance(loaded, EncodedClip) and loaded.digest == "abc"
    assert loaded.embeds is clip.embeds and len(loaded) == pcm.size
    assert clip.nbytes == 4 * pcm.size and clip.base is not None


def test_processor_hands_encoded_clips_over(container):
    from gmlx.models.nemotron_h.transcribe_model import EncodedClip

    proc = _processor()
    clip = EncodedClip.of(2560, "abc", mx.zeros((2, 24)), np.zeros((2, 8)))
    out = proc(text=["hi <|audio|>"], audio=[clip])
    assert [id(c) for c in out["audio_clips"]] == [id(clip)]
    assert out["audio_digests"] == ["abc"]
    assert "input_features" not in out
    assert np.asarray(out["input_ids"]).tolist()[0].count(AUDIO_ID) == 2
    with pytest.raises(ValueError, match="all encoded or none"):
        proc(text=["<|audio|> <|audio|>"], audio=[clip, np.zeros(1280)])
    with pytest.raises(ValueError, match="7 audio clips"):
        proc(text=["<|audio|>" * 7], audio=[clip] * 7)


def test_audio_preflight_prices_only_clips_to_encode(container):
    model, mel, pcm = container
    clip = _encoded(model, mel, pcm)
    tokens = speech.num_audio_tokens(pcm.size)
    assert model.audio_preflight([clip]) == (tokens, 0)
    hour = np.broadcast_to(np.float32(0), (3600 * 16000,))
    n, size = model.audio_preflight([clip, hour])
    assert n == tokens + 45_000
    cfg = model.audio_tower.cfg
    assert size == 45_000 * 2 * max(cfg.proj_dim * 2, cfg.asr.d_model * 4) + (160 << 20)


def test_mtp_prefill_reads_audio_prompts_as_media(container):
    from types import SimpleNamespace

    from gmlx.spec.mtp_prefill import _carries_media

    lm = container[0].language_model
    batch = SimpleNamespace(model=lm, _prompt_kwargs={},
                            _input_ids=mx.array([[1, AUDIO_ID, 2]]))
    assert _carries_media(batch)
    batch._input_ids = mx.array([[1, 2, 3]])
    assert not _carries_media(batch)
    batch._prompt_kwargs = {"audio_clips": ("clip",)}
    assert _carries_media(batch)


@pytest.fixture()
def engine(monkeypatch):
    """``run_on_engine`` driven in place: it records each job's steps."""
    import gmlx.serve.engine_jobs as ej
    from gmlx.serve.patches import native_audio

    seen = {"steps": [], "encoding": []}

    def run(rg, job, *, stop, what, stepwise, **kw):
        assert stepwise and what == "the audio encoder"
        seen["encoding"].append(native_audio.encoding_requests(rg))
        gen, steps = job(rg, stop.is_set), 0
        while True:
            try:
                next(gen)
                steps += 1
            except StopIteration as done:
                seen["steps"].append(steps)
                return done.value
    monkeypatch.setattr(ej, "run_on_engine", run)
    return seen


def test_encode_runs_a_window_per_step_and_caches_the_rows(container, engine):
    import threading

    from gmlx.serve.patches import native_audio

    model, mel, pcm = container
    rg = object()
    clips = native_audio.encode(rg, model, [pcm, pcm], threading.Event(), None)
    wins = len(model.audio_tower.windows(speech.num_audio_tokens(pcm.size)))
    assert engine["steps"] == [2 * wins + 1]          # one clip, encoded once
    assert engine["encoding"] == [1] and native_audio.encoding_requests(rg) == 0
    assert clips[0].digest == clips[1].digest and clips[0].embeds is clips[1].embeds
    want = _encoded(model, mel, pcm)
    assert mx.array_equal(clips[0].embeds, want.embeds)
    again = native_audio.encode(rg, model, [pcm], threading.Event(), None)
    assert len(engine["steps"]) == 1 and again[0].embeds is clips[0].embeds
    assert native_audio.held_bytes() >= clips[0].embeds.nbytes


def test_row_cache_lets_the_oldest_go_past_its_cap():
    from gmlx.serve.patches.native_audio import _RowCache

    cache = _RowCache(cap=2 * 400)
    for d in "abc":
        cache.put(d, 1, mx.zeros((100,)), None)
    assert list(cache.entries) == ["b", "c"] and cache.bytes == 800
    assert cache.get("a") is None and cache.get("b") is not None
    cache.put("d", 1, mx.zeros((100,)), None)
    assert list(cache.entries) == ["b", "d"]
    cache.put("big", 1, mx.zeros((1000,)), None)
    assert "big" not in cache.entries


def test_clip_limits():
    transcription.check_clips([16000] * 6)
    with pytest.raises(ValueError, match="7 audio clips"):
        transcription.check_clips([1] * 7)
    with pytest.raises(ValueError, match="3600.1 s"):
        transcription.check_clips([3600 * 16000 + 1600])
