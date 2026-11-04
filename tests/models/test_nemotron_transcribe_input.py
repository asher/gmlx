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
    real = model.audio_tower.__call__

    class Spy:
        cfg = model.audio_tower.cfg

        def __call__(self, *a):
            calls.append(a[2])
            return real(*a)

    model.audio_tower = Spy()
    tail = model.audio_features(feats, mask, n[1] + 3)
    assert calls == [n[1], n[0]]
    np.testing.assert_allclose(np.asarray(tail), np.asarray(full[-(n[1] + 3):]),
                               rtol=1e-5, atol=1e-6)
    calls.clear()
    model.audio_features(feats, mask, n[1])
    assert calls == [n[1]]


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
