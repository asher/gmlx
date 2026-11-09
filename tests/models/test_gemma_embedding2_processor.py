"""EmbeddingGemma 2's media preprocessing and token layout.

The goldens come from transformers 5.22.2
(scripts/embedder_oracle/embeddinggemma2_processor.py): the image
processor's resize and counts, the video processor's frame sampling, and
the audio extractor's features and masks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gmlx.models.embedders import gemma_embedding2_processor as proc

_ORACLE = Path(__file__).resolve().parents[1] / "fixtures" / "embeddinggemma2_oracle"


@pytest.fixture(scope="module")
def oracle():
    with np.load(_ORACLE / "processor.npz") as f:
        return {k: f[k] for k in f.files}


def test_vision_token_counts_match_the_reference(oracle):
    for h, w, budget, tokens in oracle["sizes"].tolist():
        assert proc.vision_tokens(h, w, budget) == tokens, (h, w, budget)
        assert tokens <= budget


def test_image_resize_matches_the_reference(oracle):
    src = oracle["image.src"]
    for budget in proc.VISION_BUDGETS:
        assert proc.vision_tokens(*src.shape[:2], budget) == int(
            oracle[f"image.{budget}.tokens"])
    want = oracle["image.70.resized"].astype(np.float32) / 255.0
    got = proc.resize(src[None], 70)[0].transpose(1, 2, 0)
    assert got.shape == want.shape
    # PIL's bicubic and torchvision's differ by a few levels at most.
    diff = np.abs(got - want)
    assert diff.mean() < 1.0 / 255 and diff.max() < 8.0 / 255


def test_frame_sampling_matches_the_reference(oracle):
    for i, (total, fps, duration) in enumerate(oracle["frames"].tolist()):
        got = proc.sample_frames(total, fps / 1000, duration / 1000)
        np.testing.assert_array_equal(got, oracle[f"frames.{i}"])


def test_frame_sampling_without_a_frame_rate_keeps_every_frame_capped():
    assert proc.sample_frames(10, None, None).tolist() == list(range(10))
    assert len(proc.sample_frames(100, None, None)) == 32


def test_audio_mask_and_tokens_match_the_extractor(oracle):
    # The reference extractor cuts a clip at 30 s first, as cut_audio does.
    for n in oracle["audio.lengths"].tolist():
        want = oracle[f"audio.{n}.mask"]
        cut = min(n, proc.AUDIO_CUT)
        np.testing.assert_array_equal(proc.audio_mask(cut), want)
        assert proc.audio_tokens(cut) == proc.audio_tokens_from_mask(want)
    # 30 s gives the reference's 750 tokens.
    assert proc.audio_tokens(proc.AUDIO_CUT) == 750


def test_audio_tokens_at_edge_lengths_equal_the_vendored_extractor():
    rng = np.random.default_rng(0)
    for n in (1, 159, 160, 161, 320, 321, 322, 641, 999, 16_001):
        clip = rng.standard_normal(n).astype(np.float32) * 0.1
        _, mask = proc.audio_features(clip)
        assert proc.audio_tokens(n) == proc.audio_tokens_from_mask(mask[0]), n


def test_audio_features_match_the_reference(oracle):
    wave = oracle["audio.wave"].astype(np.float32) / 32768.0
    for n in oracle["audio.lengths"].tolist():
        if f"audio.{n}.features" not in oracle:
            continue
        feats, mask = proc.audio_features(wave[:n])
        np.testing.assert_array_equal(mask[0], oracle[f"audio.{n}.mask"])
        np.testing.assert_allclose(feats[0], oracle[f"audio.{n}.features"],
                                   rtol=1e-4, atol=1e-4)


def test_cut_audio_copies_the_first_30_seconds():
    long = np.zeros(proc.AUDIO_CUT + 10, dtype=np.float32)
    cut = proc.cut_audio(long, proc.Options())
    assert len(cut) == proc.AUDIO_CUT and not np.shares_memory(cut, long)
    lifted = proc.cut_audio(long, proc.Options(audio_cut=False))
    assert lifted is long


@pytest.mark.parametrize("kwargs, image, frame, cut", [
    (None, 280, 140, True),
    ({}, 280, 140, True),
    ({"max_soft_tokens": 560}, 560, 560, True),
    ({"images_kwargs": {"max_soft_tokens": 70}}, 70, 140, True),
    ({"videos_kwargs": {"max_soft_tokens": 1120}}, 280, 1120, True),
    ({"max_soft_tokens": 560, "videos_kwargs": {"max_soft_tokens": 70}},
     560, 70, True),
    ({"videos_kwargs": {"max_soft_tokens": 70}, "max_soft_tokens": 560},
     560, 70, True),
    ({"audio_kwargs": {"truncation": False}}, 280, 140, False),
])
def test_processor_options(kwargs, image, frame, cut):
    assert proc.parse_options(kwargs) == proc.Options(image, frame, cut)


@pytest.mark.parametrize("kwargs, words", [
    ({"max_soft_tokens": 300}, "must be one of 70, 140, 280, 560, 1120"),
    ({"max_soft_tokens": True}, "must be one of"),
    ({"images_kwargs": {"max_soft_tokens": "280"}}, "must be one of"),
    ({"images_kwargs": {"do_resize": False}}, "takes max_soft_tokens"),
    ({"audio_kwargs": {"truncation": "no"}}, "true or false"),
    ({"audio_kwargs": {"max_length": 5}}, "takes truncation"),
    ({"fps": 2}, "unknown mm_processor_kwargs key 'fps'"),
    ([], "must be an object"),
    ({"videos_kwargs": 3}, "must be an object"),
])
def test_processor_options_refused(kwargs, words):
    with pytest.raises(ValueError, match=words):
        proc.parse_options(kwargs)


class _Tok:
    """A tokenizer that maps each character to its code point, and the
    special token strings to their ids, unless they are split."""

    SPECIAL = {"<|image|>": 900, "<|audio|>": 901, "<|video|>": 902,
               "<|image>": 903, "<image|>": 904}

    def encode(self, text, add_special_tokens=False, split_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for s, t in self.SPECIAL.items():
                if not split_special_tokens and text.startswith(s, i):
                    ids.append(t)
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids


_M = proc.Markers(bos=2, eos=1, boi=903, eoi=904, boa=905, eoa=906,
                  image=900, video=902, audio=901)


def test_layout_puts_media_blocks_between_text_and_wraps_bos_eos():
    ids = proc.layout(_Tok(), ["ab", proc.Media("image", 2), "c",
                               proc.Media("audio", 3),
                               proc.Media("video", 1, frames=2)], _M)
    assert ids == ([2, 97, 98, 903, 900, 900, 904, 99, 905, 901, 901, 901,
                    906] + [903, 902, 904] * 2 + [1])


def test_layout_joins_the_prompt_with_the_first_text():
    ids = proc.layout(_Tok(), ["xy"], _M, prompt="p: ")
    assert ids == [2] + [ord(c) for c in "p: xy"] + [1]
    no_special = proc.layout(_Tok(), ["xy"], _M, add_special_tokens=False)
    assert no_special == [ord("x"), ord("y")]


def test_layout_reads_placeholders_as_text_without_media():
    ids = proc.layout(_Tok(), ["see <|image|> and <|image>"], _M)
    assert 900 not in ids and 903 in ids
    assert ids[1:5] == [ord(c) for c in "see "]


def test_layout_places_media_at_named_placeholders():
    ids = proc.layout(_Tok(), [proc.Media("image", 1), "a<|image|>b"], _M)
    assert ids == [2, 97, 903, 900, 904, 98, 1]


@pytest.mark.parametrize("parts, words", [
    ([proc.Media("image", 1), "<|image|><|image|>"], "more <|image|>"),
    ([proc.Media("image", 1), proc.Media("audio", 1), "<|image|>"],
     "more audio parts"),
])
def test_layout_refuses_unpaired_placeholders(parts, words):
    with pytest.raises(ValueError, match=words):
        proc.layout(_Tok(), parts, _M)
