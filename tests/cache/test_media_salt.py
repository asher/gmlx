"""Per-prefix image salts: a turn that adds an image restores the prefix an
earlier turn stored, and a prefix whose images differ never restores."""

import importlib
import logging
import pickle
import queue
import threading
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from gmlx._vendor.mlx_vlm import apc as _apc
from gmlx._vendor.mlx_vlm.utils import StoppingCriteria

import gmlx.lora_rows as lora_rows
import gmlx.load.vlm as gvlm
import gmlx.spec.mtp_prefill as mtp_prefill
from gmlx.cache import fresh_gate, snapshot
from gmlx.cache.apc_manager import GmlxAPCManager
from gmlx.cache.apc_media import install_media_token_ids
from gmlx.cache.media_salt import (
    MediaSalt,
    image_spans,
    image_token_ids,
    install_media_salt,
    salt_at,
    salt_windows,
)
from gmlx.cache.prefix_cache import SpecPrefixCache
from gmlx.gen import media_spans

from test_apc_media import _HIDDEN, _gemma4_config
from test_ckpt_tier import KVCache

ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
server = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")

# Warm and cold prefill the suffix in different GEMM shapes, and f32 GEMM
# runs at TF32 on the GPU, so their logprobs differ by about 4e-3 on a
# text-only prompt too. A wrong restore moves them by 0.5 or more.
_ATOL = 1e-2
_V = 128
_BS = 16


# -- the salt ---------------------------------------------------------------

def test_a_prefix_keys_on_the_images_that_start_inside_it():
    s = MediaSalt(7, [(40, 111), (10, 222)])
    assert s.spans == ((10, 222), (40, 111))
    assert s.at(0) == s.at(10) == 7
    assert s.at(11) == s.at(40) == MediaSalt(7, [(10, 222)]).at(11)
    assert s.at(41) == int(s) == s.at(10 ** 6)
    assert len({s.at(10), s.at(11), s.at(41)}) == 3
    # The start and the hash both count.
    assert MediaSalt(7, [(10, 222)]).at(50) != MediaSalt(7, [(11, 222)]).at(50)
    assert MediaSalt(7, [(10, 222)]).at(50) != MediaSalt(7, [(10, 223)]).at(50)


def test_xor_masks_every_prefix_and_keeps_the_type():
    s = MediaSalt(7, [(10, 222)])
    m = s ^ 5
    assert isinstance(m, MediaSalt) and isinstance(5 ^ s, MediaSalt)
    assert int(m) == int(s) ^ 5
    assert m.at(4) == 7 ^ 5 and m.at(20) == s.at(20) ^ 5
    assert (m ^ 5).at(20) == s.at(20)
    back = pickle.loads(pickle.dumps(m))
    assert isinstance(back, MediaSalt) and back.at(4) == m.at(4)
    assert salt_at(99, 4) == 99 and salt_at(m, 4) == m.at(4)


def test_windows_run_longest_first_inside_the_caller_bounds():
    s = MediaSalt(7, [(10, 1), (40, 2)])
    assert salt_windows(s) == [(s.at(41), 40, None), (s.at(11), 10, 40),
                               (s.at(1), 0, 10)]
    assert salt_windows(s, 20, 30) == [(s.at(11), 20, 30)]
    assert salt_windows(s, 0, 0) == salt_windows(s)
    assert salt_windows(99, 3, 8) == [(99, 3, 8)]


def _cfg():
    """Image id 9, and a Gemma-4 style vision config whose image of H x W
    pixels expands to H * W tokens."""
    return SimpleNamespace(image_token_id=9, vision_config=SimpleNamespace(
        patch_size=1, pooling_kernel_size=1))


def test_image_spans_map_runs_to_images_in_order():
    ids = mx.array([[1, 9, 9, 2, 9, 9, 3]])
    a, b = mx.full((3, 1, 2), 1.0), mx.full((3, 1, 2), 2.0)
    stacked = image_spans(ids, {"pixel_values": mx.stack([a, b])}, _cfg())
    listed = image_spans(ids, {"pixel_values": [a, b]}, _cfg())
    assert [s for s, _h in stacked] == [1, 4]
    assert stacked == listed
    assert stacked[0][1] != stacked[1][1]
    swapped = image_spans(ids, {"pixel_values": mx.stack([b, a])}, _cfg())
    assert swapped[0][1] == stacked[1][1] and swapped[1][1] == stacked[0][1]
    assert image_spans(mx.array([[1, 2, 3]]), {"pixel_values": a[None]},
                       _cfg()) is None
    assert image_spans(ids, {}, _cfg()) is None


def test_image_spans_split_flattened_patches_by_grid():
    # 8 and 4 patches, 4 per token.
    ids = mx.array([[1, 9, 9, 2, 9, 3]])
    pv = mx.arange(12 * 4, dtype=mx.float32).reshape(12, 4)
    grid = mx.array([[1, 2, 4], [1, 2, 2]])
    spans = image_spans(ids, {"pixel_values": pv, "image_grid_thw": grid},
                        _cfg())
    assert [s for s, _h in spans] == [1, 4]
    # The grid row is part of an image's hash: the same patches under a
    # different grid tokenize differently.
    other = image_spans(ids, {"pixel_values": pv,
                              "image_grid_thw": mx.array([[1, 1, 8],
                                                          [1, 2, 2]])},
                        _cfg())
    assert other[0][1] != spans[0][1] and other[1][1] == spans[1][1]


def test_qwen_runs_pair_by_the_image_processor_merge_size():
    """Qwen3VLProcessor expands an image to t*h*w of its grid over the
    image processor's merge_size squared, whatever the vision config
    says."""
    from gmlx._vendor.mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
        Qwen3VLImageProcessor,
    )
    from PIL import Image

    ip = Qwen3VLImageProcessor(patch_size=14, merge_size=1, min_pixels=0,
                               max_pixels=14 * 14 * 64)
    raw = {k: mx.array(v) for k, v in ip([
        Image.new("RGB", (56, 28), (200, 0, 0)),
        Image.new("RGB", (28, 42), (0, 200, 0))]).items()}
    assert raw["image_grid_thw"].tolist() == [[1, 2, 4], [1, 3, 2]]
    cfg = SimpleNamespace(image_token_id=9, vision_config=SimpleNamespace(
        spatial_merge_size=2))
    ids = mx.array([[1] + [9] * 8 + [2] + [9] * 6 + [3]])
    spans = image_spans(ids, raw, cfg, SimpleNamespace(image_processor=ip))
    assert [s for s, _h in spans] == [1, 10]
    # The vision config's merge size alone gives 2 and 1 tokens.
    assert len(image_spans(ids, raw, cfg)) == 1


def _kimi(*sizes):
    """The Kimi K2.5 processor's rows for solid images of ``sizes`` (width,
    height), and ids holding each image's run between text tokens."""
    from gmlx._vendor.mlx_vlm.models.kimi_k25.processing_kimi_k25 import (
        KimiK25ImageProcessor,
    )
    from PIL import Image

    out = KimiK25ImageProcessor().preprocess(
        [Image.new("RGB", size, (200, 30, 30)) for size in sizes])
    ids = [1]
    for h, w in np.asarray(out["image_grid_hws"]).tolist():
        ids += [9] * (h * w // 4) + [2]
    return mx.array([ids]), {"pixel_values": out["pixel_values"],
                             "image_grid_hws": out["image_grid_hws"]}


def test_kimi_images_of_swapped_sides_key_apart():
    """448x224 and 224x448 solid images patchify to the same pixel rows and
    the same token count; only the grid tells them apart."""
    cfg = SimpleNamespace(image_token_index=9)
    ids, wide = _kimi((448, 224))
    tall_ids, tall = _kimi((224, 448))
    assert ids.tolist() == tall_ids.tolist()
    assert mx.array_equal(wide["pixel_values"], tall["pixel_values"])
    assert image_spans(ids, wide, cfg) != image_spans(ids, tall, cfg)
    # Two images split by the grid: the second keys as it does alone.
    both_ids, both = _kimi((448, 224), (224, 448))
    spans = image_spans(both_ids, both, cfg)
    assert [s for s, _h in spans] == [1, 130]
    assert spans[0][1] == image_spans(ids, wide, cfg)[0][1]
    assert spans[1][1] == image_spans(ids, tall, cfg)[0][1]


def test_deepseek_v4_images_key_on_their_vit_grid():
    """A (8, 9) and a (9, 8) vit grid both pool to a 3x3 block, so a solid
    image gives the same pixel rows and ids; only ``image_meta`` differs."""
    from gmlx.models.deepseek_v4.image_block import (
        N_BLOCK_TYPES,
        build_image_block,
    )

    vocab = 100
    types, _perm = build_image_block(3, 3, 1)
    block = [vocab + int(t) for t in types]
    ids = mx.array([[1] + block + [2]])
    cfg = SimpleNamespace(image_token_id=7, media_token_ids=[
        vocab + t for t in range(N_BLOCK_TYPES)])
    pv = mx.full((72, 3 * 14 * 14), 0.25)

    def raw(meta):
        return {"pixel_values": pv, "image_meta": mx.array([meta]),
                "image_spans": [[1, 1 + len(block)]]}

    a = image_spans(ids, raw([8, 9, 3, 3]), cfg)
    b = image_spans(ids, raw([9, 8, 3, 3]), cfg)
    assert a[0][0] == b[0][0] == 1
    assert a != b


def _starts(spans):
    return [s for s, _h in spans or ()]


def test_muse_glimmer_runs_pair_by_the_image_sizes():
    """An image resized to h x w pixels expands to (h // cell) *
    (w // cell) patch ids between its start and end markers, with cell =
    patch * spatial merge."""
    from gmlx._vendor.mlx_vlm.models.base import to_mlx
    from PIL import Image

    from gmlx.load.vlm import _MuseGlimmerGgufImageProcessor

    ip = _MuseGlimmerGgufImageProcessor([0.5] * 3, [0.5] * 3, patch_size=14,
                                        spatial_merge_size=2)
    raw = dict(to_mlx(ip([Image.new("RGB", (60, 90), (200, 0, 0)),
                          Image.new("RGB", (120, 50), (0, 200, 0))])))
    counts = [ip.soft_tokens(h, w) for h, w in raw["image_sizes"].tolist()]
    assert counts == [6, 10]
    cfg = SimpleNamespace(image_token_id=9, image_token_index=9,
                          vision_config=SimpleNamespace(
                              patch_size=14, spatial_merge_size=2))
    processor = SimpleNamespace(image_processor=ip)

    def ids(a, b):
        return mx.array([[1, 5] + [9] * a + [6, 2, 5] + [9] * b + [6, 3]])

    for proc in (processor, None):
        assert _starts(image_spans(ids(6, 10), raw, cfg, proc)) == [2, 11]
    # Runs of other lengths hold other images.
    assert len(image_spans(ids(10, 6), raw, cfg, processor)) == 1


def test_deepseek_v41_blocks_pair_by_the_image_meta():
    """Each image's block, START (IMAGE * n_llm_w + NEWLINE) * n_llm_h END,
    is one run of media ids."""
    from gmlx._vendor.mlx_vlm.models.base import to_mlx
    from PIL import Image

    from gmlx.load.vlm import _DeepseekV41VGgufImageProcessor
    from gmlx.models.deepseek_v41.image_block import (
        N_BLOCK_TYPES,
        build_image_block,
    )

    vocab = 100
    cfg = SimpleNamespace(image_token_id=7, image_token_index=7,
                          media_token_ids=[vocab + t
                                           for t in range(N_BLOCK_TYPES)])
    ip = _DeepseekV41VGgufImageProcessor(patch_size=14, downsample_ratio=3,
                                         max_n_token=1024, min_pixels=0)
    images = [Image.new("RGB", (84, 126), (200, 0, 0)),
              Image.new("RGB", (168, 84), (0, 200, 0))]

    def prompt(metas):
        ids = [1]
        for i, (_vh, _vw, h, w) in enumerate(metas):
            ids += [vocab + int(t) for t in build_image_block(h, w)[0]]
            ids.append(2 + i)
        return mx.array([ids])

    both = dict(to_mlx(ip(images)))
    metas = both["image_meta"].tolist()
    first = len(build_image_block(*metas[0][2:])[0])
    spans = image_spans(prompt(metas), both, cfg)
    assert _starts(spans) == [1, 2 + first]
    # The second image keys as it does alone.
    alone = dict(to_mlx(ip(images[1:])))
    assert spans[1][1] == image_spans(prompt(alone["image_meta"].tolist()),
                                      alone, cfg)[0][1]
    # Blocks laid out for the other order hold other images.
    assert len(image_spans(prompt(metas[::-1]), both, cfg)) == 1


def _llava(strategy):
    """mlx-vlm's llava processor over 28 x 28 crops with patch 14, and a
    tokenizer whose ``<image>`` is id 3."""
    from gmlx._vendor.mlx_vlm.models.llava.processing_llava import LlavaProcessor
    from tokenizers import pre_tokenizers
    from transformers import CLIPImageProcessor

    from gmlx.load.tokenizer import load_tokenizer_from_gguf

    tokens = (["<s>", "</s>", "<pad>", "<image>"]
              + sorted(pre_tokenizers.ByteLevel.alphabet()) + ["He", "wo"])
    tokenizer = load_tokenizer_from_gguf({
        "general.architecture": "qwen2",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": ["H e", "w o"],
        "tokenizer.ggml.token_type": [3] * 4 + [1] * (len(tokens) - 4),
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.padding_token_id": 2,
    }, "qwen2")
    image_processor = CLIPImageProcessor(
        size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28})
    return LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer,
                          patch_size=14, vision_feature_select_strategy=strategy,
                          num_additional_image_tokens=1)


@pytest.mark.parametrize("strategy", ["default", "full"])
def test_llava_runs_pair_by_the_patch_grid(strategy):
    """An image expands to (H // patch) * (W // patch) tokens, plus the CLS
    token under the "full" strategy."""
    from PIL import Image

    processor = _llava(strategy)
    out = processor(images=[Image.new("RGB", (40, 30), (200, 0, 0)),
                            Image.new("RGB", (30, 40), (0, 200, 0))],
                    text=["a<image>b<image>c"])
    ids, raw = out["input_ids"], {"pixel_values": out["pixel_values"]}
    per = 5 if strategy == "full" else 4
    assert ids.tolist()[0].count(3) == 2 * per
    cfg = SimpleNamespace(image_token_index=3,
                          vision_feature_select_strategy=strategy,
                          vision_config=SimpleNamespace(patch_size=14,
                                                        image_size=28))
    spans = image_spans(ids, raw, cfg, processor)
    assert _starts(spans) == [1, 2 + per]
    # The config alone gives the same count.
    assert image_spans(ids, raw, cfg) == spans
    other = "default" if strategy == "full" else "full"
    cfg.vision_feature_select_strategy = other
    assert len(image_spans(ids, raw, cfg)) == 1


@pytest.mark.parametrize("as_dict", [False, True], ids=["attrs", "dict"])
def test_omni_images_pair_by_the_thinker_ids(as_dict):
    """qwen3_omni_moe names its image id and vision config on the thinker
    alone."""
    thinker = {"image_token_id": 9, "video_token_id": 8,
               "vision_config": {"spatial_merge_size": 2}}
    cfg = ({"model_type": "qwen3_omni_moe", "thinker_config": thinker}
           if as_dict else SimpleNamespace(
               model_type="qwen3_omni_moe",
               thinker_config=SimpleNamespace(
                   image_token_id=9, video_token_id=8,
                   vision_config=SimpleNamespace(spatial_merge_size=2))))
    assert image_token_ids(cfg) == {9}
    raw = {"pixel_values": mx.arange(12 * 4, dtype=mx.float32).reshape(12, 4),
           "image_grid_thw": mx.array([[1, 2, 4], [1, 2, 2]])}
    ids = mx.array([[1, 9, 9, 2, 9, 3]])
    assert _starts(image_spans(ids, raw, cfg)) == [1, 4]


def test_unmatched_runs_key_every_image_at_the_first_run():
    ids = mx.array([[1, 9, 2, 9, 3, 9]])
    pv = mx.stack([mx.full((3, 1, 1), 1.0), mx.full((3, 1, 1), 2.0)])
    spans = image_spans(ids, {"pixel_values": pv}, _cfg())
    assert len(spans) == 1 and spans[0][0] == 1


def test_a_run_maps_to_an_image_only_when_it_holds_that_image():
    """Two adjacent images merge into one run, and a placeholder typed as
    text makes a second one: two runs for two images, but the second
    holds neither. A grid gives each image one token."""
    raw = {"pixel_values": mx.arange(8 * 4, dtype=mx.float32).reshape(8, 4),
           "image_grid_thw": mx.array([[1, 2, 2], [1, 2, 2]])}
    merged = image_spans(mx.array([[1, 9, 9, 2, 3, 9, 4]]), raw, _cfg())
    assert len(merged) == 1 and merged[0][0] == 1
    apart = image_spans(mx.array([[1, 9, 2, 3, 9, 4]]), raw, _cfg())
    assert [s for s, _h in apart] == [1, 4]
    # Gemma-4 rows: each image of 2 x 3 pixels expands to 6 tokens.
    pv = mx.stack([mx.full((3, 2, 3), 1.0), mx.full((3, 2, 3), 2.0)])
    ids = [1] + [9] * 12 + [2, 9, 3]
    assert len(image_spans(mx.array([ids]), {"pixel_values": pv},
                           _cfg())) == 1
    ids = [1] + [9] * 6 + [2] + [9] * 6 + [3]
    assert len(image_spans(mx.array([ids]), {"pixel_values": pv},
                           _cfg())) == 2


# -- the block and exact tiers ---------------------------------------------

def _kv(n, seed):
    k = mx.random.normal((1, 2, n, 8), key=mx.random.key(seed))
    return [k], [k + 1]


def _turns():
    """Turn 1 holds image A at 40; turn 2 repeats it and adds B at 80."""
    t1 = list(range(1, 65))
    t2 = t1 + list(range(65, 97)) + [120] * 16 + list(range(97, 113))
    a, b = (40, 111), (80, 222)
    return t1, t2, a, b


def test_the_block_walk_restores_the_prefix_before_a_new_image():
    t1, t2, a, b = _turns()
    m = GmlxAPCManager(num_blocks=64, block_size=_BS)
    m.release(m.store_kv_blocks(t1, *_kv(len(t1), 1),
                                extra_hash=MediaSalt(7, [a])))
    blocks, n = m.lookup_prefix(t2, extra_hash=MediaSalt(7, [a, b]))
    m.release(blocks)
    assert n == len(t1)
    assert (m.stats.hits, m.stats.misses, m.stats.matched_tokens) == (1, 0, 64)
    # Another image at A's column restores only the text before it, and
    # another tenant salt or LoRA mask restores nothing.
    for salt, want in ((MediaSalt(7, [(40, 112), b]), 32),
                       (MediaSalt(8, [a, b]), 0),
                       (MediaSalt(7, [a, b]) ^ lora_rows.lora_salt((1.0, 0.0)),
                        0)):
        blocks, n = m.lookup_prefix(t2, extra_hash=salt)
        m.release(blocks)
        assert n == want
    assert (m.stats.hits, m.stats.misses) == (2, 2)


def test_the_block_store_covers_the_prefix_a_warm_turn_skipped():
    t1, t2, a, b = _turns()
    m = GmlxAPCManager(num_blocks=64, block_size=_BS)
    m.release(m.store_kv_blocks(t1, *_kv(len(t1), 1),
                                extra_hash=MediaSalt(7, [a])))
    s2 = MediaSalt(7, [a, b])
    m.release(m.store_kv_blocks(t2, *_kv(len(t2), 2), extra_hash=s2,
                                skip_first_n_tokens=len(t1)))
    t3 = t2 + [121] * 16 + [5] * 16
    blocks, n = m.lookup_prefix(t3, extra_hash=MediaSalt(7, [a, b, (128, 3)]))
    m.release(blocks)
    assert n == 128


def _stored(m):
    return sorted(b.token_ids for b in m.pool if b.block_hash is not None)


def test_two_image_turns_store_their_shared_prefix_once():
    """Turn 2 repeats turn 1's tokens and image and adds image B. Each
    block keys on the images that start before its end, so turn 2 finds
    turn 1's blocks and stores only the blocks after them."""
    t1, t2, a, b = _turns()
    m = GmlxAPCManager(num_blocks=64, block_size=_BS)
    k1, v1 = _kv(len(t1), 1)
    m.release(m.store_kv_blocks(t1, k1, v1, extra_hash=MediaSalt(7, [a])))
    k2, v2 = _kv(len(t2), 2)
    k2 = [mx.concatenate([k1[0], k2[0][..., len(t1):, :]], axis=2)]
    v2 = [mx.concatenate([v1[0], v2[0][..., len(t1):, :]], axis=2)]
    m.release(m.store_kv_blocks(t2, k2, v2, extra_hash=MediaSalt(7, [a, b]),
                                skip_first_n_tokens=len(t1)))
    chunks = [tuple(t2[i:i + _BS]) for i in range(0, len(t2), _BS)]
    assert _stored(m) == sorted(chunks)
    assert m.stats.stores == len(chunks)


def _restores(t2):
    """Lookups over turn 2's tokens and more, with the prefix length each
    restores from turn 2's chain: all of turn 2 with a third image after
    it, the text up to another image B, the text up to another image A,
    and nothing under another text salt or LoRA mask."""
    _t1, _t2, a, b = _turns()
    t3 = t2 + [121] * 16 + [5] * 16
    return t3, ((MediaSalt(7, [a, b, (128, 3)]), len(t2)),
                (MediaSalt(7, [a, (80, 223)]), 80),
                (MediaSalt(7, [(40, 112), b]), 32),
                (MediaSalt(8, [a, b]), 0),
                (MediaSalt(7, [a, b]) ^ lora_rows.lora_salt((1.0, 0.0)), 0))


def test_a_block_restore_holds_the_images_of_its_own_prefix():
    _t1, t2, a, b = _turns()
    m = GmlxAPCManager(num_blocks=64, block_size=_BS)
    k2, v2 = _kv(len(t2), 2)
    m.release(m.store_kv_blocks(t2, k2, v2, extra_hash=MediaSalt(7, [a, b])))
    t3, cases = _restores(t2)
    for salt, want in cases:
        blocks, n = m.lookup_prefix(t3, extra_hash=salt)
        assert n == want
        if n:
            got = mx.concatenate([blk.keys[0] for blk in blocks], axis=2)
            assert mx.array_equal(got, k2[0][..., :n, :])
        m.release(blocks)
    # A text prompt restores the text before the first image.
    blocks, n = m.lookup_prefix(t2[:40] + [9] * 24, extra_hash=7)
    m.release(blocks)
    assert n == 32
    # The admission gate's peek reads the chain by the same rule.
    for salt, want in cases:
        assert fresh_gate._covered_len(m, t3, salt) == want


def test_the_disk_tier_restores_the_images_of_its_own_prefix(tmp_path):
    _t1, t2, a, b = _turns()
    disk = _apc.DiskBlockStore(tmp_path, namespace="t")
    try:
        m = GmlxAPCManager(num_blocks=64, block_size=_BS, disk=disk)
        m._disk_min_free_ram_bytes = 0
        k2, v2 = _kv(len(t2), 2)
        m.release(m.store_kv_blocks(t2, k2, v2,
                                    extra_hash=MediaSalt(7, [a, b])))
        disk.flush()
        t3, cases = _restores(t2)
        for salt, want in cases:
            got, n = m.lookup_prefix_disk_cache(t3, extra_hash=salt,
                                                allow_memory_overlap=True)
            assert n == want and (got is None) == (want == 0)
            if got is not None:
                assert mx.array_equal(got[0].keys[..., :n, :],
                                      k2[0][..., :n, :])
        # The stock bounds: a hit is longer than the minimum and no longer
        # than a positive maximum.
        salt = cases[0][0]
        assert m.lookup_prefix_disk_cache(
            t3, extra_hash=salt, max_prefix_tokens=72,
            allow_memory_overlap=True)[1] == 64
        assert m.lookup_prefix_disk_cache(
            t3, extra_hash=salt, min_prefix_tokens=len(t2),
            allow_memory_overlap=True) == (None, 0)
        # Without the overlap flag, a prefix the pool holds stays there.
        assert m.lookup_prefix_disk_cache(t3, extra_hash=salt) == (None, 0)
    finally:
        disk.close()


def test_a_low_ram_disk_skip_logs_as_stock(tmp_path, monkeypatch, caplog):
    """Below the free RAM floor, the image restore skips the disk tier
    with the stock restore's info line."""
    _t1, t2, a, b = _turns()
    disk = _apc.DiskBlockStore(tmp_path, namespace="t")
    try:
        m = GmlxAPCManager(num_blocks=64, block_size=_BS, disk=disk)
        m._disk_min_free_ram_bytes = 2 << 30
        monkeypatch.setattr(_apc, "_free_ram_bytes", lambda: 1 << 30)
        lines = []
        for salt in (7, MediaSalt(7, [a, b])):
            caplog.clear()
            with caplog.at_level(logging.INFO, logger=_apc.logger.name):
                assert m.lookup_prefix_disk_cache(
                    t2, extra_hash=salt) == (None, 0)
            lines.append([r.getMessage() for r in caplog.records
                          if r.name == _apc.logger.name])
        assert len(lines[0]) == 1 and lines[1] == lines[0]
    finally:
        disk.close()


def _row(n, seed):
    c = KVCache()
    c.update_and_fetch(*(x[0] for x in _kv(n, seed)))
    return [c]


def test_the_exact_tier_restores_an_entry_under_fewer_images():
    t1, t2, a, b = _turns()
    m = GmlxAPCManager(num_blocks=64, block_size=_BS)
    m._exact_cache_max = 8
    assert m.store_exact_cache(t1, _row(len(t1), 1),
                               extra_hash=MediaSalt(7, [a]))
    got, p = m.lookup_exact_cache(t2, extra_hash=MediaSalt(7, [a, b]))
    assert got is not None and p == len(t1)
    for salt in (MediaSalt(7, [(40, 112), b]), MediaSalt(8, [a, b]),
                 MediaSalt(7, [a, b]) ^ lora_rows.lora_salt((1.0, 0.0))):
        assert m.lookup_exact_cache(t2, extra_hash=salt) == (None, 0)
    # A text entry ending before the first image restores for that prefix.
    m.store_exact_cache(t1[:32], _row(32, 2), extra_hash=7)
    got, p = m.lookup_exact_cache(t2, extra_hash=MediaSalt(7, [(40, 112), b]))
    assert got is not None and p == 32


def test_the_spec_prefix_cache_keys_each_entry_at_its_length():
    t1, t2, a, b = _turns()
    cache = SpecPrefixCache(max_entries=4, min_prefix=8)
    hidden = mx.zeros((1, 1, 4))
    cache.store(mx.array(t1), _row(len(t1), 1), hidden,
                salt=(MediaSalt(7, [a]), 0))
    hit = cache.lookup(mx.array(t2), salt=(MediaSalt(7, [a, b]), 0))
    assert hit is not None and hit[0] == len(t1)
    for salt in ((MediaSalt(7, [(40, 112), b]), 0), (MediaSalt(7, [a, b]), 1)):
        assert cache.lookup(mx.array(t2), salt=salt) is None


def test_an_anchor_keys_on_its_own_length():
    t1, t2, a, b = _turns()
    m = GmlxAPCManager(num_blocks=64, block_size=_BS)
    assert snapshot.anchor_exact_store(m, t1, _row(len(t1), 1),
                                       extra_hash=MediaSalt(7, [a]))
    got, p = snapshot.anchor_exact_lookup(m, t2,
                                          extra_hash=MediaSalt(7, [a, b]))
    assert got is not None and p == len(t1)
    assert snapshot.anchor_exact_lookup(
        m, t2, extra_hash=MediaSalt(7, [(40, 112), b])) == (None, 0)


@pytest.fixture
def _static_rows():
    lora_rows.configure("static", 1)
    yield
    lora_rows.configure("static", 1)


def test_a_lora_row_keeps_the_per_prefix_salt(_static_rows):
    lora_rows.install_row_channel()
    lora_rows.ensure_rows(2)
    bg = SimpleNamespace(max_tokens=8, logits_processors=None,
                         _unprocessed_sequences=[], uid_count=100,
                         apc_manager=object(),
                         _apc_extra_hash=lambda kw: 1234)
    args = SimpleNamespace(
        logit_bias=None, repetition_penalty=None, repetition_context_size=20,
        presence_penalty=None, presence_context_size=20,
        frequency_penalty=None, frequency_context_size=20,
        logits_processors=None, enable_thinking=False, _kq_lora=(1.0, 0.0))
    s = MediaSalt(7, [(40, 111)])
    kw = {"_apc_semantic_hash": s}
    server.ResponseGenerator._make_logits_processors(SimpleNamespace(), args,
                                                     None)
    ar.BatchGenerator.insert(bg, [[1, 2]], prompt_kwargs=[kw])
    got = kw["_apc_semantic_hash"]
    mask = lora_rows.lora_salt((1.0, 0.0))
    assert isinstance(got, MediaSalt)
    assert got.at(40) == 7 ^ mask and got.at(41) == s.at(41) ^ mask


# -- the serve path, on a tiny gemma-4 --------------------------------------

@pytest.fixture
def serve(monkeypatch):
    monkeypatch.setenv("PREFILL_STEP_SIZE", "8")
    monkeypatch.setenv("GMLX_PREFILL_MIN_TAIL", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    # The loader's installs, then the server's.
    mtp_prefill.install_full_prompt_mtp_prefill()
    assert media_spans.install_span_aware_prompt_step()
    install_media_token_ids()
    assert install_media_salt()


_RIGS = {}


def _rig(bidirectional: bool):
    rig = _RIGS.get(bidirectional)
    if rig is not None:
        return rig
    mx.random.seed(3)
    text = {"use_bidirectional_attention": "vision"} if bidirectional else {}
    model, _ = gvlm.build_vlm_model(_gemma4_config("gemma4", **text))
    lm = model.language_model
    mx.eval(lm.parameters())
    img = int(model.config.image_token_id)
    table = mx.random.normal((_V, _HIDDEN), key=mx.random.key(1)) * 0.5
    runs = mx.random.normal((64, _HIDDEN), key=mx.random.key(2)) * 0.5

    def embed(input_ids, pixel_values=None, mask=None, **_kw):
        ids = input_ids.reshape(-1).tolist()
        emb = table[mx.array([min(t, _V - 1) for t in ids])][None]
        starts = [i for i, t in enumerate(ids)
                  if t == img and (i == 0 or ids[i - 1] != img)]
        for k, a in enumerate(starts):
            b = a
            while b < len(ids) and ids[b] == img:
                b += 1
            # An image's rows depend on its pixels.
            emb[:, a:b] = runs[:b - a] * (
                1.0 + mx.mean(mx.array(pixel_values[k])))
        return SimpleNamespace(to_dict=lambda: {"inputs_embeds": emb})

    served = SimpleNamespace(config=model.config, language_model=lm,
                             get_input_embeddings=embed)
    rig = _RIGS[bidirectional] = SimpleNamespace(lm=lm, img=img,
                                                 model=served)
    return rig


# The pixel shapes of a Gemma-4 image that expands to 20 tokens (16-pixel
# patches, pooled 3 x 3), wide and tall.
_WIDE, _TALL = (3, 96, 480), (3, 480, 96)


def _pix(v):
    return mx.full(_WIDE, float(v))


def _listed(*values):
    """Images of different sizes, as the server holds them: a list of
    numpy arrays."""
    return [np.full((_WIDE, _TALL)[k % 2], float(v), np.float32)
            for k, v in enumerate(values)]


def serve_request(model, raw, tenant=None, processor=None):
    """One request through the stock ``ResponseGenerator.generate`` on a
    stand-in server whose preprocess returns ``raw``. A thread named
    ``request`` runs ``generate``, and the calling thread plays the GPU
    thread: it runs ``_gpu_embed`` on the queued request, so the model
    runs on the thread that built its inputs. Returns the
    ``(input_ids, gen_kwargs)`` of ``_gpu_embed``."""
    mx.eval([v for v in raw.values() if isinstance(v, mx.array)])
    ready = threading.Event()
    ready.set()
    gen = object.__new__(server.ResponseGenerator)
    gen.__dict__.update(
        model=model, processor=processor, config=model.config,
        tokenizer=None, draft_model=None, vision_cache=None, apc_mode="on",
        requests=queue.Queue(), _ready=ready, _load_error=None,
        _tokenizer_lock=threading.Lock(), _cancel_lock=threading.Lock(),
        _cancelled=set(), _cpu_preprocess=lambda *_a, **_k: raw)
    failed = []

    def handle():
        args = server.GenerationArguments(max_tokens=4)
        args.tenant_id = tenant
        try:
            _ctx, tokens = gen.generate("prompt", args=args)
            tokens.close()
        except Exception as e:  # noqa: BLE001  raised on the caller
            failed.append(e)
            gen.requests.put(None)

    handler = threading.Thread(target=handle, name="request")
    handler.start()
    request = gen.requests.get()
    embed = None
    if request is not None:
        embed = server.ResponseGenerator._gpu_embed(
            gen, request.raw_inputs, request.images,
            apc_semantic_hash=request.apc_semantic_hash)
        request.rqueue.put(server.GenerationContext(
            uid=1, prompt_tokens=request.prompt_tokens))
    handler.join()
    if failed:
        raise failed[0]
    return embed


def _request(rig, ids, pixels, tenant=None):
    """A request through the server's generate, as ``serve_request`` runs
    it. A list of numpy arrays stays a list."""
    raw = {"input_ids": mx.array([ids]),
           "mm_token_type_ids": mx.array([[int(t == rig.img) for t in ids]])}
    listed = bool(pixels) and isinstance(pixels[0], np.ndarray)
    pv = (list(pixels) if listed else mx.stack(pixels)) if pixels else None
    if pv is not None:
        raw["pixel_values"] = pv
    _ids, kw = serve_request(rig.model, raw, tenant)
    return ids, kw


def _run(rig, manager, rows):
    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    gen = ar.BatchGenerator(rig.lm, proc,
                            sampler=lambda x: mx.argmax(x, axis=-1),
                            max_tokens=4, apc_manager=manager,
                            top_logprobs_k=_V)
    uids = gen.insert([r[0] for r in rows], [4] * len(rows),
                      prompt_kwargs=[r[1] for r in rows])
    out = {u: [] for u in uids}
    while gen.has_work:
        _p, resp = gen.next()
        for r in resp:
            if r.uid in out:
                out[r.uid].append((int(r.token), dict(r.top_logprobs or ())))
    gen.close()
    return [out[u] for u in uids]


def _manager():
    return GmlxAPCManager(num_blocks=512, block_size=_BS)


def _matched(manager):
    return int(manager.stats_snapshot().get("matched_tokens", 0) or 0)


def _assert_same(warm, cold):
    assert [t for t, _l in warm] == [t for t, _l in cold]
    for (_t, lw), (_c, lc) in zip(warm, cold):
        assert max(abs(lw[k] - lc[k]) for k in lc) < _ATOL


def _prompts(img):
    """A 96-token first turn with image A at 42, and a second turn that
    repeats it and adds image B at 126."""
    sysp = [1 + i % 50 for i in range(40)]
    turn1 = (sysp + [60, 61] + [img] * 20 + [62, 63, 64, 65]
             + [66 + i % 30 for i in range(30)])
    turn2 = (turn1 + [70, 71, 72] + [73 + i % 20 for i in range(26)]
             + [80] + [img] * 20 + [81, 82] + [83 + i % 10 for i in range(14)])
    return turn1, turn2


@pytest.mark.parametrize("bidirectional", [False, True],
                         ids=["causal", "bidirectional"])
def test_turn_two_restores_the_prefix_before_its_new_image(serve,
                                                           bidirectional):
    rig = _rig(bidirectional)
    turn1, turn2 = _prompts(rig.img)
    m = _manager()
    _run(rig, m, [_request(rig, turn1, [_pix(1)])])
    before = _matched(m)
    warm = _run(rig, m, [_request(rig, turn2, [_pix(1), _pix(2)])])
    assert _matched(m) - before == len(turn1)
    cold = _run(rig, _manager(), [_request(rig, turn2, [_pix(1), _pix(2)])])
    _assert_same(warm[0], cold[0])


@pytest.mark.parametrize("change", ["other_image", "moved", "tenant"])
def test_a_prefix_with_other_images_does_not_restore(serve, change):
    rig = _rig(False)
    turn1, turn2 = _prompts(rig.img)
    m = _manager()
    # Two images, so a swap moves each one to the other's column.
    first = turn1[:62] + [5] * 10 + [rig.img] * 20 + [6] * 4
    _run(rig, m, [_request(rig, first, [_pix(1), _pix(2)])])
    second = first + [7] * 30
    pixels, tenant = [_pix(1), _pix(2)], None
    if change == "other_image":
        pixels = [_pix(3), _pix(2)]
    elif change == "moved":
        pixels = [_pix(2), _pix(1)]
    else:
        tenant = "other"
    before = _matched(m)
    warm = _run(rig, m, [_request(rig, second, pixels, tenant)])
    restored = _matched(m) - before
    # The text before the first image may restore, never more.
    assert restored <= 42
    if tenant is not None:
        assert restored == 0
    cold = _run(rig, _manager(), [_request(rig, second, pixels, tenant)])
    _assert_same(warm[0], cold[0])


def test_images_of_different_sizes_key_on_their_pixels(serve):
    """The stock payload hash reads a list of images of different sizes
    as no images, so a request with other images of the same sizes
    restored the whole prompt."""
    rig = _rig(False)
    _turn1, turn2 = _prompts(rig.img)
    m = _manager()
    _run(rig, m, [_request(rig, turn2, _listed(1, 2))])
    before = _matched(m)
    warm = _run(rig, m, [_request(rig, turn2, _listed(3, 2))])
    # Image A differs, so only the text before it may restore.
    assert _matched(m) - before <= 42
    cold = _run(rig, _manager(), [_request(rig, turn2, _listed(3, 2))])
    _assert_same(warm[0], cold[0])


def test_turn_two_with_images_of_different_sizes_restores_turn_one(serve):
    """Turn 1 holds its one image as an mx array, turn 2 its two images
    as a list of numpy arrays. Image A hashes alike in both."""
    rig = _rig(False)
    turn1, turn2 = _prompts(rig.img)
    m = _manager()
    _run(rig, m, [_request(rig, turn1, [_pix(1)])])
    before = _matched(m)
    warm = _run(rig, m, [_request(rig, turn2, _listed(1, 2))])
    assert _matched(m) - before == len(turn1)
    cold = _run(rig, _manager(), [_request(rig, turn2, _listed(1, 2))])
    _assert_same(warm[0], cold[0])


def test_a_text_prompt_keeps_the_stock_salt(serve):
    rig = _rig(False)
    turn1, _turn2 = _prompts(rig.img)
    text = [t if t != rig.img else 90 for t in turn1]
    _ids, kw = _request(rig, text, [])
    stock = _apc.semantic_extra_hash.__wrapped__(
        tenant=None, image_hash=0, media={"audio": None, "video": None},
        model=rig.lm, processor=None)
    assert type(kw["_apc_semantic_hash"]) is int
    assert kw["_apc_semantic_hash"] == stock
    # An image prompt keys its text before the first image the same way.
    _ids, kw = _request(rig, turn1, [_pix(1)])
    salt = kw["_apc_semantic_hash"]
    assert isinstance(salt, MediaSalt)
    assert salt.at(42) == stock and salt.at(43) != stock


@pytest.mark.parametrize("images", [1, 2])
def test_each_image_hashes_once_before_the_request_queues(serve, monkeypatch,
                                                         images):
    """The request thread hashes each image's pixels once; the GPU worker
    hashes none."""
    import gmlx.cache.media_salt as media_salt

    rig = _rig(False)
    turn1, turn2 = _prompts(rig.img)
    seen = []

    def spy(fn):
        def hashed(x):
            if int(getattr(x, "size", 0)) >= 1000:
                seen.append((threading.current_thread().name, int(x.size)))
            return fn(x)
        return hashed

    monkeypatch.setattr(media_salt, "_content_hash",
                        spy(media_salt._content_hash))
    monkeypatch.setattr(_apc, "_tensor_content_hash",
                        spy(_apc._tensor_content_hash))
    ids = turn1 if images == 1 else turn2
    _ids, kw = _request(rig, ids, [_pix(k) for k in range(images)])
    assert isinstance(kw["_apc_semantic_hash"], MediaSalt)
    one = int(np.prod(_WIDE))
    assert seen == [("request", one)] * images
