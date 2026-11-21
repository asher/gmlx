"""Inkling 2 vision (mmproj projector inkling2v): the tower against a numpy
reference written from llama.cpp's inkling2v graph, the preprocessing, the
remap, the config synth, the processor and the container. CPU-only,
synthetic weights with the released mmproj's tensor names."""

import math
import os
import sys

import mlx.core as mx
import mlx.utils as mu
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import inkling2_tiny as T  # noqa: E402

from gmlx.load.config_synth import synthesize_config  # noqa: E402
from gmlx.load.vlm import (  # noqa: E402
    INKLING2_IMAGE_TOKEN,
    INKLING2_VISION,
    UnsupportedVLMError,
    _Inkling2GgufImageProcessor,
    _synthesize_inkling2_processor,
    build_vlm_model,
    remap_vision_arrays,
    resolve_vlm_model_type,
    synthesize_vlm_config,
)
from gmlx.models.inkling2.vision import (  # noqa: E402
    VisionConfig,
    VisionModel,
    merge_patches,
    resampled_positions,
    rope_2d,
    rope_positions,
)

DEPTH, DIM, HEADS, FFN, PATCH, GRID, MERGE_W = 2, 32, 4, 48, 4, 4, 40
OUT = T.D

MM_META = {
    "general.architecture": "clip",
    "general.type": "mmproj",
    "clip.has_vision_encoder": True,
    "clip.vision.projector_type": "inkling2v",
    "clip.vision.projection_dim": OUT,
    "clip.vision.image_size": GRID * PATCH,
    "clip.vision.patch_size": PATCH,
    "clip.vision.embedding_length": DIM,
    "clip.vision.feed_forward_length": FFN,
    "clip.vision.block_count": DEPTH,
    "clip.vision.attention.head_count": HEADS,
    "clip.vision.attention.layer_norm_epsilon": 1e-5,
    "clip.vision.spatial_merge_size": 2,
    "clip.vision.image_mean": [0.5, 0.5, 0.5],
    "clip.vision.image_std": [0.5, 0.5, 0.5],
}

_BLOCK = {"attn_q": (DIM, DIM), "attn_k": (DIM, DIM), "attn_v": (DIM, DIM),
          "attn_out": (DIM, DIM), "ffn_up": (FFN, DIM),
          "ffn_down": (DIM, FFN), "ln1": (DIM,), "ln2": (DIM,)}


def _mmproj_arrays(seed=5):
    """Every tensor name of the released file, at a small size, in numpy
    (out, in) order."""
    rng = np.random.default_rng(seed)

    def arr(*shape, scale=0.2, base=0.0):
        return (base + rng.standard_normal(shape) * scale).astype(np.float32)

    a = {
        "v.class_embd": arr(DIM),
        "v.class_position_embd.weight": arr(DIM),
        "v.patch_embd.weight": arr(DIM, 3, PATCH, PATCH),
        "v.position_embd.weight": arr(GRID * GRID, DIM),
        "v.pre_ln.weight": arr(DIM, base=1.0),
        "v.pre_ln.bias": arr(DIM),
        "mm.1.weight": arr(MERGE_W, 4 * DIM),
        "mm.2.weight": arr(OUT, MERGE_W),
    }
    for i in range(DEPTH):
        for sub, shape in _BLOCK.items():
            base = 1.0 if sub.startswith("ln") else 0.0
            a[f"v.blk.{i}.{sub}.weight"] = arr(*shape, base=base)
            a[f"v.blk.{i}.{sub}.bias"] = arr(shape[0])
        a[f"v.blk.{i}.ls1.weight"] = arr(DIM, base=0.5)
        a[f"v.blk.{i}.ls2.weight"] = arr(DIM, base=0.5)
    return a


# A numpy reference from inkling2v.cpp and mtmd-graph.cpp.

def _ref_resample(table, edge, n_h, n_w):
    """ggml_interpolate, bilinear, no corner alignment."""
    grid = table.reshape(edge, edge, -1)
    out = np.zeros((n_h, n_w, grid.shape[-1]), np.float64)
    for i in range(n_h):
        y = (i + 0.5) * edge / n_h - 0.5
        y0 = int(math.floor(y))
        y1 = min(max(y0 + 1, 0), edge - 1)
        y0 = min(max(y0, 0), edge - 1)
        dy = min(max(y - y0, 0.0), 1.0)
        for j in range(n_w):
            x = (j + 0.5) * edge / n_w - 0.5
            x0 = int(math.floor(x))
            x1 = min(max(x0 + 1, 0), edge - 1)
            x0 = min(max(x0, 0), edge - 1)
            dx = min(max(x - x0, 0.0), 1.0)
            out[i, j] = ((1 - dy) * ((1 - dx) * grid[y0, x0] + dx * grid[y0, x1])
                         + dy * ((1 - dx) * grid[y1, x0] + dx * grid[y1, x1]))
    return out.reshape(n_h * n_w, -1)


def _ref_rope(x, pos, base):
    """ggml_rope_ext mode 0 over all of x's last axis: adjacent pairs."""
    d = x.shape[-1]
    out = x.copy()
    for j in range(d // 2):
        theta = pos * base ** (-2.0 * j / d)
        a, b = x[..., 2 * j], x[..., 2 * j + 1]
        c, s = np.cos(theta), np.sin(theta)
        out[..., 2 * j] = a * c - b * s
        out[..., 2 * j + 1] = a * s + b * c
    return out


def _ref_rope_2d(x, pos_w, pos_h, base):
    """build_rope_2d: the first half by pos_w, the second by pos_h."""
    half = x.shape[-1] // 2
    return np.concatenate([_ref_rope(x[..., :half], pos_w, base),
                           _ref_rope(x[..., half:], pos_h, base)], axis=-1)


def _ln(x, w, b, eps=1e-5):
    mu_ = x.mean(-1, keepdims=True)
    var = ((x - mu_) ** 2).mean(-1, keepdims=True)
    return (x - mu_) / np.sqrt(var + eps) * w + b


def _gelu(x):
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def _ref_tower(a, image):
    """One image [C, H, W] -> projected rows, all in float64."""
    a = {k: v.astype(np.float64) for k, v in a.items()}
    c, h, w = image.shape
    n_h, n_w = h // PATCH, w // PATCH
    patches = np.zeros((n_h * n_w, DIM))
    kernel = a["v.patch_embd.weight"]
    for y in range(n_h):
        for x in range(n_w):
            win = image[:, y * PATCH:(y + 1) * PATCH, x * PATCH:(x + 1) * PATCH]
            patches[y * n_w + x] = np.einsum("ocij,cij->o", kernel, win)
    patches += _ref_resample(a["v.position_embd.weight"], GRID, n_h, n_w)
    cls = a["v.class_embd"] + a["v.class_position_embd.weight"]
    cur = np.concatenate([cls[None], patches])
    cur = _ln(cur, a["v.pre_ln.weight"], a["v.pre_ln.bias"])
    pos_w = np.array([0] + [x + 1 for y in range(n_h) for x in range(n_w)],
                     np.float64)
    pos_h = np.array([0] + [y + 1 for y in range(n_h) for x in range(n_w)],
                     np.float64)
    hd = DIM // HEADS

    def lin(x, name):
        return x @ a[f"{name}.weight"].T + a[f"{name}.bias"]

    for i in range(DEPTH):
        p = f"v.blk.{i}."
        x = _ln(cur, a[p + "ln1.weight"], a[p + "ln1.bias"])
        q = lin(x, p + "attn_q").reshape(-1, HEADS, hd).transpose(1, 0, 2)
        k = lin(x, p + "attn_k").reshape(-1, HEADS, hd).transpose(1, 0, 2)
        v = lin(x, p + "attn_v").reshape(-1, HEADS, hd).transpose(1, 0, 2)
        q = _ref_rope_2d(q, pos_w, pos_h, 1e4)
        k = _ref_rope_2d(k, pos_w, pos_h, 1e4)
        s = q @ k.transpose(0, 2, 1) / math.sqrt(hd)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        att = (s @ v).transpose(1, 0, 2).reshape(-1, DIM)
        cur = cur + lin(att, p + "attn_out") * a[p + "ls1.weight"]
        x = _ln(cur, a[p + "ln2.weight"], a[p + "ln2.bias"])
        ffn = lin(_gelu(lin(x, p + "ffn_up")), p + "ffn_down")
        cur = cur + ffn * a[p + "ls2.weight"]
    grid = cur[1:].reshape(n_h, n_w, DIM)
    merged = []
    for y in range(0, n_h, 2):
        for x in range(0, n_w, 2):
            merged.append(np.concatenate(
                [grid[y, x], grid[y, x + 1], grid[y + 1, x], grid[y + 1, x + 1]]))
    merged = np.stack(merged)
    return _gelu(merged @ a["mm.1.weight"].T) @ a["mm.2.weight"].T


@pytest.fixture
def _on_cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _tower(arrays):
    out, skipped, _ = remap_vision_arrays(
        {k: mx.array(v) for k, v in arrays.items()}, INKLING2_VISION)
    assert skipped == []
    tower = VisionModel(VisionConfig(
        depth=DEPTH, hidden_size=DIM, num_heads=HEADS, intermediate_size=FFN,
        patch_size=PATCH, merge_hidden_size=MERGE_W, out_hidden_size=OUT,
        position_grid=GRID))
    tower.load_weights([(k[len("vision_tower."):], v) for k, v in out.items()])
    return tower


@pytest.mark.parametrize("n_h,n_w", [(4, 4), (2, 6), (6, 2)])
def test_the_tower_matches_the_reference(_on_cpu, n_h, n_w):
    # 4 x 4 keeps the position grid as it is; the others resample it.
    arrays = _mmproj_arrays()
    image = np.random.default_rng(1).standard_normal(
        (3, n_h * PATCH, n_w * PATCH)).astype(np.float32)
    got = np.array(_tower(arrays)(mx.array(image)))
    want = _ref_tower(arrays, image.astype(np.float64))
    assert got.shape == (n_h * n_w // 4, OUT)
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4)


def test_the_reference_fails_on_swapped_rope_halves(_on_cpu):
    """The 2D rope check is not vacuous: rows and columns swapped give
    another answer on a non-square grid."""
    arrays = _mmproj_arrays()
    image = np.random.default_rng(1).standard_normal(
        (3, 2 * PATCH, 6 * PATCH)).astype(np.float32)
    want = _ref_tower(arrays, image.astype(np.float64))
    import gmlx.models.inkling2.vision as vision

    real = vision.rope_positions
    try:
        vision.rope_positions = lambda n_h, n_w: real(n_h, n_w)[::-1]
        got = np.array(_tower(arrays)(mx.array(image)))
    finally:
        vision.rope_positions = real
    assert not np.allclose(got, want, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("n_h,n_w", [(4, 4), (3, 5), (7, 2), (1, 9), (8, 8)])
def test_the_position_grid_resamples_as_ggml(n_h, n_w):
    table = np.random.default_rng(2).standard_normal((16, 8)).astype(np.float32)
    got = np.array(resampled_positions(mx.array(table), 4, n_h, n_w))
    np.testing.assert_allclose(got, _ref_resample(table, 4, n_h, n_w),
                               atol=1e-5)


def test_rope_positions_put_the_class_token_at_the_origin():
    pos_w, pos_h = rope_positions(2, 3)
    assert pos_w.tolist() == [0, 1, 2, 3, 1, 2, 3]
    assert pos_h.tolist() == [0, 1, 1, 1, 2, 2, 2]


def test_rope_2d_turns_adjacent_pairs_column_then_row():
    from gmlx.models.inkling2.vision import _rope_tables

    x = np.random.default_rng(3).standard_normal((1, 2, 5, 16)).astype(
        np.float32)
    pw = np.array([0, 1, 2, 3, 4], np.float32)
    ph = np.array([0, 4, 1, 7, 2], np.float32)
    got = np.array(rope_2d(mx.array(x), _rope_tables(mx.array(pw), 8, 1e4),
                           _rope_tables(mx.array(ph), 8, 1e4)))
    np.testing.assert_allclose(got, _ref_rope_2d(x, pw, ph, 1e4), atol=1e-5)


def test_merge_order_follows_build_patch_merge_permute():
    n_h, n_w = 4, 6
    x = mx.arange(n_h * n_w)[:, None]
    rows = np.array(merge_patches(x, n_h, n_w, 2))
    # (x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1), cells row-major.
    assert rows[0].tolist() == [0, 1, 6, 7]
    assert rows[1].tolist() == [2, 3, 8, 9]
    assert rows[3].tolist() == [12, 13, 18, 19]
    assert rows.shape == (6, 4)


# Preprocessing.

def _image(w, h, seed=0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_padding_is_raw_minus_one_then_normalized():
    ip = _Inkling2GgufImageProcessor([0.5] * 3, [0.5] * 3)
    img = _image(50, 30)
    feat = ip([img])
    assert feat["image_sizes"] == [(40, 80)]
    pv = feat["pixel_values"][0]
    raw = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
    np.testing.assert_allclose(pv[:, :30, :50], (raw - 127.5) / 127.5,
                               atol=1e-6)
    pad = (-1.0 - 127.5) / 127.5
    assert np.all(pv[:, 30:, :] == np.float32(pad))
    assert np.all(pv[:, :, 50:] == np.float32(pad))
    assert ip.soft_tokens(40, 80) == 2


def test_an_image_resizes_only_past_the_token_cap():
    ip = _Inkling2GgufImageProcessor([0.5] * 3, [0.5] * 3)
    # 2560 x 2560 is 64 x 64 cells, exactly the 4096-token cap.
    assert ip.resized(2560, 2560) is None
    assert ip.resized(2541, 2541) is None
    # 2600 x 2600 rounds up to 65 x 65 cells: scaled into the cap.
    w, h = ip.resized(2600, 2600)
    assert (w, h) == (2560, 2560)
    w, h = ip.resized(8000, 1000)
    assert w % 40 == 0 and h % 40 == 0 and (w // 40) * (h // 40) <= 4096
    feat = ip([_image(2600, 2600)])
    assert feat["image_sizes"] == [(2560, 2560)]


def test_the_serve_estimate_bounds_the_largest_image():
    from gmlx.serve.estimate import _MEDIA_TOKENS_PER_IMAGE

    ip = _Inkling2GgufImageProcessor([0.5] * 3, [0.5] * 3)
    most = 0
    for w in (39, 640, 2559, 2600, 4032, 9000):
        for h in (w, 3 * w, 40 * w):
            size = ip.resized(w, h) or (ip._aligned(w), ip._aligned(h))
            most = max(most, ip.soft_tokens(size[1], size[0]))
    assert most <= _MEDIA_TOKENS_PER_IMAGE[INKLING2_VISION]


# Loader plumbing.

def test_the_projector_resolves_for_both_layouts():
    assert resolve_vlm_model_type("inkling2", MM_META) == INKLING2_VISION
    assert resolve_vlm_model_type("inkling2_loop", MM_META) == INKLING2_VISION
    with pytest.raises(UnsupportedVLMError, match="inkling2v.*'llama'"):
        resolve_vlm_model_type("llama", MM_META)


def _llm_meta(image_token=True):
    v = T.values()
    meta = T.metadata("root", v["token_embd.weight"].shape[0])
    if image_token:
        meta["tokenizer.ggml.tokens"] = list(meta["tokenizer.ggml.tokens"])
        meta["tokenizer.ggml.tokens"][1] = INKLING2_IMAGE_TOKEN
    return meta, T.shapes("root", v)


def _config(image_token=True):
    meta, shapes = _llm_meta(image_token)
    arrays = _mmproj_arrays()
    return synthesize_vlm_config(
        INKLING2_VISION, meta, shapes, MM_META,
        mm_shapes={k: tuple(reversed(v.shape)) for k, v in arrays.items()})


def test_config_synth_fields():
    cfg = _config()
    assert cfg["model_type"] == "inkling2_vl"
    assert cfg["text_config"]["model_type"] == "inkling2"
    vc = cfg["vision_config"]
    assert (vc["depth"], vc["hidden_size"], vc["num_heads"]) == (
        DEPTH, DIM, HEADS)
    assert (vc["patch_size"], vc["spatial_merge_size"]) == (PATCH, 2)
    assert vc["merge_hidden_size"] == MERGE_W
    assert vc["position_grid"] == GRID and vc["out_hidden_size"] == OUT
    # The placeholder id comes from the vocabulary, else the released id.
    assert cfg["image_token_id"] == cfg["image_token_index"] == 1
    assert _config(image_token=False)["image_token_id"] == 200054


def test_remap_covers_the_whole_tower_and_nothing_else():
    arrays = {k: mx.array(v) for k, v in _mmproj_arrays().items()}
    out, skipped, kq = remap_vision_arrays(arrays, INKLING2_VISION)
    assert skipped == [] and kq == {}
    model, _ = build_vlm_model(_config())
    params = dict(mu.tree_flatten(model.parameters()))
    vision = sorted(k for k in params if not k.startswith("language_model."))
    assert sorted(out) == vision
    for k in vision:
        assert tuple(out[k].shape) == tuple(params[k].shape), k


def test_the_container_feeds_normed_text_rows_and_raw_image_rows():
    cfg = _config()
    model, _ = build_vlm_model(cfg)
    assert type(model).__module__ == "gmlx.models.inkling2.vlm_model"
    out, _, _ = remap_vision_arrays(
        {k: mx.array(v) for k, v in _mmproj_arrays().items()}, INKLING2_VISION)
    model.load_weights(list(out.items()), strict=False)
    lm = model.language_model
    img = cfg["image_token_id"]
    ids = mx.array([[5, 6, img, img, 7]])
    pixels = mx.random.normal((1, 3, 2 * PATCH, 4 * PATCH))
    emb = model.get_input_embeddings(
        ids, pixels, image_sizes=[(2 * PATCH, 4 * PATCH)]).inputs_embeds
    rows = model.vision_tower(pixels[0])
    text = lm.model.embed_norm(lm.model.embed_tokens(ids))
    np.testing.assert_allclose(np.array(emb[0, 2:4]), np.array(rows),
                               atol=1e-5)
    for i in (0, 1, 4):
        np.testing.assert_allclose(np.array(emb[0, i]), np.array(text[0, i]),
                                   atol=1e-6)
    with pytest.raises(ValueError, match="3 image placeholder"):
        model.get_input_embeddings(
            mx.array([[img, img, img]]), pixels,
            image_sizes=[(2 * PATCH, 4 * PATCH)])
    # A text prompt through the container is the text model's prompt.
    plain = mx.array([[5, 6, 7]])
    want = lm(plain).logits
    got = model(plain).logits
    np.testing.assert_allclose(np.array(got), np.array(want), atol=1e-5)


def test_sanitize_splits_the_packed_loop_queries_under_the_text_model():
    model, _ = build_vlm_model(_config())
    core = T.END - T.START
    w = {"language_model.model.loop_res_query": mx.zeros((2 * core, T.D)),
         "language_model.model.loop_kv_res_query": mx.ones((2 * core, T.D)),
         "vision_tower.class_embedding": mx.zeros((DIM,))}
    out = model.sanitize(w)
    assert "vision_tower.class_embedding" in out
    assert f"language_model.model.layers.{T.START}.loop.msg_query" in out
    assert "language_model.model.loop_res_query" not in out


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "hi": 1, "what": 2, "is": 3,
             "<|content_image|>": 4, INKLING2_IMAGE_TOKEN: 5}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["<|content_image|>", INKLING2_IMAGE_TOKEN])
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")


def test_processor_repeats_each_placeholder_once_per_image_row():
    meta = dict(MM_META, **{"clip.vision.patch_size": 20})
    proc = _synthesize_inkling2_processor(_tokenizer(), meta)
    ph = INKLING2_IMAGE_TOKEN
    text = f"what <|content_image|>{ph} is <|content_image|>{ph}"
    feat = proc(images=[_image(100, 70), _image(40, 40, seed=1)], text=text)
    # 100 x 70 pads to 120 x 80: 3 x 2 cells. 40 x 40 is one cell.
    assert [tuple(s) for s in feat["image_sizes"]] == [(80, 120), (40, 40)]
    ids = [int(t) for t in feat["input_ids"][0].tolist()]
    assert ids == [2, 4] + [5] * 6 + [3, 4] + [5]
    with pytest.raises(ValueError, match="2 image placeholders but 1"):
        proc(images=[_image(40, 40)], text=text)


def test_media_salt_counts_each_image_s_rows():
    from gmlx.cache.media_salt import _expected_tokens

    meta = dict(MM_META, **{"clip.vision.patch_size": 20})
    proc = _synthesize_inkling2_processor(_tokenizer(), meta)
    raw = dict(proc.image_processor([_image(100, 70), _image(40, 40)]))
    assert _expected_tokens(raw, 2, _config(), proc) == ([6, 1], False)


def test_synth_config_text_half_is_the_text_synth():
    meta, shapes = _llm_meta()
    assert _config()["text_config"] == synthesize_config(meta, shapes)
