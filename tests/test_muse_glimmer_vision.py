"""Muse Glimmer vision: the pure index math, the mmproj remap and the image
preprocessor's grid search. CPU-only - no GGUF, no weights, no image decode.

Everything here is a function of the patch grid or of a tensor name, and every
one of them is a silent-failure surface: a wrong permutation, a transposed patch
conv or an off-by-one grid produces a model that runs and describes the wrong
picture. The expectations are derived from llama.cpp (``clip.cpp`` window/ds
permutations and ``muse_glimmer_grid_size`` in ``tools/mtmd/mtmd-image.cpp``),
not read back off this port.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from gmlx.muse_glimmer_vlm_model import (  # noqa: E402
    pixel_shuffle_order,
    window_order,
)
from gmlx.vlm import (  # noqa: E402
    _MuseGlimmerGgufImageProcessor,
    _muse_glimmer_vision_name,
    remap_vision_arrays,
)

# --- window permutation -------------------------------------------------------


def test_window_order_is_a_permutation_with_one_segment_per_patch():
    perm, segment = window_order(7, 5, 3)
    assert sorted(perm) == list(range(35))
    assert len(segment) == 35


def test_window_order_is_the_identity_when_the_grid_fits_one_window():
    perm, segment = window_order(4, 4, 32)
    assert perm == list(range(16))
    assert set(segment) == {0}


def test_window_order_partial_edge_windows():
    # 3x3 grid, window 2: windows of 4, 2, 2 and 1 patches, in row-major window
    # order, exactly as llama.cpp builds them.
    perm, segment = window_order(3, 3, 2)
    assert perm == [0, 1, 3, 4, 2, 5, 6, 7, 8]
    assert segment == [0, 0, 0, 0, 1, 1, 2, 2, 3]


def test_segments_are_contiguous_so_the_mask_is_block_diagonal():
    _, segment = window_order(9, 6, 4)
    assert segment == sorted(segment)


def test_two_patches_share_a_segment_iff_they_share_a_window():
    grid_w, grid_h, window = 7, 5, 3
    perm, segment = window_order(grid_w, grid_h, window)
    for i, p in enumerate(perm):
        for j, q in enumerate(perm):
            same_window = ((p % grid_w) // window == (q % grid_w) // window
                           and (p // grid_w) // window == (q // grid_w) // window)
            assert (segment[i] == segment[j]) is same_window


def test_inverse_permutation_restores_row_major_order():
    # The tower undoes the permutation with a scatter of arange; if that is
    # wrong the patch features land under the wrong soft tokens.
    perm = mx.array(window_order(7, 5, 3)[0])
    inverse = mx.zeros(perm.shape, dtype=mx.int32)
    inverse[perm] = mx.arange(perm.size, dtype=mx.int32)
    x = mx.arange(35).reshape(35, 1)
    assert mx.take(mx.take(x, perm, axis=0), inverse, axis=0).reshape(-1).tolist() \
        == list(range(35))


# --- pixel shuffle ------------------------------------------------------------


def test_pixel_shuffle_groups_each_cell_contiguously():
    # 4x2 grid, merge 2: cell (0,0) is patches 0,1,4,5 and cell (0,1) is 2,3,6,7.
    assert pixel_shuffle_order(4, 2, 2) == [0, 1, 4, 5, 2, 3, 6, 7]


def test_pixel_shuffle_covers_every_patch_once():
    order = pixel_shuffle_order(6, 4, 2)
    assert sorted(order) == list(range(24))


def test_pixel_shuffle_merge_one_is_the_identity():
    assert pixel_shuffle_order(5, 3, 1) == list(range(15))


def test_pixel_shuffle_drops_the_ragged_edge():
    # An odd grid side has no partial cell: llama.cpp merges floor(side/merge)
    # cells and the preprocessor guarantees even sides anyway.
    assert len(pixel_shuffle_order(5, 3, 2)) == (5 // 2) * (3 // 2) * 4


# --- mmproj tensor remap ------------------------------------------------------

_ROOTS = (
    "mm.0.weight", "mm.1.weight", "mm.2.weight",
    "v.patch_embd.weight", "v.position_embd.weight",
    "v.post_ln.bias", "v.post_ln.weight", "v.pre_ln.bias", "v.pre_ln.weight",
)
_BLK_LEAVES = (
    "attn_q", "attn_k", "attn_v", "attn_out", "ln1", "ln2", "ffn_up", "ffn_down",
)


def _mmproj_names(n_layers=2):
    names = list(_ROOTS)
    for i in range(n_layers):
        for leaf in _BLK_LEAVES:
            names += [f"v.blk.{i}.{leaf}.weight", f"v.blk.{i}.{leaf}.bias"]
    return names


def test_every_mmproj_tensor_is_claimed():
    assert all(_muse_glimmer_vision_name(n) is not None for n in _mmproj_names())


def test_only_the_patch_conv_is_flagged_for_transpose():
    flagged = [n for n in _mmproj_names() if _muse_glimmer_vision_name(n)[1]]
    assert flagged == ["v.patch_embd.weight"]


def test_projector_splits_across_the_adapter_and_the_llm():
    # mm.0/mm.1 are the mmproj-side adapter; mm.2 projects into the text
    # residual width and lives outside it, where the HF checkpoint keeps it.
    assert _muse_glimmer_vision_name("mm.0.weight")[0] == "vision_adapter.fc1.weight"
    assert _muse_glimmer_vision_name("mm.1.weight")[0] == "vision_adapter.fc2.weight"
    assert _muse_glimmer_vision_name("mm.2.weight")[0] == "vision_projection.weight"


def test_ffn_up_and_down_keep_their_direction():
    # fc1 widens and fc2 narrows; swapping them still loads (both linears exist)
    # and silently produces garbage features.
    assert _muse_glimmer_vision_name("v.blk.3.ffn_up.weight")[0] \
        == "vision_tower.layers.3.mlp.fc1.weight"
    assert _muse_glimmer_vision_name("v.blk.3.ffn_down.bias")[0] \
        == "vision_tower.layers.3.mlp.fc2.bias"


def test_block_norms_and_attention_land_on_the_vendored_paths():
    got = {leaf: _muse_glimmer_vision_name(f"v.blk.0.{leaf}.weight")[0]
           for leaf in _BLK_LEAVES}
    assert got["ln1"] == "vision_tower.layers.0.layer_norm1.weight"
    assert got["ln2"] == "vision_tower.layers.0.layer_norm2.weight"
    assert got["attn_q"] == "vision_tower.layers.0.self_attn.q_proj.weight"
    assert got["attn_out"] == "vision_tower.layers.0.self_attn.o_proj.weight"


def test_position_embedding_is_a_bare_array_not_a_module_weight():
    # The tower holds it as a plain mx.array attribute, so the target name has
    # no ``.weight`` suffix.
    assert _muse_glimmer_vision_name("v.position_embd.weight")[0] \
        == "vision_tower.position_embedding"


def test_unknown_tensor_is_skipped_not_guessed():
    assert _muse_glimmer_vision_name("v.blk.0.mystery.weight") is None
    assert _muse_glimmer_vision_name("a.blk.0.attn_q.weight") is None


def test_remap_transposes_the_patch_conv_and_skips_nothing():
    arrays = {n: mx.zeros((2, 2)) for n in _mmproj_names()}
    arrays["v.patch_embd.weight"] = mx.zeros((8, 3, 14, 14))   # [out, in, kH, kW]
    out, skipped, kq = remap_vision_arrays(arrays, "muse_glimmer")
    assert skipped == [] and kq == {}
    assert out["vision_tower.patch_embed.weight"].shape == (8, 14, 14, 3)
    assert len(out) == len(arrays)


# --- image preprocessing grid search -----------------------------------------

CELL = 28   # patch_size 14 * spatial_merge_size 2


def _proc(max_image_tokens=4096):
    return _MuseGlimmerGgufImageProcessor(
        image_mean=[0.5] * 3, image_std=[0.5] * 3,
        max_image_tokens=max_image_tokens)


@pytest.mark.parametrize("h,w,expect", [
    (280, 280, (280, 280)),        # exact square, no search needed
    (280, 140, (280, 140)),        # exact 2:1
    (300, 200, (308, 196)),        # 10.71 x 7.14: the 11x7 tie wins on tokens
    (10, 100, (28, 112)),          # thinner than one cell: clamps the short side
    (10, 10, (28, 28)),            # smaller than one cell
])
def test_target_grid_matches_the_llama_cpp_search(h, w, expect):
    assert _proc()._target_hw(h, w) == expect


def test_ties_go_to_the_larger_grid():
    # 300x200 has |11/7 - 3/2| == |10/7 - 3/2|; llama.cpp breaks the tie toward
    # more tokens, which is 11x7 rather than 10x7.
    h_out, w_out = _proc()._target_hw(300, 200)
    assert (h_out // CELL, w_out // CELL) == (11, 7)


@pytest.mark.parametrize("h,w", [(4000, 4000), (8000, 1000), (1000, 8000),
                                 (1234, 567), (33, 4001)])
def test_output_is_always_whole_cells_under_the_token_cap(h, w):
    proc = _proc()
    h_out, w_out = proc._target_hw(h, w)
    assert h_out % CELL == 0 and w_out % CELL == 0
    assert h_out >= CELL and w_out >= CELL
    assert proc.soft_tokens(h_out, w_out) <= proc.max_image_tokens


def test_oversized_image_shrinks_to_the_cap_keeping_its_ratio():
    proc = _proc()
    h_out, w_out = proc._target_hw(20000, 10000)
    assert (h_out // CELL, w_out // CELL) == (90, 45)
    assert proc.soft_tokens(h_out, w_out) == 4050


def test_square_oversized_image_lands_exactly_on_the_cap():
    proc = _proc()
    h_out, w_out = proc._target_hw(10000, 10000)
    assert (h_out, w_out) == (64 * CELL, 64 * CELL)
    assert proc.soft_tokens(h_out, w_out) == 4096


def test_no_candidate_under_the_cap_falls_back_to_round_and_clamp():
    # A 100:1 image against a 1-token budget: every floor/ceil pair either has a
    # zero side or exceeds the cap, so the search rounds and clamps instead of
    # returning nothing.
    h_out, w_out = _proc(max_image_tokens=1)._target_hw(1000, 10)
    assert (h_out // CELL, w_out // CELL) == (10, 1)


def test_soft_tokens_counts_merged_cells():
    proc = _proc()
    assert proc.soft_tokens(280, 140) == 10 * 5
    assert proc.soft_tokens(CELL, CELL) == 1
