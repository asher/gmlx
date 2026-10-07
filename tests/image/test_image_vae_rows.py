"""The VAE rows of the image memory plan: a row per family, the fits of
the Qwen-Image 2.1 VAE against the passes measured on its official file,
the rule that prices a pass on an unmeasured row, held on that VAE's steps
as it was priced before, the plan's decode term against the choice and
the price of the guarded decode under either, and the Z-Image values held
to what the FLUX.1 fits gave before the rows. Arithmetic only: no array at
an official width is allocated. CPU."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from image_fixtures import Q_HEADER, Z_HEADER, fixture_view, load_json
from test_image_memfit import ENCODER, MEASURED, VAE

from gmlx.image import memfit
from gmlx.image.memfit import GiB, ImageMemoryError, plan_memory
from gmlx.image.vae import qwen_image21_memory as mem
from gmlx.image.vae import qwen_image21_vae as q
from gmlx.image.vae import reader, tiling

FAMILY = "qwen_image21"
WEIGHTS = 330987188
SIZES = (512, 1024, 1536, 2048)
DTYPES = {2: "bfloat16", 4: "float32"}


@pytest.fixture(scope="module")
def q_view():
    return fixture_view(Q_HEADER)


@pytest.fixture(scope="module")
def z_view():
    return fixture_view(Z_HEADER)


@pytest.fixture(autouse=True)
def _no_overcommit(monkeypatch):
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)


def _unmeasured(monkeypatch):
    import dataclasses

    row = dataclasses.replace(
        memfit.VAE_ROWS[FAMILY], measured=False, whole=None, tiled=None,
        one_tile=False, tiles_first=False, scores=0.0)
    monkeypatch.setitem(memfit.VAE_ROWS, FAMILY, row)
    return row


@pytest.fixture
def by_the_rule(monkeypatch):
    """The row of this family as it stood before its file was measured:
    the floor alone, priced by the rule for an unmeasured row. The rule
    stays for the next family, and these tests hold it on this VAE."""
    return _unmeasured(monkeypatch)


@pytest.fixture(params=["fits", "rule"])
def priced_by(request, monkeypatch):
    """The row as it is, measured, and as the rule priced it."""
    if request.param == "rule":
        _unmeasured(monkeypatch)
    return request.param


# The rows


def test_every_family_has_a_vae_row():
    assert set(memfit.VAE_ROWS) == set(memfit.PEAKS) == set(reader.VAES)
    z, row = memfit.VAE_ROWS["z_image"], memfit.VAE_ROWS[FAMILY]
    assert z.measured is True and z.floor is None
    assert z.whole == {"decode": memfit.VAE_DECODE,
                       "encode": memfit.VAE_ENCODE}
    assert z.tiled is memfit.VAE_TILED
    assert z.channels == {"decode": 128, "encode": 128}
    assert (z.one_tile, z.tiles_first, z.scores, z.crop) == (
        False, False, 0.0, None)
    assert z.auto_tiles is False
    # The Qwen-Image 2.1 row is measured and keeps its floor beside the
    # fits.
    assert row.measured is True and callable(row.floor)
    assert row.whole == {"decode": memfit.QWEN_VAE_DECODE,
                         "encode": memfit.QWEN_VAE_ENCODE}
    assert row.tiled is memfit.QWEN_VAE_TILED
    assert (row.one_tile, row.tiles_first, row.scores) == (True, True, 1.5)
    assert row.auto_tiles is True and callable(row.crop)
    # No family reads a VAE fit by a channel count of its step row.
    assert not hasattr(memfit.PEAKS[FAMILY], "vae_channels")
    for price in (memfit.vae_peak, memfit.vae_tiling):
        with pytest.raises(ValueError, match="no VAE memory row for image "
                           "family 'flux'; known: qwen_image21, z_image"):
            price("decode", 512, 512, family="flux", itemsize=2)


def test_the_qwen_row_is_the_official_file():
    from gmlx.image.registry import get_family

    head = load_json("qwen_image21_vae.header.json")
    cfg = q.config_from_shapes({n: tuple(s) for n, s, _ in head["tensors"]})
    assert mem.OFFICIAL == cfg
    row = memfit.VAE_ROWS[FAMILY]
    assert row.channels == {"decode": cfg.decoder_base_dim,
                            "encode": cfg.base_dim}
    assert row.channels == {"decode": 144, "encode": 96}
    # One factor of 16: the config's levels, the family's constant and the
    # plan's.
    consts = get_family(FAMILY).constants.values
    assert cfg.scale == consts["vae_scale_factor"] == 16
    assert memfit.vae_scale(FAMILY) == 16 and memfit.vae_scale("z_image") == 8
    steps, count = row.floor("decode", 1024, 1024, itemsize=2, batch=1,
                             tiled=False, tile=512, overlap=64)
    assert count == mem.weight_count(cfg) == WEIGHTS
    assert steps == mem.memory_terms(cfg, "decode", 1024, 1024, itemsize=2)


def test_the_tile_layout_follows_the_scale_of_the_vae():
    for px in range(16, 4112, 16):
        lat = px // 16
        length = tiling.tile_length(lat, memfit.TILE_PX // 16,
                                    memfit.OVERLAP_PX // 16)
        starts = tiling.tile_starts(lat, length, memfit.OVERLAP_PX // 16)
        assert memfit.tile_layout(px, scale=16) == (len(starts),
                                                    16 * length), px
    # Two tiles cover 720 pixels in 392 at 8 pixels per latent pixel, and
    # in 400 at 16.
    assert memfit.tile_layout(720) == (2, 392)
    assert memfit.tile_layout(720, scale=16) == (2, 400)


def test_a_measured_row_lays_its_tiles_out_at_the_scale_of_its_family(
        monkeypatch):
    """What the row of this family prices once a ladder has set its fits:
    tiles of whole latent pixels at 16 pixels each."""
    import dataclasses

    fit = {2: memfit.VaeFit(0.0, 1.0)}
    plain = dataclasses.replace(
        memfit.VAE_ROWS[FAMILY], measured=True, floor=None,
        whole={"decode": {"exact": fit}},
        tiled={"decode": {2: memfit.TiledFit(0.0, 0.5, 0.0)}},
        one_tile=False, tiles_first=False, scores=0.0)
    monkeypatch.setitem(memfit.VAE_ROWS, FAMILY, plain)
    kw = dict(family=FAMILY, itemsize=2)
    assert memfit.vae_peak("decode", 720, 720, **kw) == 720 * 720 * 144 * 2
    # Four tiles of 400 pixels a side, at half a byte per tile pixel,
    # channel and activation byte.
    assert memfit.vae_peak("decode", 720, 720, tiled=True,
                           **kw) == 2 * 400 * 400 * 144 * 2
    # A measured row follows the free figure, as the Z-Image row does.
    whole = 720 * 720 * 144 * 2
    assert memfit.vae_tiling("decode", 720, 720, free=whole, **kw) is False
    assert memfit.vae_tiling("decode", 720, 720, free=whole - 1,
                             **kw) is True
    # A row whose tiles run one at a time counts one of the four.
    monkeypatch.setitem(memfit.VAE_ROWS, FAMILY, dataclasses.replace(
        plain, one_tile=True))
    assert memfit.vae_peak("decode", 720, 720, tiled=True,
                           **kw) == 400 * 400 * 144 * 2 // 2
    # A row that takes tiles first runs in tiles with room for a whole
    # pass, and whole when that is forced.
    monkeypatch.setitem(memfit.VAE_ROWS, FAMILY, dataclasses.replace(
        plain, tiles_first=True))
    assert memfit.vae_tiling("decode", 720, 720, free=whole, **kw) is True
    assert memfit.vae_tiling("decode", 720, 720, free=whole, tiled=False,
                             **kw) is False
    # The scores of the mid attention: 45 x 45 latent pixels, squared, at
    # 2 bytes, times the factor of the row, on a whole pass and a tiled.
    monkeypatch.setitem(memfit.VAE_ROWS, FAMILY, dataclasses.replace(
        plain, scores=1.5))
    held = int(1.5 * (45 * 45) ** 2 * 2)
    assert memfit.vae_peak("decode", 720, 720, **kw) == whole + held
    assert memfit.vae_peak("decode", 720, 720, tiled=True,
                           **kw) == 2 * 400 * 400 * 144 * 2 + held
    assert memfit.vae_peak("decode", 720, 720, batch=2,
                           **kw) == 2 * (whole + held)


# The measured row of the Qwen-Image 2.1 VAE

# The ladder on the official file, on an M5 Max under bounded_cache: what
# a pass added over the settled process with the weights loaded, in GiB,
# as (growth of the process footprint, MLX peak over the active bytes at
# the start), each the largest over two fresh processes of a cold and a
# warm pass. (op, whole or tiled, pixels a side, itemsize, conv path). No
# pass ran whole above 1024 pixels. The tiled decodes blend 3, 4 and 5
# tiles a side of 384, 432 and 464 pixels, and the tiled encodes crop 4, 7
# and 9 tiles a side of 496, 496 and 512 pixels.
CELLS = {
    ("decode", "whole", 256, 2, "exact"): (1.10, 0.77),
    ("decode", "whole", 512, 2, "exact"): (2.45, 1.06),
    ("decode", "whole", 1024, 2, "exact"): (5.50, 3.66),
    ("decode", "whole", 256, 4, "exact"): (2.19, 1.54),
    ("decode", "whole", 512, 4, "exact"): (4.31, 2.12),
    ("decode", "whole", 1024, 4, "exact"): (8.74, 7.32),
    ("decode", "whole", 256, 2, "winograd"): (1.90, 1.43),
    ("decode", "whole", 512, 2, "winograd"): (3.53, 2.34),
    ("decode", "whole", 1024, 2, "winograd"): (7.83, 7.00),
    ("decode", "whole", 256, 4, "winograd"): (2.77, 2.85),
    ("decode", "whole", 512, 4, "winograd"): (5.11, 4.67),
    ("decode", "whole", 1024, 4, "winograd"): (13.58, 14.00),
    ("encode", "whole", 256, 2, "exact"): (0.37, 0.35),
    ("encode", "whole", 512, 2, "exact"): (1.04, 0.81),
    ("encode", "whole", 1024, 2, "exact"): (2.79, 1.56),
    ("encode", "whole", 256, 4, "exact"): (0.72, 0.70),
    ("encode", "whole", 512, 4, "exact"): (2.05, 1.65),
    ("encode", "whole", 1024, 4, "exact"): (3.96, 3.11),
    ("encode", "whole", 256, 2, "winograd"): (0.48, 0.46),
    ("encode", "whole", 512, 2, "winograd"): (1.95, 1.25),
    ("encode", "whole", 1024, 2, "winograd"): (3.28, 2.62),
    ("encode", "whole", 256, 4, "winograd"): (0.94, 0.92),
    ("encode", "whole", 512, 4, "winograd"): (2.51, 2.50),
    ("encode", "whole", 1024, 4, "winograd"): (4.40, 5.22),
    ("decode", "tiled", 1024, 2, "exact"): (1.79, 0.80),
    ("decode", "tiled", 1536, 2, "exact"): (2.29, 0.86),
    ("decode", "tiled", 2048, 2, "exact"): (2.95, 1.18),
    ("decode", "tiled", 1024, 4, "exact"): (3.46, 1.61),
    ("decode", "tiled", 1536, 4, "exact"): (3.86, 1.72),
    ("decode", "tiled", 2048, 4, "exact"): (4.00, 2.08),
    ("encode", "tiled", 1024, 2, "exact"): (1.18, 0.80),
    ("encode", "tiled", 1536, 2, "exact"): (1.48, 0.81),
    ("encode", "tiled", 2048, 2, "exact"): (1.73, 1.01),
    ("encode", "tiled", 1024, 4, "exact"): (2.07, 1.58),
    ("encode", "tiled", 1536, 4, "exact"): (2.70, 1.60),
    ("encode", "tiled", 2048, 4, "exact"): (3.39, 1.77),
}
# How far above its cell a price may stand, in GiB. A whole fit raises its
# slope 10 percent and its base 0.25 GiB over every cell. A tiled price
# adds the scores of the mid attention to one tile, which do not peak
# together, and takes 1.5 times the bytes of the scores where fp32 held
# 1.18 times.
SLACK = {"whole": 1.6, "tiled": 2.0}


def _fit_price(op: str, size: int, item: int, **kw) -> int:
    return memfit.vae_peak(op, size, size, family=FAMILY, itemsize=item,
                           **kw)


@pytest.mark.parametrize("cell", list(CELLS), ids=lambda c: "-".join(
    str(v) for v in c))
def test_the_fits_cover_every_measured_pass(cell):
    op, mode, size, item, conv = cell
    added = max(CELLS[cell])
    price = _fit_price(op, size, item, conv=conv,
                       tiled=mode == "tiled") / GiB
    # At or above the pass, and within the slack of it.
    assert added <= price <= added + SLACK[mode], (cell, price)


# The prices of the official file by its fits, in GiB over the weights:
# whole at 512, 1024, 1536 and 2048 pixels, and in tiles at the last three.
FIT_PRICES = {
    ("decode", 2): ((2.71, 6.48, 12.90, 22.12), (2.24, 2.67, 3.38)),
    ("decode", 4): ((4.57, 10.18, 19.77, 33.69), (3.89, 4.59, 5.89)),
    ("encode", 2): ((1.30, 3.42, 7.08, 12.45), (1.50, 1.74, 2.36)),
    ("encode", 4): ((2.32, 5.05, 9.84, 17.03), (2.53, 2.96, 4.10)),
}
FIT_WINOGRAD = {
    ("decode", 2): ((3.79, 8.91, 17.58, 29.95), (3.13, 3.62, 4.38)),
    ("decode", 4): ((5.57, 15.49, 32.25, 56.21), (4.27, 5.19, 6.64)),
    ("encode", 2): ((2.21, 4.43, 8.26, 13.85), (2.41, 2.65, 3.27)),
    ("encode", 4): ((2.77, 6.43, 12.76, 22.12), (2.96, 3.39, 4.55)),
}


@pytest.mark.parametrize("op,item", list(FIT_PRICES))
def test_the_prices_of_the_official_file(op, item):
    for conv, table in (("exact", FIT_PRICES), ("winograd", FIT_WINOGRAD)):
        whole, tiled = table[op, item]
        assert tuple(round(_fit_price(op, s, item, conv=conv) / GiB, 2)
                     for s in SIZES) == whole, conv
        assert tuple(round(_fit_price(op, s, item, conv=conv,
                                      tiled=True) / GiB, 2)
                     for s in SIZES[1:]) == tiled, conv
    # An image of one tile is priced whole.
    assert _fit_price(op, 512, item, tiled=True) == _fit_price(op, 512, item)


@pytest.mark.parametrize("item", [2, 4])
def test_a_price_is_the_fit_and_the_scores_over_the_weights(item):
    row = memfit.VAE_ROWS[FAMILY]
    # 128 x 128 latent pixels at 2048: every score of the mid attention
    # once, times 1.5.
    held = int(1.5 * (128 * 128) ** 2 * item)
    assert held == {2: 0.75, 4: 1.5}[item] * GiB
    fit = row.whole["decode"]["exact"][item]
    assert _fit_price("decode", 2048, item) == fit.bytes(
        2048 * 2048, 144, item) + held
    # In tiles: one tile of 464 pixels a side of the 25, the image term,
    # and the same scores, since the part at latent resolution runs whole.
    assert memfit.tile_layout(2048, scale=16) == (5, 464)
    tiled = row.tiled["decode"][item]
    assert _fit_price("decode", 2048, item, tiled=True) == tiled.bytes(
        464 * 464, 2048 * 2048, 144, item) + held
    # The tile term of a decode is the whole fit on the exact path, so a
    # tile alone costs what a whole pass of its size does.
    assert (tiled.base, tiled.k_tiles) == (fit.base, fit.k)
    # A tiled encode is priced at the overlap of its crop: 9 tiles a side
    # of 512 pixels at 2048, where a blend would lay out 5 of 464.
    assert memfit.tile_overlap(FAMILY, "encode") == 320
    assert memfit.tile_layout(2048, 512, 320, 16) == (9, 512)
    assert _fit_price("encode", 2048, item, tiled=True) == row.tiled[
        "encode"][item].bytes(512 * 512, 2048 * 2048, 96, item) + held


def test_a_vae_of_another_config_is_priced_at_its_own_channels_and_rows():
    """Tiling passes the channels and the config of the VAE at hand. The
    plan passes none, which is the official file."""
    small = q.QwenImage21VAEConfig(
        in_channels=4, out_channels=4, latent_channels=64, base_dim=4,
        decoder_base_dim=6, dim_mult=(1, 2, 4), num_res_blocks=1,
        temporal=(False, True))
    assert small.scale == 4
    kw = dict(family=FAMILY, itemsize=4)
    fit = memfit.QWEN_VAE_DECODE["exact"][4]
    got = memfit.vae_peak("decode", 60, 44, config=small, channels=6, **kw)
    # 15 x 11 latent pixels at 4 pixels each.
    assert got == fit.bytes(60 * 44, 6, 4) + int(1.5 * (15 * 11) ** 2 * 4)
    assert got < memfit.vae_peak("decode", 64, 48, **kw)
    assert memfit.vae_peak("decode", 512, 512, config=mem.OFFICIAL,
                           channels=144, **kw) == memfit.vae_peak(
                               "decode", 512, 512, **kw)
    # The tiles of a measured row are laid out at the scale of its family:
    # 100 pixels in tiles of 48 that overlap by 8 are priced as three of
    # 48, where that VAE cuts three of 40. The scores are those of its own
    # 25 x 25 latent pixels.
    assert memfit.tile_layout(100, 48, 8, 4) == (3, 40)
    assert memfit.tile_layout(100, 48, 8, 16) == (3, 48)
    assert memfit.vae_peak(
        "decode", 100, 100, config=small, channels=6, tiled=True, tile=48,
        overlap=8, **kw) == memfit.QWEN_VAE_TILED["decode"][4].bytes(
            48 * 48, 100 * 100, 6, 4) + int(1.5 * (25 * 25) ** 2 * 4)


def test_tiles_come_first_whatever_is_free():
    kw = dict(family=FAMILY, itemsize=2)
    whole = _fit_price("decode", 1024, 2)
    tiles = _fit_price("decode", 1024, 2, tiled=True)
    assert tiles < whole
    for free in (1e15, whole, tiles):
        assert memfit.vae_tiling("decode", 1024, 1024, free=free,
                                 **kw) is True
    # One tile runs whole, and a larger image whole when that is forced.
    assert memfit.vae_tiling("decode", 512, 512, free=1e15, **kw) is False
    assert memfit.vae_tiling("decode", 1024, 1024, free=whole, tiled=False,
                             **kw) is False
    with pytest.raises(ImageMemoryError) as e:
        memfit.vae_tiling("decode", 1024, 1024, free=whole - 1, tiled=False,
                          overcommit=False, **kw)
    # A measured row says nothing of the rule.
    assert str(e.value).endswith(
        f"a tiled pass needs {tiles / GiB:.2f} GiB. GMLX_OVERCOMMIT=1 "
        "overrides.")
    with pytest.raises(ImageMemoryError, match="in tiles needs"):
        memfit.vae_tiling("decode", 1024, 1024, free=tiles - 1,
                          overcommit=False, **kw)


def test_a_16_gb_mac_has_the_tiled_decode_and_not_the_whole_one():
    """Why tiles come first on this row. The working budget of a 16 GB
    Mac, the Q4_K_M DiT and the bf16 VAE resident and the encoder
    released: a tiled decode at 2048 pixels fits, and a whole one at 1024
    does not."""
    budget = 17_179_869_184 - 8e9
    resident = 4199565024 + WEIGHTS * 2
    assert resident + _fit_price("decode", 2048, 2, tiled=True) < budget
    assert resident + _fit_price("decode", 1024, 2) > budget


# The rule of an unmeasured row


def test_the_rule_of_an_unmeasured_row():
    bound = memfit.CACHE_BOUND
    assert bound == 2 * GiB
    # (name, device, softmax, host). The ratio covers the device bytes and
    # the softmax, and the host bytes of the same step come on top: 360,
    # 824 and 1036 bytes at 3.6, and 540, 986 and 1054 at 5.4.
    steps = [("a", 100, 0, 0), ("b", 60, 30, 500), ("c", 10, 0, 1000)]
    assert memfit.unmeasured_bytes(steps, 7) == 1036 + 7 + bound
    assert memfit.unmeasured_bytes(steps[:2], 7) == 824 + 7 + bound
    assert memfit.unmeasured_bytes(steps[:1], 0) == 360 + bound
    assert memfit.unmeasured_bytes(steps, 7, "winograd") == 1054 + 7 + bound
    assert memfit.unmeasured_bytes(steps[:2], 0, "winograd") == 986 + bound
    with pytest.raises(KeyError):
        memfit.unmeasured_bytes(steps, 7, "other")


def test_the_ratio_is_the_flux1_decode_at_1024_over_its_floor():
    # Five 128 channel activations at 1024 pixels in bf16.
    floor = 5 * 128 * 1024 * 1024 * 2
    assert floor == 1.25 * GiB
    assert memfit.UNMEASURED_RATIO == {"exact": 3.6, "winograd": 5.4}
    for conv, ratio in memfit.UNMEASURED_RATIO.items():
        bf16 = MEASURED[conv, "decode", 2][1024] * GiB / floor
        fp32 = MEASURED[conv, "decode", 4][1024] * GiB / (2 * floor)
        assert math.ceil(bf16 * 10) / 10 == ratio, conv
        assert fp32 < bf16


def test_the_rule_covers_the_decodes_of_the_vae_it_borrows_from():
    """The rule, given the FLUX.1 autoencoder's own floor and weights,
    stays above every measured whole decode of it."""
    for (conv, op, item), sizes in MEASURED.items():
        if op != "decode":
            continue
        for size, gib in sizes.items():
            floor = 5 * 128 * size * size * item
            price = memfit.unmeasured_bytes([("peak", floor, 0, 0)],
                                            VAE // 2 * item, conv)
            assert price >= gib * GiB, (conv, item, size)
    # The Winograd path needs its own ratio: at 3.6 the rule falls under
    # the measured bf16 decode at 512 pixels.
    floor = 5 * 128 * 512 * 512 * 2
    exact = memfit.unmeasured_bytes([("peak", floor, 0, 0)], VAE, "exact")
    assert exact < MEASURED["winograd", "decode", 2][512] * GiB


# The prices of the official file by the rule, in GiB with the weights:
# whole at 512, 1024, 1536 and 2048 pixels, and in tiles at the last three.
# The tiles of an encode are cropped and overlap by 320 pixels, so a side
# of 1024 takes four tiles of 496 pixels where a decode takes three of 384.
PRICES = {
    ("decode", 2): ((4.14, 8.69, 16.29, 26.92), (3.52, 3.87, 4.15)),
    ("decode", 4): ((6.27, 15.38, 30.57, 51.83), (5.02, 5.74, 6.30)),
    ("encode", 2): ((3.29, 5.32, 8.69, 13.42), (3.30, 3.88, 4.15)),
    ("encode", 4): ((4.58, 8.63, 15.38, 24.83), (4.55, 5.63, 6.09)),
}


def _price(op: str, size: int, item: int, **kw) -> int:
    return memfit.vae_peak(op, size, size, family=FAMILY, itemsize=item,
                           **kw)


@pytest.mark.parametrize("op,item", list(PRICES))
def test_the_prices_of_the_official_file_by_the_rule(op, item, by_the_rule):
    whole, tiled = PRICES[op, item]
    assert tuple(round(_price(op, s, item) / GiB, 2) for s in SIZES) == whole
    assert tuple(round(_price(op, s, item, tiled=True) / GiB, 2)
                 for s in SIZES[1:]) == tiled
    # An image of one tile is priced whole.
    assert _price(op, 512, item, tiled=True) == _price(op, 512, item)
    weights, bound = WEIGHTS * item, memfit.CACHE_BOUND
    for s in SIZES:
        # A whole pass: 3.6 times the op-order floor, the weights at the
        # pass dtype and the cache bound.
        floor, _ = mem.activation_peak(mem.OFFICIAL, op, s, s, itemsize=item)
        assert _price(op, s, item) == int(3.6 * floor + weights + bound)
        for in_tiles in (False, True):
            steps = mem.memory_terms(mem.OFFICIAL, op, s, s, itemsize=item,
                                     tiled=in_tiles,
                                     overlap=memfit.tile_overlap(FAMILY, op))
            assert _price(op, s, item, tiled=in_tiles) == int(max(
                3.6 * (device + softmax) + host
                for _, device, softmax, host in steps) + weights + bound)


@pytest.mark.parametrize("item", [2, 4])
def test_a_tiled_decode_is_priced_on_one_tile_and_the_host_buffers(
        item, by_the_rule):
    """At 1024 pixels the step that sets the price is in a tile: three
    tiles of 384 pixels a side, one at a time beside the part run whole,
    with the blended image and its weights on the host."""
    steps = mem.memory_terms(mem.OFFICIAL, "decode", 1024, 1024,
                             itemsize=item, tiled=True)
    _, name, device, host = max((3.6 * (d + a) + h, k, d, h)
                                for k, d, a, h in steps)
    assert name.startswith("tile decoder.up_blocks.3.")
    assert host == 4 * 1024 * 1024 * 4 + 4 * 1024 * 1024
    head = 1152 * 64 * 64 * item
    one, _ = mem.activation_peak(mem.OFFICIAL, "decode", 384, 384,
                                 itemsize=item)
    # The tile's own floor is that of a whole 384 pixel decode, which
    # peaks in the same block.
    assert device == one + head
    assert _price("decode", 1024, item, tiled=True) == int(
        3.6 * device + host + WEIGHTS * item + memfit.CACHE_BOUND)


def test_the_winograd_path_takes_its_own_ratio(by_the_rule):
    kw = dict(conv="winograd")
    assert [round(_price("decode", 1024, i, **kw) / GiB, 2)
            for i in (2, 4)] == [11.73, 21.46]
    floor, _ = mem.activation_peak(mem.OFFICIAL, "decode", 1024, 1024,
                                   itemsize=2)
    assert _price("decode", 1024, 2, **kw) == int(
        5.4 * floor + WEIGHTS * 2 + memfit.CACHE_BOUND)


def test_a_vae_of_another_config_is_priced_by_its_own_steps(by_the_rule):
    """On an unmeasured row the steps are those of the config of the VAE
    at hand. The plan passes none, which is the official file."""
    small = q.QwenImage21VAEConfig(
        in_channels=4, out_channels=4, latent_channels=64, base_dim=4,
        decoder_base_dim=6, dim_mult=(1, 2, 4), num_res_blocks=1,
        temporal=(False, True))
    assert small.scale == 4
    kw = dict(family=FAMILY, itemsize=4)
    got = memfit.vae_peak("decode", 60, 44, config=small, **kw)
    steps = mem.memory_terms(small, "decode", 60, 44, itemsize=4)
    assert got == int(max(3.6 * (d + a) for _, d, a, _ in steps)
                      + mem.weight_count(small) * 4 + memfit.CACHE_BOUND)
    assert got < memfit.vae_peak("decode", 64, 48, **kw)
    assert memfit.vae_peak("decode", 512, 512, config=mem.OFFICIAL,
                           **kw) == memfit.vae_peak("decode", 512, 512, **kw)
    with pytest.raises(ValueError, match="60x44 is not a multiple of 16"):
        memfit.vae_peak("decode", 60, 44, **kw)


# The plan and the guarded decode


def _stub(item: int, conv=None):
    """What tiling reads of a VAE to price a pass, for the official config
    and with no module behind it."""
    return SimpleNamespace(
        family=FAMILY, dtype=SimpleNamespace(size=item), config=mem.OFFICIAL,
        channels=lambda op: memfit.VAE_ROWS[FAMILY].channels[op], conv=conv)


def _plan(view, size: int, item: int, budget: float):
    return plan_memory(view, ENCODER, WEIGHTS * item, family=FAMILY,
                       height=size, width=size, dtype=DTYPES[item],
                       budget=budget, overcommit=False)


def _at_decode(plan, budget: float) -> float:
    """The bytes the plan leaves for the decode pass."""
    return budget - (plan.phases["decode"] - plan.terms["decode_peak"])


def _decode(plan, size: int, item: int, free: float, tiled="plan") -> bool:
    """The choice of the guarded decode with ``free`` bytes free, asked as
    ``gmlx image`` asks for a plan's request."""
    if tiled == "plan":
        tiled = True if plan.tiled_decode else None
    return tiling._plan(_stub(item), "decode", 1, size, size, tiled,
                        tiling.TILE, tiling.OVERLAP, free)


@pytest.mark.parametrize("item", [2, 4])
@pytest.mark.parametrize("size", SIZES)
def test_the_plan_and_tiling_agree_on_the_decode(q_view, size, item,
                                                 priced_by):
    plan = _plan(q_view, size, item, 200 * GiB)
    assert plan.levers == ()
    kw = tiling.price_args(_stub(item), "decode", 1)
    assert kw["family"] == FAMILY and kw["config"] == mem.OFFICIAL
    # The automatic choice, with room for a whole pass and without.
    for free in (1e15, plan.terms["decode_peak"]):
        choice = memfit.vae_tiling("decode", size, size, free=free, **kw)
        assert choice is plan.tiled_decode is (size > 512)
    price = memfit.vae_peak("decode", size, size, tiled=choice, **kw)
    assert price == plan.terms["decode_peak"]
    table = FIT_PRICES if priced_by == "fits" else PRICES
    whole, tiled = table["decode", item]
    assert round(price / GiB, 2) == (whole[0] if size == 512
                                     else tiled[SIZES.index(size) - 1])
    assert _decode(plan, size, item, price) is plan.tiled_decode
    with pytest.raises(ImageMemoryError,
                       match=f"needs {price / GiB:.2f} GiB"):
        _decode(plan, size, item, price - 1)


@pytest.mark.parametrize("item", [2, 4])
@pytest.mark.parametrize("size", SIZES)
def test_a_request_the_plan_admits_is_not_refused_at_decode_time(
        q_view, size, item, priced_by):
    admitted = []
    for gib in range(4, 42, 2):
        budget = gib * GiB
        try:
            plan = _plan(q_view, size, item, budget)
        except ImageMemoryError:
            continue
        admitted.append(gib)
        assert "tiled decode" not in plan.levers or size == 512
        free = _at_decode(plan, budget)
        assert free >= plan.terms["decode_peak"]
        assert _decode(plan, size, item, free) is plan.tiled_decode
    # A budget one step under the first that admits the request refuses
    # it, so the scan holds the tightest plans.
    assert admitted and admitted[0] > 4
    assert admitted == list(range(admitted[0], 42, 2))


@pytest.mark.parametrize("item", [2, 4])
def test_a_2048_request_on_16_gib_decodes_in_tiles(q_view, item,
                                                   monkeypatch, priced_by):
    budget = 16 * GiB
    try:
        plan = _plan(q_view, 2048, item, budget)
    except ImageMemoryError as e:
        # In fp32 the denoise step is what does not fit.
        plan = e.plan
        assert item == 4 and plan.phases["denoise"] > budget
        assert "The largest term is step_peak" in str(e)
    assert plan.tiled_decode and "tiled decode" not in plan.levers
    assert plan.phases["decode"] <= budget
    table = FIT_PRICES if priced_by == "fits" else PRICES
    whole, tiled = (t[-1] for t in table["decode", item])
    assert round(plan.terms["decode_peak"] / GiB, 2) == tiled
    free = _at_decode(plan, budget)
    assert _decode(plan, 2048, item, free) is True
    # A whole pass, forced, is refused at its price: that of the fit
    # carried on from 1024 pixels, or of the rule.
    with pytest.raises(ImageMemoryError, match="a tiled pass needs") as e:
        _decode(plan, 2048, item, free, tiled=False)
    msg = str(e.value)
    assert f"VAE decode of 1 x 2048x2048 whole needs {whole:.2f} GiB" in msg
    assert msg.endswith("GMLX_OVERCOMMIT=1 overrides." + (
        "" if priced_by == "fits" else
        " The Qwen-Image 2.1 autoencoder is not measured, and this is the "
        "price of the rule for an unmeasured VAE."))
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    assert _decode(plan, 2048, item, free, tiled=False) is False


@pytest.mark.parametrize("item", [2, 4])
def test_the_plan_counts_the_vae_weights_once(q_view, item, monkeypatch):
    """A fit is what a pass adds over the loaded weights, and the plan
    holds the weights as a resident term. The rule for an unmeasured row
    counts them in its price, so they stand twice in the decode phase of
    such a row."""
    plan = _plan(q_view, 1024, item, 200 * GiB)
    t = plan.terms
    assert t["vae"] == WEIGHTS * item
    resident = t["dit"] + t["encoder"] + t["vae"]
    assert plan.phases["decode"] == resident + t["decode_peak"]
    # Three tiles a side of 384 pixels, the image term and the scores of
    # 64 x 64 latent pixels.
    assert t["decode_peak"] == memfit.QWEN_VAE_TILED["decode"][item].bytes(
        384 * 384, 1024 * 1024, 144, item) + int(1.5 * 4096 ** 2 * item)
    _unmeasured(monkeypatch)
    rule = _plan(q_view, 1024, item, 200 * GiB)
    steps = mem.memory_terms(mem.OFFICIAL, "decode", 1024, 1024,
                             itemsize=item, tiled=True)
    assert rule.terms["decode_peak"] == memfit.unmeasured_bytes(
        steps, WEIGHTS * item)
    assert rule.phases["decode"] == resident + rule.terms["decode_peak"]


def test_a_measured_row_says_nothing_of_the_rule_when_it_refuses():
    with pytest.raises(ImageMemoryError) as e:
        memfit.vae_tiling("decode", 2048, 2048, family="z_image", itemsize=2,
                          free=GiB, overcommit=False)
    assert str(e.value).endswith("GMLX_OVERCOMMIT=1 overrides.")


# Z-Image: the prices, the plan's terms and the choices of the FLUX.1 fits
# before the rows, in bytes. (op, pixels, itemsize): whole, in tiles.
Z_PRICES = {
    ("decode", 512, 2): (2787702210, 2787702210),
    ("decode", 512, 4): (4321810841, 4321810841),
    ("decode", 1024, 2): (6029060341, 3701305507),
    ("decode", 1024, 4): (9073118412, 5058330624),
    ("decode", 2048, 2): (18994492866, 8580637655),
    ("decode", 2048, 4): (28078348697, 14609363763),
    ("encode", 512, 2): (2360889835, 2360889835),
    ("encode", 512, 4): (3964791685, 3964791685),
    ("encode", 1024, 2): (4515084369, 3201260584),
    ("encode", 1024, 4): (6904159928, 4253443686),
    ("encode", 2048, 2): (13131862507, 6768078356),
    ("encode", 2048, 4): (18661632901, 11618768650),
}
Z_WINOGRAD = {
    ("decode", 1024, 2): (8026220134, 4683443732),
    ("decode", 2048, 4): (40909563494, 15420791194),
    ("encode", 1024, 4): (9448928051, 4352932577),
}
# (pixels, activation dtype, budget): levers, and the decode term and the
# encode, denoise and decode phases.
Z_PLANS = {
    (1024, "bfloat16", 100e9): ((), 6029060341, (
        8153814535, 10371421703, 13704724220)),
    (2048, "bfloat16", 100e9): ((), 18994492866, (
        8153814535, 14696797703, 26670156745)),
    (1024, "float16", 100e9): ((), 9073118412, (
        8153814535, 10371421703, 16748782291)),
    (2048, "float16", 100e9): ((), 28078348697, (
        8153814535, 14696797703, 35754012576)),
    (1024, "bfloat16", 17_179_869_184 - 8e9): (
        ("release the encoder", "tiled decode"), 3701305507, (
            8153814535, 7880340672, 8885888355)),
    (2048, "bfloat16", 17_179_869_184 - 8e9): (
        ("release the encoder", "tiled decode"), 8580637655, (
            8153814535, 12205716672, 13765220503)),
}


def test_the_z_image_prices_are_those_before_the_rows():
    for conv, table in (("exact", Z_PRICES), ("winograd", Z_WINOGRAD)):
        for (op, size, item), (whole, tiled) in table.items():
            kw = dict(family="z_image", itemsize=item, conv=conv)
            assert memfit.vae_peak(op, size, size, **kw) == whole
            assert memfit.vae_peak(op, size, size, tiled=True,
                                   **kw) == tiled
            # The row's own channels are those tiling reads of the module.
            assert memfit.vae_peak(op, size, size, channels=128,
                                   **kw) == whole


def test_the_z_image_plans_are_those_before_the_rows(z_view):
    for (size, dtype, budget), (levers, decode, phases) in Z_PLANS.items():
        try:
            plan = plan_memory(z_view, ENCODER, VAE, family="z_image",
                               height=size, width=size, dtype=dtype,
                               budget=budget, overcommit=False)
        except ImageMemoryError as e:
            plan = e.plan
            assert (size, budget) == (2048, 17_179_869_184 - 8e9)
        assert plan.levers == levers
        assert plan.tiled_decode is ("tiled decode" in levers)
        assert plan.terms["decode_peak"] == decode
        assert tuple(int(plan.phases[k]) for k in
                     ("encode", "denoise", "decode")) == phases


def _z_choices(op: str, size: int, item: int, tiled, frees) -> list:
    out = []
    for free in frees:
        try:
            out.append(memfit.vae_tiling(
                op, size, size, family="z_image", itemsize=item, tiled=tiled,
                free=free, overcommit=False))
        except ImageMemoryError:
            out.append("refused")
    return out


def test_the_z_image_choices_are_those_before_the_rows():
    """With a free figure a measured row runs whole when that fits, in
    tiles when only they fit, and is refused otherwise."""
    for (op, size, item), (whole, tiles) in Z_PRICES.items():
        if size == 512:
            continue
        frees = (whole, whole - 1, tiles, tiles - 1)
        assert _z_choices(op, size, item, None, frees) == [
            False, True, True, "refused"]
        assert _z_choices(op, size, item, False, frees) == [
            False, "refused", "refused", "refused"]
        assert _z_choices(op, size, item, True, frees) == [
            True, True, True, "refused"]
