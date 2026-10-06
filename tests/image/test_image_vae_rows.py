"""The VAE rows of the image memory plan: a row per family, the rule that
prices a pass on an unmeasured row, the prices of the Qwen-Image 2.1 VAE
at the widths of the official header, the plan's decode term against the
choice and the price of the guarded decode, and the Z-Image values held
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


# The rows


def test_every_family_has_a_vae_row():
    assert set(memfit.VAE_ROWS) == set(memfit.PEAKS) == set(reader.VAES)
    z, row = memfit.VAE_ROWS["z_image"], memfit.VAE_ROWS[FAMILY]
    assert z.measured is True and z.floor is None
    assert z.whole == {"decode": memfit.VAE_DECODE,
                       "encode": memfit.VAE_ENCODE}
    assert z.tiled is memfit.VAE_TILED
    assert z.channels == {"decode": 128, "encode": 128}
    assert row.measured is False and callable(row.floor)
    assert row.whole is None and row.tiled is None
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
    monkeypatch.setitem(memfit.VAE_ROWS, FAMILY, dataclasses.replace(
        memfit.VAE_ROWS[FAMILY], measured=True, floor=None,
        whole={"decode": {"exact": fit}},
        tiled={"decode": {2: memfit.TiledFit(0.0, 0.5, 0.0)}}))
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


# The prices of the official file, in GiB: whole at 512, 1024, 1536 and
# 2048 pixels, and in tiles at the last three.
PRICES = {
    ("decode", 2): ((4.14, 8.69, 16.29, 26.92), (3.52, 3.87, 4.15)),
    ("decode", 4): ((6.27, 15.38, 30.57, 51.83), (5.02, 5.74, 6.30)),
    ("encode", 2): ((3.29, 5.32, 8.69, 13.42), (3.05, 3.88, 4.15)),
    ("encode", 4): ((4.58, 8.63, 15.38, 24.83), (4.05, 5.63, 6.09)),
}


def _price(op: str, size: int, item: int, **kw) -> int:
    return memfit.vae_peak(op, size, size, family=FAMILY, itemsize=item,
                           **kw)


@pytest.mark.parametrize("op,item", list(PRICES))
def test_the_prices_of_the_official_file(op, item):
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
                                     tiled=in_tiles)
            assert _price(op, s, item, tiled=in_tiles) == int(max(
                3.6 * (device + softmax) + host
                for _, device, softmax, host in steps) + weights + bound)


@pytest.mark.parametrize("item", [2, 4])
def test_a_tiled_decode_is_priced_on_one_tile_and_the_host_buffers(item):
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


def test_the_winograd_path_takes_its_own_ratio():
    kw = dict(conv="winograd")
    assert [round(_price("decode", 1024, i, **kw) / GiB, 2)
            for i in (2, 4)] == [11.73, 21.46]
    floor, _ = mem.activation_peak(mem.OFFICIAL, "decode", 1024, 1024,
                                   itemsize=2)
    assert _price("decode", 1024, 2, **kw) == int(
        5.4 * floor + WEIGHTS * 2 + memfit.CACHE_BOUND)


def test_a_vae_of_another_config_is_priced_by_its_own_steps():
    """Tiling passes the config of the VAE at hand. The plan passes none,
    which is the official file."""
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
def test_the_plan_and_tiling_agree_on_the_decode(q_view, size, item):
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
    whole, tiled = PRICES["decode", item]
    assert round(price / GiB, 2) == (whole[0] if size == 512
                                     else tiled[SIZES.index(size) - 1])
    assert _decode(plan, size, item, price) is plan.tiled_decode
    with pytest.raises(ImageMemoryError,
                       match=f"needs {price / GiB:.2f} GiB"):
        _decode(plan, size, item, price - 1)


@pytest.mark.parametrize("item", [2, 4])
@pytest.mark.parametrize("size", SIZES)
def test_a_request_the_plan_admits_is_not_refused_at_decode_time(
        q_view, size, item):
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
                                                   monkeypatch):
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
    assert round(plan.terms["decode_peak"] / GiB, 2) == {2: 4.15,
                                                        4: 6.30}[item]
    free = _at_decode(plan, budget)
    assert _decode(plan, 2048, item, free) is True
    # A whole pass, forced, is refused at the price of the rule.
    whole = {2: 26.92, 4: 51.83}[item]
    with pytest.raises(ImageMemoryError, match="a tiled pass needs") as e:
        _decode(plan, 2048, item, free, tiled=False)
    msg = str(e.value)
    assert f"VAE decode of 1 x 2048x2048 whole needs {whole:.2f} GiB" in msg
    assert msg.endswith(
        "GMLX_OVERCOMMIT=1 overrides. The Qwen-Image 2.1 autoencoder is not "
        "measured, and this is the price of the rule for an unmeasured VAE.")
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    assert _decode(plan, 2048, item, free, tiled=False) is False


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
