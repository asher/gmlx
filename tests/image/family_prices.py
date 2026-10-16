"""What an image family prices and pulls, as one JSON-ready dict: its peak
row, its plans at four budgets, its VAE passes, its recipes, their default
quants and sizes, and the families its VAE serves. A test compares it with
tests/fixtures/image/memfit/family_prices.json to show that new rows leave
the families before them as they were."""
from __future__ import annotations

import itertools
import json

GiB = 1 << 30
BUDGETS = (8 * GiB, 15 * GiB, 20 * GiB, 48 * GiB)
REQUESTS = tuple(itertools.product((512, 1024), (1.0, 4.0), (0, 1)))


def _plan(memfit, family, budget, size, cfg, refs):
    try:
        p = memfit.plan_memory(4 * GiB, 4 * GiB, GiB // 4, family=family,
                               height=size, width=size, cfg=cfg, refs=refs,
                               ref_px=1024 if refs else 0, budget=budget,
                               overcommit=False)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return [p.phases, p.terms, p.rows, list(p.levers), p.fits,
            p.release_encoder, p.tiled_decode]


def _vae(memfit, family, op, px, tiled):
    try:
        return memfit.vae_peak(op, px, px, family=family, itemsize=2,
                               tiled=tiled)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def family_prices(skip=()) -> dict:
    """The prices and pulls of every family but those in ``skip``."""
    from gmlx.image import catalog, memfit
    from gmlx.image.registry import families
    from gmlx.image.vae.reader import VAES

    out = {"order": [[f.name, f.order] for f in families()
                     if f.name not in skip]}
    for fam in sorted(memfit.PEAKS):
        if fam in skip:
            continue
        out[fam] = {
            "row": repr(memfit.PEAKS[fam]),
            "plans": {f"{b} {s} {c} {r}": _plan(memfit, fam, b, s, c, r)
                      for b in BUDGETS for s, c, r in REQUESTS},
            "vae": {f"{op} {px} {t}": _vae(memfit, fam, op, px, t)
                    for op in ("decode", "encode") for px in (512, 1024)
                    for t in (False, True)},
        }
    names = [n for n, r in catalog.RECIPES.items() if r.family not in skip]
    out["recipes"] = {n: repr(catalog.RECIPES[n]) for n in names}
    out["quants"] = {n: catalog.DEFAULT_QUANTS[n] for n in names}
    out["sizes"] = {n: s for n, s in catalog.DEFAULT_SIZES.items()
                    if n in names}
    out["vae_families"] = {
        f: [g for g in catalog.vae_families(f) if g not in skip]
        for f in VAES if f not in skip}
    return json.loads(json.dumps(out))
