"""Patch table entries from LoRA files and from ComfyUI patches.

``load_adapter`` reads a LoRA file through the dialect reader, places its
pairs with the family key map, checks that each fits its module and keeps
the pairs in the pipeline's cache. ``table_entries`` turns loaded adapters
and their strengths into the entries ``define_patch_table`` takes, and
stacked adapters share an entry per module and band. A strength change is
a new list of entries over the same cached pairs.

``plans_from_patches`` gives the same entries for a ComfyUI patch list.
ComfyUI's key names equal the GGUF names, so the module path is the key
without the model prefix and ``.weight``, and the band is the patch offset
of (dim, start, length). The pairs arrive through ``register_patch`` under
their content hash, and the dialect reader is not involved.

``apply_text_adapters`` installs the text encoder halves through the text
adapter wrap, on an encoder the pipeline owns.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass

import mlx.core as mx

from . import plan
from .install import linear_dims, linear_leaves, register_pair
from .reader import (AdapterError, UnsupportedAdapterError, read_lora,
                     resolve)


@dataclass(frozen=True)
class LoadedAdapter:
    """A LoRA file placed on a pipeline. ``dit`` rows are (module path,
    band, pair hash) and ``text`` rows (encoder module path, pair hash).
    ``sha256`` names the content: every row with its hash."""
    path: str
    sha256: str
    dit: tuple
    text: tuple
    warnings: tuple = ()
    nbytes: int = 0

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def key_map(pipe):
    """The family's LoRA key map for this pipeline's DiT."""
    hit = getattr(pipe, "_lora_key_map", None)
    if hit is not None:
        return hit
    make = pipe.family.lora_map
    if make is None:
        raise AdapterError(f"{pipe.family.title} has no LoRA key map in "
                           "gmlx")
    cfg = pipe.dit.cfg
    cfg = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) \
        and not isinstance(cfg, type) else dict(cfg)
    pipe._lora_key_map = hit = make(cfg)
    return hit


def _file_key(path: str) -> tuple:
    st = os.stat(path)
    return (os.path.realpath(path), st.st_size, st.st_mtime_ns)


def load_adapter(pipe, path: str, *,
                 allow_unmatched: bool = False) -> LoadedAdapter:
    """Read a LoRA file and keep its pairs in the pipeline's cache. A file
    this pipeline already read, unchanged on disk and with its pairs still
    held, is not read again."""
    path = os.path.expanduser(str(path))
    try:
        fkey = _file_key(path)
    except OSError as e:
        raise AdapterError(f"{path}: {e.strerror or e}") from None
    hit = pipe.adapter_files.get(fkey)
    if hit is not None and all(pipe.adapters.get(row[-1]) is not None
                               for row in hit.dit + hit.text):
        return hit
    raw = read_lora(path, allow_unmatched=allow_unmatched)
    res = resolve(raw, key_map(pipe), allow_unmatched=allow_unmatched)
    leaves = linear_leaves(pipe.dit)
    for t in res.dit:
        leaf = leaves.get(t.module)
        if leaf is None:
            raise AdapterError(f"{path}: {t.pair.stem} targets {t.module}, "
                               "which this DiT does not have")
        why = plan.fit(tuple(t.pair.up.shape), tuple(t.pair.down.shape),
                       t.band, linear_dims(leaf))
        if why is not None:
            raise AdapterError(
                f"{path}: {t.pair.stem} does not fit {t.module} of this "
                f"{pipe.family.title} DiT: {why}")
    dit, text, nbytes = [], [], 0
    for t in res.dit:
        key = register_pair(pipe, t.pair.down, t.pair.up, t.pair.alpha)
        dit.append((t.module, t.band, key))
        nbytes += int(t.pair.down.nbytes) + int(t.pair.up.nbytes)
    for t in res.text:
        key = register_pair(pipe, t.pair.down, t.pair.up, t.pair.alpha)
        text.append((t.module, key))
        nbytes += int(t.pair.down.nbytes) + int(t.pair.up.nbytes)
    doc = json.dumps({"dit": sorted([m, b, k] for m, b, k in dit),
                      "text": sorted(text)}, sort_keys=True,
                     separators=(",", ":"), default=list)
    loaded = LoadedAdapter(path=path,
                           sha256=hashlib.sha256(doc.encode()).hexdigest(),
                           dit=tuple(dit), text=tuple(text),
                           warnings=res.warnings, nbytes=nbytes)
    pipe.adapter_files[fkey] = loaded
    return loaded


def table_entries(adapters) -> list:
    """Entries for ``define_patch_table`` from (LoadedAdapter, strength)
    rows. Adapters on the same module and band share one entry, in the
    order given."""
    rows: dict = {}
    for ad, strength in adapters:
        for module, band, key in ad.dit:
            rows.setdefault((module, band), []).append((key, float(strength)))
    return [(m, b, p) for (m, b), p in rows.items()]


# ComfyUI patches.

def register_patch(pipe, up, down, alpha=None, mid=None, dora_scale=None,
                   reshape=None) -> str:
    """Keep the pair of a ComfyUI LoRA patch, given in its tuple order, and
    return its content hash. A patch with a mid tensor, a DoRA scale or a
    reshape is refused by name."""
    for value, kind in ((mid, "Tucker LoCon"), (dora_scale, "DoRA"),
                        (reshape, "reshape")):
        if value is not None:
            raise UnsupportedAdapterError(
                f"a {kind} patch is not applied by gmlx image. It applies "
                "plain LoRA pairs only", kind)
    up, down = mx.array(up), mx.array(down)
    if up.ndim > 2:
        up = up.reshape(up.shape[0], -1)
    if down.ndim > 2:
        down = down.reshape(down.shape[0], -1)
    if alpha is not None:
        alpha = float(mx.array(alpha).astype(mx.float32).reshape(-1)[0]
                      .item())
    return register_pair(pipe, down, up, alpha)


def _module_of(key: str, family) -> str:
    for p in family.strip_prefixes:
        if key.startswith(p):
            key = key[len(p):]
            break
    if key.endswith(".bias"):
        raise UnsupportedAdapterError(
            f"a patch on {key} is a bias patch, which gmlx image does not "
            "apply", "full-weight diff")
    return key[:-len(".weight")] if key.endswith(".weight") else key


def plans_from_patches(family, patches) -> list:
    """Entries for ``define_patch_table`` from a ComfyUI patch list. Each
    patch is a mapping of ``key``, the model's weight key, ``band``, the
    offset (dim, start, length) or None, and ``patches``, a list of
    mappings of ``adapter``, the pair hash, ``strength`` and an optional
    ``strength_model``. A model strength other than 1 scales the base
    weight, which is refused."""
    rows: dict = {}
    for patch in patches:
        module = _module_of(str(patch["key"]), family)
        band = patch.get("band", patch.get("offset"))
        band = None if band is None else tuple(int(v) for v in band)
        for p in patch["patches"]:
            if float(p.get("strength_model", 1.0)) != 1.0:
                raise UnsupportedAdapterError(
                    f"the patch on {module} scales the base weight by "
                    f"{p['strength_model']}, which gmlx image does not "
                    "apply", "model strength")
            rows.setdefault((module, band), []).append(
                (str(p["adapter"]), float(p["strength"])))
    return [(m, b, p) for (m, b), p in rows.items()]


# Text encoder halves.

def text_state(rows) -> str:
    """The text adapter state of (LoadedAdapter, strength) rows, as it
    enters a conditioning key. Empty when no row changes the encoder."""
    live = [[ad.sha256, float(s)] for ad, s in rows if ad.text and s != 0.0]
    if not live:
        return ""
    return hashlib.sha256(json.dumps(live).encode()).hexdigest()


def _text_plans(pipe, enc, rows) -> list:
    """The text adapter plan of each (LoadedAdapter, strength) row, with
    every pair checked against its encoder layer. Nothing is installed."""
    from gmlx.load.adapter import LoraAdapter, LoraModule

    leaves = dict(enc.model.named_modules())
    plans = []
    for ad, strength in rows:
        mods = {}
        for path, key in ad.text:
            pair = pipe.adapters.get(key)
            if pair is None:
                raise AdapterError(f"{ad.name}: pair {key[:16]} is no "
                                   "longer held. Load the adapter again")
            dims = linear_dims(getattr(leaves.get(path), "base",
                                       leaves.get(path)))
            why = "the encoder has no such linear layer" if dims is None \
                else plan.fit(tuple(pair.up.shape), tuple(pair.down.shape),
                              None, dims)
            if why is not None:
                raise AdapterError(
                    f"{ad.name}: the text encoder pair on {path} does not "
                    f"fit the {enc.spec.name} encoder: {why}")
            rank = int(pair.down.shape[0])
            mods[path] = LoraModule(
                module_path=path, a=pair.down.astype(mx.float32),
                b=pair.up.astype(mx.float32), rank=rank,
                scale=strength * plan.pair_scale(pair.alpha, rank))
        plans.append((ad, LoraAdapter(alpha=0.0, arch=None, modules=mods,
                                      path=ad.path)))
    return plans


def _unwrap_text(enc) -> int:
    """Put the encoder's base layers back in place of every text adapter
    wrap and return the number of layers unwrapped."""
    from gmlx.load.modules import LoRAKQuantLinear

    from .install import _set

    wraps = [(p, m) for p, m in enc.model.named_modules()
             if isinstance(m, LoRAKQuantLinear)]
    for path, wrap in wraps:
        _set(enc.model, path, wrap.base)
    return len(wraps)


def apply_text_adapters(pipe, adapters) -> int:
    """Install the text encoder halves of (LoadedAdapter, strength) rows
    on the pipeline's encoder and return the number of adapters installed.
    A row at strength 0 and a row with no text half are skipped. The rest
    need an encoder the pipeline owns: on a borrowed chat model they are
    refused by name, since they would change its chat output. Every row is
    checked before any is installed, and a failure while installing takes
    the installed ones off again, so the encoder carries all of the rows
    or none."""
    rows = [(ad, float(s)) for ad, s in adapters if ad.text and s != 0.0]
    if not rows:
        return 0
    enc = pipe.encoder
    names = ", ".join(ad.name for ad, _ in rows)
    if enc is None or enc.released:
        raise AdapterError(f"{names}: the text encoder halves need a loaded "
                           "text encoder, and this pipeline has none")
    if not enc.owned:
        raise AdapterError(
            f"{names}: {sum(len(ad.text) for ad, _ in rows)} text encoder "
            "pairs are refused, because the text encoder is a resident "
            "chat model that an image adapter must not change. Pass a text "
            "strength of 0 to apply the image half alone")
    state = text_state(rows)
    if enc.adapters:
        if enc.adapters == state:
            return 0
        raise AdapterError(
            f"{names}: the text encoder already carries other adapters. "
            "Load a new pipeline to change them")
    from gmlx.load.modules import install_lora_adapter

    plans = _text_plans(pipe, enc, rows)
    try:
        for slot, (ad, text_plan) in enumerate(plans):
            try:
                install_lora_adapter(enc.model, text_plan, slot=slot)
            except (ValueError, NotImplementedError) as e:
                raise AdapterError(
                    f"{ad.name}: the text encoder half does not fit the "
                    f"{enc.spec.name} encoder: {e}") from None
    except BaseException:
        _unwrap_text(enc)
        raise
    enc.adapters = state
    return len(rows)
