"""Patch table entries from LoRA files and from ComfyUI patches.

``load_adapter`` reads a LoRA file through the dialect reader, places its
pairs with the family key map, checks that each fits its module and keeps
the pairs in the pipeline's cache. ``table_entries`` turns loaded adapters
and their strengths into the entries ``define_patch_table`` takes, and
stacked adapters share an entry per module and band. A strength change is
a new list of entries over the same cached pairs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass

from . import plan
from .install import linear_dims, linear_leaves, register_pair
from .reader import AdapterError, read_lora, resolve


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
