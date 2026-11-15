"""MiniMax-H3-Image identify and load by the family's own rule, for the
checks to run before the family is discovered: ``identify`` gives the
image loader's ``load_dit`` an Identified from the header alone, as the
registry's identify does for a discovered family."""

from __future__ import annotations

from gmlx.image import fingerprint as fp
from gmlx.image import loader as image_loader

from .family import FAMILY, TITLE


def identify(path: str) -> fp.Identified:
    """The family, config and variant of an H3-Image GGUF from its header.
    Raises ``ImageModelError`` for another file or a refused one."""
    from gmlx.load.preflight import find_split_shards

    view = fp.view_from_scans([image_loader._scan_shard(s)
                               for s in find_split_shards(path)])
    v = fp.strip_prefix(view, FAMILY.strip_prefixes)
    res = FAMILY.fingerprint(v)
    if res is None:
        raise fp.ImageModelError(f"not a {TITLE} file. "
                                 f"{fp.describe_unknown(v)}")
    if isinstance(res, fp.Refusal):
        raise fp.ImageModelError(res.reason)
    if view.arch not in FAMILY.arch_strings | FAMILY.shared_arch_strings:
        raise fp.ImageModelError(
            f"arch {view.arch!r} contradicts the tensor layout, which is "
            f"{TITLE}")
    return fp.Identified(
        family=FAMILY, config=FAMILY.synth_config(v), prefix=v.prefix,
        layout=res.layout, **fp._variant(FAMILY, v, None, res.variant),
        view=v)


def load(path: str, **kw):
    """The DiT and its DitInfo through ``gmlx.image.loader.load_dit``."""
    return image_loader.load_dit(path, identified=identify(path), **kw)
