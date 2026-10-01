"""What kind of GGUF a file is, from its header alone.

``classify`` reads ``general.type``, the arch string and the tensor names
and shapes, and returns one of ``KINDS``. Shared load code calls it only to
improve a refusal message, so it never raises on a readable header.
"""

from __future__ import annotations

KINDS = ("text", "image_dit", "vae", "mmproj", "adapter", "unknown")

IMAGE_HINT = ("This file is an image diffusion GGUF, which gmlx image "
              "loads.")
VAE_HINT = ("This file is an image VAE GGUF, which gmlx image loads beside "
            "a DiT.")


def _text_arch(arch: str | None) -> bool:
    if not arch:
        return False
    from gmlx.load.arch_table import ARCH_TABLE
    from gmlx.load.remap import ARCH_ALIAS

    return arch in ARCH_TABLE or arch in ARCH_ALIAS


def _image_dit(view) -> bool:
    from .fingerprint import strip_prefix
    from .registry import families

    for fam in families():
        try:
            v = strip_prefix(view, fam.strip_prefixes)
        except ValueError:
            continue
        if fam.fingerprint(v) is not None:
            return True
    return False


def _vae(view) -> bool:
    from .vae import flux1_ae
    from .vae.reader import strip_names

    names = strip_names(view.tensors)
    if flux1_ae.is_original_layout(names):
        levels = flux1_ae.original_levels(names)
        names = {flux1_ae.original_name(n, levels): f
                 for n, f in names.items()}
    shapes = {n: tuple(reversed(view.tensors[f].shape))
              for n, f in names.items()}
    return flux1_ae.config_from_shapes(shapes) is not None


def classify_view(view) -> str:
    """The kind of a header view, names unstripped."""
    kind = str(view.kv.get("general.type") or "").lower()
    arch = view.arch
    if kind == "adapter":
        return "adapter"
    if kind == "mmproj" or arch == "clip":
        return "mmproj"
    if kind == "vae" or arch == "autoencoder" or _vae(view):
        return "vae"
    if _text_arch(arch):
        return "text"
    from .registry import is_image_arch

    if is_image_arch(arch) or _image_dit(view):
        return "image_dit"
    return "unknown"


def classify(path: str) -> str:
    """The kind of the GGUF at ``path``, from its first shard's header."""
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.preflight import find_split_shards

    from .fingerprint import view_from_scans

    return classify_view(view_from_scans([scan_gguf(
        find_split_shards(path)[0])]))


def arch_hint(arch: str | None) -> str:
    """The sentence a text-load refusal appends for an image arch string,
    or an empty string. Never raises."""
    try:
        if arch == "autoencoder":
            return VAE_HINT
        from .registry import is_image_arch

        return IMAGE_HINT if is_image_arch(arch) else ""
    except Exception:
        return ""


def file_hint(path: str) -> str:
    """The sentence a text-load refusal appends for an image GGUF at
    ``path``, or an empty string. Never raises."""
    try:
        kind = classify(path)
    except Exception:
        return ""
    return {"image_dit": IMAGE_HINT, "vae": VAE_HINT}.get(kind, "")
