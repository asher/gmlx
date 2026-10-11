"""The image models gmlx can pull and the image files a model folder holds.

``RECIPES`` names each pullable image model: its DiT repository and quants,
its text encoder (and vision tower), its VAE source and the licenses.
``DEFAULT_QUANTS`` gives the quants a recipe pulls on each memory class.
``classify`` and ``scan`` find the image files on disk by their weights:
DiTs by their fingerprint, VAEs by their family and source digest, and
encoders and vision towers by the identity digests of their spec.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass

# The memory classes a default quant is chosen for, in GB of RAM. A Mac
# takes the largest class at or under its RAM, and the smallest below it.
MEMORY_CLASSES = (16, 24, 32, 64)


@dataclass(frozen=True)
class Recipe:
    """One pullable image model."""
    name: str
    title: str
    family: str                       # registry family name
    variant: str                      # the variant its DiT identifies as
    dit_repo: str
    dit_file: str                     # with {quant}
    dit_quants: tuple
    encoder_repo: str
    encoder_file: str                 # with {quant}
    encoder_quants: tuple
    vae_repo: str
    vae_revision: str
    vae_file: str                     # the safetensors source in the repo
    vae_config: str                   # its config.json in the repo, or ""
                                      # for a file that has none
    vae_sha256: str                   # sha256 of the source file
    vae_pack: str                     # the packed GGUF's file name
    license: str
    license_url: str
    encoder_license: str
    mmproj_file: str | None = None    # the encoder's vision tower
    encoder_bytes: int = 0            # the default encoder's size, for a fit
    vae_bytes: int = 0                # the packed VAE's size, for a fit
    aliases: tuple = ()
    license_note: str = ""            # what the license allows, if limited

    def dit_name(self, quant: str) -> str:
        return self.dit_file.format(quant=quant)

    def encoder_name(self, quant: str) -> str:
        return self.encoder_file.format(quant=quant)


_K_QUANTS = ("Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")

# Krea 2's VAE is ComfyUI's file of the Qwen-Image autoencoder, which comes
# without a config.json.
_KREA2_VAE = dict(
    vae_repo="Comfy-Org/Krea-2",
    vae_revision="eb1eddd3983a54678545a9b2c178c5853b30f7be",
    vae_file="vae/qwen_image_vae.safetensors", vae_config="",
    vae_sha256="a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
    vae_pack="qwen-image-vae-BF16.gguf")
_KREA2_ENCODER = "unsloth/Qwen3-VL-4B-Instruct-GGUF"
_KREA2_LICENSE = ("https://cdn.jsdelivr.net/gh/krea-ai/krea-2@"
                  "db3984fbc6e13b34c0064990fc2d95ac64d00058/assets/"
                  "hf_samples/LICENSE.pdf")
_KREA2_NOTE = ("free commercial use only under $1M annual revenue, "
               "affiliates counted, and a deployer must filter content")

RECIPES = {r.name: r for r in (
    Recipe(
        name="z-image-turbo", title="Z-Image-Turbo", family="z_image",
        variant="turbo",
        dit_repo="unsloth/Z-Image-Turbo-GGUF",
        dit_file="z-image-turbo-{quant}.gguf", dit_quants=_K_QUANTS + ("BF16",),
        encoder_repo="unsloth/Qwen3-4B-GGUF", encoder_file="Qwen3-4B-{quant}.gguf",
        encoder_quants=("Q4_K_M", "Q8_0"),
        vae_repo="Tongyi-MAI/Z-Image-Turbo",
        vae_revision="f332072aa78be7aecdf3ee76d5c247082da564a6",
        vae_file="vae/diffusion_pytorch_model.safetensors",
        vae_config="vae/config.json",
        vae_sha256="f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",
        vae_pack="z-image-vae-BF16.gguf",
        license="apache-2.0",
        license_url="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        encoder_license="apache-2.0",
        encoder_bytes=4_280_405_792, vae_bytes=167_660_192),
    Recipe(
        name="qwen-image", title="Qwen-Image 2.1", family="qwen_image21",
        variant="base",
        dit_repo="unsloth/Qwen-Image-2.1-GGUF",
        dit_file="qwen-image-2.1-{quant}.gguf", dit_quants=_K_QUANTS + ("F16",),
        encoder_repo="unsloth/Qwen3-VL-8B-Instruct-GGUF",
        encoder_file="Qwen3-VL-8B-Instruct-{quant}.gguf",
        encoder_quants=("Q4_K_M", "Q8_0"), mmproj_file="mmproj-BF16.gguf",
        vae_repo="Qwen/Qwen-Image-2.1",
        vae_revision="790c92633540aa0cb11d9abf19eb46d861714758",
        vae_file="vae/diffusion_pytorch_model.safetensors",
        vae_config="vae/config.json",
        vae_sha256="a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417",
        vae_pack="qwen-image-2.1-vae-F32.gguf",
        license="qwen-research",
        license_url="https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
        license_note="non-commercial use only",
        encoder_license="apache-2.0",
        encoder_bytes=8_709_520_224 + 1_162_569_280, vae_bytes=1_350_985_376,
        aliases=("qwen-image-2.1",)),
    Recipe(
        name="qwen-image-turbo", title="Qwen-Image 2.1 Turbo",
        family="qwen_image21", variant="turbo",
        dit_repo="AtomicChat/Qwen-Image-2.1-Turbo-GGUF",
        dit_file="Qwen-Image-2.1-Turbo-{quant}.gguf",
        dit_quants=("AD-Q3_K", "AD-Q4_K", "AD-Q5_K", "AD-Q6_K", "Q8_0", "BF16"),
        encoder_repo="unsloth/Qwen3-VL-8B-Instruct-GGUF",
        encoder_file="Qwen3-VL-8B-Instruct-{quant}.gguf",
        encoder_quants=("Q4_K_M", "Q8_0"), mmproj_file="mmproj-BF16.gguf",
        vae_repo="Qwen/Qwen-Image-2.1",
        vae_revision="790c92633540aa0cb11d9abf19eb46d861714758",
        vae_file="vae/diffusion_pytorch_model.safetensors",
        vae_config="vae/config.json",
        vae_sha256="a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417",
        vae_pack="qwen-image-2.1-vae-F32.gguf",
        license="qwen-research",
        license_url="https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE",
        license_note="non-commercial use only",
        encoder_license="apache-2.0",
        encoder_bytes=8_709_520_224 + 1_162_569_280, vae_bytes=1_350_985_376,
        aliases=("qwen-image-2.1-turbo",)),
    Recipe(
        name="krea-2-turbo", title="Krea 2 Turbo", family="krea2",
        variant="turbo",
        dit_repo="Abiray/Krea-2-Turbo-GGUF",
        dit_file="Krea-2-Turbo-{quant}.gguf", dit_quants=_K_QUANTS,
        encoder_repo=_KREA2_ENCODER,
        encoder_file="Qwen3-VL-4B-Instruct-{quant}.gguf",
        encoder_quants=("Q4_K_M", "Q8_0"), **_KREA2_VAE,
        license="krea-2-community", license_url=_KREA2_LICENSE,
        license_note=_KREA2_NOTE, encoder_license="apache-2.0",
        encoder_bytes=4_280_406_816, vae_bytes=253_807_712),
)}

# The DiT and encoder quants a recipe pulls on each memory class.
DEFAULT_QUANTS = {
    "z-image-turbo": {16: ("Q4_K_M", "Q8_0"), 24: ("Q8_0", "Q8_0"),
                      32: ("BF16", "Q8_0"), 64: ("BF16", "Q8_0")},
    # An F16 DiT loads at twice its size under bf16 activations, so the
    # 32 GB class keeps Q8_0.
    "qwen-image": {16: ("Q4_K_M", "Q8_0"), 24: ("Q8_0", "Q8_0"),
                   32: ("Q8_0", "Q8_0"), 64: ("F16", "Q8_0")},
    "qwen-image-turbo": {16: ("AD-Q4_K", "Q8_0"), 24: ("Q8_0", "Q8_0"),
                         32: ("BF16", "Q8_0"), 64: ("BF16", "Q8_0")},
    # No Krea 2 DiT fits a 1024 pixel image in the 16 GB class: Q3_K_M,
    # the smallest, fits up to 512 pixels with the encoder released and
    # the staged load. Q8_0 needs the 32 GB class.
    "krea-2-turbo": {16: ("Q3_K_M", "Q8_0"), 24: ("Q6_K", "Q8_0"),
                     32: ("Q8_0", "Q8_0"), 64: ("Q8_0", "Q8_0")},
}


# The image size an entry pulled with the class default DiT gets in its
# defaults, where that DiT makes images smaller than the family's size.
DEFAULT_SIZES = {"krea-2-turbo": {16: "512x512"}}


def recipe(name: str) -> Recipe | None:
    """The recipe of ``name`` or one of its aliases, or None."""
    key = str(name).strip().lower()
    for r in RECIPES.values():
        if key == r.name or key in r.aliases:
            return r
    return None


def recipe_for(family: str, variant: str) -> Recipe | None:
    """The recipe that pulls a DiT of this family and variant, or None."""
    for r in RECIPES.values():
        if r.family == family and r.variant == variant:
            return r
    for r in RECIPES.values():
        if r.family == family:
            return r
    return None


def memory_class(ram_bytes: int | None) -> int:
    """The memory class of a Mac with ``ram_bytes`` of RAM."""
    if not ram_bytes:
        return MEMORY_CLASSES[0]
    gb = ram_bytes / (1 << 30)
    fit = [c for c in MEMORY_CLASSES if c <= gb + 0.5]
    return fit[-1] if fit else MEMORY_CLASSES[0]


def default_quants(r: Recipe, ram_bytes: int | None) -> tuple[str, str]:
    """(DiT quant, encoder quant) the recipe pulls on this Mac."""
    return DEFAULT_QUANTS[r.name][memory_class(ram_bytes)]


def default_size(r: Recipe, ram_bytes: int | None,
                 dit_quant: str) -> str | None:
    """The ``defaults.size`` a pull of ``r`` at ``dit_quant`` writes on
    this Mac: the class's size when the quant is the class default."""
    cls = memory_class(ram_bytes)
    if dit_quant != DEFAULT_QUANTS[r.name][cls][0]:
        return None
    return DEFAULT_SIZES.get(r.name, {}).get(cls)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()




# What the model folders hold. Each check reads a GGUF header, and an
# encoder or vision tower check also reads the few small norm tensors its
# identity digests cover, so a file is matched by its weights and never by
# its name.

def _view(path: str):
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.preflight import find_split_shards

    from .fingerprint import view_from_scans

    return view_from_scans(scan_gguf(s) for s in find_split_shards(path))


def _kv(path: str) -> dict:
    from gmlx.load.headerscan import scan_gguf

    return dict(scan_gguf(path, include_tensors=False).kv)


def _spec(family: str):
    from .fingerprint import ImageModelError
    from .text_encoder import encoder_spec

    try:
        return encoder_spec(family)
    except ImageModelError:
        return None


def wants_tower(family: str) -> bool:
    """Whether the family's encoder comes with a vision tower."""
    spec = _spec(family)
    return bool(spec is not None and spec.vision_identity)


def dit_identity(path: str) -> tuple[str, str] | None:
    """(family, variant) of an image DiT GGUF, or None for any other
    file."""
    from .fingerprint import ImageModelError, identify

    try:
        ident = identify(_view(path))
    except (ImageModelError, OSError, ValueError):
        return None
    return ident.family.name, ident.variant


def vae_identity(path: str) -> tuple[str, str | None] | None:
    """(family, source sha256) of a packed VAE GGUF, or None."""
    kv = _kv(path)
    if kv.get("general.type") != "vae" \
            and kv.get("general.architecture") != "autoencoder":
        return None
    fam = kv.get("gmlx.vae.family")
    if not fam:
        return None
    sha = kv.get("gmlx.vae.source_sha256")
    return str(fam), (str(sha) if sha else None)


def _shape_matches(kv: dict, spec) -> bool:
    arch = kv.get("general.architecture")
    return (arch == spec.arch
            and kv.get(f"{arch}.embedding_length") == spec.hidden
            and kv.get(f"{arch}.block_count") == spec.layers)


def encoder_matches(path: str, family: str, kv: dict | None = None) -> bool:
    """Whether the text GGUF at ``path`` holds the weights the family
    conditions on: its arch and shape, then the identity digests."""
    from .fingerprint import ImageModelError
    from .text_encoder import check_encoder_file

    spec = _spec(family)
    if spec is None or not _shape_matches(kv or _kv(path), spec):
        return False
    try:
        check_encoder_file(_view(path), spec)
    except (ImageModelError, OSError, ValueError):
        return False
    return True


def mmproj_matches(path: str, family: str) -> bool:
    """Whether the GGUF at ``path`` is the vision tower of the family's
    encoder."""
    from .fingerprint import ImageModelError
    from .text_encoder import check_mmproj_file

    spec = _spec(family)
    if spec is None or not spec.vision_identity:
        return False
    try:
        check_mmproj_file(_view(path), spec)
    except (ImageModelError, OSError, ValueError):
        return False
    return True


def image_arches() -> frozenset:
    """The arch strings that an image DiT gmlx loads can carry, those that
    other models share too included, so a file with one is worth a look at
    its tensors."""
    from .registry import families

    out: set = set()
    for fam in families():
        out |= {a for a in fam.arch_strings | fam.shared_arch_strings if a}
    return frozenset(out)


@dataclass(frozen=True)
class ImageFile:
    """One image file on disk. ``role`` is ``dit``, ``vae``, ``encoder``
    or ``mmproj``; ``family`` is the image family it serves; ``variant``
    is a DiT's variant and ``sha256`` a packed VAE's source digest."""
    path: str
    role: str
    family: str
    variant: str | None = None
    sha256: str | None = None


def classify(path: str, *, families: tuple | None = None) -> list:
    """The image roles of the GGUF at ``path``, one ImageFile per family
    it serves, or none. ``families`` limits the encoder and tower checks
    to those families."""
    from .registry import families as all_families

    names = families or tuple(f.name for f in all_families())
    kv = _kv(path)
    arch = kv.get("general.architecture")
    kind = str(kv.get("general.type") or "").lower()
    if kind == "vae" or arch == "autoencoder":
        got = vae_identity(path)
        return [ImageFile(path, "vae", got[0], sha256=got[1])] if got else []
    if kind == "mmproj" or arch == "clip":
        return [ImageFile(path, "mmproj", f) for f in names
                if mmproj_matches(path, f)]
    if kind == "adapter":
        return []
    if arch is None or arch in image_arches():
        got = dit_identity(path)
        return [ImageFile(path, "dit", got[0], got[1])] if got else []
    return [ImageFile(path, "encoder", f) for f in names
            if encoder_matches(path, f, kv)]


def scan(dirs, *, recursive: bool = True,
         families: tuple | None = None) -> list:
    """The image files under ``dirs``, first shards only. A file that
    cannot be read is skipped."""
    from gmlx.load.discovery import _iter_gguf_files

    out: list = []
    seen: set = set()
    for d in dirs or ():
        root = os.path.abspath(os.path.expanduser(os.path.expandvars(d)))
        if not os.path.isdir(root):
            continue
        for p in _iter_gguf_files(root, recursive):
            real = os.path.realpath(p)
            if real in seen:
                continue
            seen.add(real)
            try:
                out += classify(p, families=families)
            except Exception as e:  # noqa: BLE001 - a bad file is skipped
                print(f"[image] skipping {p}: {e}", file=sys.stderr)
    return out


def nearest(path: str, files: list) -> list:
    """``files`` with those in the folder of ``path`` first, then by
    path."""
    d = os.path.dirname(os.path.abspath(path))
    return sorted(files, key=lambda f: (
        os.path.dirname(os.path.abspath(f.path)) != d, f.path))
