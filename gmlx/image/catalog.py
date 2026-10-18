"""The image models gmlx can pull and the image files a model folder holds.

``RECIPES`` names each pullable image model: its DiT repository and quants,
its text encoder (and vision tower), its VAE source, the upsampler's
system prompt of a family that ships none, and the licenses.
``DEFAULT_QUANTS`` gives the quants a recipe pulls on each memory class,
and ``NO_IMAGE`` the classes where none of them makes an image.
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
    vae_license: str = ""             # the VAE's license, when it has its
                                      # own
    # The upsampler's system prompt of a family that ships none, which the
    # pull fetches when it can and the entry names as upsample_prompt.
    prompt_repo: str = ""
    prompt_revision: str = ""
    prompt_file: str = ""
    prompt_sha256: str = ""
    prompt_license: str = ""
    prompt_license_url: str = ""

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

# The FLUX.2 autoencoder, which FLUX.2 klein and FLUX.3 klein decode with:
# BFL's file in the FLUX.2 klein 4B repository, BF16 in the diffusers
# layout with its config.json, packed in its own dtype. One pack serves the
# six recipes.
_FLUX2_VAE = dict(
    vae_repo="black-forest-labs/FLUX.2-klein-4B",
    vae_revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
    vae_file="vae/diffusion_pytorch_model.safetensors",
    vae_config="vae/config.json",
    vae_sha256="ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",
    vae_pack="flux2-vae-BF16.gguf", vae_license="apache-2.0",
    vae_bytes=168_114_016)

# FLUX.2 klein 4B: unsloth's GGUFs of BFL's checkpoints, conditioned on
# Qwen3-4B, the encoder file of Z-Image.
_FLUX2 = dict(
    family="flux2_klein", dit_quants=("Q4_K_M", "Q8_0"),
    encoder_repo="unsloth/Qwen3-4B-GGUF", encoder_file="Qwen3-4B-{quant}.gguf",
    encoder_quants=("Q8_0",), **_FLUX2_VAE,
    license="apache-2.0", encoder_license="apache-2.0",
    encoder_bytes=4_280_405_792)

# FLUX.2 klein 9B: unsloth's GGUFs of BFL's checkpoints, conditioned on
# Qwen3-8B, under the FLUX Non-Commercial License. Its VAE is the 4B
# repository's file, which the 9B repositories hold too.
_FLUX2_9B = dict(
    family="flux2_klein_9b", dit_quants=("Q4_K_M", "Q8_0"),
    encoder_repo="unsloth/Qwen3-8B-GGUF", encoder_file="Qwen3-8B-{quant}.gguf",
    encoder_quants=("Q8_0",), **_FLUX2_VAE,
    license="flux-non-commercial-license",
    # LICENSE.md 2.b and 2.e.
    license_note=("non-commercial use only, with content filters or a review "
                  "of each image for unlawful or infringing content before it "
                  "is shared, and a disclosure that it is made with AI where "
                  "the law requires one"),
    encoder_license="apache-2.0", encoder_bytes=8_709_519_168)

# Both FLUX.3 klein checkpoints share the encoder file of Krea 2.
_FLUX3 = dict(
    family="flux3_klein", dit_repo="realrebelai/FLUX-3-Image-Klein_GGUFs",
    dit_quants=("Q4_K_M", "Q8_0"),
    encoder_repo=_KREA2_ENCODER,
    encoder_file="Qwen3-VL-4B-Instruct-{quant}.gguf", encoder_quants=("Q8_0",),
    **_FLUX2_VAE,
    license="flux-kommunity",
    license_url="https://huggingface.co/black-forest-labs/flux-3-image-klein/"
                "blob/main/LICENSE.md",
    # LICENSE.md 2.a, 2.e and the Qualifying User of 1.e.
    license_note=("non-commercial use, and commercial use of the images only "
                  "under $5M annual revenue, affiliates counted, with content "
                  "filters or a review of each image for unlawful or "
                  "infringing content before it is shared, and a disclosure "
                  "that it is made with AI where the law requires one"),
    encoder_license="apache-2.0", encoder_bytes=4_280_406_816)

# Qwen's text to image rewriter prompt, which gmlx does not ship: a Qwen
# Research License file of 10045 bytes in the repository of its rewriter
# model.
_QWEN_PROMPT = dict(
    prompt_repo="Qwen/Qwen-Image-2.1-PE-T2I",
    prompt_revision="f3ed7985c788ad75b3ab7223e0c4c51e2a43545b",
    prompt_file="system_prompt.txt",
    prompt_sha256="a77c9a06c59b120741141d9514b95682bb8761d02bec49ca61def7b2b3d9fb99",
    prompt_license="the Qwen Research License, non-commercial use only",
    prompt_license_url="https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I/"
                       "blob/f3ed7985c788ad75b3ab7223e0c4c51e2a43545b/LICENSE")

# MOSS-Image: realrebelai's GGUFs of OpenMOSS's SFT and Base checkpoints,
# conditioned on Qwen3.5-9B. Its VAE is the release's file, float32 in the
# diffusers layout with its config.json, in the SFT repository.
_MOSS = dict(
    family="moss_image", dit_repo="realrebelai/MOSS-Image_GGUFs",
    dit_quants=_K_QUANTS, encoder_repo="unsloth/Qwen3.5-9B-GGUF",
    encoder_file="Qwen3.5-9B-{quant}.gguf", encoder_quants=("Q4_K_M", "Q8_0"),
    vae_repo="OpenMOSS-Team/MOSS-Image-SFT",
    vae_revision="fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
    vae_file="vae/diffusion_pytorch_model.safetensors",
    vae_config="vae/config.json",
    vae_sha256="c3bdf94dc67ec69b30ea9fe1fa8340085679cae64d019f7c7e4a9921f1a7f17a",
    vae_pack="moss-image-vae-F32.gguf",
    license="apache-2.0", encoder_license="apache-2.0",
    encoder_bytes=9_527_502_048, vae_bytes=249_514_688)

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
        aliases=("qwen-image-2.1",), **_QWEN_PROMPT),
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
        aliases=("qwen-image-2.1-turbo",), **_QWEN_PROMPT),
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
    Recipe(
        name="moss-image", title="MOSS-Image", variant="sft",
        dit_file="MOSS-Image-SFT-{quant}.gguf",
        license_url="https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/"
                    "blob/main/LICENSE",
        aliases=("moss-image-sft",), **_MOSS),
    Recipe(
        name="moss-image-base", title="MOSS-Image Base", variant="base",
        dit_file="MOSS-Image-Base-{quant}.gguf",
        license_url="https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/"
                    "blob/main/LICENSE", **_MOSS),
    Recipe(
        name="flux-2-klein", title="FLUX.2 klein", variant="distilled",
        dit_repo="unsloth/FLUX.2-klein-4B-GGUF",
        dit_file="flux-2-klein-4b-{quant}.gguf",
        license_url="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/"
                    "blob/main/LICENSE.md", **_FLUX2),
    Recipe(
        name="flux-2-klein-base", title="FLUX.2 klein Base", variant="base",
        dit_repo="unsloth/FLUX.2-klein-base-4B-GGUF",
        dit_file="flux-2-klein-base-4b-{quant}.gguf",
        license_url="https://huggingface.co/black-forest-labs/"
                    "FLUX.2-klein-base-4B/blob/main/LICENSE.md", **_FLUX2),
    Recipe(
        name="flux-2-klein-9b", title="FLUX.2 klein 9B", variant="distilled",
        dit_repo="unsloth/FLUX.2-klein-9B-GGUF",
        dit_file="flux-2-klein-9b-{quant}.gguf",
        license_url="https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/"
                    "blob/main/LICENSE.md", **_FLUX2_9B),
    Recipe(
        name="flux-2-klein-base-9b", title="FLUX.2 klein Base 9B",
        variant="base", dit_repo="unsloth/FLUX.2-klein-base-9B-GGUF",
        dit_file="flux-2-klein-base-9b-{quant}.gguf",
        license_url="https://huggingface.co/black-forest-labs/"
                    "FLUX.2-klein-base-9B/blob/main/LICENSE.md", **_FLUX2_9B),
    # QuantStack's GGUFs of BFL's KV checkpoint, whose LICENSE is the text
    # of the 9B's LICENSE.md.
    Recipe(
        name="flux-2-klein-9b-kv", title="FLUX.2 klein 9B KV",
        variant="distilled", dit_repo="QuantStack/FLUX.2-Klein-9B-KV-GGUF",
        dit_file="Flux-2-Klein-9B-KV-{quant}.gguf",
        license_url="https://huggingface.co/black-forest-labs/"
                    "FLUX.2-klein-9b-kv/blob/main/LICENSE",
        **dict(_FLUX2_9B, family="flux2_klein_9b_kv")),
    Recipe(
        name="flux-3-klein", title="FLUX.3 klein", variant="distilled",
        dit_file="DISTILLED/flux3_image_klein_sd-{quant}.gguf", **_FLUX3),
    Recipe(
        name="flux-3-klein-base", title="FLUX.3 klein Base", variant="base",
        dit_file="BASE/flux3_image_klein-{quant}.gguf", **_FLUX3),
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
    # On 16 GB a Q4_K_M FLUX.2 klein DiT makes a 1024 pixel image from up
    # to two 1024 pixel references with the encoder released, and Base runs
    # its two guided rows one at a time. A Q8_0 DiT fits there only with
    # the staged load and a tiled decode, and takes no reference.
    "flux-2-klein": {16: ("Q4_K_M", "Q8_0"), 24: ("Q8_0", "Q8_0"),
                     32: ("Q8_0", "Q8_0"), 64: ("Q8_0", "Q8_0")},
    "flux-2-klein-base": {16: ("Q4_K_M", "Q8_0"), 24: ("Q8_0", "Q8_0"),
                          32: ("Q8_0", "Q8_0"), 64: ("Q8_0", "Q8_0")},
    # No FLUX.2 klein 9B DiT these recipes pull makes an image in the 16 GB
    # class: a Q4_K_M step at 256 pixels needs 8.91 GiB of its 8.55. The
    # 16 GB row names the smallest, which a pull takes there only when the
    # quant is given (NO_IMAGE), and the memory plan of a request there
    # refuses it. On 24 GB a Q4_K_M DiT makes a 1024 pixel image from up to
    # four 1024 pixel references with the encoder released, and Base runs
    # its two guided rows one at a time. A Q8_0 DiT fits there only with
    # the staged load and a tiled decode, and from one reference, so it
    # takes the 32 GB class, with the encoder released.
    "flux-2-klein-9b": {16: ("Q4_K_M", "Q8_0"), 24: ("Q4_K_M", "Q8_0"),
                        32: ("Q8_0", "Q8_0"), 64: ("Q8_0", "Q8_0")},
    "flux-2-klein-base-9b": {16: ("Q4_K_M", "Q8_0"), 24: ("Q4_K_M", "Q8_0"),
                             32: ("Q8_0", "Q8_0"), 64: ("Q8_0", "Q8_0")},
    # KV keeps the keys and values of its references, so an edit takes more
    # memory than on the 9B. As on the 9B, no DiT makes an image in the
    # 16 GB class: a Q4_K_M request at 512 pixels needs 9.01 GiB of 8.55. On
    # 24 GB a Q4_K_M DiT makes a 1024 pixel image from up to two 1024 pixel
    # references with the encoder released, and a Q8_0 DiT makes no 1024
    # pixel edit. On 32 GB a Q8_0 DiT takes up to two of them, and a Q4_K_M
    # DiT four.
    "flux-2-klein-9b-kv": {16: ("Q4_K_M", "Q8_0"), 24: ("Q4_K_M", "Q8_0"),
                           32: ("Q8_0", "Q8_0"), 64: ("Q8_0", "Q8_0")},
    # A MOSS-Image request at 1024 pixels with guidance on the longest
    # prompt: on 16 GB the Q8_0 encoder alone needs 9.49 GiB of 8.55, so the
    # class takes the Q4_K_M encoder, and a Q6_K DiT needs 8.10 GiB with the
    # staged load, the encoder released and a tiled decode. On 24 GB a Q8_0
    # DiT with the Q8_0 encoder needs 14.92 GiB of 15.2 with the encoder
    # released.
    "moss-image": {16: ("Q6_K", "Q4_K_M"), 24: ("Q8_0", "Q8_0"),
                   32: ("Q8_0", "Q8_0"), 64: ("Q8_0", "Q8_0")},
    "moss-image-base": {16: ("Q6_K", "Q4_K_M"), 24: ("Q8_0", "Q8_0"),
                        32: ("Q8_0", "Q8_0"), 64: ("Q8_0", "Q8_0")},
    # A Q8_0 DiT fits a 1024 pixel image from the 24 GB class up, with the
    # encoder released on 24 GB, where Base also runs its two guided rows
    # one at a time.
    "flux-3-klein": {16: ("Q4_K_M", "Q8_0"), 24: ("Q8_0", "Q8_0"),
                     32: ("Q8_0", "Q8_0"), 64: ("Q8_0", "Q8_0")},
    "flux-3-klein-base": {16: ("Q4_K_M", "Q8_0"), 24: ("Q8_0", "Q8_0"),
                          32: ("Q8_0", "Q8_0"), 64: ("Q8_0", "Q8_0")},
}


# The memory classes where no DiT a recipe pulls makes an image. A pull
# there refuses unless the quant is given, and names the least class that
# makes one.
NO_IMAGE = {"flux-2-klein-9b": frozenset({16}),
            "flux-2-klein-base-9b": frozenset({16}),
            "flux-2-klein-9b-kv": frozenset({16})}


# The output matrix of each encoder file a recipe pulls that has one, from
# its header. The load reads and drops it, so a pull counts it as the
# memory plan of the file on disk does before the file is there. The 4B
# encoders tie it to the embedding and have none.
ENCODER_OUTPUT_BYTES = {
    "Qwen3-8B-Q8_0.gguf": 661_225_472,
    "Qwen3-VL-8B-Instruct-Q8_0.gguf": 661_225_472,
    "Qwen3-VL-8B-Instruct-Q4_K_M.gguf": 510_504_960,
    "Qwen3.5-9B-Q8_0.gguf": 1_080_688_640,
    "Qwen3.5-9B-Q4_K_M.gguf": 834_355_200,
}


# Encoder files a class default pulls other than the one encoder_bytes
# sizes, by name.
ENCODER_BYTES = {
    "Qwen3.5-9B-Q4_K_M.gguf": 5_680_522_464,
    "Qwen3.5-9B-Q8_0.gguf": 9_527_502_048,
}


def companion_bytes(r: Recipe, ram_bytes: int | None) -> int:
    """Bytes of the encoder (with its vision tower) and the VAE that a
    pull of ``r`` takes on this Mac, at the class's encoder quant."""
    name = r.encoder_name(default_quants(r, ram_bytes)[1])
    return ENCODER_BYTES.get(name, r.encoder_bytes) + r.vae_bytes


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


def makes_image(r: Recipe, cls: int) -> bool:
    """Whether a DiT the recipe pulls makes an image in memory class
    ``cls``."""
    return cls not in NO_IMAGE.get(r.name, ())


def least_class(r: Recipe) -> int:
    """The smallest memory class in which a DiT the recipe pulls makes an
    image."""
    return min(c for c in MEMORY_CLASSES if makes_image(r, c))


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


def vae_families(label: str) -> tuple:
    """The families a VAE packed for the family ``label`` serves:
    ``label`` first, then each family that decodes with the same
    autoencoder, as FLUX.2 klein and FLUX.3 klein do."""
    from .vae.reader import VAES

    row = VAES.get(label)
    return (label, *(f for f, r in VAES.items()
                     if row is not None and r is row and f != label))


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
        return [ImageFile(path, "vae", f, sha256=got[1])
                for f in vae_families(got[0])] if got else []
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
