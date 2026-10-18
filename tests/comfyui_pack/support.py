"""Helpers the pack's tests share: the stub server, ComfyUI on the path and
the pack loaded as a package.

``COMFYUI_DIR`` names the ComfyUI checkout. ``GMLX_PACK_DIR`` names the
pack folder to test, by default the source in the gmlx tree. ``GMLX_TEST_MPS=1`` leaves the
MPS device on; otherwise ComfyUI runs on the cpu and its MPS probe answers
False, so no Metal device opens.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import subprocess
import sys
import types
import urllib.request

TESTS = os.path.dirname(os.path.abspath(__file__))
# The pack's source in the gmlx tree, or the folder GMLX_PACK_DIR names,
# such as the link `gmlx comfyui install` made in a custom_nodes folder.
PACK = os.environ.get("GMLX_PACK_DIR") or os.path.join(os.path.dirname(os.path.dirname(TESTS)),
                                                       "gmlx", "comfyui_pack")
FIXTURES = os.path.join(TESTS, "fixtures")
PACK_NAME = "comfyui_gmlx"


class Stub:
    """A stub server in its own process."""

    def __init__(self, *args: str):
        self.proc = subprocess.Popen([sys.executable, os.path.join(TESTS, "stub_server.py"), *args],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        line = self.proc.stdout.readline()
        try:
            self.url = json.loads(line)["listening"]
        except (ValueError, KeyError):
            self.proc.kill()
            raise RuntimeError(f"the stub server did not start: {line!r}") from None

    def call(self, path: str, obj=None):
        data = None if obj is None else json.dumps(obj).encode()
        req = urllib.request.Request(self.url + path, data=data, method="POST" if data else "GET",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or b"{}")

    def log(self) -> list:
        return self.call("/stub/log")["log"]

    def events(self, kind: str) -> list:
        return [e for e in self.log() if e["kind"] == kind]

    def reset(self):
        self.call("/stub/log/reset", {})

    def forget(self):
        self.call("/stub/forget", {})

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def load_client():
    """client.py and the pack modules it imports, without ComfyUI: the pack
    folder as a package whose __init__ does not run."""
    name = "gmlx_pack_standalone"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [PACK]
        sys.modules[name] = pkg
    return importlib.import_module(f"{name}.client")


_COMFY = {}


def comfy_dir() -> str | None:
    d = os.environ.get("COMFYUI_DIR")
    return d if d and os.path.isfile(os.path.join(d, "main.py")) else None


def setup_comfy() -> str:
    """Put ComfyUI on the path with its device chosen, once per process.
    Returns the checkout."""
    if _COMFY:
        return _COMFY["dir"]
    d = comfy_dir()
    if d is None:
        raise RuntimeError("set COMFYUI_DIR to a ComfyUI checkout")
    mps = os.environ.get("GMLX_TEST_MPS") == "1"
    import torch
    if not mps:
        torch.backends.mps.is_available = functools.lru_cache()(lambda: False)
    sys.path.insert(0, d)
    os.chdir(d)
    import comfy.options
    comfy.options.enable_args_parsing()
    argv = sys.argv
    sys.argv = [os.path.join(d, "main.py")] + ([] if mps else ["--cpu"])
    try:
        import comfy.cli_args  # noqa: F401
        import comfy.model_management  # noqa: F401
    finally:
        sys.argv = argv
    _COMFY["dir"] = d
    return d


def load_pack():
    """The pack as the package ``comfyui_gmlx``, after ComfyUI is set up."""
    setup_comfy()
    if PACK_NAME in sys.modules:
        return sys.modules[PACK_NAME]
    spec = importlib.util.spec_from_file_location(PACK_NAME, os.path.join(PACK, "__init__.py"),
                                                  submodule_search_locations=[PACK])
    m = importlib.util.module_from_spec(spec)
    sys.modules[PACK_NAME] = m
    spec.loader.exec_module(m)
    return m


def write_zimage_lora(path: str, manifest: dict, layers=(0, 1), rank: int = 4, alpha: float = 2.0,
                      seed: int = 0) -> int:
    """A small Z-Image LoRA in the dotted diffusion_model form, which
    ComfyUI maps with to_q, to_k and to_v as bands of attention.qkv.
    Returns the number of target modules."""
    import torch
    from safetensors.torch import save_file
    shapes = {t["name"].split("diffusion_model.")[-1]: t["shape"] for t in manifest["tensors"]}
    g = torch.Generator().manual_seed(seed)
    sd = {}
    n = 0
    for b in [f"layers.{i}" for i in layers]:
        dim = shapes[f"{b}.attention.qkv.weight"][0] // 3
        din = shapes[f"{b}.attention.qkv.weight"][1]
        targets = [(f"{b}.attention.{q}", (dim, din)) for q in ("to_q", "to_k", "to_v")]
        targets.append((f"{b}.attention.to_out.0", tuple(shapes[f"{b}.attention.out.weight"])))
        targets += [(f"{b}.feed_forward.{w}", tuple(shapes[f"{b}.feed_forward.{w}.weight"]))
                    for w in ("w1", "w2", "w3")]
        for stem, (dout, dn) in targets:
            k = "diffusion_model." + stem
            sd[f"{k}.lora_down.weight"] = (torch.randn(rank, dn, generator=g) * 0.01).to(torch.bfloat16)
            sd[f"{k}.lora_up.weight"] = (torch.randn(dout, rank, generator=g) * 0.01).to(torch.bfloat16)
            sd[f"{k}.alpha"] = torch.tensor(alpha)
            n += 1
    save_file(sd, path)
    return n


def write_flux3_lora(path: str, manifest: dict, rank: int = 4, alpha: float = 2.0,
                     seed: int = 0) -> list:
    """A small FLUX.3 LoRA on native layer names, inside the blocks and out:
    the single blocks in the kohya ``lora_unet_`` form, the rest in the
    dotted ``diffusion_model`` form, both of which ComfyUI's generic key map
    reads. Returns the target layer names."""
    import torch
    from safetensors.torch import save_file
    shapes = {t["name"]: t["shape"] for t in manifest["tensors"]}
    stems = [f"single_blocks.0.{leaf}" for leaf in
             ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")]
    stems += ["content_mode_blocks.image.1.q_proj", "content_mode_blocks.image_cond.0.v_proj",
              "txt_mode_blocks.0.mlp_out", "txt_in", "time_in.in_layer",
              "early_stream_modulations.txt.lin", "final_layer.image.linear",
              "ref_time_in.in_layer"]
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for stem in stems:
        dout, dn = shapes[f"{stem}.weight"]
        k = ("lora_unet_" + stem.replace(".", "_")) if stem.startswith("single_blocks.") \
            else "diffusion_model." + stem
        sd[f"{k}.lora_down.weight"] = (torch.randn(rank, dn, generator=g) * 0.01).to(torch.bfloat16)
        sd[f"{k}.lora_up.weight"] = (torch.randn(dout, rank, generator=g) * 0.01).to(torch.bfloat16)
        sd[f"{k}.alpha"] = torch.tensor(alpha)
    save_file(sd, path)
    return stems


# The pairs of write_flux2_lora: the file's key, the GGUF layer ComfyUI's
# key map puts it on and the third of a fused q, k and v it is, if any.
FLUX2_PAIRS = (
    ("transformer.transformer_blocks.0.attn.to_q", "double_blocks.0.img_attn.qkv", 0),
    ("transformer.transformer_blocks.0.attn.to_k", "double_blocks.0.img_attn.qkv", 1),
    ("transformer.transformer_blocks.0.attn.to_v", "double_blocks.0.img_attn.qkv", 2),
    ("transformer.transformer_blocks.1.attn.add_k_proj", "double_blocks.1.txt_attn.qkv", 1),
    ("transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj", "single_blocks.0.linear1", None),
    ("transformer.norm_out.linear", "final_layer.adaLN_modulation.1", None),
    ("lora_unet_single_blocks_1_linear2", "single_blocks.1.linear2", None),
    ("diffusion_model.double_blocks.2.txt_mlp.2", "double_blocks.2.txt_mlp.2", None),
    ("diffusion_model.double_stream_modulation_img.lin", "double_stream_modulation_img.lin", None),
    ("diffusion_model.txt_in", "txt_in", None),
)


def write_flux2_lora(path: str, manifest: dict, rank: int = 4, alpha: float = 2.0,
                     seed: int = 0) -> list:
    """A small FLUX.2 LoRA in the three spellings ComfyUI's key map reads:
    diffusers names, which put q, k and v on thirds of a fused weight and
    norm_out.linear on the final modulation with its halves swapped, the
    kohya ``lora_unet_`` form and the dotted ``diffusion_model`` form.
    Returns the GGUF layer of each pair, in the order of ``FLUX2_PAIRS``."""
    import torch
    from safetensors.torch import save_file
    shapes = {t["name"]: t["shape"] for t in manifest["tensors"]}
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for key, layer, third in FLUX2_PAIRS:
        dout, dn = shapes[f"{layer}.weight"]
        if third is not None:
            dout //= 3
        sd[f"{key}.lora_down.weight"] = (torch.randn(rank, dn, generator=g) * 0.01).to(torch.bfloat16)
        sd[f"{key}.lora_up.weight"] = (torch.randn(dout, rank, generator=g) * 0.01).to(torch.bfloat16)
        sd[f"{key}.alpha"] = torch.tensor(alpha)
    save_file(sd, path)
    return [layer for _, layer, _ in FLUX2_PAIRS]


# The layers of write_moss_lora, inside the blocks and out: the fused
# feed-forward input and the text refiner's qkv among them.
MOSS_STEMS = ("layers.0.proj.q", "layers.0.proj.ffn_up_gate", "layers.41.proj.ffn_down",
              "text_embedder.blocks.1.qkv", "text_embedder.proj", "time_embedding.2",
              "shared_block_modulation", "video_head_modulation", "video_embedder",
              "final_linear_video")


def write_moss_lora(path: str, manifest: dict, rank: int = 4, alpha: float = 2.0,
                    seed: int = 0) -> list:
    """A small MOSS-Image LoRA on the GGUF's names, which are the training
    names: the blocks in the kohya ``lora_unet_`` form, the rest in the
    dotted ``diffusion_model`` form. Returns the target layer names."""
    import torch
    from safetensors.torch import save_file
    shapes = {t["name"]: t["shape"] for t in manifest["tensors"]}
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for stem in MOSS_STEMS:
        dout, dn = shapes[f"{stem}.weight"]
        k = ("lora_unet_" + stem.replace(".", "_")) if stem.startswith("layers.") \
            else "diffusion_model." + stem
        sd[f"{k}.lora_down.weight"] = (torch.randn(rank, dn, generator=g) * 0.01).to(torch.bfloat16)
        sd[f"{k}.lora_up.weight"] = (torch.randn(dout, rank, generator=g) * 0.01).to(torch.bfloat16)
        sd[f"{k}.alpha"] = torch.tensor(alpha)
    save_file(sd, path)
    return list(MOSS_STEMS)


def manifest(family: str) -> dict:
    with open(os.path.join(FIXTURES, f"manifest-{family}.json")) as f:
        return json.load(f)
