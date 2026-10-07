#!/usr/bin/env python3
"""Check 2 oracle for Qwen-Image 2.1 with reference images: the encoder
tap over the edit template.

Each case is one call of the pinned QwenImage21Pipeline with a prompt and
its reference images. The call itself makes each image RGBA, resizes it
with calculate_dimensions and its Lanczos resize, composites the alpha
over white for the encoder, renders the template with one image slot per
reference and runs Qwen3-VL-8B with its vision tower. The script stops the
call after encode_prompt, so it needs no transformer and no VAE, and reads
what the call made: the resized RGBA image the VAE would see, the RGB
image and the patches the vision tower saw, the token ids, the image rows
and the features. The features are checked equal, bit for bit, to the
output of decoder layer 35 read by a forward hook, less the rows of the
system turn.

The language model is set by name from the text GGUF as
check2_qwen_image21.py sets it. The vision tower is set by name from the
mmproj GGUF, whose tensors are bfloat16 and float32, by the name map of
this script. As a check of that map the script asks the model for the
colour of a red and of a blue square and reads its next token.

The image processor is the PIL backend of the official processor config.
The config names the torchvision backend, which the oracle environment
lacks. The two differ in how they resize. No image is resized here: the
pipeline hands the processor sizes that are multiples of 32 within its
pixel range, and the script checks the grid of each.

The source images are drawn by this script from a seed. One of them has
an alpha channel with a transparent region.

The bf16 run of the gate has the language model in bfloat16 and the
vision tower in float32. A second bf16 run has the tower in bfloat16 too,
as the pipeline loads it. That run sets no gate: the tower's residual
stream reaches the thousands, a 16-bit float drops the small values
beside them, and the image rows of the features move by about 30 percent
of their norm, some to a cosine under 0.4. The manifest records it.

For each case the fixture holds the token ids, the rows of the features
that are image slots, the grid of each reference, and the fp32 and bf16
taps. For each image it holds the source, the resized RGBA image and the
patches the tower saw. The tolerances follow check2_qwen_image21.py:
relative L2 at twice the oracle's own bf16 against fp32 gap of the case,
and a per-token cosine floor of 1 - 4 (1 - c), with the same relative L2
rule over the image rows and over the text rows alone. The manifest also
records the relative L2 between the taps of the cases that differ in one
thing: the order of two references.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check2_qwen_image21_edit.py \\
      --encoder Qwen3-VL-8B-Instruct-Q4_K_M.gguf --mmproj mmproj-BF16.gguf \\
      --config <official repo dir>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import check2_qwen_image21 as c2
import fixtures
import oracle
from check2_z_image import token_cosine

NAME = "check2_qwen_image21_edit"
RESOLUTION = 512        # output_resolution of every case
SEED = 21
RULE = (c2.RULE + "; the bf16 run of the gate has the language model in "
        "bfloat16 and the vision tower in float32; the same relative L2 "
        "rule over the image rows and over the text rows alone")

# name -> (width, height, alpha) of the source image.
IMAGES = {"a": (768, 768, False), "b": (640, 480, False),
          "c": (480, 640, True)}
NEGATIVE = "blurry, low quality, distorted, watermark"
CASES = {
    "one": ("Paint the largest rectangle bright red and leave the rest of "
            "the picture as it is.", ("a",)),
    "one_neg": (NEGATIVE, ("a",)),
    "two": ("Place the shapes of the first picture over the background of "
            "the second picture.", ("b", "c")),
    "two_neg": (NEGATIVE, ("b", "c")),
    "two_swapped": ("Place the shapes of the first picture over the "
                    "background of the second picture.", ("c", "b")),
}

# mmproj tensor name to the name under model.visual.
VISION = {"v.patch_embd.bias": "patch_embed.proj.bias",
          "v.position_embd.weight": "pos_embed.weight",
          "v.post_ln.weight": "merger.norm.weight",
          "v.post_ln.bias": "merger.norm.bias",
          "mm.0.weight": "merger.linear_fc1.weight",
          "mm.0.bias": "merger.linear_fc1.bias",
          "mm.2.weight": "merger.linear_fc2.weight",
          "mm.2.bias": "merger.linear_fc2.bias"}
BLOCK = {"attn_qkv": "attn.qkv", "attn_out": "attn.proj",
         "ffn_up": "mlp.linear_fc1", "ffn_down": "mlp.linear_fc2",
         "ln1": "norm1", "ln2": "norm2"}
MERGER = {"norm": "norm", "fc1": "linear_fc1", "fc2": "linear_fc2"}
PATCH = ("v.patch_embd.weight", "v.patch_embd.weight.1")


def source_image(name: str) -> np.ndarray:
    """The source image ``name``: (H, W, 3) or (H, W, 4) uint8. A smooth
    gradient with filled rectangles and discs over it, and for an image
    with alpha a transparent band and a half transparent disc."""
    w, h, alpha = IMAGES[name]
    rng = np.random.default_rng([SEED, sorted(IMAGES).index(name)])
    yy, xx = np.mgrid[0:h, 0:w]
    base = rng.integers(0, 256, 3)
    img = np.stack([(xx * 255 // w + base[0]) % 256,
                    (yy * 255 // h + base[1]) % 256,
                    ((xx + yy) * 255 // (w + h) + base[2]) % 256], -1)
    for _ in range(9):
        x0, y0 = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        dx, dy = int(rng.integers(40, w // 3)), int(rng.integers(40, h // 3))
        img[y0:y0 + dy, x0:x0 + dx] = rng.integers(0, 256, 3)
    for _ in range(5):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(20, min(w, h) // 5))
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = rng.integers(0, 256, 3)
    img = img.astype(np.uint8)
    if not alpha:
        return img
    a = np.full((h, w), 255, np.uint8)
    a[h // 8:h // 4] = 0
    a[(xx - w // 2) ** 2 + (yy - 2 * h // 3) ** 2 < (w // 4) ** 2] = 128
    return np.concatenate([img, a[..., None]], -1)


def bf16_to_f32(data: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(data).view(np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def vision_tensor(torch, tensor):
    """A tensor of the mmproj as a float32 torch tensor in torch's axis
    order."""
    kind = tensor.tensor_type.name
    shape = tuple(int(d) for d in reversed(tensor.shape))
    if kind == "BF16":
        a = bf16_to_f32(np.asarray(tensor.data))
    elif kind == "F32":
        a = np.asarray(tensor.data, np.float32)
    else:
        raise SystemExit(f"error: {tensor.name} is {kind}; the vision "
                         "tower is read from BF16 and F32 tensors")
    return torch.from_numpy(np.ascontiguousarray(a.reshape(shape)))


def vision_name(name: str, deep: list) -> str | None:
    if name in VISION:
        return VISION[name]
    part = name.split(".")
    if len(part) == 5 and part[:2] == ["v", "blk"] and part[3] in BLOCK:
        return f"blocks.{int(part[2])}.{BLOCK[part[3]]}.{part[4]}"
    if (len(part) == 5 and part[:2] == ["v", "deepstack"]
            and part[3] in MERGER and int(part[2]) in deep):
        return (f"deepstack_merger_list.{deep.index(int(part[2]))}."
                f"{MERGER[part[3]]}.{part[4]}")
    return None


def set_vision(torch, te, mmproj: Path) -> dict:
    """Set model.visual from the mmproj GGUF by name, in float32. Returns
    the codec census."""
    import gguf
    from accelerate.utils import set_module_tensor_to_device

    visual = te.model.visual
    deep = list(te.config.vision_config.deepstack_visual_indexes)
    want = {n: tuple(p.shape) for n, p in visual.named_parameters()}
    reader = gguf.GGUFReader(str(mmproj))
    census: dict[str, int] = {}
    seen, left, patch = set(), [], {}
    for t in reader.tensors:
        census[t.tensor_type.name] = census.get(t.tensor_type.name, 0) + 1
        if t.name in PATCH:
            patch[t.name] = vision_tensor(torch, t)
            continue
        name = vision_name(t.name, deep)
        if name is None:
            left.append(t.name)
            continue
        value = vision_tensor(torch, t)
        if tuple(value.shape) != want.get(name):
            raise SystemExit(f"error: {t.name} has shape "
                             f"{tuple(value.shape)}, and {name} wants "
                             f"{want.get(name)}")
        set_module_tensor_to_device(visual, name, "cpu", value=value,
                                    dtype=torch.float32)
        seen.add(name)
    if sorted(patch) != sorted(PATCH):
        raise SystemExit(f"error: patch embedding tensors {sorted(patch)}")
    # The two frames of the patch embedding are the two tensors, in order.
    proj = torch.stack([patch[n] for n in PATCH], dim=2)
    if tuple(proj.shape) != want["patch_embed.proj.weight"]:
        raise SystemExit(f"error: patch embedding {tuple(proj.shape)}")
    set_module_tensor_to_device(visual, "patch_embed.proj.weight", "cpu",
                                value=proj, dtype=torch.float32)
    seen.add("patch_embed.proj.weight")
    if left or seen != set(want):
        raise SystemExit(f"error: mmproj tensors left over {left}, tower "
                         f"parameters not set {sorted(set(want) - seen)}")
    for b in visual.buffers():
        if b.is_floating_point() and b.dtype != torch.float32:
            b.data = b.data.float()
    return census


def image_processor(config: Path):
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil)

    return Qwen2VLImageProcessorPil.from_pretrained(
        str(config / "processor"), local_files_only=True)


def colour_check(torch, te, reader, proc) -> dict:
    """The model's greedy next token for the colour of a red and of a blue
    square, through the tower, the final norm and the output matrix."""
    import gguf
    from PIL import Image

    head = next(t for t in reader.tensors if t.name == c2.HEAD)
    text = proc.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What color is this picture? Answer "
                                     "in one word."}]}],
        tokenize=False, add_generation_prompt=True)
    out = {}
    for name, rgb in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
        inputs = proc(text=[text], images=[Image.new("RGB", (256, 256), rgb)],
                      return_tensors="pt")
        last = te.model(input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        pixel_values=inputs.pixel_values,
                        image_grid_thw=inputs.image_grid_thw,
                        mm_token_type_ids=inputs.mm_token_type_ids
                        ).last_hidden_state[0, -1]
        best, at = None, -1
        for lo in range(0, int(head.shape[1]), c2.ROWS):
            w = torch.from_numpy(np.ascontiguousarray(gguf.quants.dequantize(
                head.data[lo:lo + c2.ROWS], head.tensor_type),
                dtype=np.float32))
            logits = w @ last
            top = int(logits.argmax())
            if best is None or float(logits[top]) > best:
                best, at = float(logits[top]), lo + top
        out[name] = {"token_id": at, "token": proc.tokenizer.decode([at]),
                     "logit": best}
    return out


class Stop(Exception):
    pass


class Recorder:
    """Reads what one pipeline call makes on its way to the features and
    stops the call after encode_prompt."""

    def __init__(self, torch, pipe):
        self.torch, self.pipe = torch, pipe
        self.resized, self.vae_inputs, self.seen, self.patches = [], [], [], []
        self.grids, self.ids, self.result = None, None, None
        vae_proc, proc = pipe.image_processor, pipe.processor
        self._resize, self._preprocess = vae_proc.resize, vae_proc.preprocess
        self._encode = pipe.encode_prompt
        rec = self

        nested = []

        def resize(image, height=None, width=None, **kw):
            out = rec._resize(image, height=height, width=width, **kw)
            # preprocess resizes through this method too: the image the
            # encoder gets is the one the call resizes itself.
            if not nested:
                rec.resized.append(np.asarray(out).copy())
            return out

        def preprocess(image, height=None, width=None, **kw):
            nested.append(1)
            try:
                out = rec._preprocess(image, height=height, width=width,
                                      **kw)
            finally:
                nested.pop()
            rec.vae_inputs.append(oracle.to_numpy(out))
            return out

        vae_proc.resize, vae_proc.preprocess = resize, preprocess
        base = type(proc.image_processor)

        class Seen(base):
            def __call__(self, images=None, *a, **kw):
                rec.seen = [np.asarray(i).copy() for i in images]
                out = base.__call__(self, images, *a, **kw)
                rec.patches = oracle.to_numpy(out["pixel_values"])
                rec.grids = np.asarray(out["image_grid_thw"]).astype(np.int64)
                return out

        self._base = base
        proc.image_processor.__class__ = Seen

        def encode_prompt(**kw):
            tap = c2.Tap(pipe.text_encoder.model.language_model)
            try:
                t0 = time.perf_counter()
                rec.result = (*rec._encode(**kw), tap.out,
                              time.perf_counter() - t0)
            finally:
                tap.remove()
            raise Stop

        pipe.encode_prompt = encode_prompt

    def close(self):
        vae_proc = self.pipe.image_processor
        vae_proc.resize, vae_proc.preprocess = self._resize, self._preprocess
        self.pipe.processor.image_processor.__class__ = self._base
        del self.pipe.encode_prompt


def run_case(torch, pipe, case: str, sources: dict) -> dict:
    """One call of the pipeline for ``case``, stopped after its features."""
    from PIL import Image

    prompt, names = CASES[case]
    images = [Image.fromarray(sources[n]) for n in names]
    rec = Recorder(torch, pipe)
    try:
        pipe(prompt=prompt, image=images, height=RESOLUTION, width=RESOLUTION,
             output_resolution=RESOLUTION, num_inference_steps=2,
             output_type="latent")
        raise SystemExit("error: the call ran past encode_prompt")
    except Stop:
        pass
    finally:
        rec.close()
    embeds, mask, pads, hooked, secs = rec.result
    drop = int(pipe._drop_idx)
    n = len(names)
    if mask is not None or embeds.shape[0] != 1 or len(rec.resized) != n:
        raise SystemExit(f"error: {case}: a mask, a batch or "
                         f"{len(rec.resized)} resized images for {n}")
    slots = np.flatnonzero(pads[0].numpy()).astype(np.int64)
    grids = rec.grids
    want = [int(g[1] * g[2]) // 4 for g in grids]
    if len(grids) != n or sum(want) != len(slots):
        raise SystemExit(f"error: {case}: grids {grids.tolist()} for "
                         f"{len(slots)} image rows")
    for img, rgb, vae, g in zip(rec.resized, rec.seen, rec.vae_inputs, grids):
        h, w = img.shape[:2]
        if img.shape[2] != 4 or (int(g[1]) * 16, int(g[2]) * 16) != (h, w):
            raise SystemExit(f"error: {case}: the processor resized a "
                             f"{w}x{h} image to grid {g.tolist()}")
        # The VAE input is the resized RGBA image on the [-1, 1] scale.
        ref = (img.astype(np.float32) / 255.0) * 2.0 - 1.0
        got = vae[0].transpose(1, 2, 0)
        if not np.array_equal(got, ref):
            raise SystemExit(f"error: {case}: the VAE input is not the "
                             f"resized image, max {np.abs(got - ref).max()}")
        a = img[..., 3:4].astype(np.float64) / 255.0
        white = img[..., :3] * a + 255.0 * (1.0 - a)
        if np.abs(rgb.astype(np.float64) - white).max() > 1.0:
            raise SystemExit(f"error: {case}: the encoder image is not the "
                             "alpha over white")
    return {"prompt": prompt, "names": names, "tap": oracle.to_numpy(embeds[0]),
            "slots": slots, "grids": grids, "resized": rec.resized,
            "encoder_rgb": rec.seen, "patches": rec.patches,
            "seconds": secs, "rows": int(embeds.shape[1]),
            "hook_equals_features": bool(torch.equal(hooked[0, drop:],
                                                     embeds[0]))}


def rendered(pipe, case: str) -> str:
    """The template text of a case, as the pipeline renders it."""
    prompt, names = CASES[case]
    slot = "<image1><|vision_start|><|image_pad|><|vision_end|>"
    more = "".join(f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
                   for i in range(2, len(names) + 1))
    return pipe.prompt_template_ti2i.replace(slot, slot + more).format(prompt)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", required=True, type=Path,
                    help="Qwen3-VL-8B text GGUF")
    ap.add_argument("--mmproj", required=True, type=Path,
                    help="Qwen3-VL-8B mmproj GGUF (BF16)")
    ap.add_argument("--config", required=True, type=Path,
                    help="local copy of Qwen/Qwen-Image-2.1 (processor, "
                         "text_encoder and scheduler configs)")
    ap.add_argument("--config-revision",
                    default="790c92633540aa0cb11d9abf19eb46d861714758",
                    help="revision of the --config copy, for the manifest")
    ap.add_argument("--cases", nargs="+", default=list(CASES),
                    choices=list(CASES))
    ap.add_argument("--vision-only", action="store_true",
                    help="load the tower alone and run it on each image; "
                         "writes no fixture")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()

    from diffusers import (FlowMatchEulerDiscreteScheduler,
                           QwenImage21Pipeline)
    from transformers import Qwen3VLProcessor

    text_proc = c2.text_processor(a.config)
    check = Qwen3VLProcessor.check_argument_for_proper_class
    Qwen3VLProcessor.check_argument_for_proper_class = (
        lambda self, name, arg: None if arg is None
        else check(self, name, arg))
    try:
        proc = Qwen3VLProcessor(
            image_processor=image_processor(a.config),
            tokenizer=text_proc.tokenizer, video_processor=None,
            chat_template=text_proc.chat_template)
    finally:
        del Qwen3VLProcessor.check_argument_for_proper_class
    sources = {n: source_image(n) for n in IMAGES}

    if a.vision_only:
        from accelerate import init_empty_weights
        from PIL import Image
        from transformers import (AutoConfig,
                                  Qwen3VLForConditionalGeneration)

        cfg = AutoConfig.from_pretrained(str(a.config / "text_encoder"),
                                         local_files_only=True)
        with init_empty_weights():
            te = Qwen3VLForConditionalGeneration(cfg)
        census = set_vision(torch, te, a.mmproj)
        print(f"[check2e] tower set, {census}", flush=True)
        for n, src in sources.items():
            inputs = proc.image_processor(images=[Image.fromarray(
                src[..., :3])], return_tensors="pt")
            out = te.model.visual(inputs["pixel_values"],
                                  grid_thw=inputs["image_grid_thw"])
            feats = out.pooler_output
            print(f"[check2e] {n}: grid "
                  f"{inputs['image_grid_thw'].tolist()}, features "
                  f"{tuple(feats.shape)}, finite "
                  f"{bool(torch.isfinite(feats).all())}, row norm "
                  f"{float(feats.float().norm(dim=-1).mean()):.1f}, "
                  f"deepstack {len(out.deepstack_features)}", flush=True)
        return 0

    t0 = time.perf_counter()
    te, reader, census = c2.load_encoder(
        torch, a, vision=lambda model: set_vision(torch, model, a.mmproj))
    load_s = time.perf_counter() - t0
    print(f"[check2e] loaded in {load_s:.0f} s, {census}", flush=True)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(a.config), subfolder="scheduler", local_files_only=True)
    pipe = QwenImage21Pipeline(scheduler=sched, vae=None, text_encoder=te,
                               processor=proc, transformer=None)
    sanity = c2.next_token(torch, te, reader, proc)
    colours = colour_check(torch, te, reader, proc)
    print(f"[check2e] next token {sanity['token']!r}, colours "
          f"{ {k: v['token'] for k, v in colours.items()} }", flush=True)
    if "Paris" not in sanity["token"] or any(
            k not in v["token"].lower() for k, v in colours.items()):
        raise SystemExit(f"error: name map checks: {sanity}, {colours}")

    fx = fixtures.FixtureWriter(
        a.name, check=2, family="qwen_image21",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--encoder", a.encoder.name, "--mmproj",
              a.mmproj.name, "--config", a.config.name])
    fx.add_file("encoder", a.encoder)
    fx.add_file("mmproj", a.mmproj)
    for role, rel in (("tokenizer", "processor/tokenizer.json"),
                      ("chat_template", "processor/chat_template.jinja"),
                      ("preprocessor_config",
                       "processor/preprocessor_config.json"),
                      ("text_encoder_config", "text_encoder/config.json")):
        fx.add_file(role, a.config / rel)
    cases = list(a.cases)
    runs = {"fp32": {}, "bf16": {}, "bf16_tower": {}}
    for case in cases:
        r = runs["fp32"][case] = run_case(torch, pipe, case, sources)
        print(f"[check2e] fp32 {case}: {r['rows']} rows, "
              f"{len(r['slots'])} image rows, {r['seconds']:.1f} s",
              flush=True)
    lm, visual = te.model.language_model, te.model.visual
    cast_s = 0.0
    oracle.emulate_bf16_matmul(True)
    matmul = oracle.matmul_mode()
    # The language model first, then the tower too. Each cast rounds the
    # rotary frequencies of its module. They stay float32, so a gap
    # measures the arithmetic and not the position tables.
    for mode, module, rotary in (("bf16", lm, lm.rotary_emb),
                                 ("bf16_tower", visual,
                                  visual.rotary_pos_emb)):
        t0 = time.perf_counter()
        inv = rotary.inv_freq.clone()
        module.to(torch.bfloat16)
        rotary.inv_freq = inv
        cast_s += time.perf_counter() - t0
        kinds = ({p.dtype for p in lm.parameters()},
                 {p.dtype for p in visual.parameters()})
        want = ({torch.bfloat16}, {torch.float32 if mode == "bf16"
                                   else torch.bfloat16})
        if kinds != want:
            raise SystemExit(f"error: the {mode} model holds {kinds}")
        for case in cases:
            r = runs[mode][case] = run_case(torch, pipe, case, sources)
            print(f"[check2e] {mode} {case}: {r['seconds']:.1f} s",
                  flush=True)
    oracle.emulate_bf16_matmul(False)

    for n, src in sources.items():
        fx.add(f"source_{n}", src)
    stored: dict = {}
    gaps, checks, texts, rows, secs = {}, {}, {}, {}, {}
    tol_rel, tol_cos, layout, tower_gaps = {}, {}, {}, {}
    tol_part = {"image_rows": {}, "text_rows": {}}
    drop = int(pipe._drop_idx)
    for case in cases:
        r32, r16 = runs["fp32"][case], runs["bf16"][case]
        rt = runs["bf16_tower"][case]
        prompt, names = CASES[case]
        text = rendered(pipe, case)
        pad = proc.tokenizer.encode("<|image_pad|>")[0]
        ids = []
        at = 0
        for i in proc.tokenizer(text).input_ids:
            if i == pad:
                g = r32["grids"][at]
                ids.extend([pad] * (int(g[1] * g[2]) // 4))
                at += 1
            else:
                ids.append(i)
        ids = np.asarray(ids, np.int64)
        if (len(ids) - drop != r32["rows"] or not np.array_equal(
                np.flatnonzero(ids[drop:] == pad), r32["slots"])):
            raise SystemExit(f"error: {case}: the ids of the rendered "
                             "template do not give the rows of the call")
        for other in ("slots", "grids", "patches"):
            if not (np.array_equal(r16[other], r32[other])
                    and np.array_equal(rt[other], r32[other])):
                raise SystemExit(f"error: {case}: {other} differ by mode")
        for name, img, pix in zip(
                names, r32["resized"],
                np.split(r32["patches"], np.cumsum(
                    [int(g[1] * g[2]) for g in r32["grids"]])[:-1])):
            if name in stored:
                if not np.array_equal(stored[name], img):
                    raise SystemExit(f"error: image {name} resized two ways")
                continue
            stored[name] = img
            fx.add(f"resized_{name}", img)
            fx.add(f"patches_{name}", pix)
        fx.add(f"ids_{case}", ids)
        fx.add(f"slots_{case}", r32["slots"])
        fx.add(f"grids_{case}", r32["grids"])
        fx.add(f"tap_{case}", r32["tap"])
        fx.add(f"tap_bf16_{case}", r16["tap"])
        gap = fixtures.compare(r16["tap"], r32["tap"])
        cos = token_cosine(r16["tap"], r32["tap"])
        is_slot = np.zeros(r32["rows"], bool)
        is_slot[r32["slots"]] = True
        whole = fixtures.compare(rt["tap"], r32["tap"])
        whole_cos = token_cosine(rt["tap"], r32["tap"])
        whole["token_cosine_min"] = float(whole_cos.min())
        for part, sel in (("image_rows", is_slot), ("text_rows", ~is_slot)):
            gap[part] = {
                "rows": int(sel.sum()),
                "rel_l2": fixtures.rel_l2(r16["tap"][sel], r32["tap"][sel]),
                "token_cosine_min": float(cos[sel].min()),
                "row_norm_median": float(np.median(np.linalg.norm(
                    r32["tap"][sel].astype(np.float64), axis=-1)))}
            tol_part[part][case] = 2 * gap[part]["rel_l2"]
            whole[part] = {
                "rel_l2": fixtures.rel_l2(rt["tap"][sel], r32["tap"][sel]),
                "token_cosine_min": float(whole_cos[sel].min())}
        tower_gaps[case] = whole
        row = int(cos.argmin())
        gap["token_cosine_min"] = float(cos.min())
        gap["token_cosine_min_row"] = row
        gap["token_cosine_min_token_id"] = int(ids[drop + row])
        tol_cos[case] = min(c2.STARTING["token_cosine_min"],
                            1.0 - 4.0 * (1.0 - gap["token_cosine_min"]))
        tol_rel[case] = 2 * gap["rel_l2"]
        gaps[case] = gap
        checks[case] = {m: r["hook_equals_features"]
                        for m, r in (("fp32", r32), ("bf16", r16),
                                     ("bf16_tower", rt))}
        texts[case] = text
        rows[case] = r32["rows"]
        layout[case] = {
            "images": list(names),
            "sizes": [[int(i.shape[1]), int(i.shape[0])]
                      for i in r32["resized"]],
            "image_rows": [int(g[1] * g[2]) // 4 for g in r32["grids"]],
            "first_slot": int(r32["slots"][0])}
        secs[case] = {"fp32_s": round(r32["seconds"], 1),
                      "bf16_s": round(r16["seconds"], 1),
                      "bf16_tower_s": round(rt["seconds"], 1)}
        print(f"[check2e] {case}: {len(ids)} ids, {rows[case]} rows, bf16 "
              f"vs fp32 rel L2 {gap['rel_l2']:.3e} (image rows "
              f"{gap['image_rows']['rel_l2']:.3e}, text rows "
              f"{gap['text_rows']['rel_l2']:.3e}), token cos min "
              f"{gap['token_cosine_min']:.6f}; with the tower in bf16 "
              f"{whole['rel_l2']:.3e} (image rows "
              f"{whole['image_rows']['rel_l2']:.3e}), token cos min "
              f"{whole['token_cosine_min']:.6f}", flush=True)
        if not all(checks[case].values()):
            raise SystemExit(f"error: {case}: tap check failed: "
                             f"{checks[case]}")
    # What a wrong order of two references moves, on the oracle's own
    # taps: the fixture's gate must be under it.
    order = {}
    if {"two", "two_swapped"} <= set(cases):
        x, y = runs["fp32"]["two"]["tap"], runs["fp32"]["two_swapped"]["tap"]
        order = {"rel_l2": fixtures.rel_l2(y, x),
                 "token_cosine_min": float(token_cosine(y, x).min())}
        print(f"[check2e] two against two_swapped: {order}", flush=True)

    fx.inputs.update(
        cases={k: {"prompt": CASES[k][0], "images": list(CASES[k][1])}
               for k in cases},
        images={n: {"width": w, "height": h, "alpha": alpha}
                for n, (w, h, alpha) in IMAGES.items()},
        image_seed=SEED, output_resolution=RESOLUTION, rendered=texts,
        rows=rows, layout=layout, drop_index=drop,
        tap="output of decoder layer 35 (all 36 layers run), before the "
        "final norm, less the first drop_index rows",
        template="QwenImage21Pipeline.prompt_template_ti2i with one image "
        "slot per reference, as _get_qwen_prompt_embeds renders it",
        call="QwenImage21Pipeline.__call__(prompt, image, height, width, "
        "output_resolution), stopped after encode_prompt",
        padding="none: batch 1",
        config_revision=a.config_revision,
        encoder="Qwen3VLForConditionalGeneration with the language model "
        "set by name from the text GGUF, gguf-py float32 dequantization, "
        "and the vision tower set by name from the mmproj GGUF, whose "
        "BF16 and F32 tensors are read as float32; the output head is not "
        "loaded; the bf16 models are the float32 one rounded in place, the "
        "language model first and then the tower too, with the rotary "
        "frequencies left in float32",
        processor="Qwen3VLProcessor with the tokenizer and chat template of "
        "the official processor folder, the PIL backend of its image "
        "processor config and no video processor",
        gguf_codecs=census)
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "bf16_matmul": matmul},
        name_map_check=sanity, vision_map_check=colours, tap_checks=checks,
        bf16_vs_fp32=gaps, bf16_with_tower_vs_fp32=tower_gaps,
        reference_order=order,
        vae_input="the resized RGBA image on the [-1, 1] scale, equal in "
        "bits for every image",
        seconds={"load_s": round(load_s, 1), "cast_s": round(cast_s, 1),
                 **secs})
    fx.tolerances.update(rule=RULE, starting=c2.STARTING, rel_l2=tol_rel,
                         token_cosine_min=tol_cos,
                         rel_l2_image_rows=tol_part["image_rows"],
                         rel_l2_text_rows=tol_part["text_rows"])
    fx.note("The pipeline encodes one prompt with its images at batch 1 "
            "with no padding and no mask, as gmlx encodes one.")
    fx.note("The pipeline loads the encoder in bfloat16, tower and all. "
            "The run with the tower in bfloat16 is recorded under "
            "bf16_with_tower_vs_fp32 and sets no gate.")
    path = fx.write(a.out)
    print(f"[check2e] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
