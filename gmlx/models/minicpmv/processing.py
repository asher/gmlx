# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 OpenBMB and the HuggingFace Inc. team.
# Ported from transformers src/transformers/models/minicpmv4_6/
# image_processing_pil_minicpmv4_6.py and processing_minicpmv4_6.py, and the
# 4.7 input check of src/transformers/models/minicpmv4_7/modular_minicpmv4_7.py,
# at commit 674c0e3, and modified for gmlx (Apache-2.0; see
# licenses/transformers-LICENSE).
"""MiniCPM-V 4.6 / 4.7 image processor and processor, torch-free.

An image becomes an overview crop of about 448 x 448 pixels and, when it
is larger than that, a grid of slice crops of the refined image. Every
side is a multiple of 56 pixels (4 patches), so each crop folds evenly
into text tokens. Each ``<|image_pad|>`` the chat template writes expands
to::

    <image_id>N</image_id><image>{pad x overview}</image>
    <slice>{pad x slice}</slice>...   (one row per grid row, joined by "\\n")
"""

from __future__ import annotations

import math

import numpy as np
from transformers.image_processing_utils import ImageProcessingMixin


def ensure_divide(length: float, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


class MiniCPMVImageProcessor(ImageProcessingMixin):
    """Slices, resizes and patchifies images the way the reference PIL
    image processor does. Returns ``pixel_values [N, p * p * C]`` (every
    crop's patches row-major in ``(kh, kw, C)`` order) and
    ``target_sizes [n_crops, 2]``."""

    model_input_names = ["pixel_values", "target_sizes"]

    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
                 patch_size: int = 14, scale_resolution: int = 448,
                 max_slice_nums: int = 9, downsample_mode: str = "16x",
                 slice_mode: bool = True):
        super().__init__()
        self.image_mean = [float(v) for v in image_mean]
        self.image_std = [float(v) for v in image_std]
        self.patch_size = int(patch_size)
        self.scale_resolution = int(scale_resolution)
        self.max_slice_nums = int(max_slice_nums)
        self.downsample_mode = downsample_mode
        self.slice_mode = bool(slice_mode)

    @property
    def token_divisor(self) -> int:
        """Patches per text token."""
        return 4 if self.downsample_mode == "4x" else 16

    def find_best_resize(self, image_size, allow_upscale: bool = False):
        height, width = image_size
        res = self.scale_resolution
        if height * width > res * res or allow_upscale:
            aspect = width / height
            height = int(res / math.sqrt(aspect))
            width = int(height * aspect)
        align = self.patch_size * 4
        return ensure_divide(height, align), ensure_divide(width, align)

    def get_refine_size(self, image_size, grid):
        height, width = image_size
        grid_y, grid_x = grid
        refine_w = ensure_divide(width, grid_x)
        refine_h = ensure_divide(height, grid_y)
        best_h, best_w = self.find_best_resize(
            (refine_h / grid_y, refine_w / grid_x), allow_upscale=True)
        return best_h * grid_y, best_w * grid_x

    def get_sliced_grid(self, image_size):
        height, width = image_size
        log_ratio = math.log(width / height)
        ratio = width * height / (self.scale_resolution ** 2)
        multiple = min(math.ceil(ratio), self.max_slice_nums)
        if multiple <= 1:
            return None
        best, min_error = [1, 1], float("inf")
        for n in (multiple - 1, multiple, multiple + 1):
            if n == 1 or n > self.max_slice_nums:
                continue
            for rows in range(1, n + 1):
                if n % rows:
                    continue
                cols = n // rows
                error = abs(log_ratio - math.log(cols / rows))
                if error < min_error:
                    best, min_error = [rows, cols], error
                elif error == min_error and rows > best[0]:
                    best = [rows, cols]
        return best

    def _patches(self, img) -> tuple[np.ndarray, list[int]]:
        """``[n, p * p * C]`` normalized patches of one crop, row-major."""
        arr = np.asarray(img, dtype=np.float32) / 255.0          # [H, W, C]
        mean = np.asarray(self.image_mean, dtype=np.float32)
        std = np.asarray(self.image_std, dtype=np.float32)
        arr = (arr - mean) / std
        p = self.patch_size
        h, w, c = arr.shape
        nh, nw = h // p, w // p
        patches = arr.reshape(nh, p, nw, p, c).transpose(0, 2, 1, 3, 4)
        return patches.reshape(nh * nw, p * p * c), [nh, nw]

    def process_one(self, img):
        """``(patch arrays, target sizes, grid)`` of one image; ``grid`` is
        ``(rows, cols)`` of the slices, or None."""
        from PIL import Image

        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        if img.mode != "RGB":
            img = img.convert("RGB")
        size = (img.height, img.width)
        grid = self.get_sliced_grid(size) if self.slice_mode else None
        bicubic = Image.Resampling.BICUBIC

        oh, ow = self.find_best_resize(size, allow_upscale=grid is None)
        crops = [img.resize((ow, oh), bicubic)]
        if grid is not None:
            rh, rw = self.get_refine_size(size, grid)
            refined = img.resize((rw, rh), bicubic)
            rows, cols = grid
            ch, cw = rh // rows, rw // cols
            for r in range(rows):
                for c in range(cols):
                    crops.append(refined.crop((c * cw, r * ch, (c + 1) * cw,
                                               (r + 1) * ch)))
        arrays, sizes = [], []
        for crop in crops:
            a, s = self._patches(crop)
            arrays.append(a)
            sizes.append(s)
        return arrays, sizes, grid

    def __call__(self, images, **kwargs):
        out = self.preprocess(images)
        return {"pixel_values": out["pixel_values"],
                "target_sizes": out["target_sizes"]}

    def preprocess(self, images):
        """Pixel arrays plus the per-image layout the processor expands
        placeholders from (``grids``, ``crops_per_image``)."""
        if not isinstance(images, (list, tuple)):
            images = [images]
        arrays, sizes, grids, counts = [], [], [], []
        for img in _flatten(images):
            a, s, g = self.process_one(img)
            arrays += a
            sizes += s
            grids.append(g)
            counts.append(len(a))
        return {
            "pixel_values": np.concatenate(arrays, axis=0),
            "target_sizes": np.asarray(sizes, dtype=np.int32),
            "grids": grids,
            "crops_per_image": counts,
        }


def _flatten(images):
    out = []
    for img in images:
        if isinstance(img, (list, tuple)):
            out += _flatten(img)
        else:
            out.append(img)
    return out


def build_processor(tokenizer, image_processor: MiniCPMVImageProcessor, *,
                    use_image_id: bool = True):
    """The MiniCPM-V processor over a GGUF tokenizer."""
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.processing_utils import ProcessorMixin

    from gmlx._vendor.mlx_vlm.models.base import to_mlx
    from gmlx.upstream.single_bos_fix import KWARGS_REACH_TOKENIZER

    class MiniCPMVProcessor(ProcessorMixin):
        attributes = ["image_processor", "tokenizer"]
        image_processor_class = "AutoImageProcessor"
        tokenizer_class = "AutoTokenizer"

        image_token = "<|image_pad|>"
        image_start_token = "<image>"
        image_end_token = "</image>"
        slice_start_token = "<slice>"
        slice_end_token = "</slice>"
        image_id_start_token = "<image_id>"
        image_id_end_token = "</image_id>"

        # Images and text only; the base signature also names audio, which
        # reaches the tokenizer through kwargs here.
        def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
                self, images=None, text=None, videos=None, **kwargs):
            if videos is not None:
                raise ValueError("MiniCPM-V video input is not supported")
            if text is None:
                raise ValueError("MiniCPM-V needs a text prompt")
            if isinstance(text, str):
                text = [text]
            n_placeholders = sum(t.count(self.image_token) for t in text)
            n_images = 0
            if images is not None:
                if not isinstance(images, (list, tuple)):
                    images = [images]
                n_images = len(_flatten(images))
            if n_placeholders != n_images:
                raise ValueError(
                    f"Number of image placeholders does not match the number "
                    f"of images: found {n_placeholders} placeholder(s) in the "
                    f"prompt but received {n_images} image(s)")

            image_inputs = {}
            if n_images:
                pre = self.image_processor.preprocess(images)
                text = self._expand(text, pre)
                image_inputs = {"pixel_values": pre["pixel_values"],
                                "target_sizes": pre["target_sizes"]}

            kwargs.pop("return_tensors", None)
            data = {**self.tokenizer(text, **kwargs), **image_inputs}
            return BatchFeature(data=to_mlx(data))

        def _replacement(self, pre, index: int, start: int) -> str:
            sizes = pre["target_sizes"]
            div = self.image_processor.token_divisor
            n = pre["crops_per_image"][index]
            tokens = [int(h) * int(w) // div for h, w in sizes[start:start + n]]
            out = (self.image_start_token + self.image_token * tokens[0]
                   + self.image_end_token)
            grid = pre["grids"][index]
            if grid is not None:
                rows, cols = grid
                one = (self.slice_start_token + self.image_token * tokens[1]
                       + self.slice_end_token)
                out += "\n".join(one * cols for _ in range(rows))
            return out

        def _expand(self, texts, pre):
            out, index, start = [], 0, 0
            for sample in texts:
                parts = sample.split(self.image_token)
                rebuilt = parts[0]
                for local, part in enumerate(parts[1:]):
                    if use_image_id:
                        rebuilt += (f"{self.image_id_start_token}{local}"
                                    f"{self.image_id_end_token}")
                    rebuilt += self._replacement(pre, index, start) + part
                    start += pre["crops_per_image"][index]
                    index += 1
                out.append(rebuilt)
            return out

        def batch_decode(self, *args, **kwargs):
            return self.tokenizer.batch_decode(*args, **kwargs)

        def decode(self, *args, **kwargs):
            return self.tokenizer.decode(*args, **kwargs)

        @property
        def model_input_names(self):
            return list(dict.fromkeys(
                list(self.tokenizer.model_input_names)
                + list(self.image_processor.model_input_names)))

    # __call__ hands **kwargs to self.tokenizer (single_bos_fix).
    setattr(MiniCPMVProcessor, KWARGS_REACH_TOKENIZER, True)
    return MiniCPMVProcessor(
        image_processor=image_processor, tokenizer=tokenizer,
        chat_template=getattr(tokenizer, "chat_template", None))
