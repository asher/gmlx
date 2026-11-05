"""The text encoder of a video family.

A video family conditions on a text model through the image stack's
``EncoderSpec`` and ``EncoderHandle``. Its spec carries what the
reference's tokenizer adds to the encoder's own: tokens at fixed ids and
the ids of an empty prompt. MiniMax-H3's tokenizer adds seven tokens
whose ids the stock Qwen3-VL GGUF holds as filler entries, so the GGUF
tokenizer cannot produce them. The handle splits them out of the prompt
and puts their ids between the encoded pieces, which leaves the GGUF
tokenizer as it is for chat.

An encode with keyframes needs the vision tower from the entry's mmproj.
It loads in stages: the text encoder alone, cut after its tap layer as
for text to video, then the tower, cast to float32 as the image stack
casts it, and the mmproj's 16-bit tensors are released after the cast.
Loading the two files together holds the whole encoder beside the
mmproj and its float32 copy.

Importing this module loads no mlx.
"""

from __future__ import annotations

import os


def header_tokenizer(family, path: str):
    """The tokenizer the encoder's load builds, from the GGUF header
    alone, with no tensor reads. A file that is not the family's encoder
    is refused first, by its arch and shape."""
    from gmlx.image import text_encoder as te
    from gmlx.image.fingerprint import view_from_scans
    from gmlx.load import loadlog
    from gmlx.load.config_synth import GGUF_ARCH_TO_MODEL_TYPE
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.tokenizer import (finish_gguf_tokenizer,
                                     load_tokenizer_from_gguf)

    if family.encoder is None:
        raise ValueError(f"video family {family.name!r} has no text encoder")
    scan = scan_gguf(os.path.expanduser(path), include_tensors=False,
                     array_limit=1 << 62)
    # The identity digests read tensors; the load checks them.
    te.check_encoder_file(view_from_scans([scan]), family.encoder(),
                          override=True)
    arch = scan.kv["general.architecture"]
    with loadlog.quiet():
        tok = load_tokenizer_from_gguf(scan.kv, arch)
        finish_gguf_tokenizer(tok, GGUF_ARCH_TO_MODEL_TYPE.get(arch))
    return tok


def load_encoder(family, path: str, *, override: bool = False,
                 mmproj: str | None = None, log=print):
    """An owned ``EncoderHandle`` for the family's spec from the encoder
    GGUF at ``path``, with its vision tower from ``mmproj`` when one is
    given. Both headers are checked against the spec before any weights
    load."""
    from gmlx.image import text_encoder as te

    if family.encoder is None:
        raise ValueError(f"video family {family.name!r} has no text encoder")
    spec = family.encoder()
    if mmproj is None:
        return te.load_encoder(path, spec, override=override, log=log,
                               tag="video")
    if spec.vision_type != _TOWER:
        raise te.ImageModelError(
            f"the {spec.name} text encoder takes no mmproj: its family "
            "conditions on no reference images" if spec.vision_type is None
            else f"the {spec.name} vision tower is {spec.vision_type}, and "
            f"the video encoder attaches a {_TOWER} tower alone")
    te.check_encoder_file(te._header_view(path), spec, override=override)
    te.check_mmproj_file(te._header_view(mmproj), spec, override=override)
    # The encoder's identity was checked just above.
    handle = te.load_encoder(path, spec, override=True, log=log, tag="video")
    try:
        attach_tower(handle, path, mmproj)
    except BaseException:
        handle.release()
        raise
    log(f"[video] vision tower of {spec.name} from "
        f"{os.path.basename(mmproj)}, in float32")
    return handle


# The multimodal model whose language model is the text load's own.
_TOWER = "qwen3_vl"
_PREFIX = "vision_tower."


def attach_tower(handle, path: str, mmproj: str) -> None:
    """Give an owned encoder loaded without its mmproj the vision tower,
    as the joint load builds it: the multimodal model of the two headers,
    its language model made of the loaded encoder's modules, its tower
    from ``mmproj`` with every 16-bit float tensor cast to float32. The
    handle takes the model and its processor."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from gmlx.gen import prefill_decay as pd
    from gmlx.image import text_encoder as te
    from gmlx.load import loadlog, vlm
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.loader import (collect_after_load,
                                  materialize_module_arrays,
                                  weights_source_key)
    from gmlx.load.preflight import find_split_shards
    from gmlx.load.tokenizer import (bundled_chat_template_for_arch,
                                     load_tokenizer_from_gguf)
    from gmlx.load.wire import load_gguf_wire_bytes

    text = handle.model
    if not handle.owned or text is None:
        raise RuntimeError("a vision tower attaches to an owned, loaded "
                           "encoder")
    if getattr(text, "vision_tower", None) is not None:
        raise RuntimeError("the encoder holds its vision tower already")
    shards = find_split_shards(path)
    scans = [scan_gguf(s, array_limit=1 << 62) for s in shards]
    kv = scans[0].kv
    arch = kv["general.architecture"]
    shapes = {t.name: list(t.shape) for s in scans for t in s.tensors}
    kind = vlm.resolve_vlm_model_type(
        arch, scan_gguf(mmproj, include_tensors=False).kv)
    if kind != _TOWER:
        raise te.ImageModelError(
            f"the mmproj is a {kind} vision tower, not {_TOWER}")

    active = mx.get_active_memory()
    arrays, codecs, _, meta, mm_shapes = load_gguf_wire_bytes(
        mmproj, expect_quant=False)
    weights, _, packed = vlm.remap_vision_arrays(
        arrays, kind, with_audio=bool(meta.get("clip.has_audio_encoder")),
        mm_codecs=codecs, mm_meta=meta)
    names = set(arrays)
    del arrays
    if packed:
        raise te.ImageModelError(
            f"the mmproj stores {len(packed)} vision tower weights quantized; "
            f"the {handle.spec.name} keyframe encode runs the tower in "
            "float32 from a 16- or 32-bit mmproj")
    outside = sorted(k for k in weights if not k.startswith(_PREFIX))
    if outside:
        raise te.ImageModelError(
            f"the mmproj maps {', '.join(outside[:3])} outside the "
            f"{_TOWER} vision tower")
    # The text load printed the config and tokenizer lines once.
    with loadlog.quiet():
        config = vlm.synthesize_vlm_config(kind, kv, shapes, meta,
                                           mm_tensor_names=names,
                                           mm_shapes=mm_shapes)
    model, config = vlm.build_vlm_model(config)
    # No array of the built model evaluates until each is replaced: the
    # language model takes the loaded modules, the tower the mmproj's.
    _take_text(model.language_model, text)
    vlm.scale_text_rope(model, config)

    tower = model.vision_tower
    slots = dict(tree_flatten(tower.parameters()))
    # A tensor with no slot is dropped, as the joint load drops it. A
    # 16-bit float is cast as the joint load casts the qwen3_vl tower
    # (loader._FP32_CAST_BY_MODEL_TYPE); 2 bytes to 4, so no cast can
    # write through the file mapping.
    loaded = []
    for k in list(weights):
        v = weights.pop(k)
        k = k[len(_PREFIX):]
        if k in slots:
            loaded.append((k, v.astype(mx.float32)
                           if v.dtype in (mx.float16, mx.bfloat16) else v))
    tower.load_weights(loaded, strict=True)
    del loaded, weights, slots
    mx.eval(tower.parameters())
    held = sum(int(v.nbytes)
               for v in dict(tree_flatten(tower.parameters())).values())
    # The tower's 32-bit tensors stay views of the file, which MLX's
    # active memory does not count.
    window = (handle.weights_owner is not None
              and pd.untracked_weights_owner() is None)
    if window:
        pd.set_untracked_weights_owner(handle.weights_owner)
    try:
        pd.note_untracked_weights(
            max(0, held - max(0, mx.get_active_memory() - active)),
            key=weights_source_key(mmproj))
    finally:
        if window:
            pd.set_untracked_weights_owner(None)
    object.__setattr__(model, "_kq_weights_key",
                       getattr(text, "_kq_weights_key", None))
    materialize_module_arrays(model)
    with loadlog.quiet():
        tokenizer = load_tokenizer_from_gguf(
            kv, arch,
            chat_template_override=bundled_chat_template_for_arch(arch))
    handle.processor = vlm._synthesize_vlm_processor(
        kind, tokenizer, meta, config=config)
    handle.model = model
    handle.source_key = weights_source_key(*shards, mmproj)
    del text
    collect_after_load()
    mx.clear_cache()


def _take_text(lm, text) -> None:
    """Put the loaded text model's modules into the multimodal model's
    language model ``lm``: its kept decoder layers, embedding and norm,
    leaf by leaf, by the paths the two trees share. The layers past the
    kept ones and the output head go unbuilt. A loaded module with
    weights and no place in ``lm`` is refused, and so is a parameter of
    ``lm`` left as built, which would run on random weights."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    src, dst = text.model, lm.model
    n = len(src.layers)
    if not 1 <= n <= len(dst.layers):
        raise RuntimeError(f"the encoder keeps {n} layers, and its "
                           f"multimodal model has {len(dst.layers)}")
    dst.layers = dst.layers[:n]
    if "lm_head" in lm:
        del lm["lm_head"]

    def leaves(m):
        return dict(tree_flatten(m.leaf_modules(),
                                 is_leaf=nn.Module.is_module))

    have, slots = leaves(src), leaves(dst)
    lost = sorted(p for p, m in have.items()
                  if p not in slots and m.parameters())
    if lost:
        raise RuntimeError("the multimodal language model has no place for "
                           f"{', '.join(lost[:3])} of the text load")
    for path, mod in have.items():
        if path not in slots:
            continue
        *up, name = path.split(".")
        node = dst
        for part in up:
            node = node[int(part)] if part.isdigit() else node[part]
        setattr(node, name, mod)
    loaded = {id(v) for _, v in tree_flatten(src.parameters())}
    stray = [k for k, v in tree_flatten(lm.parameters()) if id(v) not in loaded]
    if stray:
        raise RuntimeError("the multimodal language model keeps parameters "
                           f"the text load did not fill: {', '.join(stray[:3])}")
