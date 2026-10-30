"""``gmlx video bench``: step time and peak memory of video runs.

The bench runs what ``generate`` runs, one model at a time: it encodes
the prompt and releases the encoder, loads the DiT once and samples each
shape, then decodes each shape's sound and frames. It times every step,
takes MLX's peak memory over each phase of each shape beside what the
memory plan predicts, and writes records of schema
``gmlx.video.bench/1``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .video import (_adapter, _from_config, _positive, _release,
                    _schedules, _seed, _shift, _size, _Usage)

BENCH_SCHEMA = "gmlx.video.bench/1"
DEFAULT_SHAPES = "512x512x22,768x768x124,1344x768x124"
BENCH_PROMPT = ("A red bicycle leans against a brick wall on a quiet street "
                "after rain. A dog trots past and barks twice.")
GiB = 1 << 30


def _shape(text: str) -> tuple[int, int, int]:
    """WxHxF: a size in multiples of 32 and a clip's frame count."""
    from .video import _frames

    if text.lower().count("x") != 2:
        raise argparse.ArgumentTypeError(f"expected WxHxF, got {text!r}")
    size, _, frames = text.lower().rpartition("x")
    w, h = _size(size)
    return w, h, _frames(frames)


def _shapes(text: str) -> list:
    shapes = [_shape(s.strip()) for s in text.split(",") if s.strip()]
    if not shapes:
        raise argparse.ArgumentTypeError("expected one or more WxHxF shapes")
    return shapes


def _parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Time the steps of video runs and take their peak "
                    "memory at each shape, with a JSON record of schema "
                    f"{BENCH_SCHEMA} per shape.")
    p.add_argument("model", metavar="MODEL",
                   help="The id of a video_models entry in the config, or "
                        "a DiT GGUF.")
    p.add_argument("--config", default=None, metavar="FILE",
                   help="The config whose video_models MODEL names.")
    p.add_argument("--variant", default=None, metavar="NAME",
                   help="Run the DiT as this variant.")
    p.add_argument("--shapes", type=_shapes, default=None, metavar="LIST",
                   help="Comma-separated WxHxF shapes (default "
                        f"{DEFAULT_SHAPES}).")
    p.add_argument("--steps", type=_positive, default=3, metavar="N",
                   help="Steps per shape (default 3).")
    p.add_argument("--seed", type=_seed, default=0, metavar="N",
                   help="Seed every shape (default 0).")
    p.add_argument("--shift", type=_shift, default=None, metavar="S",
                   help="The video flow shift (default: the entry's, else "
                        "the variant's).")
    p.add_argument("--adapter", action="append", type=_adapter, default=None,
                   metavar="PATH[:STRENGTH]",
                   help="Time with this LoRA applied after the entry's. "
                        "Repeat the flag to stack adapters.")
    p.add_argument("--prompt", default=BENCH_PROMPT,
                   help="Encode this prompt (default: a fixed bench "
                        "prompt).")
    p.add_argument("--no-decode", action="store_true",
                   help="Skip the sound and frame decodes, so no VAE "
                        "loads.")
    p.add_argument("--encoder", default=None, metavar="GGUF",
                   help="The text encoder GGUF.")
    p.add_argument("--vae", default=None, metavar="PATH",
                   help="The video VAE safetensors file.")
    p.add_argument("--audio-vae", default=None, metavar="PATH",
                   help="The audio VAE safetensors file.")
    p.add_argument("--label", default=None, metavar="TEXT",
                   help="Record this label with the results.")
    p.add_argument("--json", action="store_true",
                   help="Print one JSON record per shape on stdout.")
    p.add_argument("--json-out", default=None, metavar="FILE",
                   help="Write one JSON record per shape to this file.")
    return p


def _need(a) -> None:
    need = [("--encoder", a.encoder)]
    if not a.no_decode:
        need += [("--vae", a.vae), ("--audio-vae", a.audio_vae)]
    for flag, path in need:
        if not path:
            raise _Usage(f"{flag} is required with a DiT path as MODEL")
        if not os.path.isfile(os.path.expanduser(path)):
            raise FileNotFoundError(f"{flag} {path}: no such file")


def _median(xs: list) -> float | None:
    import statistics

    return float(statistics.median(xs)) if xs else None


class _Peak:
    """MLX's peak memory over a block, in bytes."""

    def __enter__(self):
        import mlx.core as mx

        mx.reset_peak_memory()
        return self

    def __exit__(self, *exc):
        import mlx.core as mx

        self.bytes = int(mx.get_peak_memory())
        return False


def _shifts(a, fam, variant: str) -> dict:
    """The variant's shifts with the video shift of the flag, else the
    entry's."""
    shifts = dict(fam.sampling_row(variant).shifts)
    video = a.shift if a.shift is not None else a.entry_defaults.get("shift")
    if video is not None:
        shifts["video"] = video
    return {k: float(v) for k, v in shifts.items()}


def _rows(lay, text: int) -> dict:
    return {"total": int(lay.rows), "text": text,
            **{k: int(lay.span(k).rows) for k in ("video", "audio")}}


def _files(a, adapters) -> dict:
    """The name and bytes of each file the bench reads."""
    from gmlx.video.memfit import file_bytes

    def one(path):
        p = os.path.expanduser(path)
        return {"file": os.path.basename(p), "bytes": file_bytes(p)}
    out = {"dit": one(a.model), "encoder": one(a.encoder),
           "adapters": [one(p) for p, _ in adapters]}
    if not a.no_decode:
        out.update(vae=one(a.vae), audio_vae=one(a.audio_vae))
    return out


def _plan(fam, files, shape, text: int, decode: bool):
    from gmlx.video import memfit

    w, h, frames = shape
    rows, audio = memfit.clip_rows(fam, frames=frames, width=w, height=h,
                                   text=text)
    return memfit.plan_memory(
        fam.name, dit=files["dit"]["bytes"],
        encoder=files["encoder"]["bytes"],
        vae=files["vae"]["bytes"] if decode else 0,
        audio_vae=files["audio_vae"]["bytes"] if decode else 0,
        loras=[f["bytes"] for f in files["adapters"]], rows=rows, text=text,
        audio_latents=audio, pixels=w * h, decode=decode)


def _emit(a, records: list) -> None:
    lines = "".join(json.dumps(r) + "\n" for r in records)
    if a.json:
        sys.stdout.write(lines)
    if a.json_out is not None:
        path = os.path.expanduser(a.json_out)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(lines)


def _bench(a) -> int:
    import mlx.core as mx

    from gmlx import __version__
    from gmlx.video import encoder as video_encoder
    from gmlx.video import memfit, pipeline
    from gmlx.video.loader import identify_dit
    from gmlx.video.registry import chip_refusal

    from .image import _machine

    if a.json_out is not None:
        from gmlx.tune.lora import probe_writable

        err = probe_writable(os.path.expanduser(a.json_out))
        if err is not None:
            raise OSError(f"cannot write {a.json_out}: {err}")
    _from_config(a)
    _need(a)

    def log(*x):
        print(*x, file=sys.stderr, flush=True)

    vid = identify_dit(os.path.expanduser(a.model), variant=a.variant)
    fam = vid.family
    why = chip_refusal(fam)
    if why:
        raise RuntimeError(why)
    adapters = list(a.entry_adapters) + list(a.adapter or ())
    for path, _ in adapters:
        pipeline.vet_adapter(fam, dict(vid.config), path)
    row = fam.variant_row(vid.variant)
    if adapters and row is not None and row.merged:
        log(f"[video] warning: this {fam.title} GGUF has {row.merged} "
            f"merged in, and each adapter runs on top of it")
    shapes = a.shapes or _shapes(DEFAULT_SHAPES)
    shifts = _shifts(a, fam, vid.variant)
    if not _schedules(a.steps, shifts):
        raise _Usage(f"video shift {shifts['video']:g} gives no "
                     f"{a.steps}-step schedule")
    files = _files(a, adapters)
    decode = not a.no_decode
    # The shape with the largest plan, priced from the prompt's text, by
    # its phases largest first. The encode is the same at every shape, so
    # on a tie there the next phase decides, and the bench is named only
    # when no shape has another phase over.
    text = memfit.prompt_rows(fam, a.prompt,
                              video_encoder.header_tokenizer(fam, a.encoder))
    plan, (w, h, f) = max(
        ((_plan(fam, files, s, text, decode), s) for s in shapes),
        key=lambda x: sorted(x[0].phases.values(), reverse=True))
    memfit.check(plan, what="this bench" if plan.over == ["encode"]
                 else f"the {w}x{h}x{f} shape")

    load_s, runs, done = {}, [], {}

    def record(r: dict, error: str | None) -> dict:
        w, h, frames = r["shape"]
        rest = r["step_s"][1:] or r["step_s"]
        # Priced at the prompt's tokens, as the cells are measured.
        predicted = {k: v for k, v in _plan(fam, files, r["shape"],
                                            done["text"], decode)
                     .phases.items() if k in r["peak"]}
        rec = {"schema": BENCH_SCHEMA, "gmlx_version": __version__,
               "mlx_version": mx.__version__, "label": a.label,
               "machine": _machine(), "model": {
                   "family": fam.name, "variant": vid.variant, **files},
               "settings": {"width": w, "height": h, "frames": frames,
                            "steps": a.steps, "shifts": shifts,
                            "seed": a.seed, "prompt": a.prompt,
                            "adapters": [[p, st] for p, st in adapters]},
               "load_s": {**load_s, "total": sum(load_s.values())},
               "encode_s": done["encode_s"],
               **{k: v for k, v in r.items()
                  if k not in ("shape", "latents")},
               "predicted": predicted,
               "summary": {"s_per_step": {"first": r["step_s"][0],
                                          "median": _median(rest),
                                          "mean": sum(rest) / len(rest)},
                           "peak_bytes": max(r["peak"].values()),
                           "predicted_peak_bytes": max(predicted.values())}}
        if error is not None:
            rec["error"] = error
        return rec

    try:
        _run(a, fam, shapes, shifts, adapters, load_s, runs, done, log)
    except BaseException as e:
        # The shapes that ran keep their records, with what they measured.
        if runs:
            err = type(e).__name__ + (f": {e}" if str(e) else "")
            _emit(a, [record(r, err) for r in runs])
        raise
    records = [record(r, None) for r in runs]
    for r in records:
        s = r["settings"]
        log(f"[bench] {s['width']}x{s['height']}x{s['frames']}: peak "
            "against the plan, GiB: " + ", ".join(
                f"{k} {v / GiB:.2f} of {r['predicted'][k] / GiB:.2f}"
                for k, v in r["peak"].items()))
    _emit(a, records)
    return 0


def _run(a, fam, shapes, shifts, adapters, load_s, runs, done, log) -> None:
    """Run every phase of every shape, appending to ``runs`` as each
    shape's denoise ends."""
    import mlx.core as mx

    from gmlx.image.memfit import bounded_cache
    from gmlx.video import encoder as video_encoder
    from gmlx.video import layout, pipeline
    from gmlx.video.vae.reader import load_vae

    with bounded_cache():
        t0 = time.perf_counter()
        with _Peak() as enc_peak:
            handle = video_encoder.load_encoder(
                fam, os.path.expanduser(a.encoder), log=log)
            load_s["encoder"] = time.perf_counter() - t0
            try:
                t1 = time.perf_counter()
                feats = mx.array(handle.encode(a.prompt).features)[None]
                mx.eval(feats)
                done["encode_s"] = time.perf_counter() - t1
            finally:
                handle.release()
        _release()
        text_rows = done["text"] = int(feats.shape[1])

        t0 = time.perf_counter()
        with _Peak() as load_peak:
            pipe = pipeline.load(os.path.expanduser(a.model),
                                 variant=a.variant, log=log)
            load_s["dit"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            table = pipeline.adapter_table(
                pipe, [(os.path.expanduser(p), st) for p, st in adapters])
            mx.eval()
            load_s["adapters"] = time.perf_counter() - t0
        for w, h, frames in shapes:
            marks = [time.perf_counter()]
            with _Peak() as peak:
                xv, xa = pipeline.generate(
                    pipe, feats, frames=frames, width=w, height=h,
                    steps=a.steps, shifts=shifts, seed=a.seed, table=table,
                    on_step=lambda i, n: marks.append(time.perf_counter()))
                mx.eval(xv, xa)
            vshape, ashape = fam.latent_shapes(frames, w, h)
            lay = layout.pack(fam.timeline, text_rows, vshape[2], vshape[3],
                              vshape[4], ashape[3])
            steps_s = [b - x for x, b in zip(marks, marks[1:])]
            denoise = max(peak.bytes, load_peak.bytes)
            runs.append({"shape": (w, h, frames), "latents": (xv, xa),
                         "rows": _rows(lay, text_rows), "step_s": steps_s,
                         "peak": {"encode": enc_peak.bytes,
                                  "denoise": denoise}})
            log(f"[bench] {w}x{h}x{frames}: {int(lay.rows)} rows, step "
                f"first {steps_s[0]:.2f} s, median "
                f"{_median(steps_s[1:]) or steps_s[0]:.2f} s, denoise peak "
                f"{denoise / GiB:.2f} GiB")
        del pipe, table
        _release()

        if a.no_decode:
            return
        # Each decode peak takes its VAE's load, as the denoise takes the
        # DiT's.
        t0 = time.perf_counter()
        with _Peak() as vae_peak:
            avae = load_vae(os.path.expanduser(a.audio_vae),
                            family="h3_audio", dtype=mx.float32)
        load_s["audio_vae"] = time.perf_counter() - t0
        for r in runs:
            with _Peak() as peak:
                t1 = time.perf_counter()
                mx.eval(avae.decode(r["latents"][1]))
                r["audio_s"] = time.perf_counter() - t1
            r["peak"]["audio"] = max(peak.bytes, vae_peak.bytes)
        del avae
        _release()
        t0 = time.perf_counter()
        with _Peak() as vae_peak:
            vae = load_vae(os.path.expanduser(a.vae), family="h3",
                           dtype=mx.float16)
        load_s["vae"] = time.perf_counter() - t0
        for r in runs:
            with _Peak() as peak:
                t1 = time.perf_counter()
                for chunk in vae.decode_frames(r["latents"][0]):
                    mx.eval(chunk)
                    del chunk
                r["decode_s"] = time.perf_counter() - t1
            r["peak"]["decode"] = max(peak.bytes, vae_peak.bytes)
        del vae
        _release()


def cmd_bench(argv: list[str], prog: str = "gmlx video bench") -> int:
    a = _parser(prog).parse_args(argv)
    try:
        return _bench(a)
    except _Usage as e:
        print(f"{prog}: error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 (refusals and failures exit 1)
        print(f"error: {e}", file=sys.stderr)
        return 1
