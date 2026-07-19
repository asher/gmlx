#!/usr/bin/env python3
"""Publication-quality SVG charts from serve-bench JSON (pure stdlib).

The three chart grammars behind gmlx's docs/benchmarks.md figures. One
encoding grammar across the set: ENGINE owns color (gmlx = accent, reference
engine = neutral gray), ARM owns line style (solid = speculative/best-
available, dashed = baseline), log-scaled x so "widening with depth" is
geometrically honest.

  panels       per-model: prefill panel over decode panel, shared log-x
  fleet-ratio  hero: gmlx/reference speedup vs depth, one line per model
  mtp-lift     speculative (MTP) / baseline decode lift vs depth

Samples aggregate as MEDIAN over ok requests pooled across rounds; decode uses
full-length samples only (>= 150 output tokens). Model names render verbatim;
retitle with --label OLD=NEW. serve-bench.py invokes this automatically after
a run (see --no-svg there).
"""
import argparse
import json
import math
import os
import statistics
import sys

# ---------------------------------------------------------------- themes

FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        # Okabe-Ito colorblind-safe ordering (pale yellow swapped for a dark
        # gold that reads on white); distinguishable under deutan/protan/tritan.
        "series": ["#0072b2", "#e69f00", "#009e73", "#d55e00",
                   "#cc79a7", "#56b4e9", "#8c6d1f", "#525252"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        # Okabe-Ito, lightened leads for dark ground; yellow usable here.
        "series": ["#56b4e9", "#e69f00", "#009e73", "#d55e00",
                   "#cc79a7", "#0072b2", "#f0e442", "#bdbdbd"],
    },
}


def die(msg):
    sys.stderr.write("plot-bench: error: %s\n" % msg)
    sys.exit(2)


def need(d, key, ctx):
    if not isinstance(d, dict) or key not in d:
        die("missing field '%s' in %s" % (key, ctx))
    return d[key]


# ---------------------------------------------------------------- svg core

def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


class Svg:
    def __init__(self, w, h, theme):
        self.w, self.h, self.t = w, h, theme
        self.parts = []

    def raw(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size, fill, anchor="start", weight=None,
             opacity=None, rotate=None, tabular=False):
        a = ['x="%.1f"' % x, 'y="%.1f"' % y, 'font-size="%.1f"' % size,
             'fill="%s"' % fill]
        if anchor != "start":
            a.append('text-anchor="%s"' % anchor)
        if weight:
            a.append('font-weight="%s"' % weight)
        if opacity is not None:
            a.append('opacity="%.2f"' % opacity)
        if rotate is not None:
            a.append('transform="rotate(%.1f %.1f %.1f)"' % (rotate, x, y))
        if tabular:
            a.append('style="font-variant-numeric: tabular-nums"')
        self.parts.append("<text %s>%s</text>" % (" ".join(a), esc(s)))

    def line(self, x1, y1, x2, y2, stroke, width=1.0, dash=None, opacity=None):
        a = ['x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"' % (x1, y1, x2, y2),
             'stroke="%s"' % stroke, 'stroke-width="%.1f"' % width]
        if dash:
            a.append('stroke-dasharray="%s"' % dash)
        if opacity is not None:
            a.append('opacity="%.2f"' % opacity)
        self.parts.append("<line %s/>" % " ".join(a))

    def rect(self, x, y, w, h, fill, rx=0, opacity=None, title=None):
        a = ['x="%.1f" y="%.1f" width="%.1f" height="%.1f"' % (x, y, w, h),
             'fill="%s"' % fill]
        if rx:
            a.append('rx="%.1f"' % rx)
        if opacity is not None:
            a.append('opacity="%.2f"' % opacity)
        if title:
            self.parts.append("<rect %s><title>%s</title></rect>"
                              % (" ".join(a), esc(title)))
        else:
            self.parts.append("<rect %s/>" % " ".join(a))

    def path(self, d, fill="none", stroke=None, width=None, dash=None,
             linejoin="round", linecap="round", title=None, opacity=None):
        a = ['d="%s"' % d, 'fill="%s"' % fill]
        if stroke:
            a.append('stroke="%s"' % stroke)
        if width:
            a.append('stroke-width="%.1f"' % width)
        if opacity is not None:
            a.append('opacity="%.2f"' % opacity)
        if dash:
            a.append('stroke-dasharray="%s"' % dash)
        a.append('stroke-linejoin="%s" stroke-linecap="%s"' % (linejoin, linecap))
        if title:
            self.parts.append("<path %s><title>%s</title></path>"
                              % (" ".join(a), esc(title)))
        else:
            self.parts.append("<path %s/>" % " ".join(a))

    def circle(self, cx, cy, r, fill, stroke=None, sw=None, title=None):
        a = ['cx="%.1f" cy="%.1f" r="%.1f"' % (cx, cy, r), 'fill="%s"' % fill]
        if stroke:
            a.append('stroke="%s" stroke-width="%.1f"' % (stroke, sw or 1.0))
        if title:
            self.parts.append("<circle %s><title>%s</title></circle>"
                              % (" ".join(a), esc(title)))
        else:
            self.parts.append("<circle %s/>" % " ".join(a))

    def render(self):
        head = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
                'viewBox="0 0 %d %d" font-family="%s">'
                % (self.w, self.h, self.w, self.h, esc(FONT)))
        bg = '<rect x="0" y="0" width="%d" height="%d" fill="%s"/>' % (
            self.w, self.h, self.t["surface"])
        return head + "\n" + bg + "\n" + "\n".join(self.parts) + "\n</svg>\n"


def est_w(s, size):
    return 0.58 * size * len(str(s))


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / max(n, 1)))
    for m in (1, 2, 2.5, 5, 10, 20):
        if span / (step * m) <= n:
            step *= m
            break
    t0 = math.ceil(lo / step) * step
    ticks, v = [], t0
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def fmt_num(v):
    if v >= 1000:
        return "%.0f" % v
    if v >= 100:
        return "%.0f" % v
    if v >= 10:
        return "%.1f" % v
    return "%.2f" % v


def fmt_ratio(v):
    return "%.2fx" % v


def fmt_tick(v):
    return "%g" % round(v, 6)


# ---------------------------------------------------------------- loader

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        die("input file not found: %s" % path)
    except json.JSONDecodeError as e:
        die("not valid JSON: %s (%s)" % (path, e))


# decode_tps / tpot_ms use FULL-LENGTH samples only: llama-server truncates
# gpt-oss (harmony) generation at deep context, and those short samples poison
# the decode median. prefill_tps is measured at TTFT so it uses all ok samples.
FULL_TOK = 150

# depths to exclude from serve charts (set from --drop-depth in main)
_DROP_DEPTHS = set()


def med_cv(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0.0
    med = statistics.median(vals)
    cv = (statistics.pstdev(vals) / med * 100.0) if med and len(vals) > 1 else 0.0
    return med, cv


def parse_serve(path, data):
    """-> list of cell dicts: model, runtime, arm, depth, conc, metric medians."""
    results = need(data, "results", path)
    cells = []
    for key, cell in results.items():
        parts = key.split("|")
        if len(parts) != 5:
            continue  # e.g. trailing "<model>|llama|mtp@3" acceptance blob
        model, runtime, arm, depth_s, conc_s = parts
        try:
            depth, conc = int(depth_s), int(conc_s)
        except ValueError:
            continue
        if depth in _DROP_DEPTHS:
            continue
        samples = need(cell, "samples", "%s cell %s" % (path, key))
        ok = [s for s in samples if s.get("ok")]
        if not ok:
            continue
        full = [s for s in ok if (s.get("output_tokens") or 0) >= FULL_TOK]
        row = {"model": model, "runtime": runtime, "arm": arm,
               "depth": depth, "conc": conc, "n": len(ok), "n_full": len(full)}
        for metric in ("decode_tps", "prefill_tps", "tpot_ms"):
            src = ok if metric == "prefill_tps" else (full or ok)
            row[metric], row[metric + "_cv"] = med_cv([s.get(metric) for s in src])
        cells.append(row)
    if not cells:
        die("no usable result cells in %s" % path)
    return cells


# ---------------------------------------------------------------- filters

def relabel(label, maps):
    for old, new in maps:
        label = label.replace(old, new)
    return label


def keep(label, include, exclude):
    if include and not any(s in label for s in include):
        return False
    if exclude and any(s in label for s in exclude):
        return False
    return True


# ---------------------------------------------------------------- figure

class Fig:
    """Title/subtitle/legend chrome; leaves a plot rect."""

    def __init__(self, args, default_title):
        self.t = THEMES[args.theme]
        if args.palette:
            self.t = dict(self.t, series=[c if c.startswith("#") else "#" + c
                                          for c in args.palette.split(",")])
        self.fs = args.font_size
        # legend font scales up so it stays legible when the SVG is shrunk into
        # an HTML contact-sheet pane.
        self.lfs = args.font_size * getattr(args, "legend_scale", 1.0)
        self.svg = Svg(args.width, args.height, self.t)
        self.args = args
        self.title = args.title if args.title is not None else default_title
        self.subtitle = args.subtitle

    def color(self, i):
        return self.t["series"][i % len(self.t["series"])]

    def layout(self, legend_items, left=None, bottom=None, right=None):
        """legend_items: [(label, color, kind)] kind in line|dash|swatch."""
        fs, t, svg = self.fs, self.t, self.svg
        pad = fs * 1.4
        y = pad + fs * 0.4
        if self.title:
            y += fs * 0.9
            svg.text(pad, y, self.title, fs * 1.45, t["ink"], weight="600")
            y += fs * 0.8
        if self.subtitle:
            y += fs * 0.5
            svg.text(pad, y, self.subtitle, fs * 1.0, t["ink2"])
            y += fs * 0.4
        top = y + fs * 1.2
        pleft = pad + (left if left is not None else fs * 3.4)
        pright = self.svg.w - pad - (right or 0)
        pbottom = self.svg.h - pad - (bottom if bottom is not None else fs * 2.6)
        lpos = self.args.legend
        if not legend_items or lpos == "none":
            return pleft, top, pright, pbottom
        lfs = self.lfs
        item_ws = [lfs * 2.0 + est_w(lbl, lfs) + lfs * 1.4
                   for lbl, *_ in legend_items]
        if lpos == "right":
            lw = max(item_ws) + lfs
            pright = self.svg.w - pad - lw
            ly = top + lfs * 0.5
            for it, _ in zip(legend_items, item_ws):
                self._legend_item(pright + lfs, ly, *it)
                ly += lfs * 1.7
            return pleft, top, pright, pbottom
        # top / bottom: flow-wrap rows
        rows, cur, curw = [], [], 0.0
        avail = self.svg.w - 2 * pad
        for it, w in zip(legend_items, item_ws):
            if cur and curw + w > avail:
                rows.append(cur)
                cur, curw = [], 0.0
            cur.append((it, w))
            curw += w
        rows.append(cur)
        rh = lfs * 1.7
        if lpos == "bottom":
            ly = self.svg.h - pad - rh * (len(rows) - 1) - fs * 0.2
            pbottom -= rh * len(rows)
        else:
            ly = top + fs * 0.3
            top += rh * len(rows) + fs * 0.4
        for row in rows:
            lx = pad
            for it, w in row:
                self._legend_item(lx, ly, *it)
                lx += w
            ly += rh
        return pleft, top, pright, pbottom

    def _legend_item(self, x, y, label, color, kind, shape="circle"):
        fs, svg = self.lfs, self.svg
        if kind == "swatch":
            svg.rect(x, y - fs * 0.75, fs * 0.9, fs * 0.9, color, rx=3)
        else:
            dash = "5,4" if kind == "dash" else None
            svg.line(x, y - fs * 0.3, x + fs * 1.3, y - fs * 0.3, color,
                     width=2.5, dash=dash)
            draw_marker(svg, x + fs * 0.65, y - fs * 0.3, fs * 0.22, shape,
                        color, self.t["surface"], sw=1.5)
        svg.text(x + fs * 1.7, y, label, fs, self.t["ink2"])

    def save(self):
        out = self.args.out
        d = os.path.dirname(out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(out, "w") as f:
            f.write(self.svg.render())
        print("wrote %s" % out)


# ------------------------------------------------------- shared axes

def y_scale(vals, logy, zero=False):
    lo, hi = min(vals), max(vals)
    if logy:
        lo = max(lo, 1e-6)
        l0, l1 = math.log2(lo), math.log2(hi)
        ticks = [2 ** e for e in range(int(math.floor(l0)),
                                       int(math.ceil(l1)) + 1)]
        while len(ticks) > 8:
            ticks = ticks[::2]
        lo, hi = ticks[0], max(ticks[-1], hi)

        def to_f(v):
            return (math.log2(max(v, 1e-6)) - math.log2(lo)) / \
                   (math.log2(hi) - math.log2(lo) or 1.0)
        return ticks, to_f
    if zero:
        lo = 0.0
    pad = (hi - lo) * 0.08 or 1.0
    hi += pad
    if not zero:
        lo = max(0.0, lo - pad)
    ticks = nice_ticks(lo, hi)

    def to_f(v):
        return (v - lo) / (hi - lo)
    return ticks, to_f


def draw_grid_y(fig, px0, py0, px1, py1, ticks, to_f, fmt=fmt_tick):
    svg, t, fs = fig.svg, fig.t, fig.fs
    for v in ticks:
        y = py1 - to_f(v) * (py1 - py0)
        if y < py0 - 1 or y > py1 + 1:
            continue
        svg.line(px0, y, px1, y, t["grid"], width=1)
        svg.text(px0 - fs * 0.5, y + fs * 0.35, fmt(v), fs * 0.9, t["muted"],
                 anchor="end", tabular=True)
    svg.line(px0, py1, px1, py1, t["axis"], width=1.2)


def draw_marker(svg, cx, cy, r, shape, color, stroke, sw=2):
    """Filled marker: 'triangle' (area-matched to the circle, centroid on cy)
    or 'circle' (default). Shape is a redundant channel alongside color."""
    if shape == "triangle":
        rr = r * 1.5
        svg.path("M %.1f,%.1f L %.1f,%.1f L %.1f,%.1f Z" % (
            cx, cy - rr, cx + rr * 0.92, cy + rr * 0.5, cx - rr * 0.92,
            cy + rr * 0.5), fill=color, stroke=stroke, width=sw * 0.75)
    else:
        svg.circle(cx, cy, r, color, stroke=stroke, sw=sw)


# ------------------------------------------------------- chart grammar

REF_NAME = {"llama": "llama.cpp", "ds4": "ds4-server"}


def _is_dark(t):
    return t["ink"].lower() in ("#ffffff", "#fff")


def gmlx_accent(t):
    return t["series"][0]                 # #0072b2 light / #56b4e9 dark


def ref_gray(t):
    return "#9a9a95" if _is_dark(t) else "#6f6f6d"


def fmt_depth_k(d):
    if d < 1000:
        return str(d)
    if d < 10000:
        return ("%.1fk" % (d / 1000.0)).replace(".0k", "k")
    return "%dk" % round(d / 1000.0)


def log_x(depths, x0, x1, pad_frac=0.06):
    pos = sorted({d for d in depths if d > 0})
    lo, hi = pos[0], pos[-1]
    l0, l1 = math.log10(lo), math.log10(hi)
    span = (l1 - l0) or 1.0
    pad = pad_frac * (x1 - x0)
    ax0, ax1 = x0 + pad, x1 - pad

    def fx(d):
        return ax0 + (math.log10(max(d, lo)) - l0) / span * (ax1 - ax0)
    return fx


def _lines_panel(fig, rect, series, ylabel, depths, fx, draw_x=True,
                 ratio=False, refline=None, zero=True, label_last_solid=False,
                 valfmt=None):
    """Draw one line panel into rect=(px0,py0,px1,py1). series entries:
    {pts:{depth:(val,cv)}, color, dash(bool), shape?}."""
    svg, t, fs = fig.svg, fig.t, fig.fs
    px0, py0, px1, py1 = rect
    vals = [v for s in series for v, _ in s["pts"].values()]
    if refline is not None:
        vals.append(refline)
    if not vals:
        return
    ticks, to_f = y_scale(vals, False, zero=zero)
    fmt = (lambda v: fmt_tick(v) + "x") if ratio else fmt_tick
    draw_grid_y(fig, px0, py0, px1, py1, ticks, to_f, fmt=fmt)
    if refline is not None:
        yr = py1 - to_f(refline) * (py1 - py0)
        svg.line(px0, yr, px1, yr, t["ink2"], width=1.6, dash="2,3")
        svg.text(px1 - fs * 0.2, yr - fs * 0.35, fmt(refline), fs * 0.8,
                 t["muted"], anchor="end")
    last_lx = None
    for d in depths:
        x = fx(d)
        svg.line(x, py1, x, py1 + (fs * 0.35 if draw_x else 0), t["axis"],
                 width=1)
        if draw_x:
            lab = fmt_depth_k(d)
            # skip a tick label that would collide with the previous one
            if last_lx is not None and x - last_lx < est_w(lab, fs * 0.85) * 0.7:
                continue
            svg.text(x, py1 + fs * 1.4, lab, fs * 0.85, t["muted"],
                     anchor="middle", tabular=True)
            last_lx = x
    fnum = valfmt or (fmt_ratio if ratio else fmt_num)
    # direct-label the final point of the strongest gmlx series in this panel
    lbl_s = None
    if label_last_solid:
        gm = [s for s in series if s.get("_gmlx") and s["pts"]]
        if gm:
            lbl_s = max(gm, key=lambda s: s["pts"][max(s["pts"])][0])
    for s in series:
        pts = sorted(s["pts"].items())
        seq = [(fx(d), py1 - to_f(v) * (py1 - py0)) for d, (v, _) in pts]
        w = s.get("width", 2.4)
        op = s.get("opacity")
        if len(seq) > 1:
            svg.path("M" + " L".join("%.1f,%.1f" % xy for xy in seq),
                     stroke=s["color"], width=w, opacity=op,
                     dash="6,5" if s["dash"] else None)
        if not s.get("nomark"):
            for (x, y) in seq:
                draw_marker(svg, x, y, fs * 0.26, s.get("shape", "circle"),
                            s["color"], t["surface"], sw=2)
        if s is lbl_s and seq:
            x, y = seq[-1]
            _, (v, _) = pts[-1]
            svg.text(x + fs * 0.5, y - fs * 0.45, fnum(v), fs * 0.85,
                     s["color"], anchor="start", weight="600", tabular=True)
    svg.text(px0 - fs * 3.0, (py0 + py1) / 2, ylabel, fs * 0.95, t["muted"],
             anchor="middle", rotate=-90)
    return to_f          # so callers can place direct labels on the same scale


def _panel_split(fig, top, bottom, pleft, pright, cap_lines=0):
    """Two stacked panels sharing x; reserve caption space below."""
    fs = fig.fs
    cap_h = cap_lines * fs * 1.35 + (fs * 0.6 if cap_lines else 0)
    xtick_h = fs * 2.6                    # room for shared x labels + title
    usable = bottom - top - cap_h
    gap = fs * 3.0
    ph = (usable - gap - xtick_h) / 2
    r_top = (pleft, top, pright, top + ph)
    r_bot = (pleft, top + ph + gap, pright, top + ph + gap + ph)
    return r_top, r_bot, top + ph + gap + ph + xtick_h


def _serve_cells(args):
    cells = []
    for path in args.inputs:
        data = load_json(path)
        if not (isinstance(data, dict) and "results" in data):
            die("%s: expected serve-bench JSON" % path)
        for c in parse_serve(path, data):
            if c["conc"] == args.concurrency:
                cells.append(c)
    return cells


def chart_panels(args):
    """Per-model: prefill panel stacked above decode panel, shared log-x.
    Engine=color, arm=style. One model per invocation; --model picks one out
    of a multi-model run JSON."""
    cells = _serve_cells(args)
    if args.model:
        cells = [c for c in cells if c["model"] == args.model]
    models = sorted({c["model"] for c in cells})
    if len(models) != 1:
        die("panels expects exactly one model, got %s (pick with --model)"
            % (models or "no cells"))
    model = models[0]
    ref_rt = "ds4" if any(c["runtime"] == "ds4" for c in cells) else "llama"
    t = THEMES[args.theme]
    # arm owns style GLOBALLY: dashed = baseline whenever the model has a
    # speculative arm for that engine (consistent across both panels).
    spec_rt = {rt: any(c["runtime"] == rt and c["arm"] != "baseline"
                       for c in cells) for rt in ("gmlx", ref_rt)}

    def build(metric, arms=None):
        by = {}                            # (rt,arm) -> {depth:(v,cv)}
        for c in cells:
            if c["runtime"] not in ("gmlx", ref_rt) or c[metric] is None:
                continue
            if arms is not None and c["arm"] not in arms:
                continue
            by.setdefault((c["runtime"], c["arm"]), {})[c["depth"]] = (
                c[metric], c[metric + "_cv"])
        out = []
        for (rt, arm), pts in by.items():
            # PER-PANEL style: dashed only when THIS panel also shows a
            # speculative arm for this engine; a lone-baseline panel is solid.
            panel_spec = any(a != "baseline" for (r, a) in by if r == rt)
            out.append({"rt": rt, "arm": arm, "pts": pts,
                        "color": gmlx_accent(t) if rt == "gmlx"
                        else ref_gray(t), "_gmlx": rt == "gmlx",
                        "dash": arm == "baseline" and panel_spec})
        return out

    pre = build("prefill_tps", arms={"baseline"})   # prefill: base-vs-base
    dec = build("decode_tps")                        # decode: base + speculative
    depths = sorted({d for s in pre + dec for d in s["pts"]})
    if not depths:
        die("no data for %s" % model)
    # legend: one entry per (engine, arm) drawn. An engine with no speculative
    # arm reads as best-available -> bare engine name; engines with both show
    # "<eng> speculative (MTP)" (solid) and "<eng> baseline" (dashed).
    seen, items = set(), []
    for s in dec + pre:
        key = (s["rt"], s["arm"])
        if key in seen:
            continue
        seen.add(key)
        eng = "gmlx" if s["rt"] == "gmlx" else REF_NAME.get(ref_rt, ref_rt)
        if not spec_rt[s["rt"]]:
            lbl = eng
        elif s["arm"] == "baseline":
            lbl = "%s baseline" % eng
        else:
            lbl = "%s speculative (MTP)" % eng
        items.append((lbl, s["color"], "dash" if s["dash"] else "line",
                      "circle"))
    fig = Fig(args, "%s -- prefill & decode vs KV depth"
              % relabel(model, args.labels))
    pleft, top, pright, pbottom = fig.layout(items, left=fig.fs * 3.6)
    fx = log_x(depths, pleft, pright)
    r_top, r_bot, xlab_y = _panel_split(fig, top, pbottom, pleft, pright)
    # "(baseline)" on the prefill axis: prefill is base-vs-base, so its solid
    # lines are baselines -- the qualifier stops a reader mapping them to the
    # shared legend's "speculative (MTP)" entry.
    _lines_panel(fig, r_top, pre, "prefill tok/s (baseline)", depths, fx,
                 draw_x=False, label_last_solid=True)
    _lines_panel(fig, r_bot, dec, "decode tok/s", depths, fx, draw_x=True,
                 label_last_solid=True)
    fig.svg.text((pleft + pright) / 2, xlab_y, "KV depth (tokens, log scale)",
                 fig.fs * 0.95, fig.t["muted"], anchor="middle")
    fig.save()


def _ratio_series(cells, num_rt, den_rt, metric, arm):
    """{model: {depth: ratio}} for num_rt/den_rt at the given arm."""
    idx = {}
    for c in cells:
        if c["arm"] == arm and c[metric] is not None:
            idx[(c["model"], c["runtime"], c["depth"])] = c[metric]
    out = {}
    for (m, rt, d), v in idx.items():
        if rt != num_rt:
            continue
        den = idx.get((m, den_rt, d))
        if den:
            out.setdefault(m, {})[d] = v / den
    return out


def _direct_labels(fig, rect, entries):
    """entries: [(y, text, color)] placed at the right edge, de-collided."""
    svg, fs = fig.svg, fig.fs
    px0, py0, px1, py1 = rect
    entries = sorted(entries, key=lambda e: e[0])
    minsp = fs * 1.02
    for i in range(1, len(entries)):
        y0 = entries[i - 1][0]
        if entries[i][0] - y0 < minsp:
            entries[i] = (y0 + minsp, entries[i][1], entries[i][2])
    if entries and entries[-1][0] > py1:            # overflowed bottom: shift up
        shift = entries[-1][0] - py1
        entries = [(y - shift, tx, c) for (y, tx, c) in entries]
    for y, tx, c in entries:
        svg.text(px1 + fs * 0.5, y + fs * 0.32, tx, fs * 0.82, c,
                 anchor="start", weight="600")


def chart_fleet_ratio(args):
    """Hero: gmlx/reference throughput ratio vs depth, one line per model.
    Two panels -- prefill (base/base) and decode (speculative/speculative,
    MTP models only). Direct-labeled, no legend box."""
    vs = args.vs

    def _mlabel(m):
        return relabel(m, args.labels)
    cells = [c for c in _serve_cells(args) if keep(c["model"], args.include,
                                                   args.exclude)]
    pre = _ratio_series(cells, "gmlx", vs, "prefill_tps", "baseline")
    dec = {}
    for m, pts in _ratio_series(cells, "gmlx", vs, "decode_tps", "mtp@3").items():
        dec[m] = pts
    if not pre:
        die("no gmlx/%s prefill baseline pairs found" % vs)
    t = THEMES[args.theme]
    # 3 exemplar hues (Okabe-Ito, CVD-safe); the rest of the fleet is ghosted.
    # Deliberately skip series[0] (the gmlx engine accent) so blue keeps one
    # meaning across the doc -- these are rank-keyed, not entity-keyed.
    EX = [t["series"][1], t["series"][2], t["series"][3]]  # gold, green, verm.
    depths = sorted({d for pts in pre.values() for d in pts}
                    | {d for pts in dec.values() for d in pts})
    n_pre, n_dec = len(pre), len(dec)
    fig = Fig(args, "gmlx vs %s -- throughput speedup across the fleet"
              % REF_NAME.get(vs, vs))

    def pick_exemplars(dct):
        # fastest / median / slowest by each model's deepest-depth ratio
        rank = sorted(dct, key=lambda m: dct[m][max(dct[m])], reverse=True)
        idx = list(range(len(rank))) if len(rank) <= 3 \
            else [0, len(rank) // 2, len(rank) - 1]
        return {rank[j]: EX[k] for k, j in enumerate(idx)}

    ex_pre, ex_dec = pick_exemplars(pre), pick_exemplars(dec)

    def exlabel(dct, m):
        return "%s %s" % (_mlabel(m), fmt_ratio(dct[m][max(dct[m])]))
    rmargin = max(est_w(exlabel(d, m), fig.fs * 0.82)
                  for d, ex in ((pre, ex_pre), (dec, ex_dec)) for m in ex)
    pleft, top, pright, pbottom = fig.layout([], left=fig.fs * 3.6,
                                             right=rmargin + fig.fs * 1.2)
    fx = log_x(depths, pleft, pright)
    r_top, r_bot, xlab_y = _panel_split(fig, top, pbottom, pleft, pright,
                                        cap_lines=2)

    def render(dct, ex, rect, ylabel, draw_x):
        ser = []
        for m in sorted(dct, key=lambda m: m not in ex):   # ghosts first
            is_ex = m in ex
            ser.append({"pts": {d: (v, None) for d, v in dct[m].items()},
                        "color": ex[m] if is_ex else t["muted"], "dash": False,
                        "width": 2.8 if is_ex else 1.3,
                        "opacity": None if is_ex else 0.32,
                        "nomark": not is_ex, "_gmlx": False})
        # crop: floor just below the data min (1.0 included so 1x stays visible)
        to_f = _lines_panel(fig, rect, ser, ylabel, depths, fx, draw_x=draw_x,
                            ratio=True, refline=1.0, zero=False)
        _, py0, _, py1 = rect
        ents = [(py1 - to_f(dct[m][max(dct[m])]) * (py1 - py0),
                 exlabel(dct, m), c) for m, c in ex.items()]
        _direct_labels(fig, rect, ents)

    render(pre, ex_pre, r_top, "prefill speedup", False)
    if dec:
        render(dec, ex_dec, r_bot, "decode speedup", True)
    fig.svg.text((pleft + pright) / 2, xlab_y, "KV depth (tokens, log scale)",
                 fig.fs * 0.95, fig.t["muted"], anchor="middle")
    cap1 = ("Each ghosted line is one model; fastest / median / slowest are "
            "highlighted and labeled.")
    cap2 = ("Prefill: baseline, %d models. Decode: speculative (MTP), %d models "
            "(models with no MTP arm on both engines appear in prefill only)."
            % (n_pre, n_dec))
    fig.svg.text(pleft, xlab_y + fig.fs * 1.4, cap1, fig.fs * 0.78,
                 fig.t["muted"], anchor="start")
    fig.svg.text(pleft, xlab_y + fig.fs * 2.6, cap2, fig.fs * 0.78,
                 fig.t["muted"], anchor="start")
    fig.save()


def chart_mtp_lift(args):
    """MTP decode / baseline decode (same engine) vs depth, one line per MTP
    model. gmlx accent; optional llama.cpp own-lift in gray."""
    def _mlabel(m):
        return relabel(m, args.labels)
    cells = [c for c in _serve_cells(args) if keep(c["model"], args.include,
                                                   args.exclude)]
    gm = _mtp_lift(cells, "gmlx")
    ll = _mtp_lift(cells, "llama") if args.with_ref else {}
    if not gm:
        die("no gmlx MTP/baseline pairs found")
    t = THEMES[args.theme]
    pal = t["series"]
    order = sorted(gm, key=lambda m: gm[m][max(gm[m])], reverse=True)
    color = {m: pal[i % len(pal)] for i, m in enumerate(order)}
    depths = sorted({d for pts in gm.values() for d in pts})
    fig = Fig(args, "Speculative (MTP) decode lift vs KV depth")
    rmargin = max(est_w(_mlabel(m), fig.fs * 0.82) for m in order)
    # legend conveys the style grammar only; per-model identity is direct-labeled
    items = [("gmlx (per-model color)", t["ink2"], "line", "circle")]
    if ll:
        items.append(("llama.cpp (gray)", ref_gray(t), "dash", "circle"))
    pleft, top, pright, pbottom = fig.layout(
        items, left=fig.fs * 3.6, right=rmargin + fig.fs * 1.2)
    fx = log_x(depths, pleft, pright)
    ser = []
    for m in order:
        if m in ll:                            # gray dashed ref lift (recedes)
            ser.append({"pts": {d: (v, None) for d, v in ll[m].items()},
                        "color": ref_gray(t), "dash": True, "width": 1.4,
                        "opacity": 0.55, "nomark": True, "_gmlx": False})
    for m in order:                            # gmlx per-model colored on top
        ser.append({"pts": {d: (v, None) for d, v in gm[m].items()},
                    "color": color[m], "dash": False, "_gmlx": False})
    py1 = pbottom
    # crop: floor just below the data min (gm + gray ll + 1.0 all in scale)
    to_f = _lines_panel(fig, (pleft, top, pright, pbottom), ser,
                        "decode lift (MTP / baseline)", depths, fx, draw_x=True,
                        ratio=True, refline=1.0, zero=False)
    ents = [(py1 - to_f(gm[m][max(gm[m])]) * (py1 - top),
             _mlabel(m), color[m]) for m in order]
    _direct_labels(fig, (pleft, top, pright, pbottom), ents)
    fig.svg.text((pleft + pright) / 2, pbottom + fig.fs * 2.4,
                 "KV depth (tokens, log scale)", fig.fs * 0.95,
                 fig.t["muted"], anchor="middle")
    fig.save()


def _mtp_lift(cells, rt):
    base = {(c["model"], c["depth"]): c["decode_tps"] for c in cells
            if c["runtime"] == rt and c["arm"] == "baseline"
            and c["decode_tps"] is not None}
    out = {}
    for c in cells:
        if c["runtime"] == rt and c["arm"] != "baseline" \
                and c["decode_tps"] is not None:
            b = base.get((c["model"], c["depth"]))
            if b:
                out.setdefault(c["model"], {})[c["depth"]] = c["decode_tps"] / b
    return out


# ---------------------------------------------------------------- cli

def lbl_pair(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError("expected old=new")
    return tuple(s.split("=", 1))


def add_common(p):
    p.add_argument("inputs", nargs="+", help="serve-bench JSON file(s)")
    p.add_argument("--out", required=True, help="output .svg path")
    p.add_argument("--title", help="chart title (default: per-chart)")
    p.add_argument("--subtitle", help="subtitle line")
    p.add_argument("--width", type=int, default=1200, help="px (default 1200)")
    p.add_argument("--height", type=int, default=675, help="px (default 675)")
    p.add_argument("--font-size", type=float, default=15.0,
                   help="base px (default 15)")
    p.add_argument("--legend-scale", type=float, default=1.35,
                   help="legend font x base (default 1.35)")
    p.add_argument("--theme", choices=("light", "dark"), default="light",
                   help="default light")
    p.add_argument("--palette", help="series colors, comma hex list")
    p.add_argument("--include", action="append", default=[],
                   help="keep series containing substring (repeatable)")
    p.add_argument("--exclude", action="append", default=[],
                   help="drop series containing substring (repeatable)")
    p.add_argument("--label", dest="labels", action="append", default=[],
                   type=lbl_pair, metavar="OLD=NEW",
                   help="rename substring in labels (repeatable)")
    p.add_argument("--drop-depth", type=int, action="append", default=[],
                   metavar="D", help="exclude serve cells at depth D (repeatable)")
    p.add_argument("--legend", choices=("top", "bottom", "right", "none"),
                   default="top", help="default top")
    p.add_argument("--concurrency", type=int, default=1,
                   help="serve-bench c filter (default 1)")


def main():
    ap = argparse.ArgumentParser(
        prog="plot-bench.py",
        description="Render serve-bench SVG charts (stdlib only, no deps).",
        epilog="The x axis is log-scaled KV depth, so depth-0 cells cannot be "
               "placed on it -- pass --drop-depth 0 (serve-bench's SVG hook "
               "does).")
    sub = ap.add_subparsers(dest="chart", required=True)

    p = sub.add_parser("panels",
                       help="per-model: prefill panel over decode panel, "
                            "shared log-x (engine=color, arm=style)")
    add_common(p)
    p.add_argument("--model", help="model name filter (required when the "
                                   "JSON holds several models)")
    p.set_defaults(fn=chart_panels)

    p = sub.add_parser("fleet-ratio",
                       help="hero: gmlx/reference speedup vs depth, prefill+"
                            "decode panels, one direct-labeled line per model")
    add_common(p)
    p.add_argument("--vs", default="llama",
                   help="reference runtime in the denominator (default llama)")
    p.set_defaults(fn=chart_fleet_ratio)

    p = sub.add_parser("mtp-lift",
                       help="speculative(MTP)/baseline decode lift vs depth, "
                            "one line per MTP model")
    add_common(p)
    p.add_argument("--with-ref", action="store_true",
                   help="also plot llama.cpp own mtp/base lift (dashed)")
    p.set_defaults(fn=chart_mtp_lift)

    args = ap.parse_args()
    _DROP_DEPTHS.update(getattr(args, "drop_depth", []) or [])
    if args.palette:
        for c in args.palette.split(","):
            cc = c.lstrip("#")
            if len(cc) not in (3, 6) or any(x not in "0123456789abcdefABCDEF"
                                            for x in cc):
                die("bad --palette color: %s" % c)
    args.fn(args)


if __name__ == "__main__":
    main()
