"""Streaming fit planner.

Prices a MoE GGUF from its header only. The routed experts stream from
disk. Every other tensor is an every-token weight and stays resident.
A box streams the model when the every-token weights and the KV room
fit under the memory ceiling. What is left is the decode arena, and its
share of the experts sets the decode speed. ``gmlx validate`` and
``gmlx doctor`` print this plan; the loader prints the live budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from gmlx.stream.budget import KvRoom

GiB = 1 << 30
_MIN_ARENA = GiB           # decode_feeder refuses a smaller arena

VERDICT_RESIDENT = "resident"
VERDICT_STREAMS = "streams"
VERDICT_PAGE_CACHE = "page_cache"
VERDICT_TOO_BIG = "too_big"

GROUP_LABELS = {
    "attention": "attention",
    "recurrent": "recurrent layers",
    "shared_experts": "shared experts",
    "ffn": "dense ffn and routers",
    "embedding": "embeddings",
    "output": "output head",
    "other": "other",
}


def group_of(name: str) -> str | None:
    """The every-token group of a tensor, or None for a routed-expert
    stack (the tensors ``stream: experts`` serves from disk)."""
    from gmlx.stream.prefetch import _EXPS_RE

    if _EXPS_RE.fullmatch(name):
        return None
    if "shexp" in name:
        return "shared_experts"
    if name.startswith("token_embd"):
        return "embedding"
    if name.startswith("output"):
        return "output"
    if ".ssm_" in name or "linear_attn" in name:
        return "recurrent"
    if ".attn_" in name:
        return "attention"
    if ".ffn_" in name:
        return "ffn"
    return "other"


@dataclass(frozen=True)
class ModelPlan:
    arch: str | None
    total_bytes: int
    expert_bytes: int
    groups: dict
    moe_layers: int
    n_experts: int | None
    experts_per_token: int | None
    ring_bytes: int
    kv_costs: tuple | None
    trained_ctx: int | None

    @property
    def every_token_bytes(self) -> int:
        return self.total_bytes - self.expert_bytes

    @property
    def streamable(self) -> bool:
        return self.expert_bytes > 0

    @property
    def per_token_read_bytes(self) -> int | None:
        """Expert bytes one token reads with a cold arena."""
        if not (self.n_experts and self.experts_per_token):
            return None
        return int(self.expert_bytes * self.experts_per_token / self.n_experts)

    def kv_bytes(self, tokens: int) -> float | None:
        if self.kv_costs is None:
            return None
        from gmlx.serve.mem_preflight import prompt_kv_bytes

        return prompt_kv_bytes(self.kv_costs, tokens)


@dataclass(frozen=True)
class BoxPlan:
    ram_bytes: int
    working_set_bytes: float
    ceiling_bytes: float
    room: KvRoom
    arena_bytes: int
    expert_bytes: int
    ring_fits: bool
    floor_bytes: int
    short_bytes: int
    verdict: str

    @property
    def arena_share(self) -> float | None:
        if self.expert_bytes <= 0:
            return None
        return self.arena_bytes / self.expert_bytes


def _int(v) -> int | None:
    return v if isinstance(v, int) and v > 0 else None


def model_plan(scans, env: dict | None = None) -> ModelPlan:
    """Price a model from its header scans (one per shard, the metadata
    shard first)."""
    kv = scans[0].kv
    arch = kv.get("general.architecture")
    from gmlx.stream.prefetch import _EXPS_RE
    from gmlx.stream.prefill_feeder import ring_bytes

    groups: dict[str, int] = {}
    # layer -> offsets-shaped entries, so the ring is the runtime's formula
    stacks: dict[int, list] = {}
    total = expert = 0
    exps_shape = None
    for hs in scans:
        for t in hs.tensors:
            total += t.nbytes
            m = _EXPS_RE.fullmatch(t.name)
            if m is not None:
                expert += t.nbytes
                stacks.setdefault(int(m.group(1)), []).append(
                    (hs.path, 0, t.nbytes, 0, m.group(2)))
                exps_shape = exps_shape or t.shape
            else:
                g = group_of(t.name)
                groups[g] = groups.get(g, 0) + t.nbytes
    n_experts = _int(kv.get(f"{arch}.expert_count"))
    if n_experts is None and exps_shape and len(exps_shape) == 3:
        n_experts = _int(exps_shape[-1])
    costs = trained = None
    if expert:
        from gmlx.load import loadlog
        from gmlx.serve.capacity import boot_costs

        with loadlog.quiet():
            priced = boot_costs(None, env, scans=scans)
        if priced is not None:
            cfg, costs, _ = priced
            costs = tuple(costs)
            trained = _int(cfg.get("max_position_embeddings"))
    return ModelPlan(
        arch=arch, total_bytes=total, expert_bytes=expert, groups=groups,
        moe_layers=len(stacks), n_experts=n_experts,
        experts_per_token=_int(kv.get(f"{arch}.expert_used_count")),
        ring_bytes=ring_bytes(stacks),
        kv_costs=costs, trained_ctx=trained)


def ceiling_for(ram_bytes: float, ws_bytes: float) -> float:
    """``capacity.ceiling_bytes`` for a box of these sizes."""
    from gmlx.serve.capacity import margin, reserve_bytes

    return min(ws_bytes * (1.0 - margin()), ram_bytes - reserve_bytes(ram_bytes))


def box_plan(model: ModelPlan, *, ram_bytes: int | None = None,
             ws_bytes: float | None = None) -> BoxPlan | None:
    """Fit ``model`` on a box. Defaults to this machine; pass both sizes
    for another one. None when the machine cannot be read."""
    from gmlx.load.memfit import total_ram_bytes
    from gmlx.serve.capacity import classify_weight_share, working_set_bytes
    from gmlx.stream.budget import host_floor_bytes, price_room, transient_bytes

    if ws_bytes is None:
        ws_bytes = working_set_bytes()
    if ram_bytes is None:
        ram_bytes = total_ram_bytes()
    if not ws_bytes or not ram_bytes:
        return None
    ceiling = ceiling_for(ram_bytes, ws_bytes)
    room = price_room(model.kv_costs, model.trained_ctx, ws_bytes,
                      transient_bytes(ws_bytes))
    every = model.every_token_bytes
    left = int(ceiling - every - room.bytes)
    ring_fits = model.ring_bytes <= left
    floor = host_floor_bytes(int(ram_bytes))
    arena = min(max(0, left - (model.ring_bytes if ring_fits else 0) - floor),
                model.expert_bytes)
    if classify_weight_share(model.total_bytes, ram_bytes) != "over":
        verdict = VERDICT_RESIDENT
    elif left < 0:
        verdict = VERDICT_TOO_BIG
    elif arena < _MIN_ARENA:
        verdict = VERDICT_PAGE_CACHE
    else:
        verdict = VERDICT_STREAMS
    return BoxPlan(
        ram_bytes=int(ram_bytes), working_set_bytes=float(ws_bytes),
        ceiling_bytes=ceiling, room=room, arena_bytes=arena,
        expert_bytes=model.expert_bytes, ring_fits=ring_fits,
        floor_bytes=floor,
        short_bytes=max(0, -left), verdict=verdict)


def scan_path(gguf_path: str) -> list:
    """Header scans of every shard of a local file."""
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.preflight import find_split_shards

    return [scan_gguf(p, include_tensors=True) for p in find_split_shards(gguf_path)]


def plan_path(gguf_path: str, env: dict | None = None
              ) -> tuple[ModelPlan, BoxPlan | None] | None:
    """The plan for a local file on this machine, or None when the
    header cannot be read or the file has no routed experts."""
    try:
        model = model_plan(scan_path(gguf_path), env)
    except Exception:
        return None
    if not model.streamable:
        return None
    return model, box_plan(model)


# Rendering shared by validate and doctor
def gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def model_line(m: ModelPlan) -> str:
    detail = f"{m.moe_layers} layers"
    if m.n_experts:
        detail += f", {m.n_experts} experts"
    if m.experts_per_token:
        detail += f", {m.experts_per_token} per token"
    return (f"every-token weights {gb(m.every_token_bytes)}, routed experts "
            f"{gb(m.expert_bytes)} ({detail}), prefill ring {gb(m.ring_bytes)}")


def group_line(m: ModelPlan) -> str:
    parts = [f"{GROUP_LABELS[g]} {gb(n)}" for g, n in
             sorted(m.groups.items(), key=lambda kv: -kv[1]) if n]
    return "every-token by group: " + ", ".join(parts)


def box_summary(m: ModelPlan, b: BoxPlan) -> str:
    """One clause on the outcome, for a doctor row."""
    if b.verdict == VERDICT_RESIDENT:
        return "fits in RAM, streaming is optional"
    if b.verdict == VERDICT_TOO_BIG:
        return (f"every-token weights {gb(m.every_token_bytes)} + KV room "
                f"{gb(b.room.bytes)} exceed the {gb(b.ceiling_bytes)} ceiling "
                f"by {gb(b.short_bytes)}")
    if b.verdict == VERDICT_PAGE_CACHE:
        return (f"no decode arena ({gb(b.arena_bytes)} left after the "
                "ring and the host floor), decode runs from the page cache")
    return f"decode arena {gb(b.arena_bytes)}{share_text(b.arena_share)}"


def share_text(share: float | None) -> str:
    if share is None:
        return ""
    if share < 0.01:
        return " (under 1% of the experts)"
    return f" ({share:.0%} of the experts)"


def box_lines(m: ModelPlan, b: BoxPlan) -> list[str]:
    """The per-box lines of a validate report, verdict last."""
    from gmlx.serve.lifecycle import human_gb

    room = f"KV room {gb(b.room.bytes)}"
    room += (f" at {b.room.depth} tokens" if b.room.priced
             else " (flat, header not priced)")
    lines = [f"this Mac: {human_gb(b.ram_bytes, 0)} RAM, ceiling "
             f"{gb(b.ceiling_bytes)}, {room}, host floor {gb(b.floor_bytes)}"]
    if b.verdict == VERDICT_RESIDENT:
        lines.append("=> the whole file fits in RAM; streaming is optional "
                     f"({box_summary(m, b)} if streamed)")
        return lines
    if b.verdict == VERDICT_TOO_BIG:
        lines.append(f"=> cannot stream: {box_summary(m, b)}; pick a quant "
                     "with smaller every-token tensors")
        return lines
    if b.verdict == VERDICT_PAGE_CACHE:
        lines.append(f"=> streams, but {box_summary(m, b)}; expect slow "
                     "decode")
        return lines
    line = box_summary(m, b)
    read = m.per_token_read_bytes
    if read:
        line += f"; a cold token reads about {gb(read)} of experts"
    lines.append(line)
    if not b.ring_fits:
        lines.append(f"prefill ring {gb(m.ring_bytes)} exceeds the arena; "
                     "prefill runs from the page cache")
    lines.append("=> streams with --stream-experts (server: stream: experts)")
    return lines


def to_dict(m: ModelPlan, b: BoxPlan | None) -> dict:
    out = {
        "arch": m.arch,
        "total_bytes": m.total_bytes,
        "expert_bytes": m.expert_bytes,
        "every_token_bytes": m.every_token_bytes,
        "groups": dict(m.groups),
        "moe_layers": m.moe_layers,
        "n_experts": m.n_experts,
        "experts_per_token": m.experts_per_token,
        "per_token_read_bytes": m.per_token_read_bytes,
        "ring_bytes": m.ring_bytes,
        "trained_ctx": m.trained_ctx,
        "box": None,
    }
    if b is not None:
        box = asdict(b)
        box["arena_share"] = b.arena_share
        out["box"] = box
    return out
