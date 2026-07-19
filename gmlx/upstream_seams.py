"""Contract registry for every upstream symbol gmlx patches or deep-imports.

gmlx monkeypatches ~30 private symbols across mlx-vlm and mlx-lm and
deep-imports model internals. Those seams are guarded structurally (try/except
ImportError + idempotence flags), so upstream renames or rewrites fail
silently - stock behavior quietly returns, or a confusing error surfaces far
downstream (mlx-vlm 0.6.4 vendoring switch_layers turned into a gather_mm
shape error at MoE prefill). This module makes the contract explicit:

- ``SEAMS`` declares each (module, attr) we touch, why, and whether a missing
  seam must abort install (``critical``) or only costs an acceleration.
- ``upstream_seams.json`` pins a source fingerprint for each seam, captured
  at the qualified upstream versions. ``tests/test_upstream_seams.py`` fails
  with the symbol's name when upstream drifts under the pin.
- ``check_upstream_versions()`` is the runtime gate: below-floor upstream
  versions raise with an actionable message (stale venvs resolve pyproject
  constraints exactly once - a floor pin alone cannot catch them); versions
  newer than the qualified set warn once.

On a deliberate upstream bump: re-audit the drifted seams, then
``python -m gmlx.upstream_seams --regen`` (see docs/upstream-upgrades.md).
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass

_JSON_PATH = os.path.join(os.path.dirname(__file__), "upstream_seams.json")

# Hard floors: below these, core wiring (switch_layers geometry, cache
# protocol, server engine) predates what gmlx targets; refuse to run.
FLOORS = {
    "mlx": "0.31",
    "mlx-lm": "0.31",
    "mlx-vlm": "0.6.3",
}


@dataclass(frozen=True)
class Seam:
    module: str            # upstream import path
    attr: str              # dotted attribute path; "" = module existence only
    used_by: str           # our patch/import site (for the failure message)
    critical: bool = False  # True: a missing seam must abort, not degrade
    module_optional: bool = False  # module absent on some qualified versions
    attr_optional: bool = False    # attr absent on some qualified versions


SEAMS: tuple[Seam, ...] = (
    # --- qwen3_5 MTP-target verify seams (loader + qwen35_verify_fold) ---
    Seam("mlx_vlm.models.qwen3_5.language",
         "_target_verify_left_padded_attention",
         "gdn_patches._patch_batched_verify_sdpa / qwen35_verify_fold"),
    Seam("mlx_vlm.models.qwen3_5.language", "_target_verify_linear",
         "gdn_patches._patch_bf16_verify_linear"),
    Seam("mlx_vlm.models.qwen3_5.language", "scaled_dot_product_attention",
         "qwen35_verify_fold (B>=2 left-padded fold)"),
    Seam("mlx_vlm.models.qwen3_5.language", "Qwen3_5Model.__call__",
         "gdn_patches._patch_qwen35_empty_sequence_guard", critical=True),
    Seam("mlx_vlm.models.qwen3_5.language", "Qwen3_5GatedDeltaNet.__call__",
         "gdn_patches._patch_gated_delta_fused_verify"),
    Seam("mlx_vlm.models.qwen3_5.gated_delta", "gated_delta_ops",
         "gdn_patches._patch_mlxvlm_gated_delta_tiled_v", critical=True),
    Seam("mlx_vlm.models.qwen3_5.gated_delta", "gated_delta_kernel",
         "gdn_patches._patch_mlxvlm_gated_delta_tiled_v", critical=True),
    Seam("mlx_vlm.models.qwen3_5.gated_delta", "_gated_delta_with_states_ops",
         "gdn_patches._patch_mlxvlm_gated_delta_tiled_v", critical=True),
    Seam("mlx_vlm.models.qwen3_5.gated_delta", "_gated_delta_state_ops",
         "gdn_patches._patch_mlxvlm_gated_delta_tiled_v", critical=True),
    # --- speculative / AR batch engine (spec_engine owns these methods) ---
    Seam("mlx_vlm.generate.ar", "BatchGenerator.__init__",
         "spec_engine._install_apc_manager_stash", critical=True),
    Seam("mlx_vlm.generate.ar", "PromptProcessingBatch.prompt_step",
         "spec_engine.install_full_prompt_mtp_prefill", critical=True),
    Seam("mlx_vlm.generate.ar", "PromptProcessingBatch.generate",
         "spec_engine.install_full_prompt_mtp_prefill", critical=True),
    Seam("mlx_vlm.generate.ar", "SpeculativeGenerationBatch.next",
         "spec_engine.install_continuous_batch_admission", critical=True),
    Seam("mlx_vlm.generate.ar", "run_speculative_server_rounds",
         "spec_engine.install_owned_spec_engine", critical=True),
    Seam("mlx_vlm.speculative.utils", "make_speculative_prompt_cache",
         "spec_engine.install_spec_kv_quant (B=1 KV_BITS)", critical=True),
    Seam("mlx_vlm.generate.ar", "BatchGenerator._apc_pick_for",
         "spec_engine._bind_l1_view (L1 APC helpers)"),
    Seam("mlx_vlm.generate.ar", "BatchGenerator._apc_exact_checkpoint_len",
         "spec_engine._bind_l1_view (L1 APC helpers)"),
    Seam("mlx_vlm.generate.ar", "BatchGenerator._apc_extra_hash",
         "spec_engine._bind_l1_view (L1 APC helpers)"),
    # --- server engine ---
    Seam("mlx_vlm.server.generation", "run_speculative_server_rounds",
         "spec_engine.install_owned_spec_engine", critical=True),
    Seam("mlx_vlm.server.generation", "load_model_resources",
         "server_bridge_vlm (GGUF model resource loader)", critical=True),
    Seam("mlx_vlm.server.generation", "ResponseGenerator._make_sampler",
         "server_patches.install_fast_sampler"),
    Seam("mlx_vlm.server.generation", "GenerationArguments.to_template_kwargs",
         "server_patches (chat_template_kwargs transform)"),
    Seam("mlx_vlm.server.runtime", "runtime",
         "residency._RuntimeProxy wrap", critical=True),
    Seam("mlx_vlm.server.app", "app",
         "server_patches route surgery / server.py uvicorn root",
         critical=True),
    Seam("mlx_vlm.speculative.drafters", "load_drafter",
         "server_bridge_vlm (in-memory MTP drafter injection)", critical=True),
    Seam("mlx_vlm.utils", "get_model_path",
         "server_patches (HF download gate)", critical=True),
    Seam("mlx_vlm.utils", "StoppingCriteria.__call__",
         "server_patches (ignore-EOS)"),
    # --- tool-parser registry (hy_v3_tools.ensure_registered) ---
    Seam("mlx_vlm.tool_parsers", "_TEMPLATE_MARKERS",
         "hy_v3_tools.ensure_registered (Hy3 marker prepend)"),
    Seam("mlx_vlm.tool_parsers", "load_tool_module",
         "hy_v3_tools (sys.modules graft resolves through it)"),
    # --- APC internals (server_patches lone-harvest + apc_pooling) ---
    Seam("mlx_vlm.apc", "harvest_blocks_from_batch_cache",
         "server_patches.install_apc_lone_harvest", critical=True),
    Seam("mlx_vlm.apc", "_clone_layer_major_kv_cache_for_apc",
         "server_patches.install_apc_lone_harvest", critical=True),
    Seam("mlx_vlm.apc", "_sequence_hash",
         "server_patches.install_apc_lone_harvest", critical=True),
    Seam("mlx_vlm.apc", "APCExactCacheEntry",
         "server_patches.install_apc_lone_harvest", critical=True),
    Seam("mlx_vlm.apc", "_hash_tokens",
         "server_patches.install_apc_lone_harvest", critical=True),
    Seam("mlx_vlm.apc", "_DiskLayerMajorBlock",
         "server_patches.install_apc_lone_harvest", critical=True),
    Seam("mlx_vlm.apc", "_copy_mlx_array",
         "server_patches.install_apc_lone_harvest", critical=True),
    Seam("mlx_vlm.apc", "SEED_PARENT_HASH",
         "server_patches.install_apc_lone_harvest", critical=True),
    Seam("mlx_vlm.apc", "_cache_entry_supports_exact_apc",
         "apc_pooling (PoolingCache exact-APC predicate)", critical=True),
    Seam("mlx_vlm.apc", "_merge_exact_cache_entries",
         "apc_pooling (PoolingCache merge arm)", critical=True),
    Seam("mlx_vlm.apc", "_read_safetensors_tensor",
         "apc_pooling (disk-tier zero-width spill)", critical=True),
    Seam("mlx_vlm.apc", "_clone_cache_entry_for_apc",
         "apc_pooling", critical=True),
    Seam("mlx_vlm.apc", "_safetensors_dtype_info",
         "apc_pooling", critical=True),
    # <= 0.6.3 the wrapper delegates to this mlx-lm alias; 0.6.4 inlined it.
    Seam("mlx_vlm.generate.common", "mlx_maybe_quantize_kv_cache",
         "apc_pooling (rotating-safe KV-quant replacement, <=0.6.3 arm)",
         critical=True, attr_optional=True),
    Seam("mlx_vlm.generate", "maybe_quantize_kv_cache",
         "apc_pooling (rotating-safe KV-quant replacement)", critical=True),
    # --- cache classes (dual-origin via gmlx.cache_compat; on <= 0.6.3
    # the vlm module re-exports mlx-lm's classes, so these fingerprint the
    # same source) ---
    Seam("mlx_vlm.models.cache", "KVCache",
         "cache_compat (ckpt sidecar tails, snapshots)", critical=True),
    Seam("mlx_vlm.models.cache", "RotatingKVCache",
         "cache_compat (ds4 make_cache, rollback attach, spec_helpers)",
         critical=True),
    Seam("mlx_vlm.models.cache", "CacheList",
         "cache_compat (ds4 make_cache, prefix_cache snapshots)",
         critical=True),
    Seam("mlx_vlm.models.cache", "ArraysCache",
         "cache_compat (ckpt_supported, prefix_cache snapshots)"),
    Seam("mlx_vlm.models.cache", "BatchKVCache",
         "mtp_drafter / cache_snapshot row round-trip"),
    Seam("mlx_vlm.models.cache", "BatchRotatingKVCache",
         "cache_compat (rollback attach, safe KV-quant exclusion)"),
    Seam("mlx_vlm.models.cache", "BufferedRotatingKVCache",
         "spec_helpers rollback-slack wrap", critical=True),
    Seam("mlx_vlm.models.cache", "make_prompt_cache",
         "chat MTP cache construction", critical=True),
    # 0.6.4 replaced the runtime's model_cache dict with a registry;
    # residency and the wire tests speak both shapes.
    Seam("mlx_vlm.server.runtime", "ModelCacheRegistry",
         "residency pool / test_wire_contract fake loader",
         attr_optional=True),
    # --- vendored switch_layers (exists only on mlx-vlm >= 0.6.4) ---
    Seam("mlx_vlm.models.switch_layers", "SwitchLinear.__call__",
         "modules.switch_layer_types (dual-origin leaf swap)",
         critical=True, module_optional=True),
    Seam("mlx_vlm.models.switch_layers", "SwitchGLU.__call__",
         "modules.switch_layer_types (dual-origin fused GLU)",
         critical=True, module_optional=True),
    # --- mlx-lm seams ---
    Seam("mlx_lm.models.switch_layers", "SwitchLinear.__call__",
         "modules leaf swap + fused subclass signature contract",
         critical=True),
    Seam("mlx_lm.models.switch_layers", "SwitchGLU.__call__",
         "modules.install_fused_moe_glu / loader.install_expert_streaming",
         critical=True),
    Seam("mlx_lm.server", "ModelProvider._load",
         "server_bridge_lm (GGUF ModelProvider)", critical=True),
    Seam("mlx_lm.generate", "maybe_quantize_kv_cache",
         "apc_pooling", critical=True),
    Seam("mlx_lm.models.cache", "RotatingKVCache",
         "rotating_cache_fix / prefix_cache snapshots"),
    Seam("mlx_lm.models.base", "create_causal_mask", "rotating_cache_fix"),
    Seam("mlx_lm.models.hunyuan", "MoeBlock",
         "loader._patch_hunyuan_norm_topk"),
    Seam("mlx_lm.models.deepseek_v32", "MoEGate",
         "dsv32_patches (fp32 router patch)"),
)


def _resolve(seam: Seam):
    """('ok', obj) | ('missing-module', None) | ('missing-attr', None).

    Modules resolve through import; ``sys.modules`` wins for already-loaded
    ones, which matters for e.g. mlx_vlm.server.runtime (the package
    __init__ shadows the submodule attribute with the runtime instance).
    """
    try:
        mod = sys.modules.get(seam.module) or importlib.import_module(
            seam.module)
    except ImportError:
        return "missing-module", None
    obj = mod
    for part in seam.attr.split(".") if seam.attr else ():
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return "missing-attr", None
    return "ok", obj


def _fingerprint(obj) -> str | None:
    """sha256 of the object's source, or None when unhashable (constants,
    kernel objects, module-level values)."""
    try:
        src = inspect.getsource(obj)
    except (TypeError, OSError):
        return None
    return hashlib.sha256(src.encode()).hexdigest()


def _rebound(seam: Seam, obj) -> bool:
    """True when the seam currently holds a gmlx replacement - one of
    our installers already ran in this interpreter. Source comparison is
    then meaningless (the patch being present proves the install succeeded);
    existence checks still apply. Upstream's own cross-package from-imports
    (e.g. qwen3_5.gated_delta re-exporting mlx-lm functions) keep their
    original __module__ and are fingerprinted normally."""
    mod = getattr(obj, "__module__", None) or getattr(
        type(obj), "__module__", "")
    return bool(mod) and mod.split(".")[0] == "gmlx"


def _key(seam: Seam) -> str:
    return f"{seam.module}:{seam.attr}"


def collect_fingerprints() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for seam in SEAMS:
        status, obj = _resolve(seam)
        if status == "ok":
            if _rebound(seam, obj):
                raise RuntimeError(
                    f"{_key(seam)} already holds a gmlx patch - "
                    f"regenerate fingerprints in a fresh interpreter")
            out[_key(seam)] = _fingerprint(obj)
        elif ((status == "missing-module" and seam.module_optional)
              or (status == "missing-attr" and seam.attr_optional)):
            continue  # not present on this qualified version
        else:
            raise RuntimeError(
                f"cannot fingerprint {_key(seam)}: {status} "
                f"(used by {seam.used_by})")
    return out


def load_pinned() -> dict:
    with open(_JSON_PATH) as f:
        return json.load(f)


def check_seams() -> list[str]:
    """Problems found against the pinned fingerprints; [] when clean."""
    pinned = load_pinned()
    fps = pinned["fingerprints"]
    problems = []
    for seam in SEAMS:
        status, obj = _resolve(seam)
        if status == "missing-module":
            if not seam.module_optional:
                problems.append(
                    f"{_key(seam)}: module missing (used by {seam.used_by})")
            continue
        if status == "missing-attr":
            if not seam.attr_optional:
                problems.append(
                    f"{_key(seam)}: attribute missing - upstream renamed or "
                    f"removed it (used by {seam.used_by})")
            continue
        if _rebound(seam, obj):
            continue  # our installer already replaced it in this process
        want = fps.get(_key(seam), "<unpinned>")
        got = _fingerprint(obj)
        if want == "<unpinned>":
            problems.append(
                f"{_key(seam)}: no pinned fingerprint - run "
                f"`python -m gmlx.upstream_seams --regen`")
        elif got != want:
            problems.append(
                f"{_key(seam)}: source changed under the pin - re-audit "
                f"{seam.used_by}, then regenerate "
                f"(pinned for {pinned.get('generated_with')})")
    return problems


def vendored_upstream_collisions() -> list[str]:
    """Vendored mlx-lm model modules that upstream now ships natively.

    Our vendored modules register themselves into ``sys.modules`` under the
    mlx_lm.models namespace, so an upstream module of the same name would be
    silently shadowed; flag it so the vendored copy gets reconciled/dropped.
    """
    from .arch_table import _VENDORED_MLX_LM_MODULES
    import mlx_lm.models as lm_models
    root = os.path.dirname(lm_models.__file__)
    hits = []
    for mod_name in _VENDORED_MLX_LM_MODULES:
        leaf = mod_name.rsplit(".", 1)[-1]
        if os.path.exists(os.path.join(root, f"{leaf}.py")):
            hits.append(
                f"{mod_name}: upstream mlx-lm now ships this module; the "
                f"vendored copy shadows it - reconcile and drop the vendor "
                f"entry (arch_table._VENDORED_MLX_LM_MODULES)")
    return hits


def _parse_version(v: str) -> tuple[int, ...]:
    m = re.match(r"(\d+(?:\.\d+)*)", v)
    if not m:
        return ()
    return tuple(int(p) for p in m.group(1).split("."))


_VERSION_WARNED = False


def check_upstream_versions(quiet: bool = False) -> list[str]:
    """Runtime gate for the mlx / mlx-lm / mlx-vlm environment.

    Below a floor: raises RuntimeError (a stale venv resolved pyproject
    constraints once and never again - pip pins cannot catch it). Newer than
    the qualified versions recorded at the last --regen: warns once. Missing
    or unparsable metadata (source installs): warns, never raises. Returns
    the warning lines (also printed unless ``quiet``).
    """
    global _VERSION_WARNED
    import importlib.metadata as md
    try:
        tested = load_pinned().get("generated_with", {})
    except (OSError, json.JSONDecodeError):
        tested = {}
    warnings, errors = [], []
    for pkg, floor in FLOORS.items():
        try:
            found = md.version(pkg)
        except md.PackageNotFoundError:
            warnings.append(
                f"[versions] {pkg}: no package metadata (source install?) - "
                f"cannot verify the supported floor >= {floor}")
            continue
        found_t = _parse_version(found)
        if not found_t:
            warnings.append(
                f"[versions] {pkg}: unparsable version {found!r} - cannot "
                f"verify the supported floor >= {floor}")
            continue
        floor_t = _parse_version(floor)
        if found_t < floor_t:
            errors.append(
                f"{pkg} {found} is below the supported floor {floor}; "
                f"upgrade it (pip install -U '{pkg}>={floor}') or reinstall "
                f"gmlx into a fresh venv")
            continue
        qual = tested.get(pkg)
        if qual and found_t > _parse_version(qual):
            warnings.append(
                f"[versions] {pkg} {found} is newer than the qualified "
                f"{qual}; untested - see docs/upstream-upgrades.md")
    if errors:
        raise RuntimeError(
            "unsupported upstream package versions:\n  " + "\n  ".join(errors))
    if warnings and not quiet and not _VERSION_WARNED:
        _VERSION_WARNED = True
        for w in warnings:
            print(w, file=sys.stderr, flush=True)
    return warnings


def regen(path: str = _JSON_PATH) -> dict:
    import importlib.metadata as md
    generated_with = {}
    for pkg in FLOORS:
        try:
            generated_with[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            generated_with[pkg] = None
    data = {
        "generated_with": generated_with,
        "fingerprints": collect_fingerprints(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    return data


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Check (default) or regenerate the upstream seam "
                    "fingerprints.")
    p.add_argument("--regen", action="store_true",
                   help="re-capture fingerprints from the installed "
                        "upstream versions (do this only after a deliberate, "
                        "audited upstream bump)")
    args = p.parse_args(argv)
    if args.regen:
        data = regen()
        n = len(data["fingerprints"])
        print(f"pinned {n} seam fingerprints for {data['generated_with']}")
        return 0
    problems = check_seams() + vendored_upstream_collisions()
    try:
        problems += check_upstream_versions(quiet=True)
    except RuntimeError as e:
        problems.append(str(e))
    if problems:
        print("\n".join(problems))
        return 1
    print(f"{len(SEAMS)} seams OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
