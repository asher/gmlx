"""The LoRA file reader: safetensors keys to pairs, then pairs to targets.

``read_lora`` parses the suffix of every key on its own, so the kohya,
diffusers, PEFT and OneTrainer spellings and their hybrids all read as the
same up, down and alpha roles. A file that carries another adapter type is
refused by that type's name. ``resolve`` then strips the prefix of every
stem on its own and looks the rest up in a family key map. Underscore
stems are found by spelling every known target with underscores, since an
underscore name cannot be split back into a path.

A key or a stem that matches nothing raises by default and lists the first
few, because a partly applied adapter still produces an image.

A pair is scaled by its alpha over its rank. The alpha is the pair's
``alpha`` tensor, in every spelling of the pair's keys. A file that
diffusers' ``save_lora_weights`` wrote has no alpha tensors: its header
holds the PEFT config of each component under ``lora_adapter_metadata``,
and ``header_configs`` reads the rank and the alpha of every pair from it.
A rank-stabilized config scales a pair by its alpha over the square root
of its rank, as PEFT does. A config field that would change the delta in
another way is refused by name. An alpha that is not finite and a factor with a value that is not
finite are refused at the read.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Mapping

import mlx.core as mx

from ..fingerprint import ImageModelError


class AdapterError(ImageModelError):
    """A LoRA file or patch that gmlx image refuses, with the reason."""


class UnsupportedAdapterError(AdapterError):
    """An adapter type gmlx image does not apply, named in ``kind``."""

    def __init__(self, message: str, kind: str):
        super().__init__(message)
        self.kind = kind


class AdapterFamilyError(AdapterError):
    """An adapter none of whose keys names a module of this family."""


class UnmatchedKeysError(AdapterError):
    """Keys that match no suffix, or stems that match no target."""

    def __init__(self, message: str, keys: tuple):
        super().__init__(message)
        self.keys = keys


# Suffix spelling to role, longest first.
SUFFIXES = (
    (".lora_linear_layer.down.weight", "down"),
    (".lora_linear_layer.up.weight", "up"),
    (".lora_A.default.weight", "down"),
    (".lora_B.default.weight", "up"),
    (".lora_down.weight", "down"),
    (".lora_up.weight", "up"),
    (".lora.down.weight", "down"),
    ("_lora.down.weight", "down"),
    (".lora_A.weight", "down"),
    (".lora_B.weight", "up"),
    (".lora.up.weight", "up"),
    ("_lora.up.weight", "up"),
    (".lora_A", "down"),
    (".lora_B", "up"),
    (".alpha", "alpha"),
)

# The key part that marks an adapter type gmlx does not apply.
REFUSED_PARTS = {
    "dora_scale": "DoRA",
    "lora_magnitude_vector": "DoRA",
    "diff": "full-weight diff",
    "diff_b": "full-weight diff",
    "set_weight": "full-weight diff",
    "w_norm": "full-weight diff",
    "b_norm": "full-weight diff",
    "lora_mid": "Tucker LoCon",
    "reshape_weight": "reshape",
}
REFUSED_STARTS = (("hada_", "LoHa"), ("lokr_", "LoKr"))

# Prefixes of a dotted stem, tried in order. The empty one is a bare path.
DOTTED_PREFIXES = ("model.diffusion_model.", "diffusion_model.",
                   "transformer.", "base_model.model.", "unet.", "")
UNDERSCORE_PREFIXES = ("lora_unet_", "lycoris_", "lora_transformer_")
# The underscore prefix under which ComfyUI also finds the names that only
# the diffusers layout has.
COMFY_DIFFUSERS_UNDERSCORE = ("lycoris_",)
TEXT_DOTTED = ("text_encoders.", "text_encoder.")
TEXT_UNDERSCORE = ("lora_te1_", "lora_te2_", "lora_te3_", "lora_te_")

_LISTED = 4

# The entry of a safetensors header where diffusers' save_lora_weights
# keeps the PEFT LoRA config of each component, as JSON of
# ``<component>.<field>`` to value. The diffusers loader builds the
# component's config from the entry in place of what the tensors say.
HEADER_ENTRY = "lora_adapter_metadata"
DIT_COMPONENT, TEXT_COMPONENT = "transformer", "text_encoder"
# Config fields that give the rank and the alpha of a pair and the
# modules that have one.
_HEADER_READ = frozenset({"r", "lora_alpha", "rank_pattern", "alpha_pattern",
                          "target_modules", "exclude_modules", "use_rslora"})
# Config fields that leave the delta of a stored pair as it is: settings
# of training and of initialization, and labels.
_HEADER_INERT = frozenset({
    "lora_dropout", "init_lora_weights", "inference_mode", "task_type",
    "peft_type", "peft_version", "base_model_name_or_path", "revision",
    "auto_mapping", "megatron_config", "megatron_core", "loftq_config",
    "eva_config", "corda_config", "runtime_config", "qalora_group_size",
    "ensure_weight_tying"})
# Config fields that change the delta when set, to the adapter type that
# the refusal names.
_HEADER_REFUSED = {
    "use_dora": "DoRA",
    "lora_bias": "LoRA with a bias",
    "bias": "LoRA with trained base biases",
    "use_qalora": "QALoRA",
    "alora_invocation_tokens": "activated LoRA",
    "arrow_config": "Arrow routing",
    "fan_in_fan_out": "LoRA on a transposed weight",
    "layer_replication": "LoRA on replicated layers",
    "layers_to_transform": "LoRA on chosen layers",
    "layers_pattern": "LoRA on chosen layers",
    "modules_to_save": "LoRA with replaced modules",
    "trainable_token_indices": "LoRA with trained token rows",
    "target_parameters": "LoRA on bare parameters",
}
# The value of a refused field that leaves it off, beside an empty one.
_HEADER_OFF = {"bias": "none"}


@dataclass(frozen=True)
class RawPair:
    """One low-rank pair as the file stores it: ``down`` is (rank, in),
    ``up`` is (out, rank), and ``alpha`` is None when the file has none.
    ``alpha_dtype`` is the dtype of the file's alpha tensor."""
    stem: str
    down: mx.array
    up: mx.array
    alpha: float | None
    alpha_dtype: mx.Dtype | None = None

    @property
    def rank(self) -> int:
        return int(self.down.shape[0])


@dataclass(frozen=True)
class RawLora:
    path: str
    pairs: Mapping            # stem -> RawPair, in file order
    metadata: Mapping
    stray: tuple = ()         # keys with no known suffix, kept on request


@dataclass(frozen=True)
class LoraKeyMap:
    """A family's adapter targets. ``targets`` maps a target name to its
    module path and band of (dim, start, length), or None for the whole
    weight. ``diffusers_only`` names the targets that only the diffusers
    layout has, which ComfyUI finds under ``lycoris_`` and not under the
    other underscore prefixes.
    ``text_names`` are the names a text encoder key may carry after
    ``text_encoders.``, and ``text_targets`` the encoder's module paths.
    ``absent`` maps a target name that the family has in another layout
    of its DiT file, and not in this one, to what the name is and what
    this file has in its place, which ends the refusal of such a key.
    ``swapped`` names the targets of a layout whose weight holds the two
    halves of the module's output rows in the other order, as a diffusers
    final norm puts the scale rows before the shift rows. The halves of
    the up rows of a pair on such a target are swapped back.
    ``comfy_prefixes``, when set, are the only stem prefixes under which
    ComfyUI finds the family's targets, and only those that are not
    ``diffusers_only``: ComfyUI drops every other spelling.
    ``comfy_stems``, when not None, is every DiT stem that ComfyUI's key
    map reads for the family, and ComfyUI drops every other stem;
    ``comfy_reads`` says which stems those are, for the warning. None
    keeps the prefix rules above."""
    family: str
    title: str
    targets: Mapping
    diffusers_only: frozenset = frozenset()
    text_names: tuple = ()
    text_targets: tuple = ()
    absent: Mapping = field(default_factory=dict)
    swapped: frozenset = frozenset()
    comfy_prefixes: tuple = ()
    comfy_stems: frozenset | None = None
    comfy_reads: str = ""


@dataclass(frozen=True)
class Target:
    """One pair placed on a module."""
    module: str
    band: tuple | None
    pair: RawPair


@dataclass(frozen=True)
class ResolvedLora:
    path: str
    dit: tuple                # Target rows on DiT modules
    text: tuple               # Target rows on text encoder modules
    unmatched: tuple = ()     # stems that named no target
    warnings: tuple = ()


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def _selector(value, what: str):
    """A ``target_modules`` or ``exclude_modules`` value as a compiled
    regular expression or a tuple of names, or None when it is empty."""
    if not value:
        return None
    if isinstance(value, str):
        try:
            return re.compile(value)
        except re.error as e:
            raise AdapterError(f"{what} is not a regular expression "
                               f"({e})") from None
    if isinstance(value, (list, tuple)) and all(isinstance(v, str)
                                                for v in value):
        return tuple(value)
    raise AdapterError(f"{what} is neither a regular expression nor a list "
                       "of names")


def _selects(selector, name: str) -> bool:
    """Whether a selector names the module ``name``, as PEFT reads it: a
    regular expression matches the whole name, and a listed name matches
    the name or its end after a dot."""
    if isinstance(selector, tuple):
        return name in selector or any(name.endswith("." + t)
                                       for t in selector)
    return selector.fullmatch(name) is not None


def _patterns(value, what: str, ok) -> tuple:
    """A ``rank_pattern`` or ``alpha_pattern`` value as rows of (compiled
    pattern, value) in header order. PEFT matches a pattern at the end of
    a module name, after a dot or at its start."""
    if not value:
        return ()
    if not isinstance(value, dict):
        raise AdapterError(f"{what} is not a mapping")
    rows = []
    for key, v in value.items():
        if not ok(v):
            raise AdapterError(f"{what} gives {key} the value {v!r}")
        try:
            rows.append((re.compile(rf"(.*\.)?({key})$"), v))
        except re.error as e:
            raise AdapterError(f"{what} has the key {key!r}, which is not "
                               f"a regular expression ({e})") from None
    return tuple(rows)


def _first(rows: tuple, name: str, default):
    for pattern, value in rows:
        if pattern.match(name):
            return value
    return default


@dataclass(frozen=True)
class HeaderConfig:
    """The LoRA config of one component from a file's header: the rank
    and the alpha of its pairs, the patterns that give some modules
    another rank or alpha, the modules that it names, and whether its
    pairs are rank-stabilized, scaled by the alpha over the square root
    of the rank."""
    rank: int
    alpha: float
    rank_pattern: tuple = ()
    alpha_pattern: tuple = ()
    targets: object = None
    excluded: object = None
    rslora: bool = False

    def of(self, name: str) -> tuple[int, float]:
        """(rank, alpha) of the module ``name``: those of the first
        pattern in header order that matches it, else ``r`` and
        ``lora_alpha``."""
        return (_first(self.rank_pattern, name, self.rank),
                float(_first(self.alpha_pattern, name, self.alpha)))

    def names(self, name: str) -> bool:
        """Whether the config places a pair on the module ``name``."""
        return _selects(self.targets, name) and not (
            self.excluded is not None and _selects(self.excluded, name))


def _header_config(path: str, comp: str, fields: dict) -> HeaderConfig:
    where = f"{path}: the adapter header"
    for name, value in fields.items():
        if name in _HEADER_READ or name in _HEADER_INERT or not value \
                or value == _HEADER_OFF.get(name):
            continue
        kind = _HEADER_REFUSED.get(name)
        if kind is None:
            raise UnsupportedAdapterError(
                f"{where} sets {comp}.{name}, a field gmlx image does not "
                "read. It reads r, lora_alpha, rank_pattern, alpha_pattern, "
                "target_modules and exclude_modules", "adapter header")
        raise UnsupportedAdapterError(
            f"{where} sets {comp}.{name} to {value!r}, which makes this "
            f"{kind}. gmlx image does not apply it. It applies plain LoRA "
            "pairs only", kind)
    rank, alpha = fields.get("r"), fields.get("lora_alpha")
    positive = lambda v: _number(v) and int(v) == v and v > 0  # noqa: E731
    if not positive(rank):
        raise AdapterError(f"{where} gives {comp}.r as {rank!r}, which is "
                           "not a positive integer")
    if not _number(alpha):
        raise AdapterError(f"{where} gives {comp}.lora_alpha as {alpha!r}, "
                           "which is not a finite number")
    targets = _selector(fields.get("target_modules"),
                        f"{where}: {comp}.target_modules")
    if targets is None:
        raise AdapterError(f"{where} gives no {comp}.target_modules")
    rslora = fields.get("use_rslora") or False
    if rslora is not False and rslora is not True:
        raise AdapterError(f"{where} gives {comp}.use_rslora as {rslora!r}, "
                           "which is neither true nor false")
    return HeaderConfig(
        rank=int(rank), alpha=float(alpha),
        rank_pattern=tuple((p, int(v)) for p, v in _patterns(
            fields.get("rank_pattern"), f"{where}: {comp}.rank_pattern",
            positive)),
        alpha_pattern=_patterns(fields.get("alpha_pattern"),
                                f"{where}: {comp}.alpha_pattern", _number),
        targets=targets,
        excluded=_selector(fields.get("exclude_modules"),
                           f"{where}: {comp}.exclude_modules"),
        rslora=rslora)


def header_configs(metadata: Mapping | None,
                   path: str = "<arrays>") -> dict:
    """Component to ``HeaderConfig`` from the ``lora_adapter_metadata``
    entry of a safetensors header, as diffusers' ``save_lora_weights``
    writes it. Empty without the entry.

    The diffusers loader passes the fields of a component to PEFT's
    ``LoraConfig`` in place of what it reads from the tensors. Of those
    fields gmlx applies ``r``, ``lora_alpha``, ``rank_pattern`` and
    ``alpha_pattern``, which scale a pair by its alpha over its rank, or
    over the square root of its rank when ``use_rslora`` is set, and
    holds every pair to ``target_modules`` and ``exclude_modules``. A
    field that changes the delta in another way and is set raises
    ``UnsupportedAdapterError`` by its name: ``use_dora``,
    ``lora_bias``, ``bias``, ``fan_in_fan_out`` and the others of
    ``_HEADER_REFUSED``. So does a field gmlx does not know,
    when it is set. An entry gmlx cannot read raises ``AdapterError``."""
    raw = (metadata or {}).get(HEADER_ENTRY)
    if not raw:
        return {}
    where = f"{path}: the {HEADER_ENTRY} entry of the header"
    try:
        doc = json.loads(raw)
    except (TypeError, ValueError):
        raise AdapterError(f"{where} is not JSON") from None
    if not isinstance(doc, dict):
        raise AdapterError(f"{where} is not a JSON object")
    parts: dict[str, dict] = {}
    for key, value in doc.items():
        comp, dot, name = str(key).partition(".")
        if not dot or not comp or not name:
            raise AdapterError(f"{where} has the key {key!r}, which names "
                               "no component and field")
        parts.setdefault(comp, {})[name] = value
    return {comp: _header_config(path, comp, fields)
            for comp, fields in parts.items()}


def _header_covers(path: str, configs: Mapping, comp: str, name: str,
                   stem: str) -> HeaderConfig | None:
    """The header config that covers the module ``name`` of component
    ``comp``, or None for a file without a header config. Raises
    ``AdapterError`` when the header has configs and none for the
    component, or when its config does not name the module: the reference
    loader does not load such a pair."""
    if not configs:
        return None
    cfg = configs.get(comp)
    if cfg is None:
        raise AdapterError(
            f"{path}: the adapter header has a LoRA config for "
            f"{', '.join(sorted(configs))} and none for {comp}, which "
            f"{stem} belongs to")
    if not cfg.names(name):
        raise AdapterError(
            f"{path}: {stem} is a pair on {name}, which the "
            f"{comp}.target_modules and exclude_modules of the adapter "
            "header leave out")
    return cfg


def _under_header(path: str, configs: Mapping, comp: str, name: str,
                  pair: RawPair) -> tuple[RawPair, bool, float | None]:
    """``pair`` with the alpha that the header gives the module ``name``
    of component ``comp``, whether its scale is not the one ComfyUI gives
    it, which reads no header, and for a rank-stabilized config the
    factor of the one over the other. A pair without an alpha tensor
    takes the header's. A rank-stabilized pair is scaled by the alpha
    over the square root of its rank, and carries the alpha times that
    root. Raises ``AdapterError`` for a pair whose rank is not the
    header's, and for an alpha tensor that is not the header's alpha in
    the tensor's dtype."""
    cfg = _header_covers(path, configs, comp, name, pair.stem)
    if cfg is None:
        return pair, False, None
    rank, alpha = cfg.of(name)
    if rank != pair.rank:
        raise AdapterError(
            f"{path}: {pair.stem} has rank {pair.rank}, and the adapter "
            f"header gives {name} the rank {rank}")
    if pair.alpha is not None:
        same = mx.array(alpha, pair.alpha_dtype or mx.float32).astype(
            mx.float32)
        if float(same.item()) != pair.alpha:
            raise AdapterError(
                f"{path}: {pair.stem} has an alpha tensor of "
                f"{pair.alpha:g}, and the adapter header gives {name} the "
                f"alpha {alpha:g}")
    if not cfg.rslora:
        if pair.alpha is None:
            return replace(pair, alpha=alpha), alpha != rank, None
        return pair, False, None
    ours = alpha / math.sqrt(rank)
    comfy = 1.0 if pair.alpha is None else pair.alpha / rank
    moved = ours != comfy
    factor = (ours / comfy if comfy else math.inf) if moved else None
    return replace(pair, alpha=alpha * math.sqrt(rank)), moved, factor


# The key part PEFT writes two places before the end, as
# ``lora_magnitude_vector.<adapter name>.weight``.
REFUSED_NAMED = ("lora_magnitude_vector",)


def _refused_kind(key: str) -> str | None:
    parts = key.split(".")
    for part in parts[-2:]:
        kind = REFUSED_PARTS.get(part)
        if kind is not None:
            return kind
        for start, name in REFUSED_STARTS:
            if part.startswith(start):
                return name
    if len(parts) >= 3 and parts[-3] in REFUSED_NAMED:
        return REFUSED_PARTS[parts[-3]]
    return None


def _split(key: str) -> tuple[str, str] | None:
    for suffix, role in SUFFIXES:
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)], role
    return None


def _listed(keys) -> str:
    keys = list(keys)
    text = ", ".join(keys[:_LISTED])
    more = len(keys) - _LISTED
    return text + (f", and {more} more" if more > 0 else "")


def _matrix(arr: mx.array, key: str, path: str) -> mx.array:
    """A factor as a matrix. A 1x1 kernel's trailing axes are dropped."""
    shape = tuple(arr.shape)
    if len(shape) > 2 and all(d == 1 for d in shape[2:]):
        return arr.reshape(shape[0], shape[1])
    if len(shape) != 2:
        raise AdapterError(f"{path}: {key} has shape {shape}; gmlx image "
                           "applies LoRA pairs on linear layers only")
    return arr


def _need_pair(path: str, stem: str, roles) -> None:
    """Raise ``AdapterError`` for a stem that lacks a half of its pair.
    ``roles`` holds the roles the stem has tensors for."""
    if "down" in roles and "up" in roles:
        return
    if set(roles) == {"alpha"}:
        raise AdapterError(f"{path}: {stem} has an alpha and no pair")
    missing = "down" if "down" not in roles else "up"
    raise AdapterError(f"{path}: {stem} has no {missing} tensor")


def _one_stem_per_target(path: str, rows) -> None:
    """Raise ``AdapterError`` when two of the (stem, module, band) rows
    target one module and band."""
    seen: dict = {}
    for stem, module, band in rows:
        key = (module, band)
        if key in seen:
            raise AdapterError(
                f"{path}: {stem} and {seen[key]} both target "
                f"{module}" + ("" if band is None else f" band {band}"))
        seen[key] = stem


def parse_pairs(arrays: Mapping, path: str = "<arrays>", *,
                metadata: Mapping | None = None,
                allow_unmatched: bool = False) -> RawLora:
    """The pairs of a key to array mapping. Raises
    ``UnsupportedAdapterError`` for a refused adapter type,
    ``UnmatchedKeysError`` for keys with no known suffix unless
    ``allow_unmatched``, and ``AdapterError`` for a broken pair, for an
    alpha that is not finite and for a factor that holds a value that is
    not finite."""
    for key in arrays:
        kind = _refused_kind(key)
        if kind is not None:
            raise UnsupportedAdapterError(
                f"{path} is a {kind} adapter (key {key}), which gmlx image "
                "does not apply. It applies plain LoRA pairs only", kind)
    parts: dict[str, dict] = {}
    stray = []
    for key, arr in arrays.items():
        hit = _split(key)
        if hit is None:
            stray.append(key)
            continue
        stem, role = hit
        slot = parts.setdefault(stem, {})
        if role in slot:
            raise AdapterError(f"{path}: {stem} has two {role} tensors")
        slot[role] = (key, arr)
    if stray and not allow_unmatched:
        raise UnmatchedKeysError(
            f"{path}: {len(stray)} of {len(arrays)} keys have no known LoRA "
            f"suffix: {_listed(stray)}", tuple(stray))
    pairs = {}
    factors: list = []
    for stem, slot in parts.items():
        _need_pair(path, stem, slot)
        down = _matrix(slot["down"][1], slot["down"][0], path)
        up = _matrix(slot["up"][1], slot["up"][0], path)
        if up.shape[1] != down.shape[0]:
            raise AdapterError(
                f"{path}: {stem} pairs up {tuple(up.shape)} with down "
                f"{tuple(down.shape)}, whose ranks differ")
        alpha = alpha_dtype = None
        if "alpha" in slot:
            a = slot["alpha"][1]
            if a.size != 1:
                raise AdapterError(f"{path}: {slot['alpha'][0]} is not a "
                                   "scalar")
            alpha = float(a.astype(mx.float32).reshape(-1)[0].item())
            alpha_dtype = a.dtype
            if not math.isfinite(alpha):
                raise AdapterError(f"{path}: {slot['alpha'][0]} is {alpha}, "
                                   "which is not finite")
        pairs[stem] = RawPair(stem, down, up, alpha, alpha_dtype)
        factors += [(slot["down"][0], down), (slot["up"][0], up)]
    if not pairs:
        raise AdapterError(f"{path} holds no LoRA pair")
    finite = [mx.all(mx.isfinite(a)) for _, a in factors]
    mx.eval(finite)
    for (key, _), ok in zip(factors, finite):
        if not ok.item():
            raise AdapterError(f"{path}: {key} holds a value that is not "
                               "finite")
    return RawLora(path=path, pairs=pairs, metadata=dict(metadata or {}),
                   stray=tuple(stray))


def _header_names(path: str) -> tuple[list, dict]:
    """The tensor names and the metadata of a safetensors file, from its
    header. Raises ``UnsupportedAdapterError`` when a name marks an
    adapter type gmlx does not apply. No tensor is read."""
    from ..vae.reader import VAEError, safetensors_header

    try:
        header, _ = safetensors_header(path)
    except VAEError:
        raise AdapterError(f"{path} is not a safetensors file; gmlx image "
                           "reads LoRA adapters from safetensors") from None
    except OSError as e:
        raise AdapterError(f"{path}: {e.strerror or e}") from None
    names = [k for k in header if k != "__metadata__"]
    for key in names:
        kind = _refused_kind(key)
        if kind is not None:
            raise UnsupportedAdapterError(
                f"{path} is a {kind} adapter (key {key}), which gmlx image "
                "does not apply. It applies plain LoRA pairs only", kind)
    meta = header.get("__metadata__")
    return names, meta if isinstance(meta, dict) else {}


def vet_header(path: str, *, allow_unmatched: bool = False,
               keymap: LoraKeyMap | None = None, modules=None) -> int:
    """Check a LoRA safetensors file by its header alone and return the
    number of pair stems it names. Raises as ``read_lora`` does for a file
    that is not safetensors, for a refused adapter type, for keys with no
    known suffix, for a stem with two tensors of one role or without
    both halves of its pair, and for a header config that
    ``header_configs`` refuses. It reads no tensor, so a caller can
    refuse a file before it loads a model.

    With ``keymap``, the stems are placed as ``resolve`` places them, so a
    file for another family raises ``AdapterFamilyError``, a stem that
    names no module raises ``UnmatchedKeysError``, and two stems on one
    module or a stem that the header config leaves out raise
    ``AdapterError``. With ``modules``, the module paths that the DiT
    file holds a weight for, a stem that targets a module outside them
    raises ``AdapterError``. Whether a pair's shape fits its module, its
    rank is the header's and its alpha tensor is the header's alpha is
    checked at the load."""
    import os

    path = os.path.expanduser(str(path))
    names, meta = _header_names(path)
    configs = header_configs(meta, path)
    # Stems in file order, as ``parse_pairs`` holds them.
    parts: dict[str, set] = {}
    stray = []
    for key in names:
        hit = _split(key)
        if hit is None:
            stray.append(key)
            continue
        stem, role = hit
        roles = parts.setdefault(stem, set())
        if role in roles:
            raise AdapterError(f"{path}: {stem} has two {role} tensors")
        roles.add(role)
    if stray and not allow_unmatched:
        raise UnmatchedKeysError(
            f"{path}: {len(stray)} of {len(names)} keys have no known LoRA "
            f"suffix: {_listed(stray)}", tuple(stray))
    for stem, roles in parts.items():
        _need_pair(path, stem, roles)
    if not parts:
        raise AdapterError(f"{path} holds no LoRA pair")
    if keymap is not None:
        _vet_targets(path, list(parts), keymap, modules, allow_unmatched,
                     configs)
    return len(parts)


def _vet_targets(path: str, stems: list, keymap: LoraKeyMap, modules,
                 allow_unmatched: bool, configs: Mapping) -> None:
    """Raise as ``resolve`` and ``load_adapter`` do for stems that name no
    module of the family, two stems on one module, a stem that the header
    config leaves out, or a module the DiT file does not hold."""
    under = _underscore_index(keymap.targets)
    text_under = _underscore_index(keymap.text_targets)
    dit, text, unmatched, absent, other = [], [], [], [], []
    for stem in stems:
        if is_text_stem(stem):
            target = _text_target(stem, keymap, text_under)
            if target is None:
                unmatched.append(stem)
            else:
                _header_covers(path, configs, TEXT_COMPONENT, target, stem)
                text.append((stem, target, None))
            continue
        hit = _dit_target(stem, keymap.targets, under)
        if hit is None:
            unmatched.append(stem)
            if _other_layout(stem, keymap) is not None:
                other.append(stem)
            continue
        _header_covers(path, configs, DIT_COMPONENT, hit[0], stem)
        module, band = keymap.targets[hit[0]]
        dit.append((stem, module, band))
        if modules is not None and module not in modules:
            absent.append((stem, module))
    _need_targets(path, keymap, len(stems), bool(dit or text), unmatched,
                  other, allow_unmatched)
    _one_stem_per_target(path, dit + text)
    if absent:
        stem, module = absent[0]
        raise AdapterError(f"{path}: {stem} targets {module}, which this "
                           "DiT does not have")


def _other_layout(stem: str, keymap: LoraKeyMap) -> str | None:
    """What a DiT stem names when the family has its target in another
    layout of the DiT file and not in this one, else None."""
    if not keymap.absent:
        return None
    hit = _dit_target(stem, keymap.absent, _underscore_index(keymap.absent))
    return None if hit is None else keymap.absent[hit[0]]


def _need_targets(path: str, keymap: LoraKeyMap, total: int, placed: bool,
                  unmatched: list, other: list, allow_unmatched: bool) -> str:
    """Raise for the stems that name no target, and return the warning
    for them when they are allowed, or an empty string. A stem that names
    a target of another layout of the DiT file is refused with that
    layout's name, never as an adapter of another family."""
    msg = ""
    if other:
        msg = (f"{path}: {len(other)} of {total} targets name "
               f"{_other_layout(other[0], keymap)}: {_listed(other)}")
        if not placed or not allow_unmatched:
            raise UnmatchedKeysError(msg, tuple(other))
    if not placed:
        raise AdapterFamilyError(
            f"{path} is not a {keymap.title} adapter: none of its "
            f"{total} targets names a {keymap.title} module "
            f"({_listed(unmatched)})")
    rest = [s for s in unmatched if s not in other]
    if rest:
        named = (f"{path}: {len(rest)} of {total} targets name no "
                 f"{keymap.title} module: {_listed(rest)}")
        if not allow_unmatched:
            raise UnmatchedKeysError(named, tuple(rest))
        msg = f"{msg}. {named}" if msg else named
    return msg + ". They are skipped" if msg else ""


def read_lora(path: str, *, allow_unmatched: bool = False) -> RawLora:
    """The pairs of a LoRA safetensors file."""
    import os

    path = os.path.expanduser(str(path))
    # Refuse by name from the header, before any tensor loads.
    names, _ = _header_names(path)
    try:
        arrays, meta = mx.load(path, return_metadata=True)
    except (RuntimeError, ValueError) as e:
        raise AdapterError(f"{path}: cannot read the safetensors data "
                           f"({e})") from None
    return parse_pairs({k: arrays[k] for k in names if k in arrays}, path,
                       metadata=meta, allow_unmatched=allow_unmatched)


def _underscore_index(names) -> dict:
    out: dict[str, str] = {}
    for name in names:
        out.setdefault(name.replace(".", "_"), name)
    return out


def _dit_target(stem: str, targets: Mapping, under: dict):
    """(target name, underscore form) of a DiT stem among ``targets``,
    whose underscore index is ``under``, or None."""
    for p in UNDERSCORE_PREFIXES:
        if stem.startswith(p):
            name = under.get(stem[len(p):])
            return None if name is None else (name, True)
    rest = stem
    for _ in range(4):
        if rest in targets:
            return rest, False
        for p in DOTTED_PREFIXES:
            if p and rest.startswith(p):
                rest = rest[len(p):]
                break
        else:
            return None
    return None


def _text_target(stem: str, keymap: LoraKeyMap, under: dict) -> str | None:
    """The encoder module path of a text stem, or None."""
    known = set(keymap.text_targets)
    for p in TEXT_UNDERSCORE:
        if stem.startswith(p):
            rest = stem[len(p):]
            return under.get(rest) or under.get("model_" + rest)
    rest = stem
    for p in TEXT_DOTTED:
        if rest.startswith(p):
            rest = rest[len(p):]
            break
    for name in keymap.text_names:
        for mid in (f"{name}.transformer.", f"{name}."):
            if rest.startswith(mid):
                rest = rest[len(mid):]
    for cand in (rest, "model." + rest):
        if cand in known:
            return cand
    return None


def _comfy_drops(stem: str, name: str, underscored: bool,
                 keymap: LoraKeyMap) -> bool:
    """Whether ComfyUI's key map lacks the DiT stem ``stem`` of target
    ``name``."""
    if keymap.comfy_stems is not None:
        return stem not in keymap.comfy_stems
    if keymap.comfy_prefixes:
        if name in keymap.diffusers_only:
            return True
        return not any(
            stem == p + (name.replace(".", "_") if p.endswith("_") else name)
            for p in keymap.comfy_prefixes)
    return (underscored and name in keymap.diffusers_only
            and not stem.startswith(COMFY_DIFFUSERS_UNDERSCORE))


def is_text_stem(stem: str) -> bool:
    return stem.startswith(TEXT_DOTTED + TEXT_UNDERSCORE)


def _factors(values: list) -> str:
    """The distinct factors of ``values``, the first few, for a warning."""
    got = sorted(set(values))
    shown = [f"{v:.4g}" for v in got[:_LISTED]]
    more = len(got) - len(shown)
    return " or ".join(shown) + (f" and {more} more" if more else "")


def _halves_swapped(pair: RawPair) -> RawPair:
    """The pair with the two halves of its up rows in the other order. An
    odd count of rows is left as it is, for the shape check of the load
    to refuse."""
    n = int(pair.up.shape[0])
    if n % 2:
        return pair
    h = n // 2
    return replace(pair, up=mx.concatenate([pair.up[h:], pair.up[:h]],
                                           axis=0))


def resolve(raw: RawLora, keymap: LoraKeyMap, *,
            allow_unmatched: bool = False) -> ResolvedLora:
    """Place every pair of ``raw`` on a module of the family, or on a
    module of its text encoder, with the rank and the alpha that the
    file's header config gives it. Raises ``AdapterFamilyError`` when no
    DiT or text stem names a target, ``UnmatchedKeysError`` for the stems
    left over unless ``allow_unmatched``, and as ``header_configs`` and
    ``_under_header`` do for a header config and a pair that does not
    agree with it."""
    configs = header_configs(raw.metadata, raw.path)
    under = _underscore_index(keymap.targets)
    text_under = _underscore_index(keymap.text_targets)
    dit, text, unmatched, dropped, other, scaled = [], [], [], [], [], []
    stable: list = []
    for stem, pair in raw.pairs.items():
        if is_text_stem(stem):
            path = _text_target(stem, keymap, text_under)
            if path is None:
                unmatched.append(stem)
                continue
            pair, moved, factor = _under_header(
                raw.path, configs, TEXT_COMPONENT, path, pair)
            text.append(Target(path, None, pair))
            scaled += [stem] * moved
            stable += [factor] * (factor is not None)
            continue
        hit = _dit_target(stem, keymap.targets, under)
        if hit is None:
            unmatched.append(stem)
            if _other_layout(stem, keymap) is not None:
                other.append(stem)
            continue
        name, underscored = hit
        module, band = keymap.targets[name]
        pair, moved, factor = _under_header(raw.path, configs,
                                            DIT_COMPONENT, name, pair)
        if name in keymap.swapped:
            pair = _halves_swapped(pair)
        dit.append(Target(module, band, pair))
        scaled += [stem] * moved
        stable += [factor] * (factor is not None)
        if _comfy_drops(stem, name, underscored, keymap):
            dropped.append(stem)
    skipped = _need_targets(raw.path, keymap, len(raw.pairs),
                            bool(dit or text), unmatched, other,
                            allow_unmatched)
    warnings = [skipped] if skipped else []
    if scaled:
        warnings.append(
            f"{raw.path}: {len(scaled)} pairs take an alpha from the "
            f"adapter header that is not their rank ({_listed(scaled)}). "
            "gmlx applies it as the diffusers loader does, and ComfyUI "
            "does not read the header, so the image differs from ComfyUI's"
            + (f". The header makes {len(stable)} of them rank-stabilized, "
               "scaled by the alpha over the square root of the rank, and "
               f"gmlx applies those at {_factors(stable)} times the "
               "strength that ComfyUI gives them" if stable else ""))
    if dropped and keymap.comfy_stems is not None:
        warnings.append(
            f"{raw.path}: {len(dropped)} keys have a spelling that ComfyUI "
            f"does not read for {keymap.title} ({_listed(dropped)}). "
            f"ComfyUI reads {keymap.comfy_reads}. gmlx applies them and "
            "ComfyUI drops them, so the image differs from ComfyUI's")
    elif dropped and keymap.comfy_prefixes:
        warnings.append(
            f"{raw.path}: {len(dropped)} keys have a spelling that ComfyUI "
            f"does not read for {keymap.title} ({_listed(dropped)}). "
            "ComfyUI reads the GGUF names under "
            f"{' or '.join(keymap.comfy_prefixes)}. gmlx applies them and "
            "ComfyUI drops them, so the image differs from ComfyUI's")
    elif dropped:
        warnings.append(
            f"{raw.path}: {len(dropped)} underscore keys spell diffusers "
            f"names ({_listed(dropped)}). gmlx applies them and ComfyUI "
            "drops them, so the image differs from ComfyUI's")
    _one_stem_per_target(raw.path, [(t.pair.stem, t.module, t.band)
                                    for t in dit + text])
    return ResolvedLora(path=raw.path, dit=tuple(dit), text=tuple(text),
                        unmatched=tuple(unmatched), warnings=tuple(warnings))
