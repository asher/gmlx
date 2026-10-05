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
"""

from __future__ import annotations

from dataclasses import dataclass
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
TEXT_DOTTED = ("text_encoders.", "text_encoder.")
TEXT_UNDERSCORE = ("lora_te1_", "lora_te2_", "lora_te3_", "lora_te_")

_LISTED = 4


@dataclass(frozen=True)
class RawPair:
    """One low-rank pair as the file stores it: ``down`` is (rank, in),
    ``up`` is (out, rank), and ``alpha`` is None when the file has none."""
    stem: str
    down: mx.array
    up: mx.array
    alpha: float | None

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
    layout has, which ComfyUI does not find under an underscore prefix.
    ``text_names`` are the names a text encoder key may carry after
    ``text_encoders.``, and ``text_targets`` the encoder's module paths."""
    family: str
    title: str
    targets: Mapping
    diffusers_only: frozenset = frozenset()
    text_names: tuple = ()
    text_targets: tuple = ()


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


def _refused_kind(key: str) -> str | None:
    parts = key.split(".")
    for part in parts[-2:]:
        kind = REFUSED_PARTS.get(part)
        if kind is not None:
            return kind
        for start, name in REFUSED_STARTS:
            if part.startswith(start):
                return name
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


def parse_pairs(arrays: Mapping, path: str = "<arrays>", *,
                metadata: Mapping | None = None,
                allow_unmatched: bool = False) -> RawLora:
    """The pairs of a key to array mapping. Raises
    ``UnsupportedAdapterError`` for a refused adapter type,
    ``UnmatchedKeysError`` for keys with no known suffix unless
    ``allow_unmatched``, and ``AdapterError`` for a broken pair."""
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
    for stem, slot in parts.items():
        if "down" not in slot or "up" not in slot:
            if set(slot) == {"alpha"}:
                raise AdapterError(f"{path}: {stem} has an alpha and no "
                                   "pair")
            missing = "down" if "down" not in slot else "up"
            raise AdapterError(f"{path}: {stem} has no {missing} tensor")
        down = _matrix(slot["down"][1], slot["down"][0], path)
        up = _matrix(slot["up"][1], slot["up"][0], path)
        if up.shape[1] != down.shape[0]:
            raise AdapterError(
                f"{path}: {stem} pairs up {tuple(up.shape)} with down "
                f"{tuple(down.shape)}, whose ranks differ")
        alpha = None
        if "alpha" in slot:
            a = slot["alpha"][1]
            if a.size != 1:
                raise AdapterError(f"{path}: {slot['alpha'][0]} is not a "
                                   "scalar")
            alpha = float(a.astype(mx.float32).reshape(-1)[0].item())
        pairs[stem] = RawPair(stem, down, up, alpha)
    if not pairs:
        raise AdapterError(f"{path} holds no LoRA pair")
    return RawLora(path=path, pairs=pairs, metadata=dict(metadata or {}),
                   stray=tuple(stray))


def read_lora(path: str, *, allow_unmatched: bool = False) -> RawLora:
    """The pairs of a LoRA safetensors file."""
    import os

    from ..vae.reader import VAEError, safetensors_header

    path = os.path.expanduser(str(path))
    try:
        header, _ = safetensors_header(path)
    except VAEError:
        raise AdapterError(f"{path} is not a safetensors file; gmlx image "
                           "reads LoRA adapters from safetensors") from None
    except OSError as e:
        raise AdapterError(f"{path}: {e.strerror or e}") from None
    names = [k for k in header if k != "__metadata__"]
    # Refuse by name from the header, before any tensor loads.
    for key in names:
        kind = _refused_kind(key)
        if kind is not None:
            raise UnsupportedAdapterError(
                f"{path} is a {kind} adapter (key {key}), which gmlx image "
                "does not apply. It applies plain LoRA pairs only", kind)
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


def _dit_target(stem: str, keymap: LoraKeyMap, under: dict):
    """(target name, underscore form) of a DiT stem, or None."""
    for p in UNDERSCORE_PREFIXES:
        if stem.startswith(p):
            name = under.get(stem[len(p):])
            return None if name is None else (name, True)
    rest = stem
    for _ in range(4):
        if rest in keymap.targets:
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


def is_text_stem(stem: str) -> bool:
    return stem.startswith(TEXT_DOTTED + TEXT_UNDERSCORE)


def resolve(raw: RawLora, keymap: LoraKeyMap, *,
            allow_unmatched: bool = False) -> ResolvedLora:
    """Place every pair of ``raw`` on a module of the family, or on a
    module of its text encoder. Raises ``AdapterFamilyError`` when no DiT
    or text stem names a target, and ``UnmatchedKeysError`` for the stems
    left over unless ``allow_unmatched``."""
    under = _underscore_index(keymap.targets)
    text_under = _underscore_index(keymap.text_targets)
    dit, text, unmatched, dropped = [], [], [], []
    for stem, pair in raw.pairs.items():
        if is_text_stem(stem):
            path = _text_target(stem, keymap, text_under)
            if path is None:
                unmatched.append(stem)
            else:
                text.append(Target(path, None, pair))
            continue
        hit = _dit_target(stem, keymap, under)
        if hit is None:
            unmatched.append(stem)
            continue
        name, underscored = hit
        module, band = keymap.targets[name]
        dit.append(Target(module, band, pair))
        if underscored and name in keymap.diffusers_only:
            dropped.append(stem)
    if not dit and not text:
        raise AdapterFamilyError(
            f"{raw.path} is not a {keymap.title} adapter: none of its "
            f"{len(raw.pairs)} targets names a {keymap.title} module "
            f"({_listed(unmatched)})")
    warnings = []
    if unmatched:
        msg = (f"{raw.path}: {len(unmatched)} of {len(raw.pairs)} targets "
               f"name no {keymap.title} module: {_listed(unmatched)}")
        if not allow_unmatched:
            raise UnmatchedKeysError(msg, tuple(unmatched))
        warnings.append(msg + ". They are skipped")
    if dropped:
        warnings.append(
            f"{raw.path}: {len(dropped)} underscore keys spell diffusers "
            f"names ({_listed(dropped)}). gmlx applies them and ComfyUI "
            "drops them, so the image differs from ComfyUI's")
    seen: dict = {}
    for t in dit + text:
        key = (t.module, t.band)
        if key in seen:
            raise AdapterError(
                f"{raw.path}: {t.pair.stem} and {seen[key]} both target "
                f"{t.module}" + ("" if t.band is None else f" band {t.band}"))
        seen[key] = t.pair.stem
    return ResolvedLora(path=raw.path, dit=tuple(dit), text=tuple(text),
                        unmatched=tuple(unmatched), warnings=tuple(warnings))
