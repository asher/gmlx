"""On-disk model registry for the e2e harness.

Logical handles -> candidate paths under a models root (default ``~/llm/gguf``).
First existing candidate wins; a missing handle resolves to ``None`` so a tier
whose models aren't present is *skipped*, not failed. Nothing here loads a model -
it's pure path resolution, so it runs on any machine for ``--dry-run``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_ROOT = "~/llm/gguf"

# handle -> ordered candidate relative paths (first existing wins)
_CANDIDATES = {
    # small dense
    "qwen3_0_6b_q4": ["qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf"],
    "qwen3_0_6b_q8": ["qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf"],
    "gemma3_1b": ["gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf"],
    # gemma-4 E2B text (tied embeds) - UD variants
    "gemma4_e2b": [
        "gemma-4-E2B-it-UD-Q6_K_XL/gemma-4-E2B-it-UD-Q6_K_XL.gguf",
        "gemma-4-E2B-it-UD-Q8_K_XL.gguf",
    ],
    # gemma-4 E2B companions
    "gemma4_e2b_mmproj": ["mmproj-gemma-4-E2B-it-bf16.gguf"],
    "gemma4_e2b_assistant": ["gemma-4-E2B-it-assistant.Q8_0.gguf"],
    # a stronger local judge if available (preferred), else fall back to small
    "gemma4_12b": ["gemma-4-12b-it-GGUF/gemma-4-12b-it-Q6_K.gguf"],
}

# Canonical download source per handle: an ``hf:<org>/<repo>/<file>`` ref whose
# remote filename equals the handle's candidate basename, so `gmlx pull <ref>
# --to <root>/<dir>` lands the file exactly where the registry looks for it. A handle
# with no public source (None) must be provided by the user - `print_bootstrap` says so.
# The small dense text models are sourced from public GGUF repos; the gemma-4 family is
# left unset (no public GGUF distribution): fill in your own `hf:` ref or drop the GGUF in by hand.
_SOURCES = {
    "qwen3_0_6b_q4": "hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf",
    "qwen3_0_6b_q8": "hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
    "gemma3_1b": "hf:ggml-org/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf",
    "gemma4_e2b": None,
    "gemma4_e2b_mmproj": None,
    "gemma4_e2b_assistant": None,
    "gemma4_12b": None,
}

# Preference order for the default LLM judge (a bigger, coherent model judges
# better; fall back to the small ones so the harness still runs on a lean box).
_JUDGE_PREFERENCE = ["gemma4_12b", "gemma4_e2b", "qwen3_0_6b_q8", "qwen3_0_6b_q4"]


@dataclass
class ModelRegistry:
    root: str = DEFAULT_ROOT

    def _root(self) -> str:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(self.root)))

    def find(self, handle: str) -> Optional[str]:
        root = self._root()
        for rel in _CANDIDATES.get(handle, []):
            cand = os.path.join(root, rel)
            if os.path.exists(cand):
                return cand
        return None

    def require(self, handle: str) -> str:
        p = self.find(handle)
        if p is None:
            raise FileNotFoundError(
                f"model handle {handle!r} not found under {self._root()} "
                f"(candidates: {_CANDIDATES.get(handle)})")
        return p

    def have(self, *handles: str) -> bool:
        return all(self.find(h) is not None for h in handles)

    def missing(self, *handles: str) -> list:
        return [h for h in handles if self.find(h) is None]

    def default_judge(self) -> Optional[str]:
        for h in _JUDGE_PREFERENCE:
            p = self.find(h)
            if p is not None:
                return p
        return None

    def inventory(self) -> dict:
        """handle -> resolved path or None (for the dry-run plan + report header)."""
        return {h: self.find(h) for h in _CANDIDATES}

    def _dest_for(self, handle: str) -> str:
        """The directory the handle's first candidate lives in, under the root -
        where `gmlx pull --to` should drop the download so `find` resolves it."""
        rel = (_CANDIDATES.get(handle) or [handle])[0]
        return os.path.join(self._root(), os.path.dirname(rel))

    def pull_command(self, handle: str) -> Optional[str]:
        """A ready-to-run `gmlx pull` line that fetches ``handle`` to the exact
        path the registry expects, or None when the handle has no public source."""
        src = _SOURCES.get(handle)
        if not src:
            return None
        return f"gmlx pull {src} --to {self._dest_for(handle)}"

    def bootstrap_plan(self, handles=None) -> list:
        """[(handle, present, pull_command_or_None, expected_path)] for the given
        handles (default: all). Drives `print_bootstrap` and `--print-pull`."""
        handles = list(handles) if handles is not None else list(_CANDIDATES)
        plan = []
        for h in handles:
            p = self.find(h)
            expected = os.path.join(self._root(), (_CANDIDATES.get(h) or [h])[0])
            plan.append((h, p is not None, self.pull_command(h), p or expected))
        return plan

    def print_bootstrap(self, handles=None) -> None:
        """Print copy-paste pull commands for the handles not yet on disk (and a clear
        'provide your own' note for the sourceless ones). Pure stdout; no network."""
        plan = self.bootstrap_plan(handles)
        missing = [row for row in plan if not row[1]]
        if not missing:
            print(f"all models present under {self._root()} - nothing to fetch.")
            return
        print(f"# models root: {self._root()}")
        print(f"# {len(missing)} model(s) missing. Run the lines below to fetch them:\n")
        for handle, _present, cmd, expected in missing:
            if cmd:
                print(cmd)
            else:
                print(f"# {handle}: no public source - drop a GGUF at {expected}")
        have_unsourced = any(not cmd for _h, present, cmd, _e in plan if not present)
        if have_unsourced:
            print("\n# (the gemma-4 tiers - judged/vlm/mtp - have no public GGUF source; "
                  "set their `hf:` ref in models.py _SOURCES, or skip those tiers.)")
