#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Write the mlx-vlm 0.7.3 Gemma 4 model files into gmlx/models/gemma4.

SRC is the ``mlx_vlm`` package directory of mlx-vlm 0.7.3, for example the
installed site-packages copy or an unzipped wheel. ``language.py`` is all
of ``models/gemma4/language.py``, ``gemma4.py`` all of
``models/gemma4/gemma4.py`` and ``unified.py`` all of
``models/gemma4_unified/gemma4_unified.py`` but its vision embedder. Each
edit is an exact, counted string replacement, so a source of another shape
fails here instead of producing a different copy.

``--check`` compares against the files in the tree, prints the drift
baseline as JSON (the version, the sha256 of each source file and every
counted edit) and exits 1 on a difference.
The files in the tree carry the owned edits gmlx made after vendoring,
so ``--check`` reports those edits and exits 1 by design. It is not the
check for upstream changes: ``scripts/drift_report.py`` is.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

VERSION = "0.7.3"

HEADER = """# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2025 Prince Canuma (mlx-vlm, MIT)
# Vendored from mlx-vlm {version}, mlx_vlm/{src}.
# Source sha256: {sha}
"""

# Import rewrites are single-line literals, so no line of this file starts
# with an upstream import (tests/test_pyright_scope.py scans for those).
LANGUAGE_EDITS = [
    ("from ..base import (\n"
     "    LanguageModelOutput,\n"
     "    create_attention_mask,\n"
     "    create_causal_mask,\n"
     "    scaled_dot_product_attention,\n"
     ")\n"
     "from ..cache import KVCache, RotatingKVCache\n"
     "from ..rope_utils import initialize_rope\n"
     "from .config import TextConfig\n"
     "from .speculative_verifier import Gemma4ExactSpeculativeVerifier\n"
     "\n"
     "_EXACT_SPECULATIVE_VERIFIER = Gemma4ExactSpeculativeVerifier()\n",
     "from mlx_vlm.models.base import LanguageModelOutput\n"
     "from mlx_vlm.models.gemma4.config import TextConfig\n"
     "from mlx_vlm.models.rope_utils import initialize_rope\n"
     "\n"
     "from gmlx.cache.attention import create_attention_mask, "
     "scaled_dot_product_attention\n"
     "from gmlx.cache.kv_cache import KVCache, RotatingKVCache, "
     "create_causal_mask\n"),
    # mlx-lm's SwitchGLU routes a block of positions with one gather per
    # projection at every width, as the vendored qwen3.5 MoE builds it.
    ("        from ..switch_layers import SwitchGLU\n",
     "        # gmlx: mlx-lm's SwitchGLU takes a verify block in one gather.\n"
     "        from mlx_lm.models.switch_layers import SwitchGLU\n"),
    ("    def __init__(self, config: TextConfig):\n"
     "        super().__init__()\n"
     "        self.config = config\n"
     "        self.model_type = config.model_type\n",
     "    def __init__(self, config: TextConfig):\n"
     "        super().__init__()\n"
     "        # gmlx: the exact speculative verifier is not vendored.\n"
     '        if getattr(config, "exact_speculative_verify", False):\n'
     "            raise ValueError(\n"
     '                "gmlx does not run the exact speculative verifier; the "\n'
     '                "config sets exact_speculative_verify"\n'
     "            )\n"
     "        self.config = config\n"
     "        self.model_type = config.model_type\n"),
    ('        if kwargs.pop("speculative_verify", False) and getattr(\n'
     '            self.config, "exact_speculative_verify", False\n'
     "        ):\n"
     "            return _EXACT_SPECULATIVE_VERIFIER(\n"
     "                self,\n"
     "                inputs,\n"
     "                cache=cache,\n"
     "                input_embeddings=inputs_embeds,\n"
     "                capture_layer_ids=capture_layer_ids,\n"
     "            )\n",
     "        # gmlx: the exact speculative verifier is not vendored.\n"
     '        kwargs.pop("speculative_verify", False)\n'),
    # ASCII for tests/test_ascii_hygiene.py.
    ("        # \u2014 the recorded value is the LAST",
     "        # so the recorded value is the LAST"),
]

GEMMA4_EDITS = [
    ("from ..base import InputEmbeddingsFeatures\n"
     "from .audio import AudioEncoder\n"
     "from .config import ModelConfig\n"
     "from .language import LanguageModel, RMSNormNoScale\n"
     "from .vision import VisionModel\n",
     "from mlx_vlm.models.base import InputEmbeddingsFeatures\n"
     "from mlx_vlm.models.gemma4.audio import AudioEncoder\n"
     "from mlx_vlm.models.gemma4.config import ModelConfig\n"
     "from mlx_vlm.models.gemma4.vision import VisionModel\n"
     "\n"
     "from .language import LanguageModel, RMSNormNoScale\n"),
]

UNIFIED_IMPORTS = [
    ("from ..base import InputEmbeddingsFeatures\n"
     "from ..gemma4.gemma4 import MultimodalEmbedder, masked_scatter\n"
     "from ..gemma4.language import LanguageModel\n"
     "from .config import ModelConfig, VisionConfig\n",
     "from mlx_vlm.models.base import InputEmbeddingsFeatures\n"
     "\n"
     "# The four configs are here for update_module_configs.\n"
     "from mlx_vlm.models.gemma4_unified.config import (  # noqa: F401\n"
     "    AudioConfig,\n"
     "    ModelConfig,\n"
     "    TextConfig,\n"
     "    VisionConfig,\n"
     ")\n"
     "from mlx_vlm.models.gemma4_unified.gemma4_unified import "
     "VisionEmbedder\n"
     "\n"
     "from .gemma4 import MultimodalEmbedder, masked_scatter\n"
     "from .language import LanguageModel\n"),
]
# The unified vision tower stays in mlx-vlm: the class and the alias go.
EMBEDDER_FIRST = "class VisionEmbedder(nn.Module):\n"
EMBEDDER_NEXT = "class Model(nn.Module):\n"
EMBEDDER_LINES = 40
ALIAS = "\n\nVisionModel = VisionEmbedder\n"

LANGUAGE_DOC = """mlx-vlm {version} Gemma 4 text model, owned by gmlx.

All of ``mlx_vlm/models/gemma4/language.py``. Edits to the upstream file:
imports are absolute, and the cache classes, the masks and the SDPA come
from gmlx.cache. ``Experts`` builds mlx-lm's ``SwitchGLU``, which routes
a block of positions with one gather per projection at every width. The
exact speculative verifier is not vendored: ``speculative_verify`` is
still taken from the call's keywords, and a config that sets
``exact_speculative_verify`` is refused. Comments are ASCII. Each gmlx
change past the imports carries a comment where it sits.
"""

GEMMA4_DOC = """mlx-vlm {version} Gemma 4 multimodal container, owned by gmlx.

All of ``mlx_vlm/models/gemma4/gemma4.py``. Edits to the upstream file:
imports are absolute; the configs, the vision tower and the audio encoder
stay mlx-vlm's, and the language model is the owned one.
"""

UNIFIED_DOC = """mlx-vlm {version} Gemma 4 unified container, owned by gmlx.

``mlx_vlm/models/gemma4_unified/gemma4_unified.py`` without its vision
embedder. Edits to the upstream file: imports are absolute; the configs
and ``VisionEmbedder`` stay mlx-vlm's, and the embedder class and its
``VisionModel`` alias are not copied; the language model and the
multimodal embedder are the owned ones.
"""

# owned file -> (source under SRC, docstring)
FILES = {
    "language.py": ("models/gemma4/language.py", LANGUAGE_DOC),
    "gemma4.py": ("models/gemma4/gemma4.py", GEMMA4_DOC),
    "unified.py": ("models/gemma4_unified/gemma4_unified.py", UNIFIED_DOC),
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sub(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"expected 1 x {old!r}, found {n}")
    return text.replace(old, new)


def sub_all(text: str, pairs) -> str:
    for old, new in pairs:
        text = sub(text, old, new)
    return text


def embedder_block(src: str) -> str:
    """The unified VisionEmbedder class through the blank lines after it."""
    start = src.index(EMBEDDER_FIRST)
    end = src.index(EMBEDDER_NEXT)
    block = src[start:end]
    if block.count("\n") != EMBEDDER_LINES:
        raise SystemExit(f"VisionEmbedder spans {block.count(chr(10))} "
                         f"lines, expected {EMBEDDER_LINES}")
    return block


def source_version(src: Path) -> str:
    m = re.search(r'__version__ = "([^"]+)"', (src / "version.py").read_text())
    if m is None:
        raise SystemExit(f"{src}/version.py names no __version__")
    return m.group(1)


def edits_for(name: str, src_text: str) -> list[tuple[str, str]]:
    if name == "language.py":
        return LANGUAGE_EDITS
    if name == "gemma4.py":
        return GEMMA4_EDITS
    return UNIFIED_IMPORTS + [(embedder_block(src_text), ""), (ALIAS, "")]


def build(src: Path):
    """Return (files, baseline) for the package directory ``src``."""
    version = source_version(src)
    if version != VERSION:
        raise SystemExit(f"{src} is mlx-vlm {version}, expected {VERSION}")
    files = {}
    baseline = {"package": "mlx-vlm", "version": version,
                "sources": {}, "outputs": {}}
    for name, (rel, doc) in FILES.items():
        text = (src / rel).read_text()
        edits = edits_for(name, text)
        body = sub_all(text, edits)
        head = HEADER.format(version=version, src=rel, sha=sha256(text))
        out = head + '"""' + doc.format(version=version) + '"""\n\n' + body
        if not out.isascii():
            raise SystemExit(f"{name}: non-ASCII left after the edits")
        files[name] = out
        baseline["sources"][rel] = {"sha256": sha256(text),
                                    "lines": text.count("\n")}
        baseline["outputs"][f"gmlx/models/gemma4/{name}"] = {
            "source": rel,
            "edits": [{"old": o, "new": n, "count": 1} for o, n in edits],
        }
    return files, baseline


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path,
                    help=f"mlx_vlm package directory of mlx-vlm {VERSION}")
    ap.add_argument("--dst", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "gmlx/models/gemma4",
                    help="output directory (default: gmlx/models/gemma4)")
    ap.add_argument("--check", action="store_true",
                    help="compare with the files in --dst and print the "
                         "drift baseline instead of writing")
    args = ap.parse_args()
    files, baseline = build(args.src)
    stale = []
    for name, text in files.items():
        path = args.dst / name
        out = baseline["outputs"][f"gmlx/models/gemma4/{name}"]
        out["sha256"] = sha256(text)
        if args.check:
            out["match"] = path.exists() and path.read_text() == text
            if not out["match"]:
                stale.append(name)
        else:
            path.write_text(text)
            print(f"wrote {path}")
    if args.check:
        print(json.dumps(baseline, indent=1, sort_keys=True))
    if stale:
        print("differs from the generated copy: " + ", ".join(stale),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
