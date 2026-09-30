"""The copy under gmlx/_vendor matches its manifest: every copied file has
its recorded sha256, every file under the copy is listed, every edited
file carries a ``# gmlx:`` marker, and the version files name the recorded
releases. Only scripts/vendor_upstream.py writes the manifest, on a
deliberate port."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "vendor_upstream.py"


def _tool():
    spec = importlib.util.spec_from_file_location("_vendor_upstream", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_copy_matches_its_manifest():
    problems = _tool().verify(_ROOT)
    assert not problems, "\n".join(problems[:40])
