"""What a gmlx command prints before its own output."""
import importlib.util
import subprocess
import sys

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                    reason="the note comes from an installed torch")
def test_a_command_prints_no_torch_redirect_note(tmp_path):
    # A missing file ends the command after the imports that log the note.
    out = subprocess.run(
        [sys.executable, "-m", "gmlx", "run", str(tmp_path / "none.gguf")],
        capture_output=True, text=True, timeout=300)
    assert "no such file" in out.stderr
    assert "Redirects are currently not supported" not in out.stdout + out.stderr
