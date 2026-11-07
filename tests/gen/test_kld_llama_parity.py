"""gmlx kld against the llama-perplexity fixture on Qwen3.5-9B (GGUF tier).

The fixture names its files by name and size. Each test finds them under
KQUANT_TEST_GGUF_DIR and skips when one is missing. The GGUFs are over
the 512 MiB hashing limit, so only the text is checked by sha256.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "kld" / "llama_perplexity.json"
# Mean KLD may differ from llama.cpp's by this share plus the floor.
KLD_REL_TOL = 0.03
KLD_ABS_FLOOR = 0.0003
SAME_TOP_ABS_TOL = 0.01
# A 32 GB Mac holds the BF16 reference or the test model, not both.
BUDGET_BYTES = 32 * 10**9


@pytest.fixture(scope="module")
def case(gguf_dir):
    fixture = json.loads(FIXTURE.read_text())
    row = fixture["rows"][0]
    files = fixture["files"]
    text = FIXTURE.parent / row["text"]
    if hashlib.sha256(text.read_bytes()).hexdigest() != files[row["text"]]["sha256"]:
        pytest.fail(f"{text.name} is not the text the fixture recorded")

    def find(name):
        size = files[name]["size"]
        for p in sorted(gguf_dir.rglob(name)):
            if p.stat().st_size == size:
                return str(p)
        pytest.skip(f"no {name} of {size} bytes under KQUANT_TEST_GGUF_DIR")

    return {"row": row, "model": find(row["model"]),
            "reference": find(row["reference"]), "text": str(text)}


def test_the_text_tokenizes_to_the_fixture_ids(case):
    from gmlx.commands.kld import tokenize
    from gmlx.distill.tokens import tokenizer_from_gguf
    from gmlx.gen.kld import read_text

    row = case["row"]
    ids, _ = tokenize(tokenizer_from_gguf(case["model"]), read_text(case["text"]))
    ids = np.asarray(ids[:row["ctx"] * row["chunks"]], dtype="<i4")
    assert ids[:32].tolist() == row["tokens_head"]
    assert hashlib.sha256(ids.tobytes()).hexdigest() == row["tokens_sha256"]


# The scoring runs in a child process: MLX reads MLX_ENABLE_TF32 once per
# process, and the float32 head must not run as TF32.
_CHILD = """
import json, os, sys
import mlx.core as mx
from gmlx.commands import kld as cmd
if os.environ.get("KQUANT_FORCE_CPU"):
    mx.set_default_device(mx.cpu)
active_at_load = []
real_load = cmd._load
def load(src, args, *rest):
    active_at_load.append(mx.get_active_memory())
    return real_load(src, args, *rest)
cmd._load = load
args = cmd.parse_args(sys.argv[1:])
mx.clear_cache()
mx.reset_peak_memory()
results = cmd.run(args, lambda rec: None, lambda msg: None)
print(json.dumps({"results": results, "active_at_load": active_at_load,
                  "peak": mx.get_peak_memory()}))
"""


def test_mean_kld_matches_llama_perplexity(case):
    row = case["row"]
    env = dict(os.environ, MLX_ENABLE_TF32="0")
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, case["model"], "--reference",
         case["reference"], "--text", case["text"], "--ctx", str(row["ctx"])],
        capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.splitlines()[-1])
    (result,) = out["results"]
    want = row["mean_kld"]
    print(f"mean_kld {result['mean_kld']:.6f} (fixture {want:.6f}), "
          f"same_top_p {result['same_top_p']:.5f} (fixture {row['same_top_p']:.5f})")
    assert result["chunks"] == row["chunks"]
    assert result["positions"] == row["chunks"] * (row["ctx"] - 1 - row["ctx"] // 2)
    assert abs(result["mean_kld"] - want) <= KLD_REL_TOL * want + KLD_ABS_FLOOR, \
        result["mean_kld"]
    assert abs(result["same_top_p"] - row["same_top_p"]) <= SAME_TOP_ABS_TOL
    # The reference is gone before MODEL loads.
    first, second = out["active_at_load"]
    assert second - first < 10**9
    assert out["peak"] < BUDGET_BYTES
