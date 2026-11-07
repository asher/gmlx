"""gmlx kld against the llama-perplexity fixture on Qwen3.5-9B (GGUF tier).

The fixture names its files by name and size. Each test finds them under
KQUANT_TEST_GGUF_DIR and skips when one is missing. The GGUFs are over
the 512 MiB hashing limit, so only the text is checked by sha256.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "kld" / "llama_perplexity.json"
# Mean KLD may differ from llama.cpp's by this share plus the floor.
KLD_REL_TOL = 0.05
KLD_ABS_FLOOR = 0.001
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


def test_mean_kld_matches_llama_perplexity(case, monkeypatch):
    import mlx.core as mx

    from gmlx.commands import kld as cmd

    row = case["row"]
    active_at_load = []
    real_load = cmd._load

    def load(path, args):
        active_at_load.append(mx.get_active_memory())
        return real_load(path, args)

    monkeypatch.setattr(cmd, "_load", load)
    args = cmd.build_parser().parse_args(
        [case["model"], "--reference", case["reference"], "--text",
         case["text"], "--ctx", str(row["ctx"])])
    mx.clear_cache()
    mx.reset_peak_memory()
    result = cmd.run(args, lambda rec: None, lambda msg: None)

    assert result["chunks"] == row["chunks"]
    assert result["positions"] == row["chunks"] * (row["ctx"] - 1 - row["ctx"] // 2)
    want = row["mean_kld"]
    assert abs(result["mean_kld"] - want) <= KLD_REL_TOL * want + KLD_ABS_FLOOR
    assert abs(result["same_top_p"] - row["same_top_p"]) <= SAME_TOP_ABS_TOL
    # The reference is gone before MODEL loads.
    assert active_at_load[1] - active_at_load[0] < 10**9
    assert mx.get_peak_memory() < BUDGET_BYTES
