"""The owned Gemma 4 model code gives the stock output where stock is right.

gemma4_identity_ops.py runs in a child process that imports gmlx and runs
no loader install, so stock gemma4 there is the pristine mlx-vlm 0.7.3
code over the owned caches. On tiny random configs (dense, MoE, KV-shared,
per-layer inputs, and both containers) the owned and stock trees have the
same parameters, and every call gives bit-identical outputs: prefill,
decode, a verify block with the MTP capture and sinks, rollback,
``logits_to_keep``, ``skip_final_norm`` and two rows in one cache. The MoE
reference carries gmlx's block routing on its SwitchGLUs, and the owned
experts build mlx-lm's SwitchGLU, at every width from 1 to 8. The
reference also takes the owned float32 softcap, which every logit passes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DRIVER = Path(__file__).with_name("gemma4_identity_ops.py")
CASES = ("dense", "kv_shared", "ple", "moe", "gemma4", "gemma4_unified")


@pytest.fixture(scope="module")
def result() -> dict:
    r = subprocess.run([sys.executable, str(DRIVER), "--cpu"],
                       capture_output=True, text=True, env=dict(os.environ),
                       timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout)


@pytest.mark.parametrize("case", CASES)
def test_owned_matches_stock(result, case):
    checks = dict(result[case])
    widths = checks.pop("moe_widths", {})
    checks.pop("switch_glu", None)
    module = checks.pop("language_model_class", None)
    assert checks and all(checks.values()), checks
    assert all(widths.values()), widths
    if module is not None:
        assert module == "gmlx.models.gemma4.language"


def test_the_moe_path_matches_stock_at_widths_1_to_8_on_mlx_lm_switchglu(result):
    moe = result["moe"]
    assert sorted(moe["moe_widths"], key=int) == [str(t) for t in range(1, 9)]
    assert moe["switch_glu"] == ["gmlx._vendor.mlx_lm.models.switch_layers.SwitchGLU"]


def test_a_config_asking_for_the_exact_verifier_is_refused():
    from gmlx._vendor.mlx_vlm.models.gemma4.config import TextConfig

    from gmlx.models.gemma4.language import LanguageModel

    cfg = TextConfig(model_type="gemma4_text", hidden_size=64,
                     num_hidden_layers=2, intermediate_size=128,
                     num_attention_heads=4, head_dim=16, global_head_dim=16,
                     num_key_value_heads=2, num_kv_shared_layers=0,
                     hidden_size_per_layer_input=0, vocab_size=128,
                     exact_speculative_verify=True)
    with pytest.raises(ValueError, match="exact_speculative_verify"):
        LanguageModel(cfg)
