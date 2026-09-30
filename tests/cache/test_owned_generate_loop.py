"""mlx-lm's generate loops give the same tokens on the owned cache classes.

From the binding, mlx_lm's ``stream_generate`` and ``BatchGenerator`` run
on gmlx's cache classes. One script runs a tiny random llama (plain and
with an 8-bit KV cache), a tiny gemma3 (sliding window layers on
RotatingKVCache) and a tiny qwen3_next (GatedDeltaNet layers on
ArraysCache) through both loops greedily, in a child that never imports
gmlx and in a child that imports gmlx first, and the token ids must match.
Both children start clean, so neither depends on what this process
installed, and both run on the GPU when there is one. Under
KQUANT_FORCE_CPU, or without Metal, they run on the CPU device.

Each row of a BatchGenerator run must also give the tokens a
single-stream run of its prompt gives. The ragged batch right-pads its
short prompts, and on stock qwen3_next those pad tokens enter the
GatedDeltaNet state of the short rows, so there the stock batch differs
from single-stream and the owned batch differs from stock.

The owned ArraysCache reads ``left_padding`` and ``lengths`` through an
advance offset, which costs Python calls on the GDN decode path. A decode
step on the qwen3_next model may make at most 2 more Python calls per GDN
layer than stock, counted with ``sys.setprofile``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_SCRIPT = r'''
import contextlib
import importlib
import json
import os
import sys

import mlx.core as mx
from mlx.utils import tree_map

cpu = bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available()
if cpu:
    mx.set_default_device(mx.cpu)
    # Memory policy only, the same in both arms: the stock wired-limit
    # calls read a GPU key the CPU device does not report.
    mx.metal.is_available = lambda: False
mode, arch = sys.argv[1], sys.argv[2]
if mode == "owned":
    import gmlx  # noqa: F401
gen = importlib.import_module("gmlx._vendor.mlx_lm.generate")
if cpu:
    gen.wired_limit = lambda *a, **k: contextlib.nullcontext()
cache_mod = importlib.import_module("gmlx._vendor.mlx_lm.models.cache")
from gmlx._vendor.mlx_lm.tokenizer_utils import TokenizerWrapper


class _Detok:
    def __init__(self, _tok):
        self.last_segment = ""

    def reset(self):
        pass

    def add_token(self, token):
        pass

    def finalize(self):
        pass


def tokenizer():
    tok = TokenizerWrapper.__new__(TokenizerWrapper)
    tok._tokenizer = None
    tok._detokenizer_class = _Detok
    tok._eos_token_ids = set()
    return tok


def build():
    mx.random.seed(0)
    if arch == "llama":
        from gmlx._vendor.mlx_lm.models import llama as m
        args = m.ModelArgs(
            model_type="llama", hidden_size=128, num_hidden_layers=2,
            intermediate_size=128, num_attention_heads=4,
            num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=97)
    elif arch == "gemma3":
        from gmlx._vendor.mlx_lm.models import gemma3_text as m
        args = m.ModelArgs(
            model_type="gemma3_text", hidden_size=64, num_hidden_layers=4,
            intermediate_size=128, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
            vocab_size=97, sliding_window=8, sliding_window_pattern=2)
    else:
        from gmlx._vendor.mlx_lm.models import qwen3_next as m
        args = m.ModelArgs(
            model_type="qwen3_next", hidden_size=64, num_hidden_layers=4,
            intermediate_size=128, num_attention_heads=4,
            linear_num_value_heads=4, linear_num_key_heads=2,
            linear_key_head_dim=16, linear_value_head_dim=16,
            linear_conv_kernel_dim=4, num_experts=0, num_experts_per_tok=1,
            decoder_sparse_step=1, shared_expert_intermediate_size=64,
            mlp_only_layers=[], moe_intermediate_size=64, rms_norm_eps=1e-5,
            vocab_size=97, num_key_value_heads=2, rope_theta=10000.0,
            partial_rotary_factor=0.5, max_position_embeddings=512,
            head_dim=16, full_attention_interval=2)
    model = m.Model(args)
    model.set_dtype(mx.float32)
    # Wider weights than the default init, so greedy decode does not settle
    # on one token and a cache difference shows in the ids.
    model.update(tree_map(lambda p: p * 3.0, model.parameters()))
    mx.eval(model.parameters())
    return model


def greedy(logits):
    return mx.argmax(logits, axis=-1)


model = build()
prompt = [(7 * i + 3) % 97 for i in range(11)]
cache = cache_mod.make_prompt_cache(model)
stream = [int(r.token) for r in gen.stream_generate(
    model, tokenizer(), prompt, max_tokens=12, sampler=greedy,
    prompt_cache=cache)]
classes = sorted({f"{type(c).__module__}.{type(c).__name__}" for c in cache})
kv8 = None
if arch == "llama":
    cache = cache_mod.make_prompt_cache(model)
    kv8 = [int(r.token) for r in gen.stream_generate(
        model, tokenizer(), prompt, max_tokens=12, sampler=greedy,
        prompt_cache=cache, kv_bits=8, kv_group_size=32,
        quantized_kv_start=0)]
    classes += sorted({f"{type(c).__module__}.{type(c).__name__}"
                       for c in cache})

bg = gen.BatchGenerator(model, sampler=greedy)
prompts = [prompt, prompt[:5], [(5 * i + 1) % 97 for i in range(8)]]
uids = bg.insert(prompts, [10] * len(prompts))
rows = {u: [] for u in uids}
while responses := bg.next_generated():
    for r in responses:
        if r.finish_reason != "stop":
            rows[r.uid].append(int(r.token))
bg.close()
batch = [rows[u] for u in uids]
singles = [[int(r.token) for r in gen.stream_generate(
    model, tokenizer(), p, max_tokens=10, sampler=greedy,
    prompt_cache=cache_mod.make_prompt_cache(model))] for p in prompts]

calls = None
if arch == "qwen3_next":
    step_cache = cache_mod.make_prompt_cache(model)
    mx.eval(model(mx.array([prompt]), cache=step_cache))
    tok = mx.array([[prompt[-1]]])
    mx.eval(model(tok, cache=step_cache))
    counts = []
    for _ in range(3):
        n = [0]

        def prof(frame, event, arg):
            if event == "call":
                n[0] += 1

        sys.setprofile(prof)
        out = model(tok, cache=step_cache)
        sys.setprofile(None)
        mx.eval(out)
        counts.append(n[0])
    calls = counts
print(json.dumps({"stream": stream, "kv8": kv8, "batch": batch,
                  "singles": singles, "classes": classes, "calls": calls,
                  "device": "cpu" if cpu else "gpu",
                  "gmlx": any(m == "gmlx" for m in sys.modules)}))
'''

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_GDN_LAYERS = 2


def _run(mode, arch):
    env = dict(os.environ)
    cwd = None
    if mode == "stock":
        keep = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                if p and os.path.abspath(p) != _ROOT]
        env["PYTHONPATH"] = os.pathsep.join(keep)
        cwd = "/"
    r = subprocess.run([sys.executable, "-c", _SCRIPT, mode, arch],
                       capture_output=True, text=True, timeout=600, env=env,
                       cwd=cwd)
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["gmlx"] is (mode == "owned")
    assert out["device"] == _DEVICE, out["device"]
    return out


def _device():
    import mlx.core as mx

    if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
        return "cpu"
    return "gpu"


_DEVICE = _device()


@pytest.fixture(scope="module", params=["llama", "gemma3", "qwen3_next"])
def runs(request):
    arch = request.param
    return arch, _run("stock", arch), _run("owned", arch)


def test_the_generate_loops_run_on_owned_classes(runs):
    arch, stock, owned = runs
    assert all(c.startswith("gmlx._vendor.mlx_lm.models.cache.") for c in stock["classes"])
    assert all(c.startswith("gmlx.cache.kv_cache.") for c in owned["classes"])
    names = {c.rsplit(".", 1)[1] for c in owned["classes"]}
    want = {"llama": {"KVCache", "QuantizedKVCache"},
            "gemma3": {"KVCache", "RotatingKVCache"},
            "qwen3_next": {"KVCache", "ArraysCache"}}[arch]
    assert names == want, (arch, owned["classes"])


def test_stream_generate_tokens_match_stock(runs):
    arch, stock, owned = runs
    assert len(set(stock["stream"])) > 1, arch
    assert owned["stream"] == stock["stream"], arch
    assert owned["kv8"] == stock["kv8"], arch


def test_batch_generator_rows_match_single_stream(runs):
    arch, stock, owned = runs
    assert owned["batch"] == owned["singles"], arch


def test_batch_generator_tokens_match_stock(runs):
    arch, stock, owned = runs
    if arch == "qwen3_next":
        # The stock ArraysCache masks by the zero left padding a merge
        # seeded, not by the lengths prepare set, so the short rows break.
        assert stock["batch"][0] == stock["singles"][0]
        assert stock["batch"][1:] != stock["singles"][1:]
        return
    assert owned["batch"] == stock["batch"], arch


def test_gdn_decode_step_adds_at_most_two_calls_per_layer(runs):
    arch, stock, owned = runs
    if arch != "qwen3_next":
        pytest.skip("GDN layers only")
    assert len(set(stock["calls"])) == 1 and len(set(owned["calls"])) == 1
    extra = owned["calls"][0] - stock["calls"][0]
    assert extra <= 2 * _GDN_LAYERS, (stock["calls"], owned["calls"])
