"""A load on every route leaves stock Gemma 4 as a plain import leaves it.

gmlx builds its own Gemma 4 classes on every route and patches nothing in
the stock gemma4 modules. A child process runs each route on tiny float32
models: the text and serve loaders on a tiny llama GGUF (the loader and
server installs), the owned text model, the MTP target with the stock
assistant drafter bound to it, the gemma4 and gemma4_unified containers
plain and as speculative targets, the two MLX-checkpoint routes, and the
MLX checkpoints the mlx-lm and mlx-vlm servers load from a path that is
not a GGUF. It then fingerprints every stock gemma4 module global and class attribute:
functions by name and code, classes by name, values by repr. A second
child imports the same modules after gmlx and nothing else, since the
cache bind that ``import gmlx`` installs rebinds the stock cache names in
both. The two must be equal, and the loaded child must not have imported
mlx-lm's gemma4 modules.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from test_gemma4_mlx_checkpoint import _checkpoint, _vlm_checkpoint
from test_loader_api import _mint_tiny_llama

_CHILD = textwrap.dedent('''
    import functools, hashlib, importlib, json, marshal, sys, types

    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    import gmlx  # noqa: F401

    mode, paths = sys.argv[1], json.loads(sys.argv[2])
    STOCK = {
        "mlx_vlm.models.gemma4": (
            "config", "language", "gemma4", "vision", "audio",
            "audio_feature_extractor", "processing_gemma4",
            "speculative_verifier"),
        "mlx_vlm.models.gemma4_unified": (
            "config", "gemma4_unified", "processing_gemma4_unified"),
        "mlx_vlm.speculative.drafters.gemma4_assistant": (
            "config", "gemma4_assistant", "masked_embedder", "masks"),
    }

    def text_config(**kw):
        return {
            "model_type": "gemma4_text", "hidden_size": 64,
            "num_hidden_layers": 4, "intermediate_size": 128,
            "num_attention_heads": 4, "head_dim": 16, "global_head_dim": 16,
            "num_key_value_heads": 2, "num_global_key_value_heads": 2,
            "num_kv_shared_layers": 0, "vocab_size": 128,
            "vocab_size_per_layer_input": 128,
            "hidden_size_per_layer_input": 0, "sliding_window": 64,
            "sliding_window_pattern": 2, "attention_k_eq_v": False,
            "rms_norm_eps": 1e-6, "tie_word_embeddings": True, **kw}

    def container(model_type):
        text_type = ("gemma4_unified_text" if model_type == "gemma4_unified"
                     else "gemma4_text")
        return {"model_type": model_type, "vocab_size": 128,
                "text_config": text_config(model_type=text_type),
                "vision_config": {}}

    def run_routes():
        from mlx.utils import tree_flatten
        from mlx_lm.utils import load_config

        from gmlx.config import build_config
        from gmlx.distill.student import load_mlx_student
        from gmlx.load import vlm
        from gmlx.load.loader import _install_and_load, build_model, load_model
        from gmlx.load.mtp_target import _build_mtp_target
        from gmlx.models.vlm_text_only import ModelConfig, TextOnlyModel
        from gmlx.serve import bridge_vlm
        from gmlx.serve.patches import install_server_patches
        from gmlx.spec import mtp_load
        from gmlx.spec.drafter_protocol import DrafterAdapter, validate_drafter
        from gmlx.tui.reasoning import install_template_default_thinking
        from gmlx.upstream.single_bos_fix import install_single_bos_fix
        from gmlx.upstream.system_content_fix import install_system_content_fix

        def log(*a, **k):
            pass

        def install(model):
            _install_and_load(model, dict(tree_flatten(model.parameters())),
                              {}, log=log, sanitize=False)
            return model

        ids = mx.array([[2, 17, 42, 99, 7, 63]])
        # Text and serve loaders, with the server installs.
        load_model(paths["llama"])
        bridge_vlm.load_serveable_model(paths["llama"])
        bridge_vlm.install_gguf_server_bridge()
        install_server_patches(build_config(
            {"server": {"host": "127.0.0.1", "port": 8080},
             "models": {"tiny": {"path": paths["llama"]}}}))
        # The owned text model.
        model = install(build_model(text_config())[0])
        mx.eval(model(ids))
        # The MTP target and the stock assistant drafter bound to it.
        cfg = text_config()
        target = install(_build_mtp_target(cfg)[0])
        mtp_load._arm_spec_target(target, cfg, stock_vlm_regime=False, log=log)
        mx.eval(target(ids).logits)
        dcfg = importlib.import_module(
            "mlx_vlm.speculative.drafters.gemma4_assistant.config")
        dmod = importlib.import_module(
            "mlx_vlm.speculative.drafters.gemma4_assistant.gemma4_assistant")
        drafter = dmod.Gemma4AssistantDraftModel(
            dcfg.Gemma4AssistantConfig.from_dict({
                "backbone_hidden_size": 64,
                "text_config": text_config(num_hidden_layers=2)}))
        install(drafter).bind(target)
        mtp_load._stamp_mtp_width_cap(drafter, "gemma4_assistant",
                                      target=target, log=log)
        validate_drafter(DrafterAdapter(drafter))
        # The containers, plain and as speculative targets.
        install_template_default_thinking()
        install_system_content_fix()
        install_single_bos_fix()
        for model_type in ("gemma4", "gemma4_unified"):
            plain = install(vlm.build_vlm_model(container(model_type))[0])
            mx.eval(plain.language_model(ids).logits)
            spec = vlm.build_vlm_model(container(model_type))[0]
            vlm._swap_spec_language_model(spec, model_type, log=log)
            install(spec)
            mtp_load._arm_spec_target(spec, container(model_type)["text_config"],
                                      stock_vlm_regime=True, log=log)
            mx.eval(spec.language_model(ids).logits)
        # The MLX-checkpoint routes.
        mx.eval(load_mlx_student(paths["flat"])[0](ids))
        TextOnlyModel(ModelConfig.from_dict(load_config(
            __import__("pathlib").Path(paths["container"]))))
        # The MLX checkpoints the servers load from a path that is not a
        # GGUF: the mlx-lm server loader, and mlx-vlm's loader under the
        # resolver the mlx-vlm bridge installs on that path.
        from mlx_lm import server as lm_server
        from mlx_vlm.utils import load_model as load_vlm_checkpoint

        from gmlx.serve import bridge_lm

        bridge_lm.install_gguf_bridge()
        provider = types.SimpleNamespace(
            is_distributed=False, _tokenizer_config={},
            cli_args=types.SimpleNamespace(use_default_chat_template=False))
        lm_server.ModelProvider._load(provider, paths["container"])
        mx.eval(provider.model(ids))
        vlm.install_owned_vlm_resolver()
        for key in ("vlm_gemma4", "vlm_gemma4_unified"):
            model = load_vlm_checkpoint(__import__("pathlib").Path(paths[key]))
            mx.eval(model.language_model(ids).logits)

    def code(fn):
        c = getattr(fn, "__code__", None)
        return hashlib.sha256(marshal.dumps(c)).hexdigest()[:16] if c else None

    def key(v):
        return json.dumps(v, sort_keys=True)

    def fp(v, depth=0):
        if v is None or isinstance(v, (bool, int, float, str, bytes)):
            return repr(v)
        if isinstance(v, types.ModuleType):
            return "module " + v.__name__
        if isinstance(v, (staticmethod, classmethod)):
            return [type(v).__name__, fp(v.__func__, depth)]
        if isinstance(v, property):
            return ["property", fp(v.fget, depth), fp(v.fset, depth)]
        if isinstance(v, functools.partial):
            return ["partial", fp(v.func, depth),
                    [fp(a, depth + 1) for a in v.args],
                    sorted((k, fp(a, depth + 1)) for k, a in v.keywords.items())]
        if isinstance(v, (types.FunctionType, types.MethodType)):
            fn = getattr(v, "__func__", v)
            return ["fn", fn.__module__, fn.__qualname__, code(fn),
                    sorted(vars(fn))]
        if isinstance(v, type):
            return "class " + v.__module__ + "." + v.__qualname__
        if depth < 3 and isinstance(v, (list, tuple)):
            return [type(v).__name__, [fp(x, depth + 1) for x in v]]
        if depth < 3 and isinstance(v, (set, frozenset)):
            return [type(v).__name__, sorted(key(fp(x, depth + 1)) for x in v)]
        if depth < 3 and isinstance(v, dict):
            return ["dict", sorted(([fp(k, depth + 1), fp(x, depth + 1)]
                                    for k, x in v.items()), key=key)]
        t = type(v)
        return ["object", t.__module__, t.__qualname__,
                getattr(v, "__module__", None),
                getattr(v, "__qualname__", getattr(v, "__name__", None))]

    if mode == "loaded":
        run_routes()
    stock_lm = sorted(m for m in ("mlx_lm.models.gemma4_text",
                                  "mlx_lm.models.gemma4") if m in sys.modules)
    prints = {}
    for pkg, subs in STOCK.items():
        for name in (pkg,) + tuple(pkg + "." + s for s in subs):
            mod = importlib.import_module(name)
            for attr, v in sorted(vars(mod).items()):
                if attr in ("__builtins__", "__loader__", "__spec__"):
                    continue
                prints[name + ":" + attr] = fp(v)
                if isinstance(v, type) and v.__module__ == name:
                    for cattr, cv in sorted(vars(v).items()):
                        if cattr in ("__dict__", "__weakref__"):
                            continue
                        prints[name + ":" + attr + "." + cattr] = fp(cv)
    print(json.dumps({"stock_lm": stock_lm, "prints": prints}))
''')


def _child(mode: str, paths: dict) -> dict:
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    r = subprocess.run([sys.executable, "-c", _CHILD, mode, json.dumps(paths)],
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("stock_untouched")
    llama = d / "tiny-llama.gguf"
    _mint_tiny_llama(str(llama))
    paths = {"llama": str(llama),
             "flat": str(_checkpoint(d / "flat", False, False)),
             "container": str(_checkpoint(d / "container", True, True))}
    for model_type in ("gemma4", "gemma4_unified"):
        paths[f"vlm_{model_type}"] = str(
            _vlm_checkpoint(d / model_type, model_type)[0])
    return {mode: _child(mode, paths) for mode in ("clean", "loaded")}


def test_every_route_leaves_stock_gemma4_as_a_plain_import(runs):
    clean, loaded = runs["clean"]["prints"], runs["loaded"]["prints"]
    assert len(clean) > 500
    changed = {k: (clean.get(k), loaded.get(k))
               for k in sorted(set(clean) | set(loaded))
               if clean.get(k) != loaded.get(k)}
    assert changed == {}


def test_no_route_imports_the_stock_mlx_lm_gemma4_modules(runs):
    assert runs["loaded"]["stock_lm"] == []
