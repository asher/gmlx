# SPDX-License-Identifier: Apache-2.0
"""The ModelPatcher of a gmlx MODEL, whose weights live on the gmlx server.

Its load and unload methods touch no weights, and both of its devices are
the cpu device, so ComfyUI frees nothing on its behalf. Weight patches from
stock LoRA nodes become a patch table on the server whenever
``patches_uuid`` changes. Registration methods for anything that would run
inside the DiT forward raise at node execution, so ComfyUI marks the node
that asked for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import comfy.model_patcher
import comfy.patcher_extension
import comfy.utils

from .client import GmlxRefusal, StepClient
from .patches import sync_table

# A small positive size, so ComfyUI never sums the meta tensors' bytes.
NOMINAL_SIZE = 64 << 20
# Free memory reported to the sampler, so it batches rows whenever shapes allow.
FREE_MEMORY = 1 << 50
# The patches of ComfyUI's Flux KV Cache node, which a step on the server's
# cache of reference K and V stands in for, and the reference method it sets.
KV_PATCHES = frozenset({"attn1_patch", "post_input"})
KV_METHOD = ("diffusion_model.params.default_ref_method", "index_timestep_zero")


def kv_patch(patch, name: str) -> bool:
    """Whether ``patch`` is the Flux KV Cache node's patch ``name``."""
    if name == "attn1_patch":
        cls = type(patch)
        return cls.__name__ == "KV_Attn_Input" and cls.__module__.endswith("nodes_flux")
    if name == "post_input":
        return getattr(patch, "__qualname__", "").startswith("FluxKVCache.")
    return False


@dataclass
class GmlxBinding:
    """What a gmlx MODEL knows about its server and model."""
    client: StepClient
    model_id: str
    status: dict
    manifest_count: int = 0
    events: list = field(default_factory=list)

    @property
    def family(self) -> str:
        return str(self.status.get("family"))

    @property
    def edit_capable(self) -> bool:
        return bool(self.status.get("edit_capable"))


def refuse(what: str):
    raise GmlxRefusal(f"gmlx: {what} runs inside the DiT forward and is not supported "
                      "on a gmlx model")


class GmlxModelPatcher(comfy.model_patcher.ModelPatcher):
    """A ModelPatcher whose loads touch no tensors."""

    gmlx: GmlxBinding | None = None
    gmlx_table_id: str | None = None
    gmlx_synced_uuid = None
    gmlx_run: str | None = None
    gmlx_ref_cache: frozenset = frozenset()

    def __init__(self, model, load_device, offload_device, size=0,
                 weight_inplace_update=False, fast_disk=False):
        super().__init__(model, load_device, offload_device, size=size or NOMINAL_SIZE,
                         weight_inplace_update=weight_inplace_update, fast_disk=fast_disk)

    def clone(self, *args, **kwargs):
        n = super().clone(*args, **kwargs)
        n.gmlx = self.gmlx
        n.gmlx_table_id = self.gmlx_table_id
        n.gmlx_synced_uuid = self.gmlx_synced_uuid
        n.gmlx_ref_cache = self.gmlx_ref_cache
        return n

    def is_dynamic(self):
        return False

    def get_free_memory(self, device):
        return FREE_MEMORY

    # Patches.

    def sync_patches(self, force: bool = False) -> str | None:
        """Define the table of the current patches on the server when they
        changed since the last one sent. Returns the table id."""
        if self.hook_patches:
            raise GmlxRefusal("gmlx: hook patches change weights during sampling, which a "
                              "gmlx model does not do")
        if not force and self.gmlx_synced_uuid == self.patches_uuid:
            return self.gmlx_table_id
        b = self.gmlx
        if force:
            b.client.forget()
        self.gmlx_table_id = sync_table(b.client, b.model_id, self.patches, b.family)
        self.gmlx_synced_uuid = self.patches_uuid
        b.events.append({"kind": "sync", "table": self.gmlx_table_id})
        return self.gmlx_table_id

    # Loads that touch no weights.

    def _mark_loaded(self, device_to):
        self.model.device = device_to if device_to is not None else self.load_device
        self.model.model_lowvram = False
        self.model.lowvram_patch_counter = 0
        self.model.model_loaded_weight_memory = self.model_size()
        self.model.model_offload_buffer_memory = 0
        self.model.current_weight_patches_uuid = self.patches_uuid

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False,
             full_load=False):
        with self.use_ejected():
            self.unpatch_hooks()
            self.sync_patches()
            self._mark_loaded(device_to)
            for cb in self.get_all_callbacks(comfy.patcher_extension.CallbacksMP.ON_LOAD):
                cb(self, device_to, lowvram_model_memory, force_patch_weights, full_load)
            self.apply_hooks(self.forced_hooks, force_apply=True)

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        return 0

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False,
                               return_weight=False, force_cast=False):
        return None

    def pin_weight_to_device(self, key):
        return None

    def unpin_weight(self, key):
        return None

    def unpin_all_weights(self):
        return None

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        self.eject_model()
        if unpatch_weights:
            self.unpatch_hooks()
            self.backup.clear()
            self.model.current_weight_patches_uuid = None
            self.model.model_lowvram = False
            self.model.lowvram_patch_counter = 0
            if device_to is not None:
                self.model.device = device_to
            self.model.model_loaded_weight_memory = 0
            self.model.model_offload_buffer_memory = 0
        for k in list(self.object_patches_backup):
            comfy.utils.set_attr(self.model, k, self.object_patches_backup[k])
        self.object_patches_backup.clear()

    # A run is the server's cache scope and busy marker.

    def pre_run(self):
        super().pre_run()
        self.sync_patches()
        b = self.gmlx
        self.gmlx_run = b.client.run_begin(b.model_id)
        b.events.append({"kind": "run_begin", "run": self.gmlx_run})

    def cleanup(self):
        run, self.gmlx_run = self.gmlx_run, None
        try:
            if run is not None:
                self.gmlx.client.run_end(run)
                self.gmlx.events.append({"kind": "run_end", "run": run})
        finally:
            super().cleanup()

    # Registrations for what runs inside the DiT forward.

    def _ref_cache_served(self) -> bool:
        caps = getattr(self.gmlx.client, "capabilities", None) or {}
        return "ref_cache" in (caps.get("features") or ())

    def set_model_patch(self, patch, name):
        # The Flux KV Cache node's patches become a step on the server's cache.
        if kv_patch(patch, name) and self._ref_cache_served():
            self.gmlx_ref_cache = self.gmlx_ref_cache | {name}
            return
        refuse(f"the model patch {name!r}")

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        refuse(f"the model patch replace {name!r} at {block_name} {number}")

    def set_model_rope_options(self, *args, **kwargs):
        refuse("a RoPE option")

    def set_model_optimized_attention(self, optimized_attention):
        refuse("an attention override")

    def add_weight_wrapper(self, name, function):
        refuse(f"a weight wrapper on {name}")

    def add_object_patch(self, name, obj):
        if (name, obj) == KV_METHOD and self.gmlx_ref_cache == KV_PATCHES:
            return super().add_object_patch(name, obj)
        if name == "diffusion_model" or name.startswith("diffusion_model."):
            refuse(f"an object patch on {name}")
        return super().add_object_patch(name, obj)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        if wrapper_type == comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL:
            refuse("a DIFFUSION_MODEL wrapper")
        return super().add_wrapper_with_key(wrapper_type, key, wrapper)

    def add_hook_patches(self, *args, **kwargs):
        raise GmlxRefusal("gmlx: hook patches change weights during sampling, which a gmlx "
                          "model does not do")

    def model_state_dict_for_saving(self, *args, **kwargs):
        raise GmlxRefusal("gmlx: a gmlx model has no weights in ComfyUI to save")

    def state_dict_for_saving(self, *args, **kwargs):
        raise GmlxRefusal("gmlx: a gmlx model has no weights in ComfyUI to save")


CPU = torch.device("cpu")
