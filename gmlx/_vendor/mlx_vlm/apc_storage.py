"""Component-major storage primitives for Automatic Prefix Caching (issue #1629)."""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, runtime_checkable

import mlx.core as mx

ComponentId = str


def _array_bytes(t: mx.array) -> int:
    nbytes = getattr(t, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    try:
        return int(t.size) * int(t.itemsize)
    except Exception:
        return 0


@runtime_checkable
class StateHandle(Protocol):
    """Physical storage for one component of one logical cache node.

    A handle owns its tensors plus their resident-byte accounting,
    serialization, and release. The logical index (hashes, parent links,
    lock counts) lives on the node and never touches tensors directly.
    """

    def resident_bytes(self) -> int: ...

    def release(self) -> None: ...


# gmlx: a block stores its arrays stacked, one buffer per array shape, so
# evicting it frees one buffer where it freed two a layer.
def _owned_copy(x: mx.array) -> mx.array:
    return mx.contiguous(mx.array(x, dtype=x.dtype))


class KVBlockHandle:
    """A node's pageable float K/V, one stacked array per array shape.

    The block's arrays are its layers' keys, then their values. Arrays of
    one shape and dtype stack into one part, so a model whose layers share
    a shape stores a block as one buffer, and a release frees one buffer.
    ``index[i]`` is the (part, row) of array i. The parts are copies, never
    views of the arrays they were built from. Join blocks with
    gmlx.cache.apc_blocks.join_blocks.
    """

    __slots__ = ("parts", "index", "n_layers")

    def __init__(self, parts=None, index=(), n_layers: int = 0):
        self.parts = parts
        self.index = tuple(index)
        self.n_layers = n_layers

    @classmethod
    def stack(cls, keys: List[mx.array], values: List[mx.array]) -> "KVBlockHandle":
        order: Dict[tuple, int] = {}
        members: List[List[mx.array]] = []
        index = []
        for a in list(keys) + list(values):
            c = order.setdefault((tuple(a.shape), a.dtype), len(members))
            if c == len(members):
                members.append([])
            index.append((c, len(members[c])))
            members[c].append(a)
        # A stack of two or more arrays writes a new buffer; a lone array
        # is copied first, since a stack of one may return a view of it.
        parts = tuple(
            mx.stack(m) if len(m) > 1 else mx.expand_dims(_owned_copy(m[0]), 0)
            for m in members)
        return cls(parts, index, len(keys))

    @staticmethod
    def count_parts(keys: List[mx.array], values: List[mx.array]) -> int:
        """Parts a block of these arrays stacks into."""
        return len({(tuple(a.shape[:-2]), a.shape[-1], a.dtype)
                    for a in list(keys) + list(values)})

    @property
    def tokens(self) -> int:
        return int(self.parts[0].shape[-2]) if self.parts else 0

    def buffers(self) -> int:
        return len(self.parts or ())

    def resident_bytes(self) -> int:
        return sum(_array_bytes(p) for p in self.parts or ())

    def release(self) -> None:
        self.parts = None


class APCNode:
    """Logical prefix-index entry: identity plus per-component storage handles.

    Physical tensors live only inside the handles in ``components``; the node
    carries the logical index. Concrete nodes supply the identity fields and a
    ``components`` dict, and expose them through the node-view accessors below.
    """

    KV_COMPONENT: ComponentId = "kv"

    components: Dict[ComponentId, StateHandle]

    def kv_handle(self) -> Optional[KVBlockHandle]:
        handle = self.components.get(self.KV_COMPONENT)
        if isinstance(handle, KVBlockHandle) and handle.parts:
            return handle
        return None

    def set_kv(self, keys: List[mx.array], values: List[mx.array]) -> None:
        """Store copies of one block's per-layer keys and values, stacked."""
        self.components[self.KV_COMPONENT] = KVBlockHandle.stack(keys, values)

    def release_components(self) -> None:
        for handle in self.components.values():
            handle.release()
        self.components.clear()

    def resident_bytes(self) -> int:
        return sum(handle.resident_bytes() for handle in self.components.values())
