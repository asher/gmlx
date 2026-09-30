# Copyright © 2023-2024 Apple Inc.

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten

from .base import create_causal_mask

# gmlx: gmlx.cache.kv_cache owns the cache classes and helpers this module
# shares with mlx-vlm, so their bodies are deleted here and imported by
# name. The prompt cache file helpers, TokenBuffer, PromptTrie and
# LRUPromptCache stay.
from gmlx.cache.kv_cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    ChunkedKVCache,
    ConcatenateKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    _BaseCache,
    create_attention_mask,
    dynamic_roll,
    make_prompt_cache,
)


def save_prompt_cache(file_name: str, cache: List[Any], metadata: Dict[str, str] = {}):
    """
    Save a pre-computed prompt cache to a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        cache (List[Any]): The model state.
        metadata (Dict[str, str]): Optional metadata to save along with model
            state.
    """
    cache_data = [c.state for c in cache]
    cache_info = [c.meta_state for c in cache]
    cache_data = dict(tree_flatten(cache_data))
    cache_classes = [type(c).__name__ for c in cache]
    cache_metadata = [cache_info, metadata, cache_classes]
    cache_metadata = dict(tree_flatten(cache_metadata))
    mx.save_safetensors(file_name, cache_data, cache_metadata)


def load_prompt_cache(file_name, return_metadata=False):
    """
    Load a prompt cache from a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        return_metadata (bool): Whether or not to return metadata.
            Default: ``False``.

    Returns:
        List[Any] or Tuple[List[Any], Dict[str, str]]: The prompt cache and
            the metadata if requested.
    """
    arrays, cache_metadata = mx.load(file_name, return_metadata=True)
    arrays = tree_unflatten(list(arrays.items()))
    cache_metadata = tree_unflatten(list(cache_metadata.items()))
    info, metadata, classes = cache_metadata
    cache = [
        globals()[c].from_state(state, meta_state)
        for c, state, meta_state in zip(classes, arrays, info)
    ]
    if return_metadata:
        return cache, metadata
    return cache


def can_trim_prompt_cache(cache: List[Any]) -> bool:
    """
    Check if model's cache can be trimmed.
    """
    return all(c.is_trimmable() for c in cache)


def trim_prompt_cache(cache: List[Any], num_tokens: int) -> List[Any]:
    """
    Trim the model's cache by the given number of tokens.

    This function will trim the cache if possible (in-place) and return the
    number of tokens that were trimmed.

    Args:
        cache (List[Any]): The model's cache.
        num_tokens (int): The number of tokens to trim.

    Returns:
        (int): The number of tokens that were trimmed.
    """
    if not can_trim_prompt_cache(cache) or len(cache) == 0:
        return 0
    return [c.trim(num_tokens) for c in cache][0]


class TokenBuffer:
    """A simple token buffer that can be efficiently appended to in a similar
    fashion to the KVCache.

    Perhaps these could share some logic in the future.
    """

    step = 256

    def __init__(self, tokens=[]):
        self._buffer = mx.array(tokens, dtype=mx.int32)
        self._size = len(tokens)

    def update_and_fetch(self, tokens):
        start = self._size
        end = start + len(tokens)

        new_size = ((end + self.step - 1) // self.step) * self.step
        if new_size > self._buffer.size:
            self._buffer = mx.concatenate(
                [self._buffer, mx.zeros(new_size - self._buffer.size, dtype=mx.int32)]
            )
        self._buffer[start:end] = tokens
        self._size = end

        return self._buffer[:end]

    @property
    def state(self):
        return self._buffer

    @property
    def tokens(self):
        return self._buffer[: self._size]


@dataclass
class PromptTrieResult:
    model: Any
    exact: Optional[List[int]]  # Exact match found
    shorter: Optional[List[int]]  # Longest prefix with a value
    longer: Optional[List[int]]  # Shortest value that extends beyond tokens
    common_prefix: int  # Length of common prefix with any path


class PromptTrie:
    def __init__(self):
        self._trie = {}

    def add(self, model: Any, tokens: List[int], value: Any):
        if model not in self._trie:
            self._trie[model] = {}

        current = self._trie[model]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]
        prev = current.get("__value__", None)
        current["__value__"] = value
        return prev

    def get(self, model: Any, tokens: List[int]):
        current = self._trie[model]
        for tok in tokens:
            current = current[tok]
        return current["__value__"]

    def pop(self, model: Any, tokens: List[int]):
        path = [self._trie[model]]
        for tok in tokens:
            path.append(path[-1][tok])
        value = path[-1].pop("__value__")
        for i in range(len(tokens), 0, -1):
            node = path[i]
            parent = path[i - 1]
            tok = tokens[i - 1]
            if len(node) > 0:
                break
            del parent[tok]
        return value

    def pop_prefixes(self, model: Any, tokens: List[int]):
        values = []
        current = self._trie[model]
        for i, tok in enumerate(tokens):
            if "__value__" in current:
                values.append((i, current.pop("__value__")))
            current = current[tok]
        return values

    def search(self, model: Any, tokens: List[int]) -> PromptTrieResult:
        if model not in self._trie:
            return PromptTrieResult(model, None, None, None, 0)

        current = self._trie[model]

        if not tokens and "__value__" in current:
            return PromptTrieResult(model, [], None, None, 0)

        # Walk the tokens as far as we can
        last_index = -1
        index = 0
        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "__value__" in current:
                last_index = index
            index += 1

        # Got an exact match
        if last_index == len(tokens) - 1 >= 0:
            return PromptTrieResult(model, tokens, None, None, 0)

        # Check if we found a prefix at any point
        shorter = None
        if last_index > 0:
            shorter = tokens[: last_index + 1]

        # Check for sequences that are longer
        longer = None
        common_prefix = index
        if index > 0:
            best = None
            stack = [(current, [])]
            while stack:
                current, extra = stack.pop()
                if "__value__" in current:
                    if best is None or len(extra) < len(best):
                        best = extra
                elif best is None or len(extra) < len(best):
                    for tok in current:
                        stack.append((current[tok], extra + [tok]))
            longer = tokens[:index] + best
        return PromptTrieResult(model, None, shorter, longer, common_prefix)


class LRUPromptCache:
    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        nbytes: int
        cache_type: str

    class CacheOrder:
        def __init__(self, ordering: List[str] = ["assistant", "user", "system"]):
            self._ordering = ordering
            self._lrus = {k: deque() for k in ordering}

        def __len__(self):
            return sum(len(lru) for lru in self._lrus.values())

        def push(self, model: Any, tokens: List[Any], cache_type: str = "assistant"):
            self._lrus[cache_type].append((model, tokens))

        def remove(self, model: Any, tokens: List[Any]):
            for cache_type in self._ordering:
                try:
                    self._lrus[cache_type].remove((model, tokens))
                    break
                except ValueError:
                    pass

        def pop(self):
            i = 0
            while i + 1 < len(self._ordering):
                lru_a = self._lrus[self._ordering[i]]
                lru_b = self._lrus[self._ordering[i + 1]]
                if lru_a and len(lru_a) >= len(lru_b):
                    return lru_a.popleft()
                i += 1
            return lru_b.popleft()

    def __init__(self, max_size: int = 10, max_bytes: int = 1 << 63):
        self.max_size = max_size
        self.max_bytes = max_bytes
        self._trie = PromptTrie()
        self._lru = LRUPromptCache.CacheOrder()
        self._n_bytes = 0
        self._n_bytes_by_type = {k: 0 for k in self._lru._ordering}

    def __len__(self):
        return len(self._lru)

    @property
    def nbytes(self):
        return self._n_bytes

    def fetch_nearest_cache(self, model: Any, tokens: List[int]):
        result = self._trie.search(model, tokens)
        if result.exact is not None:
            cache_entry = self._trie.get(result.model, result.exact)
            return copy.deepcopy(cache_entry.prompt_cache), []

        short_length = len(result.shorter) if result.shorter is not None else 0
        if result.longer is not None and result.common_prefix > short_length:
            cache_entry = self._trie.get(result.model, result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                cache = copy.deepcopy(cache_entry.prompt_cache)
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(cache, num_to_trim)
                return cache, tokens[prefix:]

        if short_length > 0:
            cache_entry = self._trie.get(result.model, result.shorter)
            return copy.deepcopy(cache_entry.prompt_cache), tokens[short_length:]

        return None, tokens

    def insert_cache(
        self,
        model: Any,
        tokens: List[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
    ):
        # Make the cache entry
        entry = LRUPromptCache.CacheEntry(
            prompt_cache, sum(c.nbytes for c in prompt_cache), cache_type
        )

        # Insert into the trie and update the byte counter and lru position
        self._n_bytes += entry.nbytes
        self._n_bytes_by_type[cache_type] += entry.nbytes
        prev = self._trie.add(model, tokens, entry)
        if prev is not None:
            self._n_bytes -= prev.nbytes
            self._n_bytes_by_type[prev.cache_type] -= prev.nbytes
            self._lru.remove(model, tokens)
        self._lru.push(model, tokens, cache_type)

        # If it is a trimmable cache remove all prefixes cause they just take
        # space
        if can_trim_prompt_cache(prompt_cache):
            for prefix_len, entry in self._trie.pop_prefixes(model, tokens):
                self._n_bytes -= entry.nbytes
                self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
                self._lru.remove(model, tokens[:prefix_len])

        # Ensure we match the constraints
        if len(self._lru) > self.max_size:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        while self._n_bytes > self.max_bytes:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes

    def trim_to(
        self, *, n_sequences: Optional[int] = None, n_bytes: Optional[int] = None
    ):
        n_sequences = max(0, n_sequences) if n_sequences is not None else 1 << 63
        n_bytes = max(0, n_bytes) if n_bytes is not None else 1 << 63

        while len(self._lru) > n_sequences:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        while self._n_bytes > n_bytes:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes

    def stats_by_type(self):
        result = {}
        for cache_type in self._lru._ordering:
            result[cache_type] = {
                "n_sequences": len(self._lru._lrus[cache_type]),
                "n_bytes": self._n_bytes_by_type[cache_type],
            }
        return result


# gmlx: load_prompt_cache resolves each class by name in this module, so the
# gmlx cache classes registered with gmlx.cache.bind join its globals.
import sys as _sys

from gmlx.cache import bind as _bind

_bind.apply_registry(_sys.modules[__name__])
