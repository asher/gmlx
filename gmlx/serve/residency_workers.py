"""The image and video workers: a pipeline's admission and its release."""

from __future__ import annotations

import contextlib
from typing import Any, Callable

from .residency_types import (IMAGE_KIND, VIDEO_KIND, _WORKER_KINDS,
                              _WORKER_ORDER, _Entry, _log)


class _Workers:
    """Worker reservations and the release of image and video pipelines."""

    def _image_fn(self, i: int) -> Callable[..., Any]:
        if self._image_fns is None:
            from . import image_service

            self._image_fns = (image_service.entry_footprint,
                               image_service.load_on_worker,
                               image_service.release_on_worker)
        return self._image_fns[i]

    def _kind_fn(self, kind: str, i: int) -> Callable[..., Any]:
        """Function ``i`` of a worker kind's (footprint, load, release)."""
        if kind == IMAGE_KIND:
            return self._image_fn(i)
        if self._video_fns is None:
            from . import video_service

            self._video_fns = (video_service.entry_footprint,
                               video_service.load_on_worker,
                               video_service.release_on_worker)
        return self._video_fns[i]

    def _build_image(self, cache_key, model_path, footprint, *, ttl=None,
                     build_spec=None, kind: str = IMAGE_KIND) -> _Entry:
        """An image or video entry: the kind's ``load`` on the image
        worker, gated on the weights and the activation reserve together.
        Never reaches the stock loader."""
        self._gate(footprint, model_path, image=True)
        pipe = self._kind_fn(kind, 1)(build_spec)
        return self._new_entry(cache_key, model_path, footprint, ttl,
                               kind=kind, pipeline=pipe)

    def _worker_for(self, build_kind=None, *, incoming: int | None = None,
                    force: frozenset = frozenset(), teardown=None):
        """Hold the workers before the build lock, image before video,
        that a build loads its pipeline on (``build_kind``) or may release
        one on: those whose entries the eviction for ``incoming`` bytes
        takes, or, with ``incoming`` None, a teardown's, the workers of the
        idle entries ``teardown(entry)`` picks (every idle one when None);
        and those of ``force``. Then the lock is never held
        while a worker runs a job, and a chat cold load waits behind no
        job of a worker it does not release on. Without any, nothing is
        held."""
        with self._lock:
            kinds = set(force)
            if build_kind in _WORKER_KINDS:
                kinds.add(build_kind)
            if incoming is None:
                kinds.update(e.kind for e in self._entries.values()
                             if e.is_image and e.busy == 0
                             and (teardown is None or teardown(e)))
            else:
                kinds.update(self._victim_kinds(incoming))
        if not kinds:
            return contextlib.nullcontext()
        return _reserved_workers(frozenset(kinds))

    def _release_images(self, pipelines: list) -> None:
        """Release image and video pipelines, ``(kind, pipeline, bytes)``,
        each on its kind's worker, where its arrays were made, and wait for
        each, so the next gate measures the memory back. Each one's bytes
        leave the tearing count once its release ends. Call without
        ``_lock``."""
        while pipelines:
            kind, pipe, n = pipelines.pop()
            try:
                if pipe is None:
                    continue
                box = [pipe]
                try:
                    self._kind_fn(kind, 2)(box)
                except Exception:
                    _log.warning("teardown: image pipeline release failed",
                                 exc_info=True)
                finally:
                    box.clear()
                    del pipe
                    import gc
                    gc.collect()
            finally:
                if n:
                    with self._lock:
                        self._torn(kind, n)


def _holds_image_worker() -> bool:
    from . import image_service

    return image_service.worker_reserved()


def _holds_video_worker() -> bool:
    from . import video_service

    return video_service.worker_reserved()


def _holds_worker(kind: str) -> bool:
    """Whether the calling thread holds the worker of ``kind`` or is it."""
    return _holds_image_worker() if kind == IMAGE_KIND else \
        _holds_video_worker()


@contextlib.contextmanager
def _reserved_workers(kinds: frozenset):
    """Hold the workers of ``kinds``, image before video, the one order
    every thread takes them in."""
    from . import image_service, video_service

    holds = {IMAGE_KIND: image_service.reserved_worker,
             VIDEO_KIND: video_service.reserved_worker}
    with contextlib.ExitStack() as stack:
        for k in _WORKER_ORDER:
            if k in kinds:
                stack.enter_context(holds[k]())
        yield
