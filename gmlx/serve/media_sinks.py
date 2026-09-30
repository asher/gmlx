"""Checks in the functions that open request media or load by path.

The media gate in :mod:`gmlx.serve.patches.media_gate` refuses a bad media
reference early and says what to send instead, but it has to know every
body shape and transport a route reads. The checks here sit in the functions
that open a file, fetch a URL, load a model or create a folder, so they
cover every route, body shape and transport, including ones the gate does
not know.

- ``load_image``, ``load_audio`` and ``load_video`` in ``mlx_vlm.utils``
  take a ``data:`` URI, a file in the media folder reached with no symbolic
  link, or, with ``server.media_urls``, an http(s) URL that
  :func:`gmlx.serve.media_fetch.fetch` checks. Each holds at most
  :data:`~.media_gate.MEDIA_MAX_BYTES`, except a video file in the media
  folder, which the video reader streams. Every image is read with a
  fixed set of format plugins and at most
  :data:`~.media_gate.MEDIA_MAX_PIXELS` pixels.
- ``mlx_audio.audio_io.read`` takes bytes or a file object of at most
  :data:`~.media_gate.MEDIA_MAX_BYTES`, a file in the media folder, or a
  path in the Hugging Face repo folder of a configured model or in the
  speech model's folder, such as a voice prompt the model's own code reads.
- ``mlx_vlm.utils.load`` loads only a configured model path.
- ``load_drafter`` loads only the drafter the server's own build chose.
- ``DiskBlockStore`` stores only under the APC disk path the server's own
  build chose.

The server's own loads pass because the configuration names them. gmlx code
never calls these functions for itself.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import os
import shutil
import stat
import tempfile
from pathlib import Path

from gmlx.safe_path import (
    LeavesRoot,
    NotFollowed,
    canonical,
    open_file_below,
    parts_below,
    path_inside,
)
from gmlx.serve.patches import media_gate as mg
from gmlx.serve.patches.media_gate import MediaRefused

_FLAG = "_kq_media_sink"
# The image formats a request image may be. Others, such as EPS, run
# programs or read other files.
_IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF")
_IMAGE_WANT = "a base64 data:image/... URI"
_AUDIO_WANT = "base64 audio or a data:audio/... URI"
_VIDEO_WANT = "a base64 data:video/... URI"
# The folder under the cache folder that request media may be read from.
_MEDIA_DIR = ("gmlx", "media")

# (module, attribute) -> the function or method replaced there.
_originals: dict[tuple[object, str], object] = {}
# The Hugging Face repo folders of the configured models and the speech
# model's folder, where mlx_audio's reader may open a file a model's own
# code names. Set by install.
_model_roots: tuple[str, ...] = ()


class ModelRefused(ValueError):
    """A load names a model or folder the configuration does not."""


# The media folder

def default_media_root() -> str:
    """``$XDG_CACHE_HOME/gmlx/media``, else ``~/.cache/gmlx/media``."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(os.path.abspath(base), *_MEDIA_DIR)


def ensure_media_root(root: str | None = None) -> str | None:
    """Create the media folder with mode 0700 and accept paths in it. A
    folder that is a symbolic link, belongs to another user or is open to
    others is not used, and every path is then refused. Returns the folder,
    or None when it is not used."""
    root = root or default_media_root()
    try:
        os.makedirs(os.path.dirname(root), mode=0o700, exist_ok=True)
        try:
            os.mkdir(root, 0o700)
        except FileExistsError:
            pass
        st = os.lstat(root)
    except OSError as e:
        return _no_root(f"cannot create {root} ({e.strerror or e})")
    if not stat.S_ISDIR(st.st_mode):
        return _no_root(f"{root} is a symbolic link or not a folder")
    if st.st_uid != os.getuid():
        return _no_root(f"{root} belongs to another user")
    if st.st_mode & 0o077:
        try:
            os.chmod(root, 0o700, follow_symlinks=False)
        except OSError as e:
            return _no_root(f"cannot make {root} private ({e.strerror or e})")
    mg.set_media_root(root, (root, canonical(root)))
    return root


def _no_root(reason: str) -> None:
    import logging

    logging.getLogger(__name__).warning(
        "[serve] request media files are refused: %s", reason)
    mg.set_media_root(None)
    return None


def _open_media_file(value: str, field: str, want: str) -> tuple[int, str]:
    """A descriptor of the regular file ``value`` names in the media folder,
    reached with no link followed, and its path."""
    found = mg.media_parts(value)
    if found is None:
        raise mg.reference_refusal(field, value, want)
    root, parts = found
    shown = os.path.join(root, *parts)
    try:
        fd = open_file_below(root, parts)
    except FileNotFoundError:
        raise MediaRefused(f"{field} names {shown}, which does not exist") from None
    except (LeavesRoot, NotFollowed):
        raise MediaRefused(f"{field} names {shown}, which passes through a symbolic "
                           "link or is not a file. The server follows no link in "
                           "its media folder.") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise MediaRefused(f"{field} names {shown}, which is not a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd, shown


def read_media_file(value: str, field: str, want: str) -> bytes:
    """The bytes of a file in the media folder, at most
    :data:`~.media_gate.MEDIA_MAX_BYTES` of them, as for inline data."""
    fd, _ = _open_media_file(value, field, want)
    with os.fdopen(fd, "rb") as f:
        # The read is bounded too, in case the file grows after the check.
        if os.fstat(f.fileno()).st_size > mg.MEDIA_MAX_BYTES:
            raise mg._too_large(field)
        data = f.read(mg.MEDIA_MAX_BYTES + 1)
    _check_size(len(data), field)
    return data


def reference_bytes(value: str, kind: str, field: str, want: str) -> bytes:
    """The bytes a string media reference names: an inline ``data:`` URI of
    ``kind``, a file in the media folder, or an http(s) URL when
    ``server.media_urls`` is on. Anything else is refused."""
    if value.startswith("data:"):
        data = mg._data_uri(value, kind, field)
        if data is None:
            raise MediaRefused(f"{field} must be a base64 data:{kind}/... URI")
        return data
    if mg._is_url(value):
        if not mg._allow_urls:
            raise MediaRefused(f"{field} names a URL, and this server takes media "
                               f"only as {want}. {mg._URL_HINT}")
        from gmlx.serve.media_fetch import FetchRefused, fetch

        try:
            return fetch(value, max_bytes=mg.MEDIA_MAX_BYTES)
        except FetchRefused as e:
            raise MediaRefused(f"{field}: {e}") from None
    return read_media_file(value, field, want)


def _check_size(size: int, field: str) -> None:
    if size > mg.MEDIA_MAX_BYTES:
        raise mg._too_large(field)


# The media loaders

def open_image(data: bytes | io.BytesIO, field: str = "image"):
    """A lazily decoded Pillow image of ``data``, read with the allowed
    format plugins only and within the pixel limit."""
    from PIL import Image

    stream = data if isinstance(data, io.BytesIO) else io.BytesIO(data)
    try:
        image = Image.open(stream, formats=_IMAGE_FORMATS)
    except Image.DecompressionBombError:
        # Pillow's own limit, which is above this one, stops the open.
        raise mg.too_many_pixels(field) from None
    except Exception as e:  # noqa: BLE001 - Pillow raises many types for bad data
        raise MediaRefused(f"{field} is not a readable PNG, JPEG, WebP, GIF, BMP or "
                           "TIFF image") from e
    width, height = image.size
    if width * height > mg.MEDIA_MAX_PIXELS:
        image.close()
        raise mg.too_many_pixels(field, width, height)
    return image


def _image_loader(original):
    def load_image(image_source, timeout: int = 10):
        from PIL import Image

        if isinstance(image_source, Image.Image):
            width, height = image_source.size
            if width * height > mg.MEDIA_MAX_PIXELS:
                raise mg.too_many_pixels("image", width, height)
            return original(image_source, timeout)
        if isinstance(image_source, io.BytesIO):
            _check_size(image_source.getbuffer().nbytes, "image")
            return original(open_image(image_source), timeout)
        if isinstance(image_source, (str, Path)):
            data = reference_bytes(str(image_source), "image", "image", _IMAGE_WANT)
            return original(open_image(data), timeout)
        return original(image_source, timeout)
    return load_image


def _audio_loader(original):
    def load_audio(file, sr: int, timeout: int = 10):
        import numpy as np

        if isinstance(file, np.ndarray):
            return original(file, sr, timeout)
        if isinstance(file, (bytes, bytearray)):
            file = io.BytesIO(bytes(file))
        if isinstance(file, io.BytesIO):
            _check_size(file.getbuffer().nbytes, "audio")
            return original(file, sr, timeout)
        if isinstance(file, (str, Path)):
            data = reference_bytes(str(file), "audio", "audio", _AUDIO_WANT)
            return original(io.BytesIO(data), sr, timeout)
        raise MediaRefused(f"audio of type {type(file).__name__} is not accepted")
    return load_audio


def _audio_reader(original):
    def read(file, *args, **kwargs):
        if isinstance(file, (bytes, bytearray)):
            file = io.BytesIO(bytes(file))
        elif isinstance(file, (str, os.PathLike)):
            value = str(file)
            if mg.media_parts(value) is not None:
                file = io.BytesIO(read_media_file(value, "audio", _AUDIO_WANT))
            elif any(path_inside(canonical(value), r) for r in _model_roots):
                # A model's own file, such as a voice prompt in its folder.
                return original(file, *args, **kwargs)
            else:
                raise mg.reference_refusal("audio", value, _AUDIO_WANT)
        elif hasattr(file, "read") and not isinstance(file, io.BytesIO):
            file = io.BytesIO(file.read(mg.MEDIA_MAX_BYTES + 1))
        if isinstance(file, io.BytesIO):
            _check_size(file.getbuffer().nbytes, "audio")
            return original(file, *args, **kwargs)
        raise MediaRefused(f"audio of type {type(file).__name__} is not accepted")
    return read


@contextlib.contextmanager
def _private_file(data: bytes, suffix: str):
    """Write ``data`` to a file in a new private folder and yield its path.
    The folder is removed afterwards."""
    folder = tempfile.mkdtemp(prefix="gmlx-media-")        # mode 0700
    try:
        path = os.path.join(folder, f"media{suffix}")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        yield path
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _video_loader(original):
    def load_video(video_path, *args, **kwargs):
        if not isinstance(video_path, str):
            raise MediaRefused(f"video of type {type(video_path).__name__} is not "
                               "accepted")
        if video_path.startswith("data:") or mg._is_url(video_path):
            data = reference_bytes(video_path, "video", "video", _VIDEO_WANT)
        else:
            # A video in the media folder is read where it is, so its size is
            # not limited. The video reader gets the checked path.
            fd, shown = _open_media_file(video_path, "video", _VIDEO_WANT)
            try:
                head = os.pread(fd, 16, 0)
            finally:
                os.close(fd)
            _check_container(head, "video")
            return original(shown, *args, **kwargs)
        _check_container(data, "video")
        # The video reader opens files only, and a path it is handed goes
        # through ffmpeg's protocol parsing, so inline data becomes a file.
        with _private_file(data, ".video") as path:
            return original(path, *args, **kwargs)
    return load_video


def _check_container(head: bytes, field: str) -> None:
    if not mg._video_container(head):
        raise MediaRefused(f"{field} holds no MP4, QuickTime, Matroska, WebM or AVI "
                           "video")


# Model loads and the APC disk tier

def _configured_model_paths() -> tuple[set[str], set[str]]:
    import gmlx.serve.bridge_vlm as serving

    models, adapters = set(), set()
    for rm in serving.resolved_models().values():
        models.update(p for p in (rm.path, rm.draft_gguf, rm.mmproj) if p)
        adapters.update(a for a in (getattr(rm, "adapter", None),
                                    *(getattr(rm, "adapters", ()) or ())) if a)
    return models, adapters


def _hf_repo_folder(path: str) -> str | None:
    """The Hugging Face repo folder a configured model file lies in, which
    holds the repo's snapshots and the blobs their files link to, or None
    for a file outside the Hugging Face cache."""
    from huggingface_hub.constants import HF_HUB_CACHE

    hub = canonical(HF_HUB_CACHE)
    parts = parts_below(canonical(path), hub)
    if parts and parts[0].startswith("models--"):
        return os.path.join(hub, parts[0])
    return None


def _tts_folder() -> str | None:
    """The folder of the configured speech model, which the server loads
    from a local folder or from its Hugging Face repo."""
    import gmlx.serve.bridge_vlm as serving

    value = getattr(serving._SERVER_CFG, "tts", None)
    if not value:
        return None
    from gmlx.serve.tts import resolve_tts_model

    ref = resolve_tts_model(value)
    path = os.path.expanduser(ref)
    # A local folder counts even before it exists, so a typo or a folder
    # made later never reaches the repo id check below.
    if os.path.isabs(path) or os.path.isdir(path):
        return canonical(path)
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.file_download import repo_folder_name

    try:
        folder = repo_folder_name(repo_id=ref, repo_type="model")
    except ValueError:                       # neither a folder nor a repo id
        return None
    return canonical(os.path.join(HF_HUB_CACHE, folder))


def _resolve_model_roots() -> tuple[str, ...]:
    # A model file outside the Hugging Face cache adds no root. Its code
    # reads no audio from the folder it lies in, which may be ~/Downloads.
    models, adapters = _configured_model_paths()
    roots = {r for r in map(_hf_repo_folder, models | adapters) if r}
    tts = _tts_folder()
    if tts:
        roots.add(tts)
    # A speech model folder must not open the whole home folder.
    home = canonical(os.path.expanduser("~"))
    return tuple(sorted(r for r in roots if not path_inside(home, r)))


def _same_path(a, b) -> bool:
    return os.path.abspath(os.path.expanduser(str(a))) == \
        os.path.abspath(os.path.expanduser(str(b)))


def _model_loader(original):
    def load(path_or_hf_repo, adapter_path=None, *args, **kwargs):
        models, adapters = _configured_model_paths()
        if not any(_same_path(path_or_hf_repo, m) for m in models):
            raise ModelRefused(f"{path_or_hf_repo} is not a configured model. "
                               "Configure it in models: first.")
        if adapter_path is not None and not any(_same_path(adapter_path, a)
                                                for a in adapters):
            raise ModelRefused(f"{adapter_path} is not a configured adapter")
        return original(path_or_hf_repo, adapter_path, *args, **kwargs)
    return load


def _drafter_loader(original):
    def load_drafter(path_or_repo, *args, **kwargs):
        # The server's build names its drafter in this variable for the
        # length of the build, and only then.
        chosen = os.environ.get("MLX_VLM_DRAFT_MODEL")
        if not (chosen and _same_path(path_or_repo, chosen)):
            raise ModelRefused(f"{path_or_repo} is not the drafter the server's "
                               "configuration chose")
        return original(path_or_repo, *args, **kwargs)
    return load_drafter


def _disk_store_init(original):
    def __init__(self, root, *args, **kwargs):
        chosen = os.environ.get("APC_DISK_PATH")
        if not (chosen and _same_path(root, chosen)):
            raise ModelRefused(f"{root} is not the APC disk path the server's "
                               "configuration chose")
        return original(self, root, *args, **kwargs)
    return __init__


# Install

def _replace(target, name: str, make) -> None:
    current = getattr(target, name)
    if getattr(current, _FLAG, False):
        return
    wrapped = make(current)
    wrapped.__dict__[_FLAG] = True
    wrapped.__wrapped__ = current
    _originals.setdefault((target, name), current)
    setattr(target, name, wrapped)


def install() -> None:
    """Put the checks in every function above, and in every module that
    holds its own reference to one. Idempotent."""
    utils = importlib.import_module("mlx_vlm.utils")
    loaders = {"load_image": _image_loader, "load_audio": _audio_loader,
               "load_video": _video_loader}
    for name, make in loaders.items():
        _replace(utils, name, make)
    # These modules copied the loaders at import time.
    for modname, names in (("mlx_vlm.server.reranking", ("load_image", "load_video")),
                           ("mlx_vlm.server.embeddings", ("load_image",))):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for name in names:
            if getattr(getattr(mod, name, None), _FLAG, False):
                continue
            _originals.setdefault((mod, name), getattr(mod, name))
            setattr(mod, name, getattr(utils, name))
    _replace(utils, "load", _model_loader)
    generation = importlib.import_module("mlx_vlm.server.generation")
    if not getattr(generation.load, _FLAG, False):
        _originals.setdefault((generation, "load"), generation.load)
        setattr(generation, "load", utils.load)
    drafters = importlib.import_module("mlx_vlm.speculative.drafters")
    _replace(drafters, "load_drafter", _drafter_loader)
    apc = importlib.import_module("mlx_vlm.apc")
    _replace(apc.DiskBlockStore, "__init__", _disk_store_init)
    global _model_roots
    _model_roots = _resolve_model_roots()
    try:
        audio_io = importlib.import_module("mlx_audio.audio_io")
        server_audio = importlib.import_module("mlx_vlm.server.audio")
    except ImportError:
        return
    _replace(audio_io, "read", _audio_reader)
    # The transcription module copied the reader at import time.
    if not getattr(server_audio.audio_read, _FLAG, False):
        _originals.setdefault((server_audio, "audio_read"), server_audio.audio_read)
        setattr(server_audio, "audio_read", audio_io.read)


def uninstall() -> None:
    """Restore every function :func:`install` replaced. For tests."""
    while _originals:
        (target, name), original = _originals.popitem()
        setattr(target, name, original)
