"""Remote GGUF inspection - header-only, for ``validate`` / ``pull`` before a
multi-GB download.

``gguf.GGUFReader`` can't read a truncated prefix (it eagerly materializes tensor
*data* and reshapes it), so this module carries a minimal header-only parser: it
reads the GGUF magic, the metadata KV block, and the tensor-info table (each
tensor's name + ggml type) and stops *before* any tensor bytes. Paired with an
HTTP range read of just the header (a few MB), a remote GGUF's per-tensor codec
layout + architecture can be checked without pulling the weights.

Codec support is classified with **preflight's own sets**, so a remote verdict
matches what the loader would decide locally.
"""
from __future__ import annotations

import json
import os
import struct
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from .preflight import NATIVE_FP_TYPES, NATIVE_TYPES, SUPPORTED_QUANT_TYPES

_HF_HOST = "https://huggingface.co"

# GGUF metadata value types (for skipping KV we don't care about).
_GGUF_SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                     10: 8, 11: 8, 12: 8}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


class RemoteError(RuntimeError):
    """A user-facing failure inspecting or fetching a remote GGUF."""


class AmbiguousRepo(RemoteError):
    """A directory ref matched several GGUF models; ``refs`` lists them."""

    def __init__(self, message: str, where: str, refs: list):
        super().__init__(message)
        self.where = where
        self.refs = refs


class _NeedMore(Exception):
    """The parser ran past the buffer - fetch a larger prefix and retry."""


# Minimal header-only GGUF parser
class _Reader:
    def __init__(self, buf: bytes):
        self.b = buf
        self.n = len(buf)
        self.i = 0

    def _need(self, k: int) -> None:
        if self.i + k > self.n:
            raise _NeedMore()

    def u32(self) -> int:
        self._need(4)
        v = struct.unpack_from("<I", self.b, self.i)[0]
        self.i += 4
        return v

    def u64(self) -> int:
        self._need(8)
        v = struct.unpack_from("<Q", self.b, self.i)[0]
        self.i += 8
        return v

    def raw(self, k: int) -> bytes:
        if k < 0 or k > 1 << 34:                 # guard a corrupt length
            raise RemoteError("implausible length in GGUF header (not a GGUF?)")
        self._need(k)
        v = self.b[self.i:self.i + k]
        self.i += k
        return v

    def string(self) -> bytes:
        return self.raw(self.u64())

    def skip_value(self, vtype: int) -> None:
        if vtype in _GGUF_SCALAR_SIZE:
            self.raw(_GGUF_SCALAR_SIZE[vtype])
        elif vtype == _GGUF_STRING:
            self.string()
        elif vtype == _GGUF_ARRAY:
            elem = self.u32()
            count = self.u64()
            if elem in _GGUF_SCALAR_SIZE:
                self.raw(_GGUF_SCALAR_SIZE[elem] * count)
            elif elem == _GGUF_STRING:
                for _ in range(count):
                    self.string()
            else:
                raise RemoteError(f"unsupported GGUF array element type {elem}")
        else:
            raise RemoteError(f"unsupported GGUF value type {vtype}")


def _parse_header(buf: bytes) -> tuple[str | None, str | None, list]:
    """Parse magic + KV + tensor-info table from ``buf``. Returns
    ``(arch, gguf_type, [(name, ggml_type_int), ...])``. Raises
    :class:`_NeedMore` if the buffer ends before the tensor-info table does.
    ``gguf_type`` is ``general.type`` ("adapter" for LoRA adapter GGUFs, which
    otherwise carry their base model's arch and would grade as loadable)."""
    r = _Reader(buf)
    if r.raw(4) != b"GGUF":
        raise RemoteError("not a GGUF file (bad magic)")
    version = r.u32()
    if version not in (2, 3):
        raise RemoteError(f"unsupported GGUF version {version}")
    n_tensors = r.u64()
    n_kv = r.u64()
    if n_tensors > 10_000_000 or n_kv > 1_000_000:
        raise RemoteError("implausible tensor/KV counts (not a GGUF?)")

    arch: str | None = None
    gguf_type: str | None = None
    for _ in range(n_kv):
        key = r.string()
        vtype = r.u32()
        if key == b"general.architecture" and vtype == _GGUF_STRING:
            arch = r.string().decode("utf-8", "replace")
        elif key == b"general.type" and vtype == _GGUF_STRING:
            gguf_type = r.string().decode("utf-8", "replace")
        else:
            r.skip_value(vtype)

    tensors = []
    for _ in range(n_tensors):
        name = r.string().decode("utf-8", "replace")
        ndim = r.u32()
        for _ in range(ndim):
            r.u64()
        ttype = r.u32()
        r.u64()                                  # data offset (ignored)
        tensors.append((name, ttype))
    return arch, gguf_type, tensors


def _type_name(ttype: int) -> str:
    try:
        from gguf import GGMLQuantizationType
        return GGMLQuantizationType(ttype).name
    except Exception:
        return f"TYPE_{ttype}"


@dataclass
class HeaderReport:
    arch: str | None
    histogram: dict = field(default_factory=dict)     # codec name -> count
    unsupported: dict = field(default_factory=dict)   # codec name -> count
    n_tensors: int = 0
    gguf_type: str | None = None                      # general.type ("adapter", ...)

    @property
    def loadable_codecs(self) -> bool:
        return not self.unsupported


def classify_header(buf: bytes) -> HeaderReport:
    """Build a codec report from a header prefix (re-raises :class:`_NeedMore`)."""
    arch, gguf_type, tensors = _parse_header(buf)
    hist: dict[str, int] = {}
    unsup: dict[str, int] = {}
    for _name, ttype in tensors:
        tn = _type_name(ttype)
        hist[tn] = hist.get(tn, 0) + 1
        if (tn not in SUPPORTED_QUANT_TYPES and tn not in NATIVE_TYPES
                and tn not in NATIVE_FP_TYPES):
            unsup[tn] = unsup.get(tn, 0) + 1
    return HeaderReport(arch, hist, unsup, len(tensors), gguf_type)


def aggregate_reports(reports: list[HeaderReport]) -> HeaderReport:
    """Union several shards' reports into one. A split GGUF carries the arch only
    in the metadata shard (often tensor-free) and distributes its tensors - so a
    codec used by a single tensor can live in any shard. The arch is the first
    non-empty one; the codec histograms sum. Checking every shard is the only
    sound way to catch such an isolated codec."""
    arch: str | None = None
    gguf_type: str | None = None
    hist: dict[str, int] = {}
    unsup: dict[str, int] = {}
    n = 0
    for r in reports:
        if arch is None and r.arch:
            arch = r.arch
        if gguf_type is None and r.gguf_type:
            gguf_type = r.gguf_type
        for k, v in r.histogram.items():
            hist[k] = hist.get(k, 0) + v
        for k, v in r.unsupported.items():
            unsup[k] = unsup.get(k, 0) + v
        n += r.n_tensors
    return HeaderReport(arch, hist, unsup, n, gguf_type)


# Ref parsing + HTTP range read
@dataclass
class Ref:
    kind: str                    # "local" | "url" | "hf"
    raw: str
    url: str | None = None       # fetch URL (None for an hf *directory* ref)
    repo: str | None = None      # hf repo id (org/name)
    path_in_repo: str | None = None
    revision: str = "main"
    filename: str | None = None  # basename, the download destination name
    is_dir: bool = False         # hf ref naming a folder / repo, not a .gguf


def hf_resolve_url(repo: str, path_in_repo: str, revision: str = "main") -> str:
    return f"{_HF_HOST}/{repo}/resolve/{revision}/{path_in_repo}"


def _is_hf_hostname(host: str | None) -> bool:
    return host == "huggingface.co" or bool(host and host.endswith(".huggingface.co"))


def _auth_headers(url: str) -> dict:
    if not _is_hf_hostname(urllib.parse.urlsplit(url).hostname):
        return {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            pass  # no stored hub token -> anonymous request
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


class _AuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    """stdlib urlopen re-sends every header across a redirect, so the HF token
    would follow huggingface.co's 302 to the CDN / presigned-S3 host. Drop
    Authorization whenever the redirect changes hosts - exact hostname match,
    the same rule requests' ``rebuild_auth`` applies. HF's cross-host redirect
    targets are pre-signed URLs that don't need the token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and new.has_header("Authorization"):
            old_host = urllib.parse.urlsplit(req.full_url).hostname
            new_host = urllib.parse.urlsplit(new.full_url).hostname
            if old_host != new_host:
                new.remove_header("Authorization")
        return new


_redirect_opener = urllib.request.build_opener(_AuthRedirectHandler())


def http_open(req: urllib.request.Request, *, timeout: float):
    """urlopen for any request that may carry the HF token - routes through
    :class:`_AuthRedirectHandler` so auth never leaks cross-host."""
    return _redirect_opener.open(req, timeout=timeout)


def _make_hf_ref(repo: str, path: str, revision: str, raw: str) -> Ref:
    """An hf Ref for a file (``path`` ends in ``.gguf``) or, otherwise, a directory
    - a subfolder or the bare repo root (empty ``path``), resolved later by listing."""
    is_file = path.lower().endswith(".gguf")
    return Ref(
        "hf", raw,
        url=hf_resolve_url(repo, path, revision) if is_file else None,
        repo=repo, path_in_repo=path, revision=revision,
        filename=path.rsplit("/", 1)[-1] if is_file else None,
        is_dir=not is_file,
    )


def _hf_ref_from_url(url: str) -> Ref | None:
    """Normalize a huggingface.co web/resolve URL to an hf Ref, so a pasted
    ``.../blob/main/x.gguf`` (an HTML page, not the file) or ``.../tree/main/dir``
    folder link just works. Returns ``None`` for non-HF URLs."""
    p = urllib.parse.urlsplit(url)
    if p.netloc not in ("huggingface.co", "www.huggingface.co"):
        return None
    segs = [s for s in p.path.split("/") if s]
    if len(segs) < 2:
        return None
    repo = f"{segs[0]}/{segs[1]}"
    rest = segs[2:]
    revision, path = "main", ""
    if rest and rest[0] in ("blob", "resolve", "tree"):
        if len(rest) >= 2:
            revision = rest[1]
        path = "/".join(rest[2:])
    elif rest:
        path = "/".join(rest)
    if any(seg == ".." for seg in (*segs[:2], *path.split("/"))):
        # Same rule as parse_ref's hf: branch - `..` would survive into the
        # local dest-path join on pull.
        raise RemoteError(f"hf URL must not contain '..': {url!r}")
    return _make_hf_ref(repo, path, revision, url)


def parse_ref(ref: str) -> Ref:
    """Classify a model ref into a :class:`Ref`. Accepts:

    * a **local path**;
    * ``hf:<org>/<repo>[/<path>][@<rev>]`` - a file (``.../x.gguf``), a folder, or a
      bare repo (the last two are resolved by listing - see ``manage._resolve_to_file``);
    * an ``http(s)://`` URL - a huggingface.co web/resolve URL is normalized to an hf
      ref (so a ``/blob/`` page link or a ``/tree/`` folder link works); any other URL
      is taken as a direct file download.
    """
    if ref.startswith("hf:"):
        body = ref[3:]
        revision = "main"
        if "@" in body:
            body, revision = body.rsplit("@", 1)
        parts = [p for p in body.split("/") if p]
        if len(parts) < 2:
            raise RemoteError(
                "hf ref must be at least hf:<org>/<repo> (optionally "
                f"/<path/to/file.gguf>), got {ref!r}")
        if any(p == ".." for p in parts):
            # `..` would survive into the local dest-path join on pull.
            raise RemoteError(f"hf ref must not contain '..': {ref!r}")
        return _make_hf_ref("/".join(parts[:2]), "/".join(parts[2:]), revision, ref)
    if ref.startswith(("http://", "https://")):
        hf = _hf_ref_from_url(ref)
        if hf is not None:
            return hf
        name = ref.rsplit("/", 1)[-1].split("?")[0]
        return Ref("url", ref, url=ref, filename=name)
    return Ref("local", ref, filename=os.path.basename(ref))


def http_get_prefix(url: str, end: int, *, timeout: float = 30.0) -> bytes:
    """GET bytes ``[0, end)`` of ``url`` via a Range request. Caps the socket read
    at ``end`` so a server that ignores Range can't stream a multi-GB body, and
    wraps transport errors as :class:`RemoteError`. Seam: monkeypatched in tests."""
    req = urllib.request.Request(
        url, headers={"Range": f"bytes=0-{end - 1}", **_auth_headers(url)})
    try:
        with http_open(req, timeout=timeout) as resp:
            return resp.read(end)
    except urllib.error.HTTPError as e:
        hint = (" - file or repo not found; try `gmlx validate "
                "hf:<org>/<repo>/` to list a repo's GGUFs"
                if e.code == 404 else "")
        raise RemoteError(f"HTTP {e.code} fetching {url}{hint}")
    except urllib.error.URLError as e:
        raise RemoteError(f"network error fetching {url}: {e.reason}")


def hf_list_dir(repo: str, path: str, revision: str = "main", *,
                recursive: bool = True, timeout: float = 30.0) -> list:
    """List a repo path via the HF tree API. Returns ``[(path, type, size), ...]``
    where ``type`` is ``"file"`` / ``"directory"``. Raises :class:`RemoteError`.
    Seam: monkeypatched in tests."""
    api = f"{_HF_HOST}/api/models/{repo}/tree/{revision}/{path}".rstrip("/")
    if recursive:
        api += "?recursive=true"
    req = urllib.request.Request(api, headers=_auth_headers(api))
    try:
        with http_open(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RemoteError(
                f"not found on Hugging Face: {repo}/{path} (revision {revision!r})")
        raise RemoteError(f"HTTP {e.code} listing {repo}/{path}")
    except urllib.error.URLError as e:
        raise RemoteError(f"network error listing {repo}/{path}: {e.reason}")
    if not isinstance(data, list):
        raise RemoteError(f"unexpected Hugging Face API response for {repo}/{path}")
    return [(e.get("path", ""), e.get("type", ""), e.get("size", 0)) for e in data]


def fetch_header(url: str, *, get=None,
                 initial: int = 4 * 1024 * 1024,
                 max_bytes: int = 128 * 1024 * 1024) -> HeaderReport:
    """Range-read a growing header prefix until the tensor-info table parses.

    ``get`` defaults to :func:`http_get_prefix`, resolved at call time so a test
    can monkeypatch the module attribute."""
    if get is None:
        get = http_get_prefix
    size = initial
    while True:
        buf = get(url, size)
        try:
            return classify_header(buf)
        except _NeedMore:
            if len(buf) < size:                  # server returned EOF: file < size
                raise RemoteError(
                    "reached end of file before the GGUF header was complete "
                    "(truncated or not a GGUF?)")
            if size >= max_bytes:
                raise RemoteError(
                    f"GGUF header exceeds the {max_bytes // (1024 * 1024)} MB "
                    f"range-read cap; pass a larger --max-mb")
            size = min(size * 4, max_bytes)
