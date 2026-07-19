#!/usr/bin/env python3
"""Remote GGUF header parsing + ref classification + range-read growth.

All CPU, no network: the HTTP fetch is a seam (``http_get_prefix``) and the header
bytes come from a tiny GGUF minted with gguf-py - so the real header parser runs
against a real (if minimal) GGUF.
"""

from __future__ import annotations

import urllib.request

import numpy as np
import pytest

from gguf import GGMLQuantizationType as GT  # noqa: E402
from gguf import GGUFWriter, quants  # noqa: E402

from gmlx import remote  # noqa: E402


def _weight(codec):
    """A weight tensor of ``codec`` - raw zero bytes for IQ/TQ (gguf-py can't
    quantize them), else a real quantize() of small random data."""
    if codec.name.startswith(("IQ", "TQ")):
        from gguf.constants import GGML_QUANT_SIZES
        _, tsize = GGML_QUANT_SIZES[codec]
        return np.zeros((4, 2 * tsize), dtype=np.uint8)
    s = np.random.default_rng(0).standard_normal((8, 64)).astype(np.float32) * 0.1
    return quants.quantize(s, codec)


def _mint_bytes(path, *, arch="llama", codec=GT.Q4_0) -> bytes:
    w = GGUFWriter(str(path), arch)
    w.add_tensor("plain.f32", np.zeros((4, 16), dtype=np.float32), raw_dtype=GT.F32)
    w.add_tensor("blk.0.attn_q.weight", _weight(codec), raw_dtype=codec)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with open(path, "rb") as f:
        return f.read()


# ref parsing
def test_parse_ref_local():
    r = remote.parse_ref("/tmp/model.gguf")
    assert r.kind == "local"
    assert r.filename == "model.gguf"
    assert r.url is None


def test_parse_ref_url():
    r = remote.parse_ref("https://example.com/a/b/model-Q4_K_M.gguf?x=1")
    assert r.kind == "url"
    assert r.url == "https://example.com/a/b/model-Q4_K_M.gguf?x=1"
    assert r.filename == "model-Q4_K_M.gguf"


def test_parse_ref_hf():
    r = remote.parse_ref("hf:org/repo/sub/model.gguf")
    assert r.kind == "hf"
    assert r.repo == "org/repo"
    assert r.path_in_repo == "sub/model.gguf"
    assert r.revision == "main"
    assert r.filename == "model.gguf"
    assert r.url == "https://huggingface.co/org/repo/resolve/main/sub/model.gguf"


def test_parse_ref_hf_with_revision():
    r = remote.parse_ref("hf:org/repo/model.gguf@v2")
    assert r.revision == "v2"
    assert r.url.endswith("/resolve/v2/model.gguf")


def test_parse_ref_hf_repo_root_is_dir():
    r = remote.parse_ref("hf:org/repo")          # bare repo => directory ref
    assert r.kind == "hf"
    assert r.repo == "org/repo"
    assert r.path_in_repo == ""
    assert r.is_dir is True
    assert r.url is None


def test_parse_ref_hf_folder_is_dir():
    r = remote.parse_ref("hf:org/repo/UD-Q5_K_M")
    assert r.is_dir is True
    assert r.path_in_repo == "UD-Q5_K_M"
    assert r.url is None


def test_parse_ref_hf_too_short_raises():
    with pytest.raises(remote.RemoteError):
        remote.parse_ref("hf:org")               # no repo component


# huggingface.co web/resolve URL normalization
def test_hf_blob_url_normalized_to_file():
    r = remote.parse_ref(
        "https://huggingface.co/org/repo/blob/main/sub/model.gguf")
    assert r.kind == "hf" and r.is_dir is False
    assert r.repo == "org/repo"
    assert r.path_in_repo == "sub/model.gguf"
    assert r.url == "https://huggingface.co/org/repo/resolve/main/sub/model.gguf"


def test_hf_resolve_url_parsed_as_file():
    r = remote.parse_ref(
        "https://huggingface.co/org/repo/resolve/v2/model.gguf")
    assert r.kind == "hf" and r.is_dir is False
    assert r.revision == "v2"
    assert r.path_in_repo == "model.gguf"


def test_hf_tree_url_is_dir():
    r = remote.parse_ref("https://huggingface.co/org/repo/tree/main/UD-Q5_K_M")
    assert r.kind == "hf" and r.is_dir is True
    assert r.path_in_repo == "UD-Q5_K_M"


def test_non_hf_url_stays_url():
    r = remote.parse_ref("https://example.com/a/model.gguf")
    assert r.kind == "url"


# auth headers: exact-hostname token gate
def test_auth_headers_hf_host(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok")
    h = remote._auth_headers("https://huggingface.co/org/repo/resolve/main/m.gguf")
    assert h == {"Authorization": "Bearer tok"}


def test_auth_headers_hf_subdomain(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok")
    assert remote._auth_headers("https://cdn-lfs.huggingface.co/x") == {
        "Authorization": "Bearer tok"}


@pytest.mark.parametrize("url", [
    "https://huggingface.co.evil.com/m.gguf",       # suffix-spoofed host
    "https://evil.com/?r=huggingface.co",           # host string in the query
    "https://evil.com/huggingface.co/m.gguf",       # host string in the path
    "https://nothuggingface.co/m.gguf",             # missing the dot boundary
])
def test_auth_headers_lookalike_host_gets_no_token(monkeypatch, url):
    monkeypatch.setenv("HF_TOKEN", "tok")
    assert remote._auth_headers(url) == {}


def test_auth_headers_no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    assert remote._auth_headers("https://huggingface.co/x") == {}


def test_auth_headers_token_file_fallback(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "file-tok")
    h = remote._auth_headers("https://huggingface.co/x")
    assert h == {"Authorization": "Bearer file-tok"}


# redirects: Authorization must not follow a cross-host 302
def _redirected(url: str, newurl: str) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer tok"})
    handler = remote._AuthRedirectHandler()
    new = handler.redirect_request(req, None, 302, "Found", {}, newurl)
    assert new is not None
    return new


def test_redirect_strips_auth_cross_host():
    new = _redirected("https://huggingface.co/org/repo/resolve/main/m.gguf",
                      "https://cdn-lfs.example-s3.amazonaws.com/m.gguf?sig=x")
    assert not new.has_header("Authorization")


def test_redirect_keeps_auth_same_host():
    new = _redirected("https://huggingface.co/a", "https://huggingface.co/b")
    assert new.get_header("Authorization") == "Bearer tok"


def test_redirect_strips_auth_on_hf_subdomain_hop():
    # Exact-host rule (requests' rebuild_auth semantics): even an hf-owned
    # CDN subdomain doesn't get the token - its redirect URLs are pre-signed.
    new = _redirected("https://huggingface.co/a",
                      "https://cdn-lfs.huggingface.co/b")
    assert not new.has_header("Authorization")


def test_redirect_strips_auth_hf_lookalike():
    new = _redirected("https://huggingface.co/a",
                      "https://huggingface.co.evil.com/b")
    assert not new.has_header("Authorization")


def test_http_open_opener_wires_the_handler():
    assert any(isinstance(h, remote._AuthRedirectHandler)
               for h in remote._redirect_opener.handlers)


# header parser + classify
def test_parse_header(tmp_path):
    buf = _mint_bytes(tmp_path / "m.gguf", arch="llama")
    arch, gguf_type, tensors = remote._parse_header(buf)
    assert arch == "llama"
    assert gguf_type is None                      # plain model: no general.type
    names = {n for n, _ in tensors}
    assert names == {"plain.f32", "blk.0.attn_q.weight"}


def test_classify_header_supported(tmp_path):
    buf = _mint_bytes(tmp_path / "m.gguf", codec=GT.Q4_0)
    rep = remote.classify_header(buf)
    assert rep.arch == "llama"
    assert rep.histogram.get("Q4_0") == 1
    assert rep.histogram.get("F32") == 1
    assert rep.loadable_codecs is True
    assert rep.n_tensors == 2


def test_classify_header_unsupported(tmp_path):
    # Ternary TQ stays unkernelled (the whole IQ family now loads).
    buf = _mint_bytes(tmp_path / "tq.gguf", codec=GT.TQ1_0)
    rep = remote.classify_header(buf)
    assert "TQ1_0" in rep.unsupported
    assert rep.loadable_codecs is False


def test_bad_magic_raises():
    with pytest.raises(remote.RemoteError):
        remote.classify_header(b"NOTAGGUF" + b"\x00" * 64)


# fetch_header range-read growth
def test_fetch_header_grows(tmp_path):
    full = _mint_bytes(tmp_path / "m.gguf")
    calls = []

    def fake_get(url, end, *, timeout=30.0):
        calls.append(end)
        return full[:end]

    rep = remote.fetch_header("http://x/m.gguf", get=fake_get, initial=8)
    assert rep.arch == "llama"
    assert len(calls) > 1                 # grew from the 8-byte initial prefix
    assert calls[0] == 8


def test_fetch_header_eof_truncated(tmp_path):
    full = _mint_bytes(tmp_path / "m.gguf")
    truncated = full[:40]                   # past the magic, mid metadata block

    def fake_get(url, end, *, timeout=30.0):
        return truncated[:end]             # server has only this much

    with pytest.raises(remote.RemoteError) as ei:
        remote.fetch_header("http://x/m.gguf", get=fake_get, initial=8)
    assert "end of file" in str(ei.value)


def test_fetch_header_default_get_is_call_time(tmp_path, monkeypatch):
    """Monkeypatching the module attr must take effect (default resolved late)."""
    full = _mint_bytes(tmp_path / "m.gguf")
    monkeypatch.setattr(remote, "http_get_prefix",
                        lambda url, end, *, timeout=30.0: full[:end])
    rep = remote.fetch_header("http://x/m.gguf", initial=8)
    assert rep.arch == "llama"


# shard aggregation
def test_aggregate_reports_arch_and_union():
    """Mirrors a real split GGUF: a tensor-free metadata shard carries the arch;
    the weight shards carry arch=None and the (possibly isolated) codecs."""
    meta = remote.HeaderReport("llama", {}, {}, 0)
    w1 = remote.HeaderReport(None, {"Q4_K": 5, "TQ1_0": 2}, {"TQ1_0": 2}, 7)
    w2 = remote.HeaderReport(None, {"Q4_K": 3, "TQ2_0": 1}, {"TQ2_0": 1}, 4)
    agg = remote.aggregate_reports([meta, w1, w2])
    assert agg.arch == "llama"                       # from the metadata shard
    assert agg.histogram == {"Q4_K": 8, "TQ1_0": 2, "TQ2_0": 1}
    assert agg.unsupported == {"TQ1_0": 2, "TQ2_0": 1}
    assert agg.n_tensors == 11
    assert agg.loadable_codecs is False


def test_hf_ref_dotdot_rejected():
    # `..` would survive into the local dest-path join on pull.
    with pytest.raises(remote.RemoteError, match=r"\.\."):
        remote.parse_ref("hf:org/repo/../../elsewhere/x.gguf")


def test_hf_ref_uppercase_gguf_is_a_file():
    ref = remote.parse_ref("hf:org/repo/Model.GGUF")
    assert ref.is_dir is False
    assert ref.filename == "Model.GGUF"


def test_parse_ref_hf_url_rejects_dotdot():
    """A crafted huggingface.co URL must get the same '..' rejection as an
    hf: ref - the path segments feed the local dest-path join on pull."""
    with pytest.raises(remote.RemoteError, match=r"\.\."):
        remote.parse_ref(
            "https://huggingface.co/org/repo/resolve/main/../../../x.gguf")
    # normal URLs still parse
    r = remote.parse_ref("https://huggingface.co/org/repo/blob/main/x.gguf")
    assert r.kind == "hf"
