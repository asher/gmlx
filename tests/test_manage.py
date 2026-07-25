#!/usr/bin/env python3
"""``validate`` / ``pull`` verbs - local + remote header validation and the
download path. CPU-only: the HTTP fetch (``remote.http_get_prefix``) and the
downloaders (``manage._hf_download`` / ``manage._url_download``) are seams.
"""

from __future__ import annotations

import json
import urllib.error

import numpy as np
import pytest

from gguf import GGMLQuantizationType as GT  # noqa: E402
from gguf import GGUFWriter, quants  # noqa: E402

from gmlx import manage, remote  # noqa: E402


def _weight(codec):
    if codec.name.startswith(("IQ", "TQ")):
        from gguf.constants import GGML_QUANT_SIZES
        _, tsize = GGML_QUANT_SIZES[codec]
        return np.zeros((4, 2 * tsize), dtype=np.uint8)
    s = np.random.default_rng(0).standard_normal((8, 64)).astype(np.float32) * 0.1
    return quants.quantize(s, codec)


def _mint(path, *, arch="llama", codec=GT.Q4_0) -> bytes:
    w = GGUFWriter(str(path), arch)
    w.add_tensor("plain.f32", np.zeros((4, 16), dtype=np.float32), raw_dtype=GT.F32)
    w.add_tensor("blk.0.attn_q.weight", _weight(codec), raw_dtype=codec)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with open(path, "rb") as f:
        return f.read()


def _serve(monkeypatch, payload: bytes):
    """Point remote header reads at fixed bytes (no network)."""
    monkeypatch.setattr(remote, "http_get_prefix",
                        lambda url, end, *, timeout=30.0: payload[:end])


def _serve_shards(monkeypatch, mapping: dict):
    """Serve different header bytes per shard, keyed by a substring of the URL."""
    def fake_get(url, end, *, timeout=30.0):
        for key, payload in mapping.items():
            if key in url:
                return payload[:end]
        raise AssertionError(f"unexpected url: {url}")
    monkeypatch.setattr(remote, "http_get_prefix", fake_get)


@pytest.fixture(autouse=True)
def _offline_listing(monkeypatch):
    """Keep the repo-listing call (pull's disk pre-check) off the network.
    Tests that need a listing monkeypatch ``remote.hf_list_dir`` themselves."""
    def offline(*a, **k):
        raise remote.RemoteError("no network in tests")
    monkeypatch.setattr(remote, "hf_list_dir", offline)


def _listing(monkeypatch, files: dict):
    """Fake the HF tree listing with fixed per-file sizes."""
    monkeypatch.setattr(
        remote, "hf_list_dir",
        lambda repo, path, revision="main", **k:
        [(n, "file", s) for n, s in files.items()])


# validate: local
def test_validate_local_ok(tmp_path, capsys):
    p = tmp_path / "ok.gguf"
    _mint(p, arch="llama", codec=GT.Q4_0)
    rc = manage.cmd_validate([str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "loadable" in out and "not loadable" not in out
    assert "llama" in out


def test_validate_local_adapter_companion(tmp_path, capsys):
    # A trained LoRA adapter (general.type=adapter) carries its BASE arch, so
    # without the type check it grades loadable and later 500s if served.
    import gguf

    p = tmp_path / "my-lora.gguf"
    w = GGUFWriter(str(p), "qwen3")
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_tensor("blk.0.attn_q.weight.lora_a",
                 np.zeros((8, 64), dtype=np.float32), raw_dtype=GT.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    rc = manage.cmd_validate([str(p)])
    out = capsys.readouterr().out
    assert rc == 0                                # valid for its purpose
    assert "adapter companion" in out
    assert "[LoRA adapter]" in out
    assert "=> loadable" not in out


def test_validate_local_unsupported_fails(tmp_path, capsys):
    # Ternary TQ stays unkernelled (the whole IQ family now loads).
    p = tmp_path / "tq.gguf"
    _mint(p, codec=GT.TQ1_0)
    rc = manage.cmd_validate([str(p)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not loadable" in out
    assert "TQ1_0" in out


def test_validate_unknown_arch_fails(tmp_path, capsys):
    p = tmp_path / "bogus.gguf"
    _mint(p, arch="totally_made_up_arch", codec=GT.Q4_0)
    rc = manage.cmd_validate([str(p)])
    out = capsys.readouterr().out
    assert rc == 1                       # codecs ok, but arch is unmapped
    assert "[unsupported]" in out


def test_validate_json(tmp_path, capsys):
    p = tmp_path / "ok.gguf"
    _mint(p, codec=GT.Q4_0)
    rc = manage.cmd_validate([str(p), "--json"])
    v = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert v["loadable"] is True
    assert v["arch"] == "llama"
    assert v["codecs"]["Q4_0"] == 1


def test_validate_missing_local_file(tmp_path, capsys):
    rc = manage.cmd_validate([str(tmp_path / "nope.gguf")])
    assert rc == 2
    assert "no such file" in capsys.readouterr().err


def test_validate_incomplete_split_reports_cleanly(tmp_path, capsys):
    # First shard present, 2 and 3 missing: validate must report the gap as a clean
    # verdict (rc 2 + stderr message), not crash with a FileNotFoundError traceback.
    _mint(tmp_path / "m-00001-of-00003.gguf", codec=GT.Q4_0)
    rc = manage.cmd_validate([str(tmp_path / "m-00001-of-00003.gguf")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "incomplete split GGUF" in err and "2/3 shard(s) missing" in err


# validate: remote
def test_validate_remote_ok(tmp_path, monkeypatch, capsys):
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    rc = manage.cmd_validate(["hf:org/repo/model.gguf"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "loadable" in out
    assert "source: hf" in out


def test_validate_remote_unsupported_fails(tmp_path, monkeypatch, capsys):
    _serve(monkeypatch, _mint(tmp_path / "tq.gguf", codec=GT.TQ1_0))
    rc = manage.cmd_validate(["https://example.com/iq.gguf"])
    assert rc == 1
    assert "not loadable" in capsys.readouterr().out


# pull
def _list(monkeypatch, entries):
    """Fake the HF tree listing. ``entries`` is [(path, type, size), ...]."""
    monkeypatch.setattr(remote, "hf_list_dir",
                        lambda repo, path, rev, **kw: entries)


def test_group_shard_sets():
    paths = [
        "UD-Q5_K_M/m-00001-of-00004.gguf", "UD-Q5_K_M/m-00002-of-00004.gguf",
        "UD-Q5_K_M/m-00003-of-00004.gguf", "UD-Q5_K_M/m-00004-of-00004.gguf",
        "UD-Q2_K/m-00001-of-00002.gguf", "UD-Q2_K/m-00002-of-00002.gguf",
        "plain.gguf",
    ]
    reps = sorted(manage._group_shard_sets(paths).values())
    assert reps == ["UD-Q2_K/m-00001-of-00002.gguf",
                    "UD-Q5_K_M/m-00001-of-00004.gguf", "plain.gguf"]


def test_resolve_folder_single_set(monkeypatch, capsys):
    _list(monkeypatch, [
        ("UD-Q5_K_M/m-00001-of-00002.gguf", "file", 8),
        ("UD-Q5_K_M/m-00002-of-00002.gguf", "file", 49),
    ])
    ref = manage._resolve_to_file(remote.parse_ref("hf:org/repo/UD-Q5_K_M"))
    assert ref.is_dir is False
    assert ref.path_in_repo == "UD-Q5_K_M/m-00001-of-00002.gguf"
    assert "[resolved]" in capsys.readouterr().err


def test_resolve_folder_multi_lists(monkeypatch):
    _list(monkeypatch, [
        ("UD-Q5_K_M/m-00001-of-00002.gguf", "file", 8),
        ("UD-Q5_K_M/m-00002-of-00002.gguf", "file", 49),
        ("UD-Q2_K/m.gguf", "file", 20),
    ])
    with pytest.raises(remote.RemoteError) as ei:
        manage._resolve_to_file(remote.parse_ref("hf:org/repo"))
    msg = str(ei.value)
    assert "2 GGUF models" in msg
    assert "hf:org/repo/UD-Q5_K_M/m-00001-of-00002.gguf" in msg
    assert "hf:org/repo/UD-Q2_K/m.gguf" in msg


def test_validate_repo_multi_lists_stdout_exit_zero(monkeypatch, capsys):
    _list(monkeypatch, [
        ("UD-Q5_K_M/m-00001-of-00002.gguf", "file", 8),
        ("UD-Q5_K_M/m-00002-of-00002.gguf", "file", 49),
        ("UD-Q2_K/m.gguf", "file", 20),
    ])
    rc = manage.cmd_validate(["hf:org/repo"])
    assert rc == 0
    out, err = capsys.readouterr()
    assert "2 GGUF models" in out
    assert "hf:org/repo/UD-Q5_K_M/m-00001-of-00002.gguf" in out
    assert "hf:org/repo/UD-Q2_K/m.gguf" in out
    assert "error" not in err


def test_validate_repo_multi_json(monkeypatch, capsys):
    _list(monkeypatch, [
        ("a.gguf", "file", 8),
        ("b.gguf", "file", 9),
    ])
    rc = manage.cmd_validate(["hf:org/repo", "--json"])
    assert rc == 0
    v = json.loads(capsys.readouterr().out)
    assert v["repo"] == "org/repo"
    assert v["models"] == ["hf:org/repo/a.gguf", "hf:org/repo/b.gguf"]


def test_pull_repo_multi_still_errors(tmp_path, monkeypatch, capsys):
    _list(monkeypatch, [
        ("a.gguf", "file", 8),
        ("b.gguf", "file", 9),
    ])
    rc = manage.cmd_pull(["hf:org/repo", "--to", str(tmp_path)])
    assert rc != 0
    assert "pass one" in capsys.readouterr().err


def test_resolve_folder_no_gguf_lists_subdirs(monkeypatch):
    _list(monkeypatch, [("UD-Q5_K_M", "directory", 0),
                        ("UD-Q2_K_XL", "directory", 0),
                        ("README.md", "file", 1)])
    with pytest.raises(remote.RemoteError) as ei:
        manage._resolve_to_file(remote.parse_ref("hf:org/repo"))
    msg = str(ei.value)
    assert "no .gguf files" in msg
    assert "hf:org/repo/UD-Q5_K_M" in msg


def test_validate_folder_resolves_and_checks(tmp_path, monkeypatch, capsys):
    _list(monkeypatch, [("d/m-00001-of-00001.gguf", "file", 8)])
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    rc = manage.cmd_validate(["hf:org/repo/d"])
    assert rc == 0
    assert "loadable" in capsys.readouterr().out


def test_validate_remote_split_aggregates(tmp_path, monkeypatch, capsys):
    # shard 1 is loadable on its own; shard 2 hides a TQ codec. The aggregate
    # verdict must be not loadable - the whole point of checking every shard.
    s1 = _mint(tmp_path / "s1.gguf", arch="llama", codec=GT.Q4_0)
    s2 = _mint(tmp_path / "s2.gguf", arch="llama", codec=GT.TQ1_0)
    _serve_shards(monkeypatch, {"00001-of-00002": s1, "00002-of-00002": s2})
    rc = manage.cmd_validate(["hf:org/repo/model-00001-of-00002.gguf"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "across 2 shards" in out
    assert "TQ1_0" in out and "not loadable" in out
    assert "llama" in out                    # arch came from shard 1


def test_validate_remote_split_from_any_shard(tmp_path, monkeypatch):
    # A ref to shard 2 still enumerates and checks the whole set.
    s1 = _mint(tmp_path / "s1.gguf", arch="llama", codec=GT.Q4_0)
    s2 = _mint(tmp_path / "s2.gguf", arch="llama", codec=GT.TQ1_0)
    _serve_shards(monkeypatch, {"00001-of-00002": s1, "00002-of-00002": s2})
    rc = manage.cmd_validate(["hf:org/repo/model-00002-of-00002.gguf", "--json"])
    assert rc == 1


def test_remote_shard_urls_hf_split():
    ref = remote.parse_ref("hf:org/repo/dir/m-00001-of-00002.gguf")
    urls = manage._remote_shard_urls(ref)
    assert len(urls) == 2
    assert urls[0].endswith("/resolve/main/dir/m-00001-of-00002.gguf")
    assert urls[1].endswith("/resolve/main/dir/m-00002-of-00002.gguf")


def test_remote_shard_urls_url_split():
    ref = remote.parse_ref("https://x/y/m-00001-of-00003.gguf")
    assert manage._remote_shard_urls(ref) == [
        "https://x/y/m-00001-of-00003.gguf",
        "https://x/y/m-00002-of-00003.gguf",
        "https://x/y/m-00003-of-00003.gguf",
    ]


def test_remote_shard_urls_single_keeps_original():
    ref = remote.parse_ref("hf:org/repo/model.gguf")
    assert manage._remote_shard_urls(ref) == [ref.url]


def test_pull_split_refuses_on_late_shard_codec(tmp_path, monkeypatch, capsys):
    s1 = _mint(tmp_path / "s1.gguf", arch="llama", codec=GT.Q4_0)
    s2 = _mint(tmp_path / "s2.gguf", arch="llama", codec=GT.TQ1_0)
    _serve_shards(monkeypatch, {"00001-of-00002": s1, "00002-of-00002": s2})
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: pytest.fail("must not download"))
    rc = manage.cmd_pull(
        ["hf:org/repo/model-00001-of-00002.gguf", "--to", str(tmp_path)])
    assert rc == 1
    assert "refusing to download" in capsys.readouterr().err


def test_pull_out_is_alias_for_to(tmp_path, monkeypatch, capsys):
    # --out maps to the same destination as --to (the codec refusal path proves
    # parsing reached the dest dir).
    s1 = _mint(tmp_path / "s1.gguf", arch="llama", codec=GT.Q4_0)
    s2 = _mint(tmp_path / "s2.gguf", arch="llama", codec=GT.TQ1_0)
    _serve_shards(monkeypatch, {"00001-of-00002": s1, "00002-of-00002": s2})
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: pytest.fail("must not download"))
    rc = manage.cmd_pull(
        ["hf:org/repo/model-00001-of-00002.gguf", "--out", str(tmp_path)])
    assert rc == 1
    assert "refusing to download" in capsys.readouterr().err


def test_shard_names_split():
    names = manage._shard_names("model-00001-of-00003.gguf")
    assert names == [
        "model-00001-of-00003.gguf",
        "model-00002-of-00003.gguf",
        "model-00003-of-00003.gguf",
    ]


def test_shard_names_single():
    assert manage._shard_names("model.gguf") == ["model.gguf"]


def test_pull_local_ref_rejected(capsys):
    rc = manage.cmd_pull(["/some/local/model.gguf"])
    assert rc == 2
    assert "already on disk" in capsys.readouterr().err


def test_pull_skips_local_sibling_and_pulls_the_hf_ref(tmp_path, monkeypatch,
                                                        capsys):
    """A bare sibling name that collides with a CWD file parses as a local ref.
    pull used to print `using it` and then abort the whole pull, downloading
    nothing - including the hf ref that preceded it."""
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mmproj.gguf").write_bytes(b"GGUF")
    got = []
    monkeypatch.setattr(manage, "_hf_download",
                        lambda repo, filename, revision, dest_dir:
                        got.append(filename) or f"{dest_dir}/{filename}")

    rc = manage.cmd_pull(["hf:org/repo/model.gguf", "mmproj.gguf",
                          "--to", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 0
    assert got == ["model.gguf"]           # the hf ref still downloads
    assert "skipping" in err and "hf:org/repo/mmproj.gguf" in err


def test_expand_refs_notes_a_local_collision_without_an_anchor(
        tmp_path, monkeypatch, capsys):
    """An http(s) ref resets the sibling anchor, so a following bare name that
    matches a CWD file used to be filtered with no note at all - a silent skip
    that flips with the working directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mmproj.gguf").write_bytes(b"GGUF")
    refs = manage._expand_refs(["http://host/model.gguf", "mmproj.gguf"])
    err = capsys.readouterr().err
    assert [r.kind for r in refs] == ["url", "local"]
    assert "skipping" in err and "'mmproj.gguf'" in err
    assert "repo sibling" not in err       # no anchor, so no hf: hint


def test_pull_local_only_refs_still_error(capsys):
    rc = manage.cmd_pull(["/some/local/model.gguf"])
    assert rc == 2
    assert "already on disk" in capsys.readouterr().err


def test_pull_adapter_refusal_omits_arch_reason(tmp_path, monkeypatch, capsys):
    """An unloadable LoRA adapter is refused for its codec, not for a
    nonexistent arch problem (the arch gate is skipped for adapters)."""
    import gguf

    p = tmp_path / "lora.gguf"
    w = GGUFWriter(str(p), "llama")
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_tensor("blk.0.attn_q.weight.lora_a", _weight(GT.TQ1_0),
                 raw_dtype=GT.TQ1_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    _serve(monkeypatch, p.read_bytes())
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: pytest.fail("must not download"))

    rc = manage.cmd_pull(["hf:org/repo/lora.gguf", "--to", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no kernel for codec" in err
    assert "unsupported arch" not in err


def test_pull_refuses_unloadable(tmp_path, monkeypatch, capsys):
    _serve(monkeypatch, _mint(tmp_path / "iq.gguf", codec=GT.TQ1_0))
    called = []
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: called.append((ref, dest)) or [])
    rc = manage.cmd_pull(["hf:org/repo/iq.gguf", "--to", str(tmp_path)])
    assert rc == 1
    assert called == []                  # never downloaded
    assert "refusing to download" in capsys.readouterr().err


def test_pull_force_downloads_unloadable(tmp_path, monkeypatch):
    _serve(monkeypatch, _mint(tmp_path / "iq.gguf", codec=GT.TQ1_0))
    called = []
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: called.append((ref, dest)) or ["/x"])
    rc = manage.cmd_pull(["hf:org/repo/iq.gguf", "--to", str(tmp_path), "--force"])
    assert rc == 0
    assert len(called) == 1


def test_pull_ok_downloads(tmp_path, monkeypatch, capsys):
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    got = []

    def fake_hf(repo, filename, revision, dest_dir):
        got.append((repo, filename, revision, dest_dir))
        return f"{dest_dir}/{filename}"

    monkeypatch.setattr(manage, "_hf_download", fake_hf)
    rc = manage.cmd_pull(["hf:org/repo/model.gguf", "--to", str(tmp_path)])
    assert rc == 0
    assert got == [("org/repo", "model.gguf", "main", str(tmp_path))]
    assert "downloaded 1 file" in capsys.readouterr().out


def test_pull_sharded_downloads_all(tmp_path, monkeypatch):
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    got = []
    monkeypatch.setattr(
        manage, "_hf_download",
        lambda repo, filename, revision, dest: got.append(filename) or filename)
    rc = manage.cmd_pull(
        ["hf:org/repo/model-00001-of-00003.gguf", "--to", str(tmp_path)])
    assert rc == 0
    assert got == [
        "model-00001-of-00003.gguf",
        "model-00002-of-00003.gguf",
        "model-00003-of-00003.gguf",
    ]


# pull: disk-space pre-check
def test_pull_insufficient_space_refuses(tmp_path, monkeypatch, capsys):
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    _listing(monkeypatch, {"model.gguf": 10 * 1024**3})
    monkeypatch.setattr(manage, "_disk_free", lambda p: 1 * 1024**3)
    called = []
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: called.append(dest) or ["/x"])
    rc = manage.cmd_pull(["hf:org/repo/model.gguf", "--to", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 1 and called == []
    assert "not enough disk space" in err
    assert "10.0 GB" in err and "1.0 GB" in err


def test_pull_sufficient_space_proceeds(tmp_path, monkeypatch):
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    _listing(monkeypatch, {"model.gguf": 10 * 1024**3})
    monkeypatch.setattr(manage, "_disk_free", lambda p: 20 * 1024**3)
    called = []
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: called.append(dest) or ["/x"])
    rc = manage.cmd_pull(["hf:org/repo/model.gguf", "--to", str(tmp_path)])
    assert rc == 0 and len(called) == 1


def test_pull_part_file_reduces_need(tmp_path, monkeypatch):
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    _listing(monkeypatch, {"model.gguf": 1000})
    (tmp_path / "model.gguf.part").write_bytes(b"x" * 800)
    monkeypatch.setattr(manage, "_disk_free", lambda p: 300)
    called = []
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: called.append(dest) or ["/x"])
    rc = manage.cmd_pull(["hf:org/repo/model.gguf", "--to", str(tmp_path)])
    assert rc == 0 and len(called) == 1           # need 200 < 300 free


def test_pull_sharded_space_sums_all_shards(tmp_path, monkeypatch, capsys):
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    _listing(monkeypatch, {"model-00001-of-00002.gguf": 600,
                           "model-00002-of-00002.gguf": 600})
    monkeypatch.setattr(manage, "_disk_free", lambda p: 1000)
    called = []
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: called.append(dest) or ["/x"])
    rc = manage.cmd_pull(
        ["hf:org/repo/model-00001-of-00002.gguf", "--to", str(tmp_path)])
    assert rc == 1 and called == []               # 1200 needed, 1000 free
    assert "not enough disk space" in capsys.readouterr().err


def test_pull_unknown_size_proceeds(tmp_path, monkeypatch):
    # A shard absent from the listing (or a failed listing, the autouse
    # default) means the size is unknown: never block the download.
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    _listing(monkeypatch, {"model-00001-of-00002.gguf": 600})
    monkeypatch.setattr(manage, "_disk_free", lambda p: 0)
    called = []
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: called.append(dest) or ["/x"])
    rc = manage.cmd_pull(
        ["hf:org/repo/model-00001-of-00002.gguf", "--to", str(tmp_path)])
    assert rc == 0 and len(called) == 1


def test_pull_force_bypasses_space_check(tmp_path, monkeypatch):
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    _listing(monkeypatch, {"model.gguf": 10 * 1024**3})
    monkeypatch.setattr(manage, "_disk_free", lambda p: 0)
    called = []
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: called.append(dest) or ["/x"])
    rc = manage.cmd_pull(
        ["hf:org/repo/model.gguf", "--to", str(tmp_path), "--force"])
    assert rc == 0 and len(called) == 1


def _write_config(path, model_dirs):
    import yaml
    path.write_text(yaml.safe_dump({"server": {"model_dirs": model_dirs}}))
    return str(path)


def test_pull_default_dest_is_model_dir(tmp_path, monkeypatch):
    # No --to: the default destination is the config's first model_dirs root, and
    # an hf ref nests under <dir>/<org>__<repo>/ so discovery's recursive scan
    # (and `sync-models`) picks it up.
    lib = tmp_path / "lib"
    cfg = _write_config(tmp_path / "config.yaml", [str(lib)])
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    dests = []
    monkeypatch.setattr(
        manage, "_hf_download",
        lambda repo, filename, revision, dest: dests.append(dest) or filename)
    rc = manage.cmd_pull(["hf:org/repo/model.gguf", "--config", cfg])
    assert rc == 0
    assert dests == [str(lib / "org__repo")]    # model_dirs[0]/<org>__<repo>, not cwd


def test_pull_to_is_literal_no_nesting(tmp_path, monkeypatch):
    # --to DIR writes straight into DIR - no <org>__<repo> subdir.
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    dests = []
    monkeypatch.setattr(
        manage, "_hf_download",
        lambda repo, filename, revision, dest: dests.append(dest) or filename)
    rc = manage.cmd_pull(["hf:org/repo/model.gguf", "--to", str(tmp_path)])
    assert rc == 0
    assert dests == [str(tmp_path)]


def test_pull_no_config_errors(tmp_path, monkeypatch, capsys):
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    monkeypatch.setattr(manage, "_hf_download",
                        lambda *a: pytest.fail("must not download"))
    # No --to and no config in the standard locations.
    from gmlx import config as cfgmod
    monkeypatch.setattr(cfgmod, "default_config_paths", lambda: [])
    rc = manage.cmd_pull(["hf:org/repo/model.gguf"])
    assert rc == 2
    assert "needs a server config" in capsys.readouterr().err


def test_pull_multiple_files_same_repo(tmp_path, monkeypatch):
    # A model plus a bare-named sibling (e.g. an mmproj) fetch in one go, both
    # resolved against the first ref's repo.
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    got = []
    monkeypatch.setattr(
        manage, "_hf_download",
        lambda repo, filename, revision, dest: got.append((repo, filename)) or filename)
    rc = manage.cmd_pull(
        ["hf:org/repo/model-Q4_K_M.gguf", "mmproj-F16.gguf", "--to", str(tmp_path)])
    assert rc == 0
    assert got == [("org/repo", "model-Q4_K_M.gguf"),
                   ("org/repo", "mmproj-F16.gguf")]


def test_pull_bare_filename_needs_anchor(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(manage, "_hf_download",
                        lambda *a: pytest.fail("must not download"))
    rc = manage.cmd_pull(["mmproj-F16.gguf", "--to", str(tmp_path)])
    assert rc == 2
    assert "is not a model ref" in capsys.readouterr().err


def test_pull_url_downloads(tmp_path, monkeypatch):
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    got = []

    def fake_url(url, dest_path):
        got.append((url, dest_path))
        return dest_path

    monkeypatch.setattr(manage, "_url_download", fake_url)
    rc = manage.cmd_pull(
        ["https://example.com/d/model.gguf", "--to", str(tmp_path)])
    assert rc == 0
    assert got[0][0] == "https://example.com/d/model.gguf"


def test_pull_url_single_file_keeps_query(tmp_path, monkeypatch):
    # The validated URL must be the downloaded URL - a signed/?download=true
    # query string can't be dropped between the two.
    _serve(monkeypatch, _mint(tmp_path / "m.gguf", codec=GT.Q4_0))
    got = []
    monkeypatch.setattr(manage, "_url_download",
                        lambda url, dest_path: got.append(url) or dest_path)
    url = "https://example.com/d/model.gguf?download=true&sig=abc"
    rc = manage.cmd_pull([url, "--to", str(tmp_path)])
    assert rc == 0
    assert got == [url]


def test_url_download_passes_timeout_and_writes(tmp_path, monkeypatch):
    dest = tmp_path / "m.gguf"
    seen = {}

    class Resp:
        def __init__(self):
            self.data = b"GGUFbytes"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            d, self.data = self.data, b""
            return d

    def fake_open(req, *, timeout):
        seen["timeout"] = timeout
        seen["url"] = req.full_url
        return Resp()

    monkeypatch.setattr(remote, "http_open", fake_open)
    out = manage._url_download("https://example.com/m.gguf", str(dest))
    assert out == str(dest)
    assert dest.read_bytes() == b"GGUFbytes"
    assert seen["timeout"] == 30
    assert seen["url"] == "https://example.com/m.gguf"


def test_url_download_keeps_partial_on_error(tmp_path, monkeypatch):
    # A failed transfer must leave a .part behind (so a re-run resumes) and must
    # NOT publish a half file at the final path.
    dest = tmp_path / "m.gguf"

    class Drops:
        def __init__(self):
            self.first = True

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            if self.first:                   # one chunk lands, then the wire dies
                self.first = False
                return b"partial"
            raise OSError("connection dropped")

    monkeypatch.setattr(remote, "http_open",
                        lambda req, *, timeout: Drops())
    with pytest.raises(OSError, match="connection dropped"):
        manage._url_download("https://example.com/m.gguf", str(dest))
    assert not dest.exists()                          # no half file at the final path
    assert (tmp_path / "m.gguf.part").read_bytes() == b"partial"   # kept for resume


def test_url_download_resumes_from_part(tmp_path, monkeypatch):
    # A pre-existing .part triggers a Range request; a 206 means we append.
    dest = tmp_path / "m.gguf"
    (tmp_path / "m.gguf.part").write_bytes(b"GGUF")   # 4 bytes already on disk
    seen = {}

    class Resp:
        status = 206

        def __init__(self):
            self.data = b"more"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            d, self.data = self.data, b""
            return d

    def fake_open(req, *, timeout):
        seen["range"] = req.get_header("Range")
        return Resp()

    monkeypatch.setattr(remote, "http_open", fake_open)
    out = manage._url_download("https://example.com/m.gguf", str(dest))
    assert out == str(dest)
    assert seen["range"] == "bytes=4-"
    assert dest.read_bytes() == b"GGUFmore"           # appended onto the partial
    assert not (tmp_path / "m.gguf.part").exists()     # renamed into place


def test_url_download_skips_completed(tmp_path, monkeypatch):
    # An already-complete file short-circuits - idempotent re-pull, no network.
    dest = tmp_path / "m.gguf"
    dest.write_bytes(b"done")
    monkeypatch.setattr(remote, "http_open",
                        lambda *a, **k: pytest.fail("must not hit the network"))
    assert manage._url_download("https://example.com/m.gguf", str(dest)) == str(dest)
    assert dest.read_bytes() == b"done"


def test_url_download_416_without_content_range_refuses(tmp_path, monkeypatch):
    # 416 with no Content-Range gives no way to tell a complete .part from a
    # stale oversized one - only an exact size match proves completion, so the
    # .part is kept for inspection instead of promoted unverified.
    dest = tmp_path / "m.gguf"
    (tmp_path / "m.gguf.part").write_bytes(b"GGUFcomplete")

    def fake_open(req, *, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 416, "Range Not Satisfiable", {}, None)

    monkeypatch.setattr(remote, "http_open", fake_open)
    with pytest.raises(remote.RemoteError, match="no Content-Range"):
        manage._url_download("https://example.com/m.gguf", str(dest))
    assert not dest.exists()
    assert (tmp_path / "m.gguf.part").read_bytes() == b"GGUFcomplete"


def test_url_download_rejects_truncated_transfer(tmp_path, monkeypatch):
    # A clean early close reads as EOF (read() returns b"" mid-transfer), so
    # the byte count must be checked against Content-Length before the rename;
    # promoting would make the truncated file permanent via the completed-
    # download short-circuit.
    dest = tmp_path / "m.gguf"

    class Truncated:
        def __init__(self):
            self.data = b"x" * 50

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            d, self.data = self.data, b""
            return d

        def getheader(self, name):
            return "100" if name == "Content-Length" else None

    monkeypatch.setattr(remote, "http_open",
                        lambda req, *, timeout: Truncated())
    with pytest.raises(remote.RemoteError, match="closed early"):
        manage._url_download("https://example.com/m.gguf", str(dest))
    assert not dest.exists()                          # nothing promoted
    assert (tmp_path / "m.gguf.part").read_bytes() == b"x" * 50  # resumable


def test_hf_download_delegates_to_url_download(tmp_path, monkeypatch):
    # _hf_download resolves the HF URL and delegates to _url_download.
    seen = {}

    def fake_url_download(url, dest_path):
        seen["url"] = url
        seen["dest"] = dest_path
        from pathlib import Path
        Path(dest_path).touch()
        return dest_path

    monkeypatch.setattr(manage, "_url_download", fake_url_download)
    out = manage._hf_download("org/repo", "model.gguf", "main", str(tmp_path))
    assert out == str(tmp_path / "model.gguf")
    assert seen["url"] == "https://huggingface.co/org/repo/resolve/main/model.gguf"


def test_hf_download_creates_subdirectories(tmp_path, monkeypatch):
    # Filenames with subdirectories (e.g. UD-Q5_K_M/model.gguf) are handled.
    seen = {}

    def fake_url_download(url, dest_path):
        seen["dest"] = dest_path
        from pathlib import Path
        Path(dest_path).touch()
        return dest_path

    monkeypatch.setattr(manage, "_url_download", fake_url_download)
    manage._hf_download("org/repo", "sub/model.gguf", "main", str(tmp_path))
    assert (tmp_path / "sub").is_dir()
    assert seen["dest"] == str(tmp_path / "sub" / "model.gguf")


# mmproj companion files
def _mint_mmproj(path) -> bytes:
    """A float-only GGUF with general.architecture='clip' - the mmproj shape."""
    w = GGUFWriter(str(path), "clip")
    w.add_tensor("v.patch_embd.weight", np.zeros((4, 16), dtype=np.float32),
                 raw_dtype=GT.F32)
    w.add_tensor("v.blk.0.attn_q.weight",
                 np.zeros((8, 8), dtype=np.float16), raw_dtype=GT.F16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with open(path, "rb") as f:
        return f.read()


def test_validate_local_mmproj_companion(tmp_path, capsys):
    p = tmp_path / "mmproj.gguf"
    _mint_mmproj(p)
    rc = manage.cmd_validate([str(p)])
    out = capsys.readouterr().out
    assert rc == 0                       # valid for its purpose
    assert "mmproj companion" in out
    assert "--mmproj" in out             # the message says how to use it
    assert "not loadable" not in out and "[unsupported]" not in out


def test_validate_mmproj_json(tmp_path, capsys):
    p = tmp_path / "mmproj.gguf"
    _mint_mmproj(p)
    rc = manage.cmd_validate([str(p), "--json"])
    v = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert v["mmproj"] is True
    assert v["loadable"] is False        # not a standalone model
    assert v["usable"] is True


def test_pull_mmproj_no_force_needed(tmp_path, monkeypatch):
    _serve(monkeypatch, _mint_mmproj(tmp_path / "mmproj.gguf"))
    called = []
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: called.append((ref, dest)) or ["/x"])
    rc = manage.cmd_pull(["hf:org/repo/mmproj-model-f16.gguf",
                          "--to", str(tmp_path)])
    assert rc == 0
    assert len(called) == 1


# list - the models a config defines (ids, aliases, default), not a directory scan

def _write_cfg(tmp_path, body):
    lib = tmp_path / "lib"
    lib.mkdir(exist_ok=True)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(body.replace("<LIB>", str(lib)))
    return cfg


_LIST_CFG = """
server:
  model_dirs:
    - <LIB>
  defaults:
    model: qwen3-0.6b
models:
  qwen3-0.6b:
    path: qwen.gguf
  gemma-e2b:
    path: gemma.gguf
    mmproj: mmproj.gguf
aliases:
  fast: qwen3-0.6b
"""


def test_list_config_models_aliases_default(tmp_path, capsys):
    cfg = _write_cfg(tmp_path, _LIST_CFG)
    rc = manage.cmd_list(["--config", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "qwen3-0.6b" in out and "gemma-e2b" in out
    assert "vlm" in out                       # gemma carries an mmproj
    assert "fast" in out and "->" in out      # the alias is listed
    assert "default model" in out             # the default-model footer
    assert "* qwen3-0.6b" in out              # default-marked row
    assert "NAME" in out and "SIZE" in out    # table header
    assert "missing" in out                   # paths don't exist in tmp_path
    assert "qwen.gguf" not in out             # paths hidden by default


def test_list_paths_flag_and_size(tmp_path, capsys):
    cfg = _write_cfg(tmp_path, _LIST_CFG)
    lib = tmp_path / "lib"
    (lib / "qwen.gguf").write_bytes(b"x" * 2_000_000)
    rc = manage.cmd_list(["--config", str(cfg), "-v"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "qwen.gguf" in out                 # -v shows the path lines
    assert "2 MB" in out                      # size read from disk
    assert "missing" in out                   # gemma.gguf still absent


def test_list_json(tmp_path, capsys):
    cfg = _write_cfg(tmp_path, _LIST_CFG)
    rc = manage.cmd_list(["--config", str(cfg), "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert doc["config"].endswith("cfg.yaml")
    assert doc["default_model"] == "qwen3-0.6b"
    assert doc["aliases"]["fast"] == "qwen3-0.6b"
    ids = [m["id"] for m in doc["models"]]
    assert ids == ["gemma-e2b", "qwen3-0.6b"]            # sorted by id
    gemma = next(m for m in doc["models"] if m["id"] == "gemma-e2b")
    assert gemma["source"] == "config" and "vlm" in gemma["flags"]


def test_list_missing_explicit_config_exits_nonzero(tmp_path, capsys):
    rc = manage.cmd_list(["--config", str(tmp_path / "nope.yaml")])
    assert rc == 2
    assert "no config file" in capsys.readouterr().err


def test_list_no_config_found(tmp_path, monkeypatch, capsys):
    from gmlx import config as gcfg
    monkeypatch.setattr(gcfg, "default_config_paths",
                        lambda: [tmp_path / "absent.yaml"])
    rc = manage.cmd_list([])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no server config found" in err and "gmlx init" in err


# ps

def test_ps_unreachable_server(capsys):
    rc = manage.cmd_ps(["--url", "http://127.0.0.1:1"])   # nothing listens there
    assert rc == 3                       # not-running is a status, like `status`
    assert "no gmlx server reachable" in capsys.readouterr().err


def test_ps_bare_resolves_managed_target(monkeypatch, capsys):
    # Bare `ps` must share status/stop's target resolution instead of silently
    # probing 8080 (which can be a different server than the managed one).
    from gmlx import lifecycle
    monkeypatch.setattr(lifecycle, "auto_target", lambda h, p: ("127.0.0.1", 1))
    rc = manage.cmd_ps([])
    assert rc == 3
    assert "127.0.0.1:1" in capsys.readouterr().err       # probed the resolved target


# rm - delete a model's files and its config entry

_RM_CFG = """
# keep this comment
server:
  model_dirs:
    - <LIB>
  defaults:
    model: gone
models:
  gone:
    path: gone.gguf
  keep:
    path: keep.gguf     # hand note
aliases:
  fast: gone@coding
  slow: keep
"""


def _rm_setup(tmp_path, body=_RM_CFG, files=("gone.gguf", "keep.gguf")):
    cfg = _write_cfg(tmp_path, body)
    lib = tmp_path / "lib"
    for f in files:
        _mint(lib / f)
    return cfg, lib


def test_rm_reloads_running_server(monkeypatch, tmp_path):
    # rm rewrites the config like init/sync-models/pull do - it must SIGHUP a
    # server running that config the same way, or the removed id stays served
    # (a phantom in /v1/models) until a manual reload.
    from gmlx import server as srv_mod
    calls = []
    monkeypatch.setattr(srv_mod, "_reload_running",
                        lambda path, skip: calls.append((str(path), skip)))
    cfg, lib = _rm_setup(tmp_path)
    assert manage.cmd_rm(["gone", "--config", str(cfg), "--yes"]) == 0
    assert calls == [(str(cfg), False)]
    calls.clear()
    assert manage.cmd_rm(["keep", "--config", str(cfg), "--yes",
                          "--no-reload"]) == 0
    assert calls == [(str(cfg), True)]


def test_rm_removes_file_entry_alias_default(tmp_path, capsys):
    import yaml
    cfg, lib = _rm_setup(tmp_path)
    rc = manage.cmd_rm(["gone", "--config", str(cfg), "--yes"])
    out = capsys.readouterr().out
    assert rc == 0
    assert not (lib / "gone.gguf").exists()
    assert (lib / "keep.gguf").exists()
    doc = yaml.safe_load(cfg.read_text())
    assert "gone" not in doc["models"] and "keep" in doc["models"]
    assert doc["aliases"] == {"slow": "keep"}
    assert "model" not in (doc["server"].get("defaults") or {})
    text = cfg.read_text()
    assert "# keep this comment" in text and "# hand note" in text
    assert "dropping alias fast" in out
    assert "clearing server.defaults.model" in out


def test_rm_alias_resolves_to_id(tmp_path, capsys):
    cfg, lib = _rm_setup(tmp_path)
    rc = manage.cmd_rm(["fast", "--config", str(cfg), "--yes"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "alias fast -> gone" in out
    assert not (lib / "gone.gguf").exists()


def test_rm_sharded_removes_all_and_parts(tmp_path):
    body = _RM_CFG.replace("path: gone.gguf",
                           "path: gone-00001-of-00002.gguf")
    cfg, lib = _rm_setup(tmp_path, body,
                         files=("gone-00001-of-00002.gguf",
                                "gone-00002-of-00002.gguf", "keep.gguf"))
    (lib / "gone-00002-of-00002.gguf.part").write_bytes(b"x")
    rc = manage.cmd_rm(["gone", "--config", str(cfg), "--yes"])
    assert rc == 0
    assert not list(lib.glob("gone-*"))
    assert (lib / "keep.gguf").exists()


def test_rm_shared_mmproj_kept_exclusive_removed(tmp_path, capsys):
    body = """
server:
  model_dirs:
    - <LIB>
models:
  a:
    path: a.gguf
    mmproj: mm.gguf
  b:
    path: b.gguf
    mmproj: mm.gguf
"""
    cfg, lib = _rm_setup(tmp_path, body, files=("a.gguf", "b.gguf", "mm.gguf"))
    rc = manage.cmd_rm(["a", "--config", str(cfg), "--yes"])
    assert rc == 0
    assert not (lib / "a.gguf").exists()
    assert (lib / "mm.gguf").exists()
    assert "mmproj kept" in capsys.readouterr().out
    rc = manage.cmd_rm(["b", "--config", str(cfg), "--yes"])
    assert rc == 0
    assert not (lib / "b.gguf").exists()
    assert not (lib / "mm.gguf").exists()      # now exclusive


def test_rm_keep_files(tmp_path):
    import yaml
    cfg, lib = _rm_setup(tmp_path)
    rc = manage.cmd_rm(["gone", "--config", str(cfg), "--yes", "--keep-files"])
    assert rc == 0
    assert (lib / "gone.gguf").exists()
    assert "gone" not in yaml.safe_load(cfg.read_text())["models"]


def test_rm_decline_changes_nothing(tmp_path, monkeypatch, capsys):
    import yaml
    cfg, lib = _rm_setup(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    rc = manage.cmd_rm(["gone", "--config", str(cfg)])
    assert rc == 1
    assert "aborted" in capsys.readouterr().err
    assert (lib / "gone.gguf").exists()
    assert "gone" in yaml.safe_load(cfg.read_text())["models"]


def test_rm_non_tty_needs_yes(tmp_path, monkeypatch, capsys):
    cfg, lib = _rm_setup(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = manage.cmd_rm(["gone", "--config", str(cfg)])
    assert rc == 2
    assert "pass --yes" in capsys.readouterr().err
    assert (lib / "gone.gguf").exists()


def test_rm_unknown_id(tmp_path, capsys):
    cfg, _lib = _rm_setup(tmp_path)
    rc = manage.cmd_rm(["nope", "--config", str(cfg), "--yes"])
    assert rc == 2
    assert "unknown model id" in capsys.readouterr().err


def test_rm_json(tmp_path, capsys):
    cfg, lib = _rm_setup(tmp_path)
    assert manage.cmd_rm(["gone", "--config", str(cfg), "--json"]) == 2
    assert "--json requires --yes" in capsys.readouterr().err
    rc = manage.cmd_rm(["gone", "--config", str(cfg), "--yes", "--json"])
    v = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert v["id"] == "gone" and v["config_entry_removed"] is True
    assert v["files_deleted"] == [str(lib / "gone.gguf")]
    assert v["bytes_freed"] > 0
    assert v["aliases_removed"] == ["fast"] and v["default_cleared"] is True


def test_rm_discovered_model_files_only(tmp_path, capsys):
    body = """
server:
  model_dirs:
    - <LIB>
discover:
  - {}
"""
    cfg, lib = _rm_setup(tmp_path, body, files=("loose.gguf",))
    manage.cmd_list(["--config", str(cfg), "--json"])
    data = json.loads(capsys.readouterr().out)
    (row,) = [r for r in data["models"] if r["source"] == "discovered"]
    before = cfg.read_text()
    rc = manage.cmd_rm([row["id"], "--config", str(cfg), "--yes"])
    out = capsys.readouterr().out
    assert rc == 0
    assert not (lib / "loose.gguf").exists()
    assert cfg.read_text() == before          # no config entry to touch
    assert "discovered model - no config entry" in out


def test_rm_missing_file_entry_only(tmp_path, capsys):
    import yaml
    cfg, lib = _rm_setup(tmp_path, files=("keep.gguf",))   # gone.gguf absent
    rc = manage.cmd_rm(["gone", "--config", str(cfg), "--yes"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "model file not found" in out
    assert "gone" not in yaml.safe_load(cfg.read_text())["models"]


def test_validate_junk_local_file_is_clean_verdict(tmp_path, capsys):
    # Vetting dubious files is validate's whole job: garbage bytes must be a
    # named refusal, not a GGUFReader traceback.
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"definitely not a gguf " * 8)
    rc = manage.cmd_validate([str(junk)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "junk.gguf" in err and err.startswith("error:")


def test_validate_directory_is_clean_verdict(tmp_path, capsys):
    rc = manage.cmd_validate([str(tmp_path)])
    assert rc == 2
    assert "directory" in capsys.readouterr().err


def test_shard_names_zero_total_rejected():
    from gmlx import remote
    with pytest.raises(remote.RemoteError, match="zero shard"):
        manage._shard_names("m-00001-of-00000.gguf")


def _http_416(content_range):
    import email.message
    import urllib.error

    hdr = email.message.Message()
    if content_range:
        hdr["Content-Range"] = content_range

    def opener(req, timeout=30):
        raise urllib.error.HTTPError(req.full_url, 416, "range", hdr, None)
    return opener


def test_url_download_416_stale_oversized_part_rejected(tmp_path, monkeypatch):
    from gmlx import remote
    dest = tmp_path / "m.gguf"
    (tmp_path / "m.gguf.part").write_bytes(b"12345")
    monkeypatch.setattr(remote, "http_open", _http_416("bytes */3"))
    with pytest.raises(remote.RemoteError, match="stale partial"):
        manage._url_download("http://x/m.gguf", str(dest))
    assert not dest.exists()                 # nothing promoted


def test_url_download_416_exact_size_promotes(tmp_path, monkeypatch):
    from gmlx import remote
    dest = tmp_path / "m.gguf"
    (tmp_path / "m.gguf.part").write_bytes(b"12345")
    monkeypatch.setattr(remote, "http_open", _http_416("bytes */5"))
    assert manage._url_download("http://x/m.gguf", str(dest)) == str(dest)
    assert dest.read_bytes() == b"12345"
    assert not (tmp_path / "m.gguf.part").exists()


def test_rm_drops_dangling_assistant_aliases(tmp_path, capsys):
    import yaml

    from gmlx.config import build_config

    body = _RM_CFG.replace("""  defaults:
    model: gone""", """  defaults:
    model: gone
  assistants:
    helper:
      model: gone
      memory: true
    survivor:
      model: keep""")
    cfg, lib = _rm_setup(tmp_path, body=body)
    rc = manage.cmd_rm(["gone", "--config", str(cfg), "--yes"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "helper" in err
    doc = yaml.safe_load(cfg.read_text())
    asst = doc["server"]["assistants"]
    assert set(asst) == {"survivor"}
    build_config(doc)  # config still validates after the removal
