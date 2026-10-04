"""gmlx/container/pastes.py: the file paths found in a paste, and the files
placed in the private home for them."""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path

import pytest

from gmlx.container import confine, pastes

HOME = "/Users/u"


# Reading a paste

@pytest.mark.parametrize("text, paths, forms", [
    ("/Users/u/a.png", ["/Users/u/a.png"], ["plain"]),
    ("  /Users/u/a.png\t\n", ["/Users/u/a.png"], ["plain"]),
    ("/Users/u/my\\ shot.png", ["/Users/u/my shot.png"], ["plain"]),
    ("/Users/u/\\(1\\).pdf", ["/Users/u/(1).pdf"], ["plain"]),
    ("'/Users/u/my shot.png'", ["/Users/u/my shot.png"], ["single"]),
    ('"/Users/u/my shot.png"', ["/Users/u/my shot.png"], ["double"]),
    ('"/Users/u/a \\"b\\".txt"', ['/Users/u/a "b".txt'], ["double"]),
    ("file:///Users/u/my%20shot.png", ["/Users/u/my shot.png"], ["url"]),
    ("file://localhost/Users/u/a.txt", ["/Users/u/a.txt"], ["url"]),
    ("~/notes.txt", ["/Users/u/notes.txt"], ["plain"]),
    ("/Users/u/a.png /Users/u/b.pdf\n/Users/u/c", ["/Users/u/a.png", "/Users/u/b.pdf",
                                                   "/Users/u/c"], ["plain"] * 3),
])
def test_a_paste_of_paths_in_each_terminal_form(text, paths, forms):
    words = pastes.path_words(text, HOME)
    assert words is not None
    assert [w.path for w in words] == paths
    assert [w.form for w in words] == forms


@pytest.mark.parametrize("text", [
    "look at /Users/u/a.png",              # a path inside text stays text
    "/Users/u/a.png please",
    "relative.png",
    "'~/a.png'",                           # a quoted tilde is not the home
    '"/Users/u/a.png',                     # a quote that does not close
    "file://server/Users/u/a.png",
    "file:///Users/u/a.png?x=1",
    "",
    "   \n",
])
def test_a_paste_that_is_not_only_paths_is_left_alone(text):
    assert pastes.path_words(text, HOME) is None


@pytest.mark.parametrize("text", [
    "/Users/u/a\\ b.png", "'/Users/u/it'\\''s here.png'", '"/Users/u/a $b.png"',
    "file:///Users/u/a%20b.png", "/Users/u/a\\'b.png", "/Users/u/plain.png",
])
def test_a_new_path_keeps_the_form_of_the_old_one(text):
    word = pastes.path_words(text, HOME)[0]
    new = "/home/x y/it's $here.png"
    again = pastes.path_words(pastes.written(new, word), HOME)
    assert again is not None and [(w.path, w.form) for w in again] == [(new, word.form)]


@pytest.mark.parametrize("name, clean", [
    ("shot.png", "shot.png"), ("a\x1bb.txt", "a_b.txt"), ("a\u202eb.pdf", "a_b.pdf"),
    ("a\nb", "a_b"), ("", "file"), (".", "file"), ("..", "file"),
    ("Bild \u00e4.png", "Bild \u00e4.png"),
])
def test_the_name_in_the_container_has_no_control_characters(name, clean):
    assert pastes.guest_name(name) == clean


# Placing files

@pytest.fixture
def mac(tmp_path):
    """A Mac folder outside the shares, a share at its own path and a
    private home, with the log lines of a Pastes on them."""
    class Mac:
        pass
    m = Mac()
    m.files = tmp_path / "mac"
    m.share = tmp_path / "share"
    m.home = tmp_path / "home"
    for folder in (m.files, m.share, m.home):
        folder.mkdir()
    m.lines = []
    m.pastes = pastes.Pastes(m.home, [str(m.share), str(m.home)], m.lines.append,
                             mac_home=str(tmp_path))
    m.folder = m.home / ".gmlx" / "pastes"
    return m


def _entry(path: Path) -> Path:
    return Path(pastes.entry_key(os.stat(path)))


def _target(mac, path: Path) -> Path:
    return mac.folder / _entry(path) / path.name


def _no_clone(monkeypatch, error=OSError(errno.EXDEV, "Cross-device link")):
    def fail(src_fd, dir_fd, name):
        raise error
    monkeypatch.setattr(confine, "_fclonefileat", fail)


def test_a_file_on_the_same_volume_is_cloned(mac):
    """/tmp and the private home lie on one APFS volume, so the file is
    cloned, whatever the copy limit."""
    data = b"%PDF-1.7\n" + bytes(range(256)) * 64
    (mac.files / "report.pdf").write_bytes(data)
    mac.pastes.copy_max = 1
    out = mac.pastes.rewrite(f"{mac.files}/report.pdf".encode())
    target = _target(mac, mac.files / "report.pdf")
    assert out == str(target).encode()
    assert target.read_bytes() == data
    assert not target.is_symlink()
    assert mac.lines == [f"paste: cloned {mac.files}/report.pdf to {target} "
                         f"({len(data):,} bytes)"]


def test_a_clone_changes_apart_from_the_mac_file(mac):
    (mac.files / "a.txt").write_bytes(b"mac")
    mac.pastes.rewrite(f"{mac.files}/a.txt".encode())
    target = _target(mac, mac.files / "a.txt")
    target.write_bytes(b"guest")
    assert (mac.files / "a.txt").read_bytes() == b"mac"


@pytest.mark.parametrize("error", [OSError(errno.EXDEV, "Cross-device link"),
                                   OSError(errno.ENOTSUP, "Operation not supported"),
                                   OSError(errno.EPERM, "Operation not permitted"),
                                   RuntimeError("anything")])
def test_a_failed_clone_falls_back_to_a_copy(mac, monkeypatch, error):
    _no_clone(monkeypatch, error)
    (mac.files / "a.txt").write_bytes(b"hello")
    out = mac.pastes.rewrite(f"{mac.files}/a.txt".encode())
    target = _target(mac, mac.files / "a.txt")
    assert out == str(target).encode() and target.read_bytes() == b"hello"
    assert mac.lines == [f"paste: copied {mac.files}/a.txt to {target} (5 bytes)"]


def test_a_copy_stops_at_the_limit_of_the_setting(mac, monkeypatch):
    _no_clone(monkeypatch)
    (mac.files / "big.bin").write_bytes(b"x" * 2048)
    (mac.files / "fits.bin").write_bytes(b"x" * 1024)
    mac.pastes.copy_max = 1024
    body = f"{mac.files}/big.bin {mac.files}/fits.bin".encode()
    out = mac.pastes.rewrite(body).decode().split(" ")
    assert out[0] == f"{mac.files}/big.bin"
    assert out[1] == str(_target(mac, mac.files / "fits.bin"))
    assert mac.lines[0] == (f"paste: {mac.files}/big.bin stays as it is, because launch could "
                            "not clone it (Cross-device link) and it is larger than "
                            "launch.container.paste_copy_max, 1 KiB")


def test_a_file_that_grows_past_the_limit_during_the_copy_stays(mac, monkeypatch):
    _no_clone(monkeypatch)
    (mac.files / "a.bin").write_bytes(b"x" * 10)
    mac.pastes.copy_max = 16
    real = pastes.copy_fd

    def grow(src, dst, limit):
        (mac.files / "a.bin").write_bytes(b"x" * 64)
        return real(src, dst, limit)
    monkeypatch.setattr(pastes, "copy_fd", grow)
    body = f"{mac.files}/a.bin".encode()
    assert mac.pastes.rewrite(body) == body
    assert mac.lines[-1].endswith("it grew past launch.container.paste_copy_max, 16 bytes, "
                                  "during the copy")
    assert all(os.listdir(mac.folder / e) == [] for e in os.listdir(mac.folder))


def test_any_kind_of_file_is_placed_by_name(mac):
    for name, data in (("notes.txt", b"text"), ("data.csv", b"a,b"), ("x", b"\0\1")):
        (mac.files / name).write_bytes(data)
        out = mac.pastes.rewrite(f"{mac.files}/{name}".encode())
        assert Path(out.decode()).name == name
        assert Path(out.decode()).read_bytes() == data


def test_quotes_escapes_and_urls_come_back_in_their_form(tmp_path):
    files = tmp_path / "my files"
    files.mkdir()
    (files / "a b.png").write_bytes(b"png")
    home = tmp_path / "home x"
    home.mkdir()
    p = pastes.Pastes(home, [], lambda line: None)
    target = home / ".gmlx" / "pastes" / _entry(files / "a b.png") / "a b.png"
    escaped = str(target).replace(" ", "\\ ")
    cases = {
        str(files / "a b.png").replace(" ", "\\ "): escaped,
        f"'{files}/a b.png'": f"'{target}'",
        f'"{files}/a b.png"': f'"{target}"',
        "file://" + str(files / "a b.png").replace(" ", "%20"):
            "file://" + str(target).replace(" ", "%20"),
    }
    for before, after in cases.items():
        assert p.rewrite(before.encode()).decode() == after
    two = f"{escaped}\n'{files}/a b.png'"
    assert p.rewrite(two.replace(escaped, str(files / "a b.png").replace(" ", "\\ "))
                     .encode()).decode() == f"{escaped}\n'{target}'"


def test_an_iterm2_drop_of_a_macos_screenshot(mac):
    """iTerm2 escapes the spaces of a dragged path and ends it with a space.
    A macOS screenshot name holds a narrow no-break space before PM, which
    stays part of the name."""
    name = "Screenshot 2026-08-02 at 8.25.43\u202fPM.png"
    (mac.files / name).write_bytes(b"png")
    body = (str(mac.files / name).replace(" ", "\\ ") + " ").encode()
    out = mac.pastes.rewrite(body).decode()
    target = _target(mac, mac.files / name)
    assert out == str(target).replace(" ", "\\ ") + " "
    assert target.name == name and target.read_bytes() == b"png"


def test_a_tilde_path_is_read_in_the_mac_home(mac, tmp_path):
    (tmp_path / "note.txt").write_bytes(b"n")
    out = mac.pastes.rewrite(b"~/note.txt")
    assert out == str(_target(mac, tmp_path / "note.txt")).encode()


def test_paths_inside_text_stay_text(mac):
    (mac.files / "a.png").write_bytes(b"png")
    for body in (f"look at {mac.files}/a.png", f"{mac.files}/a.png is broken",
                 f"error: {mac.files}/a.png: no such file"):
        assert mac.pastes.rewrite(body.encode()) == body.encode()
    assert not mac.folder.exists() and mac.lines == []


def test_a_path_in_a_share_at_its_own_path_stays(mac):
    (mac.share / "a.png").write_bytes(b"png")
    body = f"{mac.share}/a.png".encode()
    assert mac.pastes.rewrite(body) == body
    assert not mac.folder.exists()
    assert mac.lines == [f"paste: {mac.share}/a.png is in a folder that the container sees "
                         "at the same path, so it stays as it is"]


def test_a_paste_that_is_not_utf8_stays(mac):
    body = b"\xff/Users/u/a.png"
    assert mac.pastes.rewrite(body) == body


def test_a_second_paste_of_a_file_uses_the_same_entry(mac):
    (mac.files / "a.png").write_bytes(b"png")
    first = mac.pastes.rewrite(f"{mac.files}/a.png".encode())
    second = mac.pastes.rewrite(f"{mac.files}/a.png".encode())
    assert first == second and len(os.listdir(mac.folder)) == 1


def test_a_changed_file_gets_a_new_entry(mac):
    (mac.files / "a.txt").write_bytes(b"one")
    first = mac.pastes.rewrite(f"{mac.files}/a.txt".encode())
    os.utime(mac.files / "a.txt", ns=(1, 1))
    (mac.files / "a.txt").write_bytes(b"three")
    second = mac.pastes.rewrite(f"{mac.files}/a.txt".encode())
    assert first != second
    assert Path(first.decode()).read_bytes() == b"one"
    assert Path(second.decode()).read_bytes() == b"three"


def test_the_folder_keeps_the_newest_entries(mac, monkeypatch):
    monkeypatch.setattr(pastes, "KEEP", 3)
    mac.folder.mkdir(parents=True)
    old = []
    for i in range(4):
        name = f"{i:016x}"
        (mac.folder / name).mkdir()
        (mac.folder / name / "f").write_bytes(b"x")
        os.utime(mac.folder / name, ns=((i + 1) * 10**9, (i + 1) * 10**9))
        old.append(name)
    (mac.folder / "notes.txt").write_text("not an entry")
    (mac.files / "a.png").write_bytes(b"png")
    mac.pastes.rewrite(f"{mac.files}/a.png".encode())
    assert sorted(os.listdir(mac.folder)) == sorted(
        [str(_entry(mac.files / "a.png")), old[3], old[2], "notes.txt"])


def test_fifty_entries_are_kept_by_default():
    assert pastes.KEEP == 50


# What stays as it is

def test_a_link_a_folder_and_a_device_stay(mac, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"s")
    (mac.files / "link.txt").symlink_to(secret)
    (mac.files / "dir").mkdir()
    for path, why in ((f"{mac.files}/link.txt", "it is a symbolic link"),
                      (f"{mac.files}/dir", "it is a folder, and only files are placed"),
                      ("/dev/null", "it is not a regular file"),
                      (f"{mac.files}/missing.txt", "no such file")):
        assert mac.pastes.rewrite(path.encode()) == path.encode()
        assert mac.lines[-1] == f"paste: {path} stays as it is, because {why}"
    assert not mac.folder.exists()


def test_a_link_on_the_way_to_the_file_is_read_at_its_real_path(mac, tmp_path):
    (mac.files / "a.txt").write_bytes(b"a")
    (tmp_path / "via").symlink_to(mac.files)
    out = mac.pastes.rewrite(f"{tmp_path}/via/a.txt".encode())
    assert Path(out.decode()).read_bytes() == b"a"


def test_a_file_in_a_credentials_folder_stays(mac, monkeypatch, tmp_path):
    from gmlx.container import settings
    machome = tmp_path / "machome"
    (machome / ".ssh").mkdir(parents=True)
    (machome / ".ssh" / "id_ed25519").write_bytes(b"key")
    monkeypatch.setattr(settings, "_host_home", lambda: str(machome))
    body = f"{machome}/.ssh/id_ed25519".encode()
    assert mac.pastes.rewrite(body) == body
    assert mac.lines == [f"paste: {machome}/.ssh/id_ed25519 stays as it is, because "
                         "~/.ssh/id_ed25519 lies in ~/.ssh, which holds credentials, which "
                         "launch does not share"]
    assert not mac.folder.exists()


def test_a_path_through_a_private_home_stays(mac):
    """A file in a private home is the guest's, so launch never reads it
    as a Mac file."""
    from gmlx.container import settings
    home = settings.private_home("pi", "default")
    (home / "a.png").write_bytes(b"png")
    body = f"{home}/a.png".encode()
    other = pastes.Pastes(mac.home, [], mac.lines.append)
    assert other.rewrite(body) == body
    assert "stays as it is, because" in mac.lines[-1] and "private home" in mac.lines[-1]


def test_a_file_that_macos_does_not_let_launch_read_stays(mac, monkeypatch):
    (mac.files / "a.txt").write_bytes(b"a")

    def denied(root, parts):
        raise PermissionError(errno.EPERM, "Operation not permitted")
    monkeypatch.setattr(pastes, "open_file_below", denied)
    body = f"{mac.files}/a.txt".encode()
    assert mac.pastes.rewrite(body) == body
    assert mac.lines == [f"paste: {mac.files}/a.txt stays as it is, because macOS did not "
                         "let the terminal app read it. Allow the access in System Settings, "
                         "Privacy and Security, then paste again"]


@pytest.mark.parametrize("where", ["pastes", "gmlx"])
def test_a_planted_link_in_the_home_is_not_followed(mac, tmp_path, where):
    outside = tmp_path / "outside"
    outside.mkdir()
    if where == "gmlx":
        (mac.home / ".gmlx").symlink_to(outside)
    else:
        (mac.home / ".gmlx").mkdir()
        (mac.home / ".gmlx" / "pastes").symlink_to(outside)
    (mac.files / "a.png").write_bytes(b"png")
    body = f"{mac.files}/a.png".encode()
    assert mac.pastes.rewrite(body) == body
    assert os.listdir(outside) == []
    assert mac.lines[-1].startswith(f"paste: {mac.files}/a.png stays as it is, because launch "
                                    "could not clone it (")
    assert "symbolic link" in mac.lines[-1] and "the copy failed" in mac.lines[-1]


def test_a_planted_link_at_the_file_name_is_not_followed(mac, tmp_path):
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep")
    (mac.files / "a.png").write_bytes(b"png")
    entry = mac.folder / _entry(mac.files / "a.png")
    entry.mkdir(parents=True)
    (entry / "a.png").symlink_to(victim)
    body = f"{mac.files}/a.png".encode()
    assert mac.pastes.rewrite(body) == body
    assert victim.read_bytes() == b"keep"
    assert (entry / "a.png").is_symlink()


def test_a_paste_with_many_files_places_the_first_ones(mac, monkeypatch):
    monkeypatch.setattr(pastes, "PATHS_MAX", 2)
    names = [f"{i}.txt" for i in range(3)]
    for i, name in enumerate(names):
        (mac.files / name).write_bytes(bytes([i]))
    body = " ".join(f"{mac.files}/{n}" for n in names).encode()
    out = mac.pastes.rewrite(body).decode().split(" ")
    assert out[2] == f"{mac.files}/2.txt" and all(".gmlx" in p for p in out[:2])
    assert mac.lines[0] == ("paste: the paste names more than 2 files, so the ones after that "
                            "stay as they are")


def test_the_clone_and_the_copy_go_through_the_confined_helpers(mac, monkeypatch):
    """The private home is walked without following a link, in the thread
    that places the file."""
    seen = []
    real_clone, real_write = confine.clone, confine.write_stream

    def clone(fd, path, mode=None):
        seen.append(("clone", Path(path), confine.active()))
        raise OSError(errno.EXDEV, "Cross-device link")

    def write_stream(path, fill, mode=None):
        seen.append(("copy", Path(path), confine.active()))
        real_write(path, fill, mode)
    monkeypatch.setattr(confine, "clone", clone)
    monkeypatch.setattr(confine, "write_stream", write_stream)
    (mac.files / "a.png").write_bytes(b"png")
    mac.pastes.rewrite(f"{mac.files}/a.png".encode())
    target = _target(mac, mac.files / "a.png")
    assert seen == [("clone", target, True), ("copy", target, True)]
    assert not confine.active()
    assert real_clone is not clone


# No error leaves a paste

def _fail(*args, **kwargs):
    raise RuntimeError("boom")


@pytest.mark.parametrize("step", ["path_words", "open_file", "host_path", "clone+copy",
                                  "mkdirs", "log"])
def test_an_error_at_any_step_leaves_the_paste_as_it_came(mac, monkeypatch, step):
    (mac.files / "a.txt").write_bytes(b"a")
    if step == "path_words":
        monkeypatch.setattr(pastes, "path_words", _fail)
    elif step == "open_file":
        monkeypatch.setattr(pastes, "open_file", _fail)
    elif step == "host_path":
        monkeypatch.setattr(confine, "host_path", _fail)
    elif step == "clone+copy":
        monkeypatch.setattr(confine, "clone", _fail)
        monkeypatch.setattr(confine, "write_stream", _fail)
    elif step == "mkdirs":
        monkeypatch.setattr(confine, "_open_dir", _fail)
    body = f"{mac.files}/a.txt".encode()
    if step == "log":
        mac.pastes.log = _fail
        monkeypatch.setattr(pastes, "open_file", _fail)
        assert mac.pastes.rewrite(body) == body
        return
    assert mac.pastes.rewrite(body) == body
    assert len(mac.lines) == 1 and "boom" in mac.lines[0]
    assert mac.lines[0].startswith("paste: ")


def test_a_failed_prune_still_places_the_file(mac, monkeypatch):
    monkeypatch.setattr(confine, "listdir", _fail)
    (mac.files / "a.txt").write_bytes(b"a")
    out = mac.pastes.rewrite(f"{mac.files}/a.txt".encode())
    assert Path(out.decode()).read_bytes() == b"a"
    assert "could not remove old entries" in mac.lines[0]


def test_entry_keys_follow_the_file_version(tmp_path):
    f = tmp_path / "a"
    f.write_bytes(b"x")
    st = os.stat(f)
    ident = f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}".encode()
    assert pastes.entry_key(st) == hashlib.sha256(ident).hexdigest()[:16]


@pytest.mark.parametrize("n, text", [(1 << 30, "1 GiB"), (512 << 20, "512 MiB"),
                                     (1024, "1 KiB"), (1000, "1,000 bytes")])
def test_size_text(n, text):
    assert pastes.size_text(n) == text
