"""gmlx/safe_path.py: the walk below a folder follows no symbolic link and
never leaves the folder."""

from __future__ import annotations

import os

import pytest

from gmlx.safe_path import (
    LeavesRoot,
    NotFollowed,
    NotRegular,
    TooLarge,
    open_dir_below,
    open_file_below,
    parts_below,
    path_inside,
    read_fd,
    read_json_object,
    read_regular,
    same_name,
)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "root" / "a").mkdir(parents=True)
    (tmp_path / "root" / "a" / "f.txt").write_text("inside")
    (tmp_path / "outside.txt").write_text("outside")
    return tmp_path / "root"


@pytest.mark.parametrize("parts", [[".."], ["a", ".."], ["."], [""], ["a/.."], ["a", "../.."]])
def test_a_component_that_leaves_the_folder_is_refused(root, parts):
    with pytest.raises(LeavesRoot):
        os.close(open_dir_below(root, parts))


@pytest.mark.parametrize("parts", [["..", "outside.txt"], ["a", "..", "..", "outside.txt"],
                                   ["a", "."], ["a", ".."]])
def test_a_file_path_that_leaves_the_folder_is_refused(root, parts):
    with pytest.raises(LeavesRoot):
        os.close(open_file_below(root, parts))


def test_a_link_on_the_way_is_not_followed(root, tmp_path):
    os.symlink(tmp_path, root / "up")
    os.symlink(tmp_path / "outside.txt", root / "a" / "out.txt")
    with pytest.raises(NotFollowed):
        os.close(open_dir_below(root, ["up"]))
    with pytest.raises(NotFollowed):
        os.close(open_file_below(root, ["up", "outside.txt"]))
    with pytest.raises(NotFollowed):
        os.close(open_file_below(root, ["a", "out.txt"]))


def test_a_file_below_the_folder_opens(root):
    fd = open_file_below(root, ["a", "f.txt"])
    with os.fdopen(fd) as f:
        assert f.read() == "inside"


def test_missing_folders_are_created_only_when_asked(root):
    with pytest.raises(FileNotFoundError):
        open_dir_below(root, ["new", "deeper"])
    os.close(open_dir_below(root, ["new", "deeper"], create=True, mode=0o700))
    assert (root / "new" / "deeper").is_dir()
    assert os.stat(root / "new").st_mode & 0o777 == 0o700


def test_parts_below_matches_whole_components_only():
    assert parts_below("/m/media/a/b.png", "/m/media") == ["a", "b.png"]
    assert parts_below("/m/media-other/b.png", "/m/media") is None
    assert parts_below("/m/media/../x", "/m/media") == ["..", "x"]
    assert path_inside("/m/media", "/m/media") and not path_inside("/m", "/m/media")


def test_path_inside_asks_the_volume_only_when_the_folded_names_match(monkeypatch):
    """A check of one path against a long list of folders, such as the
    share history, asks the volume about a folder only when the path can
    lie in it. The answers stay those of a check by exact names on a
    volume that tells case apart."""
    from gmlx import safe_path

    asked = []

    def case_sensitive(folder):
        asked.append(folder)
        return False

    monkeypatch.setattr(safe_path, "_case_insensitive", case_sensitive)
    folders = [f"/m/f{i}" for i in range(300)]
    assert not any(path_inside("/m/other/x", f) for f in folders)
    assert asked == []
    assert path_inside("/m/f7/x", "/m/f7") and asked == ["/m/f7"]
    assert not path_inside("/m/F7/x", "/m/f7")
    assert not path_inside("/m/f70", "/m/f7")
    monkeypatch.setattr(safe_path, "_case_insensitive", lambda folder: True)
    assert path_inside("/m/F7/x", "/m/f7")


def test_same_name_asks_the_volume_of_the_folder_that_holds_the_names(monkeypatch):
    """Two names in one folder name one entry by the case rule of the
    folder. An entry that is a link to another volume has the rule of the
    folder too."""
    from gmlx import safe_path

    asked = []

    def volume(folder):
        asked.append(folder)
        return folder == "/m/ci"

    monkeypatch.setattr(safe_path, "_case_insensitive", volume)
    assert same_name("a.json", "a.json", "/m/cs") and asked == []
    assert not same_name("b.json", "a.json", "/m/ci") and asked == []
    assert same_name("A.json", "a.json", "/m/ci") and asked == ["/m/ci"]
    assert not same_name("A.json", "a.json", "/m/cs")


def test_read_regular_reads_a_file_up_to_its_limit(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"x" * 10)
    assert read_regular(f, 10) == b"x" * 10
    with pytest.raises(TooLarge, match="larger than 9 bytes"):
        read_regular(f, 9)


def test_read_regular_follows_a_link_only_when_asked(tmp_path):
    (tmp_path / "f").write_text("data")
    (tmp_path / "link").symlink_to(tmp_path / "f")
    with pytest.raises(OSError):
        read_regular(tmp_path / "link", 100)
    assert read_regular(tmp_path / "link", 100, follow=True) == b"data"


def test_read_regular_refuses_a_named_pipe_without_waiting(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(NotRegular, match="not a regular file"):
        read_regular(fifo, 100)


def test_read_fd_refuses_a_file_that_grew_after_its_status(tmp_path):
    small, big = tmp_path / "small", tmp_path / "big"
    small.write_bytes(b"x")
    big.write_bytes(b"x" * 100)
    fd = os.open(big, os.O_RDONLY)
    try:
        with pytest.raises(TooLarge):
            read_fd(fd, 10, os.stat(small))
    finally:
        os.close(fd)


def test_read_json_object_gives_none_for_anything_but_an_object(tmp_path):
    f = tmp_path / "f.json"
    f.write_text('{"a": 1}')
    assert read_json_object(f, 100) == {"a": 1}
    f.write_text("[1]")
    assert read_json_object(f, 100) is None
    f.write_text("{")
    assert read_json_object(f, 100) is None
    assert read_json_object(tmp_path / "missing.json", 100) is None
