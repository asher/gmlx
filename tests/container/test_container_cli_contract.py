"""The fake ``container`` accepts only the flags that the real CLI lists in
its help, as ``container_cli.json`` records them."""

from __future__ import annotations

import shutil
import subprocess

import container_cli_spec as spec
import pytest


@pytest.fixture(scope="module")
def recorded():
    return spec.load()


def test_a_flag_that_the_help_does_not_list_is_refused(recorded):
    assert spec.problem(["run", "--rm", "--no-such-flag", "img"], recorded) == (
        "container run: unknown option '--no-such-flag'")
    assert spec.problem(["system", "start", "--kernel-install"], recorded) == (
        "container system start: unknown option '--kernel-install'")
    assert spec.problem(["volume", "frobnicate"], recorded) == (
        "container volume frobnicate: a command that gmlx has no record of")


def test_a_flag_with_a_value_takes_the_next_word_or_an_equals_sign(recorded):
    assert spec.problem(["run", "-e", "--rm", "img"], recorded) is None
    assert spec.problem(["run", "--env=A=1", "--rm", "img"], recorded) is None
    assert spec.problem(["volume", "create", "--label"], recorded) == (
        "container volume create: --label takes a value")


def test_the_words_after_the_image_go_to_the_process(recorded):
    assert spec.problem(["run", "--rm", "img", "sh", "-c", "--anything"], recorded) is None
    assert spec.problem(["exec", "-i", "name", "gmlx-entry", "--hangup", "x"],
                        recorded) is None
    assert spec.problem(["image", "delete", "img", "--bogus"], recorded) == (
        "container image delete: unknown option '--bogus'")


def test_the_parser_reads_every_form_of_an_option_line():
    text = ("  -w, --workdir, --cwd <dir>\n"
            "  --rm, --remove          Remove the container after it stops\n"
            "  --enable-kernel-install/--disable-kernel-install\n"
            "  -n <n>                  Number of lines\n"
            "  <image>                 Image name\n")
    assert spec.parse_help(text) == {
        "-w": True, "--workdir": True, "--cwd": True, "--rm": False, "--remove": False,
        "--enable-kernel-install": False, "--disable-kernel-install": False, "-n": True}


def test_the_fake_refuses_a_flag_and_the_fixture_records_it(fake_container):
    proc = subprocess.run(["container", "volume", "list", "--no-such-flag"],
                          capture_output=True, text=True)
    assert proc.returncode == 64
    assert "unknown option '--no-such-flag'" in proc.stderr
    state = fake_container.load()
    assert state.pop("refused") == ["container volume list: unknown option '--no-such-flag'"]
    # The fixture fails a test that leaves such a record.
    fake_container.save(state)


_INSTALLED = shutil.which("container", path="/opt/homebrew/bin:/usr/local/bin")


@pytest.mark.skipif(_INSTALLED is None, reason="Apple's container is not installed")
def test_the_recorded_flags_match_the_installed_container(recorded):
    """On a Mac with Apple's container, the record matches its help. After
    an upgrade, run ``python tests/container/container_cli_spec.py
    --write`` and run the container tests again."""
    assert _INSTALLED is not None
    assert spec.record(_INSTALLED) == recorded
