"""gmlx/container/localhost_domains.py: the localhost domains of Apple
container, read in the formats that ``container system dns create
--localhost`` writes in 1.4.1 and 1.5.0."""

from __future__ import annotations

from pathlib import Path

from gmlx.container import localhost_domains as ld


def fake_etc(root: Path, resolvers: dict[str, str], anchor: str | None) -> Path:
    (root / "resolver").mkdir(parents=True)
    for name, text in resolvers.items():
        (root / "resolver" / name).write_text(text)
    if anchor is not None:
        (root / "pf.anchors").mkdir()
        (root / "pf.anchors" / "com.apple.container").write_text(anchor)
    return root


DOMAIN = ("domain host.container.internal\nsearch host.container.internal\n"
          "nameserver 127.0.0.1\nport 1053\noptions localhost:203.0.113.113")
PLAIN = "domain test\nsearch test\nnameserver 127.0.0.1\nport 2053\n"
RDR = "rdr inet from any to 203.0.113.113 -> 127.0.0.1 # host.container.internal\n"


def test_a_domain_is_found_from_its_resolver_file_and_its_redirect(tmp_path):
    etc = fake_etc(tmp_path, {"containerization.host.container.internal": DOMAIN,
                              "containerization.test": PLAIN}, RDR)
    assert ld.localhost_domains(etc) == [("host.container.internal", "203.0.113.113")]
    assert ld.launch_warning(etc) == (
        "[launch] warning: the Apple container localhost domain host.container.internal "
        "(203.0.113.113) sends this container to the loopback address of this Mac, where "
        "it reaches every local service. If you do not need it, remove it with sudo "
        "container system dns delete host.container.internal.")
    assert ld.doctor_note(etc) == (
        "localhost domain host.container.internal (203.0.113.113) lets every container "
        "reach the loopback services of this Mac (sudo container system dns delete "
        "host.container.internal)")


def test_a_redirect_alone_is_found_and_named_by_its_address(tmp_path):
    etc = fake_etc(tmp_path, {}, "rdr inet from any to 203.0.113.1 -> 127.0.0.1\n"
                                 "rdr inet from any to 198.51.100.2 -> 127.0.0.1 # usable\n"
                                 "rdr inet from any to 198.51.100.3 -> 10.0.0.1 # other\n")
    assert ld.localhost_domains(etc) == [(None, "203.0.113.1"), ("usable", "198.51.100.2")]
    note = ld.doctor_note(etc)
    assert note is not None and note.startswith(
        "localhost domains 203.0.113.1, usable (198.51.100.2) let every container")
    assert note.endswith(f"or remove the redirect from {etc}/pf.anchors/com.apple.container)")


def test_no_domain_gives_no_warning(tmp_path):
    etc = fake_etc(tmp_path, {"containerization.test": PLAIN}, "")
    assert ld.localhost_domains(etc) == []
    assert ld.launch_warning(etc) is None and ld.doctor_note(etc) is None
    assert ld.localhost_domains(tmp_path / "missing") == []
