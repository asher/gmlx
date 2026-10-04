"""The localhost domains of Apple container.

``sudo container system dns create <domain> --localhost <ip>`` gives every
container a name for the Mac's loopback address. It writes
``/etc/resolver/containerization.<domain>`` with ``options localhost:<ip>``,
and a packet filter rule ``rdr inet from any to <ip> -> 127.0.0.1`` to
``/etc/pf.anchors/com.apple.container``. The rule names no port, so a
container reaches every loopback service on the Mac. The gmlx server refuses
a peer that is not loopback, but other services may not, so launch and
doctor warn. Both files are readable without root.
"""

from __future__ import annotations

import re
from pathlib import Path

ETC = Path("/etc")
_RESOLVER_PREFIX = "containerization."
_ANCHOR = "pf.anchors/com.apple.container"
_OPTION = re.compile(r"^\s*options\s+localhost:(\S+)", re.MULTILINE)
_RDR = re.compile(r"^\s*rdr\s.*\bto\s+(\S+)\s+->\s+127\.0\.0\.1\b[^#]*(?:#\s*(\S+))?")


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def localhost_domains(etc: Path | None = None) -> list[tuple[str | None, str]]:
    """(domain, address) for each localhost domain, sorted. The domain is
    None for a redirect rule that names no domain."""
    etc = etc or ETC
    found: dict[str, str | None] = {}
    try:
        names = sorted(p.name for p in (etc / "resolver").iterdir()
                       if p.name.startswith(_RESOLVER_PREFIX))
    except OSError:
        names = []
    for name in names:
        match = _OPTION.search(_read(etc / "resolver" / name))
        if match:
            found[match.group(1)] = name[len(_RESOLVER_PREFIX):]
    for line in _read(etc / _ANCHOR).splitlines():
        match = _RDR.match(line)
        if match and found.get(match.group(1)) is None:
            found[match.group(1)] = match.group(2)
    return sorted(((d, ip) for ip, d in found.items()), key=lambda e: (e[0] or "", e[1]))


def _named(domains: list[tuple[str | None, str]]) -> str:
    return ", ".join(f"{d} ({ip})" if d else ip for d, ip in domains)


def _remove(domains: list[tuple[str | None, str]], etc: Path) -> str:
    if any(d is None for d, _ in domains):
        return ("sudo container system dns delete <domain>, or remove the redirect "
                f"from {etc / _ANCHOR}")
    if len(domains) == 1:
        return f"sudo container system dns delete {domains[0][0]}"
    return "sudo container system dns delete <domain> for each one"


def launch_warning(etc: Path | None = None) -> str | None:
    """The launch warning for the localhost domains of this Mac, or None."""
    etc = etc or ETC
    domains = localhost_domains(etc)
    if not domains:
        return None
    one = len(domains) == 1
    return (f"[launch] warning: the Apple container localhost "
            f"{'domain' if one else 'domains'} {_named(domains)} "
            f"{'sends' if one else 'send'} this container to the loopback address of "
            "this Mac, where it reaches every local service. If you do not need "
            f"{'it' if one else 'them'}, remove {'it' if one else 'them'} with "
            f"{_remove(domains, etc)}.")


def doctor_note(etc: Path | None = None) -> str | None:
    """The phrase for doctor's container row, or None."""
    etc = etc or ETC
    domains = localhost_domains(etc)
    if not domains:
        return None
    one = len(domains) == 1
    return (f"localhost {'domain' if one else 'domains'} {_named(domains)} "
            f"{'lets' if one else 'let'} every container reach the loopback services "
            f"of this Mac ({_remove(domains, etc)})")
