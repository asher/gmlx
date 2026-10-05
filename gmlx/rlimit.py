"""The open-file limit of a long-running gmlx process.

macOS starts a process from Terminal with a soft limit of 256 open files.
The model server and the ``gmlx launch`` supervisor each hold one descriptor
per client connection, so both raise the soft limit when they start.
"""

from __future__ import annotations

import resource

# The soft limit a process asks for, or the hard limit when that is lower.
NOFILE_TARGET = 10240
# Below this limit, the connections launch relays can hold (up to 256 per
# listener) can use a large part of the server's descriptors.
NOFILE_LOW = 3000


def raise_nofile_limit(target: int = NOFILE_TARGET) -> int | None:
    """Raise the soft limit on open files toward ``target``, never past the
    hard limit. A failure leaves the limit as it was. Returns the soft limit
    that is in effect, or None when it cannot be read."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ValueError, OSError):
        pass
    try:
        return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except (ValueError, OSError):
        return None


def low_limit_warning(limit: int | None, who: str) -> str | None:
    """One line when ``limit`` stays below :data:`NOFILE_LOW`, else None."""
    if limit is None or limit == resource.RLIM_INFINITY or limit >= NOFILE_LOW:
        return None
    return (f"{who} can open only {limit} files at a time, so many idle client "
            "connections can use them up. Raise the limit with `ulimit -n` or "
            "`launchctl limit maxfiles`.")
