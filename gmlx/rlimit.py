"""The open-file limit of a long-running gmlx process.

macOS starts a process from Terminal with a soft limit of 256 open files.
The model server and the ``gmlx launch`` supervisor each hold one descriptor
per client connection, so both raise the soft limit when they start.
"""

from __future__ import annotations

import resource

# The soft limit a process asks for, or the hard limit when that is lower.
NOFILE_TARGET = 10240


def raise_nofile_limit(target: int = NOFILE_TARGET) -> None:
    """Raise the soft limit on open files toward ``target``, never past the
    hard limit. A failure leaves the limit as it was."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ValueError, OSError):
        pass
