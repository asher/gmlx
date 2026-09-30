# vendor_upstream: skip
"""Hide any installed stock mlx-vlm and mlx-lm from the test session.

gmlx carries both packages under gmlx._vendor and never imports the stock
ones. tests/conftest.py installs this finder before anything imports gmlx,
so a stray import of ``mlx_vlm`` or ``mlx_lm`` raises ImportError here as
it does in an environment without them, even in a venv that has them. The
names and the importing lines are listed at the end of the run.
"""

from __future__ import annotations

import sys
import traceback

BLOCKED = ("mlx_vlm", "mlx_lm")
ATTEMPTS: list[tuple[str, str]] = []


class StockBlocker:
    """A meta path finder that refuses the stock top-level packages."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] not in BLOCKED:
            return None
        where = "?"
        for frame in reversed(traceback.extract_stack()[:-1]):
            if "importlib" not in frame.filename:
                where = f"{frame.filename}:{frame.lineno}"
                break
        ATTEMPTS.append((fullname, where))
        raise ImportError(
            f"stock {fullname} is blocked in the tests: gmlx imports its "
            f"copy, gmlx._vendor.{fullname}", name=fullname)


def install() -> None:
    loaded = sorted(m for m in sys.modules if m.partition(".")[0] in BLOCKED)
    if loaded:
        raise RuntimeError(f"stock packages imported before the blocker: "
                           f"{loaded}")
    if not any(isinstance(f, StockBlocker) for f in sys.meta_path):
        sys.meta_path.insert(0, StockBlocker())


def summary() -> list[str]:
    """One line per blocked name, with the first line that asked for it."""
    first: dict[str, str] = {}
    for name, where in ATTEMPTS:
        first.setdefault(name, where)
    return [f"{name}  <- {where}" for name, where in sorted(first.items())]
