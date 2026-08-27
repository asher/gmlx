"""``python -m gmlx`` -> the umbrella CLI (identical to the ``gmlx`` console
script). A lazy import keeps ``-m gmlx <verb>`` light; this is also the form the
background / launchd re-exec uses, so it must route to the umbrella (note that
``-m gmlx.commands.cli`` runs ``run``'s ``main`` instead)."""
import sys


def _main() -> int:
    from gmlx.commands.cli import umbrella_main
    return umbrella_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(_main())
