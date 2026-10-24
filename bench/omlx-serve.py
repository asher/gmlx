#!/usr/bin/env python3
"""Run `omlx` with every request forced to its max_tokens.

serve-bench.py measures decode over a fixed token window on every engine.
omlx has a forced-length mode (SamplingParams.ignore_eos) but no request
field for it, so this wrapper sets it on every request. No omlx source is
changed. It needs omlx 0.7.2 or later. Run it with the python of the venv
that has omlx installed:

  ~/.venvs/omlx/bin/python omlx-serve.py serve --model-dir DIR [omlx flags]
"""
import dataclasses
import sys

from omlx import request as _request
from omlx.cli import main

if "ignore_eos" not in {f.name for f in dataclasses.fields(_request.SamplingParams)}:
    sys.exit("this omlx has no forced-length mode (SamplingParams.ignore_eos); "
             "install omlx 0.7.2 or later")

_post_init = _request.SamplingParams.__post_init__


def _forced_length(self):
    _post_init(self)
    self.ignore_eos = True


_request.SamplingParams.__post_init__ = _forced_length

if __name__ == "__main__":
    sys.argv[0] = "omlx"
    sys.exit(main())
