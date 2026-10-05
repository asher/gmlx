# gmlx: gmlx.models.gemma4.language owns the Gemma 4 text model, so the path
# is an alias of it. The containers of this package and of gemma4_unified
# build the owned language model, and the assistant and dspark drafters take
# their layers from it. The exact speculative verifier in this package is no
# longer imported; the owned model refuses a config that asks for it.
import sys

import gmlx.models.gemma4.language as _owned

sys.modules[__name__] = _owned
