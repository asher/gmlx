# gmlx: gmlx.models.gemma4.unified owns the Gemma 4 unified container and its
# vision embedder, so the path is an alias of it.
import sys

import gmlx.models.gemma4.unified as _owned

sys.modules[__name__] = _owned
