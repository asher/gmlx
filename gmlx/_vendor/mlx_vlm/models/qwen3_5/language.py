# gmlx: gmlx.models.qwen35.language owns the qwen3.5 text model, so the path
# is an alias of it. The container in this package and every borrower of the
# text model build the owned classes. The exact speculative verifier in this
# package loses the helpers it read from the module; gmlx never enables it.
import sys

import gmlx.models.qwen35.language as _owned

sys.modules[__name__] = _owned
