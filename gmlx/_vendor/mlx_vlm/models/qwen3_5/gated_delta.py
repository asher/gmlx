# gmlx: gmlx.models.qwen35.gated_delta owns the gated-delta ops and kernels,
# so the path is an alias of it. The owned file keeps every name of this one.
import sys

import gmlx.models.qwen35.gated_delta as _owned

sys.modules[__name__] = _owned
