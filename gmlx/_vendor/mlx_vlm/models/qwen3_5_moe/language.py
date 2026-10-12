# gmlx: gmlx.models.qwen35.moe_language owns the qwen3.5 MoE text model, so
# the path is an alias of it. The container in this package builds the owned
# classes.
import sys

import gmlx.models.qwen35.moe_language as _owned

sys.modules[__name__] = _owned
