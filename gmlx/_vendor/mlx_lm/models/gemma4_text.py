# gmlx: gmlx.models.gemma4.text owns the Gemma 4 text model on the mlx-lm
# contract, so the path is an alias of it. It exports Model and ModelArgs,
# which the gemma4 container beside this file builds.
import sys

import gmlx.models.gemma4.text as _owned

sys.modules[__name__] = _owned
