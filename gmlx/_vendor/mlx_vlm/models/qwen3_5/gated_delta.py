# gmlx: gmlx.models.qwen35.gated_delta owns the gated-delta ops and kernels,
# so the path is an alias of it. The owned file keeps every name of this
# one, not every argument: its gated_delta_update takes no state_steps,
# cache or cache_index, and its gated_delta_update_with_states and
# _gated_delta_with_states_ops take no state_steps. The one caller in the
# copy that passes any, qwen3_5/speculative_verifier.py, runs for no model
# gmlx builds.
import sys

import gmlx.models.qwen35.gated_delta as _owned

sys.modules[__name__] = _owned
