# gmlx: gmlx.cache.kv_cache owns all of this module, so the path is an alias
# of it. Every import of the path, relative ones included, gets the owned
# module, and a patch or a globals() lookup lands on one dict. The import
# system then sets the attribute on the parent package.
import sys

import gmlx.cache.kv_cache as _owned

sys.modules[__name__] = _owned
