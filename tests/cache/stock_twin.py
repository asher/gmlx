"""The stock class an owned class replaced, for controls that show a test
fails on the stock body. The binding keeps each replaced stock object."""

from gmlx.cache import bind


def stock_twin(owned, origin="mlx_vlm.models.cache"):
    """The ``origin`` class or function whose name now holds ``owned``."""
    for old, new in bind.replaced():
        if new is owned and getattr(old, "__module__", None) == origin:
            return old
    raise LookupError(f"no {origin} twin of {owned!r}")


def stock_method(owned_cls, name, origin="mlx_vlm.models.cache"):
    """The stock body of ``owned_cls.name``, as the stock class defines it."""
    return stock_twin(owned_cls, origin).__dict__[name]
