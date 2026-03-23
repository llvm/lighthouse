from .dialect import register_and_load
from .dialect import TransformExtensionDialect

from .ops.wrap_in_benching_func import wrap_in_benching_func
from .ops.get_named_attribute import get_named_attribute
from .ops.param_cmp_eq import param_cmp_eq
from .ops.replace import replace

__all__ = [
    "TransformExtensionDialect",
    "get_named_attribute",
    "param_cmp_eq",
    "register_and_load",
    "replace",
    "wrap_in_benching_func",
]
