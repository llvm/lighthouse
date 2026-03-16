from mlir import ir
from mlir.dialects import transform


def linalg_morph_ops(
    target,
    named_to_generic: bool = False,
    named_to_category: bool = False,
    category_to_generic: bool = False,
    generic_to_category: bool = False,
    generic_to_named: bool = False,
) -> ir.Value:
    """
    Apply linalg morphism to all target's nested ops.

    Args:
        target: Handle to target
        named_to_generic: Named to generic ops
        named_to_category: Named to category ops
        category_to_generic: Category to generic ops
        generic_to_category: Generic to category ops
        generic_to_named: Generic to named ops
    Returns:
        New handle to target
    """
    return transform.apply_registered_pass(
        transform.any_op_t(),
        target,
        "linalg-morph-ops",
        options={
            "named-to-category": named_to_category,
            "category-to-generic": category_to_generic,
            "named-to-generic": named_to_generic,
            "generic-to-named": generic_to_named,
            "generic-to-category": generic_to_category,
        },
    )
