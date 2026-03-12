from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured


def linalg_morph_ops(
    target: ir.Operation | ir.Value,
    named_to_generic: bool = False,
    named_to_category: bool = False,
    category_to_generic: bool = False,
    generic_to_category: bool = False,
    generic_to_named: bool = False,
):
    func = structured.MatchOp.match_op_names(target, ["func.func"]).result
    transform.apply_registered_pass(
        transform.any_op_t(),
        func,
        "linalg-morph-ops",
        options={
            "named-to-category": named_to_category,
            "category-to-generic": category_to_generic,
            "named-to-generic": named_to_generic,
            "generic-to-named": generic_to_named,
            "generic-to-category": generic_to_category,
        },
    )
