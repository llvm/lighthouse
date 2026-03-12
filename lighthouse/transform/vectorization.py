from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.dialects.transform import x86


def vectorize_ops(
    target: ir.Operation | ir.Value,
    target_op: str | structured.MatchInterfaceEnum,
    vector_sizes: list = [],
    vectorize_kwargs: dict = {},
):
    if isinstance(target_op, structured.MatchInterfaceEnum):
        ops = structured.MatchOp(
            transform.any_op_t(),
            target,
            interface=target_op,
        )
    else:
        ops = structured.MatchOp.match_op_names(target, [target_op])
    foreach = transform.ForeachOp([], (ops,))
    with ir.InsertionPoint(foreach.body):
        op = foreach.bodyTargets[0]
        structured.structured_vectorize(op, vector_sizes, **vectorize_kwargs)
        transform.yield_()


def vectorize_all_ops(
    target: ir.Operation | ir.Value,
):
    func = structured.MatchOp.match_op_names(target, ["func.func"]).result
    structured.structured_vectorize_children_and_apply_patterns(
        transform.any_op_t(), func
    )


def x86_vector_patterns(target: ir.Operation | ir.Value):
    with ir.InsertionPoint(transform.ApplyPatternsOp(target).patterns):
        x86.apply_patterns_x86_vector_contract_to_packed_type_dot_product()
        x86.apply_patterns_x86_vector_contract_to_fma()
        x86.apply_patterns_x86_sink_vector_producer_ops()
