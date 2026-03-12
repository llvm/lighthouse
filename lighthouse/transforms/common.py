from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import loop
from mlir.dialects.transform import structured


def cleanup(target: ir.Operation | ir.Value):
    func = structured.MatchOp.match_op_names(target, ["func.func"]).result
    transform.apply_cse(func)
    with ir.InsertionPoint(transform.ApplyPatternsOp(func).patterns):
        transform.apply_patterns_canonicalization()


def loop_hoisting(target, target_op: str | structured.MatchInterfaceEnum):
    if isinstance(target_op, structured.MatchInterfaceEnum):
        ops = structured.MatchOp(
            transform.any_op_t(),
            target,
            interface=target_op,
        )
    else:
        ops = structured.MatchOp.match_op_names(target, [target_op])
    transform.apply_licm(ops)
    loop.loop_hoist_loop_invariant_subsets(ops)
