from collections import OrderedDict

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import smt as transform_smt, tune as transform_tune


def set_selected(op: ir.Operation, env: dict[ir.Value | ir.Operation, object]):
    def recurse(op: ir.Operation):
        for region in op.regions:
            for block in region.blocks:
                for child in block:
                    set_selected(child, env)

    match type(op):
        case transform_tune.KnobOp:
            op.attributes["selected"] = env[op.result]
        case transform_tune.AlternativesOp:
            op.attributes["selected_region"] = env[op]
            recurse(op)
        case _:
            recurse(op)
    return op


# This is a hack >;(
def constraint_results_to_constants(
    op: ir.Operation | ir.Module,
    env: dict[ir.Value | ir.Operation, object],
    undo_actions=None,
):
    undo_actions = undo_actions if undo_actions is not None else []

    def undo():
        for action in reversed(undo_actions):
            action()

    match type(op):
        case transform_smt.ConstrainParamsOp:
            with ir.InsertionPoint.after(op):
                orig_results_and_uses = OrderedDict(
                    (res, list((use.owner, use.operand_number) for use in res.uses))
                    for res in op.results
                )

                for result in op.results:
                    val = transform.param_constant(result.type, env[result])

                    for use in result.uses:
                        use.owner.operands[use.operand_number] = val

                def undo_rewrite():
                    for orig_res, orig_uses in orig_results_and_uses.items():
                        for orig_owner, orig_operand_number in orig_uses:
                            param = orig_owner.operands[orig_operand_number].owner
                            assert isinstance(param, transform.ParamConstantOp)
                            orig_owner.operands[orig_operand_number] = orig_res
                        param.erase()

                undo_actions.append(undo_rewrite)
        case _:
            for region in op.regions:
                for block in region.blocks:
                    for child in block:
                        constraint_results_to_constants(child, env, undo_actions)

    return op, undo


def nondet_to_det(op: ir.Operation, env: dict[ir.Value, ir.Value] = None):
    env = env if env is not None else {}  # TODO: nested scopes

    i64 = ir.IntegerType.get_signless(64)
    transform_param_i64 = transform.ParamType.get(i64)

    match type(op):
        case transform_tune.KnobOp:
            assert "selected" in op.attributes
            with ir.InsertionPoint.after(op):
                subst = transform.param_constant(
                    transform_param_i64, op.attributes["selected"]
                )

            for use in op.result.uses:
                use.owner.operands[use.operand_number] = subst

            op.erase()

        case transform_tune.AlternativesOp:
            assert "selected_region" in op.attributes
            region_idx = op.attributes["selected_region"].value
            with ir.InsertionPoint.after(op):
                for child in op.regions[region_idx].blocks[0]:
                    new_yield = cloned_child = child.clone()
                    for result, new_result in zip(child.results, cloned_child.results):
                        env[result] = new_result
                    for idx, operand in enumerate(cloned_child.operands):
                        if operand in env:
                            cloned_child.operands[idx] = env[operand]
                    nondet_to_det(cloned_child, env)
                for yield_operand, result in zip(new_yield.operands, op.results):
                    for res_use in result.uses:
                        res_use.owner.operands[res_use.operand_number] = yield_operand
                new_yield.erase()

            op.erase()

        case transform_smt.ConstrainParamsOp:
            for res in op.results:
                assert next(res.uses, None) is None

            op.erase()

        case _:
            for region in op.regions:
                for block in region.blocks:
                    for child in block:
                        nondet_to_det(child, env)

            return op
