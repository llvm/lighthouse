# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

from lighthouse import schedule as lh_schedule
from lighthouse import transform as lh_transform


def test_insertion_points():
    with lh_schedule.schedule_boilerplate() as (sched, named_seq):
        # Create empty foreach op between two ops.
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        foreach_op = lh_transform.foreach(funcs)
        loops = lh_transform.match_op(named_seq.bodyTarget, "scf.for")

        # Insert transforms in the foreach.
        with foreach_op as func:
            transform.apply_dce(func)

        # Create another complete foreach loop with results.
        foreach_with_res = lh_transform.foreach(
            funcs, loops, result_types=[transform.any_op_t()]
        )
        with foreach_with_res as (func, loop):
            transform.apply_cse(func)
            transform.apply_licm(loop)
            transform.yield_([func])

        # Terminate the first foreach
        with foreach_op:
            transform.yield_()

        # Insert print at the end of the schedule
        transform.print_(target=foreach_with_res.results[0])
        transform.yield_()
    sched.body.operations[0].verify()
    print(sched)


# CHECK: %[[FUNCS:.+]] = transform.structured.match ops{["func.func"]}
# CHECK: transform.foreach %[[FUNCS]]
# CHECK: ^bb0(%[[FUNC:.+]]: !transform.any_op):
# CHECK:   transform.apply_dce to %[[FUNC]]
# CHECK: %[[LOOPS:.+]] = transform.structured.match ops{["scf.for"]}
# CHECK: %[[RES:.+]] = transform.foreach %[[FUNCS]], %[[LOOPS]]
# CHECK: ^bb0(%[[FUNC:.+]]: !transform.any_op, %[[LOOP:.+]]: !transform.any_op):
# CHECK:   transform.apply_cse to %[[FUNC]]
# CHECK:   transform.apply_licm to %[[LOOP]]
# CHECK:   transform.yield %[[FUNC]]
# CHECK: transform.print %[[RES]]
# CHECK: transform.yield

with ir.Context(), ir.Location.unknown():
    test_insertion_points()
