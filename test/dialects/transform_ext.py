# RUN: %PYTHON %s | FileCheck %s

"""Tests for the lighthouse transform_ext dialect operations."""

from mlir import ir
from mlir.dialects import transform, func, arith, index
from mlir.dialects.transform import structured as transform_structured
import lighthouse.dialects as lh_dialects
from lighthouse.dialects import transform_ext


def run(f):
    print("Test: ", f.__name__, flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load(
            reload=True
        )  # reload across @run-s to ensure interfaces are registered for new contexts

        mod = ir.Module.create()
        mod.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
        with ir.InsertionPoint(mod.body):
            named_seq = transform.named_sequence(
                "main", [transform.AnyOpType.get()], []
            )
            any_op = transform.AnyOpType.get()
            with ir.InsertionPoint(named_seq.body):
                func_funcs = transform_structured.structured_match(
                    any_op, named_seq.bodyTarget, ops={"func.func"}
                )
                payload_handle = transform.get_parent_op(any_op, func_funcs)

                f(payload_handle)
                transform.yield_()

            named_seq.verify()

            try:
                payload_mod = payload()
                named_seq.apply(payload_mod)
            except Exception as e:
                print(f"Caught exception: {e}", flush=True)


def payload():
    with ir.InsertionPoint((mod := ir.Module.create()).body):

        @func.func()
        def payload_func():
            c42 = arith.constant(ir.IndexType.get(), 42)
            c67 = index.constant(67)
            twice_c42 = arith.addi(c42, c42)
            arith.subi(twice_c42, c67)
            return twice_c42

    return mod


# CHECK-LABEL: Test: test_param_cmp_eq_op
@run
def test_param_cmp_eq_op(payload_handle):
    """Exercise GetNamedAttributeOp and ParamCmpEqOp:

    * by comparing the payload's arith.constant value vs 42 and 0.
    """

    arith_constant = transform_structured.structured_match(
        transform.AnyOpType.get(), payload_handle, ops={"arith.constant"}
    )
    constant_value_param = transform_ext.get_named_attribute(arith_constant, "value")
    c42_attr = ir.IntegerAttr.get(ir.IndexType.get(), 42)
    c42_as_param = transform.param_constant(transform.AnyParamType.get(), c42_attr)

    transform_ext.ParamCmpEqOp(constant_value_param, c42_as_param)
    c0_attr = ir.IntegerAttr.get(ir.IndexType.get(), 0)
    # CHECK: got here
    transform.print_(name="got here")
    # Comparing 42 against 0 — should fail and abort the sequence.
    c0_as_param = transform.param_constant(transform.AnyParamType.get(), c0_attr)
    transform_ext.ParamCmpEqOp(constant_value_param, c0_as_param)
    # CHECK-NOT: but not here
    # CHECK: Caught exception: Failed to apply named transform sequence
    transform.print_(name="but not here")


# CHECK-LABEL: Test: test_replace_op
@run
def test_replace_op(payload_handle):
    """Exercise ReplaceOp by:

    * replacing the payload's arith.constant with an index.constant
    * replacing the arith.addi with an index.add while keeping its operands
    * replacing the arith.subi with an index.sub while replacing its operands.
    """

    arith_constant = transform_structured.structured_match(
        transform.AnyOpType.get(), payload_handle, ops={"arith.constant"}
    )
    # Case 1: replace with new name, result types, and attributes.
    c123_attr = ir.IntegerAttr.get(ir.IndexType.get(), 123)
    new_attrs = ir.DictAttr.get({"value": c123_attr})
    _new_op = transform_ext.replace(
        arith_constant,
        op_kind="index.constant",
        new_result_types=[ir.IndexType.get()],
        new_attrs=new_attrs,
    )
    # Case 2: replace name only; operands are inherited from the original op.
    arith_addi = transform_structured.structured_match(
        transform.AnyOpType.get(), payload_handle, ops={"arith.addi"}
    )
    index_add = transform_ext.replace(arith_addi, op_kind="index.add")

    # Case 3: replace with explicitly supplied new operands.
    arith_subi = transform_structured.structured_match(
        transform.AnyOpType.get(), payload_handle, ops={"arith.subi"}
    )
    add_res = transform.get_result(transform.AnyValueType.get(), index_add, [0])
    transform_ext.replace(arith_subi, "index.sub", add_res, add_res)

    # CHECK: replace succeeded
    # CHECK: %[[C123:.*]] = index.constant 123
    # CHECK-NOT: arith.constant
    # CHECK-NOT: arith.addi
    # CHECK-NOT: arith.subi
    # CHECK: %[[ADD_RES:.*]] = index.add %[[C123]], %[[C123]]
    # CHECK: index.sub %[[ADD_RES]], %[[ADD_RES]]
    transform.print_(target=payload_handle, name="replace succeeded")
