# RUN: %PYTHON %s | FileCheck %s

"""Collapsing parallel loops preserves iteration coverage and GPU parallelism."""

import itertools

import numpy as np
from mlir import ir
from mlir.dialects import transform
from mlir.passmanager import PassManager

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.execution.runner import Runner
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.xegpu.lowering_common import convert_to_gpu_launch


def payload(rank):
    indices = [f"%i{i}" for i in range(rank)]
    operands = ", ".join(indices)
    flatten = " + ".join(f"d{i} * {8 ** (rank - i - 1)}" for i in range(rank))
    dimensions = ", ".join(f"d{i}" for i in range(rank))
    return f"""
func.func @payload(%out: memref<?xi64>, %limit: memref<1xindex>) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c6 = arith.constant 6 : index
  %upper = memref.load %limit[%c0] : memref<1xindex>
  scf.parallel ({operands}) = ({", ".join(["%c1"] * rank)})
      to ({", ".join(["%c6"] * (rank - 1) + ["%upper"])})
      step ({", ".join(["%c2"] * rank)}) {{
    %offset = affine.apply affine_map<({dimensions})->({flatten})>({operands})
    %previous = memref.load %out[%offset] : memref<?xi64>
    %one = arith.constant 1 : i64
    %value = arith.addi %previous, %one : i64
    memref.store %value, %out[%offset] : memref<?xi64>
    scf.reduce
  }}
  return
}}
"""


def test_coverage():
    for rank in range(1, 6):
        for max_rank in (1, 3):
            module = ir.Module.parse(payload(rank))
            with schedule_boilerplate() as (schedule, sequence):
                loops = lh_transform.match_op(sequence.bodyTarget, "scf.parallel")
                transform_ext.collapse_parallel(loops, max_rank=max_rank)
                transform.yield_()
            schedule.body.operations[0].apply(module.operation)
            assert module.operation.verify()
            parallel = module.body.operations[0].body.blocks[0].operations
            loop = next(op for op in parallel if op.name == "scf.parallel")
            assert len(loop.lowerBound) == min(rank, max_rank)
            PassManager.parse(
                "builtin.module(canonicalize,scf-parallel-for-to-nested-fors,"
                "lower-affine,convert-scf-to-cf)"
            ).run(module.operation)
            Runner.make_function_callable(module, "payload")
            PassManager.parse(
                "builtin.module(finalize-memref-to-llvm,convert-to-llvm,"
                "reconcile-unrealized-casts)"
            ).run(module.operation)
            runner = Runner(module)
            for upper in (0, 1, 6, 7):
                output = np.full(8**rank, -1, dtype=np.int64)
                expected = output.copy()
                ranges = [range(1, 6, 2)] * (rank - 1) + [range(1, upper, 2)]
                for indices in itertools.product(*ranges):
                    offset = sum(
                        value * 8 ** (rank - dim - 1)
                        for dim, value in enumerate(indices)
                    )
                    expected[offset] = 0
                runner.execute("payload", [output, np.array([upper], dtype=np.int64)])
                np.testing.assert_array_equal(output, expected)
    print("Ranks 1-5 preserve dynamic, strided and empty iteration spaces")


# CHECK: Ranks 1-5 preserve dynamic, strided and empty iteration spaces


def test_gpu_grid():
    module = ir.Module.parse("""
func.func @payload(%out: memref<4x8x4x512xindex>) {
  scf.forall (%b, %h, %g, %q) in (4, 8, 4, 4) {
    %c128 = arith.constant 128 : index
    %query = arith.muli %q, %c128 : index
    memref.store %query, %out[%b, %h, %g, %query]
      : memref<4x8x4x512xindex>
  }
  return
}
""")
    with schedule_boilerplate() as (schedule, sequence):
        func = lh_transform.match_op(sequence.bodyTarget, "func.func")
        convert_to_gpu_launch(sequence.bodyTarget, payload_func=func)
        transform.yield_()
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    launches = []

    def visit(op):
        assert op.name != "scf.for", "Query tiles must not become a serial loop"
        if op.name == "gpu.launch":
            launches.append(op.opview)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    assert len(launches) == 1
    sizes = [
        ir.IntegerAttr(value.owner.attributes["value"]).value
        for value in launches[0].operands[:3]
    ]
    assert sizes == [32, 4, 4], sizes
    print("Four-dimensional attention grid maps to 32 x 4 x 4 GPU blocks")


# CHECK: Four-dimensional attention grid maps to 32 x 4 x 4 GPU blocks


def test_invalid_targets():
    for name, max_rank, message in (
        ("scf.parallel", 0, "max_rank must be positive"),
        ("func.func", 3, "expected unmapped scf.parallel"),
    ):
        module = ir.Module.parse(payload(4))
        before = str(module)
        with schedule_boilerplate() as (schedule, sequence):
            target = lh_transform.match_op(sequence.bodyTarget, name)
            transform_ext.collapse_parallel(target, max_rank=max_rank)
            transform.yield_()
        try:
            schedule.body.operations[0].apply(module.operation)
        except ValueError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError("Invalid collapse must fail")
        assert str(module) == before
    print("Invalid collapse requests fail without mutating IR")


# CHECK: Invalid collapse requests fail without mutating IR


with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()
    test_coverage()
    test_gpu_grid()
    test_invalid_targets()
