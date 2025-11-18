# RUN: %PYTHON %s | FileCheck %s

# A basic example of generating a payload, a schedule, and applying the latter
# to the former. Shows how to do it from Python and from the cmd given the
# payload and schedule are .mlir files. Run this file to see the concrete
# schedule IR, pre-transform payload IR and transformed payload IR.

import tempfile
import subprocess

from mlir.ir import Context, Location, InsertionPoint, Operation, Module
from mlir.ir import RankedTensorType, F32Type, FloatAttr, DenseElementsAttr, UnitAttr
from mlir.dialects import arith, func, linalg, tensor, transform
from mlir.dialects.transform import structured


def example_payload() -> Module:
    """Example payload where the results of two matmuls are summed together.

    Can be re-written so that the second matmul accumulates top of the the result of the first.
    """

    print("NOTE: example payload module:")
    payload = Module.create()
    with InsertionPoint(payload.body):
        matrixType = RankedTensorType.get([16, 16], F32Type.get())

        # NB: Do the CHECKing on the transformed output:
        # CHECK-LABEL: result of applying schedule to payload
        # CHECK: func.func @fold_add_on_two_matmuls
        # CHECK-SAME:      (%[[MATRIX_A:.*]]: {{.*}}, %[[MATRIX_B:.*]]: {{.*}})
        @func.func(matrixType, matrixType)
        def fold_add_on_two_matmuls(matrixA, matrixB):
            splat_float = FloatAttr.get(F32Type.get(), 1.111111)
            splat_attr = DenseElementsAttr.get_splat(matrixType, splat_float)
            # CHECK: %[[WEIGHTS:.*]] = arith.constant dense<1.11
            weights = arith.constant(matrixType, splat_attr)
            c0 = arith.constant(F32Type.get(), 0.0)
            empty = tensor.empty(matrixType.shape, matrixType.element_type)
            # CHECK: %[[ZERO_INIT:.*]] = linalg.fill
            zero_init = linalg.fill(c0, outs=[empty])
            # CHECK: %[[A_X_WEIGHTS:.*]] = linalg.matmul ins(%[[MATRIX_A]], %[[WEIGHTS]]{{.*}}) outs(%[[ZERO_INIT]]
            A_x_weights = linalg.matmul(matrixA, weights, outs=[zero_init])
            empty2 = tensor.empty(matrixType.shape, matrixType.element_type)
            zero_init2 = linalg.fill(c0, outs=[empty2])
            # CHECK: %[[RES:.*]] = linalg.matmul ins(%[[MATRIX_B]], %[[WEIGHTS]]{{.*}}) outs(%[[A_X_WEIGHTS]]
            B_x_weights = linalg.matmul(matrixB, weights, outs=[zero_init2])
            # CHECK-NOT: linalg.add
            added = linalg.add(A_x_weights, B_x_weights, outs=[empty])
            # CHECK: return %[[RES]]
            return added

    print(payload)
    return payload


def example_schedule() -> Module:
    """Most basic schedule which doesn't just wrap a pass -- wraps a single rewrite pattern."""
    print("NOTE: example schedule module:")
    schedule_module = Module.create()
    schedule_module.operation.attributes["transform.with_named_sequence"] = (
        UnitAttr.get()
    )
    with InsertionPoint(schedule_module.body):
        named_seq = transform.named_sequence(
            "__transform_main",
            input_types=[transform.any_op_t()],
            result_types=[],
            arg_attrs=[{"transform.readonly": UnitAttr.get()}],
        )

        with InsertionPoint(named_seq.body):
            func = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["func.func"]
            )  # TODO: fix syntax upstream
            with InsertionPoint(transform.apply_patterns(func).patterns):
                Operation.create(
                    "transform.apply_patterns.linalg.fold_add_into_dest"
                )  # TODO: expose dedicated builder upstream
            transform.yield_([])

    print(schedule_module)
    return schedule_module


with Context(), Location.unknown():
    payload = example_payload()
    schedule_module = example_schedule()
    schedule: transform.NamedSequenceOp = schedule_module.body.operations[0]

    print(
        "NOTE: result of applying schedule to payload directly within Python process:"
    )
    schedule.apply(payload)
    print(payload)

    # Demonstrate applying a schedule from file to a payload from file
    with (
        tempfile.NamedTemporaryFile("w", prefix="payload_") as payload_file,
        tempfile.NamedTemporaryFile("w", prefix="schedule_") as schedule_file,
    ):
        print(payload, file=payload_file, flush=True)
        print("NOTE: Have dumped payload to temp file:", payload_file.name)
        print(schedule_module, file=schedule_file, flush=True)
        print("NOTE: Have dumped schedule to temp file:", schedule_file.name)

        cmdline = [
            "python",
            "-m",
            "lighthouse.schedule",
            schedule_file.name,
            payload_file.name,
        ]
        print(
            "NOTE: output of applying schedule to payload from commandline:", *cmdline
        )
        print(subprocess.run(cmdline, capture_output=True).stdout.decode())
        print(
            f"NOTE: cleaning-up temp files: {payload_file.name}, {schedule_file.name}"
        )
