# RUN: python %s | FileCheck %s

# Simply demonstrates applying a schedule to a payload.
# To do so generates a basic payload and a basic schedule, purely as an example.

from mlir.ir import Context, Location, InsertionPoint, Operation, Module
from mlir.ir import RankedTensorType, F32Type, UnitAttr
from mlir.dialects import arith, func, linalg, tensor, transform
from mlir.dialects.transform import structured


def example_payload() -> Module:
    """IR for:
    Zero = ...
    X = matmul(..., C=Zero)
    Y = matmul(..., C=Zero)
    Res = add(X, Y)

    Can be re-written to:
    X = matmul(..., C=Zero)
    Res = matmul(..., C=X)
    """

    print("NOTE: example payload module:")
    payload = Module.create()
    with InsertionPoint(payload.body):
        matrixType = RankedTensorType.get([16, 16], F32Type.get())

        # NB: Do the CHECKing on the transformed output:
        # CHECK-LABEL: result of applying schedule to payload
        # CHECK: func.func @fold_add_on_two_matmuls
        # CHECK-SAME:      (%[[MATRIX_A:.*]]: {{.*}}, %[[MATRIX_B:.*]]: {{.*}}, %[[WEIGHTS:.*]]: {{.*}})
        @func.func(matrixType, matrixType, matrixType)
        def fold_add_on_two_matmuls(matrixA, matrixB, weights):
            empty = tensor.empty(matrixType.shape, matrixType.element_type)
            c0 = arith.constant(F32Type.get(), 0.0)
            # CHECK: %[[ZERO_INIT:.*]] = linalg.fill
            zero_init = linalg.fill(c0, outs=[empty])
            # CHECK: %[[A_X_WEIGHTS:.*]] = linalg.matmul ins(%[[MATRIX_A]], %[[WEIGHTS]]{{.*}}) outs(%[[ZERO_INIT]]
            A_x_weights = linalg.matmul(matrixA, weights, outs=[zero_init])
            # CHECK: %[[RES:.*]] = linalg.matmul ins(%[[MATRIX_B]], %[[WEIGHTS]]{{.*}}) outs(%[[A_X_WEIGHTS]]
            B_x_weights = linalg.matmul(matrixB, weights, outs=[zero_init])
            # CHECK-NOT: linalg.add
            added = linalg.add(A_x_weights, B_x_weights, outs=[empty])
            # CHECK: return %[[RES]]
            return added

    print(payload)
    return payload


def example_schedule() -> Module:
    """Basic schedule wrapping a single rewrite pattern."""

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
    # Actual schedule is defined by the contained transform.named_sequence:
    schedule: transform.NamedSequenceOp = schedule_module.body.operations[0]

    schedule.apply(payload)  # The actual transformation happens here.

    print("NOTE: result of applying schedule to payload:")
    print(payload)
