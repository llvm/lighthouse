import tempfile
import subprocess

from mlir import ir
from mlir.dialects import arith, func, linalg, tensor, transform
from mlir.dialects.transform import structured

from lighthouse import transform as lh_transform


def example_payload() -> ir.Module:
    payload = ir.Module.create()
    with ir.InsertionPoint(payload.body):
        matrixType = ir.RankedTensorType.get([16, 16], ir.F32Type.get())

        @func.func(matrixType, matrixType)
        def fold_add_on_two_matmuls(matrixA, matrixB):
            splat_float = ir.FloatAttr.get(ir.F32Type.get(), 1.111111)
            splat_attr = ir.DenseElementsAttr.get_splat(matrixType, splat_float)
            weights = arith.constant(matrixType, splat_attr)
            c0 = arith.constant(ir.F32Type.get(), 0.0)
            empty = tensor.empty(matrixType.shape, matrixType.element_type)
            zero_init = linalg.fill(c0, outs=[empty])
            A_x_weights = linalg.matmul(matrixA, weights, outs=[zero_init])
            empty2 = tensor.empty(matrixType.shape, matrixType.element_type)
            zero_init2 = linalg.fill(c0, outs=[empty2])
            B_x_weights = linalg.matmul(matrixB, weights, outs=[zero_init2])
            added = linalg.add(A_x_weights, B_x_weights, outs=[empty])
            return added

    return payload


def example_schedule() -> ir.Module:
    schedule = ir.Module.create()
    schedule.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    with ir.InsertionPoint(schedule.body):
        named_seq = transform.NamedSequenceOp(  # TODO: fix snake_case wrapper upstream
            sym_name="__transform_main",
            input_types=[transform.any_op_t()],
            result_types=[],
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
        )

        with ir.InsertionPoint(named_seq.body):
            func = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["func.func"]
            )  # TODO: fix syntax upstream
            with ir.InsertionPoint(
                transform.apply_patterns(func).patterns.blocks.append()
            ):  # TODO: fix snake_case wrapper upstream
                ir.Operation.create(
                    "transform.apply_patterns.linalg.fold_add_into_dest"
                )  # TODO: expose dedicated builder upstream
            transform.yield_([])
    return schedule


with ir.Context(), ir.Location.unknown():
    payload_module = example_payload()
    print("NOTE: example payload module:")
    print(payload_module)
    schedule_module = example_schedule()
    print("NOTE: example schedule module:")
    print(schedule_module)

    print("NOTE: output of applying schedule to payload directly within Python process:")
    schedule = schedule_module.body.operations[0]
    lh_transform.apply(schedule, payload_module)
    print(payload_module)

    with tempfile.NamedTemporaryFile(
        "w", prefix="payload_"
    ) as payload_file, tempfile.NamedTemporaryFile(
        "w", prefix="schedule_"
    ) as schedule_file:
        print(payload_module, file=payload_file, flush=True)
        print("NOTE: Have dumped payload to temp file:", payload_file.name)
        print(schedule_module, file=schedule_file, flush=True)
        print("NOTE: Have dumped schedule to temp file:", schedule_file.name)

        cmdline = [
            "python",
            "-m",
            "lighthouse.transform",
            schedule_file.name,
            payload_file.name,
        ]
        print("NOTE: output of applying schedule to payload from commandline:", *cmdline)
        subprocess.run(cmdline)
        print(
            f"NOTE: cleaning-up temp files: {payload_file.name}, {schedule_file.name}"
        )
