import torch
import os

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.dialects.transform import interpreter
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager

from lighthouse import utils as lh_utils


def create_kernel(ctx: ir.Context) -> ir.Module:
    """
    Create an MLIR module containing a function to execute.

    Args:
        ctx: MLIR context.
    """
    with ctx:
        module = ir.Module.parse(
            r"""
    // Compute element-wise addition.
    func.func @add(%a: memref<16x32xf32>, %b: memref<16x32xf32>, %out: memref<16x32xf32>) {
        linalg.add ins(%a, %b : memref<16x32xf32>, memref<16x32xf32>)
                   outs(%out : memref<16x32xf32>)
        return
    }
"""
        )
    return module


def create_schedule(ctx: ir.Context) -> ir.Module:
    """
    Create an MLIR module containing transformation schedule.
    The schedule provides partial lowering to scalar operations.

    Args:
        ctx: MLIR context.
    """
    with ctx, ir.Location.unknown(context=ctx):
        # Create transform module.
        schedule = ir.Module.create()
        schedule.operation.attributes["transform.with_named_sequence"] = (
            ir.UnitAttr.get()
        )

        # Create entry point transformation sequence.
        with ir.InsertionPoint(schedule.body):
            named_seq = transform.NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
            )

        # Create the schedule.
        with ir.InsertionPoint(named_seq.body):
            # For simplicity, use generic transform matchers.
            anytype = transform.AnyOpType.get()

            # Find the kernel's function op.
            func = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["func.func"]
            )
            # Use C interface wrappers - required to make function executable after jitting.
            func = transform.apply_registered_pass(
                anytype, func, "llvm-request-c-wrappers"
            )

            # Find the kernel's module op.
            mod = transform.get_parent_op(
                anytype, func, op_name="builtin.module", deduplicate=True
            )
            # Naive lowering to loops.
            mod = transform.apply_registered_pass(
                anytype, mod, "convert-linalg-to-loops"
            )
            # Cleanup.
            transform.ApplyCommonSubexpressionEliminationOp(mod)
            with ir.InsertionPoint(transform.ApplyPatternsOp(mod).patterns):
                transform.ApplyCanonicalizationPatternsOp()

            # Terminate the schedule.
            transform.YieldOp()
    return schedule


def apply_schedule(kernel: ir.Module, schedule: ir.Module) -> None:
    """
    Apply transformation schedule to a kernel module.
    The kernel is modified in-place.

    Args:
        kernel: A module with payload function.
        schedule: A module with transform schedule.
    """
    interpreter.apply_named_sequence(
        payload_root=kernel,
        transform_root=schedule.body.operations[0],
        transform_module=schedule,
    )


def create_pass_pipeline(ctx: ir.Context) -> PassManager:
    """
    Create an MLIR pass pipeline.
    The pipeline lowers operations further down to LLVM dialect.

    Args:
        ctx: MLIR context.
    """
    with ctx:
        # Create a pass manager that applies passes to the whole module.
        pm = PassManager("builtin.module")
        # Lower to LLVM.
        pm.add("convert-scf-to-cf")
        pm.add("convert-to-llvm")
        pm.add("reconcile-unrealized-casts")
        # Cleanup
        pm.add("cse")
        pm.add("canonicalize")
    return pm


# The example's entry point.
def main():
    ### Baseline computation ###
    # Create inputs.
    a = torch.randn(16, 32, dtype=torch.float32)
    b = torch.randn(16, 32, dtype=torch.float32)

    # Compute baseline result to verify numerical correctness.
    out_ref = torch.add(a, b)

    ### MLIR payload preparation ###
    # Create payload kernel.
    ctx = ir.Context()
    kernel = create_kernel(ctx)

    # Create a transform schedule and apply initial lowering.
    schedule = create_schedule(ctx)
    apply_schedule(kernel, schedule)

    # Create a pass pipeline and lower the kernel to LLVM dialect.
    pm = create_pass_pipeline(ctx)
    pm.run(kernel.operation)

    ### Compilation ###
    # External shared libraries, containing MLIR runner utilities, are are generally
    # required to execute the compiled module.
    #
    # Get paths to MLIR runner shared libraries through an environment variable.
    mlir_libs = os.environ.get("LIGHTHOUSE_SHARED_LIBS", default="").split(":")

    # JIT the kernel.
    eng = ExecutionEngine(kernel, opt_level=2, shared_libs=mlir_libs)
    # Get the kernel function.
    add_func = eng.lookup("add")

    ### Execution ###
    # Create an empty buffer to hold results.
    out = torch.empty_like(out_ref)

    # Execute the kernel.
    args = lh_utils.torch_to_packed_args([a, b, out])
    add_func(args)

    ### Verification ###
    # Check numerical correctness.
    if not torch.allclose(out_ref, out, rtol=0.01, atol=0.01):
        print("Error! Result mismatch!")
    else:
        print("Result matched!")


if __name__ == "__main__":
    main()
