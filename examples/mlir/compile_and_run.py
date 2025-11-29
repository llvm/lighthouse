# RUN: %PYTHON %s

import torch
import argparse

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
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

        # For simplicity, use generic matchers without requiring specific types.
        anytype = transform.any_op_t()

        # Create entry point transformation sequence.
        with ir.InsertionPoint(schedule.body):
            named_seq = transform.NamedSequenceOp(
                sym_name="__transform_main",
                input_types=[anytype],
                result_types=[],
                arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
            )

        # Create the schedule.
        with ir.InsertionPoint(named_seq.body):
            # Find the kernel's function op.
            func = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["func.func"]
            )
            # Use C interface wrappers - required to make function executable
            # after jitting.
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
            transform.apply_cse(mod)
            with ir.InsertionPoint(transform.ApplyPatternsOp(mod).patterns):
                transform.apply_patterns_canonicalization()

            # Terminate the schedule.
            transform.yield_([])
    return schedule


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
def main(args):
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

    # Create a transform schedule and apply initial lowering to kernel.
    # The kernel is modified in-place.
    schedule_module = create_schedule(ctx)
    named_seq: transform.NamedSequenceOp = schedule_module.body.operations[0]
    named_seq.apply(kernel)

    # Create a pass pipeline and lower the kernel to LLVM dialect.
    pm = create_pass_pipeline(ctx)
    pm.run(kernel.operation)

    ### Compilation ###
    # Parse additional libraries if present.
    #
    # External shared libraries, runtime utilities, might be needed to execute
    # the compiled module.
    # The execution engine requires full paths to the libraries.
    mlir_libs = []
    if args.shared_libs:
        mlir_libs += args.shared_libs.split(",")

    # JIT the kernel.
    eng = ExecutionEngine(kernel, opt_level=2, shared_libs=mlir_libs)

    # Initialize the JIT engine.
    #
    # The deferred initialization executes global constructors that might
    # have been created by the module during engine creation (for example,
    # when `gpu.module` is present) or registered afterwards.
    #
    # Initialization is not strictly necessary in this case.
    # However, it is a good practice to perform it regardless.
    eng.initialize()

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
    parser = argparse.ArgumentParser()

    # External shared libraries, runtime utilities, might be needed to
    # execute the compiled module.
    # For example, MLIR runner utils libraries such as:
    #   - libmlir_runner_utils.so
    #   - libmlir_c_runner_utils.so
    #
    # Full paths to the libraries should be provided.
    # For example:
    #   --shared-libs=$LLVM_BUILD/lib/lib1.so,$LLVM_BUILD/lib/lib2.so
    parser.add_argument(
        "--shared-libs",
        type=str,
        help="Comma-separated list of libraries to link dynamically",
    )
    args = parser.parse_args()
    main(args)
