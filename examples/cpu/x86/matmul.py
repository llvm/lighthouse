# RUN: %PYTHON %s --dump-kernel=vectorized | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=vectorized --tile-size=64 | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=vectorized --dtype=bf16 --avx512 | FileCheck %s --check-prefix=AVX512

# CHECK: vector.broadcast
# CHECK: vector.fma

# AVX512: x86.avx512.dot

"""
Matrix multiplication C = A * B on CPU.
"""

import ctypes
import argparse
import sys
import warnings

import ml_dtypes
import numpy as np
from mlir import ir
from mlir.dialects import linalg, transform
from mlir.dialects.transform import tensor

from lighthouse import dialects as lh_dialects
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution import (
    benchmark,
    execute,
    get_bench_wrapper_schedule,
)
from lighthouse.utils.numpy import numpy_to_mlir_type
from lighthouse.pipeline.helper import apply_registered_pass
import lighthouse.utils as lh_utils
from lighthouse import schedule as lh_schedule
import lighthouse.schedule.x86 as lh_schedule_x86
from lighthouse import transform as lh_transform
import lighthouse.transform.x86 as lh_transform_x86
import lighthouse.ingress.mlir_gen.utils as lh_mlir_utils
from functools import cached_property
from typing import Optional

from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
from mlir.dialects import bufferization


class Matmul:
    """
    Computes GEMM: C = A * B on CPU.
    """

    payload_function_name: str = "payload"

    def __init__(self, M: int, N: int, K: int, dtype=np.float32, tile_size: int = 32):
        if dtype not in [np.float32, ml_dtypes.bfloat16]:
            raise ValueError("Unsupported data type")
        if dtype == ml_dtypes.bfloat16:
            # For BF16, enforce fixed tile size due to current rewriter pattern matching limitation.
            # TODO: Relax when x86 BF16 pass supports dynamic vector transfer indexing.
            tile_size = 32
            warnings.warn(f"Overriding BF16 tile size to: {tile_size}")
        if tile_size % 32 != 0:
            raise ValueError(f"Tile must be a multiple of 32 but got: {tile_size}")
        if any(dim % tile_size != 0 for dim in [M, N, K]):
            raise ValueError("Dimensions must be divisible by the tile")

        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.tile_size = tile_size

    @cached_property
    def _input_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(123)
        A = np.random.rand(self.M, self.K).astype(self.dtype)
        B = np.random.rand(self.K, self.N).astype(self.dtype)
        C = np.random.rand(self.M, self.N).astype(self.dtype)
        return [A, B, C]

    def _get_input_arrays(self) -> list[ctypes.Structure]:
        return [get_ranked_memref_descriptor(a) for a in self._input_arrays]

    def shared_libs(self) -> list[str]:
        return []

    def get_complexity(self) -> tuple[int, int, int]:
        nbytes = np.dtype(self.dtype).itemsize
        flop_count = 2 * self.M * self.N * self.K
        memory_reads = (self.M * self.K + self.K * self.N) * nbytes  # read A and B
        memory_writes = self.M * self.N * nbytes  # write C
        return (flop_count, memory_reads, memory_writes)

    def payload_module(self) -> ir.Module:
        mod = ir.Module.create()

        with ir.InsertionPoint(mod.body):
            mlir_dtype = numpy_to_mlir_type(self.dtype)

            def tensor_t(shape, dtype=mlir_dtype):
                return ir.RankedTensorType.get(shape, dtype)

            def memref_t(shape, dtype=mlir_dtype):
                return ir.MemRefType.get(shape, dtype)

            A_shape = [self.M, self.K]
            B_shape = [self.K, self.N]
            C_shape = [self.M, self.N]
            f32_type = ir.F32Type.get()
            fargs = [memref_t(A_shape), memref_t(B_shape), memref_t(C_shape, f32_type)]

            @lh_utils.mlir.func_cif(*fargs, name=self.payload_function_name)
            def payload(A, B, C):
                a_tensor = bufferization.to_tensor(tensor_t(A_shape), A, restrict=True)
                b_tensor = bufferization.to_tensor(tensor_t(B_shape), B, restrict=True)

                # Accumulate in F32.
                fill = lh_mlir_utils.get_outputs(tensor_t(C_shape, f32_type))
                matmul = linalg.matmul(a_tensor, b_tensor, outs=[fill])

                bufferization.materialize_in_destination(
                    None, matmul, C, restrict=True, writable=True
                )

        return mod

    def schedule_modules(
        self,
        stop_at_stage: Optional[str] = None,
        parameters: Optional[dict] = None,
    ) -> list[ir.Module]:
        scheds = []

        # Insert performance measurements.
        scheds.append(get_bench_wrapper_schedule(self.payload_function_name))

        if stop_at_stage == "initial":
            return scheds

        # GEMM block packing.
        # Create cache-friendly access pattern across matmul tiles.
        scheds.append(
            lh_schedule.block_pack_matmuls(
                block_factors=[self.tile_size, self.tile_size, self.tile_size],
                rhs_transpose_outer_block=True,
                rhs_transpose_inner_block=False,
            )
        )
        scheds.append(lh_schedule_x86.lower_packs_unpacks(self.tile_size))

        # Convert to category ops for easier op matching.
        with lh_schedule.schedule_boilerplate() as (sched, named_seq):
            ops = lh_transform.match_op(named_seq.bodyTarget, "func.func")
            transform.apply_registered_pass(
                transform.any_op_t(),
                ops,
                "linalg-morph-ops",
                options={
                    "named-to-category": True,
                    "generic-to-category": True,
                },
            )
            lh_transform.cleanup(named_seq.bodyTarget)
            transform.yield_()
        scheds.append(sched)

        # GEMM cache tiling.
        # Create memory friendly access pattern.
        gemm_op = "linalg.contract"
        with lh_schedule.schedule_boilerplate() as (sched, named_seq):
            ops = lh_transform.match_op(named_seq.bodyTarget, gemm_op)
            with lh_transform.foreach(ops) as op:
                lh_transform_x86.matmul_cache_tiling(
                    op, num_tiles=6, tile_size=self.tile_size, fuse_producers=True
                )
                transform.yield_()
            transform.yield_()
        scheds.append(sched)

        # Fold extra parallel outer unit dims before further tiling to help later
        # vectorization rewrites to recognize ops.
        scheds.append(lh_schedule.linalg_contract_fold_unit_dims())

        # GEMM register tiling.
        # Ensure that computation can fit into vector registers.
        reg_tile_batch = 1
        reg_tile_m = 8
        reg_tile_n = 32
        reg_tile_k = 2
        reg_peel_loops = []
        assert self.tile_size % reg_tile_k == 0, "Invalid K dim register tiling"
        if self.tile_size % reg_tile_n != 0:
            reg_peel_loops.append(1)
        if self.tile_size % reg_tile_m != 0:
            reg_peel_loops.append(0)
        scheds.append(
            lh_schedule.tile_ops(
                gemm_op,
                tile_sizes=[reg_tile_batch, reg_tile_m, reg_tile_n, reg_tile_k],
                tile_interchange=[1, 2, 0, 3],
                peel_loops=reg_peel_loops,
            )
        )

        # GEMM register unroll.
        # Ensure that shapes are compatible with target hardware instructions.
        reg_unroll_m = 1
        reg_unroll_n = 16
        # When VNNI can be used, tuples of 32-bit elements are needed.
        reg_unroll_k = 2 if self.dtype == ml_dtypes.bfloat16 else 1
        reg_unroll_factors = [
            reg_tile_m // reg_unroll_m,
            reg_tile_n // reg_unroll_n,
            reg_tile_k // reg_unroll_k,
        ]
        scheds.append(
            lh_schedule.tile_ops(
                gemm_op,
                tile_sizes=[0, reg_unroll_m, reg_unroll_n, reg_unroll_k],
                unroll_factors=reg_unroll_factors,
            )
        )

        # Further tiling into hardware-friendly sizes for vectorization.
        scheds.append(lh_schedule.tile_ops("linalg.fill", tile_sizes=[1, 1, 1]))
        scheds.append(lh_schedule.tile_ops("linalg.generic", tile_sizes=[1, 8]))

        if stop_at_stage == "tiled":
            return scheds

        # Vectorization.
        scheds.append(lh_schedule.vectorize_linalg())
        scheds.append(lh_schedule.hoist_loops())

        with lh_schedule.schedule_boilerplate() as (sched, named_seq):
            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                tensor.apply_patterns_tensor_fold_tensor_subset_ops_into_vector_transfers()
                transform.apply_patterns_canonicalization()
            transform.yield_()
        scheds.append(sched)

        # Rewrite vector ops into x86-specific sequences.
        scheds.append(lh_schedule.x86_vectorization())

        # Lower to memrefs.
        scheds.append(lh_schedule.bufferize(deallocation_pipeline=True))

        # Apply x86 vectorization again as some patterns require memref abstraction.
        scheds.append(lh_schedule.x86_vectorization())
        # Vectorize any remaining ops.
        scheds.append(lh_schedule.vectorize_all())

        # Cleanup vector ops.
        with lh_schedule.schedule_boilerplate() as (sched, named_seq):
            lh_transform.flatten_vector_ops(named_seq.bodyTarget)
            lh_transform.cleanup(named_seq.bodyTarget)
            transform.yield_()
        scheds.append(sched)

        if stop_at_stage == "vectorized":
            return scheds

        # Lower to LLVM.
        with lh_schedule.schedule_boilerplate() as (sched, named_seq):
            target = named_seq.bodyTarget
            target = apply_registered_pass(target, "convert-linalg-to-loops")
            target = apply_registered_pass(target, "fold-memref-alias-ops")
            target = apply_registered_pass(target, "expand-strided-metadata")
            target = apply_registered_pass(target, "canonicalize")
            target = apply_registered_pass(target, "convert-vector-to-scf")
            target = apply_registered_pass(target, "lower-affine")
            target = apply_registered_pass(target, "convert-scf-to-cf")
            target = apply_registered_pass(target, "convert-vector-to-llvm")
            target = apply_registered_pass(target, "convert-to-llvm")
            target = apply_registered_pass(target, "reconcile-unrealized-casts")
            lh_transform.cleanup(target)

            transform.yield_()
        scheds.append(sched)

        return scheds


def parse_cli():
    parser = argparse.ArgumentParser(
        description="CPU x86 vectorized matmul",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs=3,
        default=[1024, 1024, 1024],
        help="M,N,K matrix sizes (A=MxK, B=KxN, C=MxN).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=32,
        help="Size to use for matmul tiling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="f32",
        choices=[
            "f32",
            "bf16",
        ],
        help="Input data type.",
    )
    parser.add_argument(
        "--avx512",
        action=argparse.BooleanOptionalAction,
        help="Enable AVX512 vectorization",
    )
    parser.add_argument(
        "--nruns",
        type=int,
        default=100,
        help="Number of runs to average the execution time.",
    )
    parser.add_argument(
        "--nwarmup",
        type=int,
        default=10,
        help="Number of warm-up iterations before benchmarking.",
    )
    parser.add_argument(
        "--dump-kernel",
        type=str,
        choices=[
            "initial",
            "tiled",
            "vectorized",
            "final",
        ],
        help="Dump kernel IR at different stages of lowering and exit without "
        "executing the kernel.",
    )
    parser.add_argument(
        "--dump-schedule",
        action="store_true",
        help="Dump transform schedule.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_cli()

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()

        match args.dtype:
            case "f32":
                in_dtype = np.float32
            case "bf16":
                in_dtype = ml_dtypes.bfloat16

        if in_dtype == ml_dtypes.bfloat16 and not args.avx512:
            print("BF16 requires AVX512 enabled")
            sys.exit(1)

        wload = Matmul(*args.sizes, dtype=in_dtype, tile_size=args.tile_size)
        pipeline = TransformDriver(
            wload.schedule_modules(stop_at_stage=args.dump_kernel)
        )
        payload = pipeline.apply(wload.payload_module())

        if args.dump_kernel or args.dump_schedule:
            if args.dump_kernel:
                print(payload)
            if args.dump_schedule:
                for schedule_module in wload.schedule_modules():
                    print(schedule_module)
            sys.exit(0)

        # check correctness
        execute(
            payload,
            host_input_buffers=wload._input_arrays,
            shared_libs=wload.shared_libs(),
            payload_function_name=wload.payload_function_name,
        )

        A, B, C = wload._input_arrays
        C_ref = np.matmul(A, B, dtype=np.float32)
        success = np.allclose(C, C_ref)
        if not success:
            print("FAILED Result mismatch.")
            sys.exit(1)

        times = benchmark(
            payload,
            host_input_buffers=wload._input_arrays,
            shared_libs=wload.shared_libs(),
            nruns=args.nruns,
            nwarmup=args.nwarmup,
        )

        times *= 1e6  # convert to microseconds

        print(f"MxNxK: {args.sizes}")
        print(f"Input dtype: {args.dtype}")

        # compute statistics
        mean = np.mean(times)
        min = np.min(times)
        max = np.max(times)
        std = np.std(times)
        print(
            f"Timings (us): mean = {mean:.2f} +/-{std:.2f} min={min:.2f} max={max:.2f}"
        )
        flop_count = wload.get_complexity()[0]
        gflops = flop_count / (mean * 1e-6) / 1e9
        print(f"Throughput: {gflops:.2f} GFLOPS")
