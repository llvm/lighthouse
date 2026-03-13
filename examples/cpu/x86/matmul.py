# RUN: %PYTHON %s | FileCheck %s
# CHECK: func.func @payload
# CHECK: PASSED
# CHECK: Throughput:
"""
Matrix multiplication C = A * B on CPU.
"""

import ctypes
from contextlib import contextmanager

import ml_dtypes
import numpy as np
import mlir
from mlir import ir
from mlir.dialects import linalg, arith, transform
import mlir.dialects.tensor
from mlir.execution_engine import ExecutionEngine
from mlir.dialects.transform import structured
from mlir.dialects.transform import loop
from mlir.dialects.transform import vector
from mlir.dialects.transform import tensor

from lighthouse.workload import benchmark
from lighthouse.utils.numpy import numpy_to_mlir_type
from lighthouse.pipeline.helper import apply_registered_pass
import lighthouse.utils as lh_utils
from lighthouse import schedule as lh_schedule
from lighthouse import transform as lh_transform
from functools import cached_property
from typing import Optional

from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
from mlir.dialects import bufferization

from lighthouse.workload import Workload


def lower_packs_for_vectorization(
    target: ir.Operation | ir.Value,
    pack_tile_sizes: list[int],
    vector_tile_sizes: list[int] | None = None,
    vector_unroll_factors: list[int] = [],
):
    packs = structured.MatchOp.match_op_names(target, ["linalg.pack"])
    foreach_pack = transform.ForeachOp([], (packs,))
    with ir.InsertionPoint(foreach_pack.body):
        pack_op = foreach_pack.bodyTargets[0]
        tiled_pack = structured.TileUsingForOp(
            pack_op, sizes=pack_tile_sizes
        ).tiled_linalg_op
        _, _, transpose = structured.structured_lower_pack(
            transform.OperationType.get("tensor.pad"),
            transform.OperationType.get("tensor.expand_shape"),
            transform.OperationType.get("linalg.transpose"),
            tiled_pack,
            lower_pad_like_with_insert_slice=False,
        )
        if vector_tile_sizes:
            _, *loops = structured.TileUsingForOp(
                transpose, sizes=vector_tile_sizes
            ).results
            for idx, factor in enumerate(reversed(vector_unroll_factors)):
                loop.loop_unroll(loops[-1 - idx], factor)
        transform.yield_()


def lower_unpacks_for_vectorization(
    target: ir.Operation | ir.Value,
    unpack_tile_sizes: list[int],
    vector_tile_sizes: list[int] | None = None,
):
    unpacks = structured.MatchOp.match_op_names(target, ["linalg.unpack"])
    foreach_unpack = transform.ForeachOp([], (unpacks,))
    with ir.InsertionPoint(foreach_unpack.body):
        unpack_op = foreach_unpack.bodyTargets[0]
        tiled_unpack = structured.TileUsingForOp(
            unpack_op, sizes=unpack_tile_sizes
        ).tiled_linalg_op
        if vector_tile_sizes:
            tiled_unpack = structured.TileUsingForOp(
                tiled_unpack, sizes=vector_tile_sizes
            ).tiled_linalg_op
        structured.structured_lower_unpack(
            transform.OperationType.get("tensor.empty"),
            transform.OperationType.get("linalg.transpose"),
            transform.OperationType.get("tensor.collapse_shape"),
            transform.OperationType.get("tensor.extract_slice"),
            transform.OperationType.get("linalg.copy"),
            tiled_unpack,
            lower_unpad_like_with_extract_slice=True,
        )
        transform.yield_()


def schedule_lower_packs_unpacks(tile_size: int) -> ir.Module:
    sched = lh_schedule.create_schedule()
    named_seq = lh_schedule.create_named_sequence(
        sched, input_types=[transform.any_op_t()]
    )

    with ir.InsertionPoint(named_seq.body):
        pack_unpack_vector_m = 8
        pack_unpack_vector_n = min(64, tile_size)
        lower_packs_for_vectorization(
            named_seq.bodyTarget,
            pack_tile_sizes=[1, 1],
            vector_tile_sizes=[1, 1, pack_unpack_vector_m, pack_unpack_vector_n],
            vector_unroll_factors=[
                tile_size // pack_unpack_vector_n,
            ],
        )
        lh_transform.cleanup(named_seq.bodyTarget)

        lower_unpacks_for_vectorization(
            named_seq.bodyTarget,
            unpack_tile_sizes=[tile_size, tile_size],
            vector_tile_sizes=[1],
        )
        lh_transform.vectorize_ops(named_seq.bodyTarget, "linalg.transpose")

        # Cleanup.
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            tensor.apply_patterns_tensor_fold_tensor_subset_ops_into_vector_transfers()
            transform.apply_patterns_canonicalization()
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            vector.apply_patterns_vector_cast_away_vector_leading_one_dim()
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return sched


def schedule_linalg_contract_fold_unit_dims() -> ir.Module:
    sched = lh_schedule.create_schedule()
    named_seq = lh_schedule.create_named_sequence(
        sched, input_types=[transform.any_op_t()]
    )

    with ir.InsertionPoint(named_seq.body):
        lh_transform.linalg_morph_ops(named_seq.bodyTarget, category_to_generic=True)
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            # Works only on generics.
            structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
        lh_transform.linalg_morph_ops(named_seq.bodyTarget, generic_to_category=True)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return sched


class Matmul(Workload):
    """
    Computes GEMM: C = A * B on CPU.
    """

    def __init__(self, M: int, N: int, K: int, dtype=np.float32, tile_size: int = 32):
        if dtype not in [np.float32, ml_dtypes.bfloat16]:
            raise ValueError("Unsupported data type")
        if dtype == ml_dtypes.bfloat16:
            # For BF16, enforce fixed tile size due to current rewriter pattern matching limitation.
            # TODO: Relax when x86 BF16 pass supports dynamic indexing.
            tile_size = 32
        if tile_size % 32 != 0:
            raise ValueError(f"Tile must be a multiple of 32 but got: {tile_size}")
        if any(dim % tile_size != 0 for dim in [M, N, K]):
            raise ValueError("Dimensions must be divisible by the tile")

        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.tile_size = tile_size

    def check_correctness(
        self, execution_engine: ExecutionEngine, verbose: int = 0
    ) -> bool:
        A, B, C = self._input_arrays
        out_ref = np.matmul(A, B, dtype=np.float32)
        return np.allclose(C, out_ref)

    @cached_property
    def _input_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(123)
        A = np.random.rand(self.M, self.K).astype(self.dtype)
        B = np.random.rand(self.K, self.N).astype(self.dtype)
        C = np.random.rand(self.M, self.N).astype(self.dtype)
        return [A, B, C]

    def _get_input_arrays(self) -> list[ctypes.Structure]:
        return [get_ranked_memref_descriptor(a) for a in self._input_arrays]

    @contextmanager
    def allocate_inputs(self, execution_engine: ExecutionEngine):
        try:
            yield self._get_input_arrays()
        finally:
            # cached numpy arrays are deallocated automatically
            pass

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
                empty = mlir.dialects.tensor.empty(C_shape, f32_type)
                zero_cst = arith.constant(f32_type, 0.0)
                fill = linalg.fill(zero_cst, outs=[empty])

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

        scheds.append(
            lh_schedule.pack_matmuls(
                block_factors=[self.tile_size, self.tile_size, self.tile_size],
                rhs_transpose_outer_block=True,
                rhs_transpose_inner_block=False,
            )
        )
        scheds.append(schedule_lower_packs_unpacks(self.tile_size))

        # Convert to category ops for easier op matching.
        scheds.append(lh_schedule.linalg_to_category())

        # GEMM cache tiling.
        # Create memory friendly access pattern.
        gemm_op = "linalg.contract"
        scheds.append(lh_schedule.tile(gemm_op, tile_sizes=[1, 1], fuse_producers=True))

        # Further tiling into hardware-friendly sizes for later vectorization.
        scheds.append(lh_schedule.tile("linalg.fill", tile_sizes=[1, 1, 1]))
        scheds.append(lh_schedule.tile("linalg.generic", tile_sizes=[1, 1, 8]))
        scheds.append(schedule_linalg_contract_fold_unit_dims())

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
            lh_schedule.tile(
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
            lh_schedule.tile(
                gemm_op,
                tile_sizes=[0, reg_unroll_m, reg_unroll_n, reg_unroll_k],
                unroll_factors=reg_unroll_factors,
            )
        )

        # Vectorization.
        scheds.append(lh_schedule.vectorize_linalg())
        scheds.append(lh_schedule.hoist_loops())

        sched = lh_schedule.create_schedule()
        named_seq = lh_schedule.create_named_sequence(
            sched, input_types=[transform.any_op_t()]
        )
        with ir.InsertionPoint(named_seq.body):
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
        sched = lh_schedule.create_schedule()
        named_seq = lh_schedule.create_named_sequence(
            sched, input_types=[transform.any_op_t()]
        )
        with ir.InsertionPoint(named_seq.body):
            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                vector.apply_patterns_vector_flatten_vector_transfer_ops()
                transform.apply_patterns_canonicalization()
            lh_transform.cleanup(named_seq.bodyTarget)
            transform.yield_()
        scheds.append(sched)

        # Lower to LLVM.
        sched = lh_schedule.create_schedule()
        named_seq = lh_schedule.create_named_sequence(
            sched, input_types=[transform.any_op_t()]
        )
        with ir.InsertionPoint(named_seq.body):
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


if __name__ == "__main__":
    with ir.Context(), ir.Location.unknown():
        wload = Matmul(1024, 1024, 1024, dtype=np.float32, tile_size=32)

        print(" Dump kernel ".center(60, "-"))
        wload.lower_payload(dump_payload="initial", dump_schedule=False)

        print(" Benchmark ".center(60, "-"))
        times = benchmark(wload, check_correctness=True)
        times *= 1e6  # convert to microseconds
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
