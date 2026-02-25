# REQUIRES: mpi4py
# RUN: mpirun -n 4 %PYTHON %s --mpilib=%VIRTUAL_ENV/lib/libmpi.so.12 | FileCheck %s
# RUN: mpirun -n 4 %PYTHON %s --mpilib=%VIRTUAL_ENV/lib/libmpi.so.12 --grid 0 0 | FileCheck %s
# RUN: mpirun -n 4 %PYTHON %s --mpilib=%VIRTUAL_ENV/lib/libmpi.so.12 --grid 4 1 | FileCheck %s
# CHECK: PASSED
"""
A single MLP that can run on multiple MPI ranks,
following a 1d/2d weight-stationary partition strategy
(see a and b from figure 2 of https://arxiv.org/pdf/2211.05102)
"""

import argparse
import ctypes
from contextlib import contextmanager
from typing import Optional

import numpy as np
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform.bufferization import OneShotBufferizeOp
from mlir.dialects.bufferization import LayoutMapOption
from mlir.execution_engine import ExecutionEngine
from mlir.runtime.np_to_memref import (
    ranked_memref_to_numpy,
    make_nd_memref_descriptor,
    as_ctype,
)
from lighthouse.utils.memref import (
    to_ctype as memref_to_ctype,
    deallocate_memrefs_on_exit,
)
from lighthouse.utils.mlir import apply_registered_pass, match
from lighthouse.workload import Workload, execute

from mlp_weight_stationary import generate_mlp_payload

from mpi4py import MPI


if not MPI.Is_initialized():
    MPI.Init()
WORLD_SIZE = MPI.COMM_WORLD.Get_size()
WORLD_RANK = MPI.COMM_WORLD.Get_rank()


def rprint(*args, **kwargs):
    if WORLD_RANK == 0:
        print(*args, **kwargs)


def parse_cla():
    parser = argparse.ArgumentParser(
        description="MLP on MPI using MLIR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        "-s",
        type=int,
        nargs=3,
        default=[64, 128, 32],
        help="M,N,K matrix sizes (Activations=MxK, WeightsIn=KxN, WeightsOut=NxK, Result=MxK).",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=[WORLD_SIZE],
        nargs="+",
        help="The shape of the device grid (1 or 2 dimensions). The product of the grid dimensions must match the number of MPI ranks. Use '0' if 2d grid dimensions should be inferred automatically.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=0,
        help="Verbosity level.",
    )
    parser.add_argument(
        "--mpilib",
        type=str,
        default="libmpi.so",
        help="MPI shared library to load.",
    )
    args = parser.parse_args()
    assert len(args.grid) in (1, 2), "Only 1D and 2D grids are supported."
    assert all(x == 0 for x in args.grid) or np.prod(args.grid) == WORLD_SIZE, (
        "Grid size must be only '0's or match the number of MPI ranks."
    )
    if len(args.grid) == 1 and args.grid[0] == 0:
        args.grid = [WORLD_SIZE]
    assert len(args.grid) == 2 or args.grid[0] == WORLD_SIZE, (
        "1D grid size must match the number of MPI ranks."
    )
    return args


class DistMLP(Workload):
    """
    A single MLP that can run on multiple MPI ranks.

    A[:] = sigmoid(A@B)@C

    where A, B, C are (M,K), (K,N), (N,K) matrices respectively.
    """

    payload_function_name = "payload"

    def __init__(self, args, P: int, R: int):
        self.M = args.sizes[0]
        self.N = args.sizes[1]
        self.K = args.sizes[2]
        self.comm_size = WORLD_SIZE  # number of MPI ranks
        self.comm_rank = WORLD_RANK  # rank of this MPI process
        self.dtype = np.float32
        self.grid = args.grid
        self.mpilibs = [args.mpilib]
        self.verbose = args.verbose

    def _alloc_inout(self, execution_engine: ExecutionEngine) -> list[ctypes.Structure]:
        rprint(" * Allocating input/output arrays...")
        memrefs = [
            make_nd_memref_descriptor(2, as_ctype(self.dtype))() for _ in range(4)
        ]
        # allocation functions use the same sharding annotations as the payload,
        # so that each rank allocates only the part of the data it owns.
        for i, v in enumerate(["act", "win", "wout"]):
            execution_engine.invoke(f"alloc_{v}", memref_to_ctype(memrefs[i + 1]))
        return memrefs

    def _init_inout(self, r: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        rprint(" * Initializing input arrays...")
        np.random.seed(self.comm_rank)  # different seed for each rank
        A = ranked_memref_to_numpy([a])
        B = ranked_memref_to_numpy([b])
        C = ranked_memref_to_numpy([c])
        A[:] = np.random.rand(*A.shape).astype(self.dtype)
        B[:] = np.random.rand(*B.shape).astype(self.dtype)
        C[:] = np.random.rand(*C.shape).astype(self.dtype)

    @contextmanager
    def allocate_inputs(self, execution_engine: ExecutionEngine):
        self.input_memrefs = self._alloc_inout(execution_engine)
        self._init_inout(*self.input_memrefs)
        # Dealloc all memrefs on exit
        with deallocate_memrefs_on_exit(
            self.input_memrefs, execution_engine, "dealloc_2d"
        ):
            yield self.input_memrefs

    def _gather(
        self,
        memref: ctypes.Structure,
        execution_engine: ExecutionEngine,
        gather_func: str,
    ) -> ctypes.Structure:
        gathered_memref = make_nd_memref_descriptor(2, as_ctype(self.dtype))()
        execution_engine.invoke(
            gather_func,
            memref_to_ctype(gathered_memref),
            memref_to_ctype(memref),
        )
        return gathered_memref

    def _reference_solution(self, execution_engine: ExecutionEngine) -> np.ndarray:
        rprint(" * Gathering input data...")
        gathered = [
            self._gather(self.input_memrefs[i + 1], execution_engine, f"gather_{v}")
            for i, v in enumerate(["act", "win", "wout"])
        ]

        rprint(" * Computing reference solution...")

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        with deallocate_memrefs_on_exit(gathered, execution_engine, "dealloc_2d"):
            A, B, C = [ranked_memref_to_numpy([m]) for m in gathered]
            result = sigmoid(A @ B) @ C
        return result

    def check_correctness(
        self, execution_engine: ExecutionEngine, verbose: int = 0
    ) -> bool:
        gathered = self._gather(self.input_memrefs[0], execution_engine, "gather_act")
        with deallocate_memrefs_on_exit([gathered], execution_engine, "dealloc_2d"):
            R = ranked_memref_to_numpy([gathered])
            R_ref = self._reference_solution(execution_engine)
            if verbose > 1:
                rprint("Reference solution:")
                rprint(R_ref)
                rprint("Computed solution:")
                rprint(R)
            success = np.allclose(R, R_ref)
        success = MPI.COMM_WORLD.allreduce(success, op=MPI.LAND)
        if success:
            rprint("PASSED")
        else:
            rprint("FAILED Result mismatch!")
        return success

    def shared_libs(self) -> list[str]:
        return self.mpilibs + ["libmlir_c_runner_utils.so", "libmlir_runner_utils.so"]

    def get_complexity(self) -> tuple[int, int, int]:
        nbytes = np.dtype(self.dtype).itemsize
        flop_count = (
            4 * self.M * self.N * self.K + 4 * self.M * self.N
        )  # 2 matmuls (4MNK) + sigmoid (~4MN)
        memory_reads = 5 * self.M * self.N * nbytes
        memory_writes = (self.M * self.N + self.M * self.K) * nbytes
        return (flop_count, memory_reads, memory_writes)

    def payload_module(self) -> ir.Module:
        if len(self.grid) == 1:
            rprint(f"Using 1D grid of size {self.comm_size}")
            grid = [self.comm_size]
        else:
            assert len(self.grid) == 2
            if all(x != 0 for x in self.grid):
                p1, p2 = self.grid
            else:
                # find two factors of comm_size that are as close as possible
                def find_factors(n):
                    for i in range(int(n**0.5), 0, -1):
                        if n % i == 0:
                            return (i, n // i)
                    return (1, n)

                p1, p2 = find_factors(self.comm_size)
            rprint(f"Using 2D grid of size {p1}x{p2}")
            grid = [p1, p2]

        common = dict(
            func_name=self.payload_function_name,
            M=self.M,
            N=self.N,
            K=self.K,
            comm_size=self.comm_size,
            comm_rank=self.comm_rank,
            grid=grid,
        )
        if len(self.grid) == 1:
            mod = generate_mlp_payload(
                **common,
                split_act=[[], [0]],
                split_win=[[], [0]],
                split_wout=[[0], []],
                split_mm0a_mm1c=[[]],
                split_mm0_c=[[], [0]],
                split_sigmoid=[[], [0]],
            )
        else:
            mod = generate_mlp_payload(
                **common,
                split_act=[[], [0, 1]],
                split_win=[[0], [1]],
                split_wout=[[1], [0]],
                split_mm0a_mm1c=[[], [0]],
                split_mm0_c=[[], [1]],
                split_sigmoid=[[], [1, 0]],
            )

        if self.verbose > 1:
            rprint("Payload MLIR:")
            count = 1
            for line in str(mod).splitlines():
                rprint(str(count) + "\t" + line)
                count += 1

        return mod

    def schedule_module(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> ir.Module:
        schedule_module = ir.Module.create()
        schedule_module.operation.attributes["transform.with_named_sequence"] = (
            ir.UnitAttr.get()
        )
        with ir.InsertionPoint(schedule_module.body):
            named_sequence = transform.named_sequence(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
            )
            with ir.InsertionPoint(named_sequence.body):
                anytype = transform.AnyOpType.get()
                func = match(named_sequence.bodyTarget, ops={"func.func"})
                func = apply_registered_pass(
                    func,
                    "sharding-propagation",
                    options={"traversal": "forward-backward"},
                )
                if self.verbose > 0:
                    transform.PrintOp(target=func)
                func = apply_registered_pass(func, "shard-partition")
                func = apply_registered_pass(func, "canonicalize")
                if self.verbose > 0:
                    transform.PrintOp(target=func)
                func = apply_registered_pass(func, "convert-shard-to-mpi")
                func = apply_registered_pass(func, "canonicalize")
                if self.verbose > 0:
                    transform.PrintOp(target=func)
                func = apply_registered_pass(func, "tosa-to-linalg")
                mod = transform.get_parent_op(
                    anytype, func, op_name="builtin.module", deduplicate=True
                )
                mod = apply_registered_pass(mod, "tosa-to-tensor")
                mod = apply_registered_pass(mod, "linalg-generalize-named-ops")
                mod = apply_registered_pass(mod, "canonicalize")
                mod = apply_registered_pass(mod, "linalg-fuse-elementwise-ops")
                mod = apply_registered_pass(mod, "arith-expand")
                mod = apply_registered_pass(mod, "memref-expand")
                mod = apply_registered_pass(mod, "empty-tensor-to-alloc-tensor")
                mod = apply_registered_pass(mod, "canonicalize")
                identity_layout = LayoutMapOption.IdentityLayoutMap
                mod = OneShotBufferizeOp(
                    mod,
                    allow_return_allocs_from_loops=True,
                    bufferize_function_boundaries=True,
                    function_boundary_type_conversion=identity_layout,
                )
                mod = apply_registered_pass(mod, "expand-realloc")
                mod = apply_registered_pass(mod, "canonicalize")
                mod = apply_registered_pass(mod, "buffer-deallocation-simplification")
                mod = apply_registered_pass(mod, "bufferization-lower-deallocations")
                mod = apply_registered_pass(mod, "cse")
                mod = apply_registered_pass(mod, "canonicalize")
                mod = apply_registered_pass(mod, "convert-bufferization-to-memref")
                mod = apply_registered_pass(mod, "convert-linalg-to-parallel-loops")
                mod = apply_registered_pass(mod, "scf-parallel-loop-fusion")
                mod = apply_registered_pass(mod, "canonicalize")
                mod = apply_registered_pass(mod, "fold-memref-alias-ops")
                mod = apply_registered_pass(mod, "expand-strided-metadata")
                mod = apply_registered_pass(mod, "convert-math-to-funcs")
                mod = apply_registered_pass(mod, "lower-affine")
                mod = apply_registered_pass(mod, "convert-scf-to-cf")
                mod = apply_registered_pass(mod, "symbol-dce")
                mod = apply_registered_pass(mod, "finalize-memref-to-llvm")
                mod = apply_registered_pass(mod, "convert-math-to-llvm")
                mod = apply_registered_pass(mod, "convert-math-to-libm")
                mod = apply_registered_pass(mod, "convert-func-to-llvm")
                mod = apply_registered_pass(mod, "canonicalize")
                mod = apply_registered_pass(mod, "convert-to-llvm")
                mod = apply_registered_pass(mod, "reconcile-unrealized-casts")
                mod = apply_registered_pass(mod, "cse")
                if self.verbose > 1:
                    transform.PrintOp(target=mod)
                transform.YieldOp()

        return schedule_module


if __name__ == "__main__":
    args = parse_cla()

    if not MPI.Is_initialized():
        MPI.Init()
    P = MPI.COMM_WORLD.Get_size()
    R = MPI.COMM_WORLD.Get_rank()

    with ir.Context(), ir.Location.unknown():
        wload = DistMLP(args, P, R)

        rprint(" Execute".center(60, "-"))
        execute(wload, verbose=args.verbose)

    MPI.Finalize()
