# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASSED
"""
A single MLP that can run on multiple MPI ranks,
following partition strategies a) and b) from figure 2 of
https://arxiv.org/pdf/2211.05102
"""

import argparse
import ctypes
from pathlib import Path
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
from lighthouse.utils.memref import to_ctype as memref_to_ctype
from lighthouse.utils.mlir import apply_registered_pass, match
from lighthouse.workload import Workload, execute

from mpi4py import MPI


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
        default=[4096, 4096, 4096],
        help="M,N,K matrix sizes (Activations=MxK, WeightsIn=KxN, WeightsOut=MxN, Result=MxK).",
    )
    parser.add_argument(
        "-grid-dims",
        "-gd",
        type=int,
        default=1,
        help="Number of dimensions in device grid.",
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
    parser.add_argument(
        "--utils_dir",
        type=str,
        default="",
        help="Directory containing the MLIR C runner utils shared libraries.",
    )
    args = parser.parse_args()
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
        self.P = P  # number of MPI ranks
        self.R = R  # rank of this MPI process
        self.dtype = np.float32
        self.griddims = args.grid_dims
        self.mpilibs = [args.mpilib]
        self.utils_dir = args.utils_dir
        self.verbose = args.verbose

    def _alloc_inout(self, execution_engine: ExecutionEngine) -> list[ctypes.Structure]:
        print(" * Allocating input/output arrays...")
        memrefs = [
            make_nd_memref_descriptor(2, as_ctype(self.dtype))() for _ in range(4)
        ]
        for i, v in enumerate(["act", "win", "wout"]):
            execution_engine.invoke(f"alloc_{v}", memref_to_ctype(memrefs[i + 1]))
        return memrefs

    def _init_inout(self, r: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        print(" * Initializing input arrays...")
        np.random.seed(self.R)
        # R = ranked_memref_to_numpy([r])
        A = ranked_memref_to_numpy([a])
        B = ranked_memref_to_numpy([b])
        C = ranked_memref_to_numpy([c])
        A[:] = np.random.rand(*A.shape).astype(self.dtype)
        B[:] = np.random.rand(*B.shape).astype(self.dtype)
        C[:] = np.random.rand(*C.shape).astype(self.dtype)

    @contextmanager
    def allocate_inputs(self, execution_engine: ExecutionEngine):
        try:
            memrefs = self._alloc_inout(execution_engine)
            self._init_inout(*memrefs)
            self._input_arrays = memrefs
            yield memrefs
        finally:
            # cached numpy arrays are deallocated automatically
            pass

    def _reference_solution(self, execution_engine: ExecutionEngine) -> np.ndarray:
        print(" * Gathering input data...")
        gathered = []
        for i, v in enumerate(["act", "win", "wout"]):
            memref = make_nd_memref_descriptor(2, as_ctype(self.dtype))()
            execution_engine.invoke(
                f"gather_{v}",
                memref_to_ctype(memref),
                memref_to_ctype(self._input_arrays[i + 1]),
            )
            gathered.append(ranked_memref_to_numpy([memref]))

        print(" * Computing reference solution...")

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        A, B, C = gathered
        return sigmoid(A @ B) @ C

    def check_correctness(
        self, execution_engine: ExecutionEngine, verbose: int = 0
    ) -> bool:
        R = ranked_memref_to_numpy([self._input_arrays[0]])
        R_ref = self._reference_solution(execution_engine)
        if verbose > 1:
            print("Reference solution:")
            print(R_ref)
            print("Computed solution:")
            print(R)
        success = np.allclose(R, R_ref)
        if success:
            print("PASSED")
        else:
            print("FAILED Result mismatch!")
        return success

    def shared_libs(self) -> list[str]:
        utils_path = Path(self.utils_dir)
        return self.mpilibs + [
            str(utils_path / "libmlir_c_runner_utils.so"),
            str(utils_path / "libmlir_runner_utils.so"),
        ]

    def get_complexity(self) -> tuple[int, int, int]:
        nbytes = np.dtype(self.dtype).itemsize
        flop_count = (
            self.M * self.N * self.K * 2 + self.M * self.K * 4
        )  # matmuls + sigmoid
        memory_reads = 5 * self.M * self.N * nbytes
        memory_writes = (self.M * self.N + self.M * self.K) * nbytes
        return (flop_count, memory_reads, memory_writes)

    def payload_module(self) -> ir.Module:
        if self.griddims == 1:
            print(f"Using 1D grid of size {self.P}")
            grid = self.P
        elif self.griddims == 2:
            # find two factors of P that are as close as possible
            def find_factors(n):
                for i in range(int(n**0.5), 0, -1):
                    if n % i == 0:
                        return (i, n // i)
                return (1, n)

            p1, p2 = find_factors(self.P)
            print(f"Using 2D grid of size {p1}x{p2}")
            grid = f"{p1}x{p2}"
        else:
            raise ValueError(
                f"Only 1D and 2D grids are supported (not {self.griddims}d).\n"
            )

        fname = "mlp_weight_stationary.mlir"
        with open(fname, "r") as f:
            txt = f.read()

        format_values = {
            "func_name": self.payload_function_name,
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "P": self.P,
            "R": self.R,
            "grid": grid,
            "split_r": "[[]]",
        }
        if self.griddims == 1:
            format_values.update(
                {
                    "split_act": "[[], [0]]",
                    "split_win": "[[], [0]]",
                    "split_wout": "[[0], []]",
                    "split_mm0_a": "[[]]",
                    "split_mm0_b": "[[], [0]]",
                    "split_mm0_c": "[[], [0]]",
                    "split_sigmoid": "[[], [0]]",
                    "split_mm1_a": "[[], [0]]",
                    "split_mm1_b": "[[0], []]",
                    "split_mm1_c": "[[]]",
                }
            )
        elif self.griddims == 2:
            format_values.update(
                {
                    "split_act": "[[], [0, 1]]",
                    "split_win": "[[0], [1]]",
                    "split_wout": "[[1], [0]]",
                    "split_mm0_a": "[[], [0]]",
                    "split_mm0_b": "[[0], [1]]",
                    "split_mm0_c": "[[], [1]]",
                    "split_sigmoid": "[[], [1, 0]]",
                    "split_mm1_a": "[[], [1]]",
                    "split_mm1_b": "[[1], [0]]",
                    "split_mm1_c": "[[], [0]]",
                }
            )
        txt = txt.format_map(format_values)

        if self.verbose > 1:
            print("Payload MLIR:")
            count = 1
            for line in txt.splitlines():
                print(str(count) + "\t" + line)
                count += 1

        return ir.Module.parse(txt)

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
                func = apply_registered_pass(func, "shard-partition")
                func = apply_registered_pass(func, "canonicalize")
                func = apply_registered_pass(func, "convert-shard-to-mpi")
                func = apply_registered_pass(func, "canonicalize")
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
                    # test_analysis_only=True,
                    # print_conflicts=True,
                ).result

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

        print(" Execute".center(60, "-"))
        execute(wload, verbose=args.verbose)

        # print(" Execute 2 ".center(60, "-"))
        # execute(wload, verbose=1)

        # print(" Benchmark ".center(60, "-"))
        # times = benchmark(wload)
        # times *= 1e6  # convert to microseconds
        # compute statistics
        # mean = np.mean(times)
        # min = np.min(times)
        # max = np.max(times)
        # std = np.std(times)
        # print(f"Timings (us): mean={mean:.2f}+/-{std:.2f} min={min:.2f} max={max:.2f}")
        # flop_count = wload.get_complexity()[0]
        # gflops = flop_count / (mean * 1e-6) / 1e9
        # print(f"Throughput: {gflops:.2f} GFLOPS")
    MPI.Finalize()
