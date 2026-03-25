# REQUIRES: mpi4py
# RUN: mpirun -n 4 %PYTHON %s --mpilib=%VIRTUAL_ENV/lib/libmpi.so.12 | FileCheck %s
# RUN: mpirun -n 4 %PYTHON %s --mpilib=%VIRTUAL_ENV/lib/libmpi.so.12 --grid 0 0 | FileCheck %s
# RUN: mpirun -n 4 %PYTHON %s --mpilib=%VIRTUAL_ENV/lib/libmpi.so.12 --grid 4 1 | FileCheck %s
# CHECK: PASSED
"""
A single feed-forward layer that can run on multiple MPI ranks,
following a 1d/2d weight-stationary partition strategy
(see a and b from figure 2 of https://arxiv.org/pdf/2211.05102)
"""

import argparse
from contextlib import contextmanager
from typing import Optional
import numpy as np

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform.bufferization import OneShotBufferizeOp
from mlir.dialects.bufferization import LayoutMapOption
from mlir.execution_engine import ExecutionEngine
from mlir.runtime.np_to_memref import ranked_memref_to_numpy

from lighthouse import dialects as lh_dialects
from lighthouse.pipeline.helper import apply_registered_pass, match
from lighthouse.workload import Workload, benchmark, get_bench_wrapper_schedule
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.x86 import tile_and_vector_matmul
from lighthouse.utils.numpy import numpy_to_mlir_type
from ff_weight_stationary import generate_ff_payload
from lighthouse.workload import ShardMemoryManager

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
        description="Feed-Forward on MPI using MLIR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        "-s",
        type=int,
        nargs=3,
        default=[128, 256, 512],
        help="M,N,K matrix sizes (Activations=MxK, WeightsIn=KxN, WeightsOut=NxK, Result=MxK).",
    )
    parser.add_argument(
        "--tile-size",
        "-t",
        type=int,
        default=64,
        help="Tile size for the tiled schedule.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=[WORLD_SIZE],
        nargs="+",
        help="The shape of the device grid (1 or 2 dimensions). The product of the grid dimensions must match the number of MPI ranks. Use '0' if 2d grid dimensions should be inferred automatically.",
    )
    parser.add_argument(
        "--nruns",
        type=int,
        default=50,
        help="Number of runs to average the execution time.",
    )
    parser.add_argument(
        "--nwarmup",
        type=int,
        default=5,
        help="Number of warm-up iterations before benchmarking.",
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


class DistFF(Workload):
    """
    A single feed-forward layer that can run on multiple MPI ranks.

    D = sigmoid(A@B)@C

    where A, B, C, D are (M,K), (K,N), (N,K), (M,K) matrices respectively.
    """

    def __init__(self, args, P: int, R: int):
        self.M = args.sizes[0]
        self.N = args.sizes[1]
        self.K = args.sizes[2]
        self.tile_size = args.tile_size
        self.comm_size = WORLD_SIZE  # number of MPI ranks
        self.comm_rank = WORLD_RANK  # rank of this MPI process
        self.dtype = np.float32
        self.grid = args.grid
        self.mpilibs = [args.mpilib]
        self.verbose = args.verbose
        self.memory_manager_class = ShardMemoryManager
        self.memory_manager = None

    @contextmanager
    def allocate_inputs(self, execution_engine: ExecutionEngine):
        rprint(" * Allocating input/output arrays...")

        if self.memory_manager is None:
            self.memory_manager = self.memory_manager_class(execution_engine)
        elem_type = numpy_to_mlir_type(self.dtype)
        np.random.seed(self.comm_rank)  # different seed for each rank

        def init(buf):
            view = ranked_memref_to_numpy([buf])
            view[:] = np.random.rand(*view.shape).astype(view.dtype) - 0.5

        # allocation functions use the same sharding annotations as the payload,
        # so that each rank allocates only the part of the data it owns.
        with self.memory_manager.get_input_buffers(
            [("act", 2), ("win", 2), ("wout", 2), ("act", 2)],
            names=["A", "B", "C", "D"],
            elem_type=elem_type,
            init_func=init,
        ) as inputs:
            yield inputs

    def _reference_solution(self, execution_engine: ExecutionEngine) -> np.ndarray:
        rprint(" * Gathering input data...")
        gathered = [
            self.memory_manager.gather(name, numpy_to_mlir_type(self.dtype))
            for name in ["A", "B", "C"]
        ]

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        rprint(" * Computing reference solution...")
        A, B, C = [ranked_memref_to_numpy([m]) for m in gathered]
        result = sigmoid(A @ B) @ C
        return result

    def check_correctness(
        self, execution_engine: ExecutionEngine, verbose: int = 0
    ) -> bool:
        R_ref = self._reference_solution(execution_engine)
        gathered = self.memory_manager.gather("D", numpy_to_mlir_type(self.dtype))
        R = ranked_memref_to_numpy([gathered])
        if verbose > 1:
            rprint("Reference solution:")
            rprint(R_ref)
            rprint("Computed solution:")
            rprint(R)
            diff = R - R_ref
            rprint(f"Difference: min={diff.min()}, max={diff.max()}")
        success = np.allclose(R, R_ref, atol=1e-5)
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
            split_act = [[], [0]]
            split_win = [[], [0]]
            split_wout = [[0], []]
            mod = generate_ff_payload(
                **common,
                split_act=split_act,
                split_win=split_win,
                split_wout=split_wout,
                split_mm0a_mm1c=[[]],
                split_mm0_c=[[], [0]],
                split_sigmoid=[[], [0]],
            )
        else:
            split_act = [[], [0, 1]]
            split_win = [[0], [1]]
            split_wout = [[1], [0]]
            mod = generate_ff_payload(
                **common,
                split_act=split_act,
                split_win=split_win,
                split_wout=split_wout,
                split_mm0a_mm1c=[[], [0]],
                split_mm0_c=[[], [1]],
                split_sigmoid=[[], [1, 0]],
            )

        self.memory_manager_class.emit_memory_management_funcs(
            mod,
            shapes=[
                ("act", [self.M, self.K], split_act),
                ("win", [self.K, self.N], split_win),
                ("wout", [self.N, self.K], split_wout),
            ],
            elem_type=numpy_to_mlir_type(self.dtype),
        )

        if self.verbose > 1:
            rprint("Payload MLIR:")
            count = 1
            for line in str(mod).splitlines():
                rprint(str(count) + "\t" + line)
                count += 1

        return mod

    def get_shard_schedule(self):
        with schedule_boilerplate() as (schedule, named_sequence):
            func = match(named_sequence.bodyTarget, ops={"func.func"})
            func = apply_registered_pass(
                func,
                "sharding-propagation",
                options={"traversal": "forward-backward"},
            )
            if self.verbose > 0:
                transform.PrintOp(target=func)
            func = apply_registered_pass(func, "shard-partition")
            if self.verbose > 0:
                transform.PrintOp(target=func)
            func = apply_registered_pass(func, "shard-simplify")
            if self.verbose > 0:
                transform.PrintOp(target=func)
            func = apply_registered_pass(func, "convert-shard-to-mpi")
            func = apply_registered_pass(func, "canonicalize")
            if self.verbose > 0:
                transform.PrintOp(target=func)
            func = apply_registered_pass(func, "tosa-to-linalg")
            transform.YieldOp()
        return schedule

    def get_bufferize_schedule(self):
        with schedule_boilerplate() as (schedule, named_sequence):
            anytype = transform.AnyOpType.get()
            func = match(named_sequence.bodyTarget, ops={"func.func"})
            mod = transform.get_parent_op(
                anytype, func, op_name="builtin.module", deduplicate=True
            )
            mod = apply_registered_pass(mod, "linalg-generalize-named-ops")
            mod = apply_registered_pass(mod, "linalg-fuse-elementwise-ops")
            identity_layout = LayoutMapOption.IdentityLayoutMap
            mod = apply_registered_pass(mod, "eliminate-empty-tensors")
            mod = OneShotBufferizeOp(
                mod,
                allow_return_allocs_from_loops=False,
                bufferize_function_boundaries=True,
                function_boundary_type_conversion=identity_layout,
            )
            mod = apply_registered_pass(
                mod,
                "drop-equivalent-buffer-results",
                options={"modify-public-functions": True},
            )

            # Run passes to inject deallocations. Don't do this for dealloc_2d, though.
            for fname in [
                self.benchmark_function_name,
                self.payload_function_name,
                "gather_act",
                "gather_win",
                "gather_wout",
                "alloc_act",
                "alloc_win",
                "alloc_wout",
            ]:
                func = match(
                    mod,
                    ops={"func.func"},
                    op_attrs={"sym_name": ir.StringAttr.get(fname)},
                )
                func = apply_registered_pass(func, "buffer-deallocation-pipeline")
                mod = transform.get_parent_op(
                    anytype, func, op_name="builtin.module", deduplicate=True
                )
            transform.YieldOp()
        return schedule

    def get_lower_schedule(self):
        with schedule_boilerplate() as (schedule, named_sequence):
            anytype = transform.AnyOpType.get()
            func = match(named_sequence.bodyTarget, ops={"func.func"})
            mod = transform.get_parent_op(
                anytype, func, op_name="builtin.module", deduplicate=True
            )
            mod = apply_registered_pass(mod, "convert-linalg-to-parallel-loops")
            mod = apply_registered_pass(mod, "scf-parallel-loop-fusion")
            mod = apply_registered_pass(mod, "canonicalize")
            mod = apply_registered_pass(mod, "expand-strided-metadata")
            mod = apply_registered_pass(mod, "lower-affine")
            mod = apply_registered_pass(mod, "convert-vector-to-scf")
            mod = apply_registered_pass(mod, "convert-scf-to-cf")
            mod = apply_registered_pass(mod, "symbol-dce")
            mod = apply_registered_pass(mod, "convert-vector-to-llvm")
            mod = apply_registered_pass(mod, "canonicalize")
            mod = apply_registered_pass(mod, "convert-to-llvm")
            mod = apply_registered_pass(mod, "reconcile-unrealized-casts")
            mod = apply_registered_pass(mod, "cse")
            if self.verbose > 1:
                transform.PrintOp(target=mod)
            transform.YieldOp()
        return schedule

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        """Generate schedules:
        - sharding propagation, partition, and MPI, tosa-to-linalg
        - adding benchmark wrapper
        - tile_and_vector
        - all the rest"""
        return [
            self.get_shard_schedule(),
            get_bench_wrapper_schedule(self),
            tile_and_vector_matmul.create_schedule(
                tile_sizes=[self.tile_size, self.tile_size]
            ),
            self.get_bufferize_schedule(),
            self.get_lower_schedule(),
        ]


if __name__ == "__main__":
    args = parse_cla()

    if not MPI.Is_initialized():
        MPI.Init()
    P = MPI.COMM_WORLD.Get_size()
    R = MPI.COMM_WORLD.Get_rank()

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()

        wload = DistFF(args, P, R)

        # execute(wload, verbose=args.verbose)
        rprint(" Benchmark".center(60, "-"))
        times = benchmark(
            wload,
            nruns=args.nruns,
            nwarmup=args.nwarmup,
            check_correctness=True,
            verbose=args.verbose,
        )
        # compute statistics
        times *= 1e6
        mean = np.mean(times)
        min = np.min(times)
        max = np.max(times)
        std = np.std(times)
        rprint(f"Timings (us): mean={mean:.2f}+/-{std:.2f} min={min:.2f} max={max:.2f}")
        flop_count = wload.get_complexity()[0]
        gflops = flop_count / (mean * 1e-6) / 1e9
        rprint(f"Throughput: {gflops:.2f} GFLOPS")

    MPI.Finalize()
