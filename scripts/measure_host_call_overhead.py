#!/usr/bin/env python3
"""Measure the host-side overhead of calling a JIT-compiled kernel.

Runs an empty kernel through lighthouse's Runner with PyTorch tensors, so all
the measured time is host overhead: memref descriptor creation (to_memref),
argument packing, function lookup and the call itself.
"""

import argparse
import statistics
import time
from collections.abc import Callable

import torch
from mlir import ir
from mlir.passmanager import PassManager

import lighthouse.dialects as lh_dialects
from lighthouse.execution.runner import Runner
from lighthouse.ingress.torch.compile import TorchMemoryManager
from lighthouse.utils.memref import to_packed_args
from lighthouse.utils.torch import to_memref


def empty_kernel(num_args: int, rank: int) -> ir.Module:
    """A function of `num_args` f32 memrefs of `rank` that does nothing."""
    shape = "x".join(["?"] * rank)
    args = ", ".join(f"%a{i}: memref<{shape}xf32>" for i in range(num_args))
    module = ir.Module.parse(f"func.func @main({args}) {{ return }}")
    Runner.make_function_callable(module, "main")
    PassManager.parse(
        "builtin.module(finalize-memref-to-llvm, convert-to-llvm, "
        "reconcile-unrealized-casts)"
    ).run(module.operation)
    return module


def time_us(fn: Callable[[], object], iters: int, repeats: int = 5) -> float:
    """The median over `repeats` of the mean time per call of `fn`, in us."""
    for _ in range(min(iters, 1000)):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter_ns() - start) / iters / 1e3)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=20000, help="calls per sample")
    parser.add_argument(
        "--num-args", type=int, nargs="+", default=[1, 3, 8], help="kernel arguments"
    )
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[1, 2, 4], help="argument ranks"
    )
    args = parser.parse_args()

    print("Time per call in us; 'call' is the JIT call with prepared arguments.")
    header = ("args", "rank", "to_memref/arg", "to_packed_args", "execute", "call")
    print("".join(f"{name:>16}" for name in header))
    for num_args in args.num_args:
        for rank in args.ranks:
            with ir.Context(), ir.Location.unknown():
                lh_dialects.register_and_load()
                module = empty_kernel(num_args, rank)
                runner = Runner(module, mem_manager_cls=TorchMemoryManager)
            tensors = [torch.zeros([4] * rank) for _ in range(num_args)]
            descriptors = [to_memref(tensor) for tensor in tensors]
            packed_args = to_packed_args(descriptors)
            func = runner.engine.lookup("main")

            times = (
                time_us(lambda: to_memref(tensors[0]), args.iters),
                time_us(lambda: to_packed_args(descriptors), args.iters),
                time_us(lambda: runner.execute("main", tensors), args.iters),
                time_us(lambda: func(packed_args), args.iters),
            )
            row = f"{num_args:>16}{rank:>16}"
            print(row + "".join(f"{t:>16.2f}" for t in times))


if __name__ == "__main__":
    main()
