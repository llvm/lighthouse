# RUN: %PYTHON %s | FileCheck %s

"""Tests that a Runner looks up each JIT function once."""

import numpy as np
from mlir import ir
from mlir.passmanager import PassManager

from lighthouse.execution.runner import Runner


class CountingEngine:
    """Forwards to an execution engine, counting function lookups."""

    def __init__(self, engine):
        self.engine = engine
        self.lookups = 0

    def lookup(self, name):
        self.lookups += 1
        return self.engine.lookup(name)

    def __getattr__(self, name):
        return getattr(self.engine, name)


with ir.Context(), ir.Location.unknown():
    module = ir.Module.parse(
        """
func.func @increment(%a: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.0 : f32
  %v = memref.load %a[%c0] : memref<4xf32>
  %s = arith.addf %v, %one : f32
  memref.store %s, %a[%c0] : memref<4xf32>
  return
}
"""
    )
    Runner.make_function_callable(module, "increment")
    PassManager.parse(
        "builtin.module(finalize-memref-to-llvm, convert-to-llvm, "
        "reconcile-unrealized-casts)"
    ).run(module.operation)
    runner = Runner(module)

engine = runner.engine = CountingEngine(runner.engine)
buffer = np.zeros(4, dtype=np.float32)
for _ in range(3):
    runner.execute("increment", [buffer])

# CHECK: value: 3.0 lookups: 1
print("value:", buffer[0], "lookups:", engine.lookups)

# A missing function still fails on every call.
for _ in range(2):
    try:
        runner.execute("missing", [buffer])
    except Exception as error:
        # CHECK-COUNT-2: error: Unknown function missing
        print("error:", error)
