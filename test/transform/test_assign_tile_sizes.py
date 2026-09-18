# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform.transform_ext import assign_tile_sizes
from lighthouse.dialects.transform.transform_ext.ops.assign_tile_sizes import (
    AssignTileSizesOp,
)
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.builders import schedule_boilerplate

PAYLOAD = """
module {
  func.func @main(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<128x128xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<128x64xf32>, tensor<64x128xf32>)
        outs(%f : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %mm : tensor<128x128xf32>
  }
}
"""

PAYLOAD_ELTWISE = """
module {
    func.func @main(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
        %sum = linalg.elementwise <add>
                ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
                outs(%a : tensor<64x64xf32>) -> tensor<64x64xf32>
        return %sum : tensor<64x64xf32>
    }
}
"""


def run(
    name: str,
    payload_text: str,
    target_op: str,
    strategy: str,
    tile_size: int | None = None,
):
    print(f"Test: {name}", flush=True)
    # Pin arch/features/core_count: the eltwise cache strategy derives tile sizes
    # from the (otherwise host-dependent) SIMD width and core count.
    with TargetInfo.override(arch="x86_64", features=["avx512f"], core_count=16):
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()
            payload = ir.Module.parse(payload_text)
            with schedule_boilerplate() as (sched, named_seq):
                ops = lh_transform.match_op(named_seq.bodyTarget, target_op)
                annotated_handle = assign_tile_sizes(
                    ops,
                    tile_size=tile_size,
                    strategy=strategy,
                )
                # Verify the strategy attribute is wired on the op's IR.
                assign_op = annotated_handle.owner.opview
                assert isinstance(assign_op, AssignTileSizesOp), (
                    f"Expected AssignTileSizesOp, got {type(assign_op)}"
                )
                assert assign_op.strategy.value == strategy, (
                    f"Expected strategy={strategy!r}, got {assign_op.strategy.value!r}"
                )
                transform.yield_()
            sched.body.operations[0].apply(payload.operation)
            print(payload)


# CHECK-LABEL: Test: strategy_attr_register_parallel
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 8, 32, 0>
run("strategy_attr_register_parallel", PAYLOAD, "linalg.matmul", "register_parallel")

# CHECK-LABEL: Test: strategy_attr_register_reduction
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 0, 2>
run("strategy_attr_register_reduction", PAYLOAD, "linalg.matmul", "register_reduction")

# CHECK-LABEL: Test: strategy_attr_cache
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32, 0>
run("strategy_attr_cache", PAYLOAD, "linalg.matmul", "cache")

# CHECK-LABEL: Test: eltwise_non_default_tile_size
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 4, 16>
run(
    "eltwise_non_default_tile_size",
    PAYLOAD_ELTWISE,
    "linalg.elementwise",
    "cache",
    tile_size=16,
)
