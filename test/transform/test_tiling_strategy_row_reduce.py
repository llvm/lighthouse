# RUN: %PYTHON %s | FileCheck %s

# Edge cases of the reduction tiling heuristics: element types, SIMD widths,
# dynamic and non-divisible dim sizes.

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform.transform_ext.utils.tiling import (
    StrategyContext,
    get_tiling_strategy,
)
from lighthouse.execution.target import TargetInfo

ROW_REDUCE = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%a: tensor<ROWSxCOLSxTYPE>, %o: tensor<ROWSxTYPE>) -> tensor<ROWSxTYPE> {
    %r = linalg.generic {indexing_maps = [#id, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%a : tensor<ROWSxCOLSxTYPE>) outs(%o : tensor<ROWSxTYPE>) {
    ^bb0(%in: TYPE, %out: TYPE):
      BODY
    } -> tensor<ROWSxTYPE>
    return %r : tensor<ROWSxTYPE>
  }
}
"""

ADD = "%s = arith.addf %in, %out : TYPE\n      linalg.yield %s : TYPE"


def run(
    name: str,
    rows: str = "64",
    cols: str = "4096",
    elem: str = "f32",
    body: str = ADD,
    features: tuple[str, ...] = ("avx512f",),
):
    payload = (
        ROW_REDUCE.replace("BODY", body)
        .replace("ROWS", rows)
        .replace("COLS", cols)
        .replace("TYPE", elem)
    )
    with TargetInfo.override(arch="x86_64", features=list(features)):
        ctx = StrategyContext(target=TargetInfo.host())
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()
            module = ir.Module.parse(payload)
            func = module.body.operations[0]
            op = func.regions[0].blocks[0].operations[0]
            tiles = {
                level: get_tiling_strategy(f"register_{level}").compute(op, ctx)
                for level in ("parallel", "reduction", "unroll")
            }
            print(
                f"{name}: parallel={tiles['parallel']}"
                f" reduction={tiles['reduction']}"
                f" unroll={tiles['unroll']}"
            )


# The reduced vector dim is tiled by lanes x 8 chains of the compute width:
# bf16 is computed as f32.
# CHECK: f32: parallel=[1, 0] reduction=[0, 128] unroll=[1, 0]
run("f32")
# CHECK: bf16: parallel=[1, 0] reduction=[0, 128]
run("bf16", elem="bf16")
# CHECK: f64: parallel=[1, 0] reduction=[0, 64]
run("f64", elem="f64")
# CHECK: i8: parallel=[1, 0] reduction=[0, 512]
run("i8", elem="i8", body=ADD.replace("addf", "addi"))
# CHECK: sse: parallel=[1, 0] reduction=[0, 32]
run("sse", features=("sse4_2",))

# Short rows are reduced whole; more rows share the chains.
# CHECK: short_row: parallel=[2, 0] reduction=None unroll=[1, 0]
run("short_row", cols="64")

# No multiple of the vector width divides 4095: the reduced dim is not tiled.
# CHECK: non_divisible: parallel=[1, 0] reduction=None unroll=[1, 0]
run("non_divisible", cols="4095")

# Dynamic reduced dim size: only the unit unroll shape is known.
# CHECK: dynamic: parallel=None reduction=None unroll=[1, 0]
run("dynamic", cols="?")
