# RUN: %PYTHON %s | FileCheck %s

"""Store-derived broadcast layouts for RoPE tables and GEMM bias."""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule import schedule_boilerplate


ROPE = """
func.func @rope(%ad: !xegpu.tensor_desc<128x64xbf16>,
                %bd: !xegpu.tensor_desc<128x64xbf16>,
                %out: !xegpu.tensor_desc<4x128x64xbf16>,
                %projection: vector<4x128x64xbf16>) {
  %a = xegpu.load_nd %ad[0,0]
    : !xegpu.tensor_desc<128x64xbf16> -> vector<128x64xbf16>
  %b = xegpu.load_nd %bd[0,0]
    : !xegpu.tensor_desc<128x64xbf16> -> vector<128x64xbf16>
  %unused = xegpu.load_nd %ad[0,0]
    : !xegpu.tensor_desc<128x64xbf16> -> vector<128x64xbf16>
  %cos = vector.broadcast %a : vector<128x64xbf16> to vector<4x128x64xbf16>
  %sin = vector.broadcast %b : vector<128x64xbf16> to vector<4x128x64xbf16>
  %m = arith.mulf %projection, %cos : vector<4x128x64xbf16>
  %r = arith.addf %m, %sin : vector<4x128x64xbf16>
  xegpu.store_nd %r, %out[0,0,0] <{layout = #xegpu.layout<
    sg_layout=[4,4,2], sg_data=[1,32,32], inst_data=[1,8,16], order=[2,0,1]>}>
    : vector<4x128x64xbf16>, !xegpu.tensor_desc<4x128x64xbf16>
  return
}
"""

BIAS = """
func.func @bias(%src: memref<256xf32>, %offsets: vector<256xindex>,
                %mask: vector<256xi1>, %projection: vector<128x256xf32>,
                %out: !xegpu.tensor_desc<128x256xf32>) {
  %a = xegpu.load %src[%offsets], %mask
    : memref<256xf32>, vector<256xindex>, vector<256xi1> -> vector<256xf32>
  %b = vector.broadcast %a : vector<256xf32> to vector<128x256xf32>
  %r = arith.addf %projection, %b : vector<128x256xf32>
  xegpu.store_nd %r, %out[0,0] <{layout = #xegpu.layout<
    sg_layout=[4,8], sg_data=[32,32], inst_data=[8,16]>}>
    : vector<128x256xf32>, !xegpu.tensor_desc<128x256xf32>
  return
}
"""


def operations(module, name):
    found = []

    def visit(op):
        if op.name == name:
            found.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def apply(module):
    with schedule_boilerplate() as (schedule, sequence):
        stores = lh_transform.match_op(sequence.bodyTarget, "xegpu.store_nd")
        transform_ext.annotate_epilogue_broadcasts(stores)
        transform.yield_([])
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()


def test(source, load_name, count):
    module = ir.Module.parse(source)
    assert module.operation.verify()
    apply(module)
    (store,) = operations(module, "xegpu.store_nd")
    layout = store.attributes["layout"]
    sliced = ir.Attribute.parse(f"#xegpu.slice<{layout}, dims=[0]>")
    loads = operations(module, load_name)
    broadcasts = operations(module, "vector.broadcast")
    assert len(broadcasts) == count
    for load, broadcast in zip(loads[:count], broadcasts):
        assert load.attributes["layout"] == sliced
        assert broadcast.attributes["layout_result_0"] == layout
        assert broadcast.attributes["layout_operand_0"] == sliced
    for load in loads[count:]:
        assert "layout" not in load.attributes
    before = str(module)
    apply(module)
    assert before == str(module)


with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()
    test(ROPE, "xegpu.load_nd", 2)
    print(
        "Both RoPE table loads use sliced store layouts; unrelated loads are unchanged"
    )
    test(BIAS, "xegpu.load", 1)
    print("Bias loads use the same store-derived broadcast handling")
    stretched = ROPE.replace("<128x64", "<1x64")
    module = ir.Module.parse(stretched)
    apply(module)
    assert all(
        "layout" not in op.attributes for op in operations(module, "xegpu.load_nd")
    )
    barrier = ROPE.replace(
        "%r = arith.addf %m, %sin : vector<4x128x64xbf16>",
        "%sum = arith.addf %m, %sin : vector<4x128x64xbf16>\n"
        "  %r = vector.shape_cast %sum : vector<4x128x64xbf16> to vector<4x128x64xbf16>",
    )
    module = ir.Module.parse(barrier)
    apply(module)
    assert all(
        "layout" not in op.attributes for op in operations(module, "xegpu.load_nd")
    )
    print("Shape-changing paths are not assigned a guessed layout")


# CHECK: Both RoPE table loads use sliced store layouts; unrelated loads are unchanged
# CHECK: Bias loads use the same store-derived broadcast handling
# CHECK: Shape-changing paths are not assigned a guessed layout
