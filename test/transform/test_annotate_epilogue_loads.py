# RUN: %PYTHON %s | FileCheck %s

"""Store-derived ordinary and broadcast epilogue load layouts."""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.xegpu.annotate_layouts_schedule import annotate_elementwise


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

ELEMENTWISE = """
gpu.module @kernels {
  gpu.func @elementwise(%table: !xegpu.tensor_desc<16x32xf32>,
                       %input: !xegpu.tensor_desc<1x16x32xf32>,
                       %out: !xegpu.tensor_desc<1x16x32xf32>)
      kernel attributes {known_block_size = array<i32: 16, 1, 1>} {
    %a = xegpu.load_nd %table[0,0]
      : !xegpu.tensor_desc<16x32xf32> -> vector<16x32xf32>
    %b = xegpu.load_nd %input[0,0,0]
      : !xegpu.tensor_desc<1x16x32xf32> -> vector<1x16x32xf32>
    %broadcast = vector.broadcast %a
      : vector<16x32xf32> to vector<1x16x32xf32>
    %result = arith.addf %b, %broadcast : vector<1x16x32xf32>
    xegpu.store_nd %result, %out[0,0,0]
      : vector<1x16x32xf32>, !xegpu.tensor_desc<1x16x32xf32>
    gpu.return
  }
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
        transform_ext.annotate_epilogue_loads(stores)
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


def test_elementwise_schedule(source, has_broadcast, expected_layout=None):
    module = ir.Module.parse(source)
    with schedule_boilerplate() as (schedule, sequence):
        gpu_func = lh_transform.match_op(sequence.bodyTarget, "gpu.func")
        annotate_elementwise(gpu_func)
        transform.yield_()
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    (store,) = operations(module, "xegpu.store_nd")
    layout = store.attributes["layout"]
    if expected_layout is not None:
        assert layout == ir.Attribute.parse(expected_layout)
    loads = operations(module, "xegpu.load_nd")
    if has_broadcast:
        sliced = ir.Attribute.parse(f"#xegpu.slice<{layout}, dims=[0]>")
        (broadcast,) = operations(module, "vector.broadcast")
        assert loads[0].attributes["layout"] == sliced
        assert broadcast.attributes["layout_result_0"] == layout
        assert broadcast.attributes["layout_operand_0"] == sliced
        loads = loads[1:]
    for load in loads:
        assert load.attributes["layout"] == layout
    before = str(module)
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    assert str(module) == before


def test_ordinary_scattered_load():
    source = BIAS.replace(
        "%projection: vector<128x256xf32>",
        "%projection: vector<256xf32>",
    ).replace("128x256", "256")
    source = (
        source.replace(
            "sg_layout=[4,8], sg_data=[32,32], inst_data=[8,16]",
            "sg_layout=[8], sg_data=[32], inst_data=[16]",
        )
        .replace(
            "%b = vector.broadcast %a : vector<256xf32> to vector<256xf32>",
            "%b = math.absf %a : vector<256xf32>",
        )
        .replace("%out[0,0]", "%out[0]")
    )
    module = ir.Module.parse(source)
    apply(module)
    (store,) = operations(module, "xegpu.store_nd")
    (load,) = operations(module, "xegpu.load")
    assert load.attributes["layout"] == store.attributes["layout"]
    print("Ordinary scattered loads inherit the store layout through math ops")


def test_conflicting_layout():
    module = ir.Module.parse(ROPE)
    loads = operations(module, "xegpu.load_nd")
    loads[1].attributes["layout"] = ir.Attribute.parse(
        "#xegpu.layout<sg_layout=[1,1], sg_data=[128,64]>"
    )
    before = str(module)
    try:
        apply(module)
    except ValueError as error:
        assert "conflicting broadcast/load layout" in str(error), str(error)
    else:
        raise AssertionError("Expected incompatible broadcast load layout to fail")
    assert str(module) == before
    print("Conflicting layouts still fail without partial annotations")


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
    test_elementwise_schedule(ELEMENTWISE, has_broadcast=True)
    print("Elementwise schedule preserves broadcast slices and anchors ordinary loads")
    ordinary = (
        ELEMENTWISE.replace("16x32", "1x16x32", 1)
        .replace("%table[0,0]", "%table[0,0,0]")
        .replace(
            ": !xegpu.tensor_desc<16x32xf32> -> vector<16x32xf32>",
            ": !xegpu.tensor_desc<1x16x32xf32> -> vector<1x16x32xf32>",
        )
        .replace(
            "%broadcast = vector.broadcast %a\n"
            "      : vector<16x32xf32> to vector<1x16x32xf32>",
            "%broadcast = math.absf %a : vector<1x16x32xf32>",
        )
    )
    test_elementwise_schedule(ordinary, has_broadcast=False)
    print("Ordinary elementwise loads inherit the store layout without broadcasts")
    identity = ordinary.replace(
        "%result = arith.addf %b, %broadcast : vector<1x16x32xf32>", ""
    ).replace("xegpu.store_nd %result", "xegpu.store_nd %b")
    module = ir.Module.parse(identity)
    with schedule_boilerplate() as (schedule, sequence):
        gpu_func = lh_transform.match_op(sequence.bodyTarget, "gpu.func")
        annotate_elementwise(gpu_func)
        transform.yield_()
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    unused, used = operations(module, "xegpu.load_nd")
    (store,) = operations(module, "xegpu.store_nd")
    assert "layout" not in unused.attributes
    assert used.attributes["layout"] == store.attributes["layout"]
    print("Directly stored loads are annotated; unrelated loads remain unchanged")
    test_ordinary_scattered_load()
    test_conflicting_layout()
    large = ELEMENTWISE.replace("16x32", "128x256").replace(
        "known_block_size = array<i32: 16, 1, 1>",
        "known_block_size = array<i32: 512, 1, 1>",
    )
    test_elementwise_schedule(
        large,
        has_broadcast=True,
        expected_layout="#xegpu.layout<sg_layout=[1,4,8], sg_data=[1,32,32]>",
    )
    print("Large elementwise stores and broadcast loads use a 4x8 SG grid")


# CHECK: Both RoPE table loads use sliced store layouts; unrelated loads are unchanged
# CHECK: Bias loads use the same store-derived broadcast handling
# CHECK: Shape-changing paths are not assigned a guessed layout
# CHECK: Elementwise schedule preserves broadcast slices and anchors ordinary loads
# CHECK: Ordinary elementwise loads inherit the store layout without broadcasts
# CHECK: Directly stored loads are annotated; unrelated loads remain unchanged
# CHECK: Ordinary scattered loads inherit the store layout through math ops
# CHECK: Conflicting layouts still fail without partial annotations
# CHECK: Large elementwise stores and broadcast loads use a 4x8 SG grid
