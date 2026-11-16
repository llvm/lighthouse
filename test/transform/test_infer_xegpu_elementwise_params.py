# RUN: %PYTHON %s | FileCheck %s

"""Test elementwise workgroup tile inference."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
from lighthouse.dialects.transform.transform_ext.ops.infer_xegpu_elementwise_params import (
    elementwise_sg_tile,
    elementwise_wg_tile,
)


PAYLOAD = """
module {
  func.func @payload(%arg: tensor<4x32x512x64xf32>,
                     %out: tensor<4x32x512x64xf32>)
                     -> tensor<4x32x512x64xf32> {
    %result = linalg.generic {
        indexing_maps=[affine_map<(b,h,s,d)->(b,h,s,d)>,
                       affine_map<(b,h,s,d)->(b,h,s,d)>],
        iterator_types=["parallel","parallel","parallel","parallel"]}
      ins(%arg : tensor<4x32x512x64xf32>)
      outs(%out : tensor<4x32x512x64xf32>) {
    ^bb0(%a: f32, %o: f32):
      linalg.yield %a : f32
    } -> tensor<4x32x512x64xf32>
    return %result : tensor<4x32x512x64xf32>
  }
}
"""


def build_schedule():
    with schedule_boilerplate() as (schedule, named_sequence):
        generic = lh_transform.match_op(named_sequence.bodyTarget, "linalg.generic")
        generic = transform_ext.extract_handle(generic, 0)
        wg_tile, sg_tile = transform_ext.infer_xegpu_elementwise_params(generic)
        sg_count = transform.num_associations(transform.AnyParamType.get(), sg_tile)
        zero = transform.param_constant(
            transform.AnyParamType.get(),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0),
        )
        transform_ext.param_cmp_eq(sg_count, zero)
        structured.TileUsingForallOp(generic, tile_sizes=wg_tile)
        transform.yield_()
    return schedule


def test():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(PAYLOAD)
        schedule = build_schedule()
        schedule.body.operations[0].apply(payload.operation)
        print(payload)


def test_permuted_result():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse("""
module {
  func.func @payload(%arg: tensor<512x64xf32>, %out: tensor<64x512xf32>)
      -> tensor<64x512xf32> {
    %result = linalg.generic {
        indexing_maps=[affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(j,i)>],
        iterator_types=["parallel","parallel"]}
      ins(%arg : tensor<512x64xf32>) outs(%out : tensor<64x512xf32>) {
    ^bb0(%a: f32, %o: f32):
      linalg.yield %a : f32
    } -> tensor<64x512xf32>
    return %result : tensor<64x512xf32>
  }
}
""")
        schedule = build_schedule()
        schedule.body.operations[0].apply(payload.operation)
        assert payload.operation.verify()
        loops = []

        def visit(op):
            if op.name == "scf.forall":
                loops.append(op.opview)
            return ir.WalkResult.ADVANCE

        payload.operation.walk(visit)
        assert len(loops) == 1
        assert list(loops[0].staticUpperBound) == [2]
        print("Permuted output tiles are returned in iteration order")


def test_unsupported_shapes():
    for shape, dimensions, iterators in (
        ("", "", ""),
        ("0x64x", "i,j", '"parallel","parallel"'),
    ):
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()
            payload = ir.Module.parse(f"""
module {{
  func.func @payload(%arg: tensor<{shape}f32>, %out: tensor<{shape}f32>)
      -> tensor<{shape}f32> {{
    %result = linalg.generic {{
        indexing_maps=[affine_map<({dimensions})->({dimensions})>,
                       affine_map<({dimensions})->({dimensions})>],
        iterator_types=[{iterators}]}}
      ins(%arg : tensor<{shape}f32>) outs(%out : tensor<{shape}f32>) {{
    ^bb0(%a: f32, %o: f32):
      linalg.yield %a : f32
    }} -> tensor<{shape}f32>
    return %result : tensor<{shape}f32>
  }}
}}
""")
            try:
                schedule = build_schedule()
                schedule.body.operations[0].apply(payload.operation)
            except ValueError as error:
                assert "positive-rank" in str(error), str(error)
            else:
                raise AssertionError("Expected unsupported shape to fail")
    print("Scalar and empty elementwise shapes fail without Python exceptions")


def test_tile_shapes():
    for shape, expected_wg, expected_sg in (
        ([1024, 1024], [128, 256], [32, 32]),
        ([4, 32, 512, 64], [1, 1, 128, 0], [1, 1, 32, 32]),
        ([1, 512, 64, 1], [0, 128, 0, 0], [1, 32, 32, 1]),
        ([16, 16], [0, 0], [16, 16]),
        ([1024], [256], [32]),
        ([1, 1], [0, 0], [1, 1]),
        ([384, 768], [128, 256], [32, 32]),
        ([96, 192], [32, 64], [32, 32]),
        ([17, 31], [1, 1], [1, 1]),
        ([131, 259], [1, 1], [1, 1]),
    ):
        wg = elementwise_wg_tile(shape)
        assert wg == expected_wg, (shape, wg)
        vector_shape = [size or extent for size, extent in zip(wg, shape)]
        sg = elementwise_sg_tile(vector_shape)
        assert sg == expected_sg, (shape, sg)
        assert all(w % s == 0 for w, s in zip(vector_shape, sg))
    print("WG and SG tiles select non-unit axes and divide small and irregular shapes")


def test_vector_anchors():
    source = """
func.func @memory(%value: vector<128x256xf32>,
                  %dest: memref<128x256xf32>,
                  %desc: !xegpu.tensor_desc<128x256xf32>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %value, %dest[%c0, %c0]
    {in_bounds = [true, true]} : vector<128x256xf32>, memref<128x256xf32>
  xegpu.store_nd %value, %desc[0,0]
    : vector<128x256xf32>, !xegpu.tensor_desc<128x256xf32>
  return
}
"""
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, sequence):
            for name in ("vector.transfer_write", "xegpu.store_nd"):
                anchor = lh_transform.match_op(sequence.bodyTarget, name)
                wg, sg = transform_ext.infer_xegpu_elementwise_params(anchor)
                threads = transform_ext.compute_num_threads(wg, sg, base=16)
                expected = transform.param_constant(
                    transform.AnyParamType.get(),
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 512),
                )
                transform_ext.param_cmp_eq(threads, expected)
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
    print("Transfer writes and XeGPU stores agree on 512 threads for a 128x256 tile")


# CHECK: scf.forall
# CHECK-SAME: in (4, 32, 4)
# CHECK: tensor.extract_slice
# CHECK-SAME: [1, 1, 128, 64]


test()
test_permuted_result()
test_unsupported_shapes()
test_tile_shapes()
test_vector_anchors()

# CHECK: Permuted output tiles are returned in iteration order
# CHECK: Scalar and empty elementwise shapes fail without Python exceptions
# CHECK: WG and SG tiles select non-unit axes and divide small and irregular shapes
# CHECK: Transfer writes and XeGPU stores agree on 512 threads for a 128x256 tile
