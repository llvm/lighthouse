# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


def apply_filter(payload: str, name: str):
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(payload)
        with schedule_boilerplate() as (sched, named_seq):
            candidates = lh_transform.match_op(
                named_seq.bodyTarget, structured.MatchInterfaceEnum.LinalgOp
            )
            filtered = transform_ext.filter_non_contraction_reductions(candidates)
            transform.print_(target=filtered, name=name)
            transform.yield_()
        sched.body.operations[0].apply(module.operation)


# Only the plain row reduction is kept: contractions (named and generic),
# pooling, fills and elementwise ops are rejected.
MIXED = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#ma = affine_map<(d0, d1, d2) -> (d0, d2)>
#mb = affine_map<(d0, d1, d2) -> (d2, d1)>
#mc = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%a: tensor<8x16xf32>, %b: tensor<16x8xf32>, %c: tensor<8x8xf32>,
      %r: tensor<8xf32>, %img: tensor<1x8x8x4xf32>, %win: tensor<2x2xf32>,
      %pool: tensor<1x4x4x4xf32>)
      -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8xf32>, tensor<1x4x4x4xf32>) {
    %cst = arith.constant 0.0 : f32
    %f = linalg.fill ins(%cst : f32) outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<8x16xf32>, tensor<16x8xf32>)
        outs(%f : tensor<8x8xf32>) -> tensor<8x8xf32>
    %gmm = linalg.generic {indexing_maps = [#ma, #mb, #mc],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<8x16xf32>, tensor<16x8xf32>) outs(%c : tensor<8x8xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %p = arith.mulf %x, %y : f32
      %s = arith.addf %p, %o : f32
      linalg.yield %s : f32
    } -> tensor<8x8xf32>
    %ew = linalg.elementwise <add> ins(%mm, %gmm : tensor<8x8xf32>, tensor<8x8xf32>)
        outs(%c : tensor<8x8xf32>) -> tensor<8x8xf32>
    %red = linalg.generic {indexing_maps = [#id, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%a : tensor<8x16xf32>) outs(%r : tensor<8xf32>) {
    ^bb0(%x: f32, %o: f32):
      %s = arith.maximumf %x, %o : f32
      linalg.yield %s : f32
    } -> tensor<8xf32>
    %pl = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>,
        strides = dense<2> : tensor<2xi64>}
        ins(%img, %win : tensor<1x8x8x4xf32>, tensor<2x2xf32>)
        outs(%pool : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    return %ew, %gmm, %red, %pl
        : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8xf32>, tensor<1x4x4x4xf32>
  }
}
"""

# CHECK-LABEL: IR printer: MIXED
# CHECK-NEXT: linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "reduction"]
# CHECK-SAME: ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8xf32>)
# CHECK-NOT: linalg.matmul
# CHECK-NOT: linalg.pooling_nhwc_max
# CHECK-NOT: linalg.fill
# CHECK-NOT: linalg.elementwise
apply_filter(MIXED, name="MIXED")


# Edge cases: a row max whose second input does not read the reduced dim is
# kept; a matvec (named or generic, both inputs read the reduced dim) and a
# multi-output argmax are rejected.
EDGE = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#col = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%a: tensor<8x16xf32>, %v: tensor<16xf32>, %s: tensor<8xf32>,
      %r: tensor<8xf32>, %i: tensor<8xi32>)
      -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xi32>) {
    %shifted = linalg.generic {indexing_maps = [#id, #row, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%a, %s : tensor<8x16xf32>, tensor<8xf32>) outs(%r : tensor<8xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %p = arith.subf %x, %y : f32
      %m = arith.maximumf %p, %o : f32
      linalg.yield %m : f32
    } -> tensor<8xf32>
    %mv = linalg.matvec ins(%a, %v : tensor<8x16xf32>, tensor<16xf32>)
        outs(%r : tensor<8xf32>) -> tensor<8xf32>
    %gmv = linalg.generic {indexing_maps = [#id, #col, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%a, %v : tensor<8x16xf32>, tensor<16xf32>) outs(%r : tensor<8xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %p = arith.mulf %x, %y : f32
      %m = arith.addf %p, %o : f32
      linalg.yield %m : f32
    } -> tensor<8xf32>
    %am:2 = linalg.generic {indexing_maps = [#id, #row, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%a : tensor<8x16xf32>) outs(%r, %i : tensor<8xf32>, tensor<8xi32>) {
    ^bb0(%x: f32, %o: f32, %oi: i32):
      %j = linalg.index 1 : index
      %ji = arith.index_cast %j : index to i32
      %gt = arith.cmpf ogt, %x, %o : f32
      %m = arith.select %gt, %x, %o : f32
      %mi = arith.select %gt, %ji, %oi : i32
      linalg.yield %m, %mi : f32, i32
    } -> (tensor<8xf32>, tensor<8xi32>)
    return %shifted, %mv, %gmv, %am#0, %am#1
        : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xi32>
  }
}
"""

# CHECK-LABEL: IR printer: EDGE
# CHECK-NEXT: linalg.generic
# CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<8x16xf32>, tensor<8xf32>)
# CHECK-NOT: linalg.matvec
# CHECK-NOT: tensor<16xf32>
# CHECK-NOT: tensor<8xi32>
apply_filter(EDGE, name="EDGE")
