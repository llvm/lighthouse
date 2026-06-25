# RUN: %PYTHON %s | FileCheck %s

"""Tests for the lighthouse transform_ext dialect operations."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured as transform_structured
import lighthouse.dialects as lh_dialects
from lighthouse.dialects.transform import transform_ext


def run(f):
    print("Test: ", f.__name__, flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load(
            reload=True
        )  # reload across @run-s to ensure interfaces are registered for new contexts

        mod = ir.Module.create()
        mod.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
        with ir.InsertionPoint(mod.body):
            named_seq = transform.named_sequence(
                "main", [transform.AnyOpType.get()], []
            )
            any_op = transform.AnyOpType.get()
            with ir.InsertionPoint(named_seq.body):
                func_funcs = transform_structured.structured_match(
                    any_op, named_seq.bodyTarget, ops={"func.func"}
                )
                payload_handle = transform.get_parent_op(any_op, func_funcs)

                f(payload_handle)
                transform.yield_()

            named_seq.verify()

            try:
                payload_mod = payload()
                named_seq.apply(payload_mod)
            except Exception as e:
                print(f"Caught exception: {e}", flush=True)


def payload():
    mod = ir.Module.parse(
        r"""
#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map5 = affine_map<(d0) -> (d0 floordiv 32)>
#map6 = affine_map<(d0) -> (d0 mod 32)>
func.func @main(%arg0: memref<4096x4096xbf16>, %arg1: memref<4096x4096xbf16>, %arg2: memref<4096x4096xbf16>) attributes {llvm.emit_c_interface} {
%c16 = arith.constant 16 : index
%cst = arith.constant 0.000000e+00 : bf16
%cst_0 = arith.constant dense<0.000000e+00> : vector<1x8xf32>
%0 = ub.poison : f32
%c8 = arith.constant 8 : index
%c4096 = arith.constant 4096 : index
%1 = ub.poison : bf16
%c32 = arith.constant 32 : index
%c1 = arith.constant 1 : index
%c128 = arith.constant 128 : index
%c0 = arith.constant 0 : index
%cst_1 = arith.constant 0.000000e+00 : f32
%alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128x32x32xbf16>
%alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<128x128x32x32xbf16>
scf.for %arg3 = %c0 to %c128 step %c1 {
    %2 = affine.apply #map(%arg3)
    scf.for %arg4 = %c0 to %c128 step %c1 {
    %3 = affine.apply #map(%arg4)
    %subview = memref.subview %arg1[%2, %3] [32, 32] [1, 1] : memref<4096x4096xbf16> to memref<32x32xbf16, strided<[4096, 1], offset: ?>>
    %expand_shape = memref.expand_shape %subview [[0, 1], [2, 3]] output_shape [1, 32, 1, 32] : memref<32x32xbf16, strided<[4096, 1], offset: ?>> into memref<1x32x1x32xbf16, strided<[131072, 4096, 32, 1], offset: ?>>
    %4 = vector.transfer_read %expand_shape[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true], permutation_map = #map1} : memref<1x32x1x32xbf16, strided<[131072, 4096, 32, 1], offset: ?>>, vector<32x32xbf16>
    vector.transfer_write %4, %alloc_2[%arg3, %arg4, %c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<128x128x32x32xbf16>
    }
}
scf.for %arg3 = %c0 to %c128 step %c1 {
    %2 = affine.apply #map(%arg3)
    scf.for %arg4 = %c0 to %c128 step %c1 {
    %3 = affine.apply #map(%arg4)
    %subview = memref.subview %arg2[%3, %2] [32, 32] [1, 1] : memref<4096x4096xbf16> to memref<32x32xbf16, strided<[4096, 1], offset: ?>>
    %expand_shape = memref.expand_shape %subview [[0, 1], [2, 3]] output_shape [1, 32, 1, 32] : memref<32x32xbf16, strided<[4096, 1], offset: ?>> into memref<1x32x1x32xbf16, strided<[131072, 4096, 32, 1], offset: ?>>
    %4 = vector.transfer_read %expand_shape[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true], permutation_map = #map1} : memref<1x32x1x32xbf16, strided<[131072, 4096, 32, 1], offset: ?>>, vector<32x32xbf16>
    vector.transfer_write %4, %alloc[%arg3, %arg4, %c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<128x128x32x32xbf16>
    }
}
%alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<128x128x32x32xf32>
scf.forall (%arg3, %arg4) in (128, 128) {
    %subview = memref.subview %alloc_3[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<128x128x32x32xf32> to memref<1x1x32x32xf32, strided<[131072, 1024, 32, 1], offset: ?>>
    %subview_4 = memref.subview %subview[0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x32x32xf32, strided<[131072, 1024, 32, 1], offset: ?>> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    scf.for %arg5 = %c0 to %c32 step %c1 {
    scf.for %arg6 = %c0 to %c32 step %c8 {
        vector.transfer_write %cst_0, %subview_4[%arg5, %arg6] {in_bounds = [true, true]} : vector<1x8xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
    }
    }
    %2 = vector.transfer_read %subview_4[%c0, %c0], %cst_1 {in_bounds = [true, true]} : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<16x16xf32>
    %3 = vector.transfer_read %subview_4[%c0, %c16], %cst_1 {in_bounds = [true, true]} : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<16x16xf32>
    %4 = vector.transfer_read %subview_4[%c16, %c0], %cst_1 {in_bounds = [true, true]} : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<16x16xf32>
    %5 = vector.transfer_read %subview_4[%c16, %c16], %cst_1 {in_bounds = [true, true]} : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<16x16xf32>
    %6:4 = scf.for %arg5 = %c0 to %c128 step %c1 iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5) -> (vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>) {
    %7 = vector.transfer_read %alloc_2[%arg3, %arg5, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<128x128x32x32xbf16>, vector<1x16x32xbf16>
    %8 = vector.transfer_read %alloc[%arg4, %arg5, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<128x128x32x32xbf16>, vector<1x32x16xbf16>
    %9 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %7, %8, %arg6 : vector<1x16x32xbf16>, vector<1x32x16xbf16> into vector<16x16xf32>
    %10 = vector.transfer_read %alloc[%arg4, %arg5, %c0, %c16], %cst {in_bounds = [true, true, true]} : memref<128x128x32x32xbf16>, vector<1x32x16xbf16>
    %11 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %7, %10, %arg7 : vector<1x16x32xbf16>, vector<1x32x16xbf16> into vector<16x16xf32>
    %12 = vector.transfer_read %alloc_2[%arg3, %arg5, %c16, %c0], %cst {in_bounds = [true, true, true]} : memref<128x128x32x32xbf16>, vector<1x16x32xbf16>
    %13 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %12, %8, %arg8 : vector<1x16x32xbf16>, vector<1x32x16xbf16> into vector<16x16xf32>
    %14 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %12, %10, %arg9 : vector<1x16x32xbf16>, vector<1x32x16xbf16> into vector<16x16xf32>
    scf.yield %9, %11, %13, %14 : vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>
    }
    vector.transfer_write %6#3, %subview_4[%c16, %c16] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
    vector.transfer_write %6#2, %subview_4[%c16, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
    vector.transfer_write %6#1, %subview_4[%c0, %c16] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
    vector.transfer_write %6#0, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
}
scf.for %arg3 = %c0 to %c128 step %c1 {
    scf.for %arg4 = %c0 to %c128 step %c8 {
    %2 = vector.transfer_read %alloc_3[%arg3, %arg4, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : memref<128x128x32x32xf32>, vector<1x8x32x32xf32>
    %3 = arith.truncf %2 : vector<1x8x32x32xf32> to vector<1x8x32x32xbf16>
    vector.transfer_write %3, %alloc[%arg3, %arg4, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x8x32x32xbf16>, memref<128x128x32x32xbf16>
    }
}
scf.for %arg3 = %c0 to %c4096 step %c32 {
    %2 = affine.apply #map5(%arg3)
    scf.for %arg4 = %c0 to %c4096 step %c32 {
    %3 = affine.apply #map5(%arg4)
    %subview = memref.subview %alloc[%2, %3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<128x128x32x32xbf16> to memref<1x1x32x32xbf16, strided<[131072, 1024, 32, 1], offset: ?>>
    %subview_4 = memref.subview %arg0[%arg3, %arg4] [32, 32] [1, 1] : memref<4096x4096xbf16> to memref<32x32xbf16, strided<[4096, 1], offset: ?>>
    scf.for %arg5 = %c0 to %c32 step %c1 {
        %4 = affine.apply #map5(%arg5)
        %5 = affine.apply #map6(%arg5)
        %subview_5 = memref.subview %subview[%4, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x32x32xbf16, strided<[131072, 1024, 32, 1], offset: ?>> to memref<1x1x32x32xbf16, strided<[131072, 1024, 32, 1], offset: ?>>
        %subview_6 = memref.subview %subview_5[0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x32x32xbf16, strided<[131072, 1024, 32, 1], offset: ?>> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
        %subview_7 = memref.subview %subview_6[%5, 0] [1, 32] [1, 1] : memref<32x32xbf16, strided<[32, 1], offset: ?>> to memref<1x32xbf16, strided<[32, 1], offset: ?>>
        %subview_8 = memref.subview %subview_4[%arg5, 0] [1, 32] [1, 1] : memref<32x32xbf16, strided<[4096, 1], offset: ?>> to memref<1x32xbf16, strided<[4096, 1], offset: ?>>
        memref.copy %subview_7, %subview_8 : memref<1x32xbf16, strided<[32, 1], offset: ?>> to memref<1x32xbf16, strided<[4096, 1], offset: ?>>
    }
    }
}
memref.dealloc %alloc : memref<128x128x32x32xbf16>
memref.dealloc %alloc_2 : memref<128x128x32x32xbf16>
memref.dealloc %alloc_3 : memref<128x128x32x32xf32>
return
}
"""
    )

    return mod


# def payload():
#     mod = ir.Module.parse(
#         r"""
# #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
# func.func @main(%arg0: memref<20x4096x4096xbf16>) -> vector<32x32xbf16> {
#     %c8 = arith.constant 8 : index
#     %c16 = arith.constant 16 : index
#     %cst = arith.constant 0.000000e+00 : bf16
#     %0 = vector.transfer_read %arg0[%c16, %c8, %c16], %cst {in_bounds = [true, true], permutation_map = #map1} : memref<20x4096x4096xbf16>, vector<32x32xbf16>
#     return %0 : vector<32x32xbf16>
# }
#     """
#     )

#     return mod


# def payload():
#     mod = ir.Module.parse(
#         r"""
# #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
# func.func @main(%arg0: memref<32x4096x4096xbf16>, %arg1: vector<32x32xbf16>) {
#     %c8 = arith.constant 8 : index
#     %c16 = arith.constant 16 : index
#     vector.transfer_write %arg1, %arg0[%c16, %c8, %c16] {in_bounds = [true, true], permutation_map = #map1} : vector<32x32xbf16>, memref<32x4096x4096xbf16>
#     return
# }
#     """
#     )

#     return mod


# CHECK-LABEL: Test: test_move_offsets_to_subview
@run
def test_move_offsets_to_subview(payload_handle):
    """Exercise GetNamedAttributeOp and ParamCmpEqOp:

    * by comparing the payload's arith.constant value vs 42 and 0.
    """

    any_op = transform.AnyOpType.get()
    transfers = transform_structured.structured_match(
        any_op, payload_handle, ops={"vector.transfer_read"}
    )
    transform_ext.move_offsets_to_subview(transfers)
    transform.print_()
