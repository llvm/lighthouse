// RUN: lh-opt --stage=BufferizationBundle %s | FileCheck %s --check-prefixes=BUFFERIZED
// RUN: lh-opt --stage=BufferizationBundle --stage=canonicalize --stage=MLIRLoweringBundle %s | FileCheck %s --check-prefixes=LINALG_LOWERED
// RUN: lh-opt --stage=BufferizationBundle --stage=canonicalize --stage=MLIRLoweringBundle --stage=canonicalize --stage=LLVMLoweringBundle %s | FileCheck %s --check-prefixes=LLVM_LOWERED
// RUN: lh-opt --stage=%TEST/opt/transforms/pipeline-check.mlir %s | FileCheck %s --check-prefixes=LLVM_LOWERED
// RUN: lh-opt --stage=%TEST/opt/stages/pipeline-check.yaml %s | FileCheck %s --check-prefixes=LLVM_LOWERED
// RUN: lh-opt --stage=%TEST/opt/stages/my-transform.yaml %s | FileCheck %s --check-prefixes=LLVM_LOWERED
// RUN: lh-opt --stage=%TEST/opt/transforms/pipeline-check.py %s | FileCheck %s --check-prefixes=LLVM_LOWERED
// RUN: lh-opt --stage=%TEST/opt/transforms/pipeline-check.py --stage=%TEST/opt/transforms/pipeline-check.py %s | FileCheck %s --check-prefixes=LLVM_LOWERED

// BUFFERIZED-LABEL: func.func @entry
// BUFFERIZED-SAME: memref
// BUFFERIZED: linalg.generic

// LINALG_LOWERED-LABEL: func.func @entry
// LINALG_LOWERED-SAME: memref
// LINALG_LOWERED-NOT: linalg.generic

// LLVM_LOWERED-LABEL: llvm.func @entry
// LLVM_LOWERED-SAME: llvm.ptr
// LLVM_LOWERED-NOT: linalg.generic
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @entry(%arg0: tensor<256x1024xf32>, %arg1: tensor<1024x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<256x2048xf32>) -> tensor<256x2048xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<256x1024xf32>, tensor<1024x2048xf32>) outs(%arg3 : tensor<256x2048xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.mulf %in, %in_2 : f32
      %10 = arith.addf %9, %out : f32
      linalg.yield %10 : f32
    } -> tensor<256x2048xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<2048xf32>) outs(%0 : tensor<256x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<256x2048xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel"]} outs(%1 : tensor<256x2048xf32>) {
    ^bb0(%out: f32):
      %9 = arith.maximumf %out, %cst : f32
      linalg.yield %9 : f32
    } -> tensor<256x2048xf32>
    return %2 : tensor<256x2048xf32>
  }
}
