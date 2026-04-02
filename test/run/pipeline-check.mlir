// RUN: lh-run --stage=%TEST/opt/stages/pipeline-check.yaml %s --entry-point=entry --input-shape=16x64,64x32,16x32 --input-type=f32

// Example from Kernel Bench, level 1, kernel 2
module {
  func.func @entry(%arg0: tensor<16x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<16x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<16x64xf32>, tensor<64x32xf32>) outs(%1 : tensor<16x32xf32>) -> tensor<16x32xf32>
    return %2 : tensor<16x32xf32>
  }
}
