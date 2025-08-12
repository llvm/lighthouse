// 3 Layer MLP auto-generated
// Linalg on tensors, named ops (contract, add, max)
// BENCH_TOTAL_FLOPS: 6445858816

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @entry(%arg0: tensor<256x1024xf32>, %arg1: tensor<1024x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<256x2048xf32>, %arg4: tensor<2048x4096xf32>, %arg5: tensor<4096xf32>, %arg6: tensor<256x4096xf32>, %arg7: tensor<4096x512xf32>, %arg8: tensor<512xf32>, %arg9: tensor<256x512xf32>) -> tensor<256x512xf32> {
    %0 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<256x1024xf32>, tensor<1024x2048xf32>) outs(%arg3 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %1 = tensor.empty() : tensor<256x2048xf32>
    %broadcasted = linalg.broadcast ins(%arg2 : tensor<2048xf32>) outs(%1 : tensor<256x2048xf32>) dimensions = [0] 
    %2 = linalg.add ins(%broadcasted, %0 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%1 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %3 = tensor.empty() : tensor<256x2048xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %5 = linalg.max ins(%2, %4 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%3 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %6 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%5, %arg4 : tensor<256x2048xf32>, tensor<2048x4096xf32>) outs(%arg6 : tensor<256x4096xf32>) -> tensor<256x4096xf32>
    %7 = tensor.empty() : tensor<256x4096xf32>
    %broadcasted_0 = linalg.broadcast ins(%arg5 : tensor<4096xf32>) outs(%7 : tensor<256x4096xf32>) dimensions = [0] 
    %8 = linalg.add ins(%broadcasted_0, %6 : tensor<256x4096xf32>, tensor<256x4096xf32>) outs(%7 : tensor<256x4096xf32>) -> tensor<256x4096xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %9 = tensor.empty() : tensor<256x4096xf32>
    %10 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<256x4096xf32>) -> tensor<256x4096xf32>
    %11 = linalg.max ins(%8, %10 : tensor<256x4096xf32>, tensor<256x4096xf32>) outs(%9 : tensor<256x4096xf32>) -> tensor<256x4096xf32>
    %12 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%11, %arg7 : tensor<256x4096xf32>, tensor<4096x512xf32>) outs(%arg9 : tensor<256x512xf32>) -> tensor<256x512xf32>
    %13 = tensor.empty() : tensor<256x512xf32>
    %broadcasted_2 = linalg.broadcast ins(%arg8 : tensor<512xf32>) outs(%13 : tensor<256x512xf32>) dimensions = [0] 
    %14 = linalg.add ins(%broadcasted_2, %12 : tensor<256x512xf32>, tensor<256x512xf32>) outs(%13 : tensor<256x512xf32>) -> tensor<256x512xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %15 = tensor.empty() : tensor<256x512xf32>
    %16 = linalg.fill ins(%cst_3 : f32) outs(%15 : tensor<256x512xf32>) -> tensor<256x512xf32>
    %17 = linalg.max ins(%14, %16 : tensor<256x512xf32>, tensor<256x512xf32>) outs(%15 : tensor<256x512xf32>) -> tensor<256x512xf32>
    return %17 : tensor<256x512xf32>
  }
}
