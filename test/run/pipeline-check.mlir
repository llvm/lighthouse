// RUN: lh-run --stage=%TEST/opt/stages/pipeline-check.yaml %s --entry-point=main --input-shape=16x64,64x32,16x32 --input-type=f32 --print-tensor=3 --seed=123 | FileCheck %s
// LATER: lh-run --stage=%TEST/opt/stages/pipeline-check.yaml %s --entry-point=main --input-shape=16x64,64x32,16x32 --input-type=f32 --benchmark | FileCheck %s --check-prefix=BENCH

// Example from Kernel Bench, level 1, kernel 2
module {
  func.func @main(%arg0: tensor<16x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<16x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<16x64xf32>, tensor<64x32xf32>) outs(%1 : tensor<16x32xf32>) -> tensor<16x32xf32>
    return %2 : tensor<16x32xf32>
  }
}

// CHECK: [9.84787568e-02 9.50287759e-01 6.18466735e-01 9.68179479e-03
// CHECK: [3.42579514e-01 1.16832502e-01 2.09235977e-02 6.68728769e-01
// CHECK: [3.74255300e-01 3.93596381e-01 3.23667049e-01 8.89222383e-01
// CHECK: [8.66959915e-02 4.87389416e-01 9.04625416e-01 4.39447194e-01
// CHECK: [2.99748063e-01 7.50455856e-01 2.91555196e-01 5.16348243e-01
// CHECK: [4.42018539e-01 7.98953593e-01 3.86109084e-01 7.41468191e-01
// CHECK: [8.85862172e-01 4.61767823e-01 5.93744159e-01 5.69888473e-01
// CHECK: [1.24294916e-02 5.09863436e-01 9.81862009e-01 2.32334837e-01
// CHECK: [3.94061446e-01 3.15508127e-01 9.17021036e-01 3.31225961e-01
// CHECK: [6.22423738e-02 6.46480322e-01 7.22611070e-01 1.42622232e-01
// CHECK: [5.09627938e-01 3.21896702e-01 2.14922383e-01 9.59206223e-01
// CHECK: [5.77826977e-01 4.78566974e-01 7.15224564e-01 9.88330960e-01
// CHECK: [8.92309606e-01 6.29615784e-01 7.03984976e-01 8.82227242e-01
// CHECK: [6.45988047e-01 9.74168777e-02 1.38656780e-01 8.75165835e-02
// CHECK: [3.26380044e-01 3.80012952e-02 4.45450991e-01 1.40048102e-01
// CHECK: [6.25901520e-01 6.27187967e-01 9.39385116e-01 9.77425352e-02
