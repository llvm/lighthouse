// RUN: lh-run --stage=%TEST/opt/stages/pipeline-check.yaml %s --entry-point=main --input-shape=16x64xf32xrnd,64x32xf32x1,16x32xf32x0 --print-tensor=3 --seed=123 | FileCheck %s
// RUN: lh-run --stage=%TEST/opt/stages/pipeline-check.yaml %s --entry-point=main --input-shape=16x64xf32xrnd,64x32xf32x1,16x32xf32x0 --benchmark | FileCheck %s --check-prefix=BENCH

// TODO: Benchmark runs need to not have a return value. The original Kernel Bench does return a value.

// Example from Kernel Bench, level 1, kernel 2
module {
  func.func @main(%arg0: tensor<16x64xf32>, %arg1: tensor<64x32xf32>, %out: tensor<16x32xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%out : tensor<16x32xf32>) -> tensor<16x32xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<16x64xf32>, tensor<64x32xf32>) outs(%out : tensor<16x32xf32>) -> tensor<16x32xf32>
    return
  }
}

// CHECK: [32.873924 32.873924 32.873924 32.873924 32.873924 32.873924 32.873924
// CHECK: [31.594452 31.594452 31.594452 31.594452 31.594452 31.594452 31.594452
// CHECK: [33.101555 33.101555 33.101555 33.101555 33.101555 33.101555 33.101555
// CHECK: [31.069946 31.069946 31.069946 31.069946 31.069946 31.069946 31.069946
// CHECK: [30.436699 30.436699 30.436699 30.436699 30.436699 30.436699 30.436699
// CHECK: [32.59776  32.59776  32.59776  32.59776  32.59776  32.59776  32.59776
// CHECK: [30.10459  30.10459  30.10459  30.10459  30.10459  30.10459  30.10459
// CHECK: [32.242332 32.242332 32.242332 32.242332 32.242332 32.242332 32.242332
// CHECK: [31.32454  31.32454  31.32454  31.32454  31.32454  31.32454  31.32454
// CHECK: [34.80585  34.80585  34.80585  34.80585  34.80585  34.80585  34.80585
// CHECK: [33.510525 33.510525 33.510525 33.510525 33.510525 33.510525 33.510525
// CHECK: [35.42333  35.42333  35.42333  35.42333  35.42333  35.42333  35.42333
// CHECK: [31.94259  31.94259  31.94259  31.94259  31.94259  31.94259  31.94259
// CHECK: [32.442146 32.442146 32.442146 32.442146 32.442146 32.442146 32.442146
// CHECK: [30.075504 30.075504 30.075504 30.075504 30.075504 30.075504 30.075504
// CHECK: [26.437485 26.437485 26.437485 26.437485 26.437485 26.437485 26.437485

// BENCH: 100 runs: {{.*}} seconds
