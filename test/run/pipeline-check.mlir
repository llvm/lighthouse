// RUN: lh-run --stage=%TEST/opt/stages/pipeline-check.yaml %s --entry-point=main --input-shape=16x64,64x32,16x32 --input-type=f32 --print-tensor=3 --seed=123 | FileCheck %s
// RUN: lh-run --stage=%TEST/opt/stages/pipeline-check.yaml %s --entry-point=main --input-shape=16x64,64x32,16x32 --input-type=f32 --benchmark | FileCheck %s --check-prefix=BENCH

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

// CHECK: [15.608599  16.920544  16.718403  17.27879   19.556364  18.397987
// CHECK: [16.873146  15.012256  14.758867  17.32999   17.961376  18.063625
// CHECK: [16.36967   15.688863  14.767131  19.238678  18.06136   19.028954
// CHECK: [15.088741  14.658587  15.7023945 17.221188  19.370567  16.725937
// CHECK: [15.462397  14.33569   13.886818  16.581299  17.575882  17.948814
// CHECK: [16.311558  15.689352  16.964294  18.590815  17.93093   18.097647
// CHECK: [16.538208  15.174422  13.982351  15.896425  16.342865  18.651571
// CHECK: [15.993649  16.75138   15.838758  16.435078  18.54076   17.443964
// CHECK: [15.44256   15.871723  15.10429   16.840515  18.197378  17.769167
// CHECK: [18.036882  16.529142  16.79706   18.977081  19.803997  21.10305
// CHECK: [16.196892  16.404594  15.761157  20.25662   18.502361  19.456942
// CHECK: [18.480625  17.17477   16.98961   20.149424  18.711601  19.232689
// CHECK: [15.480138  16.104235  14.974307  17.890814  18.092493  18.642317
// CHECK: [17.207788  14.164753  17.061205  17.912767  17.775291  18.86269
// CHECK: [15.176749  13.757328  14.85155   15.945061  18.30237   17.464027
// CHECK: [13.508607  13.725334  13.715409  14.689902  14.14056   14.543454

// BENCH: 100 runs: {{.*}} seconds
