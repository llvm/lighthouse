#!/usr/bin/env bash

# RUN: bash %s

PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$(readlink -fm "$0")")")")")"
CACHE_DIR=$PROJECT_ROOT/cache/ingress/mlir_gen

LAYERS=1024,2048,4096,512

echo "NOTE: dumping output in: $CACHE_DIR"
mkdir -p $CACHE_DIR
python -m lighthouse.ingress.mlir_gen --output named --kernel args --layers $LAYERS --batch 256 --bias --relu > $CACHE_DIR/linalg-named-3layer-mlp.mlir
# RUN: cat %CACHE/ingress/mlir_gen/linalg-named-3layer-mlp.mlir | FileCheck %s --check-prefix CHECK-NAMED
# CHECK-NAMED-DAG: linalg.matmul
# CHECK-NAMED-NOT: linalg.contract
# CHECK-NAMED-NOT: linalg.generic 

python -m lighthouse.ingress.mlir_gen --output einsum --kernel args --layers $LAYERS --batch 256 --bias --relu > $CACHE_DIR/linalg-einsum-3layer-mlp.mlir
# RUN: cat %CACHE/ingress/mlir_gen/linalg-einsum-3layer-mlp.mlir | FileCheck %s --check-prefix CHECK-EINSUM
# CHECK-EINSUM-NOT: linalg.matmul
# CHECK-EINSUM-DAG: linalg.contract
# CHECK-EINSUM-NOT: linalg.generic 

python -m lighthouse.ingress.mlir_gen --output generic --kernel args --layers $LAYERS --batch 256 --bias --relu > $CACHE_DIR/linalg-generic-3layer-mlp.mlir
# RUN: cat %CACHE/ingress/mlir_gen/linalg-generic-3layer-mlp.mlir | FileCheck %s --check-prefix CHECK-GENERIC
# CHECK-GENERIC-NOT: linalg.matmul
# CHECK-GENERIC-NOT: linalg.contract
# CHECK-GENERIC-DAG: linalg.generic 
