source mlir-gen-venv/bin/activate

LAYERS=1024,2048,4096,512

mkdir -p cache
python -m mlir_gen --output named --kernel args --layers $LAYERS --batch 256 --bias --relu > cache/linalg-named-3layer-mlp.mlir
python -m mlir_gen --output einsum --kernel args --layers $LAYERS --batch 256 --bias --relu > cache/linalg-einsum-3layer-mlp.mlir
python -m mlir_gen --output generic --kernel args --layers $LAYERS --batch 256 --bias --relu > cache/linalg-generic-3layer-mlp.mlir
