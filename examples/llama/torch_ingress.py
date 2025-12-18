# RUN: %PYTHON %s

import os
from pathlib import Path

import torch

from lighthouse.ingress.torch import import_from_model
from ref_model import ModelArgs, Transformer

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
model_path = script_dir / "ref_model.py"

model_args = ModelArgs(
    dim=512,
    n_layers=2,
    n_heads=8,
    vocab_size=10000,
    max_batch_size=1,
    max_seq_len=128,
)

model = Transformer(model_args)
sample_input = (torch.randint(0, model_args.vocab_size, (1, model_args.max_seq_len)), 0)

mlir_module_str = import_from_model(
    model, sample_args=sample_input, dialect="linalg-on-tensors"
)

dense_resource_idx = mlir_module_str.find("\n{-#\n  dialect_resources: {")
assert dense_resource_idx != -1

print(mlir_module_str[:dense_resource_idx])
