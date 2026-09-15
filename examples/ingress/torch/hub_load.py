# RUN: %PYTHON %s > /dev/null

"""
Example demonstrating how to load any Hugging Face model to MLIR using Lighthouse
without initializing the model class on the user's side.
"""

import argparse
import sys
from transformers import AutoModel
from lighthouse.ingress.torch import import_from_model


def load_from_hf(model_name: str) -> str:
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    # Exporting to a static graph: drop the KV cache so the output is plain tensors.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    # torch-mlir's importer requires every buffer in the state dict, but Hugging
    # Face registers some (e.g. 'position_ids') as non-persistent; make them all
    # persistent so they can be mapped.
    for submodule in model.modules():
        submodule._non_persistent_buffers_set.clear()
    # Each model exposes correctly-typed example inputs, so no per-model handling.
    return import_from_model(model, sample_args=(), sample_kwargs=model.dummy_inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="hf-internal-testing/tiny-random-BertModel",
        help="Hugging Face model id (e.g. 'google-bert/bert-base-uncased').",
    )
    args = parser.parse_args()

    print(f"Loading model from Hugging Face: {args.model}", file=sys.stderr)
    mlir_module = load_from_hf(args.model)
    print(mlir_module)
