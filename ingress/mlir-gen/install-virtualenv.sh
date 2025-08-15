#!/usr/bin/env bash

echo "First ensure uv is installed"

python -m pip install uv --upgrade

echo "Preparing the virtual environment"
python -m uv venv mlir-gen-venv --python 3.13

source mlir-gen-venv/bin/activate

echo "Installing mlir-python-bindings and numpy"
uv pip install numpy mlir-python-bindings -f https://makslevental.github.io/wheels
