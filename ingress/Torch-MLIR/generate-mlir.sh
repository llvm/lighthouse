#!/usr/bin/env bash

# Command line argument for model to load and MLIR dialect to generate
while getopts "m:d:" opt; do
  case $opt in
    m)
      MODEL=$OPTARG
      ;;
    d)
      DIALECT=$OPTARG
      ;;
    *)
      echo "Usage: $0 [-m model] [-d dialect]"
      exit 1
      ;;
  esac
done
if [ -z "$MODEL" ]; then
  echo "Model not specified. Please provide a model using -m option."
  exit 1
fi
if [ -z "$DIALECT" ]; then
  DIALECT="linalg"
fi

# Enable local virtualenv created by install-virtualenv.sh
if [ ! -d "torch-mlir-venv" ]; then
  echo "Virtual environment not found. Please run install-virtualenv.sh first."
  exit 1
fi
source torch-mlir-venv/bin/activate

# Find script directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Use the Python script to generate MLIR
echo "Generating MLIR for model '$MODEL' with dialect '$DIALECT'..."
python $SCRIPT_DIR/generate-mlir.py --model "$MODEL" --dialect "$DIALECT"
if [ $? -ne 0 ]; then
  echo "Failed to generate MLIR for model '$MODEL'."
  exit 1
fi
