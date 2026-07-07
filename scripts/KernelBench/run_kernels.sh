#!/usr/bin/env bash

# Script that runs the kernel bench script for a list of kernels,
# saves the logs, assembly and object files in the current directory.
#
# You can run this from anywhere, but the execution must run inside the
# git repo. To avoid poluting the git repo, you can run on the 'build'
# directory, since it's currently being ignored by git.
#
# cd build && ../scripts/KernelBench/run_kernels.sh level1/1_...
#
# Arguments with -- prefix are passed to the kernel bench script.
# Everything else is treated as a kernel name, with the format levelN/NN_...
#
# A reasonable way to use is to use a name generator in bash like:
# $(for k in $(seq 1 9) 13 $(seq 15 20); do echo level1/$k\_; done)
#  - This can be appended with more names, since the script will just
#    append them to the list of kernels to run.

KERNELS=
ARGS=
for i in $*; do
  if [[ $i == --* ]]; then
    ARGS="$ARGS $i"
  else
    KERNELS="$KERNELS $i"
  fi
done

# Verify data type, if any
dtype=f32
for arg in $ARGS; do
  if [[ $arg == --dtype=* ]]; then
    dtype=${arg#*=}
    if [[ $dtype != f32 && $dtype != f16 && $dtype != bf16 && $dtype != i8 ]]; then
      echo "Invalid data type: $dtype"
      exit 1
    fi
    break
  fi
done

GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT=$GIT_ROOT/examples/KernelBench/test-kernel-bench.py

for kernel in $KERNELS; do
  # Avoid slashes on file names
  filename="${kernel//\//_}$dtype"
  LOG=$filename.log
  RESULTS=$filename.out
  OBJ=$filename.o
  ASM=$filename.s

  # First, try to run and dump the pipeline. If it crashes, we can
  # investigate which pass is broken.
  echo -e "\nRunning kernel: $kernel with args: $ARGS"
  uv run $SCRIPT --kernel $kernel \
                 --print-mlir-after-all \
                 $ARGS 2>&1 > $LOG
  if [ $? != 0 ]; then
    echo "Error executing kernel, bailing..."
    continue
  fi
  echo "Pipeline at $LOG"

  # Benchmark the kernel and dump the assembly.
  echo "Benchmarking kernel: $kernel with args: $ARGS"
  uv run $SCRIPT --kernel $kernel \
                 --benchmark \
                 $ARGS \
                 --no-validate \
                 --dump-assembly \
                 --assembly-file $ASM \
                 --object-file $OBJ \
                 2>&1 > $RESULTS
  grep Performance $RESULTS
done
