#!/usr/bin/env bash

# Script that runs the kernel bench script for a list of kernels,
# saves the logs, assembly and object files in the current directory.
#
# You can run this from anywhere, but the execution must run inside the
# git repo. To avoid poluting the git repo, you can run on the 'build'
# directory, since it's currently being ignored by git.
#
# cd build && ../scripts/KernelBench/debug_kernels.sh bench --arg level1/1_...
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
MODE=$1
shift
for i in $*; do
  if [[ $i == --* ]]; then
    ARGS="$ARGS $i"
  else
    KERNELS="$KERNELS $i"
  fi
done

if [[ $MODE != "smoke" && $MODE != "bench" ]] || [[ -z $KERNELS ]]; then
  echo "Usage: $0 <smoke|bench> [--arg1 --arg2 ...] kernel1 kernel2 ..."
  exit 1
fi

GIT_ROOT=$(git rev-parse --show-toplevel)
SCRIPT=$GIT_ROOT/examples/KernelBench/test-kernel-bench.py
LOG_DIR=$PWD
# uv run needs to be at the root of the repo
cd $GIT_ROOT

for kernel in $KERNELS; do
  for type in f32 bf16; do
    # Avoid slashes on file names
    filename="${kernel//\//_}$type"

    if [[ $MODE == "smoke" ]]; then
      for COMP in torch-compile no-torch-compile; do
        LOG=$LOG_DIR/$filename-$COMP.log
        echo -e "\nRunning kernel: $kernel with type $type and mode $COMP with args: $ARGS"
        uv run $SCRIPT --kernel $kernel \
                       --$COMP \
                       --dtype $type \
                       --smoke-test $ARGS > $LOG
        if [ $? != 0 ]; then
          echo "Error executing kernel, log at $LOG"
        else
          echo "Smoke test passed, log at $LOG"
        fi
      done
      continue
    fi

    LOG=$LOG_DIR/$filename.log
    RESULTS=$LOG_DIR/$filename.out
    OBJ=$LOG_DIR/$filename.o
    ASM=$LOG_DIR/$filename.s

    # First, try to run and dump the pipeline. If it crashes, we can
    # investigate which pass is broken.
    echo -e "\nRunning kernel: $kernel with type $type with args: $ARGS"
    uv run $SCRIPT --kernel $kernel \
                   --dtype $type \
                   --print-mlir-after-all $ARGS \
                   --no-validate > $LOG
    if [ $? != 0 ]; then
      echo "Error executing kernel, bailing..."
      continue
    fi
    echo "Whole pipeline at $LOG"

    # Benchmark the kernel and dump the assembly.
    uv run $SCRIPT --kernel $kernel --dtype $type --benchmark $args --no-validate --dump-assembly 2>&1 > $RESULTS
    grep Performance $RESULTS
    mv *.o $OBJ
    mv *.s $ASM
  done
done
