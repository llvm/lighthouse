# Computing a MLP sigmoid(A@B)@C on multiple ranks using MPI through MLIR
## Prerequisites
You need mpi4py in your python env. If you are not using OpenMPI (e.g. not MPICH-like like Intel MPI) you need to modify the first line in mlp_weight_stationary.mlir by replacing `"MPI:Implementation" = "MPICH"` with `"MPI:Implementation" = "OpenMPI"`.

## Running
```
export MLIR_DIR=<path_to_mlir_build_dir>
export MPI_DIR=<path_to_mp_install>
export LH_DIR=<path_to_lighthouse>
PYTHONPATH=$LH_DIR:$MLIR_DIR/tools/mlir/python_packages/mlir_core \
  mpirun -n <nRanks> \
  python -u mlp-mpi.py \
  --mpilib $MPI_DIR/libmpi.so \
  --utils_dir $MLIR_DIR/lib \
  -s 64 64 64
```
Run with `--help` for mor options.
