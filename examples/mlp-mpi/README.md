# Computing a MLP sigmoid(A@B)@C on multiple ranks using MPI through MLIR

## Prerequisites

You need mpi4py in your python env. The default MPI implementation is MPICH.

For OpenMPI, change `"MPI:Implementation" = "MPICH"` to `"MPI:Implementation" = "OpenMPI"` in the first line of mlp_weight_stationary.mlir.

## Running

```
export MPI_DIR=<path_to_mpi_install>
uv sync --extra runtime_mpich
uv run mpirun -n <nRanks> python -u mlp-mpi.py --mpilib $MPI_DIR/lib/libmpi.so
```
Run with `--help` for more options.
