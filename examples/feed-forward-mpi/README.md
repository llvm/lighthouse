# Computing an FeedForward Layer sigmoid(A@B)@C on multiple ranks using MPI through MLIR

This example shows how MLIR's sharding infrastructure can be used to distribute data and computation across multiple nodes with non-shared memory.

Currently, only the lower part of the sharding pipeline is used: `shard-partition`, `convert-shard-to-mpi`, and lowering to LLVM. Therefore, the ingress MLIR is fully annotated.

The example implements a single feed-forwad layer, following a 1D/2D weight-stationary partition strategy as described in figures 2a and 2b of https://arxiv.org/pdf/2211.05102.

## Prerequisites

You need mpi4py in your python env. The default MPI implementation is MPICH.

For OpenMPI, change `"MPI:Implementation" = "MPICH"` to `"MPI:Implementation" = "OpenMPI"` in ff_weight_stationary.py.

## Running

```
export MPI_DIR=<path_to_mpi_install>
uv sync --extra runtime_mpich
uv run mpirun -n <nRanks> python -u feed-forward-mpi.py --mpilib $MPI_DIR/lib/libmpi.so
```
Run with `--help` for more options.
