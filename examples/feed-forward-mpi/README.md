# Computing an FeedForward Layer sigmoid(A@B)@C on multiple ranks using MPI through MLIR

This example demonstrates how MLIR's sharding infrastructure can be used to distribute
data and computation across multiple nodes with non-shared memory. It implements a single
feed-forward layer.

The example uses three sharding passes `sharding-propagation`, `shard-partition`, and
`convert-shard-to-mpi` and afterwards lowers the IR to LLVM. The ingress IR uses minimal
sharding annotations necessary to guide the sharding propagation to produce the desired
1D or 2D weight-stationary partition strategies as described in figures 2a and 2b of
https://arxiv.org/pdf/2211.05102.

## Prerequisites

You need mpi4py in your python env. The default MPI implementation is MPICH.

For OpenMPI, change `"MPI:Implementation" = "MPICH"` to `"MPI:Implementation" = "OpenMPI"`
in ff_weight_stationary.py.

## Running

```
export MPI_DIR=<path_to_mpi_install>
uv sync --extra runtime_mpich
uv run mpirun -n <nRanks> python -u feed-forward-mpi.py --mpilib $MPI_DIR/lib/libmpi.so
```
Run with `--help` for more options.
