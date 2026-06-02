"""Generate an MLIR module for a weight-stationary distributed feed-forward layer."""

from mlir import ir
from mlir.dialects import arith, linalg, shard, tensor, tosa, bufferization
from lighthouse.utils.mlir import func_cif
from lighthouse.ingress.mlir_gen.shard_utils import split_axes_to_mlir

_GRID = "grid0"


def generate_ff_payload(
    *,
    func_name: str,
    M: int,
    N: int,
    K: int,
    comm_size: int,
    comm_rank: int,
    grid: list[int],
    split_act: list[list[int]],
    split_win: list[list[int]],
    split_wout: list[list[int]],
    split_mm0a_mm1c: list[list[int]],
    split_mm0_c: list[list[int]],
    split_sigmoid: list[list[int]],
) -> ir.Module:
    """Generate the full MLIR module for the weight-stationary distributed feed-forward layer.
    Also adds helper functions for allocation, deallocation and gather."""
    mod = ir.Module.create()
    f32 = ir.F32Type.get()

    # Module-level DLTI attribute for MPI metadata.
    dlti = (
        f'#dlti.map<"MPI:Implementation" = "MPICH", '
        f'"MPI:comm_world_size" = {comm_size}, '
        f'"MPI:comm_world_rank" = {comm_rank}>'
    )
    mod.operation.attributes["mpi.dlti"] = ir.Attribute.parse(dlti)

    # Tensor types used throughout.
    t_mk = ir.RankedTensorType.get((M, K), f32)
    t_kn = ir.RankedTensorType.get((K, N), f32)
    t_nk = ir.RankedTensorType.get((N, K), f32)
    t_mn = ir.RankedTensorType.get((M, N), f32)

    with ir.InsertionPoint(mod.body):
        # --- grid ---
        g = shard.grid(_GRID, grid)
        g.operation.attributes["sym_visibility"] = ir.StringAttr.get("private")

        # --- payload function ---
        @func_cif(t_mk, t_kn, t_nk, t_mk, name=func_name)
        def _(a, b, c, r):
            cst = arith.constant(f32, 0.0)

            sh_act = shard.sharding(_GRID, split_axes_to_mlir(split_act), [], [])
            sh_win = shard.sharding(_GRID, split_axes_to_mlir(split_win), [], [])
            sh_wout = shard.sharding(_GRID, split_axes_to_mlir(split_wout), [], [])
            sh_ac = shard.sharding(_GRID, split_axes_to_mlir(split_mm0a_mm1c), [], [])
            sh_mc = shard.sharding(_GRID, split_axes_to_mlir(split_mm0_c), [], [])
            sh_sig = shard.sharding(_GRID, split_axes_to_mlir(split_sigmoid), [], [])

            sd_ai = shard.shard(a, sh_act)
            sd_bi = shard.shard(b, sh_win)
            sd_ci = shard.shard(c, sh_wout)
            sd_r = shard.shard(r, sh_act)

            empty0 = tensor.empty((M, N), f32)
            fill0 = linalg.fill(cst, outs=[empty0])

            sd_a1 = shard.shard(sd_ai, sh_ac, annotate_for_users=True)
            sd_fill0 = shard.shard(fill0, sh_mc, annotate_for_users=True)
            mm0 = linalg.matmul(sd_a1, sd_bi, outs=[sd_fill0])

            sd_mm0 = shard.shard(mm0, sh_sig, annotate_for_users=True)
            sig = tosa.sigmoid(t_mn, sd_mm0)

            empty1 = tensor.empty((M, K), f32)
            fill1 = linalg.fill(cst, outs=[empty1])

            sd_fill1 = shard.shard(fill1, sh_ac, annotate_for_users=True)
            mm1 = linalg.matmul(sig, sd_ci, outs=[sd_fill1])

            sd_res = shard.shard(mm1, sh_act, annotate_for_users=True)
            res = bufferization.materialize_in_destination(
                t_mk,
                sd_res,
                sd_r,
            )
            return shard.shard(res, sh_act, annotate_for_users=True)

    return mod
