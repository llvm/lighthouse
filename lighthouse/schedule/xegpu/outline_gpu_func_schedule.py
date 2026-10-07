from mlir import ir
from mlir.dialects.transform import xegpu
from mlir.dialects import transform
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    canonicalize,
)
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext

from lighthouse.schedule import schedule_boilerplate
from .matmul_constraints import NB_WORKITEMS
from .lowering_common import (
    convert_to_gpu_launch,
    get_payload_func,
)


def outline_gpu_func_schedule(
    payload_func_name: str | None = None,
    sg_tile: list[int] | None = None,
    device: str | None = None,
) -> ir.Module:
    """
    Outlines scf.forall loops to gpu.func calls.

    Assumes that the payload function has been bufferized.
    """

    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        op_names = [
            "linalg.matmul",
            "linalg.generic",
            "vector.contract",
            "vector.transfer_read",
            "vector.transfer_write",
        ]
        func = get_payload_func(
            named_seq.bodyTarget, func_name=payload_func_name, op_name=op_names
        )
        payload_mod = transform.get_parent_op(
            anytype,
            func,
            op_name="builtin.module",
            deduplicate=True,
        )
        outline(
            payload_mod,
            payload_func=func,
            sg_tile=sg_tile,
            device=device,
        )
        transform.yield_()

    return schedule


def set_gpu_threads_attention(func: ir.Operation):
    launch_ops = lh_transform.match_op(func, "gpu.launch")
    with lh_transform.foreach(launch_ops) as launch_op:
        # Ensure there's at least one multi_reduction op
        red_op = lh_transform.match_op(launch_op, "vector.multi_reduction")
        transform_ext.extract_handle(red_op, 0, silenceable=True)
        anchor_op = lh_transform.match_op(launch_op, "vector.contract")
        anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
        wg_tile, sg_tile, _ = transform_ext.infer_xegpu_attention_params(anchor_op)
        nb_threads = transform_ext.compute_num_threads(
            wg_tile, sg_tile, base=NB_WORKITEMS
        )
        xegpu.set_gpu_launch_threads(launch_op, threads=[nb_threads, 1, 1])
        transform.yield_()


def set_gpu_threads_gemm(
    func: ir.Operation, sg_tile: list[int] | None = None, device: str | None = None
):
    # Match vector.contract and call infer_xegpu_gemm_params
    launch_ops = lh_transform.match_op(func, "gpu.launch")
    with lh_transform.foreach(launch_ops) as launch_op:
        # Produces a silenceable failure if the match fails
        # Ensure there's at least one vector.contract op and use it as anchor
        anchor_op = lh_transform.match_op(launch_op, "vector.contract")
        anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
        params = transform_ext.infer_xegpu_gemm_params(
            anchor_op, force_sg_tile=sg_tile, device=device
        )
        wg_m = transform_ext.get_param_dict_entry(params, "wg_m")
        wg_n = transform_ext.get_param_dict_entry(params, "wg_n")
        sg_m = transform_ext.get_param_dict_entry(params, "sg_m")
        sg_n = transform_ext.get_param_dict_entry(params, "sg_n")

        nb_threads = transform_ext.compute_num_threads(
            [wg_m, wg_n], [sg_m, sg_n], base=NB_WORKITEMS
        )
        xegpu.set_gpu_launch_threads(launch_op, threads=[nb_threads, 1, 1])
        transform.yield_()


def set_gpu_threads_reduction(func: ir.Operation):
    # Match vector.multi_reduction and call infer_xegpu_reduction_params
    launch_ops = lh_transform.match_op(func, "gpu.launch")
    with lh_transform.foreach(launch_ops) as launch_op:
        # Produces a silenceable failure if the match fails
        anchor_op = lh_transform.match_op(launch_op, "vector.multi_reduction")
        anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
        wg_tile, sg_tile, _ = transform_ext.infer_xegpu_reduction_params(anchor_op)
        nb_threads = transform_ext.compute_num_threads(
            wg_tile, sg_tile, base=NB_WORKITEMS
        )
        xegpu.set_gpu_launch_threads(launch_op, threads=[nb_threads, 1, 1])
        transform.yield_()


def outline(
    mod: ir.Value[transform.AnyOpType],
    payload_func: ir.Operation,
    sg_tile: list[int] | None = None,
    device: str | None = None,
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering MLP-like payload to xegpu wg level."""

    payload_func = convert_to_gpu_launch(mod, payload_func=payload_func)

    # Set correct number of gpu threads.
    alt = lh_transform.alternatives(
        payload_func, num_alternatives=4, result_types=[payload_func.type]
    )
    with alt.region(0) as func:
        # Attention schedule case
        set_gpu_threads_attention(func)
        transform.yield_([func])
    with alt.region(1) as func:
        # GEMM schedule case
        set_gpu_threads_gemm(func, sg_tile=sg_tile, device=device)
        transform.yield_([func])
    with alt.region(2) as func:
        # Reduction schedule case
        set_gpu_threads_reduction(func)
        transform.yield_([func])
    with alt.region(3) as func:
        transform_ext.emit_definite_failure(
            func,
            message="outline_gpu_func: Could not apply any of the defined patterns.",
        )
        transform.yield_([func])
    payload_func = alt.results[0]

    # outline gpu func
    payload_func = apply_registered_pass(payload_func, "lower-affine")
    canonicalize(payload_func)
    payload_func = apply_registered_pass(
        payload_func, "gpu-launch-sink-index-computations"
    )
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)

    return mod
