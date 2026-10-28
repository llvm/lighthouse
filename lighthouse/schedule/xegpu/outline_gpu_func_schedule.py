from mlir import ir
from mlir.dialects.transform import xegpu
from mlir.dialects import transform
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    canonicalize,
    match,
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


def set_gpu_threads_attention(
    launch_op: ir.Value[transform.AnyOpType], force_sg_tile: list[int] | None = None
):
    # Ensure this launch contains both parts of the attention contraction.
    red_op = lh_transform.match_op(launch_op, "vector.multi_reduction")
    transform_ext.extract_handle(red_op, 0, silenceable=True)
    anchor_op = lh_transform.match_op(launch_op, "vector.contract")
    anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
    wg_tile, sg_tile, _ = transform_ext.infer_xegpu_attention_params(
        anchor_op, force_sg_tile=force_sg_tile
    )
    nb_threads = transform_ext.compute_num_threads(wg_tile, sg_tile, base=NB_WORKITEMS)
    xegpu.set_gpu_launch_threads(launch_op, threads=[nb_threads, 1, 1])


def set_gpu_threads_gemm(
    launch_op: ir.Value[transform.AnyOpType],
    force_sg_tile: list[int] | None = None,
    device: str | None = None,
):
    # Match vector.contract and call infer_xegpu_gemm_params
    # Produces a silenceable failure if this launch has no vector.contract.
    anchor_op = lh_transform.match_op(launch_op, "vector.contract")
    anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
    params = transform_ext.infer_xegpu_gemm_params(
        anchor_op, force_sg_tile=force_sg_tile, device=device
    )
    wg_m = transform_ext.get_param_dict_entry(params, "wg_m")
    wg_n = transform_ext.get_param_dict_entry(params, "wg_n")
    sg_m = transform_ext.get_param_dict_entry(params, "sg_m")
    sg_n = transform_ext.get_param_dict_entry(params, "sg_n")

    nb_threads = transform_ext.compute_num_threads(
        [wg_m, wg_n], [sg_m, sg_n], base=NB_WORKITEMS
    )
    xegpu.set_gpu_launch_threads(launch_op, threads=[nb_threads, 1, 1])


def set_gpu_threads_reduction(
    launch_op: ir.Value[transform.AnyOpType], force_sg_tile: list[int] | None = None
):
    # Match vector.multi_reduction and call infer_xegpu_reduction_params
    # Produces a silenceable failure if this launch has no multi_reduction.
    anchor_op = lh_transform.match_op(launch_op, "vector.multi_reduction")
    anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
    wg_tile, sg_tile, _ = transform_ext.infer_xegpu_reduction_params(
        anchor_op, force_sg_tile=force_sg_tile
    )
    nb_threads = transform_ext.compute_num_threads(wg_tile, sg_tile, base=NB_WORKITEMS)
    xegpu.set_gpu_launch_threads(launch_op, threads=[nb_threads, 1, 1])


def set_gpu_threads_elementwise(launch_op: ir.Value[transform.AnyOpType]):
    """Assign one subgroup to an elementwise workgroup kernel."""
    writes = lh_transform.match_op(launch_op, "vector.transfer_write")
    transform_ext.extract_handle(writes, 0, silenceable=True)
    xegpu.set_gpu_launch_threads(launch_op, threads=[NB_WORKITEMS, 1, 1])


def _annotate_kernel_kind(target: ir.Value, kind: str):
    kind_param = transform.ParamConstantOp(
        transform.AnyParamType.get(), ir.StringAttr.get(kind)
    ).param
    transform.annotate(target, "transform_ext.kernel_kind", param=kind_param)


def _classify_gpu_launches(func: ir.Value[transform.AnyOpType]):
    """Classify launches by the operations they contain.

    A launch with both a vector.contract and a multi_reduction is an attention
    kernel. A lone contract is GEMM; a lone multi_reduction is reduction. Any
    other launch is handled by the elementwise path.
    """
    anytype = transform.AnyOpType.get()
    launches = match(func, ops={"gpu.launch"})
    _annotate_kernel_kind(launches, "elementwise")

    contracts = match(func, ops={"vector.contract"})
    contract_launches = transform.get_parent_op(
        anytype, contracts, op_name="gpu.launch", deduplicate=True
    )
    transform.annotate(contract_launches, "transform_ext.outline_has_contract")
    _annotate_kernel_kind(contract_launches, "gemm")

    reductions = match(func, ops={"vector.multi_reduction"})
    reduction_launches = transform.get_parent_op(
        anytype, reductions, op_name="gpu.launch", deduplicate=True
    )
    transform.annotate(reduction_launches, "transform_ext.outline_has_reduction")
    _annotate_kernel_kind(reduction_launches, "reduction")

    attention_launches = match(
        func,
        ops={"gpu.launch"},
        op_attrs={
            "transform_ext.outline_has_contract": ir.UnitAttr.get(),
            "transform_ext.outline_has_reduction": ir.UnitAttr.get(),
        },
    )
    _annotate_kernel_kind(attention_launches, "attention")


def _set_gpu_threads_for_kind(
    func: ir.Value[transform.AnyOpType],
    kind: str,
    configure,
):
    launches = match(
        func,
        ops={"gpu.launch"},
        op_attrs={"transform_ext.kernel_kind": ir.StringAttr.get(kind)},
    )
    with lh_transform.foreach(launches) as launch_op:
        configure(launch_op)
        transform.yield_()


def outline(
    mod: ir.Value[transform.AnyOpType],
    payload_func: ir.Operation,
    sg_tile: list[int] | None = None,
    device: str | None = None,
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering MLP-like payload to xegpu wg level."""

    payload_func = convert_to_gpu_launch(mod, payload_func=payload_func)

    # Each launch is one kernel. Tag the launches by their contents, then apply
    # each independent thread-counting path only to its matching launches.
    _classify_gpu_launches(payload_func)
    _set_gpu_threads_for_kind(
        payload_func,
        "attention",
        lambda launch: set_gpu_threads_attention(launch, force_sg_tile=sg_tile),
    )
    _set_gpu_threads_for_kind(
        payload_func,
        "gemm",
        lambda launch: set_gpu_threads_gemm(
            launch, force_sg_tile=sg_tile, device=device
        ),
    )
    _set_gpu_threads_for_kind(
        payload_func,
        "reduction",
        lambda launch: set_gpu_threads_reduction(launch, force_sg_tile=sg_tile),
    )
    _set_gpu_threads_for_kind(payload_func, "elementwise", set_gpu_threads_elementwise)

    # outline gpu func
    payload_func = apply_registered_pass(payload_func, "lower-affine")
    canonicalize(payload_func)
    payload_func = apply_registered_pass(
        payload_func, "gpu-launch-sink-index-computations"
    )
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)

    return mod
