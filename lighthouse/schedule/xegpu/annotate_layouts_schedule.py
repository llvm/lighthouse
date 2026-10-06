from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import xegpu

from lighthouse.schedule import schedule_boilerplate
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext

from .matmul_constraints import (
    DPAS,
    PREFETCH_INST_DATA,
)


def add_prefetch(load_op, prefetch_nb, **layout):
    anytype = transform.AnyOpType.get()
    desc_op = xegpu.insert_prefetch(
        load_op,
        nb_prefetch=prefetch_nb,
    )
    pf_ops = transform.get_consumers_of_result(anytype, desc_op, 0)
    xegpu.set_anchor_layout(pf_ops, **layout)


def annotate_ab_load(
    dpas_op, index, load_op, layout_load, layout_dpas, layout_prefetch, prefetch_nb
):
    """Annotate A/B tile load op and dpas operand and insert prefetch ops."""
    anytype = transform.AnyOpType.get()
    user = transform.get_consumers_of_result(anytype, load_op, 0)

    # transposed case
    transpose_consumer_op = transform.select(anytype, user, "vector.transpose")
    with lh_transform.foreach(transpose_consumer_op):
        # Load op loads the transposed tile and thus sg_layout and sg_data
        # dimensions must be transposed. Keep inst_data which has been
        # validated in its current orientation.
        tr_load = layout_load.copy()
        tr_load["sg_layout"] = layout_load["sg_layout"][::-1]
        tr_load["sg_data"] = layout_load["sg_data"][::-1]
        tr_load["order"] = [0, 1]
        # annotate dpas op operand
        layout_dpas_order = layout_dpas.copy()
        layout_dpas_order["order"] = [1, 0]
        xegpu.set_anchor_layout(dpas_op, index=index, **layout_dpas_order)
        xegpu.set_anchor_layout(load_op, **tr_load)
        add_prefetch(load_op, prefetch_nb, **layout_prefetch)
        transform.yield_()

    # no transpose case
    dpas_consumer_op = transform.select(anytype, user, "xegpu.dpas")
    with lh_transform.foreach(dpas_consumer_op):
        # annotate dpas op operand
        xegpu.set_anchor_layout(dpas_op, index=index, **layout_dpas)
        xegpu.set_anchor_layout(load_op, **layout_load)
        add_prefetch(load_op, prefetch_nb, **layout_prefetch)
        transform.yield_()


def annotate_gemm(
    gpu_func_op: ir.Operation,
    dpas_op: ir.Operation,
    sg_tile: tuple[int, int] | None = None,
):
    # TODO possibility to override all remaining params, load and prefetch tiles etc.
    params = transform_ext.infer_xegpu_gemm_params(dpas_op, force_sg_tile=sg_tile)
    wg_m = transform_ext.get_param_dict_entry(params, "wg_m")
    wg_n = transform_ext.get_param_dict_entry(params, "wg_n")
    sg_m = transform_ext.get_param_dict_entry(params, "sg_m")
    sg_n = transform_ext.get_param_dict_entry(params, "sg_n")
    k_tile = transform_ext.get_param_dict_entry(params, "k_tile")
    load_a_m = transform_ext.get_param_dict_entry(params, "load_a_m")
    load_a_k = transform_ext.get_param_dict_entry(params, "load_a_k")
    load_b_k = transform_ext.get_param_dict_entry(params, "load_b_k")
    load_b_n = transform_ext.get_param_dict_entry(params, "load_b_n")
    prefetch_a_m = transform_ext.get_param_dict_entry(params, "prefetch_a_m")
    prefetch_a_k = transform_ext.get_param_dict_entry(params, "prefetch_a_k")
    prefetch_b_k = transform_ext.get_param_dict_entry(params, "prefetch_b_k")
    prefetch_b_n = transform_ext.get_param_dict_entry(params, "prefetch_b_n")
    prefetch_a_nb = transform_ext.get_param_dict_entry(params, "prefetch_a_nb")
    prefetch_b_nb = transform_ext.get_param_dict_entry(params, "prefetch_b_nb")
    transpose_a = transform_ext.get_param_dict_entry(params, "transpose_a")
    transpose_b = transform_ext.get_param_dict_entry(params, "transpose_b")

    # Compute sg_layout[i] = wg_i // sg_i.
    sg_layout, _ = transform_ext.compute_sg_layout([wg_m, wg_n], [sg_m, sg_n])

    # Prefetch parent tile is (wg_m, k_tile)/(k_tile, wg_n); transpose_* reverses
    # its dims when the operand is loaded transposed.
    prefetch_layout_a, _ = transform_ext.compute_sg_layout(
        [wg_m, k_tile], [prefetch_a_m, prefetch_a_k], transpose=transpose_a
    )
    prefetch_layout_b, _ = transform_ext.compute_sg_layout(
        [k_tile, wg_n], [prefetch_b_k, prefetch_b_n], transpose=transpose_b
    )

    anyvalue = transform.AnyValueType.get()
    anytype = transform.AnyOpType.get()

    # matmul matrix shapes
    sg_tile_a = [sg_m, k_tile]
    sg_tile_b = [k_tile, sg_n]
    load_tile_a = [load_a_m, load_a_k]
    load_tile_b = [load_b_k, load_b_n]
    prefetch_tile_a = [prefetch_a_m, prefetch_a_k]
    prefetch_tile_b = [prefetch_b_k, prefetch_b_n]

    load_op_a = xegpu.get_load_op(transform.get_operand(anyvalue, dpas_op, [0]))
    load_op_b = xegpu.get_load_op(transform.get_operand(anyvalue, dpas_op, [1]))

    # A tile load layout
    layout_load_a = {
        "sg_layout": sg_layout,
        "sg_data": sg_tile_a,
        "inst_data": load_tile_a,
    }
    # A tile dpas layout
    layout_dpas_a = layout_load_a.copy()
    layout_dpas_a["inst_data"] = DPAS.A_TILE
    # A tile prefetch layout
    layout_prefetch_a = {
        "sg_layout": prefetch_layout_a,
        "sg_data": prefetch_tile_a,
        "inst_data": PREFETCH_INST_DATA,
    }
    annotate_ab_load(
        dpas_op,
        0,
        load_op_a,
        layout_load_a,
        layout_dpas_a,
        layout_prefetch_a,
        prefetch_a_nb,
    )

    # B tile load layout
    layout_load_b = {
        "sg_layout": sg_layout,
        "sg_data": sg_tile_b,
        "inst_data": load_tile_b,
    }
    # B tile dpas layout
    layout_dpas_b = layout_load_b.copy()
    layout_dpas_b["inst_data"] = DPAS.B_TILE
    # B tile prefetch layout
    layout_prefetch_b = {
        "sg_layout": prefetch_layout_b,
        "sg_data": prefetch_tile_b,
        "inst_data": PREFETCH_INST_DATA,
    }
    annotate_ab_load(
        dpas_op,
        1,
        load_op_b,
        layout_load_b,
        layout_dpas_b,
        layout_prefetch_b,
        prefetch_b_nb,
    )

    # C tile layout
    output_layout = {
        "sg_layout": sg_layout,
        "sg_data": [sg_m, sg_n],
        "inst_data": DPAS.C_TILE,
    }
    # C tile dpas anchor layout
    xegpu.set_anchor_layout(dpas_op, index=2, **output_layout)
    # annotate store op
    # FIXME trace consumers of dpas op?
    store_op_c = lh_transform.match_op(gpu_func_op, "xegpu.store_nd")
    xegpu.set_anchor_layout(store_op_c, **output_layout)

    # annotate the 1d load of the broadcast op with a slice layout
    # NOTE assumes that xegpu.load is followed by vector.broadcast
    maybe_bcast_load = lh_transform.match_op(gpu_func_op, "xegpu.load")
    load_user = transform.get_consumers_of_result(anytype, maybe_bcast_load, 0)
    bcast_ops = transform.select(anytype, load_user, "vector.broadcast")
    with lh_transform.foreach(bcast_ops) as bcast_op:
        bcast_load = xegpu.get_load_op(transform.get_operand(anyvalue, bcast_op, [0]))
        xegpu.set_anchor_layout(bcast_load, index=0, **output_layout, slice_dims=[0])
        transform.yield_()

    lh_transform.cleanup(gpu_func_op)

    # hoist desc ops out of reduction loop
    k_loop = transform.get_parent_op(
        anytype, dpas_op, op_name="scf.for", deduplicate=True
    )
    transform.apply_licm(k_loop)

    lh_transform.cleanup(gpu_func_op)


def annotate_reduction(gpu_func: ir.Operation, anchor_op: ir.Operation):
    wg_tile, sg_tile, reduction_tile = transform_ext.infer_xegpu_reduction_params(
        anchor_op
    )
    sg_layout, sg_data = transform_ext.compute_sg_layout(
        wg_tile, sg_tile, reduction_tile
    )
    # Annotate the first store op
    store_nd_ops = lh_transform.match_op(gpu_func, "xegpu.store_nd")
    store_op = transform_ext.extract_handle(store_nd_ops, 0, silenceable=True)
    xegpu.set_anchor_layout(store_op, sg_layout=sg_layout, sg_data=sg_data)


def annotate_attention(gpu_func: ir.Operation, anchor_op: ir.Operation):
    wg_tile, sg_tile, reduction_tile = transform_ext.infer_xegpu_attention_params(
        anchor_op
    )
    # TODO infer the layout parameters - hardcoded for now

    # Insert prefetches for the K and V tiles of the reduction loop. Each
    # inserts nb_prefetch prefetches ahead of the loop plus one per iteration,
    # at induction_var + nb_prefetch * step. This must run before the wg-level
    # layouts are set below, since the prefetch descriptor is cloned from the
    # load's descriptor. The layouts of the emitted prefetch_nd ops are set
    # together with the other wg-level layouts.
    nb_prefetch = 1
    prefetch_sg_data = [16, 32]
    prefetch_sg_layout = [4, 2]
    prefetch_inst_data = list(prefetch_sg_data)

    if nb_prefetch > 0:
        reduction_loop = lh_transform.match_op(gpu_func, "scf.for")
        load_ops = lh_transform.match_op(reduction_loop, "xegpu.load_nd")
        with lh_transform.foreach(load_ops) as load_op:
            add_prefetch(
                load_op,
                nb_prefetch,
                sg_layout=prefetch_sg_layout,
                sg_data=prefetch_sg_data,
                inst_data=prefetch_inst_data,
            )
            transform.yield_()
        lh_transform.cleanup(gpu_func)

    out_sg_data = [16, 64]
    out_sg_layout = [8, 1]
    # Annotate the final store op
    store_nd_ops = lh_transform.match_op(gpu_func, "xegpu.store_nd")
    store_op = transform_ext.extract_handle(store_nd_ops, 0, silenceable=True)
    xegpu.set_anchor_layout(store_op, sg_layout=out_sg_layout, sg_data=out_sg_data)

    # Set layout for xegpu.load_nd ops (3 total: Q, K, V)
    load_nd_ops = lh_transform.match_op(gpu_func, "xegpu.load_nd")

    # First load_nd: Q layout
    load_op = transform_ext.extract_handle(load_nd_ops, 0, silenceable=True)
    q_load_inst_data = [16, 32]
    xegpu.set_anchor_layout(
        load_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
        inst_data=q_load_inst_data,
    )

    # Second load_nd: K layout
    load_op = transform_ext.extract_handle(load_nd_ops, 1, silenceable=True)
    kv_sg_layout = [1, 1]
    kv_load_sg_data = [64, 64]
    k_load_order = [0, 1]
    xegpu.set_anchor_layout(
        load_op,
        sg_layout=kv_sg_layout,
        sg_data=kv_load_sg_data,
        order=k_load_order,
    )

    # Third load_nd: V layout
    load_op = transform_ext.extract_handle(load_nd_ops, 2, silenceable=True)
    v_load_inst_data = [32, 32]
    xegpu.set_anchor_layout(
        load_op,
        sg_layout=kv_sg_layout,
        sg_data=kv_load_sg_data,
        inst_data=v_load_inst_data,
    )

    # Set layout for xegpu.dpas ops (2 total: Q@K^T and P@V)
    dpas_ops = lh_transform.match_op(gpu_func, "xegpu.dpas")

    # Layouts for the Q@K^T dpas:
    qk_dpas_op = transform_ext.extract_handle(dpas_ops, 0, silenceable=True)
    # Index 0: Q layout
    xegpu.set_anchor_layout(
        qk_dpas_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
        index=0,
    )
    # Index 1: K^T layout
    kt_sg_data = [64, 64]
    xegpu.set_anchor_layout(
        qk_dpas_op,
        sg_layout=kv_sg_layout,
        sg_data=kt_sg_data,
        index=1,
    )
    # Index 2: QK output layout
    xegpu.set_anchor_layout(
        qk_dpas_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
        index=2,
    )

    # Layouts for the P@V dpas:
    pv_dpas_op = transform_ext.extract_handle(dpas_ops, 1, silenceable=True)
    # Index 0: QK (attention weights) layout
    xegpu.set_anchor_layout(
        pv_dpas_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
        index=0,
    )
    # Index 1: V layout
    v_sg_data = [64, 64]
    xegpu.set_anchor_layout(
        pv_dpas_op,
        sg_layout=kv_sg_layout,
        sg_data=v_sg_data,
        index=1,
    )
    # Index 2: Output layout
    xegpu.set_anchor_layout(
        pv_dpas_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
        index=2,
    )


def analyze_and_annotate_gpu_func(
    gpu_func_op: ir.Operation, sg_tile: tuple[int, int] | None = None
):
    """
    Analyzes a GPU function operation and annotates it with layout information
    for the contained DPAS operations.

    Args:
        gpu_func_op (ir.Operation): The GPU function operation to analyze and annotate.
        sg_tile (tuple[int, int] | None): Optional forced subgroup tile size.
    """

    alt = lh_transform.alternatives(
        gpu_func_op, num_alternatives=4, result_types=[gpu_func_op.type]
    )
    with alt.region(0) as gpu_func:
        # Attention layer case; has both dpas and reduction ops
        dpas_ops = lh_transform.match_op(gpu_func, "xegpu.dpas")
        dpas_op = transform_ext.extract_handle(dpas_ops, 0, silenceable=True)
        anchor_op = lh_transform.match_op(gpu_func, "vector.multi_reduction")
        anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
        annotate_attention(gpu_func, anchor_op)
        transform.yield_([gpu_func])
    with alt.region(1) as gpu_func:
        # For now let's assume gpu.func op contains only a single dpas op
        dpas_ops = lh_transform.match_op(gpu_func, "xegpu.dpas")
        dpas_op = transform_ext.extract_handle(dpas_ops, 0, silenceable=True)
        annotate_gemm(gpu_func, dpas_op, sg_tile=sg_tile)
        transform.yield_([gpu_func])
    with alt.region(2) as gpu_func:
        # Match multi_reduction op
        anchor_op = lh_transform.match_op(gpu_func, "vector.multi_reduction")
        anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
        annotate_reduction(gpu_func, anchor_op)
        transform.yield_([gpu_func])
    with alt.region(3) as gpu_func:
        # Exhausting all real alternatives is a hard error, not a recoverable
        # silenceable failure.
        transform_ext.emit_definite_failure(
            gpu_func,
            message="annotate_layouts: Could not apply any of the defined patterns.",
        )
        transform.yield_([gpu_func])


def annotate_layouts_schedule(
    sg_tile: tuple[int, int] | None = None,
) -> ir.Module:
    """Adds xegpu layout annotations and prefetch ops to relevant ops."""

    with schedule_boilerplate() as (schedule, named_seq):
        gpu_func_ops = lh_transform.match_op(named_seq.bodyTarget, "gpu.func")
        with lh_transform.foreach(gpu_func_ops) as gpu_func_op:
            analyze_and_annotate_gpu_func(gpu_func_op, sg_tile=sg_tile)
            transform.yield_()
        transform.yield_()

    return schedule
