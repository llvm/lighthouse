import inspect
from typing import Optional, Annotated

from mlir import ir
from mlir.dialects.transform import loop
from mlir.dialects.transform import bufferization
from mlir.dialects.transform import xegpu
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects import transform, smt
from mlir.dialects.transform import (
    structured,
    tune as transform_tune,
    smt as transform_smt,
)
from lighthouse.utils.mlir import apply_registered_pass, canonicalize, match
from lighthouse.tune.annotate import (
    check_annotated_constraints,
    NonDet,
    ConstraintCollector,
)


class PipelineInterrupt(Exception):
    """Exception to signal early termination of the transform schedule."""

    pass


# hardware constraints
dpas_tile = [8, 16, 16]
prefetch_inst_data = [8, 16]
nb_workitems = 16  # workitems in subgroup


def get_schedule_module(
    has_bias: bool = False,
    has_relu: bool = False,
    stop_at_stage: str = "",
    params: Optional[dict] = None,
) -> ir.Module:
    """Generate transform schedule module."""
    mod = ir.Module.create()
    mod.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    with ir.InsertionPoint(mod.body):
        named_sequence = transform.named_sequence(
            "__transform_main",
            [transform.AnyOpType.get()],  # input types
            [],  # output types
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
        )
        with ir.InsertionPoint(named_sequence.body):
            # match the payload module
            anytype = transform.AnyOpType.get()
            func = match(named_sequence.bodyTarget, ops={"func.func"})
            payload_mod = transform.get_parent_op(
                anytype,
                func,
                op_name="builtin.module",
                deduplicate=True,
            )
            xegpu_matmul_transform_schedule(
                payload_mod,
                has_bias=has_bias,
                has_relu=has_relu,
                stop_at_stage=stop_at_stage,
                params=params,
            )

    return mod


def xegpu_matmul_transform_schedule(
    mod: ir.Value,
    has_bias: bool = False,
    has_relu: bool = False,
    stop_at_stage: str = "",
    params: Optional[dict] = None,
):
    """Transform schedule for matmul-like payload."""
    try:
        mod = bundle_xepu_matmul_schedule(
            mod,
            has_bias=has_bias,
            has_relu=has_relu,
            stop_at_stage=stop_at_stage,
            **params,
        )

        mod = bundle_xegpu_to_binary(
            mod,
            stop_at_stage=stop_at_stage,
        )
    except PipelineInterrupt:
        pass
    finally:
        transform.yield_()


@check_annotated_constraints
def bundle_xepu_matmul_schedule(
    mod,
    has_bias: bool = False,
    has_relu: bool = False,
    stop_at_stage: str = "",
    *,
    wg_d0: Annotated[int, lambda _: 128 <= _ <= 256 and _ % 32 == 0] = NonDet,
    wg_d1: Annotated[int, lambda _: 128 <= _ <= 256 and _ % 32 == 0] = NonDet,
    sg_d0: Annotated[int, lambda _: 16 <= _ <= 32 and _ % 8 == 0] = NonDet,
    sg_d1: Annotated[int, lambda _: 16 <= _ <= 32 and _ % 8 == 0] = NonDet,
    k_tile: Annotated[int, lambda _: 8 <= _ <= 32 and _ % 8 == 0] = NonDet,
    load_a_d0: Annotated[int, lambda _: 8 <= _ <= 32 and _ % 8 == 0] = NonDet,
    load_a_d1: Annotated[int, lambda _: 8 <= _ <= 32 and _ % 8 == 0] = NonDet,
    load_b_d0: Annotated[int, lambda _: 8 <= _ <= 32 and _ % 8 == 0] = NonDet,
    load_b_d1: Annotated[int, lambda _: 8 <= _ <= 32 and _ % 8 == 0] = NonDet,
    prefetch_a_d0: Annotated[int, lambda _: 4 <= _ <= 8] = NonDet,
    prefetch_a_d1: Annotated[int, lambda _: 16 <= _ <= 32] = NonDet,
    prefetch_b_d0: Annotated[int, lambda _: 4 <= _ <= 8] = NonDet,
    prefetch_b_d1: Annotated[int, lambda _: 8 <= _ <= 16] = NonDet,
    nb_prefetch: Annotated[int, lambda _: 1 <= _ <= 32] = NonDet,
    **_kwargs: Optional[dict],
) -> ir.Module:
    """Schedule for lowering matmul-like payload to xegpu wg level."""

    sig = inspect.signature(bundle_xepu_matmul_schedule)

    any_param = transform.AnyParamType.get()

    use_knobs = NonDet in [
        wg_d0,
        wg_d1,
        prefetch_a_d0,
        prefetch_a_d1,
        prefetch_b_d0,
        prefetch_b_d1,
        k_tile,
        load_a_d0,
        load_a_d1,
        load_b_d0,
        load_b_d1,
        prefetch_a_d0,
        prefetch_a_d1,
        prefetch_b_d0,
        prefetch_b_d1,
        nb_prefetch,
    ]

    def as_const_or_as_knob(value, knob_name):
        collector = ConstraintCollector()
        sig.parameters[knob_name].annotation.__metadata__[0](collector)
        if use_knobs:
            return transform_tune.knob(
                any_param,
                name=knob_name,
                options=collector.to_mlir(),
                selected=value if value is not NonDet else None,
            )
        return value

    wg_d0 = as_const_or_as_knob(wg_d0, "wg_d0")
    wg_d1 = as_const_or_as_knob(wg_d1, "wg_d1")
    wg_tile = [wg_d0, wg_d1]
    sg_d0 = as_const_or_as_knob(sg_d0, "sg_d0")
    sg_d1 = as_const_or_as_knob(sg_d1, "sg_d1")
    sg_tile = [sg_d0, sg_d1]

    smt_int = smt.IntType.get()
    c0 = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)
    c_nb_workitems = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), nb_workitems)

    if use_knobs:
        constraint1 = transform_smt.constrain_params(
            (any_param, any_param, any_param),
            (
                wg_d0,
                wg_d1,
                sg_d0,
                sg_d1,
            ),
            [smt_int] * 4,
        )
        with ir.InsertionPoint(constraint1.body):
            WGd0, WGd1, SGd0, SGd1 = constraint1.body.arguments
            C0 = smt.int_constant(c0)
            smt.assert_(smt.eq((smt.int_mod(WGd0, SGd0), C0)))
            smt.assert_(smt.eq((smt.int_mod(WGd1, SGd1), C0)))
            d0_step_smt = smt.int_div(WGd0, SGd0)
            d1_step_smt = smt.int_div(WGd1, SGd1)
            nb_threads_smt = smt.int_mul(
                (d0_step_smt, d1_step_smt, smt.int_constant(c_nb_workitems))
            )
            smt.yield_((d0_step_smt, d1_step_smt, nb_threads_smt))
        d0_step, d1_step, nb_threads = constraint1.results
        sg_layout = [d0_step, d1_step]
    else:
        # derived parameters
        sg_layout = [wg_d0 // sg_d0, wg_d1 // sg_d1]
        # number of threads collapsed to 1d layout
        nb_threads = sg_layout[0] * sg_layout[1] * nb_workitems

    prefetch_a_d0 = as_const_or_as_knob(prefetch_a_d0, "prefetch_a_d0")
    prefetch_a_d1 = as_const_or_as_knob(prefetch_a_d1, "prefetch_a_d1")
    prefetch_tile_a = [prefetch_a_d0, prefetch_a_d1]
    prefetch_b_d0 = as_const_or_as_knob(prefetch_b_d0, "prefetch_b_d0")
    prefetch_b_d1 = as_const_or_as_knob(prefetch_b_d1, "prefetch_b_d1")
    prefetch_tile_b = [prefetch_b_d0, prefetch_b_d1]
    k_tile = as_const_or_as_knob(k_tile, "k_tile")

    if use_knobs:
        constraint2 = transform_smt.constrain_params(
            (any_param, any_param, any_param, any_param),
            (
                wg_d0,
                wg_d1,
                k_tile,
                prefetch_a_d0,
                prefetch_a_d1,
                prefetch_b_d0,
                prefetch_b_d1,
            ),
            [smt_int] * 7,
        )
        with ir.InsertionPoint(constraint2.body):
            WGd0, WGd1, K, PFAd0, PFAd1, PFBd0, PFBd1 = constraint2.body.arguments
            C0 = smt.int_constant(c0)
            smt.assert_(smt.eq((smt.int_mod(WGd0, PFAd0), C0)))
            smt.assert_(smt.eq((smt.int_mod(K, PFAd1), C0)))
            PFAd0_step = smt.int_div(WGd0, PFAd0)
            PFAd1_step = smt.int_div(K, PFAd1)

            smt.assert_(smt.eq((smt.int_mod(K, PFBd0), C0)))
            smt.assert_(smt.eq((smt.int_mod(WGd1, PFBd1), C0)))
            PFBd0_step = smt.int_div(K, PFBd0)
            PFBd1_step = smt.int_div(WGd1, PFBd1)

            smt.yield_((PFAd0_step, PFAd1_step, PFBd0_step, PFBd1_step))
        prefetch_layout_a = constraint2.results[0:2]
        prefetch_layout_b = constraint2.results[2:4]
    else:
        prefetch_layout_a = [
            wg_d0 // prefetch_a_d0,
            k_tile // prefetch_a_d1,
        ]
        prefetch_layout_b = [
            k_tile // prefetch_b_d0,
            wg_d1 // prefetch_b_d1,
        ]

    # matmul matrix shapes
    sg_tile_a = [sg_d0, k_tile]
    sg_tile_b = [k_tile, sg_d1]

    load_a_d0 = as_const_or_as_knob(load_a_d0, "load_a_d0")
    load_a_d1 = as_const_or_as_knob(load_a_d1, "load_a_d1")
    load_b_d0 = as_const_or_as_knob(load_b_d0, "load_b_d0")
    load_b_d1 = as_const_or_as_knob(load_b_d1, "load_b_d1")

    load_tile_a = [load_a_d0, load_a_d1]
    load_tile_b = [load_b_d0, load_b_d1]

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()
    anyvalue = transform.AnyValueType.get()

    # match the payload function
    anchor = match(mod, ops={"linalg.matmul"})
    func = transform.get_parent_op(
        anytype,
        anchor,
        op_name="func.func",
        deduplicate=True,
    )

    dpas_shape_a = [dpas_tile[0], dpas_tile[2]]
    dpas_shape_b = [dpas_tile[2], dpas_tile[1]]
    dpas_shape_c = [dpas_tile[0], dpas_tile[1]]

    # wg tiling
    if has_relu:
        terminal = match(mod, ops={"linalg.max"})
    elif has_bias:
        terminal = match(mod, ops={"linalg.add"})
    else:
        terminal = match(mod, ops={"linalg.matmul"})
    # FIXME use structured.structured_fuse
    structured.FuseOp(terminal, tile_sizes=wg_tile, use_forall=True)
    transform.apply_cse(mod)
    canonicalize(mod)

    # k loop tiling
    wg_matmul = match(mod, ops={"linalg.matmul"})
    # FIXME use structured.structured_tile_using_for
    wgk_matmul, k_loop = structured.TileUsingForOp(
        wg_matmul, sizes=[0, 0, k_tile]
    ).results

    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # vectorize
    # FIXME use structured.structured_vectorize_children_and_apply_patterns
    func = structured.VectorizeChildrenAndApplyPatternsOp(
        func,
        fold_type_extensions_into_contract=True,
    ).result

    # hoist loop invariant vector read/store ops
    k_loop = match(func, ops={"scf.for"})
    loop.HoistLoopInvariantSubsetsOp(k_loop)

    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    # bufferize

    # eliminate empty tensors to avoid emitting extra copy ops
    mod = apply_registered_pass(mod, "eliminate-empty-tensors")
    identity_layout = LayoutMapOption.IdentityLayoutMap
    mod = bufferization.OneShotBufferizeOp(
        mod,
        allow_return_allocs_from_loops=True,
        bufferize_function_boundaries=True,
        function_boundary_type_conversion=identity_layout,
    ).result
    # fold memref.subviews into vector.transfer_read/write ops
    mod = apply_registered_pass(mod, "fold-memref-alias-ops")
    transform.apply_cse(mod)
    canonicalize(mod)

    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    # convert forall to parallel
    wg_loop = match(mod, ops={"scf.forall"})
    wg_loop = loop.loop_forall_to_parallel([anytype], wg_loop)
    func = transform.get_parent_op(anytype, wg_loop)

    # convert to scf.parallel to gpu.launch
    func = apply_registered_pass(func, "gpu-map-parallel-loops")
    func = apply_registered_pass(func, "convert-parallel-loops-to-gpu")
    func = apply_registered_pass(func, "lower-affine")
    transform.apply_cse(func)
    canonicalize(func)

    # set correct number of gpu threads
    launch_op = match(func, ops={"gpu.launch"})
    xegpu.set_gpu_launch_threads(launch_op, threads=[nb_threads, 1, 1])

    # outline gpu func
    func = apply_registered_pass(func, "lower-affine")
    canonicalize(func)
    func = apply_registered_pass(func, "gpu-launch-sink-index-computations")
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)

    # set xevm target
    mod = apply_registered_pass(
        mod,
        "xevm-attach-target",
        options={"O": "3", "chip": "bmg"},
    )

    # convert vector to xegpu
    gpu_mod = match(mod, ops={"gpu.module"})
    gpu_func = match(gpu_mod, ops={"gpu.func"})
    gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
    transform.apply_cse(gpu_func)

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    # add layouts to DPAS op operands
    k_loop = match(gpu_func, ops={"scf.for"})
    dpas_op = match(k_loop, ops={"xegpu.dpas"})
    tile_a = transform.get_operand(anyvalue, dpas_op, [0])
    tile_b = transform.get_operand(anyvalue, dpas_op, [1])
    tile_c = transform.get_operand(anyvalue, dpas_op, [2])

    def convert_layout(value, input, target):
        xegpu.convert_layout(
            value,
            input_sg_layout=input["sg_layout"],
            input_sg_data=input["sg_data"],
            input_inst_data=input["inst_data"],
            target_sg_layout=target["sg_layout"],
            target_sg_data=target["sg_data"],
            target_inst_data=target["inst_data"],
        )

    # insert prefetch ops for DPAS A and B tiles
    desc_prefetch_a = xegpu.insert_prefetch(
        tile_a,
        nb_prefetch=nb_prefetch,
    )
    xegpu.set_desc_layout(
        desc_prefetch_a,
        sg_layout=prefetch_layout_a,
        sg_data=prefetch_tile_a,
        inst_data=prefetch_inst_data,
    )
    desc_prefetch_b = xegpu.insert_prefetch(
        tile_b,
        nb_prefetch=nb_prefetch,
    )
    xegpu.set_desc_layout(
        desc_prefetch_b,
        sg_layout=prefetch_layout_b,
        sg_data=prefetch_tile_b,
        inst_data=prefetch_inst_data,
    )

    # A tile load layout
    layout_load_a = {
        "sg_layout": sg_layout,
        "sg_data": sg_tile_a,
        "inst_data": load_tile_a,
    }
    desc_op_a = xegpu.get_desc_op(tile_a)
    desc_op_a = xegpu.set_desc_layout(
        target=desc_op_a,
        **layout_load_a,
    )
    # A tile dpas layout
    layout_dpas_a = layout_load_a.copy()
    layout_dpas_a["inst_data"] = dpas_shape_a
    convert_layout(tile_a, layout_load_a, layout_dpas_a)

    # B tile load layout
    layout_load_b = {
        "sg_layout": sg_layout,
        "sg_data": sg_tile_b,
        "inst_data": load_tile_b,
    }
    desc_op_b = xegpu.get_desc_op(tile_b)
    desc_op_b = xegpu.set_desc_layout(
        target=desc_op_b,
        **layout_load_b,
    )
    # B tile dpas layout
    layout_dpas_b = layout_load_b.copy()
    layout_dpas_b["inst_data"] = dpas_shape_b
    convert_layout(tile_b, layout_load_b, layout_dpas_b)

    # C tile layout
    output_layout = {
        "sg_layout": sg_layout,
        "sg_data": sg_tile,
        "inst_data": dpas_shape_c,
    }
    desc_op_c = xegpu.get_desc_op(tile_c)
    desc_op_c = xegpu.set_desc_layout(desc_op_c, **output_layout)
    # C tile dpas layout
    xegpu.set_op_layout_attr(dpas_op, result=True, index=0, **output_layout)

    if has_relu:
        # for post ops we need to add C layout manually
        max_op = match(gpu_func, ops={"arith.maximumf"})
        xegpu.set_op_layout_attr(max_op, result=True, index=0, **output_layout)
        # find zero constant buffer and annotate it
        const_buffer = transform.get_producer_of_operand(anytype, max_op, 1)
        xegpu.set_op_layout_attr(const_buffer, result=True, index=0, **output_layout)
    if has_bias:
        # for post ops we need to add C layout manually
        add_op = match(gpu_func, ops={"arith.addf"})
        xegpu.set_op_layout_attr(add_op, result=True, index=0, **output_layout)

        # annotate broadcast op operands
        bcast_op = transform.get_producer_of_operand(anytype, add_op, 0)
        xegpu.set_op_layout_attr(bcast_op, result=True, index=0, **output_layout)
        bcast_load = transform.get_producer_of_operand(anytype, bcast_op, 0)
        xegpu.set_op_layout_attr(
            bcast_load, result=True, index=0, **output_layout, slice_dims=[0]
        )
        output_layout_dim1 = {
            "sg_layout": [sg_layout[1]],
            "sg_data": [sg_tile[1]],
            "inst_data": [dpas_shape_c[1]],
        }
        offset = transform.get_producer_of_operand(anytype, bcast_load, 1)
        xegpu.set_op_layout_attr(offset, result=True, index=0, **output_layout_dim1)
        aux1 = transform.get_producer_of_operand(anytype, offset, 0)
        xegpu.set_op_layout_attr(aux1, result=True, index=0, **output_layout_dim1)
        aux2 = transform.get_producer_of_operand(anytype, offset, 1)
        xegpu.set_op_layout_attr(aux2, result=True, index=0, **output_layout_dim1)
        mask = transform.get_producer_of_operand(anytype, bcast_load, 2)
        xegpu.set_op_layout_attr(mask, result=True, index=0, **output_layout_dim1)
        raise NotImplementedError("Bias layout propagation is not supported.")
    transform.apply_cse(gpu_func)
    canonicalize(gpu_func)

    # hoist desc ops out of reduction loop
    transform.apply_licm(k_loop)

    canonicalize(gpu_func)
    transform.apply_cse(gpu_func)

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod


def bundle_xegpu_to_binary(mod, stop_at_stage: str = "") -> ir.Module:
    """Schedule for lowering xegpu wg level to binary."""
    # This schedule corresponds to upstream MLIR XeVM lowering pipeline
    # and is payload independent.

    # This pipeline causes performance regression with the existing
    # xegpu transform ops.
    # FIXME Use anchor layouts in transform ops.
    mod = apply_registered_pass(
        mod, "gpu-lower-to-xevm-pipeline", options={"xegpu-op-level": "workgroup"}
    )

    return mod
