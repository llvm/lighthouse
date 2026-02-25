from mlir import ir
from mlir.dialects.transform import loop
from mlir.dialects.transform import bufferization
from mlir.dialects.transform import xegpu
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects import transform
from mlir.dialects.transform import structured
from lighthouse.utils.mlir import (
    apply_registered_pass,
    canonicalize,
    match,
)
from typing import Optional


class PipelineInterrupt(Exception):
    """Exception to signal early termination of the transform schedule."""

    pass


def match_and_split(*args, nhandles=1, **kwargs):
    """Henper function that splits matched handles."""
    matched = match(*args, **kwargs)
    anytype = transform.AnyOpType.get()
    matched_ops = transform.split_handle((anytype,) * nhandles, matched)
    if nhandles == 1:
        matched_ops = [matched_ops]
    return matched_ops


# hardware constraints
dpas_tile = [8, 16, 16]
prefetch_inst_data = [8, 16]
nb_workitems = 16  # workitems in subgroup


def get_schedule_module(
    has_bias: bool = False,
    has_relu: bool = False,
    has_convert_c: bool = True,
    skip_final_layer_relu: bool = False,
    stop_at_stage: str = "",
    nlayers: int = 1,
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
            xegpu_mlp_transform_schedule(
                payload_mod,
                has_bias=has_bias,
                has_relu=has_relu,
                has_convert_c=has_convert_c,
                skip_final_layer_relu=skip_final_layer_relu,
                stop_at_stage=stop_at_stage,
                nlayers=nlayers,
                params=params,
            )

    return mod


def xegpu_mlp_transform_schedule(
    mod: ir.Value,
    has_bias: bool = False,
    has_relu: bool = False,
    has_convert_c: bool = True,
    skip_final_layer_relu: bool = False,
    stop_at_stage: str = "",
    nlayers: int = 1,
    params: Optional[list[dict]] = None,
):
    """Transform schedule for matmul-like payload."""
    try:
        mod = bundle_xepu_mlp_schedule(
            mod,
            has_bias=has_bias,
            has_relu=has_relu,
            has_convert_c=has_convert_c,
            skip_final_layer_relu=skip_final_layer_relu,
            stop_at_stage=stop_at_stage,
            nlayers=nlayers,
            params=params,
        )

        mod = bundle_xegpu_to_binary(
            mod,
            stop_at_stage=stop_at_stage,
        )
    except PipelineInterrupt:
        pass
    finally:
        transform.yield_()


def bundle_xepu_mlp_schedule(
    mod: ir.Value,
    has_bias: bool = False,
    has_relu: bool = False,
    skip_final_layer_relu: bool = False,
    has_convert_c: bool = True,
    stop_at_stage: str = "",
    nlayers: int = 1,
    params: Optional[list[dict]] = None,
) -> ir.Module:
    """Schedule for lowering matmul-like payload to xegpu wg level."""
    if params is None:
        raise ValueError("Schedule parameters must be provided.")

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()
    anyvalue = transform.AnyValueType.get()

    for i in range(nlayers):
        assert f"layer_{i}" in params, f"Missing parameters for 'layer_{i}'"

    dpas_shape_a = [dpas_tile[0], dpas_tile[2]]
    dpas_shape_b = [dpas_tile[2], dpas_tile[1]]
    dpas_shape_c = [dpas_tile[0], dpas_tile[1]]

    # wg tiling
    if has_convert_c:
        trunc_op = match(mod, ops={"arith.truncf"})
        terminal = transform.get_parent_op(anytype, trunc_op)
        # split handle for each layer
        terminal_ops = transform.split_handle((anytype,) * nlayers, terminal)
        if nlayers == 1:
            terminal_ops = [terminal_ops]
    elif has_bias:
        terminal_ops = match_and_split(mod, ops={"linalg.add"}, nhandles=nlayers)
    else:
        terminal_ops = match_and_split(mod, ops={"linalg.matmul"}, nhandles=nlayers)
    if has_relu:
        if skip_final_layer_relu and nlayers > 1:
            # intermediate layers have relu activation function
            relu_ops = match_and_split(mod, ops={"linalg.max"}, nhandles=nlayers - 1)
            # the final layer does not have relu
            terminal_ops = list(relu_ops) + [terminal_ops[-1]]
        else:
            # relu on all layers
            terminal_ops = match_and_split(mod, ops={"linalg.max"}, nhandles=nlayers)

    # tile each layer separately
    for i_layer in range(nlayers):
        layer_params = params[f"layer_{i_layer}"]
        # tunable parameters: wg level tiling
        wg_tile = [layer_params["wg_m"], layer_params["wg_n"]]
        sg_tile = [layer_params["sg_m"], layer_params["sg_n"]]
        k_tile = layer_params["k"]

        terminal = terminal_ops[i_layer]
        # FIXME use structured.structured_fuse
        _, wg_loop = structured.FuseOp(
            terminal, tile_sizes=wg_tile, use_forall=True
        ).results
        transform.apply_cse(mod)
        canonicalize(mod)

        # k loop tiling
        wg_matmul = match(wg_loop, ops={"linalg.matmul"})
        # FIXME use structured.structured_tile_using_for
        wgk_matmul, k_loop = structured.TileUsingForOp(
            wg_matmul, sizes=[0, 0, k_tile]
        ).results

    func = transform.get_parent_op(
        anytype,
        k_loop,
        op_name="func.func",
        deduplicate=True,
    )
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
    wg_loops = match_and_split(mod, ops={"scf.forall"}, nhandles=nlayers)
    for wg_loop in wg_loops:
        wg_loop = loop.loop_forall_to_parallel([anytype], wg_loop)
    func = transform.get_parent_op(anytype, wg_loop)

    # convert to scf.parallel to gpu.launch
    func = apply_registered_pass(func, "gpu-map-parallel-loops")
    func = apply_registered_pass(func, "convert-parallel-loops-to-gpu")
    func = apply_registered_pass(func, "lower-affine")
    transform.apply_cse(func)
    canonicalize(func)

    # set correct number of gpu threads
    launch_ops = match_and_split(mod, ops={"gpu.launch"}, nhandles=nlayers)
    for i_layer, launch_op in enumerate(launch_ops):
        layer_params = params[f"layer_{i_layer}"]
        # tunable parameters
        wg_tile = [layer_params["wg_m"], layer_params["wg_n"]]
        sg_tile = [layer_params["sg_m"], layer_params["sg_n"]]

        # derived parameters
        sg_layout = [wg_tile[0] // sg_tile[0], wg_tile[1] // sg_tile[1]]
        # number of threads collapsed to 1d layout
        nb_threads = sg_layout[0] * sg_layout[1] * nb_workitems

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
    gpu_mod_ops = match_and_split(mod, ops={"gpu.module"}, nhandles=nlayers)
    for gpu_mod in gpu_mod_ops:
        gpu_func = match(gpu_mod, ops={"gpu.func"})
        gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
        transform.apply_cse(gpu_func)

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    for i_layer, gpu_mod in enumerate(gpu_mod_ops):
        gpu_func = match(gpu_mod, ops={"gpu.func"})

        # tunable parameters: xegpu layout
        layer_params = params[f"layer_{i_layer}"]

        wg_tile = [layer_params["wg_m"], layer_params["wg_n"]]
        sg_tile = [layer_params["sg_m"], layer_params["sg_n"]]
        k_tile = layer_params["k"]

        sg_layout = [wg_tile[0] // sg_tile[0], wg_tile[1] // sg_tile[1]]

        load_tile_a = [layer_params["load_a_m"], layer_params["load_a_k"]]
        load_tile_b = [layer_params["load_b_k"], layer_params["load_b_n"]]
        prefetch_tile_a = [layer_params["pf_a_m"], layer_params["pf_a_k"]]
        prefetch_tile_b = [layer_params["pf_b_k"], layer_params["pf_b_n"]]
        nb_prefetch = layer_params["pf_nb"]

        prefetch_layout_a = [
            wg_tile[0] // prefetch_tile_a[0],
            k_tile // prefetch_tile_a[1],
        ]
        prefetch_layout_b = [
            k_tile // prefetch_tile_b[0],
            wg_tile[1] // prefetch_tile_b[1],
        ]

        # matmul matrix shapes
        sg_tile_a = [sg_tile[0], k_tile]
        sg_tile_b = [k_tile, sg_tile[1]]

        # add layouts to DPAS op operands
        k_loop = match(gpu_func, ops={"scf.for"})
        dpas_op = match(k_loop, ops={"xegpu.dpas"})
        tile_a = transform.get_operand(anyvalue, dpas_op, [0])
        tile_b = transform.get_operand(anyvalue, dpas_op, [1])

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
        layout_prefetch_a = {
            "sg_layout": prefetch_layout_a,
            "sg_data": prefetch_tile_a,
            "inst_data": prefetch_inst_data,
        }
        pf_ops = transform.get_consumers_of_result(anytype, desc_prefetch_a, 0)
        for pf in transform.split_handle((anytype,) * (nb_prefetch + 1), pf_ops):
            xegpu.set_op_layout_attr(pf, **layout_prefetch_a)

        desc_prefetch_b = xegpu.insert_prefetch(
            tile_b,
            nb_prefetch=nb_prefetch,
        )
        layout_prefetch_b = {
            "sg_layout": prefetch_layout_b,
            "sg_data": prefetch_tile_b,
            "inst_data": prefetch_inst_data,
        }
        pf_ops = transform.get_consumers_of_result(anytype, desc_prefetch_b, 0)
        for pf in transform.split_handle((anytype,) * (nb_prefetch + 1), pf_ops):
            xegpu.set_op_layout_attr(pf, **layout_prefetch_b)

        # A tile load layout
        layout_load_a = {
            "sg_layout": sg_layout,
            "sg_data": sg_tile_a,
            "inst_data": load_tile_a,
        }
        desc_op_a = xegpu.get_desc_op(tile_a)
        # A tile load op anchor layout
        load_op_a = transform.get_consumers_of_result(anytype, desc_op_a, 0)
        xegpu.set_op_layout_attr(load_op_a, **layout_load_a)
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
        # B tile load op anchor layout
        load_op_b = transform.get_consumers_of_result(anytype, desc_op_b, 0)
        xegpu.set_op_layout_attr(load_op_b, **layout_load_b)
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
        # C tile dpas anchor layout
        xegpu.set_op_layout_attr(dpas_op, index=0, **layout_dpas_a)
        xegpu.set_op_layout_attr(dpas_op, index=1, **layout_dpas_b)
        xegpu.set_op_layout_attr(dpas_op, index=2, **output_layout)
        # annotate store op
        store_op_c = match(gpu_func, ops={"xegpu.store_nd"})
        xegpu.set_op_layout_attr(store_op_c, **output_layout)

        if has_bias:
            # annotate the 1d load of the broadcast op with a slice layout
            add_op = match(gpu_func, ops={"arith.addf"})
            bcast_op = transform.get_producer_of_operand(anytype, add_op, 0)
            bcast_load = transform.get_producer_of_operand(anytype, bcast_op, 0)
            xegpu.set_op_layout_attr(
                bcast_load, result=True, index=0, **output_layout, slice_dims=[0]
            )
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
    # upstream xegpu/xevm pipeline is payload independent.
    mod = apply_registered_pass(
        mod, "gpu-lower-to-xevm-pipeline", options={"xegpu-op-level": "workgroup"}
    )

    return mod
