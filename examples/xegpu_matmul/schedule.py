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


def bundle_xepu_matmul_schedule(
    mod,
    has_bias: bool = False,
    has_relu: bool = False,
    stop_at_stage: str = "",
    params: Optional[dict] = None,
) -> ir.Module:
    """Schedule for lowering matmul-like payload to xegpu wg level."""
    if params is None:
        raise ValueError("Schedule parameters must be provided.")

    # tunable parameters
    wg_tile = [params["auto_wg_d0"], params["auto_wg_d1"]]
    sg_tile = [params["auto_sg_d0"], params["auto_sg_d1"]]
    k_tile = params["auto_k"]

    load_tile_a = [params["auto_load_a_d0"], params["auto_load_a_d1"]]
    load_tile_b = [params["auto_load_b_d0"], params["auto_load_b_d1"]]

    prefetch_tile_a = [params["auto_prefetch_a_d0"], params["auto_prefetch_a_d1"]]
    prefetch_tile_b = [params["auto_prefetch_b_d0"], params["auto_prefetch_b_d1"]]
    nb_prefetch = params["auto_nb_prefetch"]

    # derived parameters
    sg_layout = [wg_tile[0] // sg_tile[0], wg_tile[1] // sg_tile[1]]
    # number of threads collapsed to 1d layout
    nb_threads = sg_layout[0] * sg_layout[1] * nb_workitems
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

    # TODO applying gpu-lower-to-xevm-pipeline pass affects performance
    # mod = apply_registered_pass(
    #     mod, "gpu-lower-to-xevm-pipeline", options={"xegpu-op-level": "workgroup"}
    # )

    gpu_mod = match(mod, ops={"gpu.module"})
    # xegpu distribution
    gpu_func = match(gpu_mod, ops={"gpu.func"})
    gpu_func = apply_registered_pass(gpu_func, "xegpu-wg-to-sg-distribute")
    transform.apply_cse(gpu_func)

    if stop_at_stage == "xegpu-sg":
        raise PipelineInterrupt()

    gpu_func = apply_registered_pass(gpu_func, "lower-affine")
    transform.apply_cse(gpu_func)
    gpu_func = apply_registered_pass(gpu_func, "xegpu-blocking")
    canonicalize(gpu_func)
    transform.apply_cse(gpu_func)

    if stop_at_stage == "xegpu-inst":
        raise PipelineInterrupt()

    gpu_func = apply_registered_pass(gpu_func, "xegpu-propagate-layout")
    gpu_mod = apply_registered_pass(gpu_mod, "xegpu-subgroup-distribute")
    canonicalize(gpu_mod)
    transform.apply_cse(gpu_mod)
    gpu_mod = apply_registered_pass(gpu_mod, "loop-invariant-code-motion")
    transform.apply_cse(gpu_mod)
    gpu_mod = apply_registered_pass(gpu_mod, "xegpu-vector-linearize")
    gpu_mod = apply_registered_pass(gpu_mod, "convert-xegpu-to-xevm")
    gpu_mod = apply_registered_pass(
        gpu_mod, "convert-gpu-to-llvm-spv", options={"use-64bit-index": "true"}
    )
    gpu_mod = apply_registered_pass(gpu_mod, "convert-xevm-to-llvm")
    transform.apply_cse(gpu_mod)

    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(func, "gpu-async-region")

    mod = apply_registered_pass(mod, "reconcile-unrealized-casts")
    mod = apply_registered_pass(mod, "convert-vector-to-scf")
    mod = apply_registered_pass(mod, "convert-scf-to-cf")
    mod = apply_registered_pass(mod, "expand-strided-metadata")
    mod = apply_registered_pass(mod, "finalize-memref-to-llvm")
    mod = apply_registered_pass(mod, "convert-cf-to-llvm")
    mod = apply_registered_pass(mod, "convert-vector-to-llvm")
    mod = apply_registered_pass(mod, "convert-arith-to-llvm")
    mod = apply_registered_pass(mod, "convert-index-to-llvm")
    mod = apply_registered_pass(mod, "convert-func-to-llvm")
    mod = apply_registered_pass(mod, "convert-math-to-llvm")
    mod = apply_registered_pass(mod, "gpu-to-llvm")
    mod = apply_registered_pass(mod, "lower-affine")
    mod = apply_registered_pass(mod, "reconcile-unrealized-casts")
    transform.apply_cse(mod)
    mod = apply_registered_pass(mod, "gpu-module-to-binary")

    return mod
