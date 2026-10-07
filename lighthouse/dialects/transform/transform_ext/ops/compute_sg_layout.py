from collections.abc import Sequence

from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class ComputeSgLayoutOp(TransformExtensionDialect.Operation, name="compute_sg_layout"):
    """
    Compute xegpu `sg_layout` and `sg_data` from wg/sg/reduction tiles.

    Produces `2 * ndims` scalar params (`sg_layout` dims then `sg_data` dims).
    Per dim: `sg_layout = wg // sg` (or `red // sg`, else 1); `sg_data = sg`
    (else `red`, else 1).

    Tiles are passed via one `tiles` operand list, split by `num_wg_handles`/
    `num_sg_handles` into wg, sg, and (optional) reduction groups. Each operand
    may carry one i64 per dim or one i64 total. The rank is fixed at build time
    because `xegpu.set_anchor_layout` needs one scalar param per dim.

    A non-zero `transpose` reverses the wg dims before dividing; only valid for
    `ndims == 2`.

    Example:

        wg_tile, sg_tile, red_tile = tr_ext.infer_xegpu_reduction_params(anchor_op)
        sg_layout, sg_data = transform_ext.compute_sg_layout(
            wg_tile, sg_tile, red_tile
        )
        xegpu.set_anchor_layout(store_op, sg_layout=sg_layout, sg_data=sg_data)

    Args:
        transpose: i64 param; when non-zero, reverse the wg dims.
        tiles: wg, then sg, then optional reduction tile operand(s).
        num_wg_handles: Number of leading `tiles` operands forming the wg tile.
        num_sg_handles: Number of `tiles` operands (after wg) forming the sg tile.
    Returns:
        `2 * ndims` params: `sg_layout` dims then `sg_data` dims.
    """

    results_: Sequence[ext.Result[transform.AnyParamType]]
    transpose: ext.Operand[transform.AnyParamType]
    tiles: Sequence[ext.Operand[transform.AnyParamType]]
    num_wg_handles: ir.IntegerAttr
    num_sg_handles: ir.IntegerAttr

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ComputeSgLayoutOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            operands = list(op.tiles)
            num_wg = op.num_wg_handles.value
            num_sg = op.num_sg_handles.value

            def flatten(handles):
                return [
                    ir.IntegerAttr(a).value
                    for handle in handles
                    for a in state.get_params(handle)
                ]

            wg = flatten(operands[:num_wg])
            sg = flatten(operands[num_wg : num_wg + num_sg])
            red = flatten(operands[num_wg + num_sg :])

            ndims = len(op.results) // 2
            if min(len(wg), len(sg)) < ndims:
                return DiagnosedSilenceableFailure.SilenceableFailure

            if ir.IntegerAttr(state.get_params(op.transpose)[0]).value:
                # Reversing the dim order is a transpose only for rank-2 tiles.
                if ndims != 2:
                    op.location.emit_error(
                        "compute_sg_layout: transpose is only defined for rank-2 tiles"
                    )
                    return DiagnosedSilenceableFailure.SilenceableFailure
                wg = wg[:ndims][::-1] + wg[ndims:]

            # Reduction tile is optional; absent dims are untiled (0).
            red = (red + [0] * ndims)[:ndims]

            sg_layout = []
            sg_data = []
            for w, s, r in zip(wg[:ndims], sg[:ndims], red):
                layout = 1
                if r > 0 and s > 0:
                    layout = r // s
                if w > 0 and s > 0:
                    layout = w // s
                sg_layout.append(layout)
                # sg_data tracks the tiled dim: subgroup tile, else reduction.
                sg_data.append(s if s > 0 else (r if r > 0 else 1))

            i64 = ir.IntegerType.get_signless(64)
            for res, value in zip(op.results, sg_layout + sg_data):
                results.set_params(res, [ir.IntegerAttr.get(i64, value)])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ComputeSgLayoutOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def compute_sg_layout(
    wg_tile: ir.Value[transform.AnyParamType] | Sequence[ir.Value],
    sg_tile: ir.Value[transform.AnyParamType] | Sequence[ir.Value],
    reduction_tile: ir.Value[transform.AnyParamType] | Sequence[ir.Value] | None = None,
    ndims: int = 2,
    transpose: ir.Value[transform.AnyParamType] | int | None = None,
) -> tuple[list[ir.Value], list[ir.Value]]:
    """
    snake_case wrapper to create a ComputeSgLayoutOp.

    Each tile is a single handle (one i64 per dim) or a list of per-dim handles.
    `ndims` is the layout rank, fixed at build time (one scalar param per dim is
    required by `xegpu.set_anchor_layout`). `transpose` (i64 param or int)
    reverses the wg dims when non-zero; only valid for `ndims == 2`.

    Args:
        wg_tile: Workgroup tile param(s).
        sg_tile: Subgroup tile param(s).
        reduction_tile: Optional reduction tile param(s); omit if no reduction.
        ndims: Layout rank, fixed at build time.
        transpose: i64 param or int; when non-zero, reverse the wg dims.
    Returns:
        `(sg_layout, sg_data)`, each a list of `ndims` scalar params.
    """

    def as_list(tile):
        if tile is None:
            return []
        return list(tile) if isinstance(tile, (list, tuple)) else [tile]

    wg_handles = as_list(wg_tile)
    sg_handles = as_list(sg_tile)
    red_handles = as_list(reduction_tile)

    i64 = ir.IntegerType.get_signless(64)
    if not isinstance(transpose, ir.Value):
        transpose = transform.ParamConstantOp(
            transform.AnyParamType.get(), ir.IntegerAttr.get(i64, int(transpose or 0))
        ).param
    result_types = [transform.AnyParamType.get()] * (2 * ndims)
    op = ComputeSgLayoutOp(
        result_types,
        transpose=transpose,
        tiles=wg_handles + sg_handles + red_handles,
        num_wg_handles=ir.IntegerAttr.get(i64, len(wg_handles)),
        num_sg_handles=ir.IntegerAttr.get(i64, len(sg_handles)),
    )
    layout_results = list(op.results)
    return layout_results[:ndims], layout_results[ndims:]
