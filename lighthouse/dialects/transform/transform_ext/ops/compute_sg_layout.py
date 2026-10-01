from collections.abc import Sequence

from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class ComputeSgLayoutOp(TransformExtensionDialect.Operation, name="compute_sg_layout"):
    """
    Compute the xegpu `sg_layout` and `sg_data` from wg/sg/reduction tiles.

    Produces `2 * ndims` scalar params: the first `ndims` are the subgroup
    layout (subgroups per dim), the next `ndims` are the subgroup data shape
    (elements per subgroup per dim). Per dimension `i`:

        sg_layout[i] = wg_i // sg_i   if wg_i and sg_i are tiled
                     = red_i // sg_i  elif red_i and sg_i are tiled
                     = 1              otherwise
        sg_data[i]   = sg_i           if sg_i is tiled
                     = red_i          elif red_i is tiled
                     = 1              otherwise

    The output rank is fixed at build time (one result per dim) because
    `set_anchor_layout` needs a static-arity size list; the tile values are read
    at apply time.

    Args:
        wg_tile: Param holding the workgroup tile sizes (one i64 per dim).
        sg_tile: Param holding the subgroup tile sizes (one i64 per dim).
        reduction_tile: Param holding the reduction tile sizes (one i64 per dim).
    Return:
        `2 * ndims` params: `sg_layout` dims followed by `sg_data` dims.
    """

    results_: Sequence[ext.Result[transform.AnyParamType]]
    wg_tile: ext.Operand[transform.AnyParamType]
    sg_tile: ext.Operand[transform.AnyParamType]
    reduction_tile: ext.Operand[transform.AnyParamType]

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
            wg = [ir.IntegerAttr(a).value for a in state.get_params(op.wg_tile)]
            sg = [ir.IntegerAttr(a).value for a in state.get_params(op.sg_tile)]
            red = [ir.IntegerAttr(a).value for a in state.get_params(op.reduction_tile)]

            ndims = len(op.results) // 2
            if min(len(wg), len(sg), len(red)) < ndims:
                return DiagnosedSilenceableFailure.SilenceableFailure

            sg_layout = []
            sg_data = []
            for w, s, r in zip(wg[:ndims], sg[:ndims], red[:ndims]):
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
    wg_tile: ir.Value[transform.AnyParamType],
    sg_tile: ir.Value[transform.AnyParamType],
    reduction_tile: ir.Value[transform.AnyParamType],
    ndims: int = 2,
) -> tuple[list[ir.Value], list[ir.Value]]:
    """
    snake_case wrapper to create a ComputeSgLayoutOp.

    Args:
        wg_tile: Param holding the workgroup tile sizes (one i64 per dim).
        sg_tile: Param holding the subgroup tile sizes (one i64 per dim).
        reduction_tile: Param holding the reduction tile sizes (one i64 per dim).
        ndims: Rank of the produced layout (fixed at build time).
    Return:
        Tuple `(sg_layout, sg_data)`, each a list of `ndims` scalar params ready
        to pass to `xegpu.set_anchor_layout`.
    """
    result_types = [transform.AnyParamType.get()] * (2 * ndims)
    op = ComputeSgLayoutOp(
        result_types,
        wg_tile=wg_tile,
        sg_tile=sg_tile,
        reduction_tile=reduction_tile,
    )
    layout_results = list(op.results)
    return layout_results[:ndims], layout_results[ndims:]
