from typing import Union

from mlir import ir
from mlir.dialects import arith, linalg, tensor

from . import named, generic
from .utils import get_outputs, get_weights, get_bias


def times_weights(
    inputs: ir.Value,
    weights_or_weights_type: Union[ir.Value, ir.RankedTensorType],
    outputs_or_outputs_type: Union[ir.Value, ir.RankedTensorType],
) -> ir.Value:
    weights: ir.Value = get_weights(weights_or_weights_type)
    outputs: ir.Value = get_outputs(outputs_or_outputs_type)

    affine_maps, _ = generic.affine_maps_and_iter_types(weights.type.rank)

    if weights.type.rank == 5:  # tiled weights with vnni blocking
        vnni_block = weights.type.get_dim_size(4)
        assert inputs.type.shape[-1] % vnni_block == 0

        expanded_shape = (
            inputs.type.shape[:-1]
            + [inputs.type.shape[-1] // vnni_block]
            + [vnni_block]
        )
        expanded_type = ir.RankedTensorType.get(
            expanded_shape, inputs.type.element_type
        )
        inputs = tensor.expand_shape(
            expanded_type,
            inputs,
            reassociation=[[0], [1], [2], [3, 4]],
            output_shape=[],
            static_output_shape=expanded_shape,
        )

    return linalg.contract(inputs, weights, outs=[outputs], indexing_maps=affine_maps)


def add_bias(inputs: ir.Value, bias_or_bias_type: Union[ir.Value, ir.Type]) -> ir.Value:
    bias: ir.Value = get_bias(bias_or_bias_type)

    if inputs.type.rank == 2:
        bcast_map = ir.AffineMap.get(2, 0, [ir.AffineDimExpr.get(0)])
    elif inputs.type.rank == 4:
        bcast_map = ir.AffineMap.get(
            4, 0, [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(2)]
        )
    else:
        assert False

    out_uninit = tensor.EmptyOp(inputs.type.shape, inputs.type.element_type)
    return linalg.elementwise(
        bias,
        inputs,
        outs=(out_uninit,),
        kind=linalg.ElementwiseKind.add,
        indexing_maps=[bcast_map, ir.AffineMap.get_identity(inputs.type.rank)],
    )


def relu(inputs: ir.Value) -> ir.Value:
    zero = arith.constant(inputs.type.element_type, 0.0)
    out_uninit = tensor.EmptyOp(inputs.type.shape, inputs.type.element_type)
    out = linalg.fill(zero, outs=out_uninit)
    return linalg.elementwise(
        inputs,
        out,
        outs=(out_uninit,),
        kind=linalg.ElementwiseKind.maximumf,
    )


softmax = named.softmax
