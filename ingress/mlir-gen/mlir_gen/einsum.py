from typing import Union

from mlir import ir
from mlir.dialects import arith, linalg, tensor

from . import named, generic
from .utils import get_outputs, get_weights, get_bias, affine_map


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

    M, N, mb, nb = [ir.AffineDimExpr.get(i) for i in range(4)]
    affine_maps = {
        2: [affine_map(2, [N]), affine_map(2, [M, N]), affine_map(2, [M, N])],
        4: [
            affine_map(4, [N, nb]),
            affine_map(4, [M, N, mb, nb]),
            affine_map(4, [M, N, mb, nb]),
        ],
    }[inputs.type.rank]

    out_uninit = tensor.EmptyOp(inputs.type.shape, inputs.type.element_type)
    return linalg.elementwise(
        bias,
        inputs,
        outs=(out_uninit,),
        kind=linalg.ElementwiseKind.add,
        indexing_maps=affine_maps,
    )


def relu(inputs: ir.Value) -> ir.Value:
    zero = arith.constant(inputs.type.element_type, 0.0)
    out_uninit = tensor.EmptyOp(inputs.type.shape, inputs.type.element_type)
    out = linalg.fill(zero, outs=out_uninit)

    return linalg.elementwise(
        inputs,
        out,
        outs=(out_uninit,),
        kind=linalg.ElementwiseKind.max_signed,  # NB: on float args, gives arith.maximumf in body
    )


softmax = named.softmax
