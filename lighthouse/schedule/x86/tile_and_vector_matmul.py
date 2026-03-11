from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import loop
from mlir.dialects.transform import structured
from mlir.dialects.transform import tensor
from mlir.dialects.transform import vector
from mlir.dialects.transform import x86
from lighthouse.pipeline.helper import cleanup_func


def create(tile_size=64) -> ir.Module:
    """
    Specialized schedule for Linalg operations.

    Tiling and vectorization is progressively applied to
    achieve SIMD code generation.

    Returns:
        MLIR transform module.
    """
    # Create a transform module.
    schedule = ir.Module.create()
    schedule.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    with ir.InsertionPoint(schedule.body):
        named_seq = transform.NamedSequenceOp(
            "__transform_main",
            [transform.any_op_t()],
            [],
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
        )

    # Create the schedule.
    with ir.InsertionPoint(named_seq.body):
        anytype = transform.any_op_t()

        # GEMM tiling.
        gemm_name = "linalg.matmul"
        mm = structured.MatchOp.match_op_names(named_seq.bodyTarget, [gemm_name]).result
        structured.FuseOp(
            mm, tile_sizes=[tile_size, tile_size], apply_cleanup=True
        ).results[0]

        # Tile buffer initialization for better vectorization.
        tiled_fill = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, ["linalg.fill"]
        ).result
        reg_fill, *loops = structured.TileUsingForOp(tiled_fill, sizes=[1]).results

        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
            structured.apply_patterns_linalg_fold_pack_unpack_into_empty()
        cleanup_func(named_seq.bodyTarget)

        # Register tiling.
        reg_tile_m = 8
        reg_tile_n = 32
        reg_tile_k = 1
        gemm = structured.MatchOp.match_op_names(
            named_seq.bodyTarget, [gemm_name]
        ).result
        _, *gemm_loops = structured.TileUsingForOp(
            gemm,
            sizes=[reg_tile_m, reg_tile_n, reg_tile_k],
        ).results
        assert tile_size % reg_tile_k == 0, "Invalid K reg tiling"
        if tile_size % reg_tile_n != 0:
            loop.LoopPeelOp(
                anytype,
                anytype,
                gemm_loops[1],
                peel_front=False,
                fail_if_already_divisible=False,
            )
        if tile_size % reg_tile_m != 0:
            loop.LoopPeelOp(
                anytype,
                anytype,
                gemm_loops[0],
                peel_front=False,
                fail_if_already_divisible=False,
            )
        cleanup_func(named_seq.bodyTarget)

        # Register unroll.
        gemms = structured.MatchOp.match_op_names(named_seq.bodyTarget, [gemm_name])
        foreach_gemm = transform.ForeachOp([], (gemms,))
        with ir.InsertionPoint(foreach_gemm.body):
            gemm = foreach_gemm.bodyTargets[0]
            _, *loops = structured.TileUsingForOp(
                gemm, sizes=[1, reg_tile_n, 1]
            ).results
            loop.loop_unroll(loops[2], reg_tile_k)
            loop.loop_unroll(loops[0], reg_tile_m)
            transform.yield_()
        cleanup_func(named_seq.bodyTarget)

        # Vectorize operations.
        gemms = structured.MatchOp.match_op_names(named_seq.bodyTarget, [gemm_name])
        foreach_gemm = transform.ForeachOp([], (gemms,))
        with ir.InsertionPoint(foreach_gemm.body):
            gemm = foreach_gemm.bodyTargets[0]
            structured.structured_vectorize(gemm, [], create_named_contraction=True)
            transform.yield_()
        structured.structured_vectorize(reg_fill, [])
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            vector.apply_patterns_vector_reduction_to_contract()
            vector.apply_patterns_vector_transfer_permutation_patterns()
        cleanup_func(named_seq.bodyTarget)

        # Loop hoisting.
        all_loops = structured.MatchOp(
            anytype,
            named_seq.bodyTarget,
            interface=structured.MatchInterfaceEnum.LoopLikeInterface,
        ).results
        transform.apply_licm(all_loops)
        loop.loop_hoist_loop_invariant_subsets(all_loops)

        # Unroll GEMM.
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            vector.apply_patterns_vector_cast_away_vector_leading_one_dim()
            tensor.apply_patterns_tensor_fold_tensor_subset_ops_into_vector_transfers()
            transform.apply_patterns_canonicalization()

        # Lower to broadcast+FMA instructions.
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            x86.apply_patterns_x86_vector_contract_to_fma()
            x86.apply_patterns_x86_sink_vector_producer_ops()
            vector.apply_patterns_vector_flatten_vector_transfer_ops()
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            vector.apply_patterns_vector_lower_contraction(
                lowering_strategy=vector.VectorContractLowering.OuterProduct
            )
            vector.apply_patterns_vector_lower_outerproduct()
        cleanup_func(named_seq.bodyTarget)

        transform.yield_()

    schedule.body.operations[0].verify()
    return schedule
