# RUN: %PYTHON %s | FileCheck %s

"""Tests for the infer_xegpu_attention_params transform op.

Two supported attention variants, distinguished by the rank of the leaf
elementwise op:
  * plain MHA: (batch, head, query_row, d_head)                    -- rank 4
  * GQA:       (batch, kv_head, group, query_row, d_head)          -- rank 5

The op tiles every leading parallel dim (including the GQA group) by 1 and the
query-row dim by the WG row size, leaving d_head untiled. Tiling the leaf with
the returned `wg_tile` surfaces those sizes as the `scf.forall` tile counts.
"""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


# Plain MHA leaf: (batch=2, head=3, query_row=256, d_head=64).
MHA_PAYLOAD = """
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%a: tensor<2x3x256x64xf32>) -> tensor<2x3x256x64xf32> {
    %e = tensor.empty() : tensor<2x3x256x64xf32>
    %g = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%a : tensor<2x3x256x64xf32>)
        outs(%e : tensor<2x3x256x64xf32>) {
    ^bb0(%i: f32, %o: f32):
      linalg.yield %i : f32
    } -> tensor<2x3x256x64xf32>
    return %g : tensor<2x3x256x64xf32>
  }
}
"""


# GQA leaf: (batch=2, kv_head=3, group=5, query_row=256, d_head=64).
GQA_PAYLOAD = """
#id = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module {
  func.func @main(%a: tensor<2x3x5x256x64xf32>) -> tensor<2x3x5x256x64xf32> {
    %e = tensor.empty() : tensor<2x3x5x256x64xf32>
    %g = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
        ins(%a : tensor<2x3x5x256x64xf32>)
        outs(%e : tensor<2x3x5x256x64xf32>) {
    ^bb0(%i: f32, %o: f32):
      linalg.yield %i : f32
    } -> tensor<2x3x5x256x64xf32>
    return %g : tensor<2x3x5x256x64xf32>
  }
}
"""

CONTRACT_PAYLOAD = """
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%a: vector<128x64xbf16>, %b: vector<64x64xbf16>,
      %c: vector<128x64xf32>) -> vector<128x64xf32> {
    %r = vector.contract {indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "parallel", "reduction"],
        kind = #vector.kind<add>} %a, %b, %c :
        vector<128x64xbf16>, vector<64x64xbf16> into vector<128x64xf32>
    return %r : vector<128x64xf32>
  }
}
"""

DPAS_PAYLOAD = """
module {
  func.func @main(%a: vector<128x64xbf16>, %b: vector<64x64xbf16>,
      %c: vector<128x64xf32>) -> vector<128x64xf32> {
    %r = xegpu.dpas %a, %b, %c :
        vector<128x64xbf16>, vector<64x64xbf16>, vector<128x64xf32>
        -> vector<128x64xf32>
    return %r : vector<128x64xf32>
  }
}
"""


def build_schedule(
    force_wg_tile=None,
    force_sg_tile=None,
    force_reduction_tile=None,
):
    with schedule_boilerplate() as (sched, named_seq):
        func = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        leaf = transform_ext.extract_handle(
            lh_transform.match_op(named_seq.bodyTarget, "linalg.generic"),
            -1,
            silenceable=True,
        )
        wg_tile, sg_tile, reduction_tile = transform_ext.infer_xegpu_attention_params(
            leaf,
            force_wg_tile=force_wg_tile,
            force_sg_tile=force_sg_tile,
            force_reduction_tile=force_reduction_tile,
        )
        # Tiling with wg_tile surfaces the per-dim sizes as forall tile counts.
        structured.TileUsingForallOp(leaf, num_threads=[], tile_sizes=wg_tile)
        # Expose individual tile sizes on the payload for FileCheck.
        for i in range(3):
            transform.annotate(
                func,
                f"wg_tile_{i}",
                param=transform_ext.extract_handle(wg_tile, i),
            )
            transform.annotate(
                func,
                f"sg_tile_{i}",
                param=transform_ext.extract_handle(sg_tile, i),
            )
        transform.annotate(
            func,
            "reduction_tile",
            param=transform_ext.extract_handle(reduction_tile, -1),
        )
        transform.yield_()
    return sched


def run(
    name,
    payload_str,
    force_wg_tile=None,
    force_sg_tile=None,
    force_reduction_tile=None,
):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        sched = build_schedule(
            force_wg_tile=force_wg_tile,
            force_sg_tile=force_sg_tile,
            force_reduction_tile=force_reduction_tile,
        )
        sched.body.operations[0].apply(payload.operation)
        print(payload)


def build_contract_schedule():
    with schedule_boilerplate() as (sched, named_seq):
        func = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        contract = transform_ext.extract_handle(
            lh_transform.match_op(named_seq.bodyTarget, "vector.contract"),
            0,
            silenceable=True,
        )
        wg_tile, sg_tile, reduction_tile = transform_ext.infer_xegpu_attention_params(
            contract
        )
        transform.annotate(
            func,
            "wg_tile",
            param=transform_ext.extract_handle(wg_tile, 0),
        )
        transform.annotate(
            func,
            "sg_tile",
            param=transform_ext.extract_handle(sg_tile, 0),
        )
        transform.annotate(
            func,
            "reduction_tile",
            param=transform_ext.extract_handle(reduction_tile, -1),
        )
        transform.yield_()
    return sched


def run_contract(name, payload_str):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        sched = build_contract_schedule()
        sched.body.operations[0].apply(payload.operation)
        print(payload)


def build_dpas_schedule():
    with schedule_boilerplate() as (sched, named_seq):
        func = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        dpas = transform_ext.extract_handle(
            lh_transform.match_op(named_seq.bodyTarget, "xegpu.dpas"),
            0,
            silenceable=True,
        )
        wg_tile, sg_tile, reduction_tile = transform_ext.infer_xegpu_attention_params(
            dpas
        )
        for name, tile in (
            ("wg_tile", wg_tile),
            ("sg_tile", sg_tile),
        ):
            for i in range(2):
                transform.annotate(
                    func,
                    f"{name}_{i}",
                    param=transform_ext.extract_handle(tile, i),
                )
        transform.annotate(
            func,
            "reduction_tile",
            param=transform_ext.extract_handle(reduction_tile, -1),
        )
        transform.yield_()
    return sched


def run_dpas(name, payload_str):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        sched = build_dpas_schedule()
        sched.body.operations[0].apply(payload.operation)
        print(payload)


# MHA: wg_tile [1, 1, 128] tiles (batch, head, query_row); query rows 256/128=2.
# d_head is left untiled. reduction (K/V seq) tile is 64 (on the func attr).
# CHECK-LABEL: Test: mha
# CHECK: reduction_tile = 64 : i64
# CHECK: scf.forall
# CHECK-SAME: in (2, 3, 2)
run("mha", MHA_PAYLOAD)

# Forced values replace each inferred tile. The forced workgroup tile
# [2, 3, 64] produces four query-row tiles; the other params are checked on the
# function attributes.
# CHECK-LABEL: Test: mha_forced
# CHECK-DAG: wg_tile_0 = 2 : i64
# CHECK-DAG: wg_tile_1 = 3 : i64
# CHECK-DAG: wg_tile_2 = 64 : i64
# CHECK-DAG: sg_tile_0 = 0 : i64
# CHECK-DAG: sg_tile_1 = 0 : i64
# CHECK-DAG: sg_tile_2 = 8 : i64
# CHECK-DAG: reduction_tile = 32 : i64
# CHECK: scf.forall
# CHECK-SAME: in (1, 1, 4)
run(
    "mha_forced",
    MHA_PAYLOAD,
    force_wg_tile=[2, 3, 64],
    force_sg_tile=[0, 0, 8],
    force_reduction_tile=[0, 0, 0, 32],
)


# GQA: wg_tile [1, 1, 1, 128] additionally tiles the group dim by 1, so each
# workgroup sees plain attention; query rows 256/128=2, d_head untiled.
# CHECK-LABEL: Test: gqa
# CHECK: reduction_tile = 64 : i64
# CHECK: scf.forall
# CHECK-SAME: in (2, 3, 5, 2)
run("gqa", GQA_PAYLOAD)

# The first vector.contract in foo.mlir has result shape (128, 64). Inference
# reduces that to a 1D workgroup tile [128] and reduction tile [0, 64].
# CHECK-LABEL: Test: infer_xegpu_attention_params_contract
# CHECK-DAG: wg_tile = 128 : i64
# CHECK-DAG: sg_tile = 16 : i64
# CHECK-DAG: reduction_tile = 64 : i64
run_contract("infer_xegpu_attention_params_contract", CONTRACT_PAYLOAD)

# The first xegpu.dpas in bar.mlir has a (128, 64) result and (128, 64)
# operand A, so it infers wg=[128, 1], sg=[16, 0], and red=[0, 64].
# CHECK-LABEL: Test: infer_xegpu_attention_params_dpas
# CHECK-DAG: wg_tile_0 = 128 : i64
# CHECK-DAG: wg_tile_1 = 1 : i64
# CHECK-DAG: sg_tile_0 = 16 : i64
# CHECK-DAG: sg_tile_1 = 0 : i64
# CHECK-DAG: reduction_tile = 64 : i64
run_dpas("infer_xegpu_attention_params_dpas", DPAS_PAYLOAD)
