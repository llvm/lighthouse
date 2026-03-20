# RUN: %PYTHON %s | lh-tune - -n 1 | FileCheck %s
# RUN: %PYTHON %s | lh-tune - -n 2147483647 --count-only | FileCheck %s --check-prefix=ENUM-CHECK
# ENUM-CHECK: count: 192

"""Enumerate concrete schedules given a schedule with tunable parameters."""

from mlir import ir

from lighthouse import dialects as lh_dialects

from matmul import XeGPUMatMul

with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()

    wload = XeGPUMatMul(512, 512, 512)
    parameters = {
        "m": wload.M,
        "n": wload.N,
        "k": wload.K,
        # CHECK: knob<"layer_0_wg_m"> = {{[0-9]+}} : i64 from options
        "wg_m": None,  # Passing `None`s generates `transform.tune.knob` ops.
        # CHECK: knob<"layer_0_wg_n"> = {{[0-9]+}} : i64 from options
        "wg_n": None,
        # CHECK: knob<"layer_0_sg_m"> = {{[0-9]+}} : i64 from options
        "sg_m": None,
        # CHECK: knob<"layer_0_sg_n"> = {{[0-9]+}} : i64 from options
        "sg_n": None,
        # CHECK: knob<"layer_0_k_tile"> = {{[0-9]+}} : i64 from options
        "k_tile": None,
        # CHECK: knob<"layer_0_load_a_m"> = {{[0-9]+}} : i64 from options
        "load_a_m": None,
        # CHECK: knob<"layer_0_load_a_k"> = {{[0-9]+}} : i64 from options
        "load_a_k": None,
        # CHECK: knob<"layer_0_load_b_k"> = {{[0-9]+}} : i64 from options
        "load_b_k": None,
        # CHECK: knob<"layer_0_load_b_n"> = {{[0-9]+}} : i64 from options
        "load_b_n": None,
        # Setting concrete values to shrink search space.
        # NB(RM): Enumerating whole space takes around 25 seconds for me.
        "prefetch_a_m": 16,
        "prefetch_a_k": 16,
        "prefetch_b_k": 16,
        "prefetch_b_n": 16,
        "prefetch_nb": 1,
    }

    # Check that at least one constraint was reified into the schedule.
    # CHECK: constrain_params
    schedules = wload.schedule_modules(parameters=parameters)
    assert len(schedules) == 2, "Expected two schedule modules to be returned"

    print(schedules[1])
