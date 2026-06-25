from time import perf_counter
import multiprocessing
import sys
import json
from csv_logger import CSVLogger
from lighthouse.pipeline.helper import PipelineInterrupt

import numpy as np

from mlir import ir
from lighthouse import dialects as lh_dialects
from lighthouse.execution.runner import Runner
from lighthouse.pipeline.driver import TransformDriver
from matmul import XeGPUMatMul, check_results


def dump_configs_json(
    param_list: list[dict] | dict, filename_prefix: str = "matmul_params"
):
    print("\nSaving parameters:")
    if isinstance(param_list, dict):
        filename = f"{filename_prefix}.json"
        params = param_list
        with open(filename, "w") as f:
            json.dump(params, f, indent=4)
        print(f"  {filename}")
        return
    for i, params in enumerate(param_list):
        filename = f"{filename_prefix}_{i:02d}.json"
        with open(filename, "w") as f:
            json.dump(params, f, indent=4)
        print(f"  {filename}")


def run_with_timeout(
    experiment_func: callable, *args, timeout: int = 20, **kwargs
) -> dict:
    """
    Wrapper to execute the experiment with a new thread and a timeout.

    This is require to recover from, e.g., IGC compiler failures.
    """

    def wrapped(res, *args, **kwargs):
        try:
            entry = experiment_func(*args, **kwargs)
            res.update(entry)
        except PipelineInterrupt:
            res["error"] = "Pipeline interrupted at specified stage."
        except Exception as e:
            if kwargs.get("debug", False):
                raise e
            err_msg = str(e)
            print(f"Experiment failed with error: {err_msg}")
            msg_lines = err_msg.split("\n")
            err_lines = [s for s in msg_lines if "error:" in s.lower()]
            if err_lines:
                err_msg = err_lines[-1]  # last error line
            else:
                err_msg = msg_lines[0]  # fallback to first line
            res["error"] = err_msg

    with multiprocessing.Manager() as manager:
        # create a shared dict results back from the child process
        res = manager.dict()
        res["time"] = 0.0
        res["throughput"] = 0.0
        res["error"] = ""
        all_args = tuple([res] + list(args))
        proc = multiprocessing.Process(target=wrapped, args=all_args, kwargs=kwargs)
        proc.start()
        proc.join(timeout)
        if proc.is_alive():
            print("TIMEOUT")
            proc.kill()
            proc.join()
            return 0, "TIMEOUT"
        if proc.exitcode != 0:
            if res["error"] == "":
                res["error"] = f"Execution failed with exit code: {proc.exitcode}"
                print(res["error"])
        results = dict(res)
        proc.close()

    return results


def execute_and_log(
    experiment_func: callable,
    params: dict,
    csv_logger: CSVLogger | None = None,
    timeout: int = 20,
    **kwargs,
) -> tuple[float, float]:
    entry = params.copy()
    elapsed, gflops = 0, 0
    try:
        tic = perf_counter()
        results = run_with_timeout(
            experiment_func=experiment_func,
            timeout=timeout,
            **params,
            **kwargs,
        )
        elapsed = results.get("time", 0)
        gflops = results.get("throughput", 0)
        duration = perf_counter() - tic
        entry["time (us)"] = elapsed
        entry["GFLOPS"] = gflops
        if csv_logger is not None:
            csv_logger.log(entry)
        print(f"Duration: {duration:.3f} s")
    except Exception as e:
        print("FAILED")
        print(entry)
        print(f"  Error: {e}")
    sys.stdout.flush()
    return elapsed, gflops


def execute_matmul(
    ab_type: str = "f16",
    c_type: str = "f32",
    nruns: int = None,
    nwarmup: int = None,
    check_result: bool = False,
    has_bias: bool = False,
    has_relu: bool = False,
    accumulate_c: bool = True,
    truncate_c: bool = False,
    **params,
) -> dict[str, float]:
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()

        wload = XeGPUMatMul(
            M=params["m"],
            N=params["n"],
            K=params["k"],
            ab_type=ab_type,
            c_type=ab_type if truncate_c else c_type,
            transpose_a=params["transpose_a"],
            transpose_b=params["transpose_b"],
            has_bias=has_bias,
            has_relu=has_relu,
            accumulate_c=accumulate_c,
            truncate_c=truncate_c,
        )
        pipeline = TransformDriver(wload.schedule_modules(parameters=params))
        payload = pipeline.apply(wload.payload_module())

        runner = Runner(
            payload,
            mem_manager_cls=wload.memory_manager_class,
            shared_libs=wload.shared_libs(),
        )
        if check_result:
            # Setup callback function to copy result from device to host.
            D_host_copy = np.zeros(wload.c_shape, dtype=wload.c_dtype)
            argument_access_callback = Runner.get_gpu_argument_access_callback(
                D_host_copy, arg_index=0
            )
            host_inputs = wload.get_input_arrays(init_int=True)
            runner.execute(
                host_input_buffers=host_inputs,
                payload_function_name=wload.payload_function_name,
                argument_access_callback=argument_access_callback,
            )
            success = check_results(
                wload,
                host_inputs,
                D_host_copy,
                verbose=1,
            )
            if not success:
                raise ValueError("Result mismatch!")
        host_inputs = wload.get_input_arrays()
        if nruns is None and nwarmup is None:
            # first run to estimate cost
            times = runner.benchmark(
                host_input_buffers=host_inputs, nruns=10, nwarmup=10
            )
            # estimate number of runs
            cost = times.mean()
            warmup_target = 0.25
            nwarmup = max(int(warmup_target / cost), 10)
            nruns = 3 * nwarmup
            print(f"{nwarmup=} {nruns=}")
        # benchmark
        times = runner.benchmark(
            host_input_buffers=host_inputs, nruns=nruns, nwarmup=nwarmup
        )

    times *= 1e6  # convert to microseconds
    elapsed = np.mean(times)
    flop_count = wload.get_complexity()[0]
    gflops = flop_count / (elapsed * 1e-6) / 1e9

    return {"time": elapsed, "throughput": gflops}
