from time import perf_counter
import multiprocessing
from multiprocessing.sharedctypes import Value
from ctypes import c_double
import sys
import json
from csv_logger import CSVLogger


def dump_configs_json(param_list: list[dict], filename_prefix: str = "matmul_params"):
    print("\nSaving parameters:")
    for i, params in enumerate(param_list):
        filename = f"{filename_prefix}_{i:02d}.json"
        with open(filename, "w") as f:
            json.dump(params, f, indent=4)
        print(f"  {filename}")


def run_with_timeout(
    experiment_func: callable, *args, timeout: int = 20, **kwargs
) -> tuple[float, float]:
    """
    Wrapper to execute the experiment with a new thread and a timeout.

    Experiments must be run in a new process to ensure reliable timings.

    Sends kill signal if timeout is reached.
    """
    # wrap return values
    timing = Value(c_double, 0.0)
    gflops = Value(c_double, 0.0)

    def wrapped(timing, gflops, *args, **kwargs):
        res = experiment_func(*args, **kwargs)
        timing.value = res[0]
        gflops.value = res[1]

    all_args = tuple([timing, gflops] + list(args))
    proc = multiprocessing.Process(target=wrapped, args=all_args, kwargs=kwargs)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        print("TIMEOUT")
        proc.kill()
        proc.join()
        return 0, 0
    proc.close()
    return timing.value, gflops.value


def execute_and_log(
    experiment_func: callable,
    csv_logger: CSVLogger,
    nruns: int,
    nwarmup: int,
    params: dict,
    check_result: bool = True,
    ab_type: str = "f16",
    c_type: str = "f32",
    has_bias: bool = False,
    has_relu: bool = False,
    accumulate_c: bool = True,
    timeout: int = 20,
) -> tuple[float, float]:
    entry = params.copy()
    elapsed, gflops = 0, 0
    try:
        tic = perf_counter()
        elapsed, gflops = run_with_timeout(
            experiment_func=experiment_func,
            ab_type=ab_type,
            c_type=c_type,
            nruns=nruns,
            nwarmup=nwarmup,
            check_result=check_result,
            timeout=timeout,
            has_bias=has_bias,
            has_relu=has_relu,
            accumulate_c=accumulate_c,
            **params,
        )
        duration = perf_counter() - tic
        entry["time (us)"] = elapsed
        entry["GFLOPS/s"] = gflops
        csv_logger.log(entry)
        print(f"Duration: {duration:.3f} s")
    except Exception as e:
        print("FAILED")
        print(entry)
        print(f"  Error: {e}")
    sys.stdout.flush()
    return elapsed, gflops
