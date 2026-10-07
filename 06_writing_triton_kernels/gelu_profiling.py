import time
from typing import Callable

import torch
from torch.profiler import ProfilerActivity

from gpu_util import cuda_if_available


def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    # Warm up because the first few runs may be slower due to compilation and other overhead.
    # Measure steady-state performance because the kernel will run multiple times.
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()  # Wait for all CUDA work to finish.

    # Measure the execution time.
    times: list[float] = []

    for trial in range(num_trials):  # Run multiple trials to capture variance.
        # Use CUDA events for accurate GPU timing without including CPU overhead.
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()  # Record the start time.
        run()  # Execute the operation.
        end_event.record()  # Record the end time.
        torch.cuda.synchronize()  # Wait for all CUDA work to finish.

        times.append((start_event.elapsed_time(end_event)))

    mean_time = mean(times)

    return mean_time


def profile(run: Callable, num_warmups: int = 1):
    # Warm up the operation.
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    # Profile one execution of the operation.
    with torch.profiler.profile(
        activities=[ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        run()
        torch.cuda.synchronize()

    # Format the profiler results as a table.
    table = prof.key_averages().table(
        sort_by="cuda_time_total", max_name_column_width=100, row_limit=10
    )

    # Append the results to `profiles.txt`.
    with open("var/profiles.txt", "a") as f:
        f.write(f"Profile at {time.ctime()}:\n")
        f.write(table)
        f.write("\n\n")
    return table


def run_operation1(dim: int, operation: Callable) -> Callable:
    # Create one random dim-by-dim matrix.
    x = torch.randn(dim, dim, device=cuda_if_available())

    # Return a function that performs the operation.
    return lambda: operation(x)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def naive_gelu(x: torch.Tensor):
    # tanh approximation of the GeLU activation function.
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))


def builtin_gelu(x: torch.Tensor):
    # PyTorch's built-in GeLU with the tanh approximation.
    return torch.nn.functional.gelu(x, approximate="tanh")


def check_equal_1d(f1, f2):
    x = torch.randn(2048, device=cuda_if_available())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


x = torch.tensor([1.])

# 1. Naive PyTorch implementation (not fused).
y1 = naive_gelu(x)

# 2. Built-in PyTorch implementation (fused).
y2 = builtin_gelu(x)
check_equal_1d(naive_gelu, builtin_gelu)

# 3. Compile the naive implementation with PyTorch.
compiled_gelu = torch.compile(naive_gelu)
y3 = compiled_gelu(x)
check_equal_1d(naive_gelu, compiled_gelu)

# Benchmark the implementations.
naive_time = benchmark(run_operation1(dim=16384, operation=naive_gelu))
builtin_time = benchmark(run_operation1(dim=16384, operation=builtin_gelu))
compiled_time = benchmark(run_operation1(dim=16384, operation=compiled_gelu))

# Profile the naive GeLU implementation.
naive_gelu_profile = profile(run_operation1(dim=16384, operation=naive_gelu))

# Profile the built-in GeLU implementation.
builtin_gelu_profile = profile(run_operation1(dim=16384, operation=builtin_gelu))

# Profile the compiled GeLU implementation.
compiled_gelu_profile = profile(run_operation1(dim=16384, operation=compiled_gelu))
