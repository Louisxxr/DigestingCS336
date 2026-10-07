from typing import Callable

import torch

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


def run_operation2(dim: int, operation: Callable) -> Callable:
    # Create two random dim-by-dim matrices.
    x = torch.randn(dim, dim, device=cuda_if_available())
    y = torch.randn(dim, dim, device=cuda_if_available())

    # Return a function that performs the operation.
    return lambda: operation(x, y)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


# Benchmark matrix multiplication.
matmul = run_operation2(dim=1024, operation=lambda a, b: a @ b)
result = benchmark(matmul)

# Measure how execution time scales with matrix dimension.
results = {}
for dim in [256, 512, 1024, 2048, 4096, 8192]:
    results[dim] = benchmark(run_operation2(dim=dim, operation=lambda a, b: a @ b))
