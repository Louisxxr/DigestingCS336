import time
from typing import Callable

import torch
from torch.profiler import ProfilerActivity

from gpu_util import cuda_if_available


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


def run_operation2(dim: int, operation: Callable) -> Callable:
    # Create two random dim-by-dim matrices.
    x = torch.randn(dim, dim, device=cuda_if_available())
    y = torch.randn(dim, dim, device=cuda_if_available())

    # Return a function that performs the operation.
    return lambda: operation(x, y)


# Profile elementwise addition (dim = 2048).
add_profile = profile(run_operation2(dim=2048, operation=lambda a, b: a + b))

# Profile matrix multiplication (dim = 2048).
matmul_profile = profile(run_operation2(dim=2048, operation=lambda a, b: a @ b))

# Profile matrix multiplication (dim = 128).
matmul_profile = profile(run_operation2(dim=128, operation=lambda a, b: a @ b))
