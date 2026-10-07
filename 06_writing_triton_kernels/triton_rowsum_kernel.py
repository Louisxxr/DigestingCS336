import torch
import triton
import triton.language as tl

from gpu_util import cuda_if_available


def triton_row_sum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 1024

    row_sum_kernel[(M,)](
        x, y, N, BLOCK_SIZE=BLOCK_SIZE
    )

    return y


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Identify the row processed by the current program instance.
    row = tl.program_id(0)

    # Initialize one accumulator per block element.
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the row tiles.
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        acc += x

    # Reduce the BLOCK_SIZE accumulators to a scalar.
    result = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, result)


def builtin_row_sum(x: torch.Tensor):
    return x.sum(dim=1)


x = torch.randn(2, 8000, device=cuda_if_available())

y1 = triton_row_sum(x)
y2 = builtin_row_sum(x)
assert torch.allclose(y1, y2, atol=1e-6)
