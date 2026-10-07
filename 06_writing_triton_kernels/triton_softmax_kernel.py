import torch
import triton
import triton.language as tl

from gpu_util import cuda_if_available


def triton_softmax(x: torch.Tensor):
    # Allocate the output tensor.
    y = torch.empty_like(x)

    # Define the grid with one Triton program per row.
    M, N = x.shape  # Number of rows and columns.
    BLOCK_SIZE = triton.next_power_of_2(N)  # Each program covers all columns; `tl.arange` boundaries must be powers of two.
    num_blocks = M  # Launch one program per row.

    # Launch the Triton kernel.
    triton_softmax_kernel[(num_blocks,)](
        x_ptr=x,
        y_ptr=y,
        x_row_stride=x.stride(0),
        y_row_stride=y.stride(0),
        num_cols=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y


@triton.jit
def triton_softmax_kernel(
    x_ptr,
    y_ptr,
    x_row_stride,  # Physical row stride; includes padding when present.
    y_row_stride,  # Same physical row-stride semantics as above.
    num_cols,  # Logical number of columns; excludes padding.
    BLOCK_SIZE: tl.constexpr,
):
    assert num_cols <= BLOCK_SIZE

    # Process each row independently.
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Load the input row from global memory.
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # Compute softmax.
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # Store the output row in global memory.
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)


def naive_softmax(x: torch.Tensor):
    # M is the number of rows; N is the number of columns.
    M, N = x.shape
    # Compute the maximum of each row (MN reads, M writes).
    x_max = x.max(dim=1)[0]
    # Subtract the maximum (MN + M reads, MN writes).
    x = x - x_max[:, None]
    # Exponentiate each element (MN reads, MN writes).
    numerator = torch.exp(x)
    # Compute the normalization constant (MN reads, M writes).
    denominator = numerator.sum(dim=1)
    # Normalize each row (MN + M reads, MN writes).
    y = numerator / denominator[:, None]
    # Total: 5MN + 2M reads and 3MN + 2M writes.
    # Ideally, use MN reads and MN writes for a potential 4x speedup.
    return y


def builtin_softmax(x: torch.Tensor):
    return torch.nn.functional.softmax(x, dim=-1)


x = torch.randn(2, 1000, device=cuda_if_available())

y1 = triton_softmax(x)
y2 = naive_softmax(x)
y3 = builtin_softmax(x)
assert torch.allclose(y1, y3, atol=1e-6)
assert torch.allclose(y2, y3, atol=1e-6)
