import torch
import triton
import triton.language as tl

from gpu_util import cuda_if_available


def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    return c


@triton.jit
def matmul_relu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,  # Output pointer for c = a @ b.
    M,
    N,
    K,  # Matrix shapes: a is M x K, b is K x N, and c is M x N.
    stride_am,
    stride_ak,  # Strides for navigating a.
    stride_bk,
    stride_bn,  # Strides for navigating b.
    stride_cm,
    stride_cn,  # Strides for navigating c.
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Process the (m, n)-th output tile.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the tile indices.
    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Row indices of a; shape: [BLOCK_M].
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # Column indices of b; shape: [BLOCK_N].
    indices_k = tl.arange(0, BLOCK_K)  # Shared K indices for a and b; shape: [BLOCK_K].

    # Construct the initial pointer matrices for a and b.
    a_ptrs = (
        a_ptr
        + indices_m[:, None] * stride_am
        + indices_k[None, :] * stride_ak
    )  # Shape: [BLOCK_M, BLOCK_K].
    b_ptrs = (
        b_ptr
        + indices_k[:, None] * stride_bk
        + indices_n[None, :] * stride_bn
    )  # Shape: [BLOCK_K, BLOCK_N].

    # Initialize the FP32 accumulator.
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Iterate over tiles along the K dimension.
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(indices_m[:, None] < M) & (indices_k[None, :] + k < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(indices_k[:, None] + k < K) & (indices_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak  # Advance to the next column tile of a.
        b_ptrs += BLOCK_K * stride_bk  # Advance to the next row tile of b.

    # Apply the ReLU activation.
    acc = tl.maximum(acc, 0.0)

    # Store the output tile.
    c_ptrs = (
        c_ptr + indices_m[:, None] * stride_cm + indices_n[None, :] * stride_cn
    )
    tl.store(
        c_ptrs,
        acc,
        mask=(indices_m[:, None] < M) & (indices_n[None, :] < N),
    )


def naive_matmul_relu(x: torch.Tensor, y: torch.Tensor):
    return torch.nn.functional.relu(x @ y)


a = torch.randn(1024, 1024, device=cuda_if_available())
b = torch.randn(1024, 1024, device=cuda_if_available())

c1 = triton_matmul_relu(a, b)
c2 = naive_matmul_relu(a, b)
# assert torch.allclose(c1, c2, atol=1e-2)
# 注：triton_matmul_relu中使用了tl.dot，走的是Tensor core，暂不清楚naive_matmul_relu的实现，但两者应该不是一个路径，所要求的精度下不相等


# Verify masking with non-divisible dimensions; expect this case to be slower than 1024 x 1024.
a = torch.randn(1025, 1025, device=cuda_if_available())
b = torch.randn(1025, 1025, device=cuda_if_available())

c1 = triton_matmul_relu(a, b)
c2 = naive_matmul_relu(a, b)
# assert torch.allclose(c1, c2, atol=1e-2)
