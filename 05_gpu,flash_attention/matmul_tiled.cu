#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int TILE = 16;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error = (call);                                          \
        if (error != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int num_k_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_k_tiles; ++t) {
        const int a_col = t * TILE + tx;
        const int b_row = t * TILE + ty;

        // Threads cooperatively load two tiles with coalesced global accesses.
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int p = 0; p < TILE; ++p) {
            acc += As[ty][p] * Bs[p][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

int main() {
    // Non-multiples of TILE exercise the boundary checks.
    constexpr int M = 127;
    constexpr int N = 131;
    constexpr int K = 129;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>((i % 13) - 6) / 7.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>((i % 11) - 5) / 6.0f;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_error = 0.0f;
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float expected = 0.0f;
            for (int p = 0; p < K; ++p) {
                expected += h_A[row * K + p] * h_B[p * N + col];
            }
            max_error = std::fmax(max_error, std::fabs(h_C[row * N + col] - expected));
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::printf("C[%d x %d] = A[%d x %d] @ B[%d x %d]\n", M, N, M, K, K, N);
    std::printf("max absolute error: %.8g\n", max_error);
    return max_error < 1e-4f ? EXIT_SUCCESS : EXIT_FAILURE;
}
