#include <hip/hip_runtime.h>
#include "../include/kernel_config.h"

#define TILE_SIZE 32

// Optimized tiled matrix multiplication using shared memory
// C = A * B where A is MxK, B is KxN, C is MxN
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory with bounds checking
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? 
            A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? 
            B[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// FP16 matrix multiply for higher throughput on supported hardware
__global__ void matmul_tiled_half(const __half* A, const __half* B, float* C,
                                   int M, int N, int K) {
    __shared__ __half As[TILE_SIZE][TILE_SIZE];
    __shared__ __half Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? 
            A[row * K + a_col] : __float2half(0.0f);
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? 
            B[b_row * N + col] : __float2half(0.0f);
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host wrapper
void launch_matmul_tiled(const float* d_A, const float* d_B, float* d_C,
                          int M, int N, int K) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
