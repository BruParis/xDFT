#include <iostream>
#include <math_constants.h>

#include "../include/01_fft_naive.cuh"
#include "../include/Complex1dFP32.cuh"

// Kernel to split data into even and odd indices for each batch
__global__ void split_even_odd(const cufftComplex *data, cufftComplex *even,
                               cufftComplex *odd, int length, int nBatch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch = blockIdx.y;

  if (idx < length / 2) {
    int base = batch * length;
    even[(base / 2) + idx] = data[base + 2 * idx];
    odd[(base / 2) + idx] = data[base + 2 * idx + 1];
  }
}

// Kernel to combine results for each batch
__global__ void combine(cufftComplex *data, const cufftComplex *d_even,
                        const cufftComplex *d_odd, int length, int nBatch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch = blockIdx.y;

  if (idx < length / 2) {
    float angle = -2 * CUDART_PI * idx / length;
    cufftComplex twiddle = {cos(angle), sin(angle)};

    int base = batch * length;
    cufftComplex t_odd = d_odd[(batch * length / 2) + idx];
    cufftComplex t_even = d_even[(batch * length / 2) + idx];

    cufftComplex tw = cuCmulf(twiddle, t_odd);

    // Combine results
    data[base + idx] = cuCaddf(t_even, tw);
    data[base + idx + length / 2] = cuCsubf(t_even, tw);
  }
}

void naive_fft_recursive(cufftComplex *data, int N, int nBatch) {
  if (N == 1)
    return;

  int half_N = N / 2;

  // Allocating memory for even and odd
  unsigned long size = half_N * nBatch * sizeof(cufftComplex);
  cufftComplex *d_even, *d_odd;
  cudaMalloc((void **)&d_even, size);
  cudaMalloc((void **)&d_odd, size);

  // Launch kernel to split data into even and odd
  int threads_per_block = 256;
  dim3 blocks_per_grid((half_N + threads_per_block - 1) / threads_per_block,
                       nBatch);
  split_even_odd<<<blocks_per_grid, threads_per_block>>>(data, d_even, d_odd, N,
                                                         nBatch);
  cudaDeviceSynchronize();

  // Recursively call FFT on even and odd parts
  naive_fft_recursive(d_even, half_N, nBatch);
  naive_fft_recursive(d_odd, half_N, nBatch);

  // Launch kernel to combine results
  combine<<<blocks_per_grid, threads_per_block>>>(data, d_even, d_odd, N,
                                                  nBatch);

  cudaDeviceSynchronize();

  // Freeing memory
  cudaFree(d_even);
  cudaFree(d_odd);
}

void naive_fft(Complex1dFP32 d_A) {
  int N = d_A.length;
  int nBatch = d_A.batch;

  // Launching recursive FFT
  naive_fft_recursive(d_A.ptr, N, nBatch);
}
