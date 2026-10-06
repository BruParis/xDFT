#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <math_constants.h>

#include "../include/02_fft_cooley_tukey.cuh"
#include "../include/Complex1dFP32.cuh"

// Fails loudly if a launch did not happen (e.g. too much shared memory), so
// that a method never reports timings for a kernel that did not run.
static void check_launch(const char *name, int N) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: kernel launch failed for N=%d: %s\n", name, N,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

static __device__ int bit_reversal(int idx, int logN) {
  int reversed = 0;
  for (int i = 0; i < logN; ++i) {
    reversed = (reversed << 1) | (idx & 1);
    idx >>= 1;
  }
  return reversed;
}

__global__ void fft_cooley_tukey_kernel(cufftComplex *data, int N) {
  int batchIdx = blockIdx.x;
  int thrIdx = threadIdx.x;
  int logN = __log2f(N);

  extern __shared__ cufftComplex smem[]; // N elements, size set at launch
  cufftComplex *g = data + batchIdx * N; // This block's FFT in global memory

  // Load into shared memory, in bit-reversed order
  for (int k = thrIdx; k < N; k += blockDim.x) {
    smem[bit_reversal(k, logN)] = g[k];
  }

  __syncthreads();

  for (int step = 1; step < N; step *= 2) {
    int doubleStep = step * 2;
    float angle = -CUDART_PI_F / step;

    for (int k = thrIdx; k < N / 2; k += blockDim.x) {
      int i = (k / step) * doubleStep + (k % step);
      int j = i + step;

      if (i > N || j > N) {
        continue;
      }

      float theta = angle * (k % step);
      cufftComplex twiddle = {cosf(theta), sinf(theta)};

      cufftComplex even = smem[i];
      cufftComplex odd = smem[j];

      // Butterfly operation
      cufftComplex t = cuCmulf(twiddle, odd);
      smem[i] = cuCaddf(even, t);
      smem[j] = cuCsubf(even, t);
    }

    __syncthreads();
  }

  // Copy the result back to global memory
  for (int k = thrIdx; k < N; k += blockDim.x) {
    g[k] = smem[k];
  }
}

void cooley_tukey_fft(Complex1dFP32 d_A) {
  int N = d_A.length;
  int nBatch = d_A.batch;

  // Launch kernel: one block per batch
  int threads_per_block = 256;
  int blocksPerGrid = nBatch;

  size_t smem = N * sizeof(cufftComplex);
  fft_cooley_tukey_kernel<<<blocksPerGrid, threads_per_block, smem>>>(d_A.ptr,
                                                                      N);
  check_launch("cooley_tukey_fft", N);

  cudaDeviceSynchronize();
}
