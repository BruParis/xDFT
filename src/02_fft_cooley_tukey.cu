#include <iostream>
#include <math_constants.h>

#include "../include/02_fft_cooley_tukey.cuh"
#include "../include/Complex1dFP32.cuh"

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

  cufftComplex *d_data = data + batchIdx * N;

  // Bit-reversal reordering
  for (int k = thrIdx; k < N; k += blockDim.x) {
    int reversedIdx = bit_reversal(k, logN);
    if (reversedIdx > k) {
      cufftComplex temp = d_data[k];
      d_data[k] = d_data[reversedIdx];
      d_data[reversedIdx] = temp;
    }
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

      cufftComplex even = d_data[i];
      cufftComplex odd = d_data[j];

      // Butterfly operation
      cufftComplex t = cuCmulf(twiddle, odd);
      d_data[i] = cuCaddf(even, t);
      d_data[j] = cuCsubf(even, t);
    }

    __syncthreads();
  }
}

void cooley_tukey_fft(Complex1dFP32 d_A) {
  int N = d_A.length;
  int nBatch = d_A.batch;

  // Launch kernel: one block per batch
  int threads_per_block = 256;
  int blocksPerGrid = nBatch;

  fft_cooley_tukey_kernel<<<blocksPerGrid, threads_per_block>>>(d_A.ptr, N);

  cudaDeviceSynchronize();
}
