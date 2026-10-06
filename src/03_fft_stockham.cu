#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <math_constants.h>

#include "../include/03_fft_stockham.cuh"
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

__global__ void fft_stockham_kernel(cufftComplex *b_input, int N) {
  unsigned int batchIdx = blockIdx.x;
  unsigned int thrIdx = threadIdx.x;

  extern __shared__ cufftComplex smem[]; // init shared memory
  cufftComplex *input = smem;
  cufftComplex *output = smem + N;

  cufftComplex *g = b_input + batchIdx * N; // This block FFT's in global memory

  // load into shared memeory
  for (int i = thrIdx; i < N; i += blockDim.x) {
    input[i] = g[i];
  }

  __syncthreads();

  int halfN = N / 2;
  for (int step = 1; step < N; step *= 2) {
    int inGroupSize = step;
    // int doubleStep = step * 2;
    // int outGroupSize = doubleStep;

    float angle = -CUDART_PI_F / step;

    // for (int k = thrIdx; k < N; k += blockDim.x) {
    for (int k = thrIdx; k < halfN; k += blockDim.x) {
      int inGroupIdx = k / inGroupSize;
      int inIdx = (k % inGroupSize) + (inGroupIdx * inGroupSize);
      int inPairIdx = inIdx + halfN;

      int outIdx = (k / step) * 2 * step + (k % step);
      int outPairIdx = outIdx + step;

      // int outGroupIdx = (k * 2) / outGroupSize;
      // printf("k %d - outGroupSize %d - outGroupIdx %d - outIdx %d -
      // outPairIdx "
      //        "%d\n",
      //        k, outGroupSize, outGroupIdx, outIdx, outPairIdx);
      if (inIdx >= N || inPairIdx >= N || outIdx >= N || outPairIdx >= N) {
        printf("Out of range\n");
        continue;
      }
      // printf("inIdx %d inPairIdx %d outIdx %d outPairIdx %d\n", inIdx,
      //        inPairIdx, outIdx, outPairIdx);

      cufftComplex even = input[inIdx];
      cufftComplex odd = input[inPairIdx];

      // Calculate the twiddle factor
      float theta = angle * (k % step);
      cufftComplex twiddle = {cosf(theta), sinf(theta)};

      // Butterfly operation
      cufftComplex t = cuCmulf(twiddle, odd);
      // printf("t %f %f\n", t.x, t.y);

      output[outIdx] = cuCaddf(even, t);
      output[outPairIdx] = cuCsubf(even, t);
    }

    // Swap input and output
    cufftComplex *temp = input;
    input = output;
    output = temp;

    __syncthreads();
  }

  __syncthreads();

  // Copy the result to the batch_output
  for (int i = thrIdx; i < N; i += blockDim.x) {
    g[i] = input[i];
  }

  __syncthreads();
}

void stockham_fft(Complex1dFP32 d_A) {
  int N = d_A.length;
  int nBatch = d_A.batch;

  // Launch kernel: one block per batch
  // num thread : max 256 - N
  int threads_per_block = (N < 256) ? N : 256;
  int blocksPerGrid = nBatch;

  // Allocate memory for the output
  // cufftComplex *d_output;
  // cudaMalloc(&d_output, N * nBatch * sizeof(cufftComplex));

  size_t smem = 2 * N * sizeof(cufftComplex);
  fft_stockham_kernel<<<blocksPerGrid, threads_per_block, smem>>>(d_A.ptr, N);
  check_launch("stockham_fft", N);

  // Copy back the result
  // cudaMemcpy(d_A.ptr, d_output, N * nBatch * sizeof(cufftComplex),
  //            cudaMemcpyDeviceToDevice);

  // cudaFree(d_output);

  cudaDeviceSynchronize();
}
