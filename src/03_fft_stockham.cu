#include <iostream>
#include <math_constants.h>

#include "../include/03_fft_stockham.cuh"
#include "../include/Complex1dFP32.cuh"

__global__ void fft_stockham_kernel(cufftComplex *b_input,
                                    cufftComplex *b_output, int N) {
  unsigned int batchIdx = blockIdx.x;
  unsigned int thrIdx = threadIdx.x;
  unsigned int numStep = log2f(N);

  cufftComplex *input = b_input + batchIdx * N;
  cufftComplex *output = b_output + batchIdx * N;

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
    b_output[batchIdx * N + i] = input[i];
  }
}

void stockham_fft(Complex1dFP32 d_A) {
  int N = d_A.length;
  int nBatch = d_A.batch;

  // Launch kernel: one block per batch
  // num thread : max 256 - N
  int threads_per_block = (N < 256) ? N : 256;
  int blocksPerGrid = nBatch;

  // Allocate memory for the output
  cufftComplex *d_output;
  cudaMalloc(&d_output, N * nBatch * sizeof(cufftComplex));

  fft_stockham_kernel<<<blocksPerGrid, threads_per_block>>>(d_A.ptr, d_output,
                                                            N);

  // Copy back the result
  cudaMemcpy(d_A.ptr, d_output, N * nBatch * sizeof(cufftComplex),
             cudaMemcpyDeviceToDevice);

  cudaFree(d_output);

  cudaDeviceSynchronize();
}
