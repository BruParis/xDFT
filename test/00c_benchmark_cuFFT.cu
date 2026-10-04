#include <ctime>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/Complex1dFP32.cuh"
#include "../include/utils_cufft.cuh"

#define BATCH 256
#define RANK 1

int main() {
  int mat_sizes[] = {256, 512, 1024, 2048, 4096, 8192};
  int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);

  cufftHandle plan;

  // For recording time
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // Store time and GFLOPS
  double cufft_time[n_sizes];
  double cufft_gflops[n_sizes];
  int *n_size = new int[RANK];

  for (int mat_size = 0; mat_size < n_sizes; mat_size++) {
    // Matrix Size
    int n = mat_sizes[mat_size];

    // Define Complex1dFP32
    Complex1dFP32 h_FP32 = Complex1dFP32(n, BATCH);

    // Initialize Complex1dFP32
    random_init_complex(h_FP32, 10, -10);

    // Move matrices to device
    Complex1dFP32 d_A_FP32 = Complex1dFP32(n, BATCH, true);
    Complex1dFP32 d_B_FP32 = Complex1dFP32(n, BATCH, true);
    h_FP32.copy_to_device(d_A_FP32);
    cudaDeviceSynchronize();

    // Create cuFFT plan
    int n_size[1] = {h_FP32.length};
    cufftPlanMany(&plan, RANK, n_size, NULL, 1, n, NULL, 1, n, CUFFT_C2C,
                  BATCH);

    //----------------------------------------------------//
    //-------------------- Warmup Run --------------------//
    //----------------------------------------------------//

    // Execute FFT forward transformation
    cufftExecC2C(plan, d_A_FP32.ptr, d_B_FP32.ptr, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    // Copy data back to host
    d_B_FP32.copy_to_host(h_FP32);
    // cufftExecC2C does not normalizes
    h_FP32.normalize();

    //----------------------------------------------------//
    //---------------------- cuFFT -----------------------//
    //----------------------------------------------------//
    cudaEventRecord(beg);
    for (int n_runs = 0; n_runs < 10; n_runs++) {
      // Execute FFT forward transformation
      cufftExecC2C(plan, d_A_FP32.ptr, d_A_FP32.ptr, CUFFT_FORWARD);
      cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);

    float elapsed_sec = elapsed_time / 1000;
    cufft_time[mat_size] = elapsed_sec;
    cufft_gflops[mat_size] =
        1e-9 * 10. * 6. * n * BATCH * log2(n) / (elapsed_sec);

    // Cleanup
    h_FP32.free_complex();
    d_A_FP32.free_complex();
    cufftDestroy(plan);
  }

  std::cout << "cufft Time (seconds): ";
  for (int mat_size = 0; mat_size < n_sizes; mat_size++)
    std::cout << cufft_time[mat_size] << " ";
  std::cout << "\n \n";

  std::cout << "cufft GFLOPS: ";
  for (int mat_size = 0; mat_size < n_sizes; mat_size++)
    std::cout << cufft_gflops[mat_size] << " ";
  std::cout << "\n \n";

  return 0;
}
