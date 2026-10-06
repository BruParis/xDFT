#include <ctime>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <stdlib.h>

#include "../include/Complex1dFP32.cuh"
#include "../include/utils_cufft.cuh"

#include "../include/02_fft_cooley_tukey.cuh"

#define BATCH 256
// #define BATCH 1
#define RANK 1

int main() {
  int n_length[] = {256, 512, 1024, 2048, 4096, 8192};
  // int n_length[] = {8192};
  // int n_length[] = {4};
  int n_sizes = sizeof(n_length) / sizeof(n_length[0]);

  // For recording time
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // Store time and GFLOPS
  double fft_time[n_sizes];
  double fft_gflops[n_sizes];

  for (int mat_size = 0; mat_size < n_sizes; mat_size++) {
    // Matrix Size
    int n = n_length[mat_size];

    // Define Complex1dFP32
    Complex1dFP32 h_A_FP32 = Complex1dFP32(n, BATCH);
    random_init_complex(h_A_FP32, 10, -10);
    Complex1dFP32 h_B_FP32 = h_A_FP32.clone();

    //----------------------------------------------------//
    //--------------------- Check Run --------------------//
    //----------------------------------------------------//
    // Test againt cuFFT
    Complex1dFP32 d_A_FP32 = Complex1dFP32(n, BATCH, true);
    h_A_FP32.copy_to_device(d_A_FP32);
    cooley_tukey_fft(d_A_FP32);
    d_A_FP32.normalize();
    d_A_FP32.copy_to_host(h_A_FP32);

    Complex1dFP32 h_C_FP32 = cufft_run(h_B_FP32, RANK);

    // printf("original\n");
    // for (int i = 0; i < 4; i++) {
    //   printf("%f %f\n", h_A_FP32.ptr[i].x, h_A_FP32.ptr[i].y); 
    // }
    // printf("cooley_tukey\n");
    // for (int i = 0; i < 4; i++) {
    //   printf("%f %f\n", h_A_FP32.ptr[i].x, h_A_FP32.ptr[i].y); 
    // }
    // printf("cufft\n");
    // for (int i = 0; i < 4; i++) {
    //   printf("%f %f\n", h_C_FP32.ptr[i].x, h_C_FP32.ptr[i].y); 
    // }

    assert_complex1d(h_A_FP32, h_C_FP32, 1e-4);

    //----------------------------------------------------//
    //---------------------- cuFFT -----------------------//
    //----------------------------------------------------//
    cudaEventRecord(beg);
    for (int n_runs = 0; n_runs < 10; n_runs++) {
      // Execute FFT forward transformation
      cooley_tukey_fft(d_A_FP32);
      d_A_FP32.normalize();
      cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);

    float elapsed_sec = elapsed_time / 1000;
    fft_time[mat_size] = elapsed_sec;
    fft_gflops[mat_size] =
        1e-9 * 10. * 6. * n * BATCH * log2(n) / (elapsed_sec);

    // Cleanup
    h_A_FP32.free_complex();
    h_B_FP32.free_complex();
  }

  std::cout << "Time (seconds): ";
  for (int mat_size = 0; mat_size < n_sizes; mat_size++)
    std::cout << fft_time[mat_size] << " ";
  std::cout << "\n \n";

  std::cout << "GFLOPS: ";
  for (int mat_size = 0; mat_size < n_sizes; mat_size++)
    std::cout << fft_gflops[mat_size] << " ";
  std::cout << "\n \n";

  return 0;
}
