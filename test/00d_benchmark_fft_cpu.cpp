#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/Complex1dFP32.cuh"
#include "../include/utils_cufft.cuh"

// #define BATCH 256
#define BATCH 1
#define RANK 1

void FFT_recursive(cufftComplex *data, int length) {
  if (length == 1) {
    return;
  }

  // Allocate memory for even and odd data
  cufftComplex *even_data =
      (cufftComplex *)malloc(length / 2 * sizeof(cufftComplex));
  cufftComplex *odd_data =
      (cufftComplex *)malloc(length / 2 * sizeof(cufftComplex));

  // Split data into even and odd
  for (int i = 0; i < length / 2; i++) {
    even_data[i] = data[2 * i];
    odd_data[i] = data[2 * i + 1];
  }

  // Recursively call FFT
  FFT_recursive(even_data, length / 2);
  FFT_recursive(odd_data, length / 2);

  // Combine results
  for (int i = 0; i < length / 2; i++) {
    float angle = -2 * M_PI * i / length;
    cufftComplex t = {cos(angle), sin(angle)};
    printf("i: %d - twiddle: %f %f - even: %f %f - odd: %f %f\n", i, t.x, t.y, even_data[i].x, even_data[i].y, odd_data[i].x, odd_data[i].y);
    t = complex_mul(t, odd_data[i]);

    data[i] = complex_add(even_data[i], t);
    data[i + length / 2] = complex_sub(even_data[i], t);
  }

  // Free memory
  free(even_data);
  free(odd_data);
}

void FFT_batch(Complex1dFP32 *data) {
  for (int b = 0; b < data->batch; b++) {
    // Execute FFT forward transformation
    FFT_recursive(&data->ptr[b * data->length], data->length);
  }

  // Normalize
  // data->normalize();
}

int main() {
  // int mat_sizes[] = {256, 512, 1024, 2048, 4096, 8192};
  int mat_sizes[] = {8};
  int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);

  // Store time and GFLOPS
  double fft_time[n_sizes];
  double fft_gflops[n_sizes];

  auto start = std::chrono::high_resolution_clock::now();
  auto stop = std::chrono::high_resolution_clock::now();
  auto elapsed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  for (int mat_size = 0; mat_size < n_sizes; mat_size++) {
    // Matrix Size
    int n = mat_sizes[mat_size];
    std::cout << "1d FFT Size: " << n << "x" << BATCH << "\n";

    // Define Complex1dFP32
    Complex1dFP32 h_A_FP32 = Complex1dFP32(n, BATCH);
    random_init_complex(h_A_FP32, 10, -10);
    Complex1dFP32 h_B_FP32 = h_A_FP32.clone();

    //----------------------------------------------------//
    //--------------------- Check Run --------------------//
    //----------------------------------------------------//
    FFT_batch(&h_B_FP32);
    // Test againt cuFFT
    Complex1dFP32 h_C_FP32 = cufft_run(h_A_FP32, RANK);
    // printf("Original\n");
    // for (int i = 0; i < n; i++) {
    //   printf("%f %f\n", h_A_FP32.ptr[i].x, h_A_FP32.ptr[i].y);
    // } 
    // printf("CPU\n");
    // for (int i = 0; i < n; i++) {
    //   printf("%f %f\n", h_B_FP32.ptr[i].x, h_B_FP32.ptr[i].y);
    // }
    // printf("cuFFT\n");
    // for (int i = 0; i < n; i++) {
    //   printf("%f %f\n", h_C_FP32.ptr[i].x, h_C_FP32.ptr[i].y);
    // }

    assert_complex1d(h_B_FP32, h_C_FP32, 1e-3);
    printf("FFT computation is correct !\n");
    exit(0);

    //----------------------------------------------------//
    //---------------------- cuFFT -----------------------//
    //----------------------------------------------------//
    start = std::chrono::high_resolution_clock::now();
    for (int n_runs = 0; n_runs < 10; n_runs++) {
      // Execute FFT forward transformation
      FFT_batch(&h_A_FP32);
    }
    stop = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    float elapsed_sec = elapsed_time.count() / (1e+6);
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
