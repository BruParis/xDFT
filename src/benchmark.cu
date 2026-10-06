#include <cmath>

#include <cuda_runtime.h>

#include "../include/benchmark.cuh"
#include "../include/utils_cufft.cuh"

static float max_abs_diff(Complex1dFP32 a, Complex1dFP32 b) {
  float err = 0.f;
  for (int i = 0; i < a.length * a.batch; i++) {
    float ex = fabsf(a.ptr[i].x - b.ptr[i].x);
    float ey = fabsf(a.ptr[i].y - b.ptr[i].y);
    // NaN-safe: a NaN difference must count as a failure
    if (!(ex <= err)) err = ex;
    if (!(ey <= err)) err = ey;
  }
  return err;
}

BenchResult run_benchmark(const Method &method, int n, int batch, int runs,
                          bool check) {
  BenchResult res = {n, 0., 0., false, true, 0.f};

  Complex1dFP32 h_in(n, batch);
  random_init_complex(h_in, 10, -10);

  Complex1dFP32 d_data(n, batch, true);
  h_in.copy_to_device(d_data);

  // Warmup run, also used for the correctness check
  method.run(d_data);
  cudaDeviceSynchronize();

  if (check && !method.reference) {
    // cuFFT does not normalize: both sides are normalized by 1/n
    d_data.normalize();
    Complex1dFP32 h_out(n, batch);
    d_data.copy_to_host(h_out);
    Complex1dFP32 h_ref = cufft_run(h_in, 1);

    res.checked = true;
    res.max_err = max_abs_diff(h_out, h_ref);
    res.pass = res.max_err < method.tol;

    h_out.free_complex();
    h_ref.free_complex();
  }

  // Timing 
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  cudaEventRecord(beg);
  for (int i = 0; i < runs; i++) {
    method.run(d_data);
    cudaDeviceSynchronize();
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, beg, end);
  res.seconds = elapsed_ms / 1000.;
  res.gflops = 1e-9 * runs * 6. * n * batch * log2((double)n) / res.seconds;

  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  h_in.free_complex();
  d_data.free_complex();
  return res;
}
