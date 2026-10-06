#include <cstring>
#include <map>
#include <utility>

#include <cufft.h>

#include "../include/01_fft_naive.cuh"
#include "../include/02_fft_cooley_tukey.cuh"
#include "../include/03_fft_stockham.cuh"
#include "../include/methods.cuh"

// cuFFT baseline. Plans are cached so that timing only covers the execution.
static void cufft_fft(Complex1dFP32 d_data) {
  static std::map<std::pair<int, int>, cufftHandle> plans;

  std::pair<int, int> key(d_data.length, d_data.batch);
  if (plans.find(key) == plans.end()) {
    cufftHandle plan;
    int n_size[1] = {d_data.length};
    cufftPlanMany(&plan, 1, n_size, NULL, 1, d_data.length, NULL, 1,
                  d_data.length, CUFFT_C2C, d_data.batch);
    plans[key] = plan;
  }
  cufftExecC2C(plans[key], d_data.ptr, d_data.ptr, CUFFT_FORWARD);
}

// To add a method: write `void my_fft(Complex1dFP32)`, add one row here.
static const Method METHODS_TABLE[] = {
    {"cufft", "NVIDIA cuFFT (baseline)", cufft_fft, 0.f, true, false},
    {"naive", "Recursive radix-2, split/combine kernels", naive_fft, 1e-3f,
     false, false},
    {"cooley-tukey", "Cooley-Tukey radix-2", cooley_tukey_fft, 1e-4f, false,
     false},
    {"stockham", "Stockham radix-2", stockham_fft, 1e-4f, false, false},
};

const Method *get_methods(int *count) {
  *count = sizeof(METHODS_TABLE) / sizeof(METHODS_TABLE[0]);
  return METHODS_TABLE;
}

const Method *find_method(const char *name) {
  int count;
  const Method *m = get_methods(&count);
  for (int i = 0; i < count; i++)
    if (strcmp(m[i].name, name) == 0)
      return &m[i];
  return NULL;
}
