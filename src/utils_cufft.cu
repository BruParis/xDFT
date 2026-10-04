// #include <cuda_runtime.h>
// #include <cufft.h>

#include "../include/utils_cufft.cuh"

cufftComplex complex_add(cufftComplex a, cufftComplex b) {
  cufftComplex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

cufftComplex complex_sub(cufftComplex a, cufftComplex b) {
  cufftComplex c;
  c.x = a.x - b.x;
  c.y = a.y - b.y;
  return c;
}

cufftComplex complex_mul(cufftComplex a, cufftComplex b) {
  cufftComplex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

void random_init_complex(Complex1dFP32 complex, int MAX_VAL, int MIN_VAL) {
  if (complex.on_device) {
    printf("Complex must be on host for random initialization\n");
  }
  // Getting Complex Dimension
  int length = complex.length;

  // Initializing val to each location
  for (int i = 0; i < length; i++) {
    complex.ptr[i].x =
        (float)(rand() % (MAX_VAL - MIN_VAL + 1) + MIN_VAL);
    complex.ptr[i].y =
        (float)(rand() % (MAX_VAL - MIN_VAL + 1) + MIN_VAL);
  }
}

void init_complex(Complex1dFP32 complex, float val) {
  // Getting Complex Dimension
  int length = complex.length;

  // Initializing val to each location
  for (int i = 0; i < length; i++) {
    complex.ptr[i].x = val;
    complex.ptr[i].y = val;
  }
}

void assert_complex(cufftComplex c1, cufftComplex c2, float eps) {
  // within eps
  float absdiff_x = fabs(c1.x - c2.x);
  float absdiff_y = fabs(c1.y - c2.y);

  assert(absdiff_x < eps && absdiff_y < eps && "Assertion failed!");
}

void assert_complex1d(Complex1dFP32 complex1, Complex1dFP32 complex2,
                    float eps) {
  // Complex must be on host
  assert(complex1.on_device == false && "For asserting complex must be on host");
  assert(complex2.on_device == false && "For asserting complex must be on host");

  // Getting Complex Dimension
  int length1 = complex1.length;
  int length2 = complex2.length;

  // Asserting that complex have same dimensions
  assert(length1 == length2 && "Complex1 length must be equal to Complex2 length");

  for (int i = 0; i < length1; i++) {
    assert_complex(complex1.ptr[i], complex2.ptr[i], eps);
  }
}

Complex1dFP32 cufft_run(Complex1dFP32 h_data_ref, int RANK) {
  // clone reference data on host, move to device
  int n = h_data_ref.length;
  int batch = h_data_ref.batch;

  Complex1dFP32 h_data = h_data_ref.clone();
  Complex1dFP32 d_data = Complex1dFP32(h_data.length, h_data.batch, true);
  h_data.copy_to_device(d_data);
  cudaDeviceSynchronize();

  // compute cufft forward
  cufftHandle plan;

  int n_size[1] = {n};
  cufftPlanMany(&plan, RANK, n_size, NULL, 1, n, NULL, 1, n, CUFFT_C2C, batch);
  cufftExecC2C(plan, d_data.ptr, d_data.ptr, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  // do not forget to normalize !
  // d_data.normalize();

  // copy back to host
  d_data.copy_to_host(h_data);

  cufftDestroy(plan);

  return h_data;
}
