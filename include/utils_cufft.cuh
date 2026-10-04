#ifndef UTILS_CUFFT
#define UTILS_CUFFT

#include "../include/Complex1dFP32.cuh"

#include <assert.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

cufftComplex complex_add(cufftComplex a, cufftComplex b);
cufftComplex complex_sub(cufftComplex a, cufftComplex b);
cufftComplex complex_mul(cufftComplex a, cufftComplex b);

// Initializing Complex1dFP32 with random between (MAX_VAL, MIN_VAL)
void random_init_complex(Complex1dFP32 complex, int MAX_VAL, int MIN_VAL);

// Initializing Complex1dFP32 with val
void init_complex(Complex1dFP32 complex, float val);

// Asserting two complex numbers are same within the tolerance (eps)
void assert_complex(cufftComplex c1, cufftComplex c2, float eps);

// Asserting two complex1d arrays are same within the tolerance (eps)
void assert_complex1d(Complex1dFP32 complex1, Complex1dFP32 complex2,
                      float eps);

Complex1dFP32 cufft_run(Complex1dFP32 h_data_ref, int RANK);

#endif
