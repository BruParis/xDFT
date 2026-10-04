#ifndef FFT_NAIVE
#define FFT_NAIVE

#include "Complex1dFP32.cuh"

// void split_even_odd_kernel(cufftComplex *d_data, cufftComplex *d_even,
//                            cufftComplex *d_odd, int length, int nBatch);
// void combine_kernel(cufftComplex *d_data, const cufftComplex *d_even,
//              const cufftComplex *d_odd, int length, int nBatch);

// void naive_fft_recursive(cufftComplex *d_data, int N, int nBatch,
//                         cufftComplex *d_even, cufftComplex *d_odd);
void naive_fft(Complex1dFP32 d_A_ptr);

#endif
