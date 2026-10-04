#ifndef FFT_STOCKHAM
#define FFT_STOCKHAM

#include "Complex1dFP32.cuh"

// __global__ void fft_stockham_kernel(cufftComplex *b_input,
//                                     cufftComplex *b_output, int N);
void stockham_fft(Complex1dFP32 d_A_ptr);

#endif
