#ifndef COMPLEX1DFP32
#define COMPLEX1DFP32

#include <cufft.h>

class Complex1dFP32 {
  public:
    const int length; // Number of elements
    const int batch;  // Number of batches

    // Pointer to dynamic array
    cufftComplex *ptr;

    // 1d-array in device memory: true; else: false
    const bool on_device;

    // Constructor to initialize 1d-complex array
    Complex1dFP32(int length_, int batch, bool on_device=false);

    void free_complex();

    Complex1dFP32 clone() const;

    void normalize();

    void copy_to_device(Complex1dFP32 d_complex);
    void copy_to_host(Complex1dFP32 h_complex);
};

#endif
