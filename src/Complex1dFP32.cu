#include "../include/Complex1dFP32.cuh"
// #include "../include/utils.cuh"

#include <assert.h>
#include <iostream>

// Kernel for normalization on the device
__global__ void normalize_kernel(cufftComplex *data, int length, int total) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < total) {
    data[idx].x /= length;
    data[idx].y /= length;
  }
}

Complex1dFP32::Complex1dFP32(int length_, int batch_, bool on_device_)
    : length(length_), batch(batch_), on_device(on_device_) {
  int num = length * batch;
  if (!on_device) {
    // Initialize dynamic array
    ptr = new cufftComplex[num];
  } else {
    cudaError_t err = cudaMalloc((void **)&ptr, num * sizeof(cufftComplex));
    // cuda_check(err);
  }
}

void Complex1dFP32::normalize() {
  if (!on_device) {
    // Normalize data on the host
    for (int i = 0; i < length * batch; ++i) {
      ptr[i].x /= length;
      ptr[i].y /= length;
    }
  } else {
    // Normalize data on the device
    int threads_per_block = 256;
    int total = length * batch;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    normalize_kernel<<<blocks, threads_per_block>>>(ptr, length, total);
    cudaDeviceSynchronize(); // Ensure the kernel completes
  }
}

Complex1dFP32 Complex1dFP32::clone() const {
  Complex1dFP32 new_clone(length, batch, on_device);

  int num = length * batch;
  if (!on_device) {
    memcpy(new_clone.ptr, ptr, num * sizeof(cufftComplex));
  } else {
    cudaError_t err = cudaMemcpy(new_clone.ptr, ptr, num * sizeof(cufftComplex),
                                 cudaMemcpyDeviceToDevice);
    // cuda_check(err);
  }

  return new_clone;
}

void Complex1dFP32::free_complex() {
  if (!on_device) {
    delete[] ptr;
  } else {
    cudaFree(ptr);
  }
}

void Complex1dFP32::copy_to_device(Complex1dFP32 d_complex) {
  // Ensure source is on host and destination is on device
  assert(!on_device && "Source must be in host memory");
  assert(d_complex.on_device && "Destination must be in device memory");

  // Copy data from host to device
  cudaError_t err =
      cudaMemcpy(d_complex.ptr, ptr, length * batch * sizeof(cufftComplex),
                 cudaMemcpyHostToDevice);
  // cuda_check(err);
}

void Complex1dFP32::copy_to_host(Complex1dFP32 h_complex) {
  // Ensure source is on device and destination is on host
  assert(on_device && "Source must be in device memory");
  assert(!h_complex.on_device && "Destination must be in host memory");

  // Copy data from device to host
  cudaError_t err =
      cudaMemcpy(h_complex.ptr, ptr, length * batch * sizeof(cufftComplex),
                 cudaMemcpyDeviceToHost);
  // cuda_check(err);
}
