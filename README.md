# xDFT

Trying DFT/FFT CUDA implementations (Naive, Cooley-Tukey, Stockham), benchmarked against cuFFT.

## Dependency

**CUDA Toolkit** is needed (`nvcc`, `cufft`, `cublas`, `curand`, `cuda_runtime`).

## Build

```bash
mkdir -p build
make <target>.out
```
