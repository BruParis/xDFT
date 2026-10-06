# xDFT

Trying DFT/FFT CUDA implementations (Naive, Cooley-Tukey, Stockham), benchmarked against cuFFT.

## Dependency

**CUDA Toolkit** is needed (`nvcc`, `cufft`, `cublas`, `curand`, `cuda_runtime`).

## Build

```bash
make all             # builds build/xdft
```

## Usage

```bash
./build/xdft help                  # commands and options
./build/xdft list                  # available methods
./build/xdft run all               # benchmark every method (skips [wip] ones)
./build/xdft run naive --sizes 256,1024 --batch 64 --runs 20
```

Each method is checked against cuFFT (`--no-check` to skip) and the exit code
is 1 if a check fails.
