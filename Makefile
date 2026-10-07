CC = nvcc
CFLAGS = -lineinfo # source lines in ncu, no perf cost

HEADERS = $(wildcard include/*.cuh)
OBJS = build/Complex1dFP32.o build/utils_cufft.o build/methods.o \
       build/01_fft_naive.o build/02_fft_cooley_tukey.o \
       build/03_fft_stockham.o build/benchmark.o build/main.o

# Old standalone benchmarks, kept until xdft reproduces their output
LEGACY_TARGETS = build/00c_benchmark_cuFFT.out build/00d_benchmark_fft_cpu.out build/01_fft_benchmark_naive.out build/02_fft_benchmark_cooley_tukey.out build/03_fft_benchmark_stockham.out

.PHONY: all legacy clean profile-nsys

all: build/xdft

legacy: $(LEGACY_TARGETS)

build:
	mkdir -p build

# Single benchmark executable: build/xdft help
build/xdft: $(OBJS) | build
	$(CC) $(CFLAGS) $(OBJS) -lcufft -o $@

# Profiling: make profile-nsys ARGS="..." GPU=1
# Reports are written to build/profile/
ARGS ?= help
GPU ?= 0

profile-nsys: build/xdft
	@mkdir -p build/profile
	CUDA_VISIBLE_DEVICES=$(GPU) nsys profile --force-overwrite=true --stats=true -o build/profile/xdft build/xdft $(ARGS)

# Every src/*.cu is compiled to build/*.o
build/%.o: src/%.cu $(HEADERS) | build
	$(CC) $(CFLAGS) -dc $< -o $@

DEVICE_USAGE = --ptxas-options=-v
HOST_COMPILE_FLAG = -c
DEVICE_COMPILE_FLAG = -dc
LINK_CUBLAS = -lcublas
LINK_CUFFT = -lcufft
CPU_OPTIMIZE = -O3 -Xcompiler "-Ofast -march=native -funroll-loops -ffast-math -msse2 -msse3 -msse4 -mavx -mavx2 -flto"

# cuFFT
build/00c_benchmark_cuFFT.out: test/00c_benchmark_cuFFT.cu build/Complex1dFP32.o build/utils_cufft.o | build
	$(CC) $(LINK_CUFFT) $(LINK_CUBLAS) build/Complex1dFP32.o build/utils_cufft.o test/00c_benchmark_cuFFT.cu -o build/00c_benchmark_cuFFT.out

# fft CPU
build/00d_benchmark_fft_cpu.out: test/00d_benchmark_fft_cpu.cpp build/Complex1dFP32.o build/utils_cufft.o | build
	$(CC) $(LINK_CUFFT) $(LINK_CUBLAS) $(CPU_OPTIMIZE) build/Complex1dFP32.o build/utils_cufft.o test/00d_benchmark_fft_cpu.cpp -o build/00d_benchmark_fft_cpu.out

# Naive fft
build/01_fft_benchmark_naive.out: src/01_fft_naive.cu test/01_fft_benchmark_naive.cu build/Complex1dFP32.o build/utils_cufft.o | build
	$(CC) $(LINK_CUFFT) $(LINK_CUBLAS) build/Complex1dFP32.o build/utils_cufft.o src/01_fft_naive.cu test/01_fft_benchmark_naive.cu -o build/01_fft_benchmark_naive.out

# Cooley-Tukey fft
build/02_fft_benchmark_cooley_tukey.out: src/02_fft_cooley_tukey.cu test/02_fft_benchmark_cooley_tukey.cu build/Complex1dFP32.o build/utils_cufft.o | build
	$(CC) $(LINK_CUFFT) $(LINK_CUBLAS) build/Complex1dFP32.o build/utils_cufft.o src/02_fft_cooley_tukey.cu test/02_fft_benchmark_cooley_tukey.cu -o build/02_fft_benchmark_cooley_tukey.out

# Stockham fft
build/03_fft_benchmark_stockham.out: src/03_fft_stockham.cu test/03_fft_benchmark_stockham.cu build/Complex1dFP32.o build/utils_cufft.o | build
	$(CC) $(LINK_CUFFT) $(LINK_CUBLAS) build/Complex1dFP32.o build/utils_cufft.o src/03_fft_stockham.cu test/03_fft_benchmark_stockham.cu -o build/03_fft_benchmark_stockham.out

# Clean executable files
clean:
	@echo "Removing object files..."
	rm -rf build
