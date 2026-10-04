CC = nvcc

DEVICE_USAGE = --ptxas-options=-v
HOST_COMPILE_FLAG = -c
DEVICE_COMPILE_FLAG = -dc
LINK_CUBLAS = -lcublas
LINK_CUFFT = -lcufft
CPU_OPTIMIZE = -O3 -Xcompiler "-Ofast -march=native -funroll-loops -ffast-math -msse2 -msse3 -msse4 -mavx -mavx2 -flto"

# Complex1dFP32
build/Complex1dFP32.o: src/Complex1dFP32.cu
	$(CC) $(DEVICE_COMPILE_FLAG) src/Complex1dFP32.cu -o build/Complex1dFP32.o

# Utils fft
build/utils_cufft.o: src/utils_cufft.cu build/Complex1dFP32.o
	$(CC) $(DEVICE_COMPILE_FLAG) $(LINK_CUFFT) src/utils_cufft.cu -o build/utils_cufft.o

# cuFFT
00c_benchmark_cuFFT.out: test/00c_benchmark_cuFFT.cu build/Complex1dFP32.o build/utils_cufft.o
	$(CC) $(LINK_CUFFT) $(LINK_CUBLAS) build/Complex1dFP32.o build/utils_cufft.o test/00c_benchmark_cuFFT.cu -o 00c_benchmark_cuFFT.out

# fft CPU
00d_benchmark_fft_cpu.out: test/00d_benchmark_fft_cpu.cpp build/Complex1dFP32.o build/utils_cufft.o
	$(CC) $(LINK_CUFFT) $(LINK_CUBLAS) $(CPU_OPTIMIZE) build/Complex1dFP32.o build/utils_cufft.o test/00d_benchmark_fft_cpu.cpp -o 00d_benchmark_fft_cpu.out

# Naive fft
01_fft_benchmark_naive.out: src/01_fft_naive.cu test/01_fft_benchmark_naive.cu build/Complex1dFP32.o build/utils_cufft.o
	$(CC) $(LINK_CUFFT) $(LINK_CUBLAS) build/Complex1dFP32.o build/utils_cufft.o src/01_fft_naive.cu test/01_fft_benchmark_naive.cu -o 01_fft_benchmark_naive.out

# Cooley-Tukey fft
02_fft_benchmark_cooley_tukey.out: src/02_fft_cooley_tukey.cu test/02_fft_benchmark_cooley_tukey.cu build/Complex1dFP32.o build/utils_cufft.o
	$(CC) $(LINK_CUFFT) $(LINK_CUBLAS) build/Complex1dFP32.o build/utils_cufft.o src/02_fft_cooley_tukey.cu test/02_fft_benchmark_cooley_tukey.cu -o 02_fft_benchmark_cooley_tukey.out

# Stockham fft
03_fft_benchmark_stockham.out: src/03_fft_stockham.cu test/03_fft_benchmark_stockham.cu build/Complex1dFP32.o build/utils_cufft.o
	$(CC) $(LINK_CUFFT) $(LINK_CUBLAS) build/Complex1dFP32.o build/utils_cufft.o src/03_fft_stockham.cu test/03_fft_benchmark_stockham.cu -o 03_fft_benchmark_stockham.out

# Clean executable files
clean:
	@echo "Removing object files..."
	rm *.out build/*.o
