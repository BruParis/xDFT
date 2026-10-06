#ifndef BENCHMARK
#define BENCHMARK

#include "methods.cuh"

struct BenchResult {
  int n;
  double seconds;  // total time of `runs` executions
  double gflops;
  bool checked;    // false for the reference method or --no-check
  bool pass;
  float max_err;
};

// Benchmarks one method for one FFT length: checks the output against cuFFT
// (unless `check` is false or the method is the reference), then times `runs`
// executions.
BenchResult run_benchmark(const Method &method, int n, int batch, int runs,
                          bool check);

#endif
