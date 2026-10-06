#ifndef METHODS
#define METHODS

#include "Complex1dFP32.cuh"

// All FFT methods share one signature: in-place forward FFT on a device array.
typedef void (*FftFn)(Complex1dFP32 d_data);

struct Method {
  const char *name;        // name used on the command line
  const char *description; // shown by `xdft list`
  FftFn run;               // in-place, unnormalized forward FFT
  float tol;               // max abs error allowed against (normalized) cuFFT
  bool reference;          // the cuFFT baseline: timed but not checked
  bool wip;                // work in progress: skipped by `run all`
};

// Registry of all available methods
const Method *get_methods(int *count);
const Method *find_method(const char *name);

#endif
