#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../include/benchmark.cuh"
#include "../include/methods.cuh"

static void usage(FILE *out) {
  fprintf(out,
          "xDFT: benchmark FFT implementations against cuFFT\n"
          "\n"
          "Usage:\n"
          "  xdft list                         list the available methods\n"
          "  xdft run <method|all> [options]   benchmark a method\n"
          "  xdft help                         show this message\n"
          "\n"
          "Options for `run`:\n"
          "  --sizes a,b,c   FFT lengths, powers of 2 (default: 256,...,2048)\n"
          "  --batch N       number of FFTs per run (default: 256)\n"
          "  --runs N        timed executions per size (default: 10)\n"
          "  --no-check      skip the comparison against cuFFT\n"
          "\n"
          "`all` runs every method that is not marked [wip]; a [wip] method\n"
          "can still be run by naming it.\n");
}

static void list_methods() {
  int count;
  const Method *m = get_methods(&count);
  for (int i = 0; i < count; i++)
    printf("  %-14s %s%s\n", m[i].name, m[i].description,
           m[i].wip ? " [wip]" : "");
}

static bool parse_sizes(const char *arg, std::vector<int> *sizes) {
  sizes->clear();
  std::string s(arg);
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t comma = s.find(',', pos);
    if (comma == std::string::npos) comma = s.size();
    int n = atoi(s.substr(pos, comma - pos).c_str());
    if (n < 2 || (n & (n - 1)) != 0) {
      fprintf(stderr, "invalid size '%s' (must be a power of 2)\n",
              s.substr(pos, comma - pos).c_str());
      return false;
    }
    sizes->push_back(n);
    pos = comma + 1;
  }
  return true;
}

static bool parse_int(const char *arg, const char *name, int *out) {
  int v = atoi(arg);
  if (v < 1) {
    fprintf(stderr, "invalid value '%s' for %s\n", arg, name);
    return false;
  }
  *out = v;
  return true;
}

int main(int argc, char **argv) {
  if (argc < 2 || !strcmp(argv[1], "help") || !strcmp(argv[1], "--help") ||
      !strcmp(argv[1], "-h")) {
    usage(argc < 2 ? stderr : stdout);
    return argc < 2 ? 2 : 0;
  }

  if (!strcmp(argv[1], "list")) {
    list_methods();
    return 0;
  }

  if (strcmp(argv[1], "run") != 0) {
    fprintf(stderr, "unknown command '%s'\n\n", argv[1]);
    usage(stderr);
    return 2;
  }
  if (argc < 3) {
    fprintf(stderr, "`run` needs a method name or `all`\n\n");
    usage(stderr);
    return 2;
  }

  // Defaults
  std::vector<int> sizes = {256, 512, 1024, 2048};
  int batch = 256, runs = 10;
  bool check = true;

  for (int i = 3; i < argc; i++) {
    bool has_val = i + 1 < argc;
    if (!strcmp(argv[i], "--no-check")) {
      check = false;
    } else if (!strcmp(argv[i], "--sizes") && has_val) {
      if (!parse_sizes(argv[++i], &sizes)) return 2;
    } else if (!strcmp(argv[i], "--batch") && has_val) {
      if (!parse_int(argv[++i], "--batch", &batch)) return 2;
    } else if (!strcmp(argv[i], "--runs") && has_val) {
      if (!parse_int(argv[++i], "--runs", &runs)) return 2;
    } else {
      fprintf(stderr, "unknown or incomplete option '%s'\n\n", argv[i]);
      usage(stderr);
      return 2;
    }
  }

  // Selecting the methods
  std::vector<const Method *> selected;
  if (!strcmp(argv[2], "all")) {
    int count;
    const Method *m = get_methods(&count);
    for (int i = 0; i < count; i++)
      if (!m[i].wip) selected.push_back(&m[i]);
  } else {
    const Method *m = find_method(argv[2]);
    if (!m) {
      fprintf(stderr, "unknown method '%s'. Available methods:\n", argv[2]);
      list_methods();
      return 2;
    }
    selected.push_back(m);
  }

  printf("batch=%d runs=%d\n", batch, runs);
  printf("%-14s %6s %12s %10s  %s\n", "method", "n", "time (s)", "GFLOPS",
         "check");

  int failures = 0;
  for (const Method *m : selected) {
    for (int n : sizes) {
      BenchResult r = run_benchmark(*m, n, batch, runs, check);
      const char *status = !r.checked ? "-" : (r.pass ? "ok" : "FAIL");
      printf("%-14s %6d %12.6f %10.2f  %s", m->name, n, r.seconds, r.gflops,
             status);
      if (r.checked) printf(" (max err %.2e)", r.max_err);
      printf("\n");
      fflush(stdout);
      if (r.checked && !r.pass) failures++;
    }
  }
  return failures ? 1 : 0;
}
