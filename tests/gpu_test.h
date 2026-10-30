// GpuTests' registry and report: numeric tests of the GPU renderer's device functions, run on the GPU. Each test file
// tests/<x>_tests.cu registers its tests under a suite name with GPU_TEST; a test makes checks through its context,
// and the program prints one line per check:
//   <suite>.<check>: <estimate>, expected <value>, <the statistic and its limit>, PASS|FAIL|SKIP (quick)
// or, for a check with no number to show, its reason in the numbers' place:
//   <suite>.<check>: SKIP (<reason>)        <suite>.<check>: FAIL (<reason>)
// The kinds of check:
//   mean         a Monte Carlo mean against its exact value: |z| = |estimate - expected| / SE at most 4;
//   chi2         Pearson's χ² test of drawn samples against a density: the p-value at least 1 - 0.99^(1/T), T being
//                the number of χ² tests in the suite (Šidák: the suite fails by chance in at most 1 run of 100);
//   must_reject  the same test on samples that are wrong on purpose (the harness's self-check): p below 10^-6;
//   exact        a deterministic identity: |estimate - expected| within a tolerance;
//   count        events that must not happen (NaN or zero-density draws), at most an allowed number;
//   skip         a check that couldn't run here (no OptiX in this build or in the driver): never a failure, so a suite
//                whose checks all skip passes; the program's last line counts them, so that they can't go unseen;
//   fail         a check that failed for a reason that is no number (OptiX unavailable under --require-optix, which
//                every gate passes: a gate must not pass on checks that never ran).
// --quick runs every kernel on at most 10^4 samples per check (for compute-sanitizer, which slows them 10 to 100
// times) and skips the statistical verdicts; exact and count checks still decide. Every suite ends with two budget
// checks: at most 10 s of wall time and 64 MB of GPU memory allocated at once (device_buffer counts it, and refuses a
// single allocation above 1 GB outright: on Windows the driver would take a too large one from the system's memory).
// The seconds a suite excludes (exclude_seconds: OptiX's creation and its first compile of the programs, which the
// first OptiX suite of a process pays for every one after it) aren't counted, and an info line before budget-seconds
// says how many. info lines print a number without a verdict.
//
// This header is plain C++, shared by gpu_tests.cpp (cl.exe) and the test files (nvcc): no CUDA in it. The device
// helpers are in gpu_test.cuh.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gpu_test
{
enum class check_kind
{
    mean,
    chi2,
    must_reject,
    exact,
    count,
    skip,
    fail,
};

// One check's numbers, kept until the suite ends: a χ² test's threshold depends on how many the suite has
struct check_result
{
    std::string name;
    check_kind kind = check_kind::exact;
    double estimate = 0;
    double expected = 0;
    double statistic = 0; // mean: z; chi2, must_reject: the p-value; exact: |estimate - expected|
    double limit = 0;     // mean: the largest |z|; must_reject: the largest p; exact: the tolerance; count: allowed
    int dof = 0;          // chi2, must_reject: degrees of freedom
    std::int64_t out_of = 0; // count: how many were looked at
    std::string reason;      // skip, fail: why, printed in the numbers' place
};

class context
{
public:
    context(bool quick, bool optix_required) : quick_(quick), optix_required_(optix_required) {}

    bool quick() const { return quick_; }
    // --require-optix was given: an OptiX fixture must turn its SKIP into fail()
    bool require_optix() const;
    // A value the command line gave a suite (GpuTests ggx-energy --write FILE): --write or --check, if given
    const std::string* argument(std::string_view name) const;
    void set_argument(std::string name, std::string value) { arguments_.emplace_back(std::move(name), std::move(value)); }
    // The number of samples a check draws: `full`, or at most 10^4 with --quick
    std::int64_t samples(std::int64_t full) const { return quick_ && full > 10000 ? 10000 : full; }

    void mean(std::string_view name, double estimate, double standard_error, double expected, double z_limit = 4);
    void chi2(std::string_view name, double statistic, int dof, double p);
    void must_reject(std::string_view name, double statistic, int dof, double p, double p_max = 1e-6);
    void exact(std::string_view name, double estimate, double expected, double tolerance);
    void count(std::string_view name, std::int64_t found, std::int64_t allowed, std::int64_t out_of);
    void info(std::string_view name, std::string_view text);
    // A check that couldn't run here (no OptiX in this build or driver, ...): "<suite>.<name>: SKIP (<reason>)", never
    // a failure
    void skip(std::string_view name, std::string_view reason);
    // A check that failed for a reason that is no number (OptiX unavailable under --require-optix):
    // "<suite>.<name>: FAIL (<reason>)"
    void fail(std::string_view name, std::string_view reason);
    // Seconds of the current suite that its 10 s budget-seconds check doesn't count (OptiX's creation and first
    // compile); printed as an INFO line. A duration measured inside the suite: at least 0 and finite, else an error
    void exclude_seconds(double seconds);

    // The runner's side: the checks of the suite in progress, and its verdict once it ends (prints the lines)
    void begin_suite(std::string_view suite);
    // Returns the number of failed checks
    int end_suite();
    // The skip()s of every suite ended so far: the program's last line counts them
    int skipped_checks() const { return skipped_; }

private:
    bool quick_;
    bool optix_required_;
    int skipped_ = 0;
    double excluded_seconds_ = 0; // exclude_seconds() of the suite in progress
    std::string suite_;
    std::vector<check_result> results_;
    std::vector<std::pair<std::size_t, std::string>> infos_; // printed after the check they follow
    std::vector<std::pair<std::string, std::string>> arguments_;
    std::int64_t start_ns_ = 0;                             // when the suite began (steady clock)
};

// The GPU memory the tests hold (device_buffer reports every allocation and every release), and the most held at once
// since reset_device_peak()
void note_device_bytes(std::int64_t change);
std::int64_t device_peak_bytes();
void reset_device_peak();

using test_function = void (*)(context&);

struct test_entry
{
    const char* suite;
    const char* name;
    test_function run;
};

// The tests of every file, in the order the files registered them (gpu_tests.cpp sorts the suites by name)
std::vector<test_entry>& registry();

struct registrar
{
    registrar(const char* suite, const char* name, test_function run) { registry().push_back({suite, name, run}); }
};
}

// GPU_TEST(harness, cosine) { ... ctx.chi2(...); } defines a test function and registers it before main() runs. The
// registrar is a static object of the test file: its constructor runs during the program's start-up. The suite is an
// identifier, and its name on the command line and in the report has '-' for '_' (GPU_TEST(env_map, ...): env-map).
#define GPU_TEST_JOIN2(a, b) a##b
#define GPU_TEST_JOIN(a, b) GPU_TEST_JOIN2(a, b)
#define GPU_TEST(suite, name)                                                                                          \
    static void GPU_TEST_JOIN(gpu_test_##suite##_, name)(gpu_test::context & ctx);                                    \
    static const gpu_test::registrar GPU_TEST_JOIN(gpu_test_registrar_##suite##_, name){                               \
        #suite, #name, &GPU_TEST_JOIN(gpu_test_##suite##_, name)};                                                     \
    static void GPU_TEST_JOIN(gpu_test_##suite##_, name)(gpu_test::context & ctx)
