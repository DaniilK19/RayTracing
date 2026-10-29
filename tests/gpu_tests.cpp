// GpuTests: numeric tests of the GPU renderer's device functions, on the GPU: the samplers draw what their densities
// say, the densities integrate to 1, the estimators converge to known values. See gpu_test.h for the checks and the
// report; the suites are the files tests/*_tests.cu.
//   GpuTests [all | SUITE...] [--quick] [--list]
// Exit code: 0 every check passes, 1 a check fails, 2 an error (an unknown suite or option, a CUDA error), 77 no CUDA
// device or no driver (as ctest's SKIP_RETURN_CODE: the tests are skipped on a computer without an NVIDIA GPU). A
// driver or runtime that is there but fails is an error, 2: a broken installation must not pass for a skipped test.

// The headers before `import std;` (see renderer.h)
#include <cuda_runtime_api.h>

#include "gpu_test.h"
#include "renderer.h"
#include "statistics.h"

import std;

namespace gpu_test
{
std::vector<test_entry>& registry()
{
    static std::vector<test_entry> tests;
    return tests;
}

namespace
{
std::string suite_name(const char* identifier)
{
    std::string name = identifier;
    std::ranges::replace(name, '_', '-');
    return name;
}

std::int64_t device_bytes = 0;
std::int64_t device_peak = 0;

std::int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

constexpr double suite_seconds_limit = 10;
constexpr std::int64_t suite_bytes_limit = 64 << 20;
}

void note_device_bytes(std::int64_t change)
{
    device_bytes += change;
    device_peak = std::max(device_peak, device_bytes);
}

std::int64_t device_peak_bytes() { return device_peak; }

void reset_device_peak() { device_peak = device_bytes; }

void context::mean(std::string_view name, double estimate, double standard_error, double expected, double z_limit)
{
    const double z = standard_error > 0 ? (estimate - expected) / standard_error
                     : estimate == expected ? 0.0
                                            : std::numeric_limits<double>::infinity();
    results_.push_back({.name = std::string(name), .kind = check_kind::mean, .estimate = estimate,
                        .expected = expected, .statistic = z, .limit = z_limit});
}

void context::chi2(std::string_view name, double statistic, int dof, double p)
{
    results_.push_back({.name = std::string(name), .kind = check_kind::chi2, .estimate = statistic,
                        .expected = static_cast<double>(dof), .statistic = p, .dof = dof});
}

void context::must_reject(std::string_view name, double statistic, int dof, double p, double p_max)
{
    results_.push_back({.name = std::string(name), .kind = check_kind::must_reject, .estimate = statistic,
                        .expected = static_cast<double>(dof), .statistic = p, .limit = p_max, .dof = dof});
}

void context::exact(std::string_view name, double estimate, double expected, double tolerance)
{
    results_.push_back({.name = std::string(name), .kind = check_kind::exact, .estimate = estimate,
                        .expected = expected, .statistic = std::abs(estimate - expected), .limit = tolerance});
}

void context::count(std::string_view name, std::int64_t found, std::int64_t allowed, std::int64_t out_of)
{
    results_.push_back({.name = std::string(name), .kind = check_kind::count, .estimate = static_cast<double>(found),
                        .limit = static_cast<double>(allowed), .out_of = out_of});
}

void context::info(std::string_view name, std::string_view text)
{
    infos_.emplace_back(results_.size(), std::format("{}.{}: {}, INFO", suite_, name, text));
}

void context::begin_suite(std::string_view suite)
{
    suite_ = suite;
    results_.clear();
    infos_.clear();
    reset_device_peak();
    start_ns_ = now_ns();
}

int context::end_suite()
{
    // The suite's budget, as two checks of its own
    const double seconds = static_cast<double>(now_ns() - start_ns_) * 1e-9;
    count("budget-gpu-memory-bytes", device_peak_bytes(), suite_bytes_limit, suite_bytes_limit);
    exact("budget-seconds", seconds, 0, suite_seconds_limit);

    const auto chi2_tests = std::ranges::count(results_, check_kind::chi2, &check_result::kind);
    const double threshold = sidak_threshold(0.01, static_cast<int>(chi2_tests));
    int failures = 0;
    std::size_t next_info = 0;
    for (std::size_t i = 0; i < results_.size(); ++i)
    {
        for (; next_info < infos_.size() && infos_[next_info].first <= i; ++next_info)
        {
            std::println("{}", infos_[next_info].second);
        }
        const check_result& r = results_[i];
        const bool statistical =
            r.kind == check_kind::mean || r.kind == check_kind::chi2 || r.kind == check_kind::must_reject;
        bool pass = false;
        std::string detail;
        switch (r.kind)
        {
        case check_kind::mean:
            pass = std::abs(r.statistic) <= r.limit;
            detail = std::format("{:.9g}, expected {:.9g}, z = {:.3f} (|z| <= {})", r.estimate, r.expected,
                                 r.statistic, r.limit);
            break;
        case check_kind::chi2:
            pass = r.statistic >= threshold;
            detail = std::format("chi2 = {:.2f} ({} dof), expected {}, p = {:.4g} (>= {:.4g})", r.estimate, r.dof,
                                 r.dof, r.statistic, threshold);
            break;
        case check_kind::must_reject:
            pass = r.statistic < r.limit;
            detail = std::format("chi2 = {:.2f} ({} dof), expected a rejection, p = {:.4g} (< {:.0e})", r.estimate,
                                 r.dof, r.statistic, r.limit);
            break;
        case check_kind::exact:
            pass = r.statistic <= r.limit;
            detail = std::format("{:.9g}, expected {:.9g}, |difference| = {:.3g} (<= {:.3g})", r.estimate, r.expected,
                                 r.statistic, r.limit);
            break;
        case check_kind::count:
            pass = r.estimate <= r.limit;
            detail = std::format("{} of {}, expected at most {}", static_cast<std::int64_t>(r.estimate), r.out_of,
                                 static_cast<std::int64_t>(r.limit));
            break;
        }
        const bool skipped = statistical && quick_;
        std::println("{}.{}: {}, {}", suite_, r.name, detail, skipped ? "SKIP (quick)" : pass ? "PASS" : "FAIL");
        failures += !skipped && !pass ? 1 : 0;
    }
    for (; next_info < infos_.size(); ++next_info)
    {
        std::println("{}", infos_[next_info].second);
    }
    return failures;
}
}

int main(int argc, char* argv[])
{
    using namespace gpu_test;
    bool quick = false;
    bool list = false;
    std::vector<std::string> wanted;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg == "--quick")
        {
            quick = true;
        }
        else if (arg == "--list")
        {
            list = true;
        }
        else if (arg == "-h" || arg == "--help")
        {
            std::println("Usage: GpuTests [all | SUITE...] [--quick] [--list]\n"
                         "Numeric tests of the GPU renderer's device functions (see tests/gpu_test.h).\n"
                         "Exit code: 0 pass, 1 a check fails, 2 an error, 77 no CUDA device.");
            return 0;
        }
        else if (arg.starts_with('-'))
        {
            std::println("Error: unknown option '{}' (GpuTests --help)", arg);
            return 2;
        }
        else
        {
            wanted.emplace_back(arg);
        }
    }

    // The suites by name, each with its tests in the order its file registered them
    std::map<std::string, std::vector<const test_entry*>> suites;
    for (const test_entry& t : registry())
    {
        suites[suite_name(t.suite)].push_back(&t);
    }
    if (list)
    {
        for (const auto& [name, tests] : suites)
        {
            std::println("{}: {}", name,
                         tests | std::views::transform([](const test_entry* t) { return std::string(t->name); }) |
                             std::views::join_with(std::string(", ")) | std::ranges::to<std::string>());
        }
        return 0;
    }
    for (const std::string& name : wanted)
    {
        if (name != "all" && !suites.contains(name))
        {
            std::println("Error: no suite '{}' (GpuTests --list names them)", name);
            return 2;
        }
    }
    if (wanted.empty() || std::ranges::contains(wanted, std::string("all")))
    {
        wanted.clear();
        for (const auto& [name, tests] : suites)
        {
            wanted.push_back(name);
        }
    }

    // Skipped only without a GPU or without a driver; any other failure to start CUDA is an error
    int devices = 0;
    const cudaError_t counted = cudaGetDeviceCount(&devices);
    if (counted == cudaErrorNoDevice || counted == cudaErrorInsufficientDriver ||
        (counted == cudaSuccess && devices == 0))
    {
        std::println("No CUDA device ({}): the tests are skipped", cudaGetErrorString(counted));
        return 77;
    }
    try
    {
        const gpu::device_info device = gpu::query_device();
        std::println("GpuTests on a GPU of compute capability {}.{}{}", device.compute_capability_major,
                     device.compute_capability_minor, quick ? ", quick: statistical verdicts skipped" : "");
    }
    catch (const std::exception& error)
    {
        std::println("Error: CUDA can't start: {}", error.what());
        return 2;
    }

    int failures = 0;
    context ctx(quick);
    for (const std::string& name : wanted)
    {
        ctx.begin_suite(name);
        try
        {
            for (const test_entry* t : suites[name])
            {
                t->run(ctx);
            }
        }
        catch (const std::exception& error)
        {
            ctx.end_suite();
            std::println("Error in suite {}: {}", name, error.what());
            return 2;
        }
        failures += ctx.end_suite();
    }
    std::println("{}", failures == 0 ? "GpuTests: PASS" : std::format("GpuTests: FAIL ({} checks)", failures));
    return failures == 0 ? 0 : 1;
}
