// Suite optix-runtime: OptiX started from the graphics driver, the programs compiled by it, and a launch of the check
// module's probes (tests/optix_check_programs.cu), before any scene or acceleration structure (Step 87's probes; P4
// and P5 come with the books' GAS in Step 88, P7 with the instances in Step 91). Everything goes through the abstract
// runtime of the fixture (optix_fixture.h): compile_programs() and the scene-less launch_test, so a build without
// OptiX links this file as it is and every check skips, saying why.
//   P1  start_optix() succeeds (else SKIP, or FAIL under --require-optix);
//   P2  the wall and CPU seconds of the render pipeline (compile_programs(): the render module, its stub programs in
//       Step 87) and of the check pipeline (the first launch_test: the check module and its raygen's pipeline), INFO,
//       outside the 10 s budget: cold in the first process after the PTX or the driver changed, warm (OptiX's disk
//       cache) in the next;
//   P3  a 64 × 64 launch of __raygen__probe_index writes optixGetLaunchIndex() into every pixel, once, and nowhere
//       else; it says whether OptiX's validation mode is on, since P9 is P3 again in a process started with
//       RAYTRACING_OPTIX_VALIDATION=1 (a context option, so it can't change inside a process), whose stderr must
//       carry no message from OptiX;
//   P6  ggx_energy_table read by the check module equals, bit for bit, the table a plain CUDA kernel reads;
//   P8  the render pipeline's continuation stack per thread, kernel().local_bytes after compile_programs(), INFO
//       (what replaced the BVH's 256-byte stack).
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "ggx_energy.cuh"
#include "gpu_test.cuh"
#include "gpu_test.h"
#include "optix/embedded.h"
#include "optix_fixture.h"
#include "optix_runtime.cuh"
#include "start_optix.h"

// GetProcessTimes, for the CPU seconds of OptiX's compile. Last, after every other header: windows.h defines macros
// (near, far) that a header's names could meet.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace
{
// The check module's PTX as the runtime takes it: always this span, since the runtime keeps one module per PTX address
// and one pipeline per (PTX, entry) for the process
std::span<const unsigned char> check_module()
{
    return {optix_check_ptx, optix_check_ptx_size};
}

// The CPU time the process has used so far, user and kernel, in seconds. OptiX compiles on the CPU, on the calling
// thread (runtime.cpp's plain optixModuleCreate), so the seconds a compile adds are its own and those of CUDA's
// threads, as gpu.ps1's cpu_seconds counts them for the whole process.
double process_cpu_seconds()
{
    FILETIME created{};
    FILETIME exited{};
    FILETIME kernel{};
    FILETIME user{};
    if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user) == 0)
    {
        return 0; // never on Windows; then the INFO says 0 s of CPU
    }
    const auto ticks = [](const FILETIME& t) {
        return (static_cast<std::uint64_t>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
    };
    return static_cast<double>(ticks(kernel) + ticks(user)) * 1e-7; // FILETIME counts 100 ns
}

struct compile_seconds
{
    double wall;
    double cpu;
};

// A first compile of OptiX's, left out of the suite's budget (the fixture's excluded_seconds), with its CPU seconds
template <class F>
compile_seconds timed_compile(gpu_test::context& ctx, F&& compile)
{
    const double cpu = process_cpu_seconds();
    const double wall = gpu_test::optix::excluded_seconds(ctx, std::forward<F>(compile));
    return {.wall = wall, .cpu = process_cpu_seconds() - cpu};
}

// Whether the context was made in validation mode: runtime.cpp's own rule (RAYTRACING_OPTIX_VALIDATION is "1" when the
// context is made), read again here, from the same environment of the same process
bool validation_on()
{
    char* value = nullptr;
    std::size_t length = 0;
    const bool on = _dupenv_s(&value, &length, "RAYTRACING_OPTIX_VALIDATION") == 0 && value != nullptr &&
                    std::string_view(value) == "1";
    std::free(value);
    return on;
}

constexpr int energy_entries = gpu::ggx_energy_size * gpu::ggx_energy_size;

// The table as GpuTests' CUDA code holds it (the fatbin nvcc made from ggx_energy.cuh), entry k by thread k. A
// template, as every kernel of the tests: nvcc's host stub of a kernel that isn't one, in an unnamed namespace, draws
// MSVC's warning C4211.
template <int Count>
__global__ void copy_energy_table_kernel(float* out)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < Count)
    {
        out[k] = gpu::ggx_energy_table[k];
    }
}

// P1: OptiX starts here, the fixture's first call (whose INFO line gives its seconds); what it says of itself
GPU_TEST(optix_runtime, start)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "start");
    if (!runtime)
    {
        return;
    }
    ctx.info("describe", gpu::describe(*runtime));
    ctx.info("disk-cache", gpu::optix_cache_enabled(*runtime)
                               ? "on: OptiX compiles a module once per PTX and driver, and keeps it"
                               : "off: every process compiles the programs again (OPTIX_CACHE_MAXSIZE=0, or a cache "
                                 "that could not start)");
    ctx.exact("start", 1, 1, 0);
}

// P2 and P8: the render pipeline, as the first upload makes it, then the check module's first pipeline (its probe
// raygen with the render module's intersection groups), made by the first launch_test of a 1 × 1 probe
GPU_TEST(optix_runtime, compile)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "render-pipeline");
    if (!runtime)
    {
        return;
    }
    const compile_seconds render = timed_compile(ctx, [&runtime] { runtime->compile_programs(); });
    ctx.info("render-pipeline-seconds",
             std::format("{:.3f} s, {:.3f} s of CPU: compile_programs(), the render module and its pipeline (once per "
                         "runtime)",
                         render.wall, render.cpu));
    const gpu::kernel_info kernel = runtime->kernel();
    ctx.info("continuation-stack-bytes",
             std::format("{} bytes per thread: the render pipeline's continuation stack (kernel().local_bytes), "
                         "programs {}",
                         kernel.local_bytes, kernel.programs));

    const gpu_test::device_buffer<unsigned long long> words(4 + 1); // one pixel, and the outside word
    const compile_seconds check = timed_compile(ctx, [&] {
        runtime->launch_test(check_module(), "__raygen__probe_index", 1, 1, {.counts = words.get()});
        gpu_test::finish_launch("the check module's first launch");
    });
    ctx.info("check-pipeline-seconds",
             std::format("{:.3f} s, {:.3f} s of CPU: the first launch_test, the check module and its probe's pipeline",
                         check.wall, check.cpu));
}

// P3 (and P9 in a process of its own): every launch index of a 64 × 64 launch, where it says, once
GPU_TEST(optix_runtime, launch_index)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "launch-index");
    if (!runtime)
    {
        return;
    }
    ctx.info("validation", validation_on() ? "on (RAYTRACING_OPTIX_VALIDATION=1): OptiX checks every call, and says "
                                             "what it finds on stderr, where P9 looks"
                                           : "off");
    constexpr std::size_t side = 64;
    constexpr std::size_t pixels = side * side;
    const gpu_test::device_buffer<unsigned long long> words(4 * pixels + 1); // zeros: a pixel never written fails
    runtime->launch_test(check_module(), "__raygen__probe_index", static_cast<int>(side), static_cast<int>(side),
                         {.counts = words.get()});
    gpu_test::finish_launch("a launch of __raygen__probe_index");
    const std::vector<unsigned long long> out = words.download();
    std::int64_t wrong = 0;
    for (std::size_t y = 0; y < side; ++y)
    {
        for (std::size_t x = 0; x < side; ++x)
        {
            const unsigned long long* const w = out.data() + 4 * (y * side + x);
            wrong += w[0] == x && w[1] == y && w[2] == 0 && w[3] == 1 ? 0 : 1;
        }
    }
    ctx.count("launch-index", wrong, 0, static_cast<std::int64_t>(pixels));
    ctx.count("launch-index-outside", static_cast<std::int64_t>(out[4 * pixels]), 0,
              static_cast<std::int64_t>(pixels));
}

// P6: the GGX energy table through OptiX's module loading and through CUDA's, bit for bit
GPU_TEST(optix_runtime, energy_table)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "ggx-energy-table");
    if (!runtime)
    {
        return;
    }
    const gpu_test::device_buffer<float> through_optix(energy_entries);
    runtime->launch_test(check_module(), "__raygen__probe_energy", energy_entries, 1,
                         {.values = through_optix.get()});
    gpu_test::finish_launch("a launch of __raygen__probe_energy");
    const gpu_test::device_buffer<float> through_cuda(energy_entries);
    copy_energy_table_kernel<energy_entries><<<(energy_entries + 127) / 128, 128>>>(through_cuda.get());
    gpu_test::finish_launch("copying the energy table");
    const std::vector<float> a = through_optix.download();
    const std::vector<float> b = through_cuda.download();
    std::int64_t differ = 0;
    for (std::size_t k = 0; k < a.size(); ++k)
    {
        differ += std::bit_cast<std::uint32_t>(a[k]) == std::bit_cast<std::uint32_t>(b[k]) ? 0 : 1;
    }
    ctx.count("ggx-energy-table", differ, 0, energy_entries);
}
}
