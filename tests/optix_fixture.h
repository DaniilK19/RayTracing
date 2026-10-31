// GpuTests' OptiX: one optix_runtime per process for every OptiX suite (optix-runtime, optix-tracer, optix-render,
// optix-triangles, optix-pbr, optix-sessions, guides, denoise), made by start_optix() when the first of them asks for it. That suite
// pays for OptiX's start (the driver loads nvoptix.dll, optixInit, the context): its seconds are printed as INFO and
// left out of the suite's 10 s budget (gpu_test.h's exclude_seconds). The programs' first compile is the asking
// suite's to leave out, through excluded_seconds around compile_programs() or a test pipeline's first launch_test:
// whichever OptiX suite runs first in a process compiles them for every one after it (OptiX keeps them per runtime).
//
// Where OptiX can't run (a build without its headers, a driver older than CUDA 13.4, without OptiX or with one older
// than 9.0), every check that asks is "SKIP (OptiX unavailable: <why>)", never a failure, or under --require-optix,
// which every gate passes, "FAIL (OptiX unavailable: <why>)": a gate must not pass on checks that never ran.
// start_optix() is asked once per process; its answer stands for the suites after.
//
// The runtime is held in a std::shared_ptr that main() empties before it returns (release_at_return, gpu_tests.cpp),
// never left to a static's destructor at the program's exit: by then cudart_static's own exit handlers may have torn
// CUDA down under the runtime's GPU memory and OptiX's context (the lifetimes RayTracingGPU and the viewer keep too).
//
// Plain C++, no OptiX or CUDA type (optix_runtime is only declared here): gpu_tests.cpp includes it before its `import
// std;`, and a suite that calls the runtime includes optix_runtime.cuh too (nvcc).
#pragma once

#include <chrono>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "gpu_test.h"
#include "start_optix.h"

namespace gpu_test::optix
{
namespace state
{
inline std::shared_ptr<gpu::optix_runtime> held; // the process's runtime, once made; emptied by release_at_return
inline bool asked = false;                       // start_optix() was called: at most once per process
inline std::string problem;                      // what it threw, when it threw
}

// The process's OptiX for the check `check`: the runtime, made on the first call of the process, or nullptr where
// OptiX can't run, after `check` was skipped, or failed under --require-optix, with the reason. A copy of the
// shared_ptr, for a session_options that wants one; the holder keeps the runtime until main returns.
inline std::shared_ptr<gpu::optix_runtime> runtime(context& ctx, std::string_view check)
{
    if (!state::asked)
    {
        state::asked = true;
        const auto start = std::chrono::steady_clock::now();
        try
        {
            state::held = gpu::start_optix();
        }
        catch (const std::runtime_error& error) // start_optix.h's reasons; anything else is a bug, the suite's error
        {
            state::problem = error.what();
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        ctx.exclude_seconds(seconds);
        ctx.info("optix-start", std::format("{:.3f} s in start_optix(), once per process (outside budget-seconds)",
                                            seconds));
    }
    if (!state::held)
    {
        const std::string reason = "OptiX unavailable: " + state::problem;
        if (ctx.require_optix())
        {
            ctx.fail(check, reason);
        }
        else
        {
            ctx.skip(check, reason);
        }
    }
    return state::held;
}

// The wall seconds `compile` takes, left out of the suite's budget: for OptiX's first compile in a process
// (compile_programs(), or the first launch_test of a check module's entry, which makes its pipeline). A call that
// finds them compiled already costs microseconds, and excludes as much.
template <class F>
double excluded_seconds(context& ctx, F&& compile)
{
    const auto start = std::chrono::steady_clock::now();
    std::forward<F>(compile)();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ctx.exclude_seconds(seconds);
    return seconds;
}

// main()'s first object (gpu_tests.cpp): when main returns, by any of its returns, its destructor frees the runtime
// (in core_deleter's order, ending with OptiX's context) while CUDA is still up
struct release_at_return
{
    release_at_return() = default;
    ~release_at_return() { state::held.reset(); }
    release_at_return(const release_at_return&) = delete;
    release_at_return& operator=(const release_at_return&) = delete;
};
}
