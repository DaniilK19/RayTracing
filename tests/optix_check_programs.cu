// GpuTests' OptiX programs: a module of its own (optix.cmake's optix_check_programs, embedded in GpuTests only as
// optix_check_ptx), never shipped. Its raygens run through optix_runtime::launch_test, in pipelines that link the
// render module's own intersection groups, so a check tests the intersection programs that ship: this module doesn't
// include intersections.cuh. It holds the optix-runtime suite's probes (Step 87), the tracer check's __raygen__check
// (track A, Step 88) and the probe intersection program __intersection__probe (P7, track B, Step 91). Not named
// *_tests.cu: GpuTests' glob would compile it as a suite of CUDA kernels.
#include <optix.h>
static_assert(OPTIX_VERSION >= 90000); // NVIDIA's optix.h, not a project header of that name
#include "optix_params.h"

// The SDK's braced form, value-initialized, as in programs.cu: a definition. Both modules declare `params` of one type,
// so a pipeline may link their groups together.
extern "C"
{
__constant__ gpu::optix_params params{};
}

#include <cstdint>

#include "ggx_energy.cuh"
#include "optix_tracer.cuh"

// --- The optix-runtime suite's probes (Step 87, tests/optix_tests.cu) -----------------------------------------------
// Launched by optix_runtime::launch_test without a scene (root 0): they trace nothing.

// P3: each launch index, written where it says. Pixel k = y·width + x owns words 4k to 4k + 3 of params.test.counts:
// x, y, z, and how many threads wrote there (1 when OptiX launches every index once). An index outside the launch's
// dimensions, which OptiX never gives, counts in the word after the last pixel instead of writing past the buffer.
extern "C" __global__ void __raygen__probe_index()
{
    const uint3 index = optixGetLaunchIndex();
    const uint3 size = optixGetLaunchDimensions();
    unsigned long long* const words = params.test.counts;
    if (index.x >= size.x || index.y >= size.y || index.z >= size.z)
    {
        atomicAdd(&words[4 * static_cast<std::uint64_t>(size.x) * size.y], 1ull);
        return;
    }
    unsigned long long* const pixel = words + 4 * (static_cast<std::uint64_t>(index.y) * size.x + index.x);
    pixel[0] = index.x;
    pixel[1] = index.y;
    pixel[2] = index.z;
    atomicAdd(&pixel[3], 1ull);
}

// P6: ggx_energy_table (ggx_energy.cuh), entry k at launch index k of a 1024 × 1 launch, into params.test.values. In
// a module the table is constant data that OptiX loads from the PTX's initializer, as the render module's GGX metal
// reads it: it must equal, bit for bit, the table of GpuTests' own CUDA code, which a plain kernel copies out.
extern "C" __global__ void __raygen__probe_energy()
{
    const unsigned int k = optixGetLaunchIndex().x;
    if (k < static_cast<unsigned int>(gpu::ggx_energy_size * gpu::ggx_energy_size))
    {
        params.test.values[k] = gpu::ggx_energy_table[k];
    }
}

// --- The tracer check (track A, Step 88): optix_tracer's closest and visibility against the BVH's, ray by ray -------
// An empty entry until then, so that the module's entries are all there from Step 87.
extern "C" __global__ void __raygen__check()
{
}
// --- end of the tracer check ----------------------------------------------------------------------------------------

// --- The probe intersection program (track B, Step 91: P7) -----------------------------------------------------------
// The world ray an intersection program reads under an identity instance, against the floats traced; launch_test's
// pipelines point every custom kind's SBT record at it under optix_upload_options::probe_intersections. An empty entry
// until then: it reports nothing.
extern "C" __global__ void __intersection__probe()
{
}
// --- end of the probe intersection program ---------------------------------------------------------------------------
