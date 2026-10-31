// What runtime_core.h's opaque `core` holds: OptiX's objects of one program, owned by runtime.cpp, which makes them and
// destroys them in the reverse order (core_deleter). Only the .cpp files of gpu_optix include this header: it brings
// OptiX's host API, which optix.h replaces by the device API under nvcc.
#pragma once

// OptiX's headers at /W4: optix_stubs.h defines static functions a file may not call (C4505), and includes windows.h
#pragma warning(push, 0)
#include <optix.h>
#include <optix_stubs.h>
#pragma warning(pop)

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "device_memory.h"
#include "runtime_core.h"

namespace gpu::optix_core
{
// Every OptiX call of runtime.cpp, accel.cpp and denoiser.cpp but optixInit goes through this one check: its error as
// std::runtime_error, with what the call was for and OptiX's own name and description of the error (runtime.cpp)
void check_optix(OptixResult result, const char* call);

// The program groups of the render module, in the order of their SBT records: the three raygens (runtime_core.h's
// raygen numbers them), the miss group (no program: optixTraverse invokes none), the books' three intersection
// programs (the books' GAS's sbtIndexOffset 0, 1, 2) and the triangles' empty hit group (a mesh instance's sbtOffset 3)
enum group : int
{
    raygen_render,
    raygen_guides,
    raygen_pick,
    miss,
    hit_sphere,
    hit_moving_sphere,
    hit_quad,
    hit_triangles,
    group_count,
};

// The denoiser's state (denoiser.cpp, track C, Step 94): made at the first denoise, destroyed before the context
struct denoiser_state;
struct denoiser_deleter
{
    void operator()(denoiser_state* state) const noexcept;
};

// A raygen of another module (GpuTests' check module) in a pipeline of its own: that raygen, the render module's miss
// group, and its four hit groups, or under probe_intersections the check module's probe for the three custom kinds
struct test_pipeline
{
    OptixProgramGroup raygen = nullptr;
    OptixProgramGroup probe = nullptr; // __intersection__probe of the test module (probe_intersections), or none
    OptixPipeline pipeline = nullptr;
    device_array<unsigned char> records{0}; // its SBT records: raygen, miss, four hit groups
    OptixShaderBindingTable sbt{};
};

class core
{
public:
    OptixDeviceContext context = nullptr;
    bool cache_enabled = false; // optixDeviceContextGetCacheEnabled, when the context was made
    // The render module, its program groups (by `group`) and its pipeline: compile_programs, once
    OptixModule module = nullptr;
    std::vector<OptixProgramGroup> groups;
    OptixPipeline pipeline = nullptr;
    device_array<unsigned char> records{0}; // the SBT records, one header each, in the order of `group`
    OptixShaderBindingTable sbt{};          // raygenRecord at raygen_render's; a launch points it at its own
    int continuation_stack = 0;             // bytes per thread, from the stack sizes of every group
    // GpuTests' modules, by the address of their PTX, and their pipelines, by (PTX, entry, probe_intersections)
    std::map<const unsigned char*, OptixModule> test_modules;
    std::map<std::tuple<const unsigned char*, std::string, bool>, test_pipeline> tests;
    std::unique_ptr<denoiser_state, denoiser_deleter> denoiser;
    unsigned int denoise_tile_limit = 0; // GpuTests: set_denoise_tile_limit's side (0: none)
};
}
