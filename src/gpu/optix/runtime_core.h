// OptiX's API behind plain types. The .cpp files of gpu_optix, compiled by cl.exe, make every OptiX call; the .cu
// files, compiled by nvcc (under which optix.h would select the device API), fill the device structs and call these.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "renderer.h" // denoise_report

namespace gpu::optix_core
{
struct aabb // OptixAabb's layout
{
    float lo[3];
    float hi[3];
};

struct instance // copied into an OptixInstance (16-byte aligned) by accel.cpp, with visibilityMask 0xFF and
{                // OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT set there (a zeroed mask would cull the instance)
    float transform[12];     // object to world, row-major 3×4
    unsigned int id;         // optixHitObjectGetInstanceId() of its hits: 0 the books' GAS, k + 1 mesh instance k
    unsigned int sbt_offset; // 0: the books' intersection programs; 3: the triangles' empty hit group
    std::uint64_t handle;    // its GAS
};

struct triangle_mesh // one triangle GAS's inputs, device pointers
{
    const void* positions; // float3, stride 12
    unsigned int vertex_count;
    const void* triangles; // uint3
    unsigned int triangle_count;
};

// An acceleration structure: its handle and the device memory holding it (freed with the last copy)
struct accel
{
    std::uint64_t handle = 0;
    std::shared_ptr<void> memory;
    std::size_t bytes = 0;
};

enum class raygen : int
{
    render = 0,
    guides = 1,
    pick = 2,
};

class core; // the context, the module, program groups, pipeline, SBT, the test pipelines, the denoiser (core.h)
struct core_deleter { void operator()(core*) const noexcept; };
using core_ptr = std::unique_ptr<core, core_deleter>;

core_ptr open_core();                       // optixInit + the context; std::runtime_error with start_optix's reason
std::string programs_sha256();              // the embedded PTX's
int headers_version();                      // OPTIX_VERSION of the headers this build compiled with: 90000 is 9.0
bool cache_enabled(const core& c);          // optixDeviceContextGetCacheEnabled, asked when the context was made
std::size_t params_bytes();                 // sizeof(optix_params), from scene.cu (nvcc knows the type): OptiX 9.1's
                                            // pipeline takes it (pipelineLaunchParamsSizeInBytes)
void compile_programs(core& c);             // the module and the pipeline, once; later calls do nothing
int continuation_stack_bytes(const core& c);
// Planned sizes (optixAccelComputeMemoryUsage, no device work): output + temp bytes
std::size_t custom_bytes(core& c, unsigned int count);
std::size_t triangles_bytes(core& c, std::span<const triangle_mesh> meshes);
// Builds on stream 0 and waits; temp memory is freed before returning (stubs until 88 and 91)
accel build_custom(core& c, const aabb* boxes, const std::uint8_t* kinds, unsigned int count); // device arrays (A)
std::vector<accel> build_triangles(core& c, std::span<const triangle_mesh> meshes);            // compacted (B)
accel build_instances(core& c, std::span<const instance> instances);                         // (B)
// optixLaunch of a raygen of the render pipeline on stream 0; params: device memory holding optix_params
void launch(core& c, raygen entry, const void* params, std::size_t params_bytes, unsigned int width,
            unsigned int height);
// GpuTests: a raygen of another module, its pipeline (that raygen + the render module's intersection groups, or the
// check module's probe group under probe_intersections) made once per entry and kept until the core is destroyed
// (before the render pipeline: core_deleter's order)
void launch_test(core& c, std::span<const unsigned char> ptx, const char* entry, const void* params,
                 std::size_t params_bytes, unsigned int width, unsigned int height, bool probe_intersections = false);
// The AI denoiser (denoiser.cpp, C): device pointers to float4 images; stream 0
void denoise(core& c, const void* beauty, const void* albedo, const void* normal, void* out, int width, int height);
// What the last denoise computed and holds (denoiser.cpp, C): renderer.h's denoise_report, valid false before the first
denoise_report last_denoise(const core& c);
}
