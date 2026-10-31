// What renderer.cu asks of OptiX, and nothing more: an abstract class, so gpu_renderer calls it through virtual
// functions without linking the library that implements it (gpu_optix links gpu_renderer). No OptiX header and no
// device type of the quality kernel: quality_scene stays opaque, as in quality.cuh. nvcc translation units only.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector_types.h>

#include "camera.cuh"
#include "quality.cuh"
#include "renderer.h"
#include "scene.cuh"

namespace gpu
{
// GpuTests only: what a test raygen of the check module gets besides the scene (optix_params::test)
struct optix_test_launch
{
    unsigned long long* counts = nullptr; // device: the test's counters
    float* values = nullptr;              // device: the test's other outputs
    int fault = 0;                        // a planted fault of the check raygen (0: none)
    int samples = 1;
    std::uint64_t seed = 0;
};

struct optix_upload_options
{
    bool force_instance_root = false; // GpuTests: the books' GAS under an identity instance, as in a mesh scene
    bool probe_intersections = false; // GpuTests (P7, Step 91): launch_test's pipelines point every custom kind's SBT
                                      // record at the check module's __intersection__probe
};

namespace detail
{
// A scene in OptiX's acceleration structures, for one session (src/gpu/optix/scene.cu)
class optix_scene
{
public:
    virtual ~optix_scene() = default;
    // The view's size: grows the per-pixel scratch slots of glTF materials (never shrinks them)
    virtual void prepare_view(int width, int height) = 0;
    // As launch_quality (quality.cuh): passes [first_pass, first_pass + pass_count) of rows [first_row, first_row +
    // row_count), on stream 0, asynchronous; a band of more than 2^30 pixels is split into several optixLaunch calls
    virtual void launch(const camera<float>& cam, std::uint64_t seed, int first_pass, int pass_count, int first_row,
                        int row_count, double* sums, unsigned long long* counts) = 0;
    // The denoiser's guides of the whole view: 16 samples per pixel, float4 per pixel, on stream 0
    virtual void guides(const camera<float>& cam, std::uint64_t seed, float4* albedo, float4* normal) = 0;
    // The nearest surface along a ray (no media), its distance in lengths of `direction`; infinity: none. Synchronous.
    virtual double pick(const std::array<double, 3>& origin, const std::array<double, 3>& direction) = 0;
    // GpuTests: raygen `entry` of the module `ptx` (its pipeline made once per runtime and entry: that raygen with the
    // render module's intersection groups) over width × height, at this scene's root. Probes that trace nothing use
    // optix_runtime::launch_test instead, which needs no scene.
    virtual void launch_test(std::span<const unsigned char> ptx, const char* entry, int width, int height,
                             const camera<float>& cam, const optix_test_launch& test) = 0;
};
}

// OptiX for the whole program (implemented in src/gpu/optix/scene.cu on runtime.cpp's core)
class optix_runtime
{
public:
    virtual ~optix_runtime() = default;
    // The scene's books' GAS, its meshes' GASes and the IAS over them, and the quality view lengthened for meshes.
    // base: device_scene's view; quality: the session's quality scene (quality_view_of reads its view); ids: the BVH's.
    // The first call compiles the programs (OptiX's disk cache serves later runs). Throws std::runtime_error.
    virtual std::unique_ptr<detail::optix_scene> upload(const scene_desc& scene, const scene_view<float>& base,
                                                        const detail::quality_scene& quality,
                                                        std::span<const int> sphere_ids, std::span<const int> quad_ids,
                                                        const optix_upload_options& options = {}) = 0;
    // name "optix", programs (8 hex digits), local_bytes the continuation stack once the programs are compiled
    virtual kernel_info kernel() const = 0;
    virtual int resident_blocks() const = 0; // the planner's one-wave floor: detail::quality_resident_blocks()
    // The AI denoiser: beauty, albedo, normal (world, unit or 0) → out, float4 per pixel, rows from the top, on stream
    // 0. Made on the first call, set up again for each new image size: one invoke for an image of at most 2048 a side,
    // else (or when that doesn't fit) tiles of 2048, 1024 or 512, the first that fits in 1024 MiB with the caller's
    // images; std::runtime_error with the sizes of the smallest when none does (denoiser.cpp).
    virtual void denoise(const float4* beauty, const float4* albedo, const float4* normal, float4* out, int width,
                         int height) = 0;
    // Not pure, these four: a test's own optix_runtime needs none of them.
    // What the last denoise computed and holds (renderer.h's denoise_report): valid false before the first
    virtual denoise_report last_denoise() const { return {}; }
    // Frees the denoiser's state and scratch, and forgets their setup (session::drop_denoised: the window's denoiser
    // turned off). The denoiser itself stays made, since making it takes a while; the next denoise sets it up again.
    virtual void release_denoiser() {}
    // GpuTests only, so that the tiled path is checked on a small image: an image wider or taller than `side` is
    // denoised in tiles of `side`, one no larger in one invoke (the memory plan still applies); 0, the default: the
    // layouts above
    virtual void set_denoise_tile_limit(int /*side*/) {}
    // Whether OptiX keeps the programs it compiled in its disk cache (start_optix.h's optix_cache_enabled asks this)
    virtual bool cache_enabled() const { return true; }

    // GpuTests only (the probes reach OptiX through the abstract runtime the fixture holds, with no upload, no build
    // and no optix_core symbol, so the build without OptiX still links and SKIPs):
    // the render module and pipeline, as the first upload would make them; kernel() then reports local_bytes
    virtual void compile_programs() = 0;
    // raygen `entry` of the module `ptx` with no scene: root 0, a zero camera, params' test fields from `test`
    virtual void launch_test(std::span<const unsigned char> ptx, const char* entry, int width, int height,
                             const optix_test_launch& test) = 0;
};
}
