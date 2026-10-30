// The quality kernel (see quality.cuh): the third kernel, for light and materials beyond the books. It reads the books'
// tables as the kernel for all three books does (scene_view), and its own beside them (quality_view.cuh), and traces
// its paths with quality_path.cuh's integrator. A translation unit of its own, as display.cu is: nvcc compiles
// renderer.cu, and the books' two kernels in it, as if this file didn't exist. Float only: the books' kernels compute
// in double as a check of their floats; the quality kernel's checks are its own (GpuTests, and the comparisons with
// the books' images).
#include "quality.cuh"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "device_memory.h"
#include "environment_on_device.cuh"
#include "lamps_on_device.cuh"
#include "microfacets_on_device.cuh"
#include "quality_path.cuh"
#include "quality_view.cuh"
#include "tracer.cuh"

namespace gpu::detail
{
// (Not in an unnamed namespace: for a kernel that isn't a template, nvcc's generated host code would then declare its
// launch function extern and define it static, which MSVC warns about at /W4. See display.cu.)

// The books' blocks of 16×8 pixels (block_width and block_height in renderer.cu, which keeps its own)
constexpr int quality_block_width = 16;
constexpr int quality_block_height = 8;

// renderer.cu's without_nan and warp_sum, for this file's kernel: a NaN channel of a sample counts as 0, and a warp
// adds up its threads' counts before one atomic addition
__device__ inline double quality_without_nan(float x)
{
    return static_cast<double>(x == x ? x : 0.0f);
}

__device__ inline unsigned long long quality_warp_sum(unsigned long long value)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// One thread per pixel, as render_pixel in renderer.cu: the launch's passes of the pixel in one loop, one ray per
// iteration, the next sample starting as soon as one ends, the sums in double read and written once per launch.
// counts[0] gets the rays traced (closest-hit searches, as the books count them), counts[1] the shadow rays.
__global__ void __launch_bounds__(quality_block_width * quality_block_height)
    quality_kernel(const camera<float> cam, const quality_view view, const std::uint64_t seed, const int first_pass,
                   const int pass_count, const int first_row, const int row_count, double* __restrict__ sums,
                   unsigned long long* __restrict__ counts)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int j = first_row + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    unsigned long long rays = 0;
    unsigned int shadow_rays = 0;
    // Threads beyond the image's right edge or the launch's last row don't return early: they take part in the
    // warp sums below
    if (i < cam.image_width && j < first_row + row_count)
    {
        const std::uint64_t pixel =
            static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(cam.image_width) + static_cast<std::uint64_t>(i);
        double* const pixel_sum = sums + 3 * pixel;
        double red = pixel_sum[0];
        double green = pixel_sum[1];
        double blue = pixel_sum[2];

        const bvh_tracer tracer{view};
        int pass = first_pass;
        const int end_pass = first_pass + pass_count;
        pcg32 rng = sample_generator(seed, pixel, static_cast<std::uint64_t>(pass));
        quality_path q = start_quality_sample(cam, view, i, j, pass, rng);
        while (true)
        {
            ++rays;
            color<float> sample;
            if (!trace_quality_ray(q, tracer, view, cam, rng, sample, shadow_rays))
            {
                continue; // the path goes on
            }
            red += quality_without_nan(sample.x);
            green += quality_without_nan(sample.y);
            blue += quality_without_nan(sample.z);
            if (++pass == end_pass)
            {
                break;
            }
            rng = sample_generator(seed, pixel, static_cast<std::uint64_t>(pass));
            q = start_quality_sample(cam, view, i, j, pass, rng);
        }
        pixel_sum[0] = red;
        pixel_sum[1] = green;
        pixel_sum[2] = blue;
    }

    rays = quality_warp_sum(rays);
    const unsigned long long shadow = quality_warp_sum(shadow_rays);
    if ((threadIdx.y * blockDim.x + threadIdx.x) % warpSize == 0)
    {
        atomicAdd(&counts[0], rays);
        atomicAdd(&counts[1], shadow);
    }
}

// The quality kernel's own arrays on the GPU, beside the books' tables (device_scene in renderer.cu): the materials'
// GGX side entries, the lamps, the environment map. Each part is a class in a header of its own, which the feature that
// needs it fills; this class only puts them together.
class quality_scene
{
public:
    quality_scene(const scene_desc& scene, std::span<const int> sphere_ids, std::span<const int> quad_ids)
        : microfacets_(scene),
          lamps_(scene, sphere_ids, quad_ids),
          environment_(scene),
          combination_(scene.quality.combination)
    {
        // The chance of picking each light of the set, the lamps and the environment alike, computed in double
        const int lights = lamps_.count() + (environment_.sampled() ? 1 : 0);
        light_pmf_ = lights > 0 ? static_cast<float>(1.0 / lights) : 0.0f;
    }

    quality_view view(const scene_view<float>& base) const
    {
        return {.base = base,
                .microfacets = microfacets_.data(),
                .lamps = lamps_.view(),
                .environment = environment_.view(),
                .light_pmf = light_pmf_,
                .combination = combination_};
    }

private:
    microfacets_on_device microfacets_;
    lamps_on_device lamps_;
    environment_on_device environment_;
    light_combination combination_;
    float light_pmf_ = 0;
};

void quality_scene_deleter::operator()(quality_scene* scene) const noexcept
{
    delete scene;
}

quality_scene_ptr upload_quality(const scene_desc& scene, std::span<const int> sphere_ids,
                                 std::span<const int> quad_ids)
{
    return quality_scene_ptr(new quality_scene(scene, sphere_ids, quad_ids));
}

void launch_quality(const quality_scene& q, const camera<float>& cam, const scene_view<float>& base,
                    std::uint64_t seed, int first_pass, int pass_count, int first_row, int row_count, double* sums,
                    unsigned long long* counts)
{
    const dim3 block(quality_block_width, quality_block_height);
    const dim3 grid(static_cast<unsigned>((cam.image_width + quality_block_width - 1) / quality_block_width),
                    static_cast<unsigned>((row_count + quality_block_height - 1) / quality_block_height));
    quality_kernel<<<grid, block>>>(cam, q.view(base), seed, first_pass, pass_count, first_row, row_count, sums,
                                    counts);
    // A launch returns no error code: one that couldn't start shows here, one that fails while it runs when the host
    // waits for it (see launch_kernel in renderer.cu)
    check(cudaGetLastError(), "launching the quality kernel");
}

namespace
{
int quality_blocks_per_multiprocessor()
{
    int blocks = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, quality_kernel,
                                                        quality_block_width * quality_block_height, 0),
          "computing the quality kernel's occupancy");
    return blocks;
}
}

kernel_info quality_kernel_info()
{
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, quality_kernel), "reading the quality kernel's attributes");
    int max_threads = 0;
    check(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, 0),
          "reading the CUDA device's properties");
    return {.name = "quality",
            .registers = attributes.numRegs,
            .local_bytes = static_cast<int>(attributes.localSizeBytes),
            .blocks_per_multiprocessor = quality_blocks_per_multiprocessor(),
            .threads_per_block = quality_block_width * quality_block_height,
            .max_threads_per_multiprocessor = max_threads};
}

int quality_resident_blocks()
{
    int multiprocessors = 0;
    check(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0),
          "reading the CUDA device's properties");
    return multiprocessors * quality_blocks_per_multiprocessor();
}

// quality_access.cuh: the view quality_kernel takes, for OptiX's programs and the denoiser's guides (host code only)
quality_view quality_view_of(const quality_scene& q, const scene_view<float>& base) { return q.view(base); }
}
