// OptiX's AI denoiser (runtime_core.h's denoise): the AOV model with albedo and normal guides, made at the first call,
// its intensity and average color computed before each invoke, set up again for each new image size, large images in
// tiles, destroyed before the context (core_deleter). Compiled by cl.exe with OptiX's host API (core.h).
//
// The model. OPTIX_DENOISER_MODEL_KIND_AOV, the one built-in model for single images since OptiX 9.0 (HDR and LDR are
// deprecated aliases of it), with both guides: the albedo, and the normal in world space, the AOV model's
// (optix_types.h: "(X, Y, Z) world space, all other models"). Every image is float4, dense, rows from the top: the
// order of the renderer's own images, and the denoiser reads them as they lie. Alpha is copied (1 everywhere).
//
// The exposure. The network works on light scaled to a usual brightness: hdrIntensity, 0.18 / exp(the mean of
// log(luminance)) over the image (optixDenoiserComputeIntensity), and for the AOV model hdrAverageColor, the mean log
// of each channel. Left null, the denoiser would compute them itself for each invoke, which for a tiled image means
// per tile: tiles of one image scaled differently (optix_types.h: they "should be set ... at least if tiling is used
// to get consistent autoexposure for all tiles"). They are computed here over the whole image before every invoke,
// into a buffer of 8 floats on the GPU that every tile reads, and that last_denoise copies back only when asked
// (renderer.h's denoise_report, with the state's and the scratch's sizes and the tiles).
//
// The layout and its memory. optixDenoiserSetup prepares the state for one size, the image's or a tile's, so it runs
// again whenever the image's size changes, and only then: an image is denoised with a setup made for it, whatever was
// denoised before (the bytes of a denoise depend on the image and on its layout, never on the history). The scratch
// buffer serves the setup, the invoke, and the intensity and average color in turn, so it is the largest of their
// sizes. Before anything is allocated, the plan: the layouts in turn, one invoke for an image of at most 2048 a side,
// then tiles of 2048, 1024 and 512 (each set up with the network's overlap around it, and tried only when that setup is
// smaller than the whole image), and the first whose state, scratch and the caller's images (four float4 images and its
// widened sums) come within 1024 MiB, with 256 MiB of the GPU's memory left free beside the state and scratch, is
// taken; if none fits, a std::runtime_error with the sizes of the smallest. The layout depends on the memory free when
// an image size is first denoised, so last_denoise reports it (RayTracingGPU names the tiles). (On Windows an
// allocation the GPU can't hold may be taken from the system's memory instead of failing: the plan is what keeps the
// denoiser from that.)
#include "core.h"
static_assert(OPTIX_VERSION >= 90000); // NVIDIA's optix.h, not a project header of that name

#pragma warning(push, 0)
#include <optix_denoiser_tiling.h> // optixUtilDenoiserInvokeTiled (includes <optix.h>: no project header may shadow it)
#pragma warning(pop)

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace gpu::optix_core
{
namespace
{
// Images up to this side are denoised in one invoke, when that plan fits
constexpr unsigned int whole_side = 2048;
// The sides of the tiles tried in turn otherwise, each tile with the network's overlap of pixels around it
// (optixUtilDenoiserInvokeTiled cuts the tiles and runs them in turn)
constexpr std::array<unsigned int, 3> tile_sides{2048, 1024, 512};
// The memory plan: the most a denoise may take on the GPU, the state, the scratch and the images together, and what
// must stay free beside it
constexpr std::size_t memory_cap = std::size_t{1024} << 20;
constexpr std::size_t free_margin = std::size_t{256} << 20;
// The caller's images per pixel (renderer.cu's session): the beauty, the two guides and the output, float4, and the
// denoised image widened to 3 doubles for the display
constexpr std::size_t image_bytes_per_pixel = 4 * 4 * sizeof(float) + 3 * sizeof(double);
// A float4 pixel
constexpr unsigned int pixel_bytes = 4 * sizeof(float);

// Bytes as MiB, rounded up, for the messages
std::size_t mib(std::size_t bytes)
{
    return (bytes + (std::size_t{1} << 20) - 1) >> 20;
}

// A float4 image of the GPU's memory as the denoiser reads it: dense rows from the top
OptixImage2D image_of(const void* data, unsigned int width, unsigned int height)
{
    OptixImage2D image = {};
    image.data = reinterpret_cast<CUdeviceptr>(data);
    image.width = width;
    image.height = height;
    image.rowStrideInBytes = width * pixel_bytes;
    image.pixelStrideInBytes = 0; // dense: 16 bytes, the format's
    image.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    return image;
}
}

// The denoiser and what it keeps between calls: its handle, the GPU's buffer of the intensity and average color, and
// the setup of the last image: its size, its layout, its state and scratch
struct denoiser_state
{
    OptixDenoiser denoiser = nullptr;
    device_array<float> inputs{8}; // [0] hdrIntensity, [4, 7) hdrAverageColor (16 bytes in, as a float4 would be)
    // The image optixDenoiserSetup was last called for; all 0 until a setup succeeds, and after forget_setup
    unsigned int image_width = 0;
    unsigned int image_height = 0;
    unsigned int limit = 0;   // the tile limit it was planned under (core::denoise_tile_limit)
    unsigned int tile = 0;    // 0: one invoke; else tiles of this side
    unsigned int overlap = 0; // tiles: the pixels of overlap on each side
    device_array<unsigned char> state{0};
    device_array<unsigned char> scratch{0};
    bool denoised = false; // the last call queued its invoke: `inputs` holds its intensity and average color

    denoiser_state() = default;
    denoiser_state(const denoiser_state&) = delete;
    denoiser_state& operator=(const denoiser_state&) = delete;
    // The GPU is done with it: core_deleter synchronizes first, and the denoiser goes before the context
    ~denoiser_state()
    {
        if (denoiser)
        {
            optixDenoiserDestroy(denoiser);
        }
    }

    // Whether the setup is the one for a width × height image under this tile limit
    bool serves(unsigned int width, unsigned int height, unsigned int tile_limit) const
    {
        return image_width > 0 && image_width == width && image_height == height && limit == tile_limit;
    }

    // No setup: the state and scratch freed (cudaFree waits for the work that may still read them)
    void forget_setup()
    {
        image_width = image_height = limit = tile = overlap = 0;
        state = device_array<unsigned char>(0);
        scratch = device_array<unsigned char>(0);
    }
};

void denoiser_deleter::operator()(denoiser_state* state) const noexcept
{
    delete state;
}

namespace
{
// The denoiser, made at the first denoise: the AOV model with both guides, alpha copied
denoiser_state& denoiser_of(core& c)
{
    if (!c.denoiser)
    {
        std::unique_ptr<denoiser_state, denoiser_deleter> made(new denoiser_state);
        OptixDenoiserOptions options = {};
        options.guideAlbedo = 1;
        options.guideNormal = 1;
        options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
        check_optix(optixDenoiserCreate(c.context, OPTIX_DENOISER_MODEL_KIND_AOV, &options, &made->denoiser),
                    "creating OptiX's denoiser");
        c.denoiser = std::move(made);
    }
    return *c.denoiser;
}

// A layout and what it needs: tile 0 for one invoke over a setup_width × setup_height image, else tiles of that side
// set up with their overlap; the state's and the scratch's bytes
struct plan
{
    unsigned int tile = 0;
    unsigned int setup_width = 0;
    unsigned int setup_height = 0;
    unsigned int overlap = 0;
    std::size_t state_bytes = 0;
    std::size_t scratch_bytes = 0;

    std::size_t own_bytes() const { return state_bytes + scratch_bytes; }
};

// The layout's sizes (optixDenoiserComputeMemoryResources: host arithmetic, no GPU work), for one invoke over a
// width × height image (tile 0) or for tiles of that side
plan plan_for(OptixDenoiser denoiser, unsigned int width, unsigned int height, unsigned int tile)
{
    plan p;
    p.tile = tile;
    OptixDenoiserSizes sizes = {};
    if (tile == 0)
    {
        p.setup_width = width;
        p.setup_height = height;
        check_optix(optixDenoiserComputeMemoryResources(denoiser, width, height, &sizes),
                    "computing the memory of OptiX's denoiser");
        p.scratch_bytes = sizes.withoutOverlapScratchSizeInBytes;
    }
    else
    {
        // The sizes of a tile without its overlap, its setup with the overlap on all four sides (the guide's rule)
        check_optix(optixDenoiserComputeMemoryResources(denoiser, tile, tile, &sizes),
                    "computing the memory of OptiX's denoiser");
        p.overlap = sizes.overlapWindowSizeInPixels;
        p.setup_width = tile + 2 * p.overlap;
        p.setup_height = tile + 2 * p.overlap;
        p.scratch_bytes = sizes.withOverlapScratchSizeInBytes;
    }
    p.state_bytes = sizes.stateSizeInBytes;
    // One scratch buffer serves the invoke and, before it, the intensity and the average color: the largest of three
    p.scratch_bytes =
        std::max({p.scratch_bytes, sizes.computeIntensitySizeInBytes, sizes.computeAverageColorSizeInBytes});
    return p;
}

// The layouts a width × height image may be denoised in, in the order the plan tries them: one invoke for an image of
// at most whole_side a side, then the tiles whose setup is smaller than the whole image (a larger one would only cost
// more), and one invoke after all when there is none of either (an image too long and thin for any tile). Under
// GpuTests' limit: one invoke for an image no larger than the limit, else tiles of the limit's side.
std::vector<plan> layouts_for(OptixDenoiser denoiser, unsigned int width, unsigned int height, unsigned int limit)
{
    if (limit > 0)
    {
        const bool whole = width <= limit && height <= limit;
        return {plan_for(denoiser, width, height, whole ? 0 : limit)};
    }
    std::vector<plan> layouts;
    if (width <= whole_side && height <= whole_side)
    {
        layouts.push_back(plan_for(denoiser, width, height, 0));
    }
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    for (const unsigned int side : tile_sides)
    {
        const plan p = plan_for(denoiser, width, height, side);
        if (static_cast<std::size_t>(p.setup_width) * p.setup_height < pixels)
        {
            layouts.push_back(p);
        }
    }
    if (layouts.empty())
    {
        layouts.push_back(plan_for(denoiser, width, height, 0));
    }
    return layouts;
}

// A new setup for a width × height image, by the memory plan (see the top of this file). The old state and scratch go
// first, so that their memory counts as free and the two are never held at once; a setup that is refused or fails
// leaves none.
void set_up(denoiser_state& d, unsigned int width, unsigned int height, unsigned int limit)
{
    d.forget_setup();
    const std::size_t images = static_cast<std::size_t>(width) * height * image_bytes_per_pixel;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "reading how much of the GPU's memory is free");
    const std::vector<plan> layouts = layouts_for(d.denoiser, width, height, limit);
    const plan* chosen = nullptr;
    const plan* smallest = &layouts.front();
    for (const plan& p : layouts)
    {
        // The caller's images are in the GPU's memory already: the free memory has them counted
        if (p.own_bytes() + images <= memory_cap && p.own_bytes() + free_margin <= free_bytes)
        {
            chosen = &p;
            break;
        }
        if (p.own_bytes() < smallest->own_bytes())
        {
            smallest = &p;
        }
    }
    if (!chosen)
    {
        const plan& p = *smallest;
        throw std::runtime_error(std::format(
            "OptiX's denoiser can't take a {}×{} image: its smallest plan, {}, needs its state ({} MiB), scratch ({} "
            "MiB) and the images ({} MiB), {} MiB of the {} MiB it may take; {} MiB of the GPU's memory are free, and "
            "{} MiB must stay so",
            width, height, p.tile == 0 ? std::string("one invoke") : std::format("tiles of {}", p.tile),
            mib(p.state_bytes), mib(p.scratch_bytes), mib(images), mib(p.own_bytes() + images), mib(memory_cap),
            free_bytes >> 20, mib(free_margin)));
    }
    d.state = device_array<unsigned char>(chosen->state_bytes);
    d.scratch = device_array<unsigned char>(chosen->scratch_bytes);
    check_optix(optixDenoiserSetup(d.denoiser, nullptr, chosen->setup_width, chosen->setup_height,
                                   reinterpret_cast<CUdeviceptr>(d.state.data()), d.state.size(),
                                   reinterpret_cast<CUdeviceptr>(d.scratch.data()), d.scratch.size()),
                "setting up OptiX's denoiser");
    d.image_width = width;
    d.image_height = height;
    d.limit = limit;
    d.tile = chosen->tile;
    d.overlap = chosen->overlap;
}
}

void denoise(core& c, const void* beauty, const void* albedo, const void* normal, void* out, int width, int height)
{
    if (width <= 0 || height <= 0 || !beauty || !albedo || !normal || !out)
    {
        throw std::invalid_argument("OptiX's denoiser takes four images of at least 1×1 pixels");
    }
    const auto w = static_cast<unsigned int>(width);
    const auto h = static_cast<unsigned int>(height);
    denoiser_state& d = denoiser_of(c);
    d.denoised = false;
    // A setup made for this very size (and tile limit) is kept; any other size is set up anew
    if (!d.serves(w, h, c.denoise_tile_limit))
    {
        set_up(d, w, h, c.denoise_tile_limit);
    }
    const auto scratch = reinterpret_cast<CUdeviceptr>(d.scratch.data());
    const auto state = reinterpret_cast<CUdeviceptr>(d.state.data());

    // The exposure of the whole image, before the invoke reads it (every call on stream 0, in order)
    const OptixImage2D input = image_of(beauty, w, h);
    const auto intensity = reinterpret_cast<CUdeviceptr>(d.inputs.data());
    const CUdeviceptr average_color = intensity + 4 * sizeof(float);
    check_optix(optixDenoiserComputeIntensity(d.denoiser, nullptr, &input, intensity, scratch, d.scratch.size()),
                "computing the image's intensity for OptiX's denoiser");
    check_optix(
        optixDenoiserComputeAverageColor(d.denoiser, nullptr, &input, average_color, scratch, d.scratch.size()),
        "computing the image's average color for OptiX's denoiser");

    OptixDenoiserParams params = {};
    params.hdrIntensity = intensity;
    params.blendFactor = 0.0f; // the denoised image alone, none of the input blended back
    params.hdrAverageColor = average_color;
    params.temporalModeUsePreviousLayers = 0;
    OptixDenoiserGuideLayer guides = {};
    guides.albedo = image_of(albedo, w, h);
    guides.normal = image_of(normal, w, h);
    OptixDenoiserLayer layer = {};
    layer.input = input;
    layer.output = image_of(out, w, h);
    layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;
    if (d.tile == 0)
    {
        check_optix(optixDenoiserInvoke(d.denoiser, nullptr, &params, state, d.state.size(), &guides, &layer, 1, 0, 0,
                                        scratch, d.scratch.size()),
                    "running OptiX's denoiser");
    }
    else
    {
        check_optix(optixUtilDenoiserInvokeTiled(d.denoiser, nullptr, &params, state, d.state.size(), &guides, &layer,
                                                 1, scratch, d.scratch.size(), d.overlap, d.tile, d.tile),
                    "running OptiX's denoiser in tiles");
    }
    // A failure the GPU reports later shows when the host waits for the result (the display's copy)
    check(cudaGetLastError(), "running OptiX's denoiser");
    d.denoised = true;
}

denoise_report last_denoise(const core& c)
{
    denoise_report report;
    if (!c.denoiser || !c.denoiser->denoised)
    {
        return report; // valid false: no denoise has completed
    }
    const denoiser_state& d = *c.denoiser;
    // The intensity and average color the last invoke read, from the GPU: the copy waits for the work queued on
    // stream 0 before it, so a report costs a wait only when it is asked for
    const std::vector<float> inputs = d.inputs.to_host();
    report.intensity = inputs[0];
    report.average_color[0] = inputs[4];
    report.average_color[1] = inputs[5];
    report.average_color[2] = inputs[6];
    report.state_bytes = d.state.size();
    report.scratch_bytes = d.scratch.size();
    report.tile = static_cast<int>(d.tile); // the setup is the last image's: its layout
    report.valid = true;
    return report;
}

void release_denoiser(core& c)
{
    if (c.denoiser)
    {
        c.denoiser->denoised = false;
        c.denoiser->forget_setup();
    }
}

void set_denoise_tile_limit(core& c, int side)
{
    if (side < 0)
    {
        throw std::invalid_argument("a tile limit is a side of at least 1 pixel, or 0 for none");
    }
    c.denoise_tile_limit = static_cast<unsigned int>(side); // the next denoise is planned again (serves compares it)
}
}
