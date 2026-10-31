// Suites guides, denoise and denoise-eval (Step 94): OptiX's AI denoiser for RayTracingGPU's images, tested against its
// specification: every reference here is the test's own, written from the interfaces (the guides' rules guides.cuh
// states, denoise.cuh, renderer.h's session::display_denoised, optix_runtime.cuh) rather than from the code behind
// them.
//
// guides: the albedo and normal guides, by their rules per surface, 16 samples per pixel (launch_guides in
// denoise.cuh, OptiX's optix_scene::guides), on scenes made here, 8 × 8 pixels (the corridors 4 × 4), put on the GPU
// by test_scene.h:
//   G1  matte: a sphere filling the view gives its color c at every pixel, exactly, both through a quality scene's view
//       and through the books' kernels' view without one (quality = nullptr); a quad facing the camera gives c and its
//       normal (0, 0, 1), exactly;
//   G2  the surfaces that stop a sample: a books' metal of fuzz 0.5 and a GGX conductor of roughness 0.5 give their
//       albedo (F0), a gloss coat the matte color under it (the coat ignored), a medium's particle its albedo, each
//       exactly, with a unit normal facing the camera (the particle's −unit(d));
//   G3  the surfaces a sample follows: a mirror sphere of albedo a (the books' metal, fuzz 0) in front of a matte quad
//       of color c gives a ⊙ c exactly and the quad's normal; the smooth conductor (r = 0) and a conductor of r = 0.05
//       (below 0.1) in the mirror's place give F0 ⊙ c within 10⁻⁵ (Schlick's F below 8° of incidence) and 10⁻² (a GGX
//       sample's weight), and the quad's normal;
//   G4  a clear glass ball inside a closed matte box of color c: c exactly at every pixel;
//   G5  a lamp: c / max(1, max channel of c) exactly, (4, 2, 1) → (1, 0.5, 0.25) and (0.5, 0.25, 0.125) itself, and
//       its normal; an escaped ray: clamp(sky, 0, 1) exactly, the sky (2, 0.5, 0) → (1, 0.5, 0), and the normal 0;
//   G6  the 8 delta vertices: a corridor of two parallel mirrors (albedo a) closed by a matte wall (color c): 7
//       mirrors before the wall give a⁷ ⊙ c exactly and the wall's normal, 12 give the albedo 0 and the normal 0;
//   G7  over G1-G6: every normal unit (within 10⁻⁵) or exactly 0, every albedo channel in [0, 1];
//   G8  G1-G7 through OptiX: the same scenes uploaded by optix_runtime::upload, their guides by optix_scene::guides;
//   G9  the CUDA and OptiX guides of cornell (-p material: the books' kernel's view) and final (-p mis: the quality
//       kernel's) through gpu::session, without and with optix_tracer, agree within 10⁻³ in every channel on at least
//       99.9% of the pixels; their normals unit or 0, their albedos in [0, 1]; the planted fault, OptiX's albedo
//       × 1.01, must not agree. OptiX's guides of cornell-triangles (the same box as triangles) against CUDA's of
//       cornell agree within 10⁻³ on at least 99% of the pixels, a threshold derived from the edges' geometry before
//       any run (see the check), its normals unit or 0 and its albedos in [0, 1].
// denoise:
//   W1  to_sums widens each float to its double exactly; the books' and the three tone maps' bytes (at 0 and 1.5 EV) of
//       the widened image at scale 1 equal those of the original sums at scale 1, and of four times them at scale 1/4
//       (display()'s case: the sums of 4 passes);
//   B1  to_beauty: float(sum · scale), NaN → 0, clamped to [0, 10⁴], alpha 1, in two bands with scales of their own,
//       whichever band is written first;
//   S1  a session's display_denoised (renderer.h): the beauty it gives the denoiser is the image's floats, clamped, in
//       display()'s two bands; its books bytes are to_rgba8's of the denoised floats as doubles at scale 1, its agx
//       bytes at +1 EV to_rgba8_toned's with float(2¹); a second display of the same state re-tones the kept floats
//       without denoising again, and so does retone_denoised after more passes, with the bytes display_denoised gave
//       for each display; a new pass denoises again. A new view (set_camera): no denoised floats or re-tone until its
//       denoise, then guides recomputed, a fresh session's for that view bit for bit; a larger view grows the buffers;
//       in an OptiX session (where OptiX runs) the guides launched once per view, not again for a second denoise.
//       float64 is refused (std::invalid_argument), a session without a runtime too (std::logic_error) until
//       attach_optix gives it one (the image kept; the same runtime again is nothing, another a std::logic_error);
//       drop_denoised leaves nothing to re-tone, asks the runtime to release its denoiser, and the next display
//       denoises again. render() with denoise leaves the pixels and the rays bit for bit as without it, its denoised
//       floats those display_denoised gives; a denoise that fails after the passes leaves the pixels too, nothing
//       denoised and the reason in denoise_error; an image too large for the denoiser's images is refused before
//       anything is built;
//   D1  the first denoise of the process: its seconds (the denoiser's creation, which re-prices denoise-eval) and the
//       GPU memory the runtime took for 480 × 270 and 960 × 540, INFO, and the state, scratch and tiles
//       optix_runtime::last_denoise reports, INFO (the report must be valid); 1920 × 1080 extrapolated, INFO;
//   D2  the synthetic case: a 256 × 256 smooth gradient plus Gaussian noise of σ 0.1, albedo 1, normal (0, 0, 1): the
//       MSE at least 4 times lower, the mean of each channel within 1%, every value finite, alpha copied; planted: the
//       criterion must reject the noisy image itself and the result made 3% brighter (a check of its thresholds, which
//       the denoiser never sees); two invokes of one input compared bytewise, INFO;
//   D3  the tiled path: a 512 × 512 image in one invoke, then in tiles of 128 (the runtime's tile limit for tests:
//       last_denoise must report each layout; with the network's overlap on each side a tile's window is still smaller
//       than the image, so the tiles see crops of it), the mean and the largest |Δ| between the two, INFO, each layout's
//       state and scratch, INFO; the tiles' MSE gain checked as D2's.
// denoise-eval: the renders of the denoiser's evaluation (scripts/verify/denoise.ps1), left out of the every-step run.
// In one process (one denoiser creation), through gpu::render with denoise = true (the library path of RayTracingGPU's
// --denoise), named as RayTracingGPU names its outputs, into the folder --write gives; PfmStats error then reads them:
//   E1  N16 at seeds 1 and 2, 480 × 270: cornell, studio and final -p mis (CUDA), cornell-bust -p mis (OptiX):
//       <scene>-N<seed>.pfm and <scene>-N<seed>-denoised.pfm, and the intensity and average color the denoiser
//       computed for each (last_denoise), INFO;
//   E2  cornell-bust's references, -s 2048 (45² = 2025 passes, as RayTracingGPU's) under OptiX at seeds 101 and 102:
//       cornell-bust-mis2048-s101.pfm and -s102.pfm (the other scenes' are references.ps1's, rendered once by Step 85's
//       quality kernel);
//   E3  -s 1024 at seed 1 on cornell and studio, denoised: <scene>-1024spp-N1.pfm and -denoised.pfm (where denoising
//       stops helping, INFO in the protocol);
//   E4  cornell's N16 at seed 1 rendered and denoised a second time: cornell-N1-again.pfm and
//       cornell-N1-again-denoised.pfm, and its pixels and denoised floats against the first's, bytewise, INFO (the
//       denoiser's determinism inside a process, which denoise.ps1 decides on from the files).
//
// Decisions where the specification leaves a choice:
//  1. A session's guides are seen through guide_tap, an optix_runtime the test gives the session (session_options::
//     optix, or attach_optix): renderer.cu reaches OptiX only through that abstract class (optix_runtime.cuh), and
//     display_denoised hands the beauty and the guides to its denoise. The tap keeps them and returns the beauty as the
//     "denoised" image (the denoiser itself is D1-D3's), counts the release_denoiser calls, and passes every other call
//     to the process's runtime, when it has one; the scenes that runtime uploads for it come back wrapped, so that
//     their guide launches are counted (OptiX sessions only: a CUDA session launches its guide kernel itself, and S1
//     sees its guides recomputed by their contents instead). So G9 compares the very guides a session computes and S1
//     the very beauty, and S1 runs in a build without OptiX but for its count of guide launches. The tap is never
//     given to optix_cache_enabled(), which casts its argument to the runtime start_optix() makes.
//  2. launch_guides and optix_scene::guides get camera_for_guides of the view's camera: it is idempotent, so the test
//     holds whichever side of the interface applies it.
//  3. Exact values: every color is a dyadic fraction of a few bits (0.75, 0.5, 0.25...), so every product the rules
//     take and the average of 16 equal samples are exact in float, in any order; a constant sky of powers of two (and
//     0) comes out of the books' (1 − a)·s + a·s exactly, contracted into fmas or not.
//  4. The guides' "at most 8 delta vertices" is checked where it can't be read two ways: 7 mirrors are followed, 12
//     give 0 (with the camera's max_depth 50, a path cut by max_depth instead would show a¹² ⊙ c). What the 8th mirror
//     itself gives is the implementation's choice.
//  5. A camera is no surface: the mirror's reflections reach the quad behind it. The views stay within 0.05 of −z, so
//     every reflection lands on that quad and every sample of a pixel meets the same surfaces. Every surface whose
//     normal is checked exactly is met on its front (u × v toward the ray), where the outward normal and the one
//     facing the ray are the same: the test doesn't decide which of the two the guides' "shading normal" is.
//  6. D1: the memory is the device's free memory (cudaMemGetInfo) before and after the first denoise at each size,
//     other programs' allocations in between included, beside the state and scratch optix_runtime::last_denoise
//     reports. 1920 × 1080 isn't run: its four float4 images alone (127 MiB) exceed the suite's 64 MB, so its memory
//     is extrapolated linearly in the pixels from the two sizes run, and nothing of that size is allocated.
//  7. D3: the runtime's own tiles are 512 a side and more, for images of thousands of pixels, so the tiled path is
//     reached on a small image through its tile limit for tests (optix_runtime::set_denoise_tile_limit), set for D3's
//     second denoise only and taken off however the check ends, since the runtime is the process's. Both denoises
//     compute the intensity and average color over the whole image, so the difference is the tiles' alone.
//  8. denoise-eval runs only with --write (its folder) and never under --quick: it SKIPs otherwise, so that GpuTests
//     all --quick (CTest) stays small. Its references are rendered without denoise (nothing reads a denoised
//     reference). The seconds of the references and of the 1024-sample renders are left out of budget-seconds
//     (exclude_seconds, whose INFO line says how many): they are the protocol's, budgeted with it; the N16 renders and
//     their denoises count. No timing is printed but the denoiser's creation (D1), which tells what the protocol costs.
//  9. Scenes by name come from host::built_scene (scene_services.cpp), as RayTracingGPU builds them for -p, -w,
//     --height and -s. Every scene the evaluation names exists: one that can't be built (cornell-bust without its
//     model's files) stops the run with the error, never a SKIP that would let denoise.ps1 miss its files. So does a
//     render whose denoise failed (render_result::denoise_error, a failed check), before anything of it is written.
// 10. Seeds (literals, never retuned; this file's 9200-9299): 9200 G1-G8, 9201 G9's sessions, 9202 W1's floats, 9203
//     B1's sums, 9204 S1's renders, 9205 D2's noise, 9206 D3's noise. denoise-eval's are the protocol's (1, 2, 101,
//     102), as RayTracingGPU is given them.
//
// The evaluation's criteria (relMSE'(D16) ≤ relMSE'(N16)/4, and the mean within 2% of the reference's in every
// channel, per scene and seed) are PfmStats error's, run by denoise.ps1 on what denoise-eval writes.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "denoise.cuh"
#include "display.cuh"
#include "guides.cuh"
#include "optix_runtime.cuh"
#include "quality.cuh"
#include "renderer.h"
#include "tonemap.cuh"

#include "gpu_test.cuh"
#include "gpu_test.h"
#include "host_services.h"
#include "optix_fixture.h"
#include "test_scene.h"

namespace
{
constexpr std::uint64_t table_seed = 9200;      // G1-G8: the guides' seed
constexpr std::uint64_t session_seed = 9201;    // G9's sessions
constexpr std::uint64_t widen_seed = 9202;      // W1's floats
constexpr std::uint64_t beauty_seed = 9203;     // B1's sums
constexpr std::uint64_t display_seed = 9204;    // S1's renders
constexpr std::uint64_t synthetic_seed = 9205;  // D2's noise
constexpr std::uint64_t tiled_seed = 9206;      // D3's noise

std::size_t count_of(int width, int height)
{
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

// A registered scene as RayTracingGPU builds it for `NAME -p mis|material -w WIDTH --height HEIGHT -s SAMPLES` (no
// strata): render() with passes = 0 then renders RayTracingGPU's passes, the camera's sqrt_spp². std::runtime_error
// with gpu_scene's message where it can't build the scene.
gpu::scene_desc scene_named(std::string_view name, gpu_test::host::scene_strategy strategy, int width, int height,
                            int samples)
{
    return gpu_test::host::built_scene(
        name, {.width = width, .height = height, .samples = samples, .sampling = strategy});
}

// 1 when f throws an exception of type E, 0 when it throws another one or none
template <class E, class F>
double throws(const F& f)
{
    try
    {
        f();
    }
    catch (const E&)
    {
        return 1;
    }
    catch (...)
    {
        return 0;
    }
    return 0;
}

std::vector<float4> from_device(const float4* data, std::size_t count)
{
    std::vector<float4> values(count);
    gpu_test::cuda_check(cudaMemcpy(values.data(), data, count * sizeof(float4), cudaMemcpyDeviceToHost),
                         "copying an image of the denoiser's back");
    return values;
}

// The positions where two arrays differ bit for bit (memcmp: −0 against 0 or two NaNs count), and every position of
// the longer one beyond the shorter
template <class T>
std::int64_t bit_differences(std::span<const T> a, std::span<const T> b)
{
    const std::size_t common = std::min(a.size(), b.size());
    std::int64_t differ = static_cast<std::int64_t>(std::max(a.size(), b.size()) - common);
    for (std::size_t i = 0; i < common; ++i)
    {
        differ += std::memcmp(&a[i], &b[i], sizeof(T)) == 0 ? 0 : 1;
    }
    return differ;
}

template <class T>
std::int64_t bit_differences(const std::vector<T>& a, const std::vector<T>& b)
{
    return bit_differences(std::span<const T>(a), std::span<const T>(b));
}

// The largest |a − b| over the positions both arrays have (a NaN in either: infinity)
template <class T>
double largest_difference(const std::vector<T>& a, const std::vector<T>& b)
{
    double largest = 0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    {
        const double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        largest = d == d ? std::max(largest, d) : std::numeric_limits<double>::infinity();
    }
    return largest;
}

// B1 and S1's beauty, as denoise.cuh's to_beauty states it: float(sum · scale), NaN → 0, clamped to [0, 10⁴]
float beauty_of(double sum, double scale)
{
    const double v = sum * scale;
    if (v != v)
    {
        return 0.0f;
    }
    return std::clamp(static_cast<float>(v), 0.0f, 1e4f);
}

// --- guide_tap: a session's beauty and guides, as its denoise receives them (decision 1) -----------------------------

// The images of one denoise call, copied back, rows from the top
struct denoise_call
{
    int width = 0;
    int height = 0;
    std::vector<float4> beauty;
    std::vector<float4> albedo;
    std::vector<float4> normal;
};

// An OptiX scene that counts its guide launches (optix_scene::guides) and passes every call on to the scene it wraps
class counting_scene final : public gpu::detail::optix_scene
{
public:
    counting_scene(std::unique_ptr<gpu::detail::optix_scene> inner, std::shared_ptr<int> guide_launches)
        : inner_(std::move(inner)), guide_launches_(std::move(guide_launches))
    {
    }

    void prepare_view(int width, int height) override { inner_->prepare_view(width, height); }
    void launch(const gpu::camera<float>& cam, std::uint64_t seed, int first_pass, int pass_count, int first_row,
                int row_count, double* sums, unsigned long long* counts) override
    {
        inner_->launch(cam, seed, first_pass, pass_count, first_row, row_count, sums, counts);
    }
    void guides(const gpu::camera<float>& cam, std::uint64_t seed, float4* albedo, float4* normal) override
    {
        ++*guide_launches_;
        inner_->guides(cam, seed, albedo, normal);
    }
    double pick(const std::array<double, 3>& origin, const std::array<double, 3>& direction) override
    {
        return inner_->pick(origin, direction);
    }
    void launch_test(std::span<const unsigned char> ptx, const char* entry, int width, int height,
                     const gpu::camera<float>& cam, const gpu::optix_test_launch& test) override
    {
        inner_->launch_test(ptx, entry, width, height, cam, test);
    }

private:
    std::unique_ptr<gpu::detail::optix_scene> inner_;
    std::shared_ptr<int> guide_launches_;
};

class guide_tap final : public gpu::optix_runtime
{
public:
    // inner: the process's runtime, for an OptiX session; nullptr for a CUDA session, which only denoises.
    // fail_with: when not empty, every denoise throws std::runtime_error with it (a denoise failing after the passes)
    explicit guide_tap(std::shared_ptr<gpu::optix_runtime> inner, std::string fail_with = {})
        : inner_(std::move(inner)), fail_with_(std::move(fail_with))
    {
    }

    // The process's runtime uploads the scene; the tap counts the guide launches of what it built
    std::unique_ptr<gpu::detail::optix_scene> upload(const gpu::scene_desc& scene, const gpu::scene_view<float>& base,
                                                     const gpu::detail::quality_scene& quality,
                                                     std::span<const int> sphere_ids, std::span<const int> quad_ids,
                                                     const gpu::optix_upload_options& options) override
    {
        return std::make_unique<counting_scene>(inner().upload(scene, base, quality, sphere_ids, quad_ids, options),
                                                guide_launches_);
    }

    gpu::kernel_info kernel() const override { return inner().kernel(); }
    int resident_blocks() const override { return inner().resident_blocks(); }

    // Keeps the three inputs and returns the beauty unchanged: a copy on stream 0, after the work queued before it
    void denoise(const float4* beauty, const float4* albedo, const float4* normal, float4* out, int width,
                 int height) override
    {
        if (!fail_with_.empty())
        {
            throw std::runtime_error(fail_with_);
        }
        const std::size_t pixels = count_of(width, height);
        last_ = {.width = width,
                 .height = height,
                 .beauty = from_device(beauty, pixels),
                 .albedo = from_device(albedo, pixels),
                 .normal = from_device(normal, pixels)};
        gpu_test::cuda_check(cudaMemcpy(out, beauty, pixels * sizeof(float4), cudaMemcpyDeviceToDevice),
                             "copying the beauty into the denoiser's output");
        ++calls_;
    }

    // Counted, and passed on to the process's runtime when there is one
    void release_denoiser() override
    {
        ++releases_;
        if (inner_)
        {
            inner_->release_denoiser();
        }
    }

    void compile_programs() override { inner().compile_programs(); }
    void launch_test(std::span<const unsigned char> ptx, const char* entry, int width, int height,
                     const gpu::optix_test_launch& test) override
    {
        inner().launch_test(ptx, entry, width, height, test);
    }

    const denoise_call& last() const { return last_; }
    int calls() const { return calls_; }
    int releases() const { return releases_; }
    int guide_launches() const { return *guide_launches_; } // OptiX sessions only: a CUDA session's aren't seen

private:
    gpu::optix_runtime& inner() const
    {
        if (!inner_)
        {
            throw std::logic_error("guide_tap: a CUDA session's tap has no OptiX behind it");
        }
        return *inner_;
    }

    std::shared_ptr<gpu::optix_runtime> inner_;
    std::string fail_with_;
    denoise_call last_;
    int calls_ = 0;
    int releases_ = 0;
    std::shared_ptr<int> guide_launches_ = std::make_shared<int>(0); // shared with the scenes upload made
};

// --- The scenes of G1-G8 ---------------------------------------------------------------------------------------------

// The surfaces' colors: dyadic fractions of a few bits (decision 3)
constexpr gpu::vec3d matte_color{0.75, 0.5, 0.25};    // c
constexpr gpu::vec3d mirror_color{0.5, 0.75, 1.0};    // a: the mirror's albedo, the conductors' F0
constexpr gpu::vec3d corridor_mirror{0.5, 1.0, 0.75}; // G6's a
constexpr gpu::vec3d corridor_wall{0.75, 0.5, 1.0};   // G6's c

gpu::vec3<float> to_float(const gpu::vec3d& v)
{
    return gpu::to_real<float>(v);
}

// a ⊙ b, exact in double for these colors, so exact in float too
gpu::vec3<float> times(const gpu::vec3d& a, const gpu::vec3d& b)
{
    return to_float({a.x * b.x, a.y * b.y, a.z * b.z});
}

int add_material(gpu::scene_desc& s, const gpu::material_desc& m)
{
    s.materials.push_back(m);
    return static_cast<int>(s.materials.size()) - 1;
}

int solid(gpu::scene_desc& s, const gpu::vec3d& rgb)
{
    s.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = rgb});
    return static_cast<int>(s.textures.size()) - 1;
}

int matte(gpu::scene_desc& s, const gpu::vec3d& c)
{
    return add_material(s, {.kind = gpu::material_kind::lambertian, .texture = solid(s, c)});
}

int emitter(gpu::scene_desc& s, const gpu::vec3d& light)
{
    return add_material(s, {.kind = gpu::material_kind::diffuse_light, .texture = solid(s, light)});
}

int particles(gpu::scene_desc& s, const gpu::vec3d& c)
{
    return add_material(s, {.kind = gpu::material_kind::isotropic, .texture = solid(s, c)});
}

int metal(gpu::scene_desc& s, const gpu::vec3d& albedo, double fuzz)
{
    return add_material(s, {.kind = gpu::material_kind::metal, .albedo = albedo, .fuzz = fuzz});
}

int glass(gpu::scene_desc& s)
{
    return add_material(s, {.kind = gpu::material_kind::dielectric, .refraction_index = 1.5});
}

// A view and the same view for scene_desc::camera
struct test_view
{
    gpu::camera<float> cam; // camera_for_guides of the view: the 4 × 4 grid of the guides' 16 passes
    gpu::camera_desc desc;
};

// A pinhole camera at `center` whose pixel (i, j) is centered on the direction to_pixel00 + i·du + j·dv, under a sky
// of one color; max_depth 50 (decision 4)
test_view make_view(const gpu::vec3d& center, const gpu::vec3d& to_pixel00, const gpu::vec3d& du, const gpu::vec3d& dv,
                    int width, int height, const gpu::vec3d& sky)
{
    test_view v{};
    v.desc = {.image_width = width,
              .image_height = height,
              .sqrt_spp = 4,
              .max_depth = 50,
              .center = center,
              .pixel00_loc = {center.x + to_pixel00.x, center.y + to_pixel00.y, center.z + to_pixel00.z},
              .pixel_delta_u = du,
              .pixel_delta_v = dv,
              .sky_bottom = sky,
              .sky_top = sky,
              .sampling = gpu::sampling_strategy::material};
    gpu::camera<float> cam{};
    cam.image_width = width;
    cam.image_height = height;
    cam.sqrt_spp = 1;
    cam.max_depth = 50;
    cam.recip_sqrt_spp = 1.0f;
    cam.center = to_float(center);
    cam.center_to_pixel00 = to_float(to_pixel00);
    cam.pixel_delta_u = to_float(du);
    cam.pixel_delta_v = to_float(dv);
    cam.defocus = false;
    cam.sky_bottom = to_float(sky);
    cam.sky_top = to_float(sky);
    v.cam = gpu::camera_for_guides(cam);
    return v;
}

// G1-G5's view: 8 × 8 pixels from the origin, within 0.05 of −z (pixel (0, 0) at the top left)
test_view ahead(const gpu::vec3d& sky = {})
{
    return make_view({0, 0, 0}, {-0.04375, 0.04375, -1}, {0.0125, 0, 0}, {0, -0.0125, 0}, 8, 8, sky);
}

// G6's view: 4 × 4 pixels 0.001 apart around the direction (0.2, 0, −1), so a ray crosses 0.198 to 0.202 along x for
// each unit across the corridor; under a grey sky, which nothing lets a ray reach
test_view down_the_corridor()
{
    return make_view({0, 0, 0}, {0.1985, 0.0015, -1}, {0.001, 0, 0}, {0, -0.001, 0}, 4, 4, {0.5, 0.5, 0.5});
}

// A sphere filling the view of ahead(): center (0, 0, −5), radius 2. The view's rays meet it within 0.15 of its front
// point, where the normal is within 0.08 rad of +z, so the incidence stays below 8°.
void sphere_ahead(gpu::scene_desc& s, int material)
{
    s.spheres.push_back({.center = {0, 0, -5}, .radius = 2, .material = material});
}

// A quad 10 units behind the camera, facing it (u × v = −z): every reflection of sphere_ahead lands on it
void quad_behind(gpu::scene_desc& s, int material)
{
    s.quads.push_back({.q = {-50, -50, 10}, .u = {0, 100, 0}, .v = {100, 0, 0}, .material = material});
}

// A quad 5 units ahead, facing the camera (u × v = +z), filling the view of ahead()
void quad_ahead(gpu::scene_desc& s, int material)
{
    s.quads.push_back({.q = {-10, -10, -5}, .u = {20, 0, 0}, .v = {0, 20, 0}, .material = material});
}

// A sphere behind the camera: the BVH's one surface in a scene whose view sees none
void out_of_sight(gpu::scene_desc& s)
{
    const int m = matte(s, {0.5, 0.5, 0.5});
    s.spheres.push_back({.center = {0, 0, 50}, .radius = 1, .material = m});
}

// The six faces of the box [−10, 10]³, closed (a quad's edges are its own)
void closed_box(gpu::scene_desc& s, int material)
{
    s.quads.push_back({.q = {-10, -10, -10}, .u = {20, 0, 0}, .v = {0, 20, 0}, .material = material});
    s.quads.push_back({.q = {-10, -10, 10}, .u = {20, 0, 0}, .v = {0, 20, 0}, .material = material});
    s.quads.push_back({.q = {-10, -10, -10}, .u = {20, 0, 0}, .v = {0, 0, 20}, .material = material});
    s.quads.push_back({.q = {-10, 10, -10}, .u = {20, 0, 0}, .v = {0, 0, 20}, .material = material});
    s.quads.push_back({.q = {-10, -10, -10}, .u = {0, 20, 0}, .v = {0, 0, 20}, .material = material});
    s.quads.push_back({.q = {10, -10, -10}, .u = {0, 20, 0}, .v = {0, 0, 20}, .material = material});
}

// G6: mirrors at z = −1 and z = +1 from x = −1 to the matte wall at x = wall_x, 20 wide in y. A ray of
// down_the_corridor() meets the mirrors at x = s, 3s, 5s... (s from 0.198 to 0.202): the k-th at x = (2k − 1)s. With
// the wall at 2.8, the 7th lies at most at 2.626 and the 8th at least at 2.97, so 7 mirrors, then the wall within
// 0.15 of z = 0; with the wall at 4.8, the 12th at most at 4.646 and the 13th at least at 4.95, so 12 mirrors.
void corridor(gpu::scene_desc& s, double wall_x)
{
    const int mirror = metal(s, corridor_mirror, 0);
    const int wall = matte(s, corridor_wall);
    const double length = wall_x + 1;
    s.quads.push_back({.q = {-1, -10, -1}, .u = {length, 0, 0}, .v = {0, 20, 0}, .material = mirror}); // faces +z
    s.quads.push_back({.q = {-1, -10, 1}, .u = {0, 20, 0}, .v = {length, 0, 0}, .material = mirror});  // faces −z
    // u × v = −x: the wall's front faces the rays, so its outward normal and the one facing the ray are the same
    s.quads.push_back({.q = {wall_x, -10, -1}, .u = {0, 0, 2}, .v = {0, 20, 0}, .material = wall});
}

enum class normal_rule
{
    exactly, // that vector at every pixel, within 10⁻⁶
    toward,  // unit, and within acos(0.99) of that vector
    unit,    // unit
    zero,    // exactly (0, 0, 0)
};

struct guide_case
{
    std::string name;
    gpu::scene_desc scene;
    test_view view;
    gpu::vec3<float> albedo{};  // at every pixel
    float albedo_tolerance = 0; // 0: exactly
    normal_rule rule = normal_rule::unit;
    gpu::vec3<float> normal{};
    bool books_view = false; // launch_guides without a quality scene, as a session of the books' kernels calls it
};

std::vector<guide_case> guide_cases()
{
    const gpu::vec3<float> c = to_float(matte_color);
    const gpu::vec3<float> a = to_float(mirror_color);
    const gpu::vec3<float> plus_z{0, 0, 1};
    const gpu::vec3<float> minus_z{0, 0, -1};
    std::vector<guide_case> cases;

    // G1
    {
        gpu::scene_desc s;
        sphere_ahead(s, matte(s, matte_color));
        cases.push_back({.name = "matte-sphere",
                         .scene = s,
                         .view = ahead(),
                         .albedo = c,
                         .rule = normal_rule::toward,
                         .normal = plus_z});
        cases.push_back({.name = "matte-sphere-books-view",
                         .scene = s,
                         .view = ahead(),
                         .albedo = c,
                         .rule = normal_rule::toward,
                         .normal = plus_z,
                         .books_view = true});
    }
    {
        gpu::scene_desc s;
        quad_ahead(s, matte(s, matte_color));
        cases.push_back({.name = "matte-quad",
                         .scene = s,
                         .view = ahead(),
                         .albedo = c,
                         .rule = normal_rule::exactly,
                         .normal = plus_z});
    }

    // G2
    {
        gpu::scene_desc s;
        sphere_ahead(s, metal(s, mirror_color, 0.5));
        cases.push_back({.name = "metal-fuzz-0.5",
                         .scene = s,
                         .view = ahead(),
                         .albedo = a,
                         .rule = normal_rule::toward,
                         .normal = plus_z});
    }
    {
        gpu::scene_desc s;
        sphere_ahead(s, metal(s, mirror_color, 0));
        s.quality.microfacets = {{.kind = gpu::microfacet_kind::conductor, .roughness = 0.5}};
        cases.push_back({.name = "conductor-0.5",
                         .scene = s,
                         .view = ahead(),
                         .albedo = a,
                         .rule = normal_rule::toward,
                         .normal = plus_z});
    }
    {
        gpu::scene_desc s;
        sphere_ahead(s, matte(s, matte_color));
        s.quality.microfacets = {{.kind = gpu::microfacet_kind::gloss, .roughness = 0.5}};
        cases.push_back({.name = "gloss-0.5",
                         .scene = s,
                         .view = ahead(),
                         .albedo = c,
                         .rule = normal_rule::toward,
                         .normal = plus_z});
    }
    {
        // A volume of density 10⁴ in the sphere of sphere_ahead: a particle within 10⁻³ of its boundary, every time
        gpu::scene_desc s;
        out_of_sight(s);
        const int m = particles(s, matte_color);
        s.media.push_back({.boundary = gpu::boundary_shape::sphere,
                           .sphere = {.center = {0, 0, -5}, .radius = 2},
                           .neg_inv_density = -1e-4,
                           .material = m});
        cases.push_back({.name = "particles",
                         .scene = s,
                         .view = ahead(),
                         .albedo = c,
                         .rule = normal_rule::toward,
                         .normal = plus_z});
    }

    // G3: materials 0 the sphere's, 1 the quad's
    {
        gpu::scene_desc s;
        sphere_ahead(s, metal(s, mirror_color, 0));
        quad_behind(s, matte(s, matte_color));
        cases.push_back({.name = "mirror",
                         .scene = s,
                         .view = ahead(),
                         .albedo = times(mirror_color, matte_color),
                         .rule = normal_rule::exactly,
                         .normal = minus_z});
    }
    for (const double r : {0.0, 0.05})
    {
        gpu::scene_desc s;
        sphere_ahead(s, metal(s, mirror_color, 0));
        quad_behind(s, matte(s, matte_color));
        s.quality.microfacets = {{.kind = gpu::microfacet_kind::conductor, .roughness = r}, {}};
        cases.push_back({.name = r == 0 ? "smooth-conductor" : "conductor-0.05",
                         .scene = s,
                         .view = ahead(),
                         .albedo = times(mirror_color, matte_color),
                         .albedo_tolerance = r == 0 ? 1e-5f : 1e-2f,
                         .rule = normal_rule::exactly,
                         .normal = minus_z});
    }

    // G4
    {
        gpu::scene_desc s;
        sphere_ahead(s, glass(s));
        closed_box(s, matte(s, matte_color));
        cases.push_back(
            {.name = "glass-in-a-box", .scene = s, .view = ahead(), .albedo = c, .rule = normal_rule::unit});
    }

    // G5
    for (const gpu::vec3d& light : {gpu::vec3d{4, 2, 1}, gpu::vec3d{0.5, 0.25, 0.125}})
    {
        gpu::scene_desc s;
        quad_ahead(s, emitter(s, light));
        const double largest = std::max({1.0, light.x, light.y, light.z});
        cases.push_back({.name = light.x > 1 ? "lamp-bright" : "lamp-dim",
                         .scene = s,
                         .view = ahead(),
                         .albedo = to_float({light.x / largest, light.y / largest, light.z / largest}),
                         .rule = normal_rule::exactly,
                         .normal = plus_z});
    }
    {
        gpu::scene_desc s;
        out_of_sight(s);
        cases.push_back({.name = "sky",
                         .scene = s,
                         .view = ahead({2, 0.5, 0}),
                         .albedo = {1.0f, 0.5f, 0.0f},
                         .rule = normal_rule::zero});
    }

    // G6
    for (const int mirrors : {7, 12})
    {
        gpu::scene_desc s;
        corridor(s, mirrors == 7 ? 2.8 : 4.8);
        gpu::vec3d expected{};
        if (mirrors == 7)
        {
            expected = corridor_wall;
            for (int k = 0; k < mirrors; ++k)
            {
                expected = {expected.x * corridor_mirror.x, expected.y * corridor_mirror.y,
                            expected.z * corridor_mirror.z};
            }
        }
        cases.push_back({.name = std::format("corridor-{}-mirrors", mirrors),
                         .scene = s,
                         .view = down_the_corridor(),
                         .albedo = to_float(expected),
                         .rule = mirrors == 7 ? normal_rule::exactly : normal_rule::zero,
                         .normal = {-1, 0, 0}});
    }

    for (guide_case& g : cases)
    {
        g.scene.camera = g.view.desc;
    }
    return cases;
}

// --- G1-G8: the guides of a case, and its verdicts -------------------------------------------------------------------

struct guide_images
{
    std::vector<float4> albedo;
    std::vector<float4> normal;
};

// CUDA's guides: the guide kernel on the scene as device_scene and upload_quality put it on the GPU
guide_images cuda_guides(const guide_case& c)
{
    const gpu_test::device_scene device(c.scene);
    const gpu::detail::quality_scene_ptr quality =
        gpu::detail::upload_quality(c.scene, device.sphere_ids(), device.quad_ids());
    const std::size_t pixels = count_of(c.view.cam.image_width, c.view.cam.image_height);
    const gpu_test::device_buffer<float4> albedo(pixels);
    const gpu_test::device_buffer<float4> normal(pixels);
    gpu::detail::launch_guides(c.books_view ? nullptr : quality.get(), device.view(), c.view.cam, table_seed,
                               albedo.get(), normal.get());
    gpu_test::finish_launch("launch_guides");
    return {albedo.download(), normal.download()};
}

// OptiX's: the same scene uploaded by the runtime, the guides of its optix_scene (destroyed before the quality scene
// and the tables it reads)
guide_images optix_guides(gpu::optix_runtime& runtime, const guide_case& c)
{
    const gpu_test::device_scene device(c.scene);
    const gpu::detail::quality_scene_ptr quality =
        gpu::detail::upload_quality(c.scene, device.sphere_ids(), device.quad_ids());
    const std::unique_ptr<gpu::detail::optix_scene> scene =
        runtime.upload(c.scene, device.view(), *quality, device.sphere_ids(), device.quad_ids());
    const int width = c.view.cam.image_width;
    const int height = c.view.cam.image_height;
    scene->prepare_view(width, height);
    const gpu_test::device_buffer<float4> albedo(count_of(width, height));
    const gpu_test::device_buffer<float4> normal(count_of(width, height));
    scene->guides(c.view.cam, table_seed, albedo.get(), normal.get());
    gpu_test::finish_launch("optix_scene::guides");
    return {albedo.download(), normal.download()};
}

bool within(float value, float expected, float tolerance)
{
    return std::abs(static_cast<double>(value) - static_cast<double>(expected)) <= tolerance; // a NaN: false
}

bool in_unit_interval(float v)
{
    return v >= 0.0f && v <= 1.0f; // a NaN: false
}

double length_of(const float4& n)
{
    const double x = n.x;
    const double y = n.y;
    const double z = n.z;
    return std::sqrt(x * x + y * y + z * z);
}

bool is_zero(const float4& n)
{
    return n.x == 0.0f && n.y == 0.0f && n.z == 0.0f;
}

bool is_unit(const float4& n)
{
    return std::abs(length_of(n) - 1) <= 1e-5;
}

// G7's counts over every pixel judged
struct guide_tally
{
    std::int64_t pixels = 0;
    std::int64_t not_unit_or_zero = 0;
    std::int64_t outside_unit_interval = 0;

    void add(const float4& albedo, const float4& normal)
    {
        ++pixels;
        not_unit_or_zero += is_unit(normal) || is_zero(normal) ? 0 : 1;
        outside_unit_interval +=
            in_unit_interval(albedo.x) && in_unit_interval(albedo.y) && in_unit_interval(albedo.z) ? 0 : 1;
    }
};

std::string describe_pixel(int i, int j, const float4& albedo, const float4& normal)
{
    return std::format("pixel ({}, {}): albedo ({:.9g}, {:.9g}, {:.9g}), normal ({:.9g}, {:.9g}, {:.9g})", i, j,
                       albedo.x, albedo.y, albedo.z, normal.x, normal.y, normal.z);
}

void judge_case(gpu_test::context& ctx, const std::string& prefix, const guide_case& c, const guide_images& g,
                guide_tally& tally)
{
    const int width = c.view.cam.image_width;
    const int height = c.view.cam.image_height;
    const std::size_t pixels = count_of(width, height);
    if (g.albedo.size() != pixels || g.normal.size() != pixels)
    {
        throw std::runtime_error(std::format("{}{}: {} albedo and {} normal texels for {} pixels", prefix, c.name,
                                             g.albedo.size(), g.normal.size(), pixels));
    }
    std::int64_t albedo_wrong = 0;
    std::int64_t normal_wrong = 0;
    std::string first;
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            const std::size_t k = count_of(width, j) + static_cast<std::size_t>(i);
            const float4 a = g.albedo[k];
            const float4 n = g.normal[k];
            tally.add(a, n);
            const float t = c.albedo_tolerance;
            const bool albedo_right =
                within(a.x, c.albedo.x, t) && within(a.y, c.albedo.y, t) && within(a.z, c.albedo.z, t);
            bool normal_right = false;
            switch (c.rule)
            {
            case normal_rule::exactly:
                normal_right = within(n.x, c.normal.x, 1e-6f) && within(n.y, c.normal.y, 1e-6f) &&
                               within(n.z, c.normal.z, 1e-6f);
                break;
            case normal_rule::toward:
                normal_right = is_unit(n) && (static_cast<double>(n.x) * c.normal.x +
                                              static_cast<double>(n.y) * c.normal.y +
                                              static_cast<double>(n.z) * c.normal.z) / length_of(n) >= 0.99;
                break;
            case normal_rule::unit:
                normal_right = is_unit(n);
                break;
            case normal_rule::zero:
                normal_right = is_zero(n);
                break;
            }
            albedo_wrong += albedo_right ? 0 : 1;
            normal_wrong += normal_right ? 0 : 1;
            if ((!albedo_right || !normal_right) && first.empty())
            {
                first = describe_pixel(i, j, a, n);
            }
        }
    }
    const auto out_of = static_cast<std::int64_t>(pixels);
    ctx.count(prefix + c.name + "-albedo", albedo_wrong, 0, out_of);
    ctx.count(prefix + c.name + "-normal", normal_wrong, 0, out_of);
    if (!first.empty())
    {
        ctx.info(prefix + c.name + "-first-wrong",
                 std::format("{}; expected albedo ({:.9g}, {:.9g}, {:.9g})", first, c.albedo.x, c.albedo.y,
                             c.albedo.z));
    }
}

void judge_tally(gpu_test::context& ctx, const std::string& prefix, const guide_tally& tally)
{
    ctx.count(prefix + "normals-unit-or-zero", tally.not_unit_or_zero, 0, tally.pixels);
    ctx.count(prefix + "albedo-in-0-1", tally.outside_unit_interval, 0, tally.pixels);
}

// --- G9: a session's guides ------------------------------------------------------------------------------------------

// The guides display_denoised computes for one pass of the scene's view, seen by a tap (decision 1)
denoise_call session_guides(const gpu::scene_desc& scene, bool optix_tracer, std::shared_ptr<gpu::optix_runtime> inner)
{
    const auto tap = std::make_shared<guide_tap>(std::move(inner));
    gpu::session_options options;
    options.seed = session_seed;
    options.optix_tracer = optix_tracer;
    options.optix = tap;
    gpu::session s(scene, options);
    s.render_to({1, 0});
    s.display_denoised(gpu::display_options{});
    return tap->last();
}

struct guide_difference
{
    std::int64_t disagreeing = 0; // pixels with a channel of the albedo or the normal more than 10⁻³ apart
    double largest = 0;           // the largest |Δ| of a channel
};

// b's albedo is multiplied by albedo_factor first (the planted fault's 1.01; 1 otherwise)
guide_difference compare_guides(const denoise_call& a, const denoise_call& b, float albedo_factor)
{
    guide_difference d;
    if (a.albedo.size() != b.albedo.size() || a.normal.size() != b.normal.size() ||
        a.albedo.size() != a.normal.size())
    {
        d.disagreeing = static_cast<std::int64_t>(std::max(a.albedo.size(), b.albedo.size()));
        d.largest = std::numeric_limits<double>::infinity();
        return d;
    }
    for (std::size_t k = 0; k < a.albedo.size(); ++k)
    {
        const std::array<double, 6> deltas{
            std::abs(static_cast<double>(a.albedo[k].x) - static_cast<double>(b.albedo[k].x * albedo_factor)),
            std::abs(static_cast<double>(a.albedo[k].y) - static_cast<double>(b.albedo[k].y * albedo_factor)),
            std::abs(static_cast<double>(a.albedo[k].z) - static_cast<double>(b.albedo[k].z * albedo_factor)),
            std::abs(static_cast<double>(a.normal[k].x) - static_cast<double>(b.normal[k].x)),
            std::abs(static_cast<double>(a.normal[k].y) - static_cast<double>(b.normal[k].y)),
            std::abs(static_cast<double>(a.normal[k].z) - static_cast<double>(b.normal[k].z))};
        bool agree = true;
        for (const double delta : deltas)
        {
            agree = agree && delta <= 1e-3; // a NaN disagrees
            d.largest = delta == delta ? std::max(d.largest, delta) : std::numeric_limits<double>::infinity();
        }
        d.disagreeing += agree ? 0 : 1;
    }
    return d;
}

guide_tally tally_of(const denoise_call& call)
{
    guide_tally tally;
    for (std::size_t k = 0; k < std::min(call.albedo.size(), call.normal.size()); ++k)
    {
        tally.add(call.albedo[k], call.normal[k]);
    }
    return tally;
}

// --- The denoiser's inputs -------------------------------------------------------------------------------------------

// The four float4 images of one denoise in the GPU's memory: beauty, albedo, normal (copied in) and the output
class denoise_images
{
public:
    denoise_images(int width, int height, const std::vector<float4>& beauty, const std::vector<float4>& albedo,
                   const std::vector<float4>& normal)
        : width_(width), height_(height), beauty_(beauty), albedo_(albedo), normal_(normal), out_(beauty.size())
    {
    }

    void denoise(gpu::optix_runtime& runtime) const
    {
        runtime.denoise(beauty_.get(), albedo_.get(), normal_.get(), out_.get(), width_, height_);
        gpu_test::finish_launch("optix_runtime::denoise");
    }

    std::vector<float4> out() const { return out_.download(); }

private:
    int width_;
    int height_;
    gpu_test::device_buffer<float4> beauty_;
    gpu_test::device_buffer<float4> albedo_;
    gpu_test::device_buffer<float4> normal_;
    gpu_test::device_buffer<float4> out_;
};

// The runtime's tiles of at most `side` (set_denoise_tile_limit) for one scope, taken off again however it ends: the
// runtime is the process's, and no later check may denoise in tiles it didn't ask for
class tile_limit
{
public:
    tile_limit(gpu::optix_runtime& runtime, int side) : runtime_(runtime) { runtime_.set_denoise_tile_limit(side); }
    ~tile_limit() { runtime_.set_denoise_tile_limit(0); } // 0 is never refused
    tile_limit(const tile_limit&) = delete;
    tile_limit& operator=(const tile_limit&) = delete;

private:
    gpu::optix_runtime& runtime_;
};

// A uniform number in (0, 1) from 53 bits of mt19937_64, whose numbers the standard fixes
double uniform_open(std::mt19937_64& rng)
{
    return (static_cast<double>(rng() >> 11) + 0.5) * 0x1p-53;
}

// D2's image: a smooth gradient, every channel in [0.3, 0.8], plus Gaussian noise of σ = 0.1 per channel (Box-Muller:
// std::normal_distribution's algorithm is the library's to choose), alpha 1
struct synthetic_image
{
    std::vector<float4> clean;
    std::vector<float4> noisy;
};

synthetic_image make_synthetic(int width, int height, std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    double spare = 0;
    bool has_spare = false;
    const auto gaussian = [&] {
        if (has_spare)
        {
            has_spare = false;
            return spare;
        }
        const double radius = std::sqrt(-2 * std::log(uniform_open(rng)));
        const double angle = 2 * std::numbers::pi * uniform_open(rng);
        spare = radius * std::sin(angle);
        has_spare = true;
        return radius * std::cos(angle);
    };
    synthetic_image image;
    image.clean.reserve(count_of(width, height));
    image.noisy.reserve(count_of(width, height));
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const double u = static_cast<double>(x) / (width - 1);
            const double v = static_cast<double>(y) / (height - 1);
            const std::array<double, 3> c{0.3 + 0.5 * u, 0.3 + 0.5 * v, 0.55 - 0.25 * (u - v)};
            std::array<float, 3> n{};
            for (std::size_t k = 0; k < 3; ++k)
            {
                n[k] = static_cast<float>(c[k] + 0.1 * gaussian());
            }
            image.clean.push_back(make_float4(static_cast<float>(c[0]), static_cast<float>(c[1]),
                                              static_cast<float>(c[2]), 1.0f));
            image.noisy.push_back(make_float4(n[0], n[1], n[2], 1.0f));
        }
    }
    return image;
}

// The mean of (a − b)² over the three color channels of every pixel, in double
double mse(const std::vector<float4>& a, const std::vector<float4>& b)
{
    double sum = 0;
    for (std::size_t k = 0; k < a.size(); ++k)
    {
        const double dx = static_cast<double>(a[k].x) - b[k].x;
        const double dy = static_cast<double>(a[k].y) - b[k].y;
        const double dz = static_cast<double>(a[k].z) - b[k].z;
        sum += dx * dx + dy * dy + dz * dz;
    }
    return sum / (3.0 * static_cast<double>(a.size()));
}

std::array<double, 3> channel_means(const std::vector<float4>& image)
{
    std::array<double, 3> sum{};
    for (const float4& p : image)
    {
        sum[0] += p.x;
        sum[1] += p.y;
        sum[2] += p.z;
    }
    for (double& s : sum)
    {
        s /= static_cast<double>(image.size());
    }
    return sum;
}

// Color channels that aren't finite numbers
std::int64_t not_finite(const std::vector<float4>& image)
{
    std::int64_t found = 0;
    for (const float4& p : image)
    {
        found += (std::isfinite(p.x) ? 0 : 1) + (std::isfinite(p.y) ? 0 : 1) + (std::isfinite(p.z) ? 0 : 1);
    }
    return found;
}

// The device's free memory now, in bytes
std::size_t free_device_bytes()
{
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    gpu_test::cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "reading the GPU's free memory");
    return free_bytes;
}

constexpr double mebibyte = 1024.0 * 1024.0;

// --- S1's scene ------------------------------------------------------------------------------------------------------

// 32 × 18 pixels, one sample per pass, the books' kernels (-p material): a matte floor, two balls (glass, and a metal
// of fuzz 0.1), and a lamp of radiance 2·10⁴ seen directly by a few pixels, whose beauty the clamp at 10⁴ decides
gpu::scene_desc display_scene()
{
    gpu::scene_desc s;
    const int floor = matte(s, {0.5, 0.5, 0.5});
    const int lamp = emitter(s, {2e4, 2e4, 2e4});
    const int ball = glass(s);
    const int chrome = metal(s, {0.8, 0.6, 0.4}, 0.1);
    s.quads.push_back({.q = {-10, 0, -10}, .u = {0, 0, 20}, .v = {20, 0, 0}, .material = floor}); // faces +y
    s.quads.push_back({.q = {-0.5, 0.5, -3}, .u = {1, 0, 0}, .v = {0, 1, 0}, .material = lamp});  // faces the camera
    s.spheres.push_back({.center = {1, 0.5, -1.5}, .radius = 0.5, .material = ball});
    s.spheres.push_back({.center = {-1, 0.5, -1.5}, .radius = 0.5, .material = chrome});
    s.camera = {.image_width = 32,
                .image_height = 18,
                .sqrt_spp = 1,
                .max_depth = 8,
                .center = {0, 1, 1},
                .pixel00_loc = {-0.775, 1.425, 0},
                .pixel_delta_u = {0.05, 0, 0},
                .pixel_delta_v = {0, -0.05, 0},
                .sky_bottom = {0.2, 0.3, 0.5},
                .sky_top = {1, 1, 1},
                .sampling = gpu::sampling_strategy::material};
    return s;
}

// The display's bytes of linear floats (3 per pixel) as doubles, at scale 1 over one band: books (to_rgba8) or a tone
// map (to_rgba8_toned) with the exposure's float(2^EV), as display_denoised computes them (denoise.cuh's to_sums)
std::vector<std::uint8_t> bytes_at_scale_1(const std::vector<float>& linear, int width, int height,
                                           const gpu::display_options& display)
{
    const std::vector<double> sums(linear.begin(), linear.end());
    const gpu_test::device_buffer<double> device_sums(sums);
    const gpu_test::device_buffer<std::uint8_t> rgba(4 * count_of(width, height));
    if (display.op == gpu::tone_map::books)
    {
        gpu::detail::to_rgba8(device_sums.get(), width, 0, height, 1.0, rgba.get());
    }
    else
    {
        gpu::detail::to_rgba8_toned(device_sums.get(), width, 0, height, 1.0, display.op,
                                    static_cast<float>(std::exp2(display.exposure)), rgba.get());
    }
    gpu_test::finish_launch("the display's bytes at scale 1");
    return rgba.download();
}

// --- denoise-eval ----------------------------------------------------------------------------------------------------

// A PFM as RayTracingGPU writes one (image::save_pfm): "PF", the size, -1.0 (little-endian floats), then the rows from
// the bottom of the image to its top. `rgb` holds 3 values per pixel, rows from the top.
template <class T>
void write_pfm(const std::filesystem::path& file, const std::vector<T>& rgb, int width, int height)
{
    const std::size_t row_values = 3 * static_cast<std::size_t>(width);
    if (rgb.size() != row_values * static_cast<std::size_t>(height))
    {
        throw std::runtime_error(
            std::format("{}: {} values for {}×{} pixels", file.string(), rgb.size(), width, height));
    }
    std::ofstream out(file, std::ios::binary);
    if (!out)
    {
        throw std::runtime_error("could not open " + file.string());
    }
    const std::string header = std::format("PF\n{} {}\n-1.0\n", width, height);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    std::vector<unsigned char> row(4 * row_values);
    for (int y = height - 1; y >= 0; --y)
    {
        for (std::size_t i = 0; i < row_values; ++i)
        {
            const auto bits =
                std::bit_cast<std::uint32_t>(static_cast<float>(rgb[static_cast<std::size_t>(y) * row_values + i]));
            for (std::size_t b = 0; b < 4; ++b)
            {
                row[4 * i + b] = static_cast<unsigned char>(bits >> (8 * b));
            }
        }
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
    }
    out.close();
    if (!out)
    {
        throw std::runtime_error("could not write " + file.string());
    }
}

struct eval_render
{
    gpu::render_result result;
    double seconds = 0; // the render() call's wall time, for exclude_seconds only (decision 8)
};

// One render of the protocol: NAME.pfm, and with denoise NAME-denoised.pfm, written into the folder
eval_render render_and_write(gpu_test::context& ctx, const std::filesystem::path& folder, const std::string& name,
                             const gpu::scene_desc& scene, std::uint64_t seed, bool optix_tracer, bool denoise,
                             const std::shared_ptr<gpu::optix_runtime>& runtime)
{
    gpu::render_options options;
    options.seed = seed;
    options.optix_tracer = optix_tracer;
    options.optix = runtime;
    options.denoise = denoise;
    const auto start = std::chrono::steady_clock::now();
    eval_render r{.result = gpu::render(scene, options)};
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    // A denoise that failed after the passes leaves render() a result without its denoised image: a failed check, and
    // nothing written, so that no noisy image ever lies in the folder without its denoised twin
    if (denoise && !r.result.denoise_error.empty())
    {
        ctx.fail(name + "-denoised", r.result.denoise_error);
        throw std::runtime_error(std::format("{}: the denoise failed, so nothing of it was written: {}", name,
                                             r.result.denoise_error));
    }
    const int width = scene.camera.image_width;
    const int height = scene.camera.image_height;
    write_pfm(folder / (name + ".pfm"), r.result.pixels, width, height);
    std::string files = name + ".pfm";
    if (denoise)
    {
        write_pfm(folder / (name + "-denoised.pfm"), r.result.denoised, width, height);
        files += ", " + name + "-denoised.pfm";
        std::int64_t bad = 0;
        for (const float v : r.result.denoised)
        {
            bad += std::isfinite(v) ? 0 : 1;
        }
        ctx.count(name + "-denoised-finite", bad, 0, static_cast<std::int64_t>(r.result.denoised.size()));
        // What the denoiser was given besides the images, computed over the whole image before its invoke
        const gpu::denoise_report report = runtime->last_denoise();
        ctx.info(name + "-denoiser-exposure",
                 report.valid ? std::format("hdrIntensity {:.6g}, hdrAverageColor ({:.6g}, {:.6g}, {:.6g})",
                                            report.intensity, report.average_color[0], report.average_color[1],
                                            report.average_color[2])
                              : std::string("the runtime reports no denoise"));
    }
    const std::int64_t passes = std::int64_t{scene.camera.sqrt_spp} * scene.camera.sqrt_spp;
    ctx.info(name, std::format("{}×{}, {} passes, kernel {}: {}", width, height, passes, r.result.kernel.name, files));
    return r;
}
}

// --- guides ----------------------------------------------------------------------------------------------------------

// G1-G7 through CUDA's guide kernel
GPU_TEST(guides, table)
{
    guide_tally tally;
    for (const guide_case& c : guide_cases())
    {
        judge_case(ctx, "", c, cuda_guides(c), tally);
    }
    judge_tally(ctx, "", tally);
}

// G8: the same scenes through OptiX
GPU_TEST(guides, optix_table)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "optix-table");
    if (!runtime)
    {
        return;
    }
    gpu_test::optix::excluded_seconds(ctx, [&runtime] { runtime->compile_programs(); });
    guide_tally tally;
    for (const guide_case& c : guide_cases())
    {
        if (c.books_view)
        {
            continue; // an OptiX scene always has its quality scene's view
        }
        judge_case(ctx, "optix-", c, optix_guides(*runtime, c), tally);
    }
    judge_tally(ctx, "optix-", tally);
}

// G9: cornell and final, CUDA's guides against OptiX's, as their sessions compute them
GPU_TEST(guides, cuda_against_optix)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "cuda-against-optix");
    if (!runtime)
    {
        return;
    }
    gpu_test::optix::excluded_seconds(ctx, [&runtime] { runtime->compile_programs(); });
    const int width = ctx.quick() ? 160 : 320;
    const int height = ctx.quick() ? 90 : 180;
    const auto pixels = static_cast<std::int64_t>(count_of(width, height));
    const std::int64_t allowed = pixels / 1000; // 99.9% must agree
    struct guide_scene
    {
        const char* name;
        const char* strategy; // its -p, in the checks' names
        gpu_test::host::scene_strategy sampling;
    };
    constexpr std::array<guide_scene, 2> scenes{{{"cornell", "material", gpu_test::host::scene_strategy::material},
                                                 {"final", "mis", gpu_test::host::scene_strategy::mis}}};
    for (const auto& [scene_name, strategy, sampling] : scenes)
    {
        const std::string name = std::format("{}-{}", scene_name, strategy);
        const gpu::scene_desc scene = scene_named(scene_name, sampling, width, height, 16);
        const denoise_call cuda = session_guides(scene, false, nullptr);
        const denoise_call optix = session_guides(scene, true, runtime);
        ctx.exact(name + "-size", static_cast<double>(cuda.albedo.size()), static_cast<double>(pixels), 0);
        const guide_difference d = compare_guides(cuda, optix, 1.0f);
        ctx.count(name, d.disagreeing, allowed, pixels);
        ctx.info(name + "-largest",
                 std::format("{:.3g}: the largest |Δ| of a channel of the albedo or the normal", d.largest));
        for (const auto& [side, call] : {std::pair<const char*, const denoise_call*>{"cuda", &cuda},
                                         std::pair<const char*, const denoise_call*>{"optix", &optix}})
        {
            const guide_tally tally = tally_of(*call);
            judge_tally(ctx, std::format("{}-{}-", name, side), tally);
        }
        if (std::string_view(scene_name) == "cornell")
        {
            // The planted fault: the criterion must see OptiX's albedo 1% too bright
            const guide_difference planted = compare_guides(cuda, optix, 1.01f);
            ctx.info("planted-albedo-1.01",
                     std::format("{} of {} pixels disagree (the criterion allows {})", planted.disagreeing, pixels,
                                 allowed));
            ctx.exact("planted-albedo-1.01-rejected", planted.disagreeing > allowed ? 1 : 0, 1, 0);
        }
    }

    // cornell-triangles (Step 91): cornell's walls and blocks as two triangles a quad, with the quads' corners, and the
    // same materials, lamp (a quad still) and camera. OptiX's guides of it against CUDA's of cornell's quads, -p
    // material, the same seed. The threshold, from the geometry before any run: a pixel whose 16 samples all meet one
    // face in both scenes has the same guides, the face's matte color exactly and its plane's normal to float rounding
    // (far below 10⁻³). A pixel can differ only where a sample passes so close to an edge between two faces of another
    // color or orientation (a corner of the room, a block's edge, the box's open front) that the quad's test and the
    // triangle's, both in float, pick different faces: within about 10⁻³ of a unit of the edge (a few ulps of
    // coordinates up to about 1355), against pixels 3 to 6 units wide at 320×180 (twice that at 160×90). Such edges
    // cross about 3% of the pixels, and one of a pixel's 16 samples lands that close to its edge with a chance of about
    // 16 × 2·10⁻³ / 3 ≈ 1%: some 0.03% of the pixels, ten times that at most where a wall is seen at a grazing angle.
    // Allowed: 1% of the pixels, each channel within 10⁻³, more than three times that pessimistic share; a fault of the
    // triangles' guides (a face's normal turned, a wrong material, a mesh out of place) changes every pixel of a face,
    // and every face in sight but the short block's top covers more than 1% of the image. The numbers measured are
    // INFO.
    {
        const std::int64_t edge_allowed = pixels / 100;
        const gpu::scene_desc quads =
            scene_named("cornell", gpu_test::host::scene_strategy::material, width, height, 16);
        const gpu::scene_desc triangles =
            scene_named("cornell-triangles", gpu_test::host::scene_strategy::material, width, height, 16);
        const denoise_call cuda = session_guides(quads, false, nullptr);
        const denoise_call optix = session_guides(triangles, true, runtime);
        ctx.exact("cornell-triangles-size", static_cast<double>(optix.albedo.size()), static_cast<double>(pixels), 0);
        const guide_difference d = compare_guides(cuda, optix, 1.0f);
        ctx.count("cornell-triangles-material", d.disagreeing, edge_allowed, pixels);
        ctx.info("cornell-triangles-material-largest",
                 std::format("{:.3g}: the largest |Δ| of a channel of the albedo or the normal, OptiX's triangles "
                             "against CUDA's quads",
                             d.largest));
        judge_tally(ctx, "cornell-triangles-material-optix-", tally_of(optix));
    }
}

// --- denoise ---------------------------------------------------------------------------------------------------------

// W1: to_sums, and the display's bytes of the widened floats at scale 1
GPU_TEST(denoise, widen)
{
    constexpr int width = 64;
    constexpr int height = 16;
    const std::size_t pixels = count_of(width, height);
    // Special floats first (the clamps' edges, a subnormal, negatives, 0 and −0), then log-uniform ones in [2⁻¹⁰, 2¹⁵)
    std::vector<float> values{0.0f,
                              -0.0f,
                              std::numeric_limits<float>::denorm_min(),
                              std::numeric_limits<float>::min(),
                              0.25f,
                              0.5f,
                              0.999f,
                              1.0f,
                              std::nextafter(1.0f, 2.0f),
                              1.5f,
                              3.0f,
                              10.0f,
                              1e4f,
                              std::nextafter(1e4f, 2e4f),
                              65504.0f,
                              1e30f,
                              -0.5f,
                              -1e-3f};
    std::mt19937_64 rng(widen_seed);
    while (values.size() < 3 * pixels)
    {
        values.push_back(static_cast<float>(std::exp2(-10.0 + 25.0 * uniform_open(rng))));
    }
    std::vector<float4> image(pixels);
    for (std::size_t k = 0; k < pixels; ++k)
    {
        image[k] = make_float4(values[3 * k], values[3 * k + 1], values[3 * k + 2], 1.0f);
    }
    const std::vector<double> original(values.begin(), values.end());
    std::vector<double> quadruple(original.size());
    std::ranges::transform(original, quadruple.begin(), [](double v) { return 4 * v; }); // exact: a power of two

    const gpu_test::device_buffer<float4> device_image(image);
    const gpu_test::device_buffer<double> widened(3 * pixels);
    gpu::detail::to_sums(device_image.get(), width, height, widened.get());
    gpu_test::finish_launch("to_sums");
    const auto values_out_of = static_cast<std::int64_t>(3 * pixels);
    ctx.count("to-sums-exact", bit_differences(widened.download(), original), 0, values_out_of);

    const gpu_test::device_buffer<double> device_original(original);
    const gpu_test::device_buffer<double> device_quadruple(quadruple);
    const gpu_test::device_buffer<std::uint8_t> expected(4 * pixels);
    const gpu_test::device_buffer<std::uint8_t> got(4 * pixels);
    const auto bytes_out_of = static_cast<std::int64_t>(4 * pixels);
    const auto compare = [&](const std::string& name, const auto& display) {
        display(widened.get(), 1.0, got.get());
        display(device_original.get(), 1.0, expected.get());
        gpu_test::finish_launch("the display's bytes");
        const std::vector<std::uint8_t> widened_bytes = got.download();
        ctx.count(name + "-scale-1", bit_differences(widened_bytes, expected.download()), 0, bytes_out_of);
        display(device_quadruple.get(), 0.25, expected.get());
        gpu_test::finish_launch("the display's bytes");
        ctx.count(name + "-quarter-of-4x", bit_differences(widened_bytes, expected.download()), 0, bytes_out_of);
    };
    compare("books", [&](const double* sums, double scale, std::uint8_t* rgba) {
        gpu::detail::to_rgba8(sums, width, 0, height, scale, rgba);
    });
    constexpr std::array<std::pair<gpu::tone_map, const char*>, 3> tone_maps{
        {{gpu::tone_map::none, "none"}, {gpu::tone_map::agx, "agx"}, {gpu::tone_map::neutral, "neutral"}}};
    for (const auto& tone : tone_maps)
    {
        const gpu::tone_map op = tone.first; // a variable of its own: the lambda below captures it
        for (const double ev : {0.0, 1.5})
        {
            const auto exposure_scale = static_cast<float>(std::exp2(ev));
            compare(std::format("{}-{}ev", tone.second, ev), [&](const double* sums, double scale, std::uint8_t* rgba) {
                gpu::detail::to_rgba8_toned(sums, width, 0, height, scale, op, exposure_scale, rgba);
            });
        }
    }
}

// B1: to_beauty in display()'s two bands, a pass in progress over the top rows (3 samples there, 2 below)
GPU_TEST(denoise, beauty)
{
    constexpr int width = 32;
    constexpr int height = 8;
    constexpr int split = 3;
    constexpr double top_scale = 1.0 / 3;
    constexpr double rest_scale = 1.0 / 2;
    const std::size_t pixels = count_of(width, height);
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    constexpr double infinity = std::numeric_limits<double>::infinity();
    std::vector<double> sums{nan,  infinity, -infinity, -1.0, -0.0, 0.0,   1e-300, 0.75,  2.9999e4, 3e4,
                             3.0000001e4, 2e4, 2.0000001e4, 1e30, 1e300, -1e300, 1e-30, 9.0, 12.5};
    std::mt19937_64 rng(beauty_seed);
    while (sums.size() < 3 * pixels)
    {
        const double u = uniform_open(rng);
        sums.push_back(sums.size() % 2 == 0 ? 90 * u : std::exp2(-10.0 + 30.0 * u)); // [0, 90) and [2⁻¹⁰, 2²⁰)
    }
    const gpu_test::device_buffer<double> device_sums(sums);
    for (const bool top_first : {true, false})
    {
        const gpu_test::device_buffer<float4> beauty(pixels);
        const auto top = [&] { gpu::detail::to_beauty(device_sums.get(), width, 0, split, top_scale, beauty.get()); };
        const auto rest = [&] {
            gpu::detail::to_beauty(device_sums.get(), width, split, height - split, rest_scale, beauty.get());
        };
        if (top_first)
        {
            top();
            rest();
        }
        else
        {
            rest();
            top();
        }
        gpu_test::finish_launch("to_beauty");
        const std::vector<float4> got = beauty.download();
        std::int64_t wrong = 0;
        std::int64_t alpha = 0;
        std::string first;
        for (std::size_t k = 0; k < pixels; ++k)
        {
            const double scale = static_cast<int>(k / width) < split ? top_scale : rest_scale;
            const std::array<float, 3> value{got[k].x, got[k].y, got[k].z};
            for (std::size_t c = 0; c < 3; ++c)
            {
                const float expected = beauty_of(sums[3 * k + c], scale);
                const bool right = value[c] == expected; // −0 and 0 alike; a NaN never
                wrong += right ? 0 : 1;
                if (!right && first.empty())
                {
                    first = std::format("sum {:.17g} × {:.17g}: {:.9g}, expected {:.9g}", sums[3 * k + c], scale,
                                        value[c], expected);
                }
            }
            alpha += got[k].w == 1.0f ? 0 : 1;
        }
        const std::string order = top_first ? "top-band-first" : "top-band-last";
        ctx.count("values-" + order, wrong, 0, static_cast<std::int64_t>(3 * pixels));
        ctx.count("alpha-" + order, alpha, 0, static_cast<std::int64_t>(pixels));
        if (!first.empty())
        {
            ctx.info("first-wrong-" + order, first);
        }
    }
}

// S1: display_denoised, retone_denoised, a new view, attach_optix, drop_denoised and render()'s denoise through
// sessions whose denoiser is a tap (no OptiX needed, but for the count of guide launches in an OptiX session)
GPU_TEST(denoise, session)
{
    const gpu::scene_desc scene = display_scene();
    const int width = scene.camera.image_width;
    const int height = scene.camera.image_height;
    const int split = height / 2;
    const std::size_t pixels = count_of(width, height);
    const auto values_out_of = static_cast<std::int64_t>(3 * pixels);
    const auto bytes_out_of = static_cast<std::int64_t>(4 * pixels);

    const auto tap = std::make_shared<guide_tap>(nullptr);
    gpu::session_options options;
    options.seed = display_seed;
    options.optix = tap;
    gpu::session s(scene, options);
    s.render_to({1, 0});
    const std::vector<double> one = s.pixels();
    s.render_to({1, split}); // the second pass over rows [0, split): display()'s two bands
    const std::span<const std::uint8_t> shown = s.display_denoised(gpu::display_options{});
    const std::vector<std::uint8_t> books_bytes(shown.begin(), shown.end());
    const std::vector<float> denoised = s.denoised_pixels();
    const denoise_call call = tap->last();
    ctx.exact("denoise-calls", tap->calls(), 1, 0);
    ctx.exact("denoised-size", static_cast<double>(denoised.size()), static_cast<double>(3 * pixels), 0);
    ctx.exact("denoise-size", call.width == width && call.height == height ? 1 : 0, 1, 0);
    std::int64_t alpha = 0;
    for (const float4& p : call.beauty)
    {
        alpha += p.w == 1.0f ? 0 : 1;
    }
    ctx.count("beauty-alpha", alpha, 0, static_cast<std::int64_t>(pixels));

    // The same state again with agx at +1 EV: the kept floats re-toned, no second denoise
    const gpu::display_options agx{.op = gpu::tone_map::agx, .exposure = 1};
    const std::span<const std::uint8_t> toned = s.display_denoised(agx);
    const std::vector<std::uint8_t> agx_bytes(toned.begin(), toned.end());
    ctx.exact("retone-without-denoise", tap->calls(), 1, 0);
    ctx.count("retone-keeps-floats", bit_differences(s.denoised_pixels(), denoised), 0, values_out_of);

    // The display at scale 1: the bytes of the denoised floats as doubles, over one band
    ctx.count("books-bytes-at-scale-1", bit_differences(books_bytes, bytes_at_scale_1(denoised, width, height, {})), 0,
              bytes_out_of);
    ctx.count("agx-bytes-at-scale-1", bit_differences(agx_bytes, bytes_at_scale_1(denoised, width, height, agx)), 0,
              bytes_out_of);

    // The beauty (the tap's "denoised" image): rows [0, split) as the image of 2 passes, the rest as that of 1
    s.render_to({2, 0});
    const std::vector<double> two = s.pixels();
    std::int64_t in_progress_wrong = 0;
    std::int64_t complete_wrong = 0;
    for (std::size_t i = 0; i < 3 * pixels && denoised.size() == 3 * pixels; ++i)
    {
        const bool top = static_cast<int>(i / (3 * static_cast<std::size_t>(width))) < split;
        const float expected = beauty_of(top ? two[i] : one[i], 1.0);
        (top ? in_progress_wrong : complete_wrong) += denoised[i] == expected ? 0 : 1;
    }
    const auto top_values = static_cast<std::int64_t>(3 * count_of(width, split));
    ctx.count("beauty-band-in-progress", in_progress_wrong, 0, top_values);
    ctx.count("beauty-band-complete", complete_wrong, 0, values_out_of - top_values);

    // The kept floats through other options after more passes (retone_denoised): the image denoised at (1, split)
    // again, the bytes display_denoised gave for each, and no denoise
    const auto copied = [](std::span<const std::uint8_t> bytes) {
        return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
    };
    ctx.count("retone-after-passes-agx", bit_differences(copied(s.retone_denoised(agx)), agx_bytes), 0, bytes_out_of);
    ctx.count("retone-after-passes-books", bit_differences(copied(s.retone_denoised({})), books_bytes), 0,
              bytes_out_of);
    ctx.exact("retone-after-passes-without-denoise", tap->calls(), 1, 0);
    ctx.exact("retone-options-differ", bit_differences(agx_bytes, books_bytes) > 0 ? 1 : 0, 1, 0);

    // A new state denoises again
    s.display_denoised(gpu::display_options{});
    ctx.exact("new-pass-denoises-again", tap->calls(), 2, 0);

    // A new view: the camera 0.5 to the right, so the balls move by about 4 pixels. No denoised image until its own
    // denoise; then guides of its own, a fresh session's for that view bit for bit (the same seed), none of the last
    // view's
    const denoise_call last_view = tap->last();
    gpu::camera_desc moved = scene.camera;
    moved.center.x += 0.5;
    moved.pixel00_loc.x += 0.5;
    s.set_camera(moved);
    s.render_to({1, 0});
    ctx.exact("new-view-no-denoised-pixels", throws<std::logic_error>([&] { (void)s.denoised_pixels(); }), 1, 0);
    ctx.exact("new-view-no-retone", throws<std::logic_error>([&] { (void)s.retone_denoised({}); }), 1, 0);
    s.display_denoised({});
    ctx.exact("new-view-denoises", tap->calls(), 3, 0);
    const denoise_call new_view = tap->last();
    denoise_call fresh_view;
    {
        gpu::scene_desc moved_scene = scene;
        moved_scene.camera = moved;
        gpu::session_options fresh_options = options;
        const auto fresh_tap = std::make_shared<guide_tap>(nullptr);
        fresh_options.optix = fresh_tap;
        gpu::session fresh(moved_scene, fresh_options);
        fresh.render_to({1, 0});
        fresh.display_denoised({});
        fresh_view = fresh_tap->last();
    }
    const auto pixels_out_of = static_cast<std::int64_t>(pixels);
    ctx.exact("new-view-guides-recomputed",
              bit_differences(new_view.albedo, last_view.albedo) + bit_differences(new_view.normal, last_view.normal) >
                      0
                  ? 1
                  : 0,
              1, 0);
    ctx.count("new-view-albedo-as-a-fresh-session", bit_differences(new_view.albedo, fresh_view.albedo), 0,
              pixels_out_of);
    ctx.count("new-view-normal-as-a-fresh-session", bit_differences(new_view.normal, fresh_view.normal), 0,
              pixels_out_of);

    // A larger view: the five buffers grow with it
    gpu::camera_desc larger = scene.camera;
    larger.image_width = 48;
    larger.image_height = 27;
    larger.pixel00_loc = {-1.175, 1.65, 0}; // the same pixels' size, the same center
    s.set_camera(larger);
    s.render_to({1, 0});
    s.display_denoised({});
    ctx.exact("larger-view-denoised",
              tap->last().width == 48 && tap->last().height == 27 && s.denoised_pixels().size() == 3 * count_of(48, 27)
                  ? 1
                  : 0,
              1, 0);

    // The guides' launches, counted where OptiX traces the session (optix_scene::guides): one per view, none for a
    // second denoise of the same view
    if (const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "optix-guide-launches"))
    {
        gpu_test::optix::excluded_seconds(ctx, [&runtime] { runtime->compile_programs(); });
        const auto counted = std::make_shared<guide_tap>(runtime);
        gpu::session_options traced = options;
        traced.optix_tracer = true;
        traced.optix = counted;
        gpu::session o(scene, traced);
        o.render_to({1, 0});
        o.display_denoised({});
        o.render_to({2, 0});
        o.display_denoised({});
        ctx.exact("optix-two-denoises", counted->calls(), 2, 0);
        ctx.exact("optix-guides-once-per-view", counted->guide_launches(), 1, 0);
        o.set_camera(moved);
        o.render_to({1, 0});
        o.display_denoised({});
        ctx.exact("optix-guides-again-for-a-new-view", counted->guide_launches(), 2, 0);
    }

    // The refusals (renderer.h): float64, and no runtime
    {
        gpu::session_options float64 = options;
        float64.arithmetic = gpu::precision::float64;
        float64.optix = std::make_shared<guide_tap>(nullptr);
        gpu::session s64(scene, float64);
        s64.render_to({1, 0});
        ctx.exact("float64-refused", throws<std::invalid_argument>([&] { s64.display_denoised({}); }), 1, 0);
        ctx.exact("float64-retone-refused", throws<std::invalid_argument>([&] { s64.retone_denoised({}); }), 1, 0);
        ctx.exact("float64-attach-refused",
                  throws<std::invalid_argument>([&] { s64.attach_optix(std::make_shared<guide_tap>(nullptr)); }), 1,
                  0);
    }

    // A session made without a runtime: refused, then given one by attach_optix (the image goes on), then
    // drop_denoised: nothing denoised kept, the runtime asked to release its denoiser, and the next display_denoised
    // denoises again
    {
        gpu::session_options bare;
        bare.seed = display_seed;
        gpu::session without(scene, bare);
        without.render_to({1, 0});
        ctx.exact("no-runtime-refused", throws<std::logic_error>([&] { without.display_denoised({}); }), 1, 0);
        ctx.exact("attach-none-refused", throws<std::invalid_argument>([&] { without.attach_optix(nullptr); }), 1, 0);
        const auto attached = std::make_shared<guide_tap>(nullptr);
        without.attach_optix(attached);
        without.attach_optix(attached); // the same runtime again: nothing happens
        ctx.exact("attach-keeps-the-image", without.state().passes == 1 ? 1 : 0, 1, 0);
        without.display_denoised({});
        ctx.exact("attach-then-denoise", attached->calls(), 1, 0);
        ctx.exact("attach-another-refused",
                  throws<std::logic_error>([&] { without.attach_optix(std::make_shared<guide_tap>(nullptr)); }), 1, 0);
        without.drop_denoised();
        ctx.exact("drop-releases-the-runtime", attached->releases(), 1, 0);
        ctx.exact("dropped-no-retone", throws<std::logic_error>([&] { (void)without.retone_denoised({}); }), 1, 0);
        ctx.exact("dropped-no-denoised-pixels", throws<std::logic_error>([&] { (void)without.denoised_pixels(); }), 1,
                  0);
        without.display_denoised({});
        ctx.exact("denoise-after-drop", attached->calls(), 2, 0);
    }

    // render() with denoise: the pixels and the rays as without it, the denoised floats display_denoised's
    {
        gpu::render_options plain;
        plain.seed = display_seed;
        plain.passes = 2;
        const gpu::render_result without = gpu::render(scene, plain);
        gpu::render_options with = plain;
        with.denoise = true;
        with.optix = std::make_shared<guide_tap>(nullptr);
        const gpu::render_result result = gpu::render(scene, with);
        ctx.exact("render-denoise-ran", result.denoise_error.empty() ? 1 : 0, 1, 0);
        if (!result.denoise_error.empty())
        {
            ctx.info("render-denoise-error", result.denoise_error);
        }
        ctx.count("render-pixels-unchanged", bit_differences(result.pixels, without.pixels), 0, values_out_of);
        ctx.exact("render-rays-unchanged", result.rays == without.rays ? 1 : 0, 1, 0);
        ctx.exact("render-denoised-rgba-size", static_cast<double>(result.denoised_rgba.size()),
                  static_cast<double>(4 * pixels), 0);
        std::int64_t wrong = static_cast<std::int64_t>(result.denoised.size() == 3 * pixels ? 0 : 3 * pixels);
        for (std::size_t i = 0; wrong == 0 && i < 3 * pixels; ++i)
        {
            wrong += result.denoised[i] == beauty_of(without.pixels[i], 1.0) ? 0 : 1;
        }
        ctx.count("render-denoised-is-beauty", wrong, 0, values_out_of);
        gpu::render_options orphan = plain;
        orphan.denoise = true;
        ctx.exact("render-denoise-needs-a-runtime", throws<std::exception>([&] { (void)gpu::render(scene, orphan); }),
                  1, 0);

        // A denoise that fails after the passes (a tap that throws, as a refused memory plan would): the render's
        // pixels kept as without denoise, nothing denoised, and the reason in denoise_error
        gpu::render_options failing = plain;
        failing.denoise = true;
        failing.optix = std::make_shared<guide_tap>(nullptr, "a planted failure of the denoiser");
        const gpu::render_result failed = gpu::render(scene, failing);
        ctx.count("failed-denoise-keeps-the-pixels", bit_differences(failed.pixels, without.pixels), 0,
                  values_out_of);
        ctx.exact("failed-denoise-nothing-denoised", failed.denoised.empty() && failed.denoised_rgba.empty() ? 1 : 0,
                  1, 0);
        ctx.exact("failed-denoise-says-why", failed.denoise_error == "a planted failure of the denoiser" ? 1 : 0, 1, 0);

        // An image too large for the denoiser's images (88 bytes a pixel past 1024 MiB) is refused before anything is
        // built: with a launch time the session would refuse before it allocates anything, a render that got past
        // the size would throw std::invalid_argument instead, so nothing of that size is ever allocated here
        gpu::scene_desc huge = scene;
        huge.camera.image_width = 8192;
        huge.camera.image_height = 4608;
        gpu::render_options oversized = with;
        oversized.launch_seconds = 0;
        ctx.exact("render-refuses-the-size-first",
                  throws<std::runtime_error>([&] { (void)gpu::render(huge, oversized); }), 1, 0);
    }
}

// D1: the first denoise of the process, and the memory the runtime takes for it (decision 6)
GPU_TEST(denoise, memory)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "memory");
    if (!runtime)
    {
        return;
    }
    constexpr std::array<std::array<int, 2>, 2> sizes{{{480, 270}, {960, 540}}};
    std::array<double, 2> taken{}; // MiB in all, after each size
    double total = 0;
    for (std::size_t s = 0; s < sizes.size(); ++s)
    {
        const int width = sizes[s][0];
        const int height = sizes[s][1];
        const std::size_t pixels = count_of(width, height);
        const denoise_images images(width, height, std::vector<float4>(pixels, make_float4(0.5f, 0.5f, 0.5f, 1.0f)),
                                    std::vector<float4>(pixels, make_float4(1.0f, 1.0f, 1.0f, 1.0f)),
                                    std::vector<float4>(pixels, make_float4(0.0f, 0.0f, 1.0f, 0.0f)));
        gpu_test::cuda_check(cudaDeviceSynchronize(), "waiting for the images");
        const std::size_t before = free_device_bytes();
        const auto start = std::chrono::steady_clock::now();
        images.denoise(*runtime);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const std::size_t after = free_device_bytes();
        total += (static_cast<double>(before) - static_cast<double>(after)) / mebibyte;
        taken[s] = total;
        const std::string size = std::format("{}x{}", width, height);
        if (s == 0)
        {
            ctx.info("first-denoise-seconds",
                     std::format("{:.3f} s: the process's first denoise, at {}: the denoiser made, set up and invoked "
                                 "once",
                                 seconds, size));
        }
        ctx.count("finite-" + size, not_finite(images.out()), 0, static_cast<std::int64_t>(3 * pixels));
        ctx.info("memory-" + size,
                 std::format("{:.1f} MiB: the GPU's free memory before the process's first denoise less after the "
                             "first at this size (other programs' allocations in between count too)",
                             total));
        // The runtime's own account of what it holds after this denoise
        const gpu::denoise_report report = runtime->last_denoise();
        ctx.exact("last-denoise-valid-" + size, report.valid ? 1 : 0, 1, 0);
        const std::string layout =
            report.tile > 0 ? std::format("tiles of {}", report.tile) : std::string("one invoke");
        ctx.info("denoiser-" + size,
                 std::format("state {:.1f} MiB, scratch {:.1f} MiB, {}",
                             static_cast<double>(report.state_bytes) / mebibyte,
                             static_cast<double>(report.scratch_bytes) / mebibyte, layout));
    }
    const double p1 = 480.0 * 270;
    const double p2 = 960.0 * 540;
    const double p3 = 1920.0 * 1080;
    const double estimate = taken[1] + (taken[1] - taken[0]) / (p2 - p1) * (p3 - p2);
    ctx.info("memory-1920x1080",
             std::format("about {:.1f} MiB, extrapolated linearly in the pixels from the two sizes above; not run: its "
                         "four float4 images alone would take {:.1f} MiB, beyond the suite's 64 MB",
                         estimate, 4 * p3 * 16 / mebibyte));
}

// D2: the synthetic case
GPU_TEST(denoise, synthetic)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "synthetic");
    if (!runtime)
    {
        return;
    }
    constexpr int width = 256;
    constexpr int height = 256;
    const std::size_t pixels = count_of(width, height);
    const synthetic_image image = make_synthetic(width, height, synthetic_seed);
    const denoise_images images(width, height, image.noisy,
                                std::vector<float4>(pixels, make_float4(1.0f, 1.0f, 1.0f, 1.0f)),
                                std::vector<float4>(pixels, make_float4(0.0f, 0.0f, 1.0f, 0.0f)));
    images.denoise(*runtime);
    const std::vector<float4> out = images.out();
    images.denoise(*runtime);
    const std::vector<float4> again = images.out();

    // The criterion, as functions of the image judged: its MSE against the clean image over the noisy image's (at most
    // 0.25: a gain of at least 4), and each channel's mean over the clean image's (within 1% of 1)
    const double noisy_mse = mse(image.noisy, image.clean);
    const std::array<double, 3> clean_means = channel_means(image.clean);
    const auto mse_ratio = [&](const std::vector<float4>& judged) { return mse(judged, image.clean) / noisy_mse; };
    const auto mean_ratios = [&](const std::vector<float4>& judged) {
        const std::array<double, 3> means = channel_means(judged);
        return std::array<double, 3>{means[0] / clean_means[0], means[1] / clean_means[1], means[2] / clean_means[2]};
    };
    const auto means_kept = [&](const std::vector<float4>& judged) {
        const std::array<double, 3> ratios = mean_ratios(judged);
        return std::ranges::all_of(ratios, [](double r) { return std::abs(r - 1) <= 0.01; }); // a NaN: false
    };
    const double ratio = mse_ratio(out);
    // A ratio is at least 0: within 0.25 of 0 is a gain of at least 4
    ctx.exact("mse-ratio", ratio, 0, 0.25);
    ctx.info("mse", std::format("{:.4g} noisy, {:.4g} denoised: a gain of {:.1f}", noisy_mse, ratio * noisy_mse,
                                1 / ratio));
    const std::array<double, 3> out_ratios = mean_ratios(out);
    for (std::size_t c = 0; c < 3; ++c)
    {
        ctx.exact(std::format("mean-{}", "rgb"[c]), out_ratios[c], 1, 0.01);
    }
    ctx.count("finite", not_finite(out), 0, static_cast<std::int64_t>(3 * pixels));
    std::int64_t alpha = 0;
    for (const float4& p : out)
    {
        alpha += p.w == 1.0f ? 0 : 1;
    }
    ctx.count("alpha-copied", alpha, 0, static_cast<std::int64_t>(pixels));

    // Planted: the same criterion on two images it must reject. They show that its thresholds can fail, and prove
    // nothing about the denoiser, which never sees them: the noisy image itself, what a denoise that returned its
    // input would give (an MSE ratio of exactly 1), and the denoised image made 3% brighter (whose means, within 1%
    // above, then lie at least 1.97% off)
    ctx.exact("planted-identity-rejected", mse_ratio(image.noisy) > 0.25 ? 1 : 0, 1, 0);
    std::vector<float4> brighter = out;
    for (float4& p : brighter)
    {
        p = make_float4(1.03f * p.x, 1.03f * p.y, 1.03f * p.z, p.w);
    }
    ctx.exact("planted-bright-rejected", means_kept(brighter) ? 0 : 1, 1, 0);

    // Two invokes of one input (INFO: denoise.ps1 decides on determinism, from denoise-eval's files)
    std::vector<float> first;
    std::vector<float> second;
    for (std::size_t k = 0; k < pixels; ++k)
    {
        first.insert(first.end(), {out[k].x, out[k].y, out[k].z, out[k].w});
        second.insert(second.end(), {again[k].x, again[k].y, again[k].z, again[k].w});
    }
    const std::int64_t differ = bit_differences(first, second);
    ctx.info("two-invokes",
             differ == 0 ? std::format("identical, all {} floats", first.size())
                         : std::format("{} of {} floats differ, the largest by {:.3g}", differ, first.size(),
                                       largest_difference(first, second)));
}

// D3: the tiled path against one invoke (decision 7)
GPU_TEST(denoise, tiled)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "tiled");
    if (!runtime)
    {
        return;
    }
    constexpr int side = 512;
    constexpr int tile = 128;
    const std::size_t pixels = count_of(side, side);
    const synthetic_image image = make_synthetic(side, side, tiled_seed);
    const denoise_images images(side, side, image.noisy,
                                std::vector<float4>(pixels, make_float4(1.0f, 1.0f, 1.0f, 1.0f)),
                                std::vector<float4>(pixels, make_float4(0.0f, 0.0f, 1.0f, 0.0f)));
    // The image in one invoke, then the same images in tiles of 128 (a window of 128 and the overlap on each side: a
    // crop of the image, where 256 and the overlap would see all of it and give one invoke's bytes)
    images.denoise(*runtime);
    const std::vector<float4> whole = images.out();
    const gpu::denoise_report whole_report = runtime->last_denoise();
    std::vector<float4> tiled;
    gpu::denoise_report tiled_report;
    {
        const tile_limit limit(*runtime, tile);
        images.denoise(*runtime);
        tiled = images.out();
        tiled_report = runtime->last_denoise();
    }
    // The runtime's own account of each layout
    ctx.exact("one-invoke", whole_report.valid && whole_report.tile == 0 ? 1 : 0, 1, 0);
    ctx.exact("in-tiles", tiled_report.valid && tiled_report.tile == tile ? 1 : 0, 1, 0);
    for (const auto& [name, report] : {std::pair<const char*, const gpu::denoise_report*>{"one-invoke", &whole_report},
                                       std::pair<const char*, const gpu::denoise_report*>{"tiles", &tiled_report}})
    {
        ctx.info(std::string("denoiser-") + name,
                 std::format("{}×{}: state {:.1f} MiB, scratch {:.1f} MiB, {}", side, side,
                             static_cast<double>(report->state_bytes) / mebibyte,
                             static_cast<double>(report->scratch_bytes) / mebibyte,
                             report->tile > 0 ? std::format("tiles of {}", report->tile) : std::string("one invoke")));
    }
    // The tiles denoise as D2's criterion asks of one invoke
    const double ratio = mse(tiled, image.clean) / mse(image.noisy, image.clean);
    ctx.exact("tiled-mse-ratio", ratio, 0, 0.25);
    ctx.count("tiled-finite", not_finite(tiled), 0, static_cast<std::int64_t>(3 * pixels));
    double sum = 0;
    double largest = 0;
    for (std::size_t k = 0; k < std::min(whole.size(), tiled.size()); ++k)
    {
        for (const double d : {std::abs(static_cast<double>(whole[k].x) - tiled[k].x),
                               std::abs(static_cast<double>(whole[k].y) - tiled[k].y),
                               std::abs(static_cast<double>(whole[k].z) - tiled[k].z)})
        {
            sum += d;
            largest = d == d ? std::max(largest, d) : std::numeric_limits<double>::infinity();
        }
    }
    ctx.info("tiled-against-one-invoke",
             std::format("mean |Δ| {:.3g}, largest {:.3g}, over all {}×{} pixels: tiles of {} against one invoke",
                         sum / static_cast<double>(3 * pixels), largest, side, side, tile));
}

// --- denoise-eval ----------------------------------------------------------------------------------------------------

// E1-E4: the evaluation's renders (decision 8)
GPU_TEST(denoise_eval, renders)
{
    const std::string* folder_argument = ctx.argument("--write");
    if (folder_argument == nullptr)
    {
        ctx.skip("renders", "the evaluation's renders need --write FOLDER (denoise.ps1 gives it)");
        return;
    }
    if (ctx.quick())
    {
        ctx.skip("renders", "the evaluation's renders never run under --quick");
        return;
    }
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "renders");
    if (!runtime)
    {
        return;
    }
    const std::filesystem::path folder = std::filesystem::absolute(*folder_argument);
    std::filesystem::create_directories(folder);
    ctx.info("folder", folder.string());
    gpu_test::optix::excluded_seconds(ctx, [&runtime] { runtime->compile_programs(); });
    constexpr int width = 480;
    constexpr int height = 270;

    // E1
    struct eval_scene
    {
        const char* name;
        bool optix; // traced by OptiX (a scene with meshes)
    };
    constexpr std::array<eval_scene, 4> scenes{
        {{"cornell", false}, {"studio", false}, {"final", false}, {"cornell-bust", true}}};
    std::optional<gpu::render_result> cornell_first; // E4's first
    for (const eval_scene& e : scenes)
    {
        // Every scene is there: one that can't be built (its model's files missing, say) throws, and the run stops with
        // the error, never a SKIP
        const gpu::scene_desc scene = scene_named(e.name, gpu_test::host::scene_strategy::mis, width, height, 16);
        for (const std::uint64_t seed : {std::uint64_t{1}, std::uint64_t{2}})
        {
            eval_render r =
                render_and_write(ctx, folder, std::format("{}-N{}", e.name, seed), scene, seed, e.optix, true, runtime);
            if (std::string_view(e.name) == "cornell" && seed == 1)
            {
                cornell_first = std::move(r.result);
            }
        }
    }

    // E2: the seconds of the references aren't the suite's (decision 8)
    const gpu::scene_desc reference =
        scene_named("cornell-bust", gpu_test::host::scene_strategy::mis, width, height, 2048);
    for (const std::uint64_t seed : {std::uint64_t{101}, std::uint64_t{102}})
    {
        const eval_render r = render_and_write(ctx, folder, std::format("cornell-bust-mis2048-s{}", seed), reference,
                                               seed, true, false, runtime);
        ctx.exclude_seconds(r.seconds);
    }

    // E3
    for (const char* name : {"cornell", "studio"})
    {
        const gpu::scene_desc scene = scene_named(name, gpu_test::host::scene_strategy::mis, width, height, 1024);
        const eval_render r =
            render_and_write(ctx, folder, std::format("{}-1024spp-N1", name), scene, 1, false, true, runtime);
        ctx.exclude_seconds(r.seconds);
    }

    // E4: cornell's N16 at seed 1 again, in the same process, written beside the first for denoise.ps1 to compare
    if (cornell_first)
    {
        const gpu::scene_desc scene = scene_named("cornell", gpu_test::host::scene_strategy::mis, width, height, 16);
        const gpu::render_result again =
            render_and_write(ctx, folder, "cornell-N1-again", scene, 1, false, true, runtime).result;
        const std::int64_t pixels_differ = bit_differences(again.pixels, cornell_first->pixels);
        ctx.info("render-determinism",
                 pixels_differ == 0
                     ? std::format("cornell-N1 rendered again: its pixels bit for bit the first's ({} values)",
                                   again.pixels.size())
                     : std::format("cornell-N1 rendered again: {} of {} values differ", pixels_differ,
                                   again.pixels.size()));
        const std::int64_t denoised_differ = bit_differences(again.denoised, cornell_first->denoised);
        ctx.info("denoise-determinism",
                 denoised_differ == 0
                     ? std::format("cornell-N1 denoised again: bit for bit the first's ({} floats)",
                                   again.denoised.size())
                     : std::format("cornell-N1 denoised again: {} of {} floats differ, the largest by {:.3g}",
                                   denoised_differ, again.denoised.size(),
                                   largest_difference(again.denoised, cornell_first->denoised)));
    }
}
