// The OptiX suites of track A: optix-runtime (Steps 87 and 88), optix-tracer (Step 88) and optix-render (Step 89),
// in that order below. Everything goes through the abstract runtime of the fixture (optix_fixture.h) and the
// renderer's interface, so a build without OptiX links this file as it is and every OptiX check skips, saying why.
//
// Suite optix-runtime: OptiX started from the graphics driver, the programs compiled by it, and launches of the check
// module's probes (tests/optix_check_programs.cu), first before any scene or acceleration structure (Step 87's probes:
// compile_programs() and the scene-less launch_test), then through an upload (P4 and P5, Step 88; P7, with the
// instances, is track B's, in Step 91).
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
//       (what replaced the BVH's 256-byte stack);
//   P4  a one-quad custom GAS (an upload) traced with tmax FLT_MAX, as optix_tracer passes it: the hit and the miss
//       exactly as the books' quad test gives them; then with +∞, which the tracer never passes (INFO: what OptiX
//       does with it);
//   P5  a scene without any of the books' primitives (an empty custom build input): its upload doesn't throw, and
//       nothing is hit (INFO: whether its root is 0, as planned: no GAS at all).
//
// Suite optix-tracer: __raygen__check, the two tracers ray by ray on the geometry of the 22 GPU scenes, and its three
// planted faults; from Step 91 the same under an instance root. Suite optix-render: gpu::render with the quality
// kernel and with OptiX on the same seed, the furnaces, determinism, the session's refusals, and a view rendered in
// three calls. Both are left out of the every-step GT run; their comments below say what they decide.
#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "ggx_energy.cuh"
#include "gpu_test.cuh"
#include "gpu_test.h"
#include "host_services.h"
#include "optix/embedded.h"
#include "optix_fixture.h"
#include "optix_runtime.cuh"
#include "quality.cuh"
#include "renderer.h"
#include "start_optix.h"
#include "test_scene.h"
#include "tracer_check_layout.h"

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

// --- What P4, P5 and the tracer check share: a scene in both tracers' memory -----------------------------------------

namespace
{
// The 22 GPU scenes the quality kernel renders, in gpu_scenes' order: the books' 13 (scenes.ixx), then milestone 4's 9
// (gpu_scenes.ixx). Named here rather than taken from gpu_scene_list(): the scenes with triangle meshes join that list
// from Step 91, and they are optix-triangles' to check.
constexpr std::array<std::string_view, 22> gpu_scene_names{
    // the books'
    "materials", "weekend", "bouncing", "checker", "earth", "perlin", "quads", "light", "cornell", "smoke", "final",
    "aluminum", "glass",
    // beyond the books
    "furnace", "ggx-furnace", "gloss-furnace", "cornell-ggx", "veach", "env-furnace", "studio", "sunlit", "sun-test"};

bool listed(std::span<const std::string_view> names, std::string_view name)
{
    return std::ranges::find(names, name) != names.end();
}

// renderer.cu's to_camera, in float (a function of that file's own): the pixel grid relative to the center is
// subtracted in double before it is rounded, as camera::get_ray needs it
gpu::camera<float> camera_of(const gpu::camera_desc& desc)
{
    const gpu::vec3d center_to_pixel00{.x = desc.pixel00_loc.x - desc.center.x,
                                       .y = desc.pixel00_loc.y - desc.center.y,
                                       .z = desc.pixel00_loc.z - desc.center.z};
    return {.image_width = desc.image_width,
            .image_height = desc.image_height,
            .sqrt_spp = desc.sqrt_spp,
            .max_depth = desc.max_depth,
            .recip_sqrt_spp = static_cast<float>(1.0 / desc.sqrt_spp),
            .center = gpu::to_real<float>(desc.center),
            .center_to_pixel00 = gpu::to_real<float>(center_to_pixel00),
            .pixel_delta_u = gpu::to_real<float>(desc.pixel_delta_u),
            .pixel_delta_v = gpu::to_real<float>(desc.pixel_delta_v),
            .defocus = desc.defocus,
            .defocus_disk_u = gpu::to_real<float>(desc.defocus_disk_u),
            .defocus_disk_v = gpu::to_real<float>(desc.defocus_disk_v),
            .sky_bottom = gpu::to_real<float>(desc.sky_bottom),
            .sky_top = gpu::to_real<float>(desc.sky_top)};
}

// A scene as both tracers see it: the BVH and the books' tables (tests/test_scene.h, with the renderer's own builder),
// the quality kernel's arrays beside them (the lamps, the GGX side entries, the environment map), and OptiX's
// acceleration structures over the same primitives, placed by the BVH's global ids. Made in this order, freed in the
// reverse one.
struct uploaded_scene
{
    gpu_test::device_scene base;
    gpu::detail::quality_scene_ptr quality;
    std::unique_ptr<gpu::detail::optix_scene> optix;

    uploaded_scene(gpu::optix_runtime& runtime, const gpu::scene_desc& scene, const gpu::optix_upload_options& options)
        : base(scene),
          quality(gpu::detail::upload_quality(scene, base.sphere_ids(), base.quad_ids())),
          optix(runtime.upload(scene, base.view(), *quality, base.sphere_ids(), base.quad_ids(), options))
    {
    }
};

// The render module and its pipeline, which the first upload makes if no suite made them before: made here, outside
// the budget (the suite that runs first in a process pays for the others), so that an upload's seconds are its own
void compile_render_programs(gpu_test::context& ctx, gpu::optix_runtime& runtime)
{
    gpu_test::optix::excluded_seconds(ctx, [&runtime] { runtime.compile_programs(); });
}

// A ray of __raygen__check_ray (tests/optix_check_programs.cu), and what it met there
struct probe_ray
{
    std::array<float, 3> origin;
    std::array<float, 3> direction;
    float tmax;
};

struct probe_hit
{
    float met; // 1 a hit, 0 a miss, -1 nothing traced (the scene's root is 0)
    float t;
    float front;     // the low bit of the hit kind: a quad's front, a sphere entered
    float primitive; // the hit's primitive index: the books' global id
};

std::vector<probe_hit> trace_probe_rays(gpu::detail::optix_scene& scene, std::span<const probe_ray> rays)
{
    std::vector<float> slots(12 * rays.size(), 0.0f);
    for (std::size_t k = 0; k < rays.size(); ++k)
    {
        float* const slot = slots.data() + 12 * k;
        std::ranges::copy(rays[k].origin, slot);
        std::ranges::copy(rays[k].direction, slot + 3);
        slot[6] = rays[k].tmax;
        slot[7] = -1; // no primitive to skip
    }
    const gpu_test::device_buffer<float> values(slots);
    scene.launch_test(check_module(), "__raygen__check_ray", static_cast<int>(rays.size()), 1, gpu::camera<float>{},
                      {.values = values.get()});
    gpu_test::finish_launch("a launch of __raygen__check_ray");
    const std::vector<float> out = values.download();
    std::vector<probe_hit> hits;
    for (std::size_t k = 0; k < rays.size(); ++k)
    {
        const float* const slot = out.data() + 12 * k;
        hits.push_back({.met = slot[8], .t = slot[9], .front = slot[10], .primitive = slot[11]});
    }
    return hits;
}

// The probes' scene: the given quads, matte, under a white sky, and a 4 × 4 camera that the probes never use
gpu::scene_desc probe_scene(std::vector<gpu::quad_desc> quads)
{
    gpu::scene_desc scene;
    scene.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = {0.5, 0.5, 0.5}});
    scene.materials.push_back({.kind = gpu::material_kind::lambertian, .texture = 0});
    scene.quads = std::move(quads);
    scene.camera = {.image_width = 4,
                    .image_height = 4,
                    .sqrt_spp = 1,
                    .max_depth = 4,
                    .center = {0, 0, 5},
                    .pixel00_loc = {-0.375, 0.375, 4},
                    .pixel_delta_u = {0.25, 0, 0},
                    .pixel_delta_v = {0, -0.25, 0},
                    .sky_bottom = {1, 1, 1},
                    .sky_top = {1, 1, 1},
                    .sampling = gpu::sampling_strategy::material};
    return scene;
}

// From (0, 0, 5) toward the probe quad and away from it, with one tmax
std::array<probe_ray, 2> probe_rays(float tmax)
{
    return {probe_ray{.origin = {0, 0, 5}, .direction = {0, 0, -1}, .tmax = tmax},
            probe_ray{.origin = {0, 0, 5}, .direction = {0, 0, 1}, .tmax = tmax}};
}

// P4: the quad z = 0, |x| ≤ 1, |y| ≤ 1 (u × v along +z: its front), the only primitive of its GAS. From (0, 0, 5)
// toward it: a hit at t = 5 exactly (hit_quad's n·(q − o)/(n·d) = −5/−1), primitive 0, its front; away from it: a miss.
// With tmax FLT_MAX, as optix_tracer passes every tmax, these decide; with +∞ the same rays are INFO, since the
// tracer never passes it.
GPU_TEST(optix_runtime, tmax)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "tmax");
    if (!runtime)
    {
        return;
    }
    compile_render_programs(ctx, *runtime);
    const uploaded_scene scene(
        *runtime, probe_scene({gpu::quad_desc{.q = {-1, -1, 0}, .u = {2, 0, 0}, .v = {0, 2, 0}, .material = 0}}), {});
    std::vector<probe_hit> finite;
    const double seconds = gpu_test::optix::excluded_seconds(
        ctx, [&] { finite = trace_probe_rays(*scene.optix, probe_rays(std::numeric_limits<float>::max())); });
    ctx.info("check-ray-pipeline-seconds",
             std::format("{:.3f} s: the first launch of __raygen__check_ray in this test (its pipeline, once per "
                         "runtime), outside budget-seconds",
                         seconds));
    ctx.exact("tmax-flt-max-hit", finite[0].met, 1, 0);
    ctx.exact("tmax-flt-max-hit-t", finite[0].t, 5, 0);
    ctx.exact("tmax-flt-max-hit-front", finite[0].front, 1, 0);
    ctx.exact("tmax-flt-max-hit-primitive", finite[0].primitive, 0, 0);
    ctx.exact("tmax-flt-max-miss", finite[1].met, 0, 0);

    const std::vector<probe_hit> infinite =
        trace_probe_rays(*scene.optix, probe_rays(std::numeric_limits<float>::infinity()));
    const bool alike = infinite[0].met == 1 && infinite[0].t == 5 && infinite[0].front == 1 &&
                       infinite[0].primitive == 0 && infinite[1].met == 0;
    ctx.info("tmax-infinity",
             alike ? std::string("accepted: the same hit at t = 5 and the same miss as with FLT_MAX (the tracer passes "
                                 "FLT_MAX all the same)")
                   : std::format("not as with FLT_MAX: toward the quad {} (t = {}, front {}, primitive {}), away {} "
                                 "(the tracer passes FLT_MAX, never +inf)",
                                 infinite[0].met, infinite[0].t, infinite[0].front, infinite[0].primitive,
                                 infinite[1].met));
}

// P5: a scene of no sphere and no quad, whose custom build input would hold no primitive. The upload builds no GAS
// for it and leaves the root 0, so that closest() misses without tracing; whatever the runtime does, its upload must
// not throw, and the probe rays then meet nothing.
GPU_TEST(optix_runtime, no_primitives)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "no-primitives");
    if (!runtime)
    {
        return;
    }
    compile_render_programs(ctx, *runtime);
    std::unique_ptr<uploaded_scene> scene;
    try
    {
        scene = std::make_unique<uploaded_scene>(*runtime, probe_scene({}), gpu::optix_upload_options{});
    }
    catch (const std::exception& error)
    {
        ctx.fail("no-primitives-upload", std::format("the upload threw: {}", error.what()));
        return;
    }
    ctx.exact("no-primitives-upload", 1, 1, 0); // it returned
    std::vector<probe_hit> hits;
    gpu_test::optix::excluded_seconds(
        ctx, [&] { hits = trace_probe_rays(*scene->optix, probe_rays(std::numeric_limits<float>::max())); });
    ctx.info("no-primitives-root", hits[0].met < 0 ? "0: no GAS, nothing traced (as planned)"
                                                   : "a traversable handle: OptiX traced the rays through it");
    ctx.exact("no-primitives-nothing-hit", (hits[0].met > 0 ? 1 : 0) + (hits[1].met > 0 ? 1 : 0), 0, 0);
}
}

// --- Suite optix-tracer (Step 88): the OptiX tracer against the BVH's, ray by ray ------------------------------------
//
// Each of the 22 GPU scenes is made geometry-only (host_services.h's geometry_only: textures folded to solid colors,
// images and the lights list dropped; every primitive, medium, lamp and map kept), uploaded as test_scene.h uploads a
// scene (the BVH is the renderer's own) and as OptiX's runtime uploads it, and __raygen__check follows 160 × 90 pixels
// × 4 samples, seed 9001, sorting every closest() into same, ulp and mismatch (tests/optix_check_programs.cu says how).
// Per scene:
//   <scene>.camera-rays             every sample started: width × height × samples first queries, exactly;
//   <scene>.mismatches              at most 1e-4 of the closest() queries;
//   <scene>.visibility-differences  at most 1e-4 of the shadow queries;
// and an INFO line with every tally: the ulp category's largest |Δt|/t, the ties at equal t among the mismatches
// (coplanar faces, such as final's ground boxes, whose order the two traversals may decide differently), and the
// first mismatch and visibility difference, the diagnosis owed before any threshold is touched (a threshold
// changes only with the user's OK). The planted faults bound the thresholds from the other side. Both tracers come
// from one compile, and the thresholds are shares of integers summed in any order: no driver moves them.
//
// --quick (compute-sanitizer's run): cornell, smoke, final and studio, which hold every kind of primitive, medium, lamp
// and map, at 48 × 27 × 1 sample. The shares then rest on a few thousand queries, too few for 1e-4, and are printed
// without a verdict; the camera rays and the planted faults still decide.

namespace
{
struct check_size
{
    int width;
    int height;
    int samples; // per pixel: a square, so that they cover the camera's grid of strata once
};
constexpr check_size full_check{.width = 160, .height = 90, .samples = 4};
constexpr check_size quick_check{.width = 48, .height = 27, .samples = 1};
constexpr std::array<std::string_view, 4> quick_check_scenes{"cornell", "smoke", "final", "studio"};

constexpr std::uint64_t check_seed = 9001;
constexpr std::uint64_t fault_seed = 9002;
constexpr std::uint64_t instance_seed = 9003;

// A registered scene at the check's size, made geometry-only
gpu::scene_desc geometry_scene(std::string_view name, const check_size& size)
{
    return gpu_test::host::geometry_only(gpu_test::host::built_scene(
        name, {.width = size.width, .height = size.height, .samples = size.samples}));
}

struct tracer_tally
{
    std::vector<unsigned long long> counts; // by tracer_check::counter
    std::vector<float> values;              // by tracer_check::value

    std::int64_t operator[](tracer_check::counter k) const { return static_cast<std::int64_t>(counts[k]); }
};

// One launch of __raygen__check over the camera's pixels, `samples` each
tracer_tally run_check(const uploaded_scene& scene, const gpu::camera<float>& cam, int samples, int fault,
                       std::uint64_t seed)
{
    const gpu_test::device_buffer<unsigned long long> counts(tracer_check::counter_count);
    const gpu_test::device_buffer<float> values(tracer_check::value_count);
    scene.optix->launch_test(check_module(), "__raygen__check", cam.image_width, cam.image_height, cam,
                             {.counts = counts.get(),
                              .values = values.get(),
                              .fault = fault,
                              .samples = samples,
                              .seed = seed});
    gpu_test::finish_launch("a launch of __raygen__check");
    return {.counts = counts.download(), .values = values.download()};
}

// The check pipeline (__raygen__check with the render module's intersection programs), made by its first launch: one
// of no samples, which traces nothing, outside the budget. INFO: its seconds, cold in the first process after the
// check module's PTX or the driver changed, warm (OptiX's disk cache) after that
void make_check_pipeline(gpu_test::context& ctx, const uploaded_scene& scene, const gpu::camera<float>& cam)
{
    const gpu_test::device_buffer<unsigned long long> counts(tracer_check::counter_count);
    const gpu_test::device_buffer<float> values(tracer_check::value_count);
    const compile_seconds made = timed_compile(ctx, [&] {
        scene.optix->launch_test(check_module(), "__raygen__check", 1, 1, cam,
                                 {.counts = counts.get(), .values = values.get(), .samples = 0});
        gpu_test::finish_launch("the first launch of __raygen__check");
    });
    ctx.info("check-pipeline-seconds",
             std::format("{:.3f} s, {:.3f} s of CPU: the first launch of __raygen__check (the check module and its "
                         "pipeline, if no suite made them before), outside budget-seconds",
                         made.wall, made.cpu));
}

std::string_view yes_no(float flag, std::string_view yes, std::string_view no)
{
    return flag != 0 ? yes : no;
}

// The first mismatch a launch wrote, in words
std::string describe_mismatch(const std::vector<float>& values)
{
    const float* const e = values.data() + tracer_check::mismatch_example;
    return std::format("pixel ({}, {}) sample {} vertex {}, from {} {}: the BVH met {} at t = {:.9g} ({}), OptiX {} at "
                       "t = {:.9g} ({}); the random numbers left {}",
                       e[0], e[1], e[2], e[3], e[4], yes_no(e[5], "inward", "outward"), e[6], e[7],
                       yes_no(e[8], "entering", "not entering"), e[9], e[10],
                       yes_no(e[11], "entering", "not entering"), yes_no(e[12], "equal", "differ"));
}

// The first visibility difference a launch wrote, in words
std::string describe_visibility(const std::vector<float>& values)
{
    const float* const e = values.data() + tracer_check::visibility_example;
    return std::format("pixel ({}, {}) sample {} vertex {}, from {} {}, skip {}, t_max {:.9g}: the BVH's visibility "
                       "{:.9g}, OptiX's {:.9g}",
                       e[0], e[1], e[2], e[3], e[4], yes_no(e[5], "inward", "outward"), e[6], e[7], e[8], e[9]);
}

// One scene's verdicts and INFO lines (name: the scene, or instance-root.<scene>)
void report_check(gpu_test::context& ctx, std::string_view name, const tracer_tally& t, const gpu::camera<float>& cam,
                  int samples, bool verdicts)
{
    using namespace tracer_check;
    ctx.info(name, std::format("{} closest queries: {} same, {} ulp (largest |dt|/t {:.3g}), {} mismatches ({} ties at "
                               "equal t, {} only in the random numbers left); {} shadow queries, {} differ. The BVH "
                               "found {} escapes, {} particles, {} rays into their own sphere, {} shadow rays blocked",
                               t[closest_queries], t[same], t[ulp], t.values[largest_ulp_ratio], t[mismatches],
                               t[ties], t[numbers_only], t[shadow_queries], t[visibility_differences], t[escaped],
                               t[particles], t[own_sphere_inward], t[shadows_blocked]));
    if (t[mismatches] > 0)
    {
        ctx.info(std::format("{}.first-mismatch", name), describe_mismatch(t.values));
    }
    if (t[visibility_differences] > 0)
    {
        ctx.info(std::format("{}.first-visibility-difference", name), describe_visibility(t.values));
    }
    ctx.exact(std::format("{}.camera-rays", name), static_cast<double>(t[camera_rays]),
              static_cast<double>(cam.image_width) * cam.image_height * samples, 0);
    if (t[shadow_queries] == 0)
    {
        ctx.fail(std::format("{}.shadow-queries", name), "no shadow query was traced: the check would pass on nothing");
    }
    if (verdicts)
    {
        // At most 1e-4 of the queries: the integer part of n/10^4
        ctx.count(std::format("{}.mismatches", name), t[mismatches], t[closest_queries] / 10000, t[closest_queries]);
        ctx.count(std::format("{}.visibility-differences", name), t[visibility_differences],
                  t[shadow_queries] / 10000, t[shadow_queries]);
    }
    else
    {
        const auto share = [](std::int64_t part, std::int64_t whole) {
            return whole > 0 ? static_cast<double>(part) / static_cast<double>(whole) : 0.0;
        };
        ctx.info(std::format("{}.shares", name),
                 std::format("{:.3g} of the closest queries mismatch, {:.3g} of the shadow queries differ (--quick: "
                             "no verdict on a share of 1e-4 from so few)",
                             share(t[mismatches], t[closest_queries]),
                             share(t[visibility_differences], t[shadow_queries])));
    }
}

GPU_TEST(optix_tracer, scenes)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "scenes");
    if (!runtime)
    {
        return;
    }
    compile_render_programs(ctx, *runtime);
    const check_size size = ctx.quick() ? quick_check : full_check;
    bool pipeline_made = false;
    for (const std::string_view name : gpu_scene_names)
    {
        if (ctx.quick() && !listed(quick_check_scenes, name))
        {
            continue;
        }
        const gpu::scene_desc scene = geometry_scene(name, size);
        const uploaded_scene uploaded(*runtime, scene, {});
        const gpu::camera<float> cam = camera_of(scene.camera);
        if (!pipeline_made)
        {
            make_check_pipeline(ctx, uploaded, cam);
            pipeline_made = true;
        }
        report_check(ctx, name, run_check(uploaded, cam, size.samples, 0, check_seed), cam, size.samples,
                     !ctx.quick());
    }
}

// The three planted faults (test.fault: on OptiX's side of __raygen__check only), each of which the check must see in
// more than 1e-2 of its queries, 100 times the thresholds above: 1 the hit kind read inverted, on cornell (every quad
// hit then has the other side); 2 the media loop skipped, on smoke (its two boxes of smoke); 3 the shadow rays' skip
// passed as -1, on cornell (a ray aimed at the lamp may then meet the lamp itself, at the end of its interval). A
// fault the check doesn't see means thresholds that can't see one either. Under --quick too: 1e-2 of a few thousand
// queries is still dozens.
GPU_TEST(optix_tracer, planted_faults)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "planted-faults");
    if (!runtime)
    {
        return;
    }
    compile_render_programs(ctx, *runtime);
    struct planted
    {
        int fault;
        std::string_view scene;
        std::string_view what;
        bool shadow_rays; // seen by visibility(), else by closest()
    };
    constexpr std::array<planted, 3> faults{{{1, "cornell", "hit-kind-inverted", false},
                                             {2, "smoke", "media-loop-skipped", false},
                                             {3, "cornell", "skip-dropped", true}}};
    const check_size size = ctx.quick() ? quick_check : full_check;
    for (const planted& f : faults)
    {
        const gpu::scene_desc scene = geometry_scene(f.scene, size);
        const uploaded_scene uploaded(*runtime, scene, {});
        const gpu::camera<float> cam = camera_of(scene.camera);
        const tracer_tally t = run_check(uploaded, cam, size.samples, f.fault, fault_seed);
        const std::int64_t seen = f.shadow_rays ? t[tracer_check::visibility_differences] : t[tracer_check::mismatches];
        const std::int64_t out_of = f.shadow_rays ? t[tracer_check::shadow_queries] : t[tracer_check::closest_queries];
        const std::string name = std::format("fault-{}-{}.{}", f.fault, f.what, f.scene);
        ctx.info(name, std::format("{} of {} {} queries see it ({:.3g}; more than 1e-2 must)", seen, out_of,
                                   f.shadow_rays ? "shadow" : "closest",
                                   out_of > 0 ? static_cast<double>(seen) / static_cast<double>(out_of) : 0.0));
        // Seen in more than 1/100 of the queries: seen ≥ ⌊n/100⌋ + 1, so the queries it left unseen are at most
        // n − ⌊n/100⌋ − 1 (and none at all may pass when there was no query)
        ctx.count(name + ".unseen", out_of - seen, out_of - out_of / 100 - 1, out_of);
    }
}

// From Step 91: the check with the books' GAS under an identity instance (optix_upload_options::force_instance_root),
// the root of every scene with meshes, on cornell, final and studio. The instance must change nothing, so the same
// thresholds hold (and the intersection programs read the world ray under it: P7, suite optix-triangles). An upload
// that fails is the suite's failure: no skip.
GPU_TEST(optix_tracer, instance_root)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "instance-root");
    if (!runtime)
    {
        return;
    }
    compile_render_programs(ctx, *runtime);
    const check_size size = ctx.quick() ? quick_check : full_check;
    for (const std::string_view name : std::array<std::string_view, 3>{"cornell", "final", "studio"})
    {
        const gpu::scene_desc scene = geometry_scene(name, size);
        const auto uploaded = std::make_unique<uploaded_scene>(*runtime, scene,
                                                               gpu::optix_upload_options{.force_instance_root = true});
        const gpu::camera<float> cam = camera_of(scene.camera);
        report_check(ctx, std::format("instance-root.{}", name),
                     run_check(*uploaded, cam, size.samples, 0, instance_seed), cam, size.samples, !ctx.quick());
    }
}
}

// --- Suite optix-render (Step 89): the same seed through both renderers ----------------------------------------------
//
// gpu::render, the library's own path, with the quality kernel and with OptiX (render_options::optix_tracer and the
// fixture's runtime) on each of the 22 GPU scenes at 240 × 135, 16 samples, seed 7 (the same-seed comparisons' seed),
// with the scene's strategy for light: -p mis where it has a lamp of the light set or an environment map, else
// -p material (gpu_scene's rule for -p mis). Both run one integrator (quality_path.cuh) with the same random numbers,
// so a pixel differs only where a path met another primitive or another t. Per scene (test survey):
//   INFO                  the share of the pixels equal bit for bit, the largest distance between a channel's two
//                         values in float steps (ulps), the rays and shadow rays of both;
//   <scene>.kernels       the quality kernel rendered the one, OptiX the other (the comparison is never vacuous);
//   <scene>.not-finite    no NaN and no infinity in OptiX's image;
//   <scene>.rays          OptiX's rays within 0.5% of the quality kernel's;
//   <scene>.shadow-rays   the same for the shadow rays (exactly the same 0 under -p material).
// The other tests: furnaces (furnace and env-furnace under OptiX with -p material: every value exactly 1.0, as for the
// CUDA kernels; under -p mis env-furnace is 1 on average only, docs/quality.md), determinism (OptiX twice on cornell,
// final and studio in this process: the same bytes; across processes is Step 89's own check, with RayTracingGPU),
// refusals (the session's, before anything renders), three_calls (a view rendered by three calls of render_to, the
// second ending inside a pass, equals one render of as many passes, bit for bit) and planted_faults (what the exact
// gates must reject).
//
// --quick: 64 × 36 pixels, 4 samples, and five scenes: cornell, final, studio, furnace and env-furnace.

namespace
{
struct survey_size
{
    int width;
    int height;
    int samples; // a square: a whole round of strata
};
constexpr survey_size full_survey{.width = 240, .height = 135, .samples = 16};
constexpr survey_size quick_survey{.width = 64, .height = 36, .samples = 4};
constexpr std::array<std::string_view, 5> quick_survey_scenes{"cornell", "final", "studio", "furnace", "env-furnace"};

constexpr std::uint64_t survey_seed = 7; // the milestone's seed of the same-seed comparisons
constexpr std::uint64_t three_calls_seed = 9004;
constexpr std::uint64_t planted_seed = 9005;

gpu::scene_desc survey_scene(std::string_view name, const survey_size& size, gpu_test::host::scene_strategy sampling)
{
    return gpu_test::host::built_scene(
        name, {.width = size.width, .height = size.height, .samples = size.samples, .sampling = sampling});
}

gpu::render_result render_quality(const gpu::scene_desc& scene, std::uint64_t seed)
{
    return gpu::render(scene, {.seed = seed, .quality_kernel = true});
}

gpu::render_result render_optix(const gpu::scene_desc& scene, std::uint64_t seed,
                                const std::shared_ptr<gpu::optix_runtime>& runtime)
{
    return gpu::render(scene, {.seed = seed, .optix_tracer = true, .optix = runtime});
}

// A float's place on the number line in steps of one ulp: neighbouring floats differ by 1, -0 and +0 are both 0
std::int64_t float_steps(float x)
{
    const auto bits = std::bit_cast<std::int32_t>(x);
    return bits >= 0 ? bits : -static_cast<std::int64_t>(bits & 0x7fffffff);
}

struct image_comparison
{
    std::int64_t pixels = 0;
    std::int64_t identical = 0;     // pixels whose three values are equal bit for bit
    std::int64_t values_differ = 0; // values (channels) that aren't
    std::int64_t largest_steps = 0; // the largest distance of two finite values, rounded to float, in float steps
};

image_comparison compare(std::span<const double> a, std::span<const double> b)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error(std::format("two images of {} and {} values", a.size(), b.size()));
    }
    image_comparison c{.pixels = static_cast<std::int64_t>(a.size() / 3)};
    for (std::size_t p = 0; p + 2 < a.size(); p += 3)
    {
        bool equal = true;
        for (std::size_t k = p; k < p + 3; ++k)
        {
            if (std::bit_cast<std::uint64_t>(a[k]) == std::bit_cast<std::uint64_t>(b[k]))
            {
                continue;
            }
            equal = false;
            ++c.values_differ;
            if (std::isfinite(a[k]) && std::isfinite(b[k]))
            {
                const std::int64_t steps =
                    float_steps(static_cast<float>(a[k])) - float_steps(static_cast<float>(b[k]));
                c.largest_steps = std::max(c.largest_steps, steps < 0 ? -steps : steps);
            }
        }
        c.identical += equal ? 1 : 0;
    }
    return c;
}

std::int64_t not_finite(std::span<const double> values)
{
    return std::ranges::count_if(values, [](double v) { return !std::isfinite(v); });
}

double percent(std::int64_t part, std::int64_t whole)
{
    return whole > 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
}

GPU_TEST(optix_render, survey)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "survey");
    if (!runtime)
    {
        return;
    }
    compile_render_programs(ctx, *runtime);
    const survey_size size = ctx.quick() ? quick_survey : full_survey;
    std::int64_t all_pixels = 0;
    std::int64_t all_identical = 0;
    for (const std::string_view name : gpu_scene_names)
    {
        if (ctx.quick() && !listed(quick_survey_scenes, name))
        {
            continue;
        }
        const gpu::scene_desc scene = survey_scene(name, size, gpu_test::host::scene_strategy::mis);
        const bool mis = scene.quality.combination == gpu::light_combination::power_heuristic;
        const gpu::render_result quality = render_quality(scene, survey_seed);
        const gpu::render_result optix = render_optix(scene, survey_seed, runtime);
        const image_comparison c = compare(quality.pixels, optix.pixels);
        all_pixels += c.pixels;
        all_identical += c.identical;
        ctx.info(name, std::format("-p {}: {:.2f}% of the pixels equal bit for bit, a channel at most {} float steps "
                                   "apart; rays {} (quality) and {} (OptiX), shadow rays {} and {}",
                                   mis ? "mis" : "material", percent(c.identical, c.pixels), c.largest_steps,
                                   quality.rays, optix.rays, quality.shadow_rays, optix.shadow_rays));
        if (const std::int64_t bad = not_finite(quality.pixels); bad > 0)
        {
            ctx.info(std::format("{}.quality-not-finite", name),
                     std::format("{} values of the quality kernel's image aren't finite", bad));
        }
        ctx.exact(std::format("{}.kernels", name),
                  quality.kernel.name == "quality" && optix.kernel.name == "optix" ? 1 : 0, 1, 0);
        ctx.count(std::format("{}.not-finite", name), not_finite(optix.pixels), 0,
                  static_cast<std::int64_t>(optix.pixels.size()));
        ctx.exact(std::format("{}.rays", name), static_cast<double>(optix.rays), static_cast<double>(quality.rays),
                  0.005 * static_cast<double>(quality.rays));
        ctx.exact(std::format("{}.shadow-rays", name), static_cast<double>(optix.shadow_rays),
                  static_cast<double>(quality.shadow_rays), 0.005 * static_cast<double>(quality.shadow_rays));
    }
    ctx.info("all-scenes",
             std::format("{:.2f}% of {} pixels equal bit for bit", percent(all_identical, all_pixels), all_pixels));
}

// furnace and env-furnace under OptiX with -p material: a white sphere under a white sky or map, every path ending in
// the sky with all of its light, so every value is exactly 1.0 (gpu_scenes.ixx says why in float too)
GPU_TEST(optix_render, furnaces)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "furnaces");
    if (!runtime)
    {
        return;
    }
    compile_render_programs(ctx, *runtime);
    const survey_size size = ctx.quick() ? quick_survey : full_survey;
    for (const std::string_view name : std::array<std::string_view, 2>{"furnace", "env-furnace"})
    {
        const gpu::scene_desc scene = survey_scene(name, size, gpu_test::host::scene_strategy::material);
        const gpu::render_result r = render_optix(scene, survey_seed, runtime);
        ctx.count(std::format("{}.not-one", name), std::ranges::count_if(r.pixels, [](double v) { return v != 1.0; }),
                  0, static_cast<std::int64_t>(r.pixels.size()));
    }
}

// OptiX twice on the same scene, seed and options in one process (one GAS build per render): the same bytes, and the
// same rays. If they differ, the OptiX carry and the viewer's byte checks turn statistical.
GPU_TEST(optix_render, determinism)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "determinism");
    if (!runtime)
    {
        return;
    }
    compile_render_programs(ctx, *runtime);
    const survey_size size = ctx.quick() ? quick_survey : full_survey;
    for (const std::string_view name : std::array<std::string_view, 3>{"cornell", "final", "studio"})
    {
        const gpu::scene_desc scene = survey_scene(name, size, gpu_test::host::scene_strategy::mis);
        const gpu::render_result first = render_optix(scene, survey_seed, runtime);
        const gpu::render_result second = render_optix(scene, survey_seed, runtime);
        ctx.count(std::format("{}.second-run-differs", name), compare(first.pixels, second.pixels).values_differ, 0,
                  static_cast<std::int64_t>(first.pixels.size()));
        ctx.exact(std::format("{}.second-run-rays", name), static_cast<double>(second.rays),
                  static_cast<double>(first.rays), 0);
        ctx.exact(std::format("{}.second-run-shadow-rays", name), static_cast<double>(second.shadow_rays),
                  static_cast<double>(first.shadow_rays), 0);
    }
}

// A tiny scene: one matte sphere under a sky, a 4 × 4 image of one sample, sampled as the material scatters
gpu::scene_desc refusal_scene()
{
    gpu::scene_desc scene;
    scene.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = {0.5, 0.5, 0.5}});
    scene.materials.push_back({.kind = gpu::material_kind::lambertian, .texture = 0});
    scene.spheres.push_back({.center = {0, 0, -2}, .radius = 1, .material = 0});
    scene.camera = {.image_width = 4,
                    .image_height = 4,
                    .sqrt_spp = 1,
                    .max_depth = 4,
                    .center = {0, 0, 0},
                    .pixel00_loc = {-0.375, 0.375, -1},
                    .pixel_delta_u = {0.25, 0, 0},
                    .pixel_delta_v = {0, -0.25, 0},
                    .sky_bottom = {1, 1, 1},
                    .sky_top = {0.5, 0.7, 1},
                    .sampling = gpu::sampling_strategy::material};
    return scene;
}

// What making a session throws: std::invalid_argument, a std::logic_error that isn't one, another exception, or nothing
// (the first word), and its message
std::pair<std::string, std::string> refusal(const gpu::scene_desc& scene, const gpu::session_options& options)
{
    try
    {
        gpu::session s(scene, options);
    }
    catch (const std::invalid_argument& error)
    {
        return {"std::invalid_argument", error.what()};
    }
    catch (const std::logic_error& error)
    {
        return {"std::logic_error", error.what()};
    }
    catch (const std::exception& error)
    {
        return {"another exception", error.what()};
    }
    return {"nothing", ""};
}

void expect_refusal(gpu_test::context& ctx, std::string_view name, const gpu::scene_desc& scene,
                    const gpu::session_options& options, std::string_view expected)
{
    const auto [thrown, message] = refusal(scene, options);
    ctx.info(name, std::format("{}: {}", thrown, message)); // the words, for the review
    ctx.exact(name, thrown == expected ? 1 : 0, 1, 0);
}

// The session's refusals for OptiX (renderer.h), each before anything renders: optix_tracer in float64, with the
// kernel for all three books, with the quality kernel, or with a strategy of the books' lights list (std::
// invalid_argument: what OptiX's tracer can't render); optix_tracer without a runtime (std::logic_error: the caller's
// mistake); and a scene with triangle meshes without optix_tracer (std::invalid_argument: only OptiX renders them),
// which needs no OptiX, so it is decided even where OptiX can't run.
GPU_TEST(optix_render, refusals)
{
    const gpu::scene_desc scene = refusal_scene();
    gpu::scene_desc meshes = scene;
    auto triangle = std::make_shared<gpu::mesh_desc>();
    triangle->positions = {-1, -1, -3, 1, -1, -3, 0, 1, -3};
    triangle->indices = {0, 1, 2};
    triangle->material_of = {0};
    meshes.meshes.meshes.push_back(std::move(triangle));
    meshes.meshes.instances.push_back({.mesh = 0});
    meshes.meshes.materials.push_back({.scene_material = 0});
    expect_refusal(ctx, "meshes-without-optix-tracer", meshes, {}, "std::invalid_argument");
    expect_refusal(ctx, "optix-tracer-without-runtime", scene, {.optix_tracer = true}, "std::logic_error");

    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "refusals");
    if (!runtime)
    {
        return;
    }
    expect_refusal(ctx, "optix-tracer-float64", scene,
                   {.arithmetic = gpu::precision::float64, .optix_tracer = true, .optix = runtime},
                   "std::invalid_argument");
    expect_refusal(ctx, "optix-tracer-all-books-kernel", scene,
                   {.all_books_kernel = true, .optix_tracer = true, .optix = runtime}, "std::invalid_argument");
    expect_refusal(ctx, "optix-tracer-quality-kernel", scene,
                   {.quality_kernel = true, .optix_tracer = true, .optix = runtime}, "std::invalid_argument");
    gpu::scene_desc mixture = scene;
    mixture.camera.sampling = gpu::sampling_strategy::mixture;
    mixture.lights.push_back({.kind = gpu::light_shape::sphere, .center = {0, 0, -2}, .radius = 1, .weight = 1});
    expect_refusal(ctx, "optix-tracer-mixture", mixture, {.optix_tracer = true, .optix = runtime},
                   "std::invalid_argument");
}

// One view of cornell (-p mis) rendered by a session in three calls of render_to, the second ending inside a pass
// (rows of the next pass done: a band that starts below the top), against one gpu::render of as many passes: the
// same doubles bit for bit, and the same rays (renderer.h's promise, here for OptiX's launches)
GPU_TEST(optix_render, three_calls)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "three-calls");
    if (!runtime)
    {
        return;
    }
    compile_render_programs(ctx, *runtime);
    const survey_size size = ctx.quick() ? quick_survey : full_survey;
    const gpu::scene_desc scene = survey_scene("cornell", size, gpu_test::host::scene_strategy::mis);
    const std::int64_t passes = size.samples;
    gpu::session split(scene, {.seed = three_calls_seed, .optix_tracer = true, .optix = runtime});
    split.render_to({.passes = passes / 4, .rows = 0});
    split.render_to({.passes = passes / 2, .rows = size.height / 2});
    split.render_to({.passes = passes, .rows = 0});
    const gpu::render_result whole =
        gpu::render(scene, {.seed = three_calls_seed, .passes = passes, .optix_tracer = true, .optix = runtime});
    ctx.count("three-calls.differ", compare(split.pixels(), whole.pixels).values_differ, 0,
              static_cast<std::int64_t>(whole.pixels.size()));
    ctx.exact("three-calls.rays", static_cast<double>(split.rays()), static_cast<double>(whole.rays), 0);
    ctx.exact("three-calls.shadow-rays", static_cast<double>(split.shadow_rays()),
              static_cast<double>(whole.shadow_rays), 0);
}

// The suite's planted faults, each of which a gate above must reject: OptiX's cornell at another seed
// (9005) against seed 7's, which determinism and three_calls must call different; and env-furnace under -p mis, where
// the map is a light and a value is 1 only on average (docs/quality.md: 1.0008 ± 0.0007 at the sphere's centre), which
// furnaces must call not exactly 1. Each passes when its gate would have seen at least one value.
GPU_TEST(optix_render, planted_faults)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, "planted-faults");
    if (!runtime)
    {
        return;
    }
    compile_render_programs(ctx, *runtime);
    const survey_size size = ctx.quick() ? quick_survey : full_survey;
    const gpu::scene_desc cornell = survey_scene("cornell", size, gpu_test::host::scene_strategy::mis);
    const gpu::render_result seven = render_optix(cornell, survey_seed, runtime);
    const gpu::render_result other = render_optix(cornell, planted_seed, runtime);
    const auto values = static_cast<std::int64_t>(seven.pixels.size());
    const std::int64_t differ = compare(seven.pixels, other.pixels).values_differ;
    ctx.info("fault-other-seed.cornell", std::format("{} of {} values differ", differ, values));
    ctx.count("fault-other-seed.cornell.equal", values - differ, values - 1, values);

    const gpu::scene_desc furnace = survey_scene("env-furnace", size, gpu_test::host::scene_strategy::mis);
    const gpu::render_result mis = render_optix(furnace, survey_seed, runtime);
    const std::int64_t not_one = std::ranges::count_if(mis.pixels, [](double v) { return v != 1.0; });
    const auto furnace_values = static_cast<std::int64_t>(mis.pixels.size());
    ctx.info("fault-mis.env-furnace", std::format("{} of {} values aren't exactly 1", not_one, furnace_values));
    ctx.count("fault-mis.env-furnace.exactly-one", furnace_values - not_one, furnace_values - 1, furnace_values);
}
}
