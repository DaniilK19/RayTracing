// OptiX for the programs and the renderer (start_optix.h, optix_runtime.cuh): the concrete runtime on runtime.cpp's
// core, and a scene in OptiX's acceleration structures with its launches. Compiled by nvcc, because it fills
// optix_params and the other device structs, which runtime.cpp (cl.exe, OptiX's host API) can't see; every OptiX call
// is runtime.cpp's or accel.cpp's (runtime_core.h).
//
// Shared code: the runtime and the launch plumbing (render, guides, pick, the tests). Track A's region: the upload
// (Step 88: the books' AABBs, their GAS, the root; Step 91 composes track B's mesh_scene and the IAS).
#include "start_optix.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "device_memory.h"
#include "mesh_scene.cuh"
#include "optix_params.h"
#include "optix_runtime.cuh"
#include "quality_access.cuh"
#include "runtime_core.h"

namespace gpu
{
namespace detail
{
// One launch's width × height × depth is at most 2^30 (OptiX Programming Guide §12): a larger band goes in parts
constexpr std::int64_t max_launch_pixels = std::int64_t{1} << 30;

// A GpuTests launch's own fields, as the check module reads them (zero in every other launch)
inline optix_test_fields test_fields(const optix_test_launch& test)
{
    return {.counts = test.counts, .values = test.values, .fault = test.fault, .samples = test.samples,
            .seed = test.seed};
}

// A scene in OptiX: what every launch reads (the quality view, lengthened in a mesh scene, the root, the mesh tables),
// what keeps the acceleration structures and the meshes alive while launches read them, and one buffer of parameters,
// which each launch fills in its turn on stream 0
class scene_in_optix final : public optix_scene
{
public:
    // view: quality_view_of(quality, base), lengthened by the meshes; root: the books' GAS, the IAS over it and the
    // meshes' GASes, or 0 (nothing to hit); accels: every acceleration structure the root reaches, kept until the scene
    // goes; meshes: the scene's mesh holder, never null (an empty one in a scene without meshes)
    scene_in_optix(optix_core::core& core, const quality_view& view, std::uint64_t root,
                   std::vector<optix_core::accel> accels, std::unique_ptr<mesh_scene> meshes,
                   const optix_upload_options& options)
        : core_(core),
          view_(view),
          root_(root),
          accels_(std::move(accels)),
          meshes_(std::move(meshes)),
          tables_(meshes_->tables()),
          options_(options)
    {
    }

    // A view with more pixels than any before may have the meshes make their lengthened tables again, longer, at new
    // addresses (mesh_scene::prepare_view): the view and the tables every launch reads are taken again, the view by
    // lengthening the one already lengthened, which gives the current pointers
    void prepare_view(int width, int height) override
    {
        meshes_->prepare_view(width, height);
        view_ = meshes_->lengthened(view_);
        tables_ = meshes_->tables();
    }

    void launch(const camera<float>& cam, std::uint64_t seed, int first_pass, int pass_count, int first_row,
                int row_count, double* sums, unsigned long long* counts) override
    {
        optix_params p = scene_params(cam);
        p.seed = seed;
        p.first_pass = first_pass;
        p.pass_count = pass_count;
        p.sums = sums;
        p.counts = counts;
        in_bands(p, optix_core::raygen::render, first_row, row_count);
    }

    // The whole view, banded as the render is: __raygen__guides takes its row as first_row + the launch index's y
    void guides(const camera<float>& cam, std::uint64_t seed, float4* albedo, float4* normal) override
    {
        optix_params p = scene_params(cam);
        p.guide_seed = seed;
        p.albedo = albedo;
        p.normal = normal;
        in_bands(p, optix_core::raygen::guides, 0, cam.image_height);
    }

    // One thread of __raygen__pick, its t copied back by a synchronous copy, which waits for the launch
    double pick(const std::array<double, 3>& origin, const std::array<double, 3>& direction) override
    {
        optix_params p = scene_params(camera<float>{});
        p.pick_ray = {.origin = {static_cast<float>(origin[0]), static_cast<float>(origin[1]),
                                 static_cast<float>(origin[2])},
                      .direction = {static_cast<float>(direction[0]), static_cast<float>(direction[1]),
                                    static_cast<float>(direction[2])},
                      .time = 0.0f};
        p.pick_t = pick_t_.data();
        send(p);
        optix_core::launch(core_, optix_core::raygen::pick, params_.data(), sizeof(optix_params), 1, 1);
        return static_cast<double>(pick_t_.to_host().front());
    }

    void launch_test(std::span<const unsigned char> ptx, const char* entry, int width, int height,
                     const camera<float>& cam, const optix_test_launch& test) override
    {
        optix_params p = scene_params(cam);
        p.test = test_fields(test);
        send(p);
        optix_core::launch_test(core_, ptx, entry, params_.data(), sizeof(optix_params), static_cast<unsigned>(width),
                                static_cast<unsigned>(height), options_.probe_intersections);
    }

private:
    // What every launch of the scene starts from: the scene's fields, the view's camera, everything else zero
    optix_params scene_params(const camera<float>& cam) const
    {
        optix_params p{};
        p.cam = cam;
        p.view = view_;
        p.root = root_;
        p.meshes = tables_;
        return p;
    }

    // The parameters to the GPU on stream 0, right before the launch that reads them. From pageable memory the copy
    // returns once it has taken the bytes, and it runs after the launches queued before it on the same stream: a band
    // never reads the next band's parameters.
    void send(const optix_params& p)
    {
        check(cudaMemcpyAsync(params_.data(), &p, sizeof p, cudaMemcpyHostToDevice, 0),
              "copying OptiX's launch parameters to the GPU");
    }

    // A launch of `entry` over rows [first_row, first_row + row_count) of the view, in bands of at most 2^30 pixels:
    // each band gets its first_row and row_count, and its launch index's y counts from first_row
    void in_bands(optix_params& p, optix_core::raygen entry, int first_row, int row_count)
    {
        const int width = p.cam.image_width;
        const auto band_rows = static_cast<int>(std::max<std::int64_t>(1, max_launch_pixels / width));
        for (int row = first_row; row < first_row + row_count; row += band_rows)
        {
            p.first_row = row;
            p.row_count = std::min(band_rows, first_row + row_count - row);
            send(p);
            optix_core::launch(core_, entry, params_.data(), sizeof(optix_params), static_cast<unsigned>(width),
                               static_cast<unsigned>(p.row_count));
        }
    }

    optix_core::core& core_;
    quality_view view_;
    std::uint64_t root_;
    std::vector<optix_core::accel> accels_;
    std::unique_ptr<mesh_scene> meshes_;
    mesh_tables tables_;
    optix_upload_options options_;
    device_array<optix_params> params_{1};
    device_array<float> pick_t_{1};
};

// --- The upload (track A, Step 88; Step 91 composes track B's mesh_scene and the IAS) ----------------------------
// The books' AABBs from bvh.cuh's host boxes, placed at each primitive's global id (sphere_ids, quad_ids), their GAS,
// the root (the books' GAS, or the IAS over it and the meshes' instances), and quality_view_of(quality, base)
// lengthened by the meshes: then a scene_in_optix of them.
namespace
{
// A primitive's kind is its sbtIndexOffset in the books' GAS (accel.cpp): which of the books' three intersection
// programs tests it, in runtime.cpp's order of the hit groups
static_assert(static_cast<int>(primitive_kind::sphere) == 0 && static_cast<int>(primitive_kind::moving_sphere) == 1 &&
              static_cast<int>(primitive_kind::quad) == 2);

// A box of the BVH's builder (in double) rounded outward to float, as the builder rounds its nodes' boxes (round_down,
// round_up in bvh.cuh): the nearest float to a corner may lie inside the primitive, and a ray grazing it there would
// miss the box and with it the primitive
optix_core::aabb rounded_outward(const build_box& box)
{
    return {.lo = {round_down<float>(box.min[0]), round_down<float>(box.min[1]), round_down<float>(box.min[2])},
            .hi = {round_up<float>(box.max[0]), round_up<float>(box.max[1]), round_up<float>(box.max[2])}};
}

// The books' primitives in global-id order (bvh.cuh: the spheres that stand still, the moving ones, the quads): each
// one's box, and its kind as a byte, the build's sbtIndexOffset
struct books_boxes
{
    std::vector<optix_core::aabb> boxes;
    std::vector<std::uint8_t> kinds;
};

// Each primitive's box is the one the BVH's builder gives it in its leaf (sphere_box, sweep_box, quad_box), around the
// primitive the GPU tests: a sphere's center and radius rounded to float first, as add_sphere and add_moving_sphere
// convert them, and for a moving sphere its whole sweep, the end computed in float as the GPU computes its center. Each
// goes to its global id through the BVH's own ids, and its kind is that id's range: the range also tells which of the
// tree's arrays the intersection program reads the primitive from.
books_boxes boxes_of(const scene_desc& scene, const bvh_view<float>& tree, std::span<const int> sphere_ids,
                     std::span<const int> quad_ids)
{
    const std::size_t count = scene.spheres.size() + scene.quads.size();
    if (sphere_ids.size() != scene.spheres.size() || quad_ids.size() != scene.quads.size() ||
        static_cast<std::size_t>(tree.quad_base) != scene.spheres.size())
    {
        throw std::logic_error("optix_runtime::upload takes the BVH's ids of the scene's spheres and quads");
    }
    books_boxes books{.boxes = std::vector<optix_core::aabb>(count), .kinds = std::vector<std::uint8_t>(count)};
    const auto place = [&books, count](int id, const build_box& box, primitive_kind kind) {
        if (id < 0 || static_cast<std::size_t>(id) >= count)
        {
            throw std::logic_error("optix_runtime::upload: a BVH id outside the scene's primitives");
        }
        books.boxes[static_cast<std::size_t>(id)] = rounded_outward(box);
        books.kinds[static_cast<std::size_t>(id)] = static_cast<std::uint8_t>(kind);
    };
    for (std::size_t i = 0; i < scene.spheres.size(); ++i)
    {
        const sphere_desc& s = scene.spheres[i];
        const int id = sphere_ids[i];
        const point3<float> center = to_real<float>(s.center);
        const auto radius = static_cast<float>(s.radius);
        if (id < tree.moving_base)
        {
            place(id, sphere_box(center, center, radius), primitive_kind::sphere);
        }
        else
        {
            const moving_sphere<float> moving{
                .center = center, .radius = radius, .displacement = to_real<float>(s.displacement), .unused = 0};
            place(id, sweep_box(moving), primitive_kind::moving_sphere);
        }
    }
    for (std::size_t i = 0; i < scene.quads.size(); ++i)
    {
        place(quad_ids[i], quad_box(scene.quads[i]), primitive_kind::quad);
    }
    return books;
}

// The books' GAS (accel.cpp), or none (handle 0) in a scene without books' primitives. The boxes and kinds are freed
// once it is built: a custom-primitive GAS keeps no reference to them.
optix_core::accel books_gas(optix_core::core& core, const books_boxes& books)
{
    if (books.boxes.empty())
    {
        return {};
    }
    const device_array<optix_core::aabb> boxes(books.boxes); // cudaMalloc: OPTIX_AABB_BUFFER_BYTE_ALIGNMENT and more
    const device_array<std::uint8_t> kinds(books.kinds);
    return optix_core::build_custom(core, boxes.data(), kinds.data(), static_cast<unsigned int>(books.boxes.size()));
}
}

std::unique_ptr<optix_scene> upload_scene(optix_core::core& core, const scene_desc& scene,
                                          const scene_view<float>& base, const quality_scene& quality,
                                          std::span<const int> sphere_ids, std::span<const int> quad_ids,
                                          const optix_upload_options& options)
{
    std::vector<optix_core::accel> accels; // every acceleration structure the root reaches, kept by the scene
    const optix_core::accel books = books_gas(core, boxes_of(scene, base.tree, sphere_ids, quad_ids));
    if (books.handle != 0)
    {
        accels.push_back(books);
    }

    // The meshes, track B's holder (empty in a scene without them): their GASes, and the integrator's view lengthened
    // for them. A scene without meshes gets the quality kernel's view unchanged.
    const quality_view view = quality_view_of(quality, base);
    auto meshes = std::make_unique<mesh_scene>(core, scene, view);
    const quality_view lengthened = meshes->lengthened(view);

    // The root: the books' GAS alone in a scene without meshes (the books' scenes see no instance); else an IAS of
    // instance 0, the books' GAS under the identity (instanceId 0, sbtOffset 0: the books' three records), and mesh
    // instance k as instance k + 1 (sbtOffset 3: the triangles' hit group). GpuTests may ask for the IAS in a books'
    // scene (force_instance_root). accel.cpp's IAS build gives every instance visibilityMask 0xFF and DISABLE_ANYHIT.
    std::uint64_t root = books.handle;
    if (!meshes->empty() || options.force_instance_root)
    {
        std::vector<optix_core::instance> instances;
        if (books.handle != 0)
        {
            instances.push_back({.transform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
                                 .id = 0,
                                 .sbt_offset = 0,
                                 .handle = books.handle});
        }
        const std::span<const optix_core::instance> placed = meshes->instances(); // ids k + 1, sbt_offset 3
        instances.insert(instances.end(), placed.begin(), placed.end());
        root = 0; // nothing at all: closest() is a miss without tracing
        if (!instances.empty())
        {
            accels.push_back(optix_core::build_instances(core, instances));
            root = accels.back().handle;
        }
    }
    // (The lengthened view is taken before the holder moves: arguments are initialized in no fixed order.)
    return std::make_unique<scene_in_optix>(core, lengthened, root, std::move(accels), std::move(meshes), options);
}
// --- end of the upload region ------------------------------------------------------------------------------------
}

namespace
{
// The program's OptiX: the core (context, programs, test pipelines, denoiser) and the buffer of parameters of the
// launches that need no scene
class runtime_on_core final : public optix_runtime
{
public:
    explicit runtime_on_core(optix_core::core_ptr core) : core_(std::move(core)) {}

    std::unique_ptr<detail::optix_scene> upload(const scene_desc& scene, const scene_view<float>& base,
                                                const detail::quality_scene& quality, std::span<const int> sphere_ids,
                                                std::span<const int> quad_ids,
                                                const optix_upload_options& options) override
    {
        // The first upload compiles the programs (if they don't compile, the error quotes OptiX's log); OptiX's disk
        // cache serves the later runs of the program
        optix_core::compile_programs(*core_);
        return detail::upload_scene(*core_, scene, base, quality, sphere_ids, quad_ids, options);
    }

    // OptiX reports no registers and no occupancy: the driver compiles the programs and schedules their threads. What
    // replaced the BVH's stack is the continuation stack, known once the programs are compiled.
    kernel_info kernel() const override
    {
        return {.name = "optix",
                .local_bytes = optix_core::continuation_stack_bytes(*core_),
                .programs = optix_core::programs_sha256().substr(0, 8)};
    }

    // The planner's one-wave floor: the same integrator as the quality kernel's (it changes the speed only)
    int resident_blocks() const override { return detail::quality_resident_blocks(); }

    void denoise(const float4* beauty, const float4* albedo, const float4* normal, float4* out, int width,
                 int height) override
    {
        optix_core::denoise(*core_, beauty, albedo, normal, out, width, height);
    }

    denoise_report last_denoise() const override { return optix_core::last_denoise(*core_); }

    void release_denoiser() override { optix_core::release_denoiser(*core_); }

    void set_denoise_tile_limit(int side) override { optix_core::set_denoise_tile_limit(*core_, side); }

    bool cache_enabled() const override { return optix_core::cache_enabled(*core_); }

    void compile_programs() override { optix_core::compile_programs(*core_); }

    // No scene: root 0 (a trace misses without traversing), a zero camera, the test's own fields
    void launch_test(std::span<const unsigned char> ptx, const char* entry, int width, int height,
                     const optix_test_launch& test) override
    {
        optix_params p{};
        p.test = detail::test_fields(test);
        check(cudaMemcpyAsync(test_params_.data(), &p, sizeof p, cudaMemcpyHostToDevice, 0),
              "copying OptiX's launch parameters to the GPU");
        optix_core::launch_test(*core_, ptx, entry, test_params_.data(), sizeof(optix_params),
                                static_cast<unsigned>(width), static_cast<unsigned>(height));
    }

private:
    optix_core::core_ptr core_; // declared first: it goes last, after every buffer of this class
    device_array<optix_params> test_params_{1};
};
}

bool optix_built()
{
    return true;
}

std::shared_ptr<optix_runtime> start_optix()
{
    return std::make_shared<runtime_on_core>(optix_core::open_core());
}

std::string describe(const optix_runtime& /*runtime*/)
{
    const int version = optix_core::headers_version();
    return std::format("OptiX {}.{}, programs {}", version / 10000, version % 10000 / 100,
                       optix_core::programs_sha256().substr(0, 8));
}

bool optix_cache_enabled(const optix_runtime& runtime)
{
    return runtime.cache_enabled();
}

namespace optix_core
{
std::size_t params_bytes()
{
    return sizeof(optix_params);
}
}
}
