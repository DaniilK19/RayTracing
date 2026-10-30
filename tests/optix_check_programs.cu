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
// The quality kernel's two tracers, compiled by one compiler into this one module, so that only the traversal and the
// intersection programs differ: bvh_tracer walks the renderer's own BVH (tests/test_scene.h's upload),
// optix_tracer the scene's GAS through the render module's intersection programs, which this module's pipelines link.
// Each sample of a pixel follows a path as the integrator draws it (quality_path.cuh's order of the random numbers):
// the camera ray (start_quality_sample), then up to max_vertices vertices, at each of them
//   - closest() of both tracers on the same ray and spawn, each with its own copy of the one generator, compared: the
//     index, entering, t's bits, and the generator each leaves behind (the media's draws). The path goes on with the
//     BVH's hit and numbers;
//   - one shadow query, visibility() of both with the same spawn: a light sample (sample_light: the lamp's own id to
//     skip, its distance, the environment's infinite rays) where the scene has lights; else a direction uniform on the
//     sphere, a length log-uniform from 0.01 to 10 times the scene's size and, half the time, a random primitive to
//     skip;
//   - the next direction, drawn by the vertex's material as the integrator draws it (make_bsdf with its GGX side entry,
//     then sample_bsdf): glass sends rays into their own sphere, whose far root the own-sphere rule must find.
// params.test: counts (tracer_check::counter), values (tracer_check::value), samples per pixel, the seed, and a planted
// fault on OptiX's side only, which the check must see (optix_tests.cu): 1 the hit kind read inverted, 2 the media loop
// skipped, 3 skip = -1 for the shadow rays. A launch with 0 samples traces nothing: it makes the pipeline.
#include <cstring>

#include "quality_path.cuh"
#include "tracer_check_layout.h" // params.test's counts and values, in the order suite optix-tracer reads them

namespace tracer_check
{
constexpr int max_vertices = 4; // closest() queries per sample: the camera ray and three bounces

// Whether two generators will draw the same numbers: pcg32's 16 bytes, its state and its stream, equal
__device__ inline bool same_numbers(const gpu::pcg32& a, const gpu::pcg32& b)
{
    static_assert(sizeof(gpu::pcg32) == 2 * sizeof(std::uint64_t));
    std::uint64_t x[2];
    std::uint64_t y[2];
    memcpy(x, &a, sizeof x);
    memcpy(y, &b, sizeof y);
    return x[0] == y[0] && x[1] == y[1];
}

// The scene's size, for the lengths of the random shadow rays: the diagonal of the BVH's root box (1 without surfaces)
__device__ inline float scene_size(const gpu::bvh_view<float>& tree)
{
    if (tree.node_count == 0)
    {
        return 1.0f;
    }
    const gpu::bvh_node<float> root = tree.nodes[0];
    return sqrtf((root.box_max - root.box_min).length_squared());
}

// The values of the first example of a kind of difference, for the one thread that claims them; nullptr for the rest
__device__ inline float* claim(counter flag, value first)
{
    return atomicCAS(&params.test.counts[flag], 0ull, 1ull) == 0ull ? params.test.values + first : nullptr;
}
}

extern "C" __global__ void __raygen__check()
{
    using namespace tracer_check;
    const uint3 index = optixGetLaunchIndex();
    const int i = static_cast<int>(index.x);
    const int j = static_cast<int>(index.y);
    const std::uint64_t pixel =
        static_cast<std::uint64_t>(j) * optixGetLaunchDimensions().x + static_cast<std::uint64_t>(i);
    const gpu::optix_test_fields& test = params.test;
    const gpu::quality_view& view = params.view;
    const gpu::bvh_tracer bvh_side{view};
    gpu::triangle_memo memo{};
    const gpu::optix_tracer optix_side{params, memo, 0}; // a scene of the books: no glTF material reads the slot
    const int surfaces = view.base.media.base;           // the surface primitives' global ids are [0, media.base)
    const float extent = scene_size(view.base.tree);

    unsigned long long n[mismatch_claimed] = {}; // this thread's tallies, added to the launch's once at the end
    for (int sample = 0; sample < test.samples; ++sample)
    {
        gpu::pcg32 rng = gpu::sample_generator(test.seed, pixel, static_cast<std::uint64_t>(sample));
        gpu::ray<float> r = gpu::start_quality_sample(params.cam, view, i, j, sample, rng).base.r;
        int from = -1;
        bool inward = false;
        for (int vertex = 0; vertex < max_vertices; ++vertex)
        {
            const gpu::spawn s{from, inward};
            gpu::pcg32 bvh_rng = rng;
            gpu::pcg32 optix_rng = rng;
            const gpu::bvh_hit<float> a = bvh_side.closest(r, s, bvh_rng);
            gpu::bvh_hit<float> b;
            if (test.fault == 2)
            {
                // Fault 2: the media loop skipped, as for a scene without media
                gpu::optix_params blind = params;
                blind.view.base.media.count = 0;
                b = gpu::optix_tracer{blind, memo, 0}.closest(r, s, optix_rng);
            }
            else
            {
                b = optix_side.closest(r, s, optix_rng);
            }
            if (test.fault == 1 && b.index >= 0 && b.index < surfaces)
            {
                b.entering = !b.entering; // fault 1: a surface's hit kind read inverted
            }

            ++n[closest_queries];
            n[camera_rays] += vertex == 0 ? 1 : 0;
            n[own_sphere_inward] += s.inward && s.from >= 0 && s.from < view.base.tree.quad_base ? 1 : 0;
            const bool numbers = same_numbers(bvh_rng, optix_rng);
            const bool alike = a.index == b.index && a.entering == b.entering;
            const bool same_t = __float_as_uint(a.t) == __float_as_uint(b.t);
            if (alike && numbers && same_t)
            {
                ++n[same];
            }
            else if (alike && numbers && fabsf(a.t - b.t) <= 1e-5f * a.t)
            {
                ++n[ulp];
                atomicMax(reinterpret_cast<unsigned int*>(test.values + largest_ulp_ratio),
                          __float_as_uint(fabsf(a.t - b.t) / a.t));
            }
            else
            {
                ++n[mismatches];
                n[ties] += a.index != b.index && a.t == b.t ? 1 : 0;
                n[numbers_only] += alike && same_t ? 1 : 0;
                if (float* const e = claim(mismatch_claimed, mismatch_example))
                {
                    const float example[13] = {static_cast<float>(i),       static_cast<float>(j),
                                               static_cast<float>(sample),  static_cast<float>(vertex),
                                               static_cast<float>(from),    inward ? 1.0f : 0.0f,
                                               static_cast<float>(a.index), a.t,
                                               a.entering ? 1.0f : 0.0f,    static_cast<float>(b.index),
                                               b.t,                         b.entering ? 1.0f : 0.0f,
                                               numbers ? 1.0f : 0.0f};
                    memcpy(e, example, sizeof example);
                }
            }
            rng = bvh_rng;
            if (a.index < 0)
            {
                ++n[escaped];
                break; // the sky
            }
            n[particles] += a.index >= surfaces ? 1 : 0;
            const gpu::surface<float> hit = bvh_side.surface_of(r, a);

            // The shadow query, from the vertex as the integrator's would leave it (quality_path.cuh: the primitive it
            // leaves, and whether it goes through its outward side)
            gpu::ray<float> shadow{};
            float t_max = 0.0f;
            int skip = -1;
            bool queried = true;
            if (gpu::light_count(view) > 0)
            {
                gpu::light_sample ls;
                queried = gpu::sample_light(view, hit.rec.p, rng, ls);
                if (queried)
                {
                    shadow = {hit.rec.p, ls.wi, r.time};
                    t_max = ls.distance;
                    skip = ls.skip;
                }
            }
            else
            {
                const gpu::vec3<float> wi = gpu::random_unit_vector<float>(rng);
                // 0.01 to 10 times the size: 0.01·1000^ξ = 0.01·2^(ξ·log2 1000)
                t_max = extent * 0.01f * exp2f(9.96578428f * gpu::random_real<float>(rng));
                if (surfaces > 0 && gpu::random_real<float>(rng) < 0.5f)
                {
                    const int k = static_cast<int>(gpu::random_real<float>(rng) * static_cast<float>(surfaces));
                    skip = k < surfaces ? k : surfaces - 1; // ξ·N can round up to N
                }
                shadow = {hit.rec.p, wi, r.time};
            }
            if (queried)
            {
                const gpu::spawn leaving{hit.from, gpu::dot(shadow.direction, hit.outward) < 0.0f};
                const float va = bvh_side.visibility(shadow, leaving, t_max, skip);
                const float vb = optix_side.visibility(shadow, leaving, t_max, test.fault == 3 ? -1 : skip);
                ++n[shadow_queries];
                n[shadows_blocked] += va == 0.0f ? 1 : 0;
                if (__float_as_uint(va) != __float_as_uint(vb))
                {
                    ++n[visibility_differences];
                    if (float* const e = claim(visibility_claimed, visibility_example))
                    {
                        const float example[10] = {static_cast<float>(i),         static_cast<float>(j),
                                                   static_cast<float>(sample),    static_cast<float>(vertex),
                                                   static_cast<float>(hit.from),  leaving.inward ? 1.0f : 0.0f,
                                                   static_cast<float>(skip),      t_max,
                                                   va,                            vb};
                        memcpy(e, example, sizeof example);
                    }
                }
            }

            // The next direction, as trace_quality_ray draws it after the light sample: a lamp ends the path
            const gpu::material<float> mat = view.base.materials[hit.material];
            if (mat.kind == gpu::material_kind::diffuse_light)
            {
                break;
            }
            const bool colored =
                mat.kind == gpu::material_kind::lambertian || mat.kind == gpu::material_kind::isotropic;
            const gpu::color<float> c = colored ? gpu::surface_color(view.base, mat, hit) : gpu::color<float>{};
            const gpu::microfacet* side = view.microfacets ? &view.microfacets[hit.material] : nullptr;
            const gpu::bsdf lobe = gpu::make_bsdf(mat, side, c, hit.rec, r.direction);
            gpu::bsdf_sample next;
            if (!gpu::sample_bsdf(lobe, r, hit.rec, rng, next))
            {
                break; // absorbed
            }
            r = {hit.rec.p, next.wi, r.time};
            from = hit.from;
            inward = gpu::dot(next.wi, hit.outward) < 0.0f;
        }
    }
    for (int k = 0; k < mismatch_claimed; ++k)
    {
        if (n[k] != 0)
        {
            atomicAdd(&test.counts[k], n[k]);
        }
    }
}

// P4 and P5 (optix-runtime, Step 88): rays the host gives, each traced by optixTraverse alone as optix_tracer's
// traverse() traces a ray that leaves no primitive, but with its tmax as given: +∞ too, which the tracer never passes
// (it clamps to FLT_MAX). Ray k, at launch index (k, 0), is params.test.values[12k, 12k + 12): its origin (3 floats),
// direction (3), tmax and skip (a global id, -1 for none), then what it met, written here: 1 a hit, 0 a miss, -1
// nothing traced (the scene's root is 0); the hit's t; the low bit of its hit kind (a quad's front, a sphere's
// entering); its primitive index.
extern "C" __global__ void __raygen__check_ray()
{
    float* const slot = params.test.values + 12 * static_cast<std::size_t>(optixGetLaunchIndex().x);
    if (params.root == 0)
    {
        slot[8] = -1.0f;
        return;
    }
    const gpu::vec3<float> direction{slot[3], slot[4], slot[5]};
    unsigned int from = 0xffffffffu; // payload words as traverse() writes them
    unsigned int inward = 0u;
    unsigned int skip = static_cast<unsigned int>(static_cast<int>(slot[7]));
    unsigned int ray_time = __float_as_uint(0.0f);
    unsigned int inverse_a = __float_as_uint(1.0f / direction.length_squared());
    optixTraverse(params.root, make_float3(slot[0], slot[1], slot[2]),
                  make_float3(direction.x, direction.y, direction.z), 0.001f, slot[6], 0.0f, OptixVisibilityMask(255),
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, from, inward, skip, ray_time, inverse_a);
    const bool hit = optixHitObjectIsHit();
    slot[8] = hit ? 1.0f : 0.0f;
    slot[9] = hit ? optixHitObjectGetRayTmax() : 0.0f;
    slot[10] = hit ? static_cast<float>(optixHitObjectGetHitKind() & 1u) : 0.0f;
    slot[11] = hit ? static_cast<float>(optixHitObjectGetPrimitiveIndex()) : 0.0f;
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
