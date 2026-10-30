// The OptiX tracer: what the quality kernel's integrator (quality_path.cuh, unchanged) asks of geometry, answered by
// OptiX's traversal on the ray tracing hardware instead of the BVH (tracer.cuh's bvh_tracer). Rays go through
// optixTraverse and its hit object: no closest-hit, any-hit or miss program runs.
#pragma once
#include <cfloat>
#include <optix.h>
#include "optix_params.h"
#include "tracer.cuh"
#include "triangles.cuh"

namespace gpu
{
// closest_event's loop over the volumes (camera.cuh:264-283), copied verbatim: camera.cuh is frozen, and GpuTests'
// tracer check compares the random numbers left after every closest(). The same media in the same order, each drawing
// its one number only where its segment overlaps [0.001, closest.t]. (Line 264, `media = scene.media`, is the
// parameter here; 265-283 follow unchanged.)
__device__ inline bvh_hit<float> media_events(const media_view<float>& media, const ray<float>& r,
                                              bvh_hit<float> closest, pcg32& rng)
{
    using Real = float; // camera.cuh's lines are a template's over Real: from here on they are the same text
    if (media.count == 0)
    {
        return closest;
    }
    const Real t_min = Real(0.001);
    const Real inverse_a = Real(1) / r.direction.length_squared();
    const vec3<Real> inverse_direction{safe_inverse(r.direction.x), safe_inverse(r.direction.y),
                                       safe_inverse(r.direction.z)};
    const vec3<Real> origin_scaled = r.origin * inverse_direction;
    for (int k = 0; k < media.count; ++k)
    {
        Real t;
        if (medium_collision(media.media[k], media.boundary_quads, r, inverse_a, inverse_direction, origin_scaled,
                             t_min, closest.t, rng, t))
        {
            closest = {.index = media.base + k, .t = t, .entering = true};
        }
    }
    return closest;
}

struct optix_tracer
{
    const optix_params& p;
    triangle_memo& memo; // a reference: the tracer object stays const, as the concept asks
    int slot;            // the pixel's scratch slot: mesh_tables::scratch_base + the pixel's index

    __device__ bvh_hit<float> closest(const ray<float>& r, spawn s, pcg32& rng) const
    {
        if (leaves_triangle(s) && s.inward)
        {
            return {.index = p.meshes.triangle_base, .t = 0.0f, .entering = false}; // through the face: absorbed
        }
        bvh_hit<float> hit{.index = -1, .t = infinity<float>, .entering = false};
        if (p.root != 0)
        {
            traverse(r, s, infinity<float>, -1, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
            if (optixHitObjectIsHit())
            {
                hit = decode_hit();
            }
        }
        hit = media_events(p.view.base.media, r, hit, rng);
        if (hit.index > p.meshes.triangle_base && wears_gltf(p.meshes, memo))
        {
            memo.lobe = random_real<float>(rng); // after the media's numbers, only at a glTF triangle
        }
        return hit;
    }

    __device__ surface<float> surface_of(const ray<float>& r, const bvh_hit<float>& h) const
    {
        if (h.index < p.meshes.triangle_base)
        {
            return surface_at(p.view.base, r, h); // the books' primitives and particles: the CUDA kernel's own code
        }
        if (h.index == p.meshes.triangle_base)
        {
            return absorber_surface(p.meshes, r);
        }
        return triangle_surface(p, r, h, memo, slot);
    }

    __device__ float visibility(const ray<float>& r, spawn s, float t_max, int skip) const
    {
        if (leaves_triangle(s) && s.inward)
        {
            return 0.0f; // the face itself is in the way: the same rule as closest(), for the light technique
        }
        if (p.root != 0)
        {
            traverse(r, s, t_max, skip, OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);
            if (optixHitObjectIsHit())
            {
                return 0.0f;
            }
        }
        return transmittance(p.view.base.media, r, t_max); // shadow.cuh, unchanged
    }

private:
    __device__ bool leaves_triangle(spawn s) const { return s.from > p.meshes.triangle_base; }

    __device__ void traverse(const ray<float>& r, spawn s, float t_max, int skip, unsigned int flags) const
    {
        point3<float> origin = r.origin;
        float t_min = 0.001f; // the books' t_min; their intersection programs test it themselves too
        if (leaves_triangle(s))
        {
            origin = offset_origin(r.origin, memo.facing); // off the face, on the side the ray goes to
            t_min = 0.0f;
        }
        unsigned int w0 = static_cast<unsigned int>(s.from);
        unsigned int w1 = s.inward ? 1u : 0u;
        unsigned int w2 = static_cast<unsigned int>(skip);
        unsigned int w3 = __float_as_uint(r.time);
        unsigned int w4 = __float_as_uint(1.0f / r.direction.length_squared());
        optixTraverse(p.root, make_float3(origin.x, origin.y, origin.z),
                      make_float3(r.direction.x, r.direction.y, r.direction.z), t_min, fminf(t_max, FLT_MAX), 0.0f,
                      OptixVisibilityMask(255), flags, 0, 1, 0, w0, w1, w2, w3, w4);
    }

    __device__ bvh_hit<float> decode_hit() const
    {
        const unsigned int kind = optixHitObjectGetHitKind();
        if (optixGetPrimitiveType(kind) == OPTIX_PRIMITIVE_TYPE_CUSTOM)
        {
            return {.index = static_cast<int>(optixHitObjectGetPrimitiveIndex()), // the books' global id
                    .t = optixHitObjectGetRayTmax(),                             // the reported t, bit for bit
                    .entering = (kind & 1u) != 0u};
        }
        memo.instance = static_cast<int>(optixHitObjectGetInstanceId()) - 1; // instance 0 is the books' GAS
        memo.primitive = optixHitObjectGetPrimitiveIndex();
        // The barycentrics are a triangle hit's attributes 0 and 1 (optix_device.h, at optixGetTriangleBarycentrics),
        // read as such rather than by optixHitObjectGetTriangleBarycentrics. OptiX 9.0's header writes that one as an
        // `asm` without `volatile` and without operands, which tells nvcc it depends on nothing: nvcc hoisted it out of
        // __raygen__render's loop, ahead of the first optixTraverse, so every triangle of a render got the
        // barycentrics of a hit object that held no hit yet, and the paths left the triangles at points that weren't
        // where the rays met them: cornell-triangles came out a fifth as bright as cornell, and a closed block leaked
        // light. optixHitObjectGetAttribute_0/1 are `asm volatile`, as optixTraverse is, so they stay after it.
        memo.b1 = __uint_as_float(optixHitObjectGetAttribute_0());
        memo.b2 = __uint_as_float(optixHitObjectGetAttribute_1());
        return {.index = p.meshes.instances[memo.instance].first_id + static_cast<int>(memo.primitive),
                .t = optixHitObjectGetRayTmax(),
                .entering = false}; // a triangle's side is decided by surface_of, from its transformed normal
    }
};
static_assert(tracer<optix_tracer>);
}
