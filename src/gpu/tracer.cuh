// How the quality kernel's integrator (quality_path.cuh) finds what a ray meets: through a tracer, never the BVH by
// name. The shading code (materials, lamps, the environment) then needs no acceleration structure, and another tracer
// (OptiX's ray tracing hardware, a later milestone) can take this one's place. bvh_tracer walks the renderer's own BVH
// with the books' functions.
#pragma once

#include <concepts>

#include "camera.cuh"
#include "quality_view.cuh"
#include "shadow.cuh"

namespace gpu
{
// The self-intersection state of a ray's origin (closest_hit's rules in bvh.cuh): the primitive it leaves (-1 for none,
// after the camera or a particle of smoke), and whether it goes into that sphere
struct spawn
{
    int from;
    bool inward;
};

// What a tracer offers. A concept (C++20): a type is a tracer if these expressions compile with these results.
template <class T>
concept tracer = requires(const T& t, const ray<float>& r, spawn s, pcg32& rng, float t_max, int skip,
                          const bvh_hit<float>& h) {
    { t.closest(r, s, rng) } -> std::same_as<bvh_hit<float>>; // surfaces, then particles of smoke
    // The hit's point, normal, material and texture coordinates. (Named surface_of: a member may not change what an
    // unqualified name means in its class, and surface is the books' struct of the result.)
    { t.surface_of(r, h) } -> std::same_as<surface<float>>;
    { t.visibility(r, s, t_max, skip) } -> std::same_as<float>; // 0 blocked, else the media's transmittance
};

struct bvh_tracer
{
    const quality_view& view;

    // closest_event: the nearest surface, or a particle of smoke in front of it, drawing the particle's distance
    __device__ bvh_hit<float> closest(const ray<float>& r, spawn s, pcg32& rng) const
    {
        return closest_event(view.base, r, s.from, s.inward, rng);
    }

    __device__ gpu::surface<float> surface_of(const ray<float>& r, const bvh_hit<float>& h) const
    {
        return surface_at(view.base, r, h);
    }

    // Whether anything blocks (0.001, t_max) of r, the primitive `skip` aside, and if not, the share of light the media
    // let through: the shadow ray of a light sample (shadow.cuh). The volumes are tested only if no surface blocks the
    // ray: they aren't the BVH's, and an OptiX tracer would keep them outside its acceleration structure too.
    __device__ float visibility(const ray<float>& r, spawn s, float t_max, int skip) const
    {
        if (occluded(view.base.tree, r, t_max, s.from, s.inward, skip))
        {
            return 0.0f;
        }
        return transmittance(view.base.media, r, t_max);
    }
};
static_assert(tracer<bvh_tracer>);
}
