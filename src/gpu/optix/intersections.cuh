// The books' primitives in OptiX: the intersection programs of the books' GAS, one per kind of primitive (its
// sbtIndexOffset 0, 1, 2), the leaf bodies of closest_hit (bvh.cuh) and occluded (shadow.cuh) on the ray OptiX hands
// them, with the integrator's skips in the payload (optix_tracer.cuh's: from, inward, skip, the ray's time, 1/|d|²).
//
// One body serves both of the BVH's searches. closest_hit tests a leaf's primitive over (0.001, closest.t), the nearest
// hit found so far; occluded over (0.001, t_max), and stops at the first hit. In an intersection program
// optixGetRayTmax() is exactly that upper end: the nearest t reported so far, or the tmax the ray was traced with while
// nothing has been; and a ray traced with OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT (optix_tracer's visibility) stops at
// its first report, as occluded returns at its first hit. The lower end is the books' t_min, applied here again inside
// each test's own interval (hit_sphere's open one, hit_quad's closed one), whatever tmin the traversal had.
//
// `params` is the module's: programs.cu defines it before it includes this file. Track A's file (Step 88).
#pragma once
#include <optix.h>
#include "optix_params.h"
#include "quad.cuh"
#include "sphere.cuh"

namespace gpu
{
// The ray of the current test, exactly as optixTraverse received it, at the time payload word 3 carries
// (optixGetRayTime is 0 in a pipeline without motion blur). Under the books' GAS as the root the world ray is the
// object ray, the floats optix_tracer traced; under the identity instance of a mesh scene, probe P7 (Step 91) checks
// that it still is.
__device__ inline ray<float> traced_ray()
{
    const float3 o = optixGetWorldRayOrigin();
    const float3 d = optixGetWorldRayDirection();
    return {{o.x, o.y, o.z}, {d.x, d.y, d.z}, __uint_as_float(optixGetPayload_3())};
}

// closest_hit's sphere leaf and occluded's, for the sphere with global id `id`: the same skips, the same function
// (hit_sphere, with 1/|d|² from payload word 4, as closest_hit computes it once per ray), the same open interval. The
// sphere the ray leaves (payload 0) is skipped when the ray goes out of it (payload 1 is 0: a sphere is convex), and
// only its far root counts when the ray goes in; the lamp a shadow ray aims at (payload 2; -1 for closest()) never
// blocks it.
__device__ inline void report_sphere(const sphere<float>& ball, int id)
{
    const bool own = id == static_cast<int>(optixGetPayload_0());
    if ((own && optixGetPayload_1() == 0u) || id == static_cast<int>(optixGetPayload_2()))
    {
        return; // a ray going out of its own sphere can't hit it; the lamp a shadow ray aims at doesn't block it
    }
    float t;
    bool entering;
    if (hit_sphere(ball, traced_ray(), __uint_as_float(optixGetPayload_4()), 0.001f, optixGetRayTmax(), own, t,
                   entering))
    {
        // OptiX accepts t in [tmin, tmax]; hit_sphere kept (t_min, t_max). The hit kind carries the root, 1 entering
        // and 0 the far one: optix_tracer's decode_hit reads it back as bvh_hit::entering.
        optixReportIntersection(t, entering ? 1u : 0u);
    }
}
}

// A sphere that stands still, global ids [0, moving_base): primitiveIndexOffset 0 makes the primitive index the global
// id, by which the tree's array of spheres is indexed
extern "C" __global__ void __intersection__sphere()
{
    const int id = static_cast<int>(optixGetPrimitiveIndex());
    gpu::report_sphere(params.view.base.tree.spheres[id], id);
}

// A moving sphere, where it is at the ray's time (moving_sphere::at, as closest_hit and occluded place it)
extern "C" __global__ void __intersection__moving_sphere()
{
    const gpu::bvh_view<float>& tree = params.view.base.tree;
    const int id = static_cast<int>(optixGetPrimitiveIndex());
    gpu::report_sphere(tree.moving_spheres[id - tree.moving_base].at(__uint_as_float(optixGetPayload_3())), id);
}

// A quad, closest_hit's quad leaf and occluded's: never the quad the ray leaves (a line meets a plane once, at the
// ray's start), nor the lamp it aims at; hit_quad's interval [t_min, t_max] closed, as on the CPU
extern "C" __global__ void __intersection__quad()
{
    const gpu::bvh_view<float>& tree = params.view.base.tree;
    const int id = static_cast<int>(optixGetPrimitiveIndex());
    if (id == static_cast<int>(optixGetPayload_0()) || id == static_cast<int>(optixGetPayload_2()))
    {
        return; // never the quad the ray leaves, nor the lamp it aims at
    }
    float t;
    bool front_face;
    if (gpu::hit_quad(tree.quads[id - tree.quad_base], gpu::traced_ray(), 0.001f, optixGetRayTmax(), t, front_face))
    {
        optixReportIntersection(t, front_face ? 1u : 0u); // the side the normal points to: bvh_hit::entering
    }
}
