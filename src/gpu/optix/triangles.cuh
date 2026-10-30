// Triangles as the integrator sees them: a mesh hit's surface, the ray origin moved off a face, the absorber
//
// Track B's file (Step 91: the bodies of triangle_surface for scene materials, offset_origin and absorber_surface;
// Step 92: the glTF branch and wears_gltf). Until then every function is a stub that fails loudly where it could be
// reached: without meshes triangle_base is INT_MAX, so no hit and no ray of a books' scene ever comes here.
#pragma once
#include "camera.cuh"
#include "optix_params.h"

namespace gpu
{
// What the OptiX tracer remembers between its calls in one step of a path (trace_quality_ray calls closest, then
// surface_of for that hit, then at most one visibility, then the next closest: quality_path.cuh:88-186): a triangle's
// hit data, which bvh_hit has no room for, and the face the next rays leave. It lives in the raygen's frame.
struct triangle_memo
{
    int instance = 0;           // closest(): the hit triangle's instance (an index into mesh_tables::instances)
    unsigned int primitive = 0; // its index in its mesh
    float b1 = 0;               // its barycentric coordinates: the weights of corners 1 and 2
    float b2 = 0;
    float lobe = 0;             // closest(): the number a glTF material chooses its lobe with
    vec3<float> facing{};       // surface_of(): the face's unit normal on the side the ray came from
};

// The hit triangle wears a glTF material (its lobe needs a number). Stub: false.
__device__ inline bool wears_gltf(const mesh_tables& m, const triangle_memo& memo)
{
    return false;
}

// A triangle's surface: the point from the barycentrics, ng, ns, outward = facing, u, v, the material (a scene
// material's index, or the pixel's scratch slot filled with the glTF mapping); sets memo.facing. Stub: __trap()
// (unreachable until 91: without meshes triangle_base is INT_MAX).
__device__ inline surface<float> triangle_surface(const optix_params& p, const ray<float>& r, const bvh_hit<float>& h,
                                                  triangle_memo& memo, int slot)
{
    __trap();
    return {};
}

// The absorber's surface: diffuse_light, black, front_face false, from -1, normal −unit(d), p the ray's origin (the
// path ends with what it found; the denoiser's guides read the normal). Stub: __trap() (unreachable until 91).
__device__ inline surface<float> absorber_surface(const mesh_tables& m, const ray<float>& r)
{
    __trap();
    return {};
}

// p moved off a face toward n: integer float steps (256·n per component) or n/65536 near 0 (Wächter & Binder 2019).
// Stub: __trap() (unreachable until 91: only a ray leaving a triangle moves its origin).
__device__ inline point3<float> offset_origin(const point3<float>& p, const vec3<float>& n)
{
    __trap();
    return p;
}
}
