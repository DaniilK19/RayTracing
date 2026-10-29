// The quality kernel's lamps: the world's own glowing quads and spheres, each a light the light sampler can aim at
// (light_sampler.cuh). Unlike the books' lights list (lights.cuh), which holds copies of shapes chosen by the scene,
// and in the glass scene a ball that doesn't glow, this set is made of the primitives rays hit: every quad and every
// sphere that stands still whose material is diffuse_light with one solid color. Instances are baked into world
// coordinates before the BVH is built, so a lamp inside rotate_y and translate is found like any other. A lamp with a
// texture (its light would vary over it) or one that moves (where would the sample aim?) stays out of the set: a ray
// that finds it counts it in full, still unbiased, as no light sample could have drawn it. No scene has one.
//
// Each lamp draws a direction from a point x with a known density per steradian, the light technique's p_L of mis.cuh:
// - A quad, by area, as the books' quad_light_random draws it: a uniform point q + a·u + b·v. Seen from x, a patch dA
//   at distance d, tilted by θ_l from the direction, covers dA·cos θ_l / d², so the point's direction has the density
//   d² / (cos θ_l · A). A quad glows on its front only, the side u × v points to (hit_record's front_face): a point of
//   its back gives no sample, and no shadow ray, and lamp_pdf gives a ray that finds the back the density 0: f·Le is 0
//   there, so Veach's condition W1 (mis.cuh) doesn't care.
// - A sphere, uniformly within the cone it fills, as the books' sphere_light_random draws it: the density
//   1 / (2π(1 − cos θmax)) is the same for every direction of the cone, and sin θmax = r / D. From inside the sphere
//   there is no sample: only its inside is in view, and a lamp glows on its outside only (a ray that enters it).
//
// The density of the hit lamp alone weighs a ray that finds it (lamp_pdf), not the sum over every lamp whose solid
// angle holds the direction, as the books' lights_pdf_value adds them up: a light sample of another lamp never reaches
// that point of this one (its shadow ray ends at its own lamp), so only this lamp's technique could have made the same
// path (path-space MIS, as pbrt-v4's PathIntegrator weighs an emissive hit).
//
// The host part at the end builds the set from the scene (lamp_table.cu).
#pragma once

#include <span>
#include <vector>

#include "lights.cuh"
#include "material.cuh"
#include "quad.cuh"
#include "random.cuh"
#include "renderer.h"
#include "vec3.cuh"

namespace gpu
{
// One lamp of the light set, 80 bytes, read with 16-byte loads. The host builds it in double and rounds to float, from
// the same numbers as the BVH's own primitive, so the lamp a light sample aims at is the one rays hit.
struct alignas(16) emitter
{
    point3<float> corner;  // quad: q; sphere: the center
    float size;            // quad: the area |u × v| (computed in double on the host); sphere: the radius
    vec3<float> u;         // quad only
    int kind;              // static_cast<int>(light_shape::quad or ::sphere)
    vec3<float> v;         // quad only
    int primitive;         // its global id in the BVH: its own shadow ray never counts it as a blocker
    vec3<float> normal;    // quad: the unit u × v, the side it lights (make_quad's, bit for bit)
    float unused0;         // padding, named, so that every byte copied to the GPU is set
    color<float> radiance; // its one color, from the front side only
    float unused1;
};
static_assert(sizeof(emitter) == 80);

// A direction toward a lamp, its distance (the shadow ray's t_max) and its density per steradian, without the chance
// of picking the lamp (light_sampler.cuh multiplies it in)
struct lamp_sample
{
    vec3<float> wi;
    float distance;
    float pdf;
};

// The densities, each in one place, so that a light sample and the ray that later finds the same lamp compute them
// with the same operations: d² / (cos θ_l · A) of a quad's point, and the cone's 1 / (2π(1 − cos θmax)) with the books'
// pinned 1 − cos θmax (one_minus_cos_theta_max in lights.cuh, which keeps its digits for a lamp seen small)
__device__ inline float quad_lamp_density(float distance_squared, float cos_light, float area)
{
    return distance_squared / (cos_light * area);
}

__device__ inline float sphere_lamp_density(float radius, float distance_squared)
{
    return 1.0f / (2.0f * pi<float> * one_minus_cos_theta_max(radius, distance_squared));
}

// A direction from x toward lamp e. It draws exactly 2 numbers in every call, before it knows whether x can see the
// lamp's glowing side, so the numbers after it don't depend on that. False: x sees the quad's back (or its edge), or
// lies inside the sphere.
__device__ inline bool sample_lamp(const emitter& e, const point3<float>& x, pcg32& rng, lamp_sample& out)
{
    const float a = random_real<float>(rng); // a before b, as quad_light_random in lights.cuh
    const float b = random_real<float>(rng);
    if (e.kind == static_cast<int>(light_shape::quad))
    {
        // The point relative to x as hit_quad and quad_light_random compute it: the corner relative to x first (one
        // difference of positions), then the sides, vectors the size of the distance
        const vec3<float> to = (e.corner - x) + a * e.u + b * e.v;
        const float distance_squared = to.length_squared();
        const float inverse_distance = rsqrt(distance_squared);
        out.wi = inverse_distance * to;
        out.distance = distance_squared * inverse_distance;
        const float cos_light = -dot(out.wi, e.normal);
        if (!(cos_light > 0.0f)) // NaN too: a degenerate quad, or x on the sampled point
        {
            return false;
        }
        out.pdf = quad_lamp_density(distance_squared, cos_light, e.size);
        return true;
    }
    // The inside test of the books' sphere lights, on the same distance², with its multiply-adds pinned (see
    // sphere_light_distance_squared): lamp_pdf decides from this very number at a hit
    const vec3<float> to_center = e.corner - x;
    const float distance_squared = sphere_light_distance_squared(to_center);
    const float radius_squared = e.size * e.size;
    if (!(distance_squared > radius_squared))
    {
        return false;
    }
    // Uniform in the cone around the z axis by the inversion method (sphere_light_random in lights.cuh): φ = 2π·a,
    // 1 − cos θ = b·(1 − cos θmax), then turned so that z points at the center (around in material.cuh)
    const float one_minus_cos_max = one_minus_cos_theta_max(e.size, distance_squared);
    float sin_phi;
    float cos_phi;
    sincospi(2.0f * a, &sin_phi, &cos_phi);
    const float one_minus_z = b * one_minus_cos_max;
    const float sin_squared = one_minus_z * (2.0f - one_minus_z); // (1 − z)(1 + z): no 1 − z² to cancel
    const float sin_theta = sqrt(sin_squared);
    const float cos_theta = 1.0f - one_minus_z;
    const float inverse_distance = rsqrt(distance_squared);
    out.wi = around(inverse_distance * to_center, vec3<float>{cos_phi * sin_theta, sin_phi * sin_theta, cos_theta});
    // The near root along wi, where the ray enters the sphere: D·cos θ − √(r² − D²·sin²θ). Inside the cone
    // D²·sin²θ ≤ r² in exact arithmetic, and only rounding can make the radicand negative, at the cone's edge (a
    // grazing ray, whose two roots meet): clamped to 0. The shadow ray skips the lamp itself, so a t_max a rounding off
    // can't shadow the sample.
    const float distance = distance_squared * inverse_distance;
    out.distance = distance * cos_theta - sqrt(fmax(0.0f, fma(-distance_squared, sin_squared, radius_squared)));
    out.pdf = sphere_lamp_density(e.size, distance_squared);
    return true;
}

// The density with which sample_lamp draws the unit direction unit_dir from x, for a ray along it that hits lamp e's
// glowing side at `distance`: a quad's from the hit's own distance and angle, a sphere's the cone's, which is the same
// for every direction that hits it. (For a direction that misses the lamp the value means nothing: the integrator asks
// only at a hit.)
__device__ inline float lamp_pdf(const emitter& e, const point3<float>& x, const vec3<float>& unit_dir, float distance)
{
    if (e.kind == static_cast<int>(light_shape::quad))
    {
        const float cos_light = -dot(unit_dir, e.normal);
        return cos_light > 0.0f ? quad_lamp_density(distance * distance, cos_light, e.size) : 0.0f;
    }
    const float distance_squared = sphere_light_distance_squared(e.corner - x);
    return distance_squared > e.size * e.size ? sphere_lamp_density(e.size, distance_squared) : 0.0f;
}

// Host code (lamp_table.cu): the light set of a scene, as quality.cu copies it to the GPU (lamps_on_device.cuh)
struct lamp_table
{
    std::vector<emitter> lamps;
    std::vector<int> lamp_of; // size = the surface primitives (quad_base + quads), -1 for a primitive that is no lamp
};

// The scene's lamps (see the top of this file), in the scene's order, its spheres before its quads. sphere_ids and
// quad_ids are the BVH's global ids of scene.spheres and scene.quads (bvh::sphere_ids and quad_ids). Throws
// std::invalid_argument when the ids don't match the scene, or a lamp's material or texture doesn't exist. How many
// lamps it finds, without the ids, is count_lamps (renderer.h).
lamp_table build_lamps(const scene_desc& scene, std::span<const int> sphere_ids, std::span<const int> quad_ids);
}
