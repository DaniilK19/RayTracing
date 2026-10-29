// The shadow ray of a light sample: does anything lie between the vertex and the point of the light it aimed at? A
// question with a yes or no answer, so the search can stop at the first primitive it finds, where closest_hit (bvh.cuh)
// must look on for a nearer one. An any-hit search: OptiX's OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT does the same on the
// ray tracing hardware. And if nothing blocks it, how much of the light the smoke and fog on the way let through.
#pragma once

#include "bvh.cuh"
#include "constant_medium.cuh"
#include "quad.cuh"
#include "sphere.cuh"
#include "vec3.cuh"

namespace gpu
{
// Anything in (0.001, t_max) of r? closest_hit's walk with a fixed t_max, the primitive `skip` not tested (the lamp the
// light sample aimed at: its own point, a rounding beyond or before t_max, mustn't shadow it), returning at the first
// hit. Every primitive of the BVH blocks the light, glass, metal and the other lamps included: the light that passes
// through glass or off a mirror arrives by the material's own rays (the books' scatter), which a light sample can't
// draw. The volumes of smoke aren't the BVH's: their particles thin the light out instead (the transmittance).
//
// The rules of closest_hit for the primitive the ray leaves (from, inward), the same tests (hit_quad over the closed
// interval [0.001, t_max], hit_sphere over the open one), and a moving sphere where it is at the ray's time: the shadow
// ray leaves the vertex exactly as the path's next ray would. With t_max fixed, a node is pushed only if the ray enters
// its box before t_max, and every node pushed is visited, so the stack holds node ids alone: 32 ints, 128 bytes of the
// thread's local memory, half of closest_hit's stack of ids and distances.
__device__ inline bool occluded(const bvh_view<float>& tree, const ray<float>& r, float t_max, int from, bool inward,
                                int skip)
{
    const float t_min = 0.001f;
    if (tree.node_count == 0)
    {
        return false;
    }
    const float inverse_a = 1.0f / r.direction.length_squared();
    const vec3<float> inverse_direction{safe_inverse(r.direction.x), safe_inverse(r.direction.y),
                                        safe_inverse(r.direction.z)};
    const vec3<float> origin_scaled = r.origin * inverse_direction;

    int stack[bvh_stack_size];
    int stack_size = 0;
    int node_index = 0; // the root
    while (node_index >= 0)
    {
        const bvh_node<float> node = tree.nodes[node_index];
        if (node.is_leaf())
        {
            const int end = node.index + leaf_size(node);
            if (leaf_kind(node) == primitive_kind::quad)
            {
                for (int i = node.index; i < end; ++i)
                {
                    const int id = tree.quad_base + i;
                    if (id == from || id == skip)
                    {
                        continue; // a ray never hits the quad it leaves (see closest_hit), nor the lamp it aims at
                    }
                    const quad<float> face = tree.quads[i];
                    float t;
                    bool front_face;
                    if (hit_quad(face, r, t_min, t_max, t, front_face))
                    {
                        return true;
                    }
                }
            }
            else
            {
                const bool moving = leaf_kind(node) == primitive_kind::moving_sphere;
                for (int i = node.index; i < end; ++i)
                {
                    const int id = moving ? tree.moving_base + i : i;
                    const bool own = id == from;
                    if ((own && !inward) || id == skip)
                    {
                        continue; // a ray going out of a sphere can't hit it (see closest_hit), nor the lamp it aims at
                    }
                    const sphere<float> s = moving ? tree.moving_spheres[i].at(r.time) : tree.spheres[i];
                    float t;
                    bool entering;
                    if (hit_sphere(s, r, inverse_a, t_min, t_max, own, t, entering)) // own: its far root only
                    {
                        return true;
                    }
                }
            }
            node_index = -1; // continue from the stack
        }
        else
        {
            // Both children's boxes, the nearer one first: an occluder found there spares the other one's walk. The
            // farther one waits on the stack if the ray enters it before t_max.
            const int first = node_index + 1; // depth first: right after its parent
            const int second = node.index;
            const float t_first = box_entry(tree.nodes[first], inverse_direction, origin_scaled, t_min, t_max);
            const float t_second = box_entry(tree.nodes[second], inverse_direction, origin_scaled, t_min, t_max);
            const bool second_is_nearer = t_second < t_first;
            if (fmax(t_first, t_second) < infinity<float>)
            {
                stack[stack_size++] = second_is_nearer ? first : second;
            }
            node_index = fmin(t_first, t_second) < infinity<float> ? (second_is_nearer ? second : first) : -1;
        }
        if (node_index < 0 && stack_size > 0)
        {
            node_index = stack[--stack_size];
        }
    }
    return false;
}

// The chance that (0.001, t_max) of r crosses every volume of smoke without meeting a particle: the expected value of
// the kernel's own free-flight test (medium_collision in constant_medium.cuh) over exactly the segment it would test.
// medium_collision lets a ray pass a medium when neg_inv_density·log(ξ) exceeds the length L of its stretch inside,
// that is when ξ < e^(−ρL) with the density ρ = −1/neg_inv_density: with probability e^(−ρL) (to the 2^-24 steps of
// ξ), for each medium on its own, so the chance to pass them all is the product. The path's own ray along the shadow
// ray would pass with that chance, and so the light the material's ray finds carries the same factor on average: it is
// common to both techniques, and the weights (mis.cuh) don't change.
//
// Computed, not drawn: the drawn test answers 0 or 1, which adds a relative noise of √((1 − T)/T) to every light
// sample, 23% through the final scene's haze (T ≈ 0.95) and 206% through a block of the smoke scene (T ≈ 0.19); its
// expected value adds none and draws no random number (Novák et al., "Monte Carlo Methods for Volumetric Light
// Transport Simulation", Eurographics STAR 2018, section 5.2: the expected-value estimator, against the track-length
// one).
//
// The steps are medium_collision's own: the box test against [0.001, t_max]; the boundary's two crossings with their
// 0.0001 exit rule (boundary_crossings, a sphere where it is at the ray's time); the segment [max(line_in, 0.001),
// min(line_out, t_max)], nothing if its start isn't below its end; its length in world units. Every volume is
// bounded, so the environment's t_max = ∞ ends at the boundary's exit.
__device__ inline float transmittance(const media_view<float>& media, const ray<float>& r, float t_max)
{
    if (media.count == 0)
    {
        return 1.0f;
    }
    const float t_min = 0.001f;
    const float inverse_a = 1.0f / r.direction.length_squared();
    const vec3<float> inverse_direction{safe_inverse(r.direction.x), safe_inverse(r.direction.y),
                                        safe_inverse(r.direction.z)};
    const vec3<float> origin_scaled = r.origin * inverse_direction;
    const float length = sqrt(r.direction.length_squared()); // t counts in lengths of the direction
    float exponent = 0.0f; // Σ L_k / neg_inv_density_k = −Σ ρ_k·L_k
    for (int k = 0; k < media.count; ++k)
    {
        const medium<float>& m = media.media[k];
        if (box_entry(m.box_min, m.box_max, inverse_direction, origin_scaled, t_min, t_max) == infinity<float>)
        {
            continue;
        }
        float line_in;
        float line_out;
        if (!boundary_crossings(m, media.boundary_quads, r, inverse_a, line_in, line_out))
        {
            continue;
        }
        float t_in = fmax(line_in, t_min);
        const float t_out = fmin(line_out, t_max);
        if (t_in >= t_out)
        {
            continue;
        }
        t_in = fmax(t_in, 0.0f); // medium_collision's max(t_in, 0), which t_min already covers
        exponent += (t_out - t_in) * length / m.neg_inv_density;
    }
    return ::expf(exponent); // e^0 = 1 exactly where no volume is crossed
}
}
