// The GPU's lights list (see pdf_value and random of hittable_list in hittable.ixx and of quad in quad.ixx): the
// shapes that the lights strategy aims at, never the ones rays hit.
#pragma once

#include "quad.cuh"
#include "random.cuh"
#include "vec3.cuh"

namespace gpu
{
// A shape of the lights list, ready to be aimed at: 96 bytes in float, 176 in double. The scenes have one or two. Every
// thread of a warp that bounces off a diffuse surface reads all of them in the same order (lights_pdf_value), and each
// picks one (lights_random): ordinary global memory, whose loads the L1 cache serves to the whole warp.
template <real_number Real>
struct alignas(16) light
{
    quad<Real> shape; // the hit test of the density (made by make_quad, as the BVH's quads)
    vec3<Real> u;     // its sides, to draw a point of it
    Real area;        // |u × v|, computed in double
    vec3<Real> v;
    Real cdf;         // the weights of the lights up to this one, added up in double on the host (see lights_random)
    Real weight;      // the chance of aiming at it (light_desc::weight)
    int twin;         // the global id of the world's quad with exactly these numbers, or -1 (see lights_pdf_value)
    int unused[sizeof(Real) == sizeof(float) ? 2 : 1]; // padding, named: to a multiple of 16 bytes
};
static_assert(sizeof(light<float>) == 96 && sizeof(light<double>) == 176);

// The lights list as the kernel sees it
template <real_number Real>
struct lights_view
{
    const light<Real>* lights;
    int count; // 0 unless the strategy is lights
};

// quad::pdf_value: the density, per steradian, with which quad_light_random draws a direction from origin. Seen from
// origin, a patch dA of the quad at distance r, tilted by θ, covers the solid angle dA·cos θ / r², so a uniform point
// of the quad has the density r² / (cos θ · area). A direction that misses the quad has density 0: it is never drawn.
template <real_number Real>
__device__ Real quad_light_pdf(const light<Real>& l, const point3<Real>& origin, const vec3<Real>& direction)
{
    // The CPU's search from t = 0.001, closed at both ends (quad::hit), with the BVH's robust quad test. Either side
    // counts (|cos θ|): a direction toward a lamp's back gets a density too, and its ray then finds the back, which
    // emits nothing, as on the CPU.
    Real t;
    bool front_face;
    if (!hit_quad(l.shape, ray<Real>{origin, direction}, Real(0.001), infinity<Real>, t, front_face))
    {
        return Real(0);
    }
    const Real length_squared = direction.length_squared();
    const Real distance_squared = t * t * length_squared; // t counts in lengths of the direction
    const Real cosine = fabs(dot(direction, l.shape.normal)) * rsqrt(length_squared);
    return distance_squared / (cosine * l.area);
}

// quad::random: a uniform point of the quad, as a unit direction from origin. The CPU computes the point, a position,
// and then subtracts the origin; here the corner is taken relative to the origin first, and the sides are added to
// that, as hit_quad does: one difference of positions, then vectors the size of the distance. The direction has unit
// length because the next search starts at t = 0.001, counted in lengths of the direction (see quad.ixx).
template <real_number Real>
__device__ vec3<Real> quad_light_random(const light<Real>& l, const point3<Real>& origin, pcg32& rng)
{
    const Real a = random_real<Real>(rng); // a before b, as on the CPU
    const Real b = random_real<Real>(rng);
    return unit_vector((l.shape.q - origin) + a * l.u + b * l.v);
}

// hittable_list::pdf_value: the density of a direction drawn by lights_random, whichever light drew it, as any light
// whose solid angle contains the direction could have drawn it: the sum of each light's density times the chance of
// aiming at that light. (The CPU averages the densities of a list, Σ p_k / n, and those of the lists inside it in turn.
// The GPU adds up weight_k · p_k, where each weight is the product of the 1/n of every list that holds the shape (see
// light_desc in renderer.h): in exact arithmetic the same sum.)
//
// from is the global id of the primitive the bounce leaves, -1 after a particle of smoke. A light that is an exact copy
// of that primitive (its twin, a matte quad in its own lights list) is treated as the primitive itself, as closest_hit
// treats it: a ray never hits the quad it leaves, so from there its copy has density 0. That is the CPU's answer too,
// whose point lies within ~10^-13 of the quad's plane, so that its test finds no t ≥ 0.001. The GPU's float point lies
// a float step off the plane, and without the rule a direction drawn nearly in the plane would meet the copy beyond
// t = 0.001 and get a huge density instead of 0.
template <real_number Real>
__device__ Real lights_pdf_value(const lights_view<Real>& lights, const point3<Real>& origin,
                                 const vec3<Real>& direction, int from)
{
    Real sum = 0;
    for (int k = 0; k < lights.count; ++k)
    {
        const light<Real>& l = lights.lights[k];
        // twin is -1 for a light without one, and from is -1 after a particle: only a real id can match
        if (from >= 0 && l.twin == from)
        {
            continue;
        }
        sum += l.weight * quad_light_pdf(l, origin, direction);
    }
    return sum;
}

// hittable_list::random: one random number picks a light by the running sums of the weights (the inversion method for
// a discrete distribution, book 3, chapter 3), and that light draws the direction. The number is drawn for a single
// light too, as the CPU's random_int(0, 0) draws one. The last light takes whatever is left, so rounding in the sums
// can't run past the end of the list. With two lights the first sum is exactly 0.5, and the pick ξ < 0.5 is exactly the
// CPU's random_int(0, 1) == 0.
template <real_number Real>
__device__ vec3<Real> lights_random(const lights_view<Real>& lights, const point3<Real>& origin, pcg32& rng)
{
    const Real xi = random_real<Real>(rng);
    int k = 0;
    while (k < lights.count - 1 && !(xi < lights.lights[k].cdf))
    {
        ++k;
    }
    return quad_light_random(lights.lights[k], origin, rng);
}
}
