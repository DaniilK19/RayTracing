// The GPU's lights list (see pdf_value and random of hittable_list in hittable.ixx, of quad in quad.ixx and of sphere
// in sphere.ixx): the shapes that the lights and mixture strategies aim at, never the ones rays hit.
#pragma once

#include "material.cuh"
#include "pdf.cuh"
#include "quad.cuh"
#include "random.cuh"
#include "sphere.cuh"
#include "vec3.cuh"

namespace gpu
{
// A shape of the lights list, ready to be aimed at: 112 bytes in float, 224 in double (16 of them padding, see unused).
// The scenes have one or two.
// Every thread of a warp that bounces off a diffuse surface reads all of them in the same order (lights_pdf_value), and
// each picks one (lights_random): ordinary global memory, whose loads the L1 cache serves to the whole warp.
template <real_number Real>
struct alignas(16) light
{
    // light_shape::quad: the hit test of the density (made by make_quad, as the BVH's quads). The type is written
    // gpu::quad: a member may not change what an unqualified name means in its class, and this one is named quad too.
    gpu::quad<Real> quad;
    vec3<Real> u;        // light_shape::quad: its sides, to draw a point of it
    Real area;           // light_shape::quad: |u × v|, computed in double
    vec3<Real> v;
    Real cdf;            // the weights of the lights up to this one, added up in double on the host (lights_random)
    point3<Real> center; // light_shape::sphere: where it is at time 0, where the CPU aims too
    Real radius;         // light_shape::sphere
    Real weight;         // the chance of aiming at it (light_desc::weight)
    light_shape kind;
    int twin; // the global id of the world's primitive with exactly these numbers (a quad, or a sphere that stands
              // still), or -1 (see lights_pdf_value)
    // Padding, named, so that light<Real> l{} sets every byte copied to the GPU: in float the members end at 108, and
    // one int makes 112, a multiple of 16. In double they already end at 208 = 13·16, but an array can't have 0
    // elements, and 1 to 3 ints would leave unnamed padding up to 224 anyway: 4 fill it.
    int unused[sizeof(Real) == sizeof(float) ? 1 : 4];
};
static_assert(sizeof(light<float>) == 112 && sizeof(light<double>) == 224);

// The lights list as the kernel sees it
template <real_number Real>
struct lights_view
{
    const light<Real>* lights;
    int count; // 0 unless the strategy is lights or the mixture
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
    if (!hit_quad(l.quad, ray<Real>{origin, direction}, Real(0.001), infinity<Real>, t, front_face))
    {
        return Real(0);
    }
    const Real length_squared = direction.length_squared();
    const Real distance_squared = t * t * length_squared; // t counts in lengths of the direction
    const Real cosine = fabs(dot(direction, l.quad.normal)) * rsqrt(length_squared);
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
    return unit_vector((l.quad.q - origin) + a * l.u + b * l.v);
}

// 1 - cos θmax of the cone that a sphere of the given radius fills, seen from √distance_squared away, in the CPU's form
// sin²θmax / (1 + cos θmax) (see one_minus_cos_theta_max in sphere.ixx), which keeps its digits when the sphere looks
// small. From the light scene's far ground, ~110 units from its lamp of radius 2, sin²θmax is ~3.3·10^-4, where
// 1 - √(1 - sin²θmax) would lose about four of float's seven digits.
template <real_number Real>
__device__ Real one_minus_cos_theta_max(Real radius, Real distance_squared)
{
    const Real sin_squared = radius * radius / distance_squared;
    return sin_squared / (Real(1) + sqrt(Real(1) - sin_squared));
}

// |center - origin|², the distance² that decides whether origin is inside a sphere light (sphere::pdf_value and random
// ask D² < r²). sphere_light_pdf and sphere_light_random must decide alike, or directions would be drawn with one
// density and weighed with another; and for a point on the sphere's own surface (a matte ball in its own lights list)
// D² lies within a rounding of r², where one last bit decides. Written as x·x + y·y + z·z, the sum would be fused into
// multiply-adds by nvcc in each of the two places on its own, and which product gets fused depends on the code around
// it (see first_book_scene in scene.cuh). So the fused multiply-adds are written out, in the order nvcc chose for both
// in Step 57.
template <real_number Real>
__device__ Real sphere_light_distance_squared(const vec3<Real>& to_center)
{
    return fma(to_center.z, to_center.z, fma(to_center.x, to_center.x, mul_rn(to_center.y, to_center.y)));
}

// sphere::pdf_value: the density, per steradian, with which sphere_light_random draws a direction from origin, uniform
// within the cone the sphere fills: 1 / (its solid angle 2π·(1 - cos θmax)). The sphere is where it is at time 0, as
// the CPU aims at it. own: the bounce leaves this very sphere (its twin in the world, see lights_pdf_value).
//
// The inside test comes first, as on the CPU, so a point on the sphere's own surface takes it whenever rounding puts
// the point inside: then every direction gets 1/(4π), outward ones too, and the rule for the own sphere below applies
// only to the other points. Under the mixture both answers are unbiased, and only the noise depends on which one a
// point gets. Under -p lights, whose estimate is biased by design, the expected image depends on it too, and rounding
// decides it differently on the two renderers: the GPU's points of a ball in its own lights list fall inside about half
// the time, the CPU's about 40% (measured by forcing every surface point outside, then inside: the CPU's image lies at
// 40% of the way). So that scene alone renders differently under -p lights, by 0.3% (an adversarial scene of Step 57).
template <real_number Real>
__device__ Real sphere_light_pdf(const light<Real>& l, const point3<Real>& origin, const vec3<Real>& direction,
                                 bool own)
{
    const vec3<Real> to_center = l.center - origin; // relative to the origin
    const Real distance_squared = sphere_light_distance_squared(to_center);
    if (distance_squared < l.radius * l.radius)
    {
        return uniform_sphere_density<Real>; // from inside, every direction hits it: see sphere_light_random
    }
    // From its own surface the CPU's double point meets the sphere only going inward, at the far root (its near root,
    // ~0, lies below 0.001): closest_hit's rule for the sphere a ray leaves. The float point lies a float step off the
    // surface, where a grazing outward direction could find a root beyond 0.001.
    if (own && !(dot(direction, to_center) > Real(0)))
    {
        return Real(0);
    }
    // A direction that misses the sphere has density 0: the CPU's search over (0.001, ∞), with the robust test of the
    // BVH's spheres (only the far root from its own surface)
    Real t;
    bool entering;
    if (!hit_sphere(sphere<Real>{l.center, l.radius}, ray<Real>{origin, direction},
                    Real(1) / direction.length_squared(), Real(0.001), infinity<Real>, own, t, entering))
    {
        return Real(0);
    }
    return Real(1) / (Real(2) * pi<Real> * one_minus_cos_theta_max(l.radius, distance_squared));
}

// sphere::random and random_to_sphere (sphere.ixx): a direction uniform within the cone the sphere fills, drawn around
// the z axis by the inversion method (book 3, chapter 7) and turned so that the axis points at the center
template <real_number Real>
__device__ vec3<Real> sphere_light_random(const light<Real>& l, const point3<Real>& origin, pcg32& rng)
{
    const vec3<Real> to_center = l.center - origin;
    const Real distance_squared = sphere_light_distance_squared(to_center);
    if (distance_squared < l.radius * l.radius)
    {
        // From inside, every direction hits the sphere: all of them, uniformly (the book's formula takes the square
        // root of a negative number here). The same test on the same number as in sphere_light_pdf: both take it.
        return random_unit_vector<Real>(rng);
    }
    const Real one_minus_cos_max = one_minus_cos_theta_max(l.radius, distance_squared);
    const Real r1 = random_real<Real>(rng); // φ = 2π·r1; r1 before r2, as on the CPU
    const Real r2 = random_real<Real>(rng);
    Real sin_phi;
    Real cos_phi;
    sincospi(Real(2) * r1, &sin_phi, &cos_phi);
    const Real one_minus_z = r2 * one_minus_cos_max;                    // 1 - cos θ
    const Real sin_theta = sqrt(one_minus_z * (Real(2) - one_minus_z)); // √((1 - z)(1 + z)): no 1 - z² to cancel
    // Any orthonormal basis around the axis gives the same distribution, as the cone is symmetric around it: the one
    // of the cosine directions (around in material.cuh), where the CPU uses its onb
    // (rsqrt(distance_squared) · to_center is the unit vector toward the center, unit_vector's own formula on the same
    // distance²)
    return around(rsqrt(distance_squared) * to_center,
                  vec3<Real>{cos_phi * sin_theta, sin_phi * sin_theta, Real(1) - one_minus_z});
}

// hittable_list::pdf_value: the density of a direction drawn by lights_random, whichever light drew it, as any light
// whose solid angle contains the direction could have drawn it: the sum of each light's density times the chance of
// aiming at that light. (The CPU averages the densities of a list, Σ p_k / n, and those of the lists inside it in turn.
// The GPU adds up weight_k · p_k, where each weight is the product of the 1/n of every list that holds the shape (see
// light_desc in renderer.h): in exact arithmetic the same sum.)
//
// from is the global id of the primitive the bounce leaves, -1 after a particle of smoke. A light that is an exact copy
// of that primitive (its twin: a matte object in its own lights list) is treated as the primitive itself, as
// closest_hit treats it. A ray never hits the quad it leaves, so from there a quad's copy has density 0. That is the
// CPU's answer too, whose point lies within ~10^-13 of the quad's plane, so that its test finds no t ≥ 0.001. The GPU's
// float point lies a float step off the plane, and without the rule a direction drawn nearly in the plane would meet
// the copy beyond t = 0.001 and get a huge density instead of 0. A sphere's copy is met only going inward, when the
// point rounds outside it (see sphere_light_pdf).
template <real_number Real>
__device__ Real lights_pdf_value(const lights_view<Real>& lights, const point3<Real>& origin,
                                 const vec3<Real>& direction, int from)
{
    Real sum = 0;
    for (int k = 0; k < lights.count; ++k)
    {
        const light<Real>& l = lights.lights[k];
        // twin is -1 for a light without one, and from is -1 after a particle: only a real id can match
        const bool own = from >= 0 && l.twin == from;
        if (l.kind == light_shape::quad)
        {
            sum += own ? Real(0) : l.weight * quad_light_pdf(l, origin, direction);
        }
        else
        {
            sum += l.weight * sphere_light_pdf(l, origin, direction, own);
        }
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
    const light<Real>& l = lights.lights[k];
    return l.kind == light_shape::quad ? quad_light_random(l, origin, rng) : sphere_light_random(l, origin, rng);
}
}
