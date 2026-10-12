// The GPU's smoke and fog (see constant_medium.ixx): volumes of particles, tested after the surfaces.
#pragma once

#include "bvh.cuh"
#include "quad.cuh"
#include "random.cuh"
#include "sphere.cuh"
#include "vec3.cuh"

namespace gpu
{
// A volume of smoke as the kernel sees it: 80 bytes in float, 160 in double. Its boundary only says where the volume
// is: it is not among the BVH's primitives, and its material is not used.
template <real_number Real>
struct alignas(4 * sizeof(Real)) medium
{
    point3<Real> box_min;       // a box around the volume over the whole shutter time, rounded outward: a ray whose
    Real neg_inv_density;       // segment between t_min and its closest hit misses the box can't meet the volume
    point3<Real> box_max;       // there, and draws nothing, as the CPU's BVH skips the medium in that case
    Real unused;                // padding, named
    moving_sphere<Real> sphere; // the boundary when quad_count is 0 (its displacement is 0 if it stands still)
    int material;               // its phase function: an isotropic material
    int first_quad;             // otherwise the boundary is media_view::boundary_quads[first_quad, + quad_count)
    int quad_count;
    int unused_ints[sizeof(Real) == sizeof(float) ? 1 : 5]; // padding, named: to 16 bytes in float, 32 in double
};

// The media as the kernel sees them
template <real_number Real>
struct media_view
{
    const medium<Real>* media;
    const quad<Real>* boundary_quads; // the boundaries' quads in world coordinates
    int count;                        // 0 in a scene without smoke: the loop over the media doesn't run
    int base;                         // the global id of media[0], right after the surface primitives: a particle hit
                                      // in medium k gets the id base + k
};

// Where the ray's whole line crosses a medium's boundary: constant_medium::hit's two questions, the first crossing over
// t from -∞ to ∞ (the entry: behind the ray's origin when the ray starts inside) and the first one at least 0.0001
// beyond it (the exit). The GPU's boundaries are convex (see is_convex in renderer.cu), and a line crosses a convex
// shape's surface at most twice: the entry is the nearest crossing, the exit the farthest.
//
// The CPU asks the second question with an interval that starts at t_in + 0.0001. In float that sum rounds back to
// t_in once |t_in| > 2048, where floats are more than 0.0002 apart, and the entry would be found again as the exit. The
// difference of the two crossings is exact when they are close (they are floats within a factor of 2 of each other),
// so the GPU tests the difference instead: the same rule, without the rounding.
template <real_number Real>
__device__ bool boundary_crossings(const medium<Real>& m, const quad<Real>* boundary_quads, const ray<Real>& r,
                                   Real inverse_a, Real& t_in, Real& t_out)
{
    if (m.quad_count == 0)
    {
        // A sphere: sphere::hit over (-∞, ∞) gives its near root, and over (near + 0.0001, ∞), an open interval, its
        // far root if that lies strictly beyond. The sphere is where it is at the ray's time.
        return sphere_roots(m.sphere.at(r.time), r, inverse_a, t_in, t_out) && t_out - t_in > Real(0.0001);
    }
    // Quads that bound a convex shape: every crossing lies at the entry or at the exit
    t_in = infinity<Real>;
    t_out = -infinity<Real>;
    for (int i = m.first_quad; i < m.first_quad + m.quad_count; ++i)
    {
        Real t;
        bool front_face;
        if (hit_quad(boundary_quads[i], r, -infinity<Real>, infinity<Real>, t, front_face))
        {
            t_in = fmin(t_in, t);
            t_out = fmax(t_out, t);
        }
    }
    // Closed, as a quad's interval is. No crossing at all gives -∞ - ∞ = -∞, a single one 0: no volume.
    return t_out - t_in >= Real(0.0001);
}

// constant_medium::hit for one medium: does the ray meet one of its particles at a t in [t_min, t_max], t_max being the
// closest event found so far? If so, at which t.
//
// The ray collides with a particle on each short stretch dL with probability density·dL, so the distance to the first
// collision is exponential: -ln(ξ) / density for a random ξ in [0, 1). The CPU draws it while it searches its BVH,
// against the closest hit found so far; the GPU after the search, against the closest event so far (the closest
// surface, or a particle of a medium tested before). The image is the same: the collision's t, the distance added to
// where the ray's stretch inside the volume begins, doesn't depend on that limit, which only decides whether a
// collision beyond it is reported, and such a collision would lose to the nearer event anyway. So in any order of the
// tests the first event is the nearest of the closest surface and the media's collisions, with the same distribution.
// Only which random numbers are drawn differs (a medium entered only beyond a nearer event draws none): other noise,
// the same picture. (That the exponential has no memory, the chance of a collision in the next stretch not depending
// on how far the ray came without one, is what lets every new ray, from a particle or from a surface inside the
// volume, draw a fresh distance from its own start, as both renderers do.)
template <real_number Real>
__device__ bool medium_collision(const medium<Real>& m, const quad<Real>* boundary_quads, const ray<Real>& r,
                                 Real inverse_a, const vec3<Real>& inverse_direction, const vec3<Real>& origin_scaled,
                                 Real t_min, Real t_max, pcg32& rng, Real& t)
{
    // A ray whose segment [t_min, t_max] misses the volume's box can't meet the volume there. The box is conservative,
    // so the test changes no result, only the work.
    if (box_entry(m.box_min, m.box_max, inverse_direction, origin_scaled, t_min, t_max) == infinity<Real>)
    {
        return false;
    }
    Real line_in;
    Real line_out;
    if (!boundary_crossings(m, boundary_quads, r, inverse_a, line_in, line_out))
    {
        return false;
    }
    // The part of the line inside the volume that lies ahead of the origin and before the closest event
    Real t_in = fmax(line_in, t_min);
    const Real t_out = fmin(line_out, t_max);
    if (t_in >= t_out)
    {
        return false;
    }
    t_in = fmax(t_in, Real(0)); // the CPU's max(t_in, 0), which t_min = 0.001 already covers
    // t counts in lengths of the direction, which isn't of length 1 for camera rays: the distance in world units
    const Real length = sqrt(r.direction.length_squared());
    const Real distance_inside = (t_out - t_in) * length;
    // ξ = 0 gives log(0) = -∞, so the distance is +∞ and the ray passes, as on the CPU
    const Real hit_distance = m.neg_inv_density * log(random_real<Real>(rng));
    if (hit_distance > distance_inside)
    {
        return false; // passed through without meeting a particle
    }
    t = t_in + hit_distance / length;
    return true;
}
}
