// The GPU's sphere (see sphere.ixx) and the ray-sphere test, made robust for float.
#pragma once

#include "vec3.cuh"

namespace gpu
{
// A sphere as the GPU stores it: center and radius, 16 bytes in float. The alignment lets one load instruction read
// all of it (a GPU thread loads up to 16 bytes at once). The material index is kept in another array (see bvh): the
// search for the closest hit reads many spheres, but only the one it finds needs its material.
template <real_number Real>
struct alignas(4 * sizeof(Real)) sphere
{
    point3<Real> center;
    Real radius;
};

// A sphere that moves while the shutter is open: at time t its center is center + t·displacement (sphere.ixx keeps
// the center's path as a ray). 32 bytes in float: two 16-byte loads. A kind of its own (see primitive_kind in
// bvh.cuh), so that a sphere that stands still keeps its single load.
template <real_number Real>
struct alignas(4 * sizeof(Real)) moving_sphere
{
    point3<Real> center; // at time 0
    Real radius;
    vec3<Real> displacement;
    Real unused; // padding, named: the alignment makes the struct 32 bytes in float either way

    // The sphere where it is at the given time. A ray and all its bounces share one time, so the tests of this sphere
    // along one path (in closest_hit) all compute the same center, bit for bit: a bounce leaves the sphere where it was
    // hit, and the self-intersection rule for the sphere the ray leaves holds as for a sphere that stands still.
    __device__ sphere<Real> at(Real time) const { return {center + time * displacement, radius}; }
};

// What only the closest hit reads about a sphere besides its material (see bvh): the two numbers that let a checker
// texture read a point exactly on the sphere (see onto_sphere). 8 bytes in float: one load.
template <real_number Real>
struct alignas(2 * sizeof(Real)) sphere_frame
{
    Real k;     // |center|² - radius², computed in double on the host from the sphere the GPU tests, then rounded
    Real limit; // 2·radius·|center|: onto_sphere helps only where its own rounding stays below this. 0 for a moving
                // sphere, whose center, and so k, change with time: onto_sphere leaves its points alone.
};

// The distance t (in lengths of the ray's direction) at which the ray hits the sphere, if a hit lies in (t_min, t_max),
// and which root that is: `entering` is true for the near root, where the ray enters the sphere, and false for the far
// one, where it leaves (see trace_first_book_ray in camera.cuh for why that matters). inverse_a is 1 / |direction|²,
// the same for every sphere, so the caller computes it once per ray. far_only takes only the far root: for a ray that
// starts on this very sphere and goes inside it (see closest_hit).
//
// The CPU's sphere::hit returns a std::optional, but GPU code can't call the standard library's functions: they are
// compiled for the CPU only. (nvcc rejects std::optional in GPU code unless an experimental flag,
// --expt-relaxed-constexpr, lets it call constexpr CPU functions; CUDA's own library has a cuda::std::optional.) So
// the GPU functions answer the C way: a bool says whether there is a result, and the result goes to out-parameters
// (or an index of -1 means "none", as in closest_hit).
template <real_number Real>
__device__ bool hit_sphere(const sphere<Real>& s, const ray<Real>& r, Real inverse_a, Real t_min, Real t_max,
                           bool far_only, Real& t, bool& entering)
{
    // As on the CPU, |origin + t·direction - center|² = radius² is the quadratic a·t² - 2h·t + c = 0, with a = |d|²,
    // h = d·oc, c = |oc|² - radius² and oc = center - origin, and its roots are (h ± √(h² - a·c)) / a. But the CPU
    // computes h² - a·c in double, and in float this difference loses its digits when the sphere is small compared to
    // its distance: h² and a·c are then nearly equal numbers of the size a·|oc|², while their difference is of the
    // size a·radius², so most of the digits cancel (Haines et al., "Precision Improvements for Ray/Sphere
    // Intersection", Ray Tracing Gems, 2019, chapter 7). The same number from a vector instead: l = oc - (h/a)·d goes
    // from the point of the ray's line closest to the center to the center, and h² - a·c = a·(radius² - |l|²). For a
    // hit |l| ≤ radius, so nothing larger than radius² is subtracted.
    //
    // The three squares that make up |l|² are subtracted from radius² one at a time, each inside an fma, so their sum
    // is never rounded on its own. Rounded to the floats near radius², every |l|² less than half a step above radius²
    // (0.03 for the ground, radius 1000) would become exactly radius²: a discriminant of 0, a hit, for a ray that
    // passes just outside the sphere. At the ground's horizon these false hits outnumbered the opposite error about
    // 4 to 1 and darkened the horizon in float, by 13 standard errors in a test that gives float and double the same
    // random numbers. With the fmas both errors are equally rare, and the darkening is gone.
    const vec3<Real> oc = s.center - r.origin;
    const Real h_over_a = dot(r.direction, oc) * inverse_a; // the t of the line's point closest to the center
    const vec3<Real> l = oc - h_over_a * r.direction;
    // radius² - |l|², which is (h² - a·c) / a
    const Real discriminant = fma(-l.x, l.x, fma(-l.y, l.y, fma(-l.z, l.z, s.radius * s.radius)));
    if (discriminant < 0)
    {
        return false;
    }
    const Real half_chord = sqrt(discriminant * inverse_a); // the roots are h/a ± this

    // The nearest root in the interval, as on the CPU
    t = h_over_a - half_chord;
    entering = !far_only && t > t_min;
    if (!entering)
    {
        t = h_over_a + half_chord;
    }
    return t_min < t && t < t_max;
}

// The texture coordinates of a point of the sphere (sphere::hit in sphere.ixx): the longitude and latitude of its
// outward unit normal, both from 0 to 1. u goes around the y axis from the direction -x, v from the south pole (-y) to
// the north pole.
template <real_number Real>
__device__ void sphere_uv(const vec3<Real>& outward, Real& u, Real& v)
{
    const Real phi = atan2(-outward.z, outward.x) + pi<Real>;
    // Rounding can put |y| a little above 1, where acos is NaN: clamped, as on the CPU
    const Real theta = acos(fmin(fmax(-outward.y, Real(-1)), Real(1)));
    u = phi / (Real(2) * pi<Real>);
    v = theta / pi<Real>;
}

// A point computed near the sphere, moved onto it, for a checker texture to read. A float hit point is known relative
// to the sphere's center only to about a float step of the center's coordinates: on the ground of radius 1000 it lies
// ~3·10^-5 above or below the surface, as hit_sphere rounds center - origin at 1000 (see closest_hit in bvh.cuh).
// Mostly that doesn't matter, but a checker's colors jump where the surface crosses a face of its cubes, and the ground
// of the bouncing scene touches the face y = 0 at the origin, right under the glass ball in the middle of the image.
// Near there the ground is at y = -(x² + z²)/2000, a little below 0, so the CPU, whose points are good to 10^-13, takes
// the cubes below y = 0. The float point's error is as large as that height out to x² + z² ≈ 0.06, so there it often
// lands above 0 and takes the cubes above, of the other color: a patch of wrong colors, a bias no number of samples
// removes.
//
// One step of Newton's method for f(p) = |p - c|² - radius² = 0 along the outward normal: p - (f / (2·radius))·outward,
// as f / (2·radius) is how far p lies outside the sphere (to first order). f must not be computed as |p - c|² - r²:
// p - c is a vector of the size of c, rounded as in hit_sphere. Multiplied out, f = |p|² - 2·p·c + k with
// k = |c|² - r² from the host (sphere_frame, computed in double): near the world's origin every term is small, for the
// ground k is exactly 0, and p·c is -1000·p.y. f is then good to about a float step of |p|² + 2|p·c| + |k|, so the
// moved point is good to ~10^-11 near the origin instead of 3·10^-5, and the checker's colors come out wrong only
// within ~10^-4 of the contact. Far from the origin those terms grow, and the step would add more rounding than it
// removes, so it is taken only while that bound is below 2·radius·|c| (frame.limit). The point moves only for the
// texture: the next ray still starts at the hit point itself.
template <real_number Real>
__device__ point3<Real> onto_sphere(const point3<Real>& p, const vec3<Real>& outward, const sphere<Real>& s,
                                    const sphere_frame<Real>& frame)
{
    const Real p_p = dot(p, p);
    const Real p_c = dot(p, s.center);
    if (!(p_p + Real(2) * fabs(p_c) + fabs(frame.k) < frame.limit))
    {
        return p; // where the step wouldn't help
    }
    const Real f = p_p - Real(2) * p_c + frame.k;
    return p - (f / (Real(2) * s.radius)) * outward;
}
}
