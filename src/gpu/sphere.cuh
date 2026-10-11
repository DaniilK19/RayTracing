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

// The distance t (in lengths of the ray's direction) at which the ray hits the sphere, if a hit lies in (t_min, t_max),
// and which root that is: `entering` is true for the near root, where the ray enters the sphere, and false for the far
// one, where it leaves (see trace_ray for why that matters). inverse_a is 1 / |direction|², the same for every sphere,
// so the caller computes it once per ray. far_only takes only the far root: for a ray that starts on this very sphere
// and goes inside it (see closest_hit).
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
}
