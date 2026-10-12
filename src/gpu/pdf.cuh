// The densities of directions that the kernel for all three books weighs its diffuse bounces with (see pdf.ixx).
#pragma once

#include <numbers>

#include "vec3.cuh"

namespace gpu
{
// The density of random_unit_vector: the same in every direction, 1 / (the area of the unit sphere) = 1 / (4π). The
// CPU's constant uniform_sphere_density, rounded to Real once: a constexpr variable template, which device code reads
// like infinity<Real>.
template <real_number Real>
constexpr Real uniform_sphere_density = static_cast<Real>(1 / (4 * std::numbers::pi));

template <real_number Real>
constexpr Real inverse_pi = static_cast<Real>(1 / std::numbers::pi);

// cosine_density of pdf.ixx: cos θ / π above the surface, θ being the angle between the direction and the unit normal,
// and 0 below it, or for a NaN cosine (a comparison with a NaN is false). The direction may have any length.
template <real_number Real>
__device__ Real cosine_density(const vec3<Real>& unit_normal, const vec3<Real>& direction)
{
    const Real cosine = dot(unit_normal, direction) * rsqrt(direction.length_squared());
    return cosine > Real(0) ? cosine * inverse_pi<Real> : Real(0);
}
}
