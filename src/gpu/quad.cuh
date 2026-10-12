// The GPU's quadrilateral (see quad.ixx): a parallelogram and the ray-quad test, made robust for float.
#pragma once

#include <cmath>

#include "renderer.h"
#include "vec3.cuh"

namespace gpu
{
// A quad as its hit test needs it: 48 bytes in float, read with three 16-byte loads (96 in double).
//
// α and β of a point p of the quad's plane are (p − q)·alpha_axis and (p − q)·beta_axis. The CPU computes
// α = w·((p − q) × v) and β = w·(u × (p − q)) with w = n / |n|² and n = u × v: two cross products per hit. The
// scalar triple product can be rotated, w·(x × v) = x·(v × w) and w·(u × x) = x·(w × u), so the crosses with the
// fixed vectors are done once, on the host: alpha_axis = v × w and beta_axis = w × u. In exact arithmetic
// alpha_axis·u = 1 and alpha_axis·v = 0, and the other way round for beta_axis: they are the dual basis of (u, v) in
// the quad's plane.
template <real_number Real>
struct alignas(16) quad
{
    point3<Real> q;        // a corner
    vec3<Real> normal;     // the unit normal: the side it points to (u × v) is the front
    vec3<Real> alpha_axis; // v × w
    vec3<Real> beta_axis;  // w × u
};
static_assert(sizeof(quad<float>) == 48 && sizeof(quad<double>) == 96);

// Host code: the quad constructor's formulas (quad.ixx) in double, each result rounded to Real only at the end. So in
// --fp64 a quad gets the CPU's own unit normal, to the last bit. A degenerate quad (u parallel to v) gets NaNs, as the
// CPU's normal is NaN; hit_quad's comparisons are written so that NaNs fail them, and the quad stays invisible, as on
// the CPU.
template <real_number Real>
quad<Real> make_quad(const quad_desc& desc)
{
    const vec3d n = cross(desc.u, desc.v);
    const vec3d normal = scale(1 / std::sqrt(dot(n, n)), n); // the CPU's unit_vector(n): (1 / |n|)·n
    const vec3d w = scale(1 / dot(n, n), n);                 // the CPU's n / dot(n, n)
    return {.q = to_real<Real>(desc.q),
            .normal = to_real<Real>(normal),
            .alpha_axis = to_real<Real>(cross(desc.v, w)),
            .beta_axis = to_real<Real>(cross(w, desc.u))};
}

// The ray against a quad (quad::hit): whether it hits it at a t in [t_min, t_max], closed as on the CPU
// (interval::contains), and so are α's and β's [0, 1]. front_face: the ray hits the side the normal points to, decided
// by the sign of the same dot product the test divides by (hit_record::set_face_normal).
//
// The CPU finds t as (d − normal·origin) / (normal·direction), with d = normal·q: a difference of two numbers the
// size of the coordinates (555 in the Cornell box), which keeps both of their rounding errors. Here the corner is
// first taken relative to the ray's origin, a vector the size of the distance, and the hit point relative to the
// corner, t·direction − (q − origin), is computed without forming a position at all.
//
// __forceinline__: the closest-hit search calls it in its innermost loop, and a call would pass the ray and the results
// through memory.
template <real_number Real>
__device__ __forceinline__ bool hit_quad(const quad<Real>& s, const ray<Real>& r, Real t_min, Real t_max, Real& t,
                                         bool& front_face)
{
    const Real denominator = dot(s.normal, r.direction);
    // Parallel to the plane: no hit, with the CPU's threshold. Written as !(x >= y) so that a NaN (the normal of a
    // degenerate quad) fails too: every comparison with a NaN is false. Real(1e-8): a plain 1e-8 is a double, and the
    // comparison would be computed in double, at 1/64 of the float speed.
    if (!(fabs(denominator) >= Real(1e-8)))
    {
        return false;
    }
    const vec3<Real> origin_to_q = s.q - r.origin;
    t = dot(s.normal, origin_to_q) / denominator;
    if (!(t_min <= t && t <= t_max))
    {
        return false;
    }
    // The hit point relative to the corner: the CPU's r.at(t) − q
    const vec3<Real> p = t * r.direction - origin_to_q;
    const Real alpha = dot(p, s.alpha_axis);
    const Real beta = dot(p, s.beta_axis);
    if (!(Real(0) <= alpha && alpha <= Real(1) && Real(0) <= beta && beta <= Real(1)))
    {
        return false;
    }
    front_face = denominator < Real(0);
    return true;
}

// The texture coordinates of a point of the quad: its α and β (quad::hit), from the same expressions as hit_quad's, for
// the hit point at distance t along the ray
template <real_number Real>
__device__ void quad_uv(const quad<Real>& s, const ray<Real>& r, Real t, Real& u, Real& v)
{
    const vec3<Real> p = t * r.direction - (s.q - r.origin);
    u = dot(p, s.alpha_axis);
    v = dot(p, s.beta_axis);
}
}
