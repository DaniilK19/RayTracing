// The GPU's vec3 (see vec3.ixx): a 3D vector for points, directions and colors, and a ray (see ray.ixx).
//
// The GPU renders in float or in double (see precision in renderer.h) from one source: the device code is made of
// templates on Real, which is one of the two, and nvcc compiles each of them twice. The float version must compute
// nothing in double but the pixel sums (see render_pixel in renderer.cu). A GeForce card does double arithmetic 64
// times slower than float, so one forgotten double in the inner loop would cost more than all the float math around it.
// That is why literals are written as Real(0.5), not 0.5 (a double), and why the math functions below have a GPU
// overload for each type.
#pragma once

#include <concepts>
#include <limits>
#include <numbers>

#include "renderer.h"

namespace gpu
{
// float or double. A concept (C++20) names the requirement, and a template constrained by it rejects other types at
// compile time (long double, for one, which CUDA doesn't have on the GPU).
template <typename T>
concept real_number = std::same_as<T, float> || std::same_as<T, double>;

// A variable template (C++14): infinity<float> and infinity<double> are two constants. constexpr makes them
// compile-time values, which GPU code may read although they are defined for the CPU.
template <real_number Real>
constexpr Real infinity = std::numeric_limits<Real>::infinity();

// π rounded to Real, from C++20's <numbers>: also a compile-time constant that GPU code may read
template <real_number Real>
constexpr Real pi = std::numbers::pi_v<Real>;

// The math functions the device code uses, in both precisions. CUDA overloads most of them for float, as C++'s <cmath>
// does, so sqrt of a float already computes in float. But not every overload it finds works on the GPU: with MSVC,
// copysign(float, float) is the standard library's CPU-only function, and calling it from GPU code only draws a
// warning, while the GPU code compiled from it is wrong. These overloads name exactly the functions the device code
// needs, each with its GPU version for float and for double, and hide the global ones inside namespace gpu: template
// code that writes sqrt(x) gets the GPU function of x's own type, and an int argument is a compile error (float or
// double? ambiguous) instead of a quiet conversion. The other side of this: host code in namespace gpu (the BVH
// builder, the conversions in renderer.cu) must name the standard library's functions in full, std::sqrt or
// std::floor, as an unqualified call would find these GPU-only functions first.
// __device__ marks a function that runs on the GPU and is called from GPU code; nvcc compiles it for the GPU only.
__device__ inline float sqrt(float x) { return ::sqrtf(x); }
__device__ inline double sqrt(double x) { return ::sqrt(x); }
// 1 / √x: the GPU has an instruction for it, so this is cheaper than a square root and a division
__device__ inline float rsqrt(float x) { return ::rsqrtf(x); }
__device__ inline double rsqrt(double x) { return ::rsqrt(x); }
__device__ inline float fabs(float x) { return ::fabsf(x); }
__device__ inline double fabs(double x) { return ::fabs(x); }
__device__ inline float fmin(float x, float y) { return ::fminf(x, y); }
__device__ inline double fmin(double x, double y) { return ::fmin(x, y); }
__device__ inline float fmax(float x, float y) { return ::fmaxf(x, y); }
__device__ inline double fmax(double x, double y) { return ::fmax(x, y); }
__device__ inline float copysign(float x, float y) { return ::copysignf(x, y); }
__device__ inline double copysign(double x, double y) { return ::copysign(x, y); }
// x·y + z with a single rounding (a fused multiply-add), one GPU instruction
__device__ inline float fma(float x, float y, float z) { return ::fmaf(x, y, z); }
__device__ inline double fma(double x, double y, double z) { return ::fma(x, y, z); }
// x·y and x + y, each rounded on its own: CUDA's intrinsics that nvcc never fuses into an fma (see ray::at_unfused)
__device__ inline float mul_rn(float x, float y) { return __fmul_rn(x, y); }
__device__ inline double mul_rn(double x, double y) { return __dmul_rn(x, y); }
__device__ inline float add_rn(float x, float y) { return __fadd_rn(x, y); }
__device__ inline double add_rn(double x, double y) { return __dadd_rn(x, y); }
// The largest whole number not above x (the checker texture's cubes)
__device__ inline float floor(float x) { return ::floorf(x); }
__device__ inline double floor(double x) { return ::floor(x); }
// The angle of the point (x, y) seen from the origin, from -π to π, and the angle whose cosine is x, from 0 to π (the
// texture coordinates of a sphere)
__device__ inline float atan2(float y, float x) { return ::atan2f(y, x); }
__device__ inline double atan2(double y, double x) { return ::atan2(y, x); }
__device__ inline float acos(float x) { return ::acosf(x); }
__device__ inline double acos(double x) { return ::acos(x); }
// The sine (the marble's stripes)
__device__ inline float sin(float x) { return ::sinf(x); }
__device__ inline double sin(double x) { return ::sin(x); }
// The natural logarithm (the distance to a particle of smoke); log(0) is -∞, as in C
__device__ inline float log(float x) { return ::logf(x); }
__device__ inline double log(double x) { return ::log(x); }
// sin(π·x) and cos(π·x) in one call. More accurate than sin(π * x), whose argument π * x is already rounded.
__device__ inline void sincospi(float x, float* sine, float* cosine) { ::sincospif(x, sine, cosine); }
__device__ inline void sincospi(double x, double* sine, double* cosine) { ::sincospi(x, sine, cosine); }

// Functions marked __host__ __device__ are compiled for both the CPU and the GPU. These simple ones can serve both,
// while the math functions above are for the GPU only (__device__): they call CUDA's GPU functions.
template <real_number Real>
struct vec3
{
    Real x = 0;
    Real y = 0;
    Real z = 0;

    __host__ __device__ vec3 operator-() const { return {-x, -y, -z}; }

    // Component by component, as colors mix: the throughput of a path is multiplied by each surface's albedo
    __host__ __device__ vec3& operator*=(const vec3& v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    __host__ __device__ Real length_squared() const { return x * x + y * y + z * z; }
};

template <real_number Real>
using point3 = vec3<Real>;
template <real_number Real>
using color = vec3<Real>;

template <real_number Real>
__host__ __device__ vec3<Real> operator+(const vec3<Real>& u, const vec3<Real>& v)
{
    return {u.x + v.x, u.y + v.y, u.z + v.z};
}

template <real_number Real>
__host__ __device__ vec3<Real> operator-(const vec3<Real>& u, const vec3<Real>& v)
{
    return {u.x - v.x, u.y - v.y, u.z - v.z};
}

template <real_number Real>
__host__ __device__ vec3<Real> operator*(const vec3<Real>& u, const vec3<Real>& v)
{
    return {u.x * v.x, u.y * v.y, u.z * v.z};
}

// Real is deduced from both arguments, so 2 * v doesn't compile (2 is an int): the number must be a Real(2), which
// keeps the float version free of doubles by construction
template <real_number Real>
__host__ __device__ vec3<Real> operator*(Real t, const vec3<Real>& v)
{
    return {t * v.x, t * v.y, t * v.z};
}

template <real_number Real>
__host__ __device__ Real dot(const vec3<Real>& u, const vec3<Real>& v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

// Host code: the arithmetic that the host does in double on the interface's vectors (a quad's normal and axes), with
// the CPU's formulas (vec3.ixx)
inline double dot(const vec3d& u, const vec3d& v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

inline vec3d cross(const vec3d& u, const vec3d& v)
{
    return {u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
}

inline vec3d scale(double s, const vec3d& v)
{
    return {s * v.x, s * v.y, s * v.z};
}

// Host code: a vector of the interface (renderer.h), which is in double, rounded to the precision the GPU computes in.
// It lives in this header, not in renderer.cu, because templates in the other headers (the BVH builder) call it too.
template <real_number Real>
vec3<Real> to_real(const vec3d& v)
{
    return {static_cast<Real>(v.x), static_cast<Real>(v.y), static_cast<Real>(v.z)};
}

template <real_number Real>
__device__ vec3<Real> unit_vector(const vec3<Real>& v)
{
    return rsqrt(v.length_squared()) * v;
}

// Mirror reflection of v off a surface with unit normal n (see vec3.ixx)
template <real_number Real>
__device__ vec3<Real> reflect(const vec3<Real>& v, const vec3<Real>& n)
{
    return v - (Real(2) * dot(v, n)) * n;
}

// Refraction of the unit vector uv at a surface with unit normal n (Snell's law, see vec3.ixx)
template <real_number Real>
__device__ vec3<Real> refract(const vec3<Real>& uv, const vec3<Real>& n, Real etai_over_etat)
{
    const Real cos_theta = fmin(dot(-uv, n), Real(1));
    const vec3<Real> r_out_perp = etai_over_etat * (uv + cos_theta * n);
    const vec3<Real> r_out_parallel = -sqrt(fabs(Real(1) - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

// A ray (see ray.ixx). time is when it was emitted, from 0 to 1 while the camera's shutter is open (see start_sample
// in camera.cuh): only moving spheres read it, and the ray's bounces keep it. Written as {origin, direction}, a ray
// gets the time 0.
template <real_number Real>
struct ray
{
    point3<Real> origin;
    vec3<Real> direction; // not normalized: t is measured in lengths of the direction, as on the CPU
    Real time = 0;

    __device__ point3<Real> at(Real t) const { return origin + t * direction; }

    // The same point computed as the CPU computes it: the product t·direction rounded, then the sum. In at(), nvcc may
    // fuse the two into one fma, rounded once: more accurate, but not the same number, and where a point's last bit
    // decides a color (see surface_at in camera.cuh), the GPU takes the CPU's.
    __device__ point3<Real> at_unfused(Real t) const
    {
        return {add_rn(origin.x, mul_rn(t, direction.x)), add_rn(origin.y, mul_rn(t, direction.y)),
                add_rn(origin.z, mul_rn(t, direction.z))};
    }
};
}
