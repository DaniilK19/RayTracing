// The GPU's materials (see material.ixx) as plain data, and the first book's three as a switch. (A lamp, diffuse_light,
// reflects nothing: trace_ray in camera.cuh ends the path there.)
//
// On the CPU a material is an object of a class with virtual functions, and a hit calls its scatter(). On the GPU the
// usual way is data: a material is its kind and its numbers, and one function switches on the kind. The CPU's objects
// couldn't be copied to the GPU anyway (their virtual function tables point into the CPU's code), and a switch lets
// the compiler see all the code and inline it. The threads of a warp (32 threads that execute every instruction
// together) whose rays hit different kinds take turns: the warp runs each case with only its threads active
// (divergence). Each case is short, so this costs little.
#pragma once

#include "random.cuh"
#include "renderer.h"
#include "vec3.cuh"

namespace gpu
{
template <real_number Real>
struct material
{
    material_kind kind;
    color<Real> albedo;    // metal; lambertian, diffuse_light, isotropic: its color, or the light it emits, when its
                           // texture is one solid color (see material_texture)
    Real fuzz;             // metal
    Real refraction_index; // dielectric
};

// What the kernel for all three books needs to know about a material besides the struct above, in an array of its own
// (scene_view::material_textures), so that the first book's kernel, which never reads it, loads exactly what it loaded
// before
struct material_texture
{
    int texture_index;      // lambertian, diffuse_light, isotropic: an index into the textures, or -1 when its
                            // texture is one solid color, which material::albedo then holds (nothing to look up)
    bool needs_uv;          // its texture reads the texture coordinates (an image, also inside a checker): only then
                            // does the hit compute them
    bool needs_exact_point; // its texture is a checker, whose colors jump across the faces of its cubes: on a sphere
                            // it reads the hit point moved exactly onto the surface (onto_sphere in sphere.cuh), on
                            // a quad the point computed as the CPU computes it (see surface_at in camera.cuh)
};

// Where a ray hit, as the materials need it (the CPU's hit_record without the texture coordinates)
template <real_number Real>
struct hit_record
{
    point3<Real> p;
    vec3<Real> normal; // unit length, facing against the ray
    bool front_face;   // the ray hit the outside of a sphere, or the front of a quad (the side u × v points to)
};

// onb(n).transform(v) of onb.ixx: v is given in an orthonormal basis whose z axis is the unit vector n, and the result
// is the same vector in world coordinates. The CPU builds the basis with cross products; this is the branchless
// construction of Duff et al., "Building an Orthonormal Basis, Revisited" (JCGT, 2017): no square root, one division.
// Any basis around n gives the same distribution here, as the cosine density is symmetric around n.
template <real_number Real>
__device__ vec3<Real> around(const vec3<Real>& n, const vec3<Real>& v)
{
    const Real sign = copysign(Real(1), n.z);
    const Real a = Real(-1) / (sign + n.z);
    const Real b = n.x * n.y * a;
    const vec3<Real> x_axis{Real(1) + sign * n.x * n.x * a, sign * b, -sign * n.x};
    const vec3<Real> y_axis{b, sign + n.y * n.y * a, -n.y};
    return v.x * x_axis + v.y * y_axis + v.z * n;
}

// Schlick's approximation of the fraction of light glass reflects (see dielectric in material.ixx). The fifth power
// is written as multiplications: pow() would go through a logarithm and an exponential.
template <real_number Real>
__device__ Real reflectance(Real cosine, Real refraction_index)
{
    Real r0 = (Real(1) - refraction_index) / (Real(1) + refraction_index);
    r0 = r0 * r0;
    const Real x = Real(1) - cosine;
    const Real x2 = x * x;
    return r0 + (Real(1) - r0) * (x2 * x2 * x);
}

// The material's scatter(): what happens to a ray with the given direction at the hit. Returns false if the light is
// absorbed: the sample is then 0. Otherwise sets the direction of the next ray and multiplies the path's throughput by
// the attenuation. As on the CPU, a matte bounce is sampled with the material's own density (the camera's material
// strategy, the only one the first book's kernel renders: see first_book_scene in renderer.cu), so its weight
// pScatter / p is exactly 1. (The first book's kernel calls it for every material; the kernel for all three books for
// metal and glass, see trace_ray in camera.cuh.)
template <real_number Real>
__device__ bool scatter(const material<Real>& mat, const vec3<Real>& direction, const hit_record<Real>& rec,
                        pcg32& rng, vec3<Real>& scattered, color<Real>& throughput)
{
    switch (mat.kind)
    {
    case material_kind::lambertian:
        scattered = around(rec.normal, random_cosine_direction<Real>(rng));
        // The CPU's camera doesn't trace a direction whose density isn't above 0, which here means one not above the
        // surface: rounding can put a direction drawn very close to the surface just below it. The sample is then 0.
        if (!(dot(rec.normal, scattered) > Real(0)))
        {
            return false;
        }
        throughput *= mat.albedo;
        return true;

    case material_kind::metal:
        scattered = unit_vector(reflect(direction, rec.normal)) + mat.fuzz * random_unit_vector<Real>(rng);
        if (dot(scattered, rec.normal) <= Real(0))
        {
            return false; // blurred below the surface: absorbed
        }
        throughput *= mat.albedo;
        return true;

    case material_kind::dielectric:
    {
        const Real ri = rec.front_face ? Real(1) / mat.refraction_index : mat.refraction_index;
        const vec3<Real> unit_direction = unit_vector(direction);
        const Real cos_theta = fmin(dot(-unit_direction, rec.normal), Real(1));
        const Real sin_theta = sqrt(Real(1) - cos_theta * cos_theta);
        const bool cannot_refract = ri * sin_theta > Real(1);
        scattered = cannot_refract || reflectance(cos_theta, ri) > random_real<Real>(rng)
                        ? reflect(unit_direction, rec.normal)
                        : refract(unit_direction, rec.normal, ri);
        return true; // glass absorbs nothing: the throughput stays
    }
    }
    // Not reached: the first book's kernel gets only these three kinds (see first_book_scene in renderer.cu), and
    // trace_ray calls this function only for metal and glass (it handles matte surfaces, lamps and smoke itself)
    return false;
}
}
