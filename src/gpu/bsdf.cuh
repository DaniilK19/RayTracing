// The quality kernel's materials, as one interface: a BSDF (bidirectional scattering distribution function) f(wo, wi),
// the share of the light arriving from wi that leaves toward wo. The integrator (quality_path.cuh) needs three things
// of it: a direction drawn with some density (sample_bsdf, the path's next ray), f·|cos θ| and that density for a
// direction it chose itself (eval_bsdf, a light sample), and whether the material has a density at all (is_delta). A
// mirror or glass reflects into one exact direction: a delta distribution, no density, so a light sample can never
// find that direction and the light comes only through the BSDF's own ray.
//
// The books' materials keep the books' sampling: a matte surface and a particle of smoke draw exactly the numbers the
// kernel for all three books draws under -p material (diffuse_bounce in camera.cuh), and metal and glass go through
// the books' scatter() (material.cuh). So under -p material the quality kernel traces the books' paths.
#pragma once

#include "material.cuh"
#include "pdf.cuh"
#include "quality_view.cuh"
#include "random.cuh"
#include "vec3.cuh"

namespace gpu
{
enum class lobe : int
{
    lambertian,       // f = albedo/π, cosine density
    isotropic,        // a particle of smoke: albedo/(4π), uniform density
    conductor,        // GGX rough metal: F0 = the metal's albedo, α from the side entry
    gloss,            // GGX coat F0 0.04 over the lambertian albedo
    smooth_conductor, // a conductor below min_ggx_alpha: the perfect mirror with Schlick's F (delta)
    books_delta,      // the books' metal (any fuzz: no density) and glass, through material.cuh's scatter() (delta)
};

struct bsdf
{
    lobe kind;
    color<float> albedo;   // lambertian, isotropic, gloss: the color at the hit; conductors: F0
    float alpha;           // conductor, gloss
    vec3<float> n;         // hit_record::normal: unit, facing the incoming ray
    vec3<float> wo;        // unit, toward where the ray came from: -unit(ray direction)
    material<float> books; // books_delta: what scatter() takes
};

struct bsdf_sample
{
    vec3<float> wi;      // the next direction (unit for every lobe but books_delta, which is scatter()'s)
    color<float> weight; // f·|cos θi|/pdf; delta: the books' attenuation or Schlick's F
    float pdf;           // solid-angle density of wi; 0 for delta lobes
    bool delta;          // no density: no light sample here, the next light counts in full
};

// The lobe from the material, its side entry (nullptr: none) and the color surface_color gave. The GGX lobes come with
// their side entries; until then validate() accepts no side entry but `none`.
__device__ inline bsdf make_bsdf(const material<float>& m, const microfacet* /*side*/, const color<float>& c,
                                 const hit_record<float>& rec, const vec3<float>& ray_direction)
{
    bsdf b{};
    b.n = rec.normal;
    b.wo = -unit_vector(ray_direction); // read by the GGX lobes only: the compiler drops it where nothing reads it
    b.albedo = c;
    switch (m.kind)
    {
    case material_kind::lambertian:
        b.kind = lobe::lambertian;
        break;
    case material_kind::isotropic:
        b.kind = lobe::isotropic;
        break;
    default: // metal and glass (a lamp ends the path before)
        b.kind = lobe::books_delta;
        b.books = m;
        break;
    }
    return b;
}

__device__ inline bool is_delta(const bsdf& b)
{
    return b.kind >= lobe::smooth_conductor;
}

// Draws the next direction. lambertian and isotropic draw exactly the numbers diffuse_bounce draws under -p material,
// in its order; books_delta draws scatter()'s. False: absorbed (a direction below the surface).
__device__ inline bool sample_bsdf(const bsdf& b, const ray<float>& in, const hit_record<float>& rec, pcg32& rng,
                                   bsdf_sample& out)
{
    switch (b.kind)
    {
    case lobe::lambertian:
        // cosine_pdf: f·cos θ / p = (albedo/π)·cos θ / (cos θ/π) = albedo. A direction that rounding put just below
        // the surface is dropped, as the books' kernels drop it.
        out.wi = around(b.n, random_cosine_direction<float>(rng));
        if (!(dot(b.n, out.wi) > 0.0f))
        {
            return false;
        }
        out.weight = b.albedo;
        out.pdf = cosine_density(b.n, out.wi);
        out.delta = false;
        return true;
    case lobe::isotropic:
        out.wi = random_unit_vector<float>(rng);
        out.weight = b.albedo;
        out.pdf = uniform_sphere_density<float>;
        out.delta = false;
        return true;
    case lobe::books_delta:
    {
        // scatter() multiplies a throughput by the metal's albedo (glass keeps it): here a weight that starts at 1
        color<float> weight{1.0f, 1.0f, 1.0f};
        if (!scatter(b.books, in.direction, rec, rng, out.wi, weight))
        {
            return false;
        }
        out.weight = weight;
        out.pdf = 0.0f;
        out.delta = true;
        return true;
    }
    default: // the GGX lobes: their sampling comes with them
        return false;
    }
}

// f·|cos θi| (particles: albedo/(4π)) for a light sample's unit direction wi, and the density sample_bsdf draws it
// with; 0 and 0 for delta lobes and below the horizon
__device__ inline color<float> eval_bsdf(const bsdf& b, const vec3<float>& wi, float& pdf)
{
    switch (b.kind)
    {
    case lobe::lambertian:
    {
        pdf = cosine_density(b.n, wi);
        return pdf * b.albedo; // (albedo/π)·cos θ = albedo · (cos θ/π), 0 below the surface
    }
    case lobe::isotropic:
        pdf = uniform_sphere_density<float>;
        return uniform_sphere_density<float> * b.albedo;
    default:
        pdf = 0.0f;
        return {};
    }
}
}
