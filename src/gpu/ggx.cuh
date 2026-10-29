// The GGX microfacet model for the quality kernel's rough metals: pure functions of the unit normal n, the unit
// direction toward the viewer v, the unit direction toward the light l, the width α and the color F0, callable without
// a scene (bsdf.cuh puts them behind its lobes; GpuTests calls them directly).
//
// A rough surface is a field of tiny mirrors, microfacets, whose normals spread around n. Light from l leaves toward v
// off the facets whose normal is the half vector h = (v + l)/|v + l|, so (Walter et al. 2007, eq. 20)
//   f(v, l) = F(v·h)·D(h)·G2(v, l) / (4 (n·v)(n·l)):
// D, the density of the facets' normals (GGX, also called Trowbridge-Reitz: Walter eq. 33); G2, the share of the
// facets toward h that both v and l see, the rest being hidden behind other facets (Smith's masking and shadowing,
// height-correlated: Heitz 2014, eq. 99); F, how much a facet reflects (Schlick 1994, with the metal's color at
// normal incidence, F0, per channel). Notation: c_v = n·v, c_l = n·l, c_h = n·h.
//
// The width is α = r², from the roughness r in [0, 1] (Burley 2012, as OpenPBR and Filament: perceptually closer to
// linear than α itself; pbrt-v4's "roughness" is another scale). The formulas are written so that float holds from
// α = 1e-3 up (min_ggx_alpha); below it the lobe is the perfect mirror (bsdf.cuh's smooth conductor).
//
// The lobe draws its directions by the facets the viewer sees, the visible normals (Heitz 2018), with the spherical
// caps of Dupuy & Benyoub (HPG 2023): two random numbers, no tangent frame of its own. Every sample's weight
// f·cos θl / p is then F·G2/G1(v), at most F: no sample carries more light than a facet reflects, while drawing h by D
// alone gives weights that reach the hundreds at grazing angles (Walter eq. 41). One bounce off the facets is all the
// model describes; ggx_compensation gives back the light that leaves after several (Turquin 2019). The code is
// re-derived from the papers' equations; no listing is copied.
#pragma once

#include "ggx_energy.cuh"
#include "material.cuh" // around
#include "pdf.cuh"      // inverse_pi
#include "random.cuh"
#include "vec3.cuh"

namespace gpu
{
// Below this width the lobe is the perfect mirror, a delta lobe (bsdf.cuh): r = 0 gives exactly that lobe. The peak of
// D needs sin²θh resolved far below α², which the cross product gives (see ggx_d) down to α = 1e-3, where D's peak is
// 1/(πα²) = 3.2·10⁵ and the largest density a direction gets, 1/(2πα³) = 1.6·10⁸ (a grazing view, the mirror
// direction), is still far inside float's range (and mis.cuh weighs densities by their ratio, never squared). pbrt-v4
// switches at the same width; GpuTests checks float's weak furnace there (ggx/min-alpha-ok).
inline constexpr float min_ggx_alpha = 1e-3f;

// a × b (vec3.cuh has the host's, in double)
__device__ inline vec3<float> cross(const vec3<float>& a, const vec3<float>& b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// A(c) = √(c²(1 − α²) + α²) = c·√(1 + α² tan²θ) for c = cos θ. Smith's Λ for GGX (Heitz 2014, §5) is then
// Λ(c) = (A(c)/c − 1)/2 without a tangent, which grows without bound toward the horizon.
__device__ inline float smith_a(float c, float alpha2)
{
    return sqrt(fma(c * c, 1.0f - alpha2, alpha2));
}

// The density of the facets' normals per unit of projected area, ∫ D(h)·c_h dω_h = 1 (Walter eq. 33):
// D(h) = α² / (π (α² c_h² + sin²θh)²) for c_h > 0, else 0. sin²θh comes from |n × h|² (Lagrange's identity), not from
// 1 − c_h²: near the peak c_h is a float just below 1, whose 1 − c_h² is quantized in steps of 1.2·10⁻⁷, 12% of α² at
// α = 1e-3 (Filament makes the same point for half floats).
__device__ inline float ggx_d(const vec3<float>& n, const vec3<float>& h, float alpha2)
{
    const float c = dot(n, h);
    if (!(c > 0.0f))
    {
        return 0.0f; // no facet faces away from the surface
    }
    const vec3<float> s = cross(n, h);
    const float d = fma(alpha2 * c, c, dot(s, s));
    return alpha2 * inverse_pi<float> / (d * d);
}

// Smith's masking (Heitz 2014, §5): the share of the facets a direction of cosine c with n sees,
// G1 = 1/(1 + Λ(c)) = 2c/(c + A(c)). For the tests: the lobe uses the combined forms below.
__device__ inline float ggx_g1(float c, float alpha2)
{
    return 2.0f * c / (c + smith_a(c, alpha2));
}

// Masking and shadowing together, height-correlated (Heitz 2014, eq. 99): G2 = 1/(1 + Λ(v) + Λ(l)) =
// 2 c_v c_l / (c_l A(c_v) + c_v A(c_l)). A facet low in the surface is likely hidden from both directions at once;
// the product G1(v)·G1(l) would count that twice and darken rough metals, at the same cost. For the tests.
__device__ inline float ggx_g2(float cv, float cl, float alpha2)
{
    return 2.0f * cv * cl / (cl * smith_a(cv, alpha2) + cv * smith_a(cl, alpha2));
}

// Schlick's approximation of the share a facet reflects at the cosine θ of its incidence: F = F0 + (1 − F0)(1 − cos θ)⁵
// per channel. A metal reflects its own color head-on and white at grazing angles. The fifth power as multiplications
// (as reflectance() in material.cuh); a cosine that rounding put above 1 counts as 1. F0 = 1 gives exactly 1.
__device__ inline color<float> schlick(const color<float>& f0, float cosine)
{
    const float x = fmax(0.0f, 1.0f - cosine);
    const float x2 = x * x;
    const float x5 = x2 * x2 * x;
    return {fma(1.0f - f0.x, x5, f0.x), fma(1.0f - f0.y, x5, f0.y), fma(1.0f - f0.z, x5, f0.z)};
}

// f(v, l)·c_l for the light from l, and in pdf the density p(l) ggx_sample draws l with (multiple importance sampling
// weighs the two). Both 0 when v or l lies at or below the horizon (a light there sends nothing: no shadow ray is
// needed), or for l = −v (no half vector).
//   f·c_l = F·D·G2/(4 c_v) = F·D·c_l / (2 (c_l A_v + c_v A_l)), with A_v = A(c_v) and A_l = A(c_l): the "V form"
//   (Filament's V_SmithGGXCorrelated), symmetric in v and l, which never divides by c_v or c_l, so a grazing direction
//   makes no 0·∞.
//   p = D_v(h)/(4 v·h) = G1(v)·D(h)/(4 c_v) = D(h) / (2 (c_v + A_v)): the visible normals' density
//   D_v(h) = G1(v)·max(0, v·h)·D(h)/c_v (Heitz 2018, eq. 3) times the reflection's Jacobian dω_h/dω_l = 1/(4 v·h)
//   (Walter eq. 14; Heitz 2018, eq. 17), which v·h cancels from.
__device__ inline color<float> ggx_eval(const vec3<float>& n, const vec3<float>& v, const vec3<float>& l, float alpha,
                                        const color<float>& f0, float& pdf)
{
    pdf = 0.0f;
    const float cv = dot(n, v);
    const float cl = dot(n, l);
    const vec3<float> sum = v + l;
    const float length2 = sum.length_squared();
    if (!(cv > 0.0f) || !(cl > 0.0f) || !(length2 > 0.0f))
    {
        return {};
    }
    const vec3<float> h = rsqrt(length2) * sum;
    const float alpha2 = alpha * alpha;
    const float d = ggx_d(n, h, alpha2);
    const float av = smith_a(cv, alpha2);
    const float al = smith_a(cl, alpha2);
    pdf = d / (2.0f * (cv + av));
    return (d * cl / (2.0f * (cl * av + cv * al))) * schlick(f0, dot(v, h));
}

// p(l) alone: ggx_eval's density, by ggx_eval itself (its color, which nothing reads here, the compiler drops): the
// same formula, though two inlined copies may round its half vector apart (GpuTests compare them within D's
// sensitivity, not bit for bit)
__device__ inline float ggx_pdf(const vec3<float>& n, const vec3<float>& v, const vec3<float>& l, float alpha)
{
    float pdf;
    (void)ggx_eval(n, v, l, alpha, color<float>{1.0f, 1.0f, 1.0f}, pdf);
    return pdf;
}

// Draws l by the visible normals with Dupuy & Benyoub's spherical caps (HPG 2023; Listings 1 and 3), in world
// coordinates. GGX's normals are those of a field of ellipsoids 1/α wide and 1 high (Heitz 2018, §3); squeezed by α
// along the tangents, they become hemispheres of radius 1, seen from v_s, v squeezed the same way. A hemispherical
// mirror lit by parallel rays from v_s reflects them uniformly into the spherical cap {c : n·c > −n·v_s} (their §3.1),
// so a direction c drawn uniformly in that cap gives the normal the mirror reflected it with,
// h_s = (c + v_s)/|c + v_s|, with exactly the density of the hemisphere's visible normals. Taken back to the ellipsoid
// as a normal (by the inverse transpose: its tangential part times α, its part along n kept), h has GGX's D_v(h). The
// lobe is isotropic, so "the tangential part times α" is the same in every frame: around() places the cap, no frame
// of the lobe's own is needed. The pdf below is ggx_eval's, D(h)/(2(c_v + A_v)), with the drawn h.
// Two random numbers in every call, u1 for the azimuth, then u2 for the height in the cap, whatever the outcome.
// false: absorbed. l = (0, 0, 0) when v lies at or below the horizon (rounding at a grazing hit: eval has no light
// there either) or the half vector is degenerate. l is the mirror direction when that lies at or below the horizon,
// or v behind the drawn facet: the light single scattering loses, a ray that the facets would pass on to one another.
// On true: l unit up to rounding; weight = f·c_l / p = F·G2/G1(v) = F·c_l (c_v + A_v)/(c_l A_v + c_v A_l), from which
// D cancels, at most F; pdf = D(h)/(2(c_v + A_v)).
__device__ inline bool ggx_sample(const vec3<float>& n, const vec3<float>& v, float alpha, const color<float>& f0,
                                  pcg32& rng, vec3<float>& l, color<float>& weight, float& pdf)
{
    const float u1 = random_real<float>(rng); // one number per statement: a fixed order
    const float u2 = random_real<float>(rng);
    const float cv = dot(n, v);
    if (!(cv > 0.0f))
    {
        l = {};
        return false;
    }
    // 1. The view, squeezed into the hemispheres' space: its tangential part times α
    const vec3<float> vs = unit_vector(alpha * (v - cv * n) + cv * n);
    const float zs = dot(n, vs);
    // 2. A direction uniform in the cap of heights z ∈ (−z_s, 1] around n: the height uniform (Archimedes: slices of a
    // sphere of equal height have equal areas), the azimuth 2π·u1. (1 − u2) puts z = 1 in the range and −z_s out.
    const float z = fma(1.0f - u2, 1.0f + zs, -zs);
    const float sin_theta = sqrt(fmax(0.0f, 1.0f - z * z));
    float sin_phi;
    float cos_phi;
    sincospi(2.0f * u1, &sin_phi, &cos_phi);
    const vec3<float> c = around(n, vec3<float>{sin_theta * cos_phi, sin_theta * sin_phi, z});
    // 3. The hemisphere's normal that reflects v_s into c (its length doesn't matter), back on the ellipsoid. Its part
    // along n is above 0 in exact arithmetic (the cap lies above −z_s); a rounding below it counts as 0.
    const vec3<float> hs = c + vs;
    const float hn = dot(n, hs);
    const vec3<float> m = alpha * (hs - hn * n) + fmax(0.0f, hn) * n;
    const float m2 = m.length_squared();
    if (!(m2 > 0.0f))
    {
        l = {};
        return false; // c = −v_s, at the cap's open rim
    }
    const vec3<float> h = rsqrt(m2) * m;
    // 4. The mirror of v about h: reflect(−v, h)
    const float vh = dot(v, h);
    l = (2.0f * vh) * h - v;
    const float cl = dot(n, l);
    if (!(cl > 0.0f) || !(vh > 0.0f))
    {
        return false;
    }
    const float alpha2 = alpha * alpha;
    const float av = smith_a(cv, alpha2);
    const float al = smith_a(cl, alpha2);
    weight = (cl * (cv + av) / (cl * av + cv * al)) * schlick(f0, vh);
    pdf = ggx_d(n, h, alpha2) / (2.0f * (cv + av));
    return true;
}

// The light a white conductor's single scattering keeps (F = 1) for the viewer at cos θv = μ and the roughness r:
// E_ss(μ, r) = ∫ f·c_l dω_l, from ggx_energy.cuh's table (GpuTests' quadrature, nodes at √μ = i/31 and r = j/31),
// interpolated bilinearly and clamped to the table. The columns are even in √μ, not in μ: a smooth metal keeps nearly
// all of its light until the view grazes the surface, then E_ss falls within a few hundredths of μ, and nodes even in
// μ (a first one at 0.032) would put 5% too much light into its silhouettes. Where both α and μ are small E_ss
// depends on μ/α alone, which a bilinear cell across the table's diagonal follows loosely: within 1.2° of the
// horizon a smooth metal's compensated energy can be a few percent off 1.
__device__ inline float ggx_energy(float cos_v, float roughness)
{
    constexpr int last = ggx_energy_size - 1;
    const float x = sqrtf(fmin(fmax(cos_v, 0.0f), 1.0f)) * static_cast<float>(last);
    const float y = fmin(fmax(roughness, 0.0f), 1.0f) * static_cast<float>(last);
    const int i = static_cast<int>(x) < last - 1 ? static_cast<int>(x) : last - 1; // ⌊x⌋ for x ≥ 0, a cell's corner
    const int j = static_cast<int>(y) < last - 1 ? static_cast<int>(y) : last - 1;
    const float fx = x - static_cast<float>(i);
    const float fy = y - static_cast<float>(j);
    const float* row = ggx_energy_table + j * ggx_energy_size + i;
    const float below = (1.0f - fx) * row[0] + fx * row[1];
    const float above = (1.0f - fx) * row[ggx_energy_size] + fx * row[ggx_energy_size + 1];
    return (1.0f - fy) * below + fy * above;
}

// Turquin's multiple-scattering compensation (Turquin 2019, eq. 16), the factor f_ms/f_ss = 1 + F0·(1 − E)/E per
// channel, E = E_ss(μ, r). Single scattering loses the light the facets pass on to one another, 1 − E of it with
// F = 1: 59% on average over the angles at r = 1, 69% head-on. Turquin puts it back as a scaled copy of the
// single-scattering lobe, closer to the reference than a diffuse lobe (Heitz et al. 2016), drawn with the same
// sampling and the same density, so only the sample's weight and eval_bsdf change. A white metal then keeps
// E·(1 + (1 − E)/E) = 1 of its light at every angle; in a colored one the light put back is tinted by F0. (The factor
// depends on the view alone, so f is no longer reciprocal, which a path traced from the camera never notices:
// sampling and eval see the same v.) E is clamped at 1, so the factor is never below 1; F0 = 0 gives exactly 1.
__device__ inline color<float> ggx_compensation(const color<float>& f0, float cos_v, float roughness)
{
    const float e = fmin(ggx_energy(cos_v, roughness), 1.0f);
    const float lost = (1.0f - e) / e; // the light lost per unit kept (E is at least 0.3 in the table)
    return {fma(f0.x, lost, 1.0f), fma(f0.y, lost, 1.0f), fma(f0.z, lost, 1.0f)};
}
}
