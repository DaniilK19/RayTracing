// What a ray of the quality kernel that hits nothing sees: the camera's sky, or the scene's environment map, and the
// environment as a light the light sampler can draw from (light_sampler.cuh).
//
// The map is sampled by its brightness: a cell with the chance Y·Ω/S from the alias table (environment_table.cu), then
// a direction uniform over the cell's solid angle, so the density is Y/S per steradian, constant over the cell, and
// stored in the cell's texel. The luminance of a sample's radiance over its density is then S (times the map's scale)
// for every sample: a sun of a few texels, which the material's own samples find once in tens of thousands of
// bounces, is drawn as often as its light deserves. (pbrt-v3 draws (u, v) from a 2D table and divides by 2π² sin θ,
// which isn't constant over a cell and needs a guard at the poles; pbrt-v4 resamples the map to an equal-area one,
// which this avoids.) The cell comes from an alias table rather than a search of a cumulative table: two dependent
// loads per sample instead of about 20, and a dim cell's chance isn't lost in a float sum's last bit (a 1k studio
// map has 197,368 cells below 2^-24).
//
// Both sides of the weights read the density through env_cell, the one map from a direction to a cell: a light sample
// takes its radiance and density from the cell its direction lands in, not from the cell the alias table picked, and
// a ray that escapes along the same direction gets the very same two numbers (multiple importance sampling needs the
// density of the function the escapes see). The two cells differ only for a direction that rounding puts on the other
// side of a cell's border, which GpuTests counts (env-sampling: at most 10⁻⁴ of the samples, 10⁻³ in the polar rows,
// where the float row cosines and the direction's length round the most). At a yaw other than 0, two inlined copies of
// the map's rotation may also round apart, in about 10⁻⁶ of the directions; at yaw 0 the rotation is exact.
#pragma once

#include "camera.cuh"
#include "environment.cuh"
#include "quality_view.cuh"
#include "random.cuh"

namespace gpu
{
// A map with light in it (S > 0) and its alias table: a light of the set (a black map gets no table)
__device__ inline bool environment_sampled(const environment_view& e)
{
    return e.texels != nullptr && e.bins != nullptr;
}

// filtered: camera rays and rays after a delta or pdf-less vertex, which see the map bilinearly filtered; the rays MIS
// combines see its nearest texel, the one its density is constant over. The caller says which (quality_path::specular,
// a property of the material, never inferred from a density: under -p material no vertex stores one, and its escapes
// must see the map the light samples of -p mis see, or the two strategies would render two environments). Without a
// map, the camera's sky: the books' arithmetic exactly. In pdf, the light sampler's density of dir per steradian (no
// pmf) from the very texel the radiance comes from, which the caller weighs the escape with: one lookup, so that a
// direction on a cell's border can't take its radiance from one cell and its density from the next. 0 for the sky, a
// filtered lookup (no light sample competes there) and a map that isn't sampled.
__device__ inline color<float> background(const quality_view& v, const camera<float>& cam, const vec3<float>& dir,
                                          bool filtered, float& pdf)
{
    pdf = 0.0f;
    const environment_view& e = v.environment;
    if (e.texels == nullptr)
    {
        return cam.sky(dir);
    }
    if (filtered)
    {
        return env_radiance_filtered(e, dir);
    }
    float texel_pdf;
    const color<float> le = env_radiance(e, dir, texel_pdf);
    pdf = environment_sampled(e) ? texel_pdf : 0.0f;
    return le;
}

// The same without the density
__device__ inline color<float> background(const quality_view& v, const camera<float>& cam, const vec3<float>& dir,
                                          bool filtered)
{
    float pdf;
    return background(v, cam, dir, filtered, pdf);
}

// A direction drawn from the map by its brightness, unit, its radiance (the nearest texel's, times the scale) and its
// density per steradian (no pmf: light_sampler.cuh multiplies by the chance of picking the map). Exactly 4 numbers:
// (1) the bin, the top log2_bins bits of one 32-bit number, uniform over the 2^k bins (a padded bin is drawn as
// often as any and gives its alias); (2) the bin's coin, random_real < q: its own cell or its alias; (3) cos θ and
// (4) φ' inside the cell. False when the map isn't sampled (no number drawn then), or when the direction's own cell,
// past a border by rounding, is a black one: a density of 0, which no light sample may carry.
__device__ inline bool sample_environment(const environment_view& e, pcg32& rng, vec3<float>& wi, color<float>& le,
                                          float& pdf)
{
    if (!environment_sampled(e))
    {
        return false;
    }
    const int bin = static_cast<int>(rng.next() >> (32 - e.log2_bins));
    const alias_bin b = e.bins[bin];
    const int cell = random_real<float>(rng) < b.q ? bin : b.alias;
    const int j = cell / e.width;
    const int i = cell - j * e.width;
    // Two statements: in one expression the order of the two draws would be unspecified
    const float s_theta = random_real<float>(rng);
    const float s_phi = random_real<float>(rng);
    wi = cell_direction(e, i, j, s_theta, s_phi);
    le = env_radiance(e, wi, pdf);
    return pdf > 0.0f;
}

// The density per steradian with which sample_environment draws dir (of any length), through env_cell: the cell an
// escaping ray along dir sees. 0 when the map isn't sampled.
__device__ inline float environment_pdf(const environment_view& e, const vec3<float>& dir)
{
    return environment_sampled(e) ? e.texels[env_cell(e, dir)].pdf : 0.0f;
}
}
