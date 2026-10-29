// Suite bsdf: the quality kernel's materials (bsdf.cuh) as the books define them: matte, smoke's particles, and the
// books' metal and glass through their own scatter(). The density sample_bsdf draws with must be the one eval_bsdf
// gives (the multiple importance sampling of the later steps weighs the two against each other), the weight must be
// f·cos θ / p, and the draws must be the books' own, number for number, so that the quality kernel's base mode
// (-p material) traces the books' paths. The spec is out/m4/spec/shared.md, section 10.
#include <cmath>
#include <numbers>

#include "bsdf.cuh"
#include "camera.cuh"
#include "gpu_test.cuh"
#include "gpu_test.h"
#include "material.cuh"
#include "random.cuh"

namespace
{
using gpu::vec3;
using gpu_test::direction_grid;

constexpr std::int64_t chi2_samples = 1'000'000;
constexpr std::int64_t draw_samples = 1'000'000;
constexpr std::int64_t small_samples = 10'000;

vec3<float> to_float(const gpu::vec3d& v) { return gpu::to_real<float>(v); }

const gpu::vec3d tilted{1 / std::sqrt(14.0), 2 / std::sqrt(14.0), 3 / std::sqrt(14.0)};
// The cases' colors, for host and device code alike (a host variable can't be read on the GPU)
__host__ __device__ inline vec3<float> matte_albedo() { return {0.73f, 0.5f, 0.2f}; }
__host__ __device__ inline vec3<float> smoke_albedo() { return {0.9f, 0.6f, 0.3f}; }
__host__ __device__ inline vec3<float> metal_albedo() { return {0.8f, 0.6f, 0.4f}; }

__device__ inline bool bits_equal(float a, float b) { return __float_as_uint(a) == __float_as_uint(b); }
__device__ inline bool bits_equal(const vec3<float>& a, const vec3<float>& b)
{
    return bits_equal(a.x, b.x) && bits_equal(a.y, b.y) && bits_equal(a.z, b.z);
}
__device__ inline bool finite(const vec3<float>& v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z); }

// The materials of the cases: 0 lambertian, 1 isotropic, 2-4 metal with fuzz 0, 0.3 and 1, 5 glass
__device__ inline gpu::material<float> case_material(int kind)
{
    gpu::material<float> m{};
    switch (kind)
    {
    case 0:
        m.kind = gpu::material_kind::lambertian;
        m.albedo = matte_albedo();
        break;
    case 1:
        m.kind = gpu::material_kind::isotropic;
        m.albedo = smoke_albedo();
        break;
    case 5:
        m.kind = gpu::material_kind::dielectric;
        m.refraction_index = 1.5f;
        break;
    default:
        m.kind = gpu::material_kind::metal;
        m.albedo = metal_albedo();
        m.fuzz = kind == 2 ? 0.0f : kind == 3 ? 0.3f : 1.0f;
        break;
    }
    return m;
}

// A random hit: a unit normal (or the given one), an incoming direction against it of length 0.5 to 3, the point
// (0.5, 1, -2), and a random side for glass
struct hit_case
{
    gpu::hit_record<float> rec;
    gpu::ray<float> in;
};

__device__ inline hit_case random_case(gpu::pcg32& rng, const vec3<float>* fixed_normal)
{
    hit_case h{};
    h.rec.p = {0.5f, 1.0f, -2.0f};
    h.rec.normal = fixed_normal ? *fixed_normal : gpu::random_unit_vector<float>(rng);
    vec3<float> d = gpu::random_unit_vector<float>(rng);
    if (gpu::dot(d, h.rec.normal) > 0.0f)
    {
        d = -d;
    }
    const float length = 0.5f + 2.5f * gpu::random_real<float>(rng);
    h.in = {h.rec.p - d, length * d};
    h.rec.front_face = gpu::random_real<float>(rng) < 0.5f;
    return h;
}

// --- S1: make_bsdf ---------------------------------------------------------------------------------------------------

struct make_mismatch
{
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const int kind = static_cast<int>(gpu::random_real<float>(rng) * 6) % 6;
        const gpu::material<float> m = case_material(kind);
        const hit_case h = random_case(rng, nullptr);
        const vec3<float> c{gpu::random_real<float>(rng), gpu::random_real<float>(rng), gpu::random_real<float>(rng)};
        const gpu::bsdf b = gpu::make_bsdf(m, nullptr, c, h.rec, h.in.direction);
        const gpu::lobe expected = kind == 0   ? gpu::lobe::lambertian
                                   : kind == 1 ? gpu::lobe::isotropic
                                               : gpu::lobe::books_delta;
        const vec3<float> wo = -gpu::unit_vector(h.in.direction);
        bool ok = b.kind == expected && bits_equal(b.albedo, c) && bits_equal(b.n, h.rec.normal) &&
                  fabsf(gpu::dot(b.wo, wo) - 1.0f) <= 1e-6f;
        if (expected == gpu::lobe::books_delta)
        {
            ok = ok && b.books.kind == m.kind && bits_equal(b.books.albedo, m.albedo) &&
                 bits_equal(b.books.fuzz, m.fuzz) && bits_equal(b.books.refraction_index, m.refraction_index);
        }
        return !ok;
    }
};

GPU_TEST(bsdf, make)
{
    const std::int64_t n = ctx.samples(small_samples);
    ctx.count("make", gpu_test::device_count(make_mismatch{}, 7301, n), 0, gpu_test::drawn(n));
}

// --- S2, S5, S7: weights, densities, finite draws, the delta lobes ---------------------------------------------------

// kind 0 or 1; the normal random. Counts a draw whose weight isn't eval/pdf or whose two densities differ.
struct weight_eval_mismatch
{
    int kind;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const hit_case h = random_case(rng, nullptr);
        const gpu::material<float> m = case_material(kind);
        const gpu::bsdf b = gpu::make_bsdf(m, nullptr, m.albedo, h.rec, h.in.direction);
        gpu::bsdf_sample s;
        if (!gpu::sample_bsdf(b, h.in, h.rec, rng, s))
        {
            return false;
        }
        float pdf = 0;
        const vec3<float> f = gpu::eval_bsdf(b, s.wi, pdf);
        const vec3<float> ratio = (1.0f / pdf) * f;
        const auto off = [](float w, float r) { return !(fabsf(w - r) <= 1e-5f * w); };
        return off(s.weight.x, ratio.x) || off(s.weight.y, ratio.y) || off(s.weight.z, ratio.z) ||
               !(fabsf(s.pdf - pdf) <= 1e-5f * pdf);
    }
};

// A true return whose direction isn't finite and unit, or whose weight or density isn't finite and positive
struct not_finite
{
    int kind;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const hit_case h = random_case(rng, nullptr);
        const gpu::material<float> m = case_material(kind);
        const gpu::bsdf b = gpu::make_bsdf(m, nullptr, m.albedo, h.rec, h.in.direction);
        gpu::bsdf_sample s;
        if (!gpu::sample_bsdf(b, h.in, h.rec, rng, s))
        {
            return false;
        }
        const bool unit = fabsf(sqrtf(s.wi.length_squared()) - 1.0f) <= 1e-5f;
        const bool weight = finite(s.weight) && s.weight.x >= 0 && s.weight.y >= 0 && s.weight.z >= 0;
        return !(finite(s.wi) && unit && weight && isfinite(s.pdf) && s.pdf > 0.0f);
    }
};

struct absorbed
{
    int kind;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const hit_case h = random_case(rng, nullptr);
        const gpu::material<float> m = case_material(kind);
        const gpu::bsdf b = gpu::make_bsdf(m, nullptr, m.albedo, h.rec, h.in.direction);
        gpu::bsdf_sample s;
        return !gpu::sample_bsdf(b, h.in, h.rec, rng, s);
    }
};

// Matte and smoke have a density: never a delta lobe (light samples are taken there, and the map seen after them is
// the nearest texel, not the filtered one)
struct non_delta_violation
{
    int kind;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const hit_case h = random_case(rng, nullptr);
        const gpu::material<float> m = case_material(kind);
        const gpu::bsdf b = gpu::make_bsdf(m, nullptr, m.albedo, h.rec, h.in.direction);
        gpu::bsdf_sample s;
        const bool drawn = gpu::sample_bsdf(b, h.in, h.rec, rng, s);
        return gpu::is_delta(b) || (drawn && s.delta);
    }
};

// The books' metal and glass: a delta lobe, no density, nothing for eval_bsdf
struct delta_violation
{
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const int kind = 2 + static_cast<int>(gpu::random_real<float>(rng) * 4) % 4;
        const gpu::material<float> m = case_material(kind);
        const hit_case h = random_case(rng, nullptr);
        const gpu::bsdf b = gpu::make_bsdf(m, nullptr, {}, h.rec, h.in.direction);
        gpu::bsdf_sample s;
        const bool drawn = gpu::sample_bsdf(b, h.in, h.rec, rng, s);
        float pdf = 1;
        const vec3<float> f = gpu::eval_bsdf(b, gpu::random_unit_vector<float>(rng), pdf);
        const bool sample_ok = !drawn || (s.pdf == 0.0f && s.delta);
        return !(gpu::is_delta(b) && sample_ok && pdf == 0.0f && f.x == 0.0f && f.y == 0.0f && f.z == 0.0f);
    }
};

GPU_TEST(bsdf, draws)
{
    const std::int64_t n = ctx.samples(draw_samples);
    const std::int64_t drawn = gpu_test::drawn(n);
    ctx.count("lambertian-weight-eval", gpu_test::device_count(weight_eval_mismatch{0}, 7302, n), 0, drawn);
    ctx.count("isotropic-weight-eval", gpu_test::device_count(weight_eval_mismatch{1}, 7303, n), 0, drawn);
    ctx.count("lambertian-finite", gpu_test::device_count(not_finite{0}, 7304, n), 0, drawn);
    ctx.count("isotropic-finite", gpu_test::device_count(not_finite{1}, 7305, n), 0, drawn);
    ctx.count("lambertian-absorbed", gpu_test::device_count(absorbed{0}, 7304, n), drawn / 100000, drawn);
    ctx.count("isotropic-absorbed", gpu_test::device_count(absorbed{1}, 7305, n), 0, drawn);
    const std::int64_t m = ctx.samples(small_samples);
    ctx.count("delta", gpu_test::device_count(delta_violation{}, 7306, m), 0, gpu_test::drawn(m));
    ctx.count("lambertian-not-delta", gpu_test::device_count(non_delta_violation{0}, 7307, m), 0, gpu_test::drawn(m));
    ctx.count("isotropic-not-delta", gpu_test::device_count(non_delta_violation{1}, 7308, m), 0, gpu_test::drawn(m));
}

// --- S3, S4, S8: the densities as distributions ------------------------------------------------------------------------

// A fixed lobe's draws, binned in a grid; -1 for an absorbed draw (its probability is 0 in the masses)
struct lobe_sampler
{
    int kind;
    vec3<float> n;
    direction_grid grid;
    __device__ int operator()(gpu::pcg32& rng) const
    {
        gpu::hit_record<float> rec{};
        rec.p = {0.5f, 1.0f, -2.0f};
        rec.normal = n;
        rec.front_face = true;
        const gpu::material<float> m = case_material(kind);
        const gpu::bsdf b = gpu::make_bsdf(m, nullptr, m.albedo, rec, -n);
        gpu::bsdf_sample s;
        return gpu::sample_bsdf(b, {rec.p + n, -n}, rec, rng, s) ? grid.cell(s.wi) : -1;
    }
};

// eval_bsdf's density in the grid (channel -1), or a channel of f·|cos θ|, times the grid's Jacobian
struct lobe_density_in_grid
{
    int kind;
    vec3<float> n;
    direction_grid grid;
    int channel;
    __device__ double operator()(double u, double v) const
    {
        const gpu::material<float> m = case_material(kind);
        gpu::hit_record<float> rec{};
        rec.normal = n;
        const gpu::bsdf b = gpu::make_bsdf(m, nullptr, m.albedo, rec, -n);
        float pdf = 0;
        const vec3<float> f = gpu::eval_bsdf(b, grid.direction(u, v), pdf);
        const float value = channel < 0 ? pdf : channel == 0 ? f.x : channel == 1 ? f.y : f.z;
        return value * grid.jacobian();
    }
};

// The planted fault: the isotropic draws against (1 + 0.05·z)/(4π), which must be rejected
struct planted_density_in_grid
{
    direction_grid grid;
    __device__ double operator()(double u, double v) const
    {
        const vec3<float> d = grid.direction(u, v);
        return (1 + 0.05 * d.z) / (4 * std::numbers::pi) * grid.jacobian();
    }
};

void test_lobe_distribution(gpu_test::context& ctx, const char* name, int kind, const gpu::vec3d& axis,
                            std::uint64_t seed, const vec3<float>& albedo)
{
    const vec3<float> n = to_float(axis);
    const direction_grid grid = gpu_test::grid_around(axis);
    const auto counts =
        gpu_test::device_histogram(lobe_sampler{kind, n, grid}, seed, ctx.samples(chi2_samples), grid.cells());
    const auto masses = gpu_test::cell_masses(lobe_density_in_grid{kind, n, grid, -1}, grid.nu, grid.nv);
    double mass = 0;
    const gpu_test::chi2_outcome chi2 = gpu_test::chi2_against(counts, masses, mass);
    const std::string prefix(name);
    ctx.chi2(prefix + "-chi2", chi2.statistic, chi2.dof, chi2.p);
    ctx.exact(prefix + "-mass", mass, 1, 1e-5);
    const float channels[] = {albedo.x, albedo.y, albedo.z};
    for (int c = 0; c < 3; ++c)
    {
        double sum = 0;
        for (const double m : gpu_test::cell_masses(lobe_density_in_grid{kind, n, grid, c}, grid.nu, grid.nv))
        {
            sum += m;
        }
        ctx.exact(prefix + "-albedo-" + "rgb"[c], sum, channels[c], 1e-5);
    }
    if (kind == 1)
    {
        double planted_mass = 0;
        const auto planted = gpu_test::cell_masses(planted_density_in_grid{grid}, grid.nu, grid.nv);
        const gpu_test::chi2_outcome wrong = gpu_test::chi2_against(counts, planted, planted_mass);
        ctx.must_reject("planted-anisotropic-chi2", wrong.statistic, wrong.dof, wrong.p);
    }
}

GPU_TEST(bsdf, distributions)
{
    test_lobe_distribution(ctx, "lambertian", 0, tilted, 7311, matte_albedo());
    test_lobe_distribution(ctx, "isotropic", 1, {0, 0, 1}, 7312, smoke_albedo());
}

// --- S6: the books' own numbers --------------------------------------------------------------------------------------

// Two copies of one generator: the books' code draws with one, the quality kernel's with the other. A case
// mismatches when the outcome, the direction (bit for bit), the weight or the next four numbers differ.
struct books_mismatch
{
    int kind;
    gpu::scene_view<float> view; // only `sampling` is read: material
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        // Two normals: random, and straight down (0, 0, -1)
        const vec3<float> down{0.0f, 0.0f, -1.0f};
        const bool fixed = gpu::random_real<float>(rng) < 0.25f;
        const hit_case h = random_case(rng, fixed ? &down : nullptr);
        const gpu::material<float> m = case_material(kind);
        gpu::pcg32 a = rng;
        gpu::pcg32 b = rng;

        vec3<float> books_direction{};
        vec3<float> books_weight{1.0f, 1.0f, 1.0f};
        bool books_drawn = false;
        if (kind <= 1)
        {
            float unused_weight = 0;
            books_drawn = gpu::diffuse_bounce(view, kind == 0, h.rec, -1, a, books_direction, unused_weight) !=
                          gpu::bounce::none;
            books_weight = m.albedo; // the throughput's factor: the texture's color, here the material's own
        }
        else
        {
            books_drawn = gpu::scatter(m, h.in.direction, h.rec, a, books_direction, books_weight);
        }

        const gpu::bsdf lobe = gpu::make_bsdf(m, nullptr, m.albedo, h.rec, h.in.direction);
        gpu::bsdf_sample s;
        const bool drawn = gpu::sample_bsdf(lobe, h.in, h.rec, b, s);

        bool same = drawn == books_drawn && bits_equal(s.wi, books_direction);
        if (drawn && books_drawn)
        {
            same = same && bits_equal(s.weight, books_weight);
        }
        for (int i = 0; i < 4; ++i)
        {
            same = same && a.next() == b.next();
        }
        return !same;
    }
};

GPU_TEST(bsdf, books_identical)
{
    gpu::scene_view<float> view{};
    view.sampling = gpu::sampling_strategy::material;
    const std::int64_t n = ctx.samples(draw_samples);
    constexpr const char* names[] = {"lambertian", "isotropic", "metal-fuzz-0", "metal-fuzz-0.3", "metal-fuzz-1",
                                     "glass"};
    for (int kind = 0; kind < 6; ++kind)
    {
        ctx.count(std::string("books-identical-") + names[kind],
                  gpu_test::device_count(books_mismatch{kind, view}, 7320 + kind, n), 0, gpu_test::drawn(n));
    }
}
}
