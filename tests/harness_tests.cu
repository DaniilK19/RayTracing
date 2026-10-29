// Suite harness: GpuTests' own checks, made on samplers known to be right, the books' own: their cosine-weighted and
// uniform directions and their sphere and quad lights (lights.cuh), which three books of renders and MonteCarlo's
// light-pdfs have checked. If the χ² machinery (the histogram, the cells' masses by quadrature, the pooling, the
// p-value) were too strict, these would fail; if it were too weak, the planted fault would pass: directions drawn with
// the density ∝ cos^1.05 θ, tested against the cosine density. The p-value function is checked against textbook
// quantiles first, on the host.
//
// The χ² tests bin each sampler's directions in a frame where its density's support is the whole grid, so that no
// cell is cut by the support's edge (a quadrature can't integrate a jump): the cosine lobe and the uniform sphere over
// the whole sphere (the lobe's horizon is a row boundary), a sphere light in the cone it fills, a quad light by the
// point of the quad a direction hits.
#include <cmath>
#include <numbers>

#include "gpu_test.cuh"
#include "gpu_test.h"
#include "lights.cuh"
#include "material.cuh"
#include "pdf.cuh"
#include "random.cuh"

namespace
{
using gpu::vec3;
using gpu_test::direction_grid;

constexpr double four_pi = 4 * std::numbers::pi;
constexpr std::int64_t chi2_samples = 1'000'000;
constexpr std::int64_t mean_samples = 1'000'000;

vec3<float> to_float(const gpu::vec3d& v) { return gpu::to_real<float>(v); }

// --- The cosine lobe around a tilted normal --------------------------------------------------------------------------

// The normal of matte_bounce in MonteCarlo's light-pdfs: tilted, so that no axis of the books' basis (around) is
// special
const gpu::vec3d tilted{1 / std::sqrt(14.0), 2 / std::sqrt(14.0), 3 / std::sqrt(14.0)};

struct cosine_sampler
{
    vec3<float> n;
    direction_grid grid;
    __device__ int operator()(gpu::pcg32& rng) const
    {
        return grid.cell(gpu::around(n, gpu::random_cosine_direction<float>(rng)));
    }
};

// The planted fault: the density ∝ cos^1.05 θ, which a χ² test of 10^6 samples against cos θ / π must reject
struct planted_sampler
{
    vec3<float> n;
    direction_grid grid;
    __device__ int operator()(gpu::pcg32& rng) const
    {
        const float r1 = gpu::random_real<float>(rng);
        const float r2 = gpu::random_real<float>(rng);
        float sin_phi;
        float cos_phi;
        sincospif(2 * r1, &sin_phi, &cos_phi);
        const float c = powf(r2, 1 / 2.05f); // the inverse of the cumulative c^2.05
        const float s = sqrtf(fmaxf(0.0f, 1 - c * c));
        return grid.cell(gpu::around(n, vec3<float>{cos_phi * s, sin_phi * s, c}));
    }
};

struct cosine_density_in_grid
{
    vec3<float> n;
    direction_grid grid;
    __device__ double operator()(double u, double v) const
    {
        return gpu::cosine_density(n, grid.direction(u, v)) * grid.jacobian();
    }
};

// 4π·p over uniform directions: 1 for any density
struct cosine_over_uniform
{
    vec3<float> n;
    __device__ double operator()(gpu::pcg32& rng) const
    {
        return four_pi * gpu::cosine_density(n, gpu::random_unit_vector<float>(rng));
    }
};

// A drawn direction whose density is not above 0 (the bounce would be absorbed)
struct cosine_zero_density
{
    vec3<float> n;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        return !(gpu::cosine_density(n, gpu::around(n, gpu::random_cosine_direction<float>(rng))) > 0.0f);
    }
};

GPU_TEST(harness, p_values)
{
    // Quantiles of χ² from the tables (Abramowitz and Stegun, table 26.8; NIST's handbook for 1000 degrees of freedom)
    ctx.exact("chi2-p(3.841459, 1 dof)", gpu_test::chi2_p_value(3.841459, 1), 0.05, 1e-6);
    ctx.exact("chi2-p(23.209251, 10 dof)", gpu_test::chi2_p_value(23.209251, 10), 0.01, 1e-6);
    ctx.exact("chi2-p(124.342113, 100 dof)", gpu_test::chi2_p_value(124.342113, 100), 0.05, 1e-6);
    ctx.exact("chi2-p(1106.969, 1000 dof)", gpu_test::chi2_p_value(1106.969, 1000), 0.01, 1e-5);
    ctx.exact("gamma-Q(1, 2) = e^-2", gpu_test::regularized_gamma_q(1, 2), std::exp(-2.0), 1e-12);
    // The series branch (x < a + 1), which computes the p-values of tests that pass: the median and the lower 5%
    // quantile of χ² with 10 degrees of freedom, closed forms for whole a (Q(a, x) = e^-x Σ_{k<a} x^k/k!, the chance
    // that a Poisson count of mean x is below a), and that sum for a = 500, computed term by term in logarithms
    ctx.exact("chi2-p(9.341818, 10 dof)", gpu_test::chi2_p_value(9.341818, 10), 0.5, 1e-6);
    ctx.exact("chi2-p(3.940299, 10 dof)", gpu_test::chi2_p_value(3.940299, 10), 0.95, 1e-6);
    ctx.exact("gamma-Q(1, 0.5) = e^-0.5", gpu_test::regularized_gamma_q(1, 0.5), std::exp(-0.5), 1e-12);
    ctx.exact("gamma-Q(3, 2) = 5 e^-2", gpu_test::regularized_gamma_q(3, 2), 5 * std::exp(-2.0), 1e-12);
    double poisson = 0;
    for (int k = 0; k < 500; ++k)
    {
        poisson += std::exp(-480 + k * std::log(480.0) - std::lgamma(k + 1.0));
    }
    ctx.exact("gamma-Q(500, 480) = Poisson sum", gpu_test::regularized_gamma_q(500, 480), poisson, 1e-10);
    ctx.exact("normal-p(1.959964)", gpu_test::normal_two_sided_p(1.959964), 0.05, 1e-6);
    // A pooled cell expected to hold nothing that holds a sample: impossible, p = 0. One bin: no test, p is NaN.
    ctx.exact("chi2-impossible-sample", gpu_test::chi2_test({100, 100, 1}, {0.5, 0.5, 0}).p, 0, 0);
    ctx.exact("chi2-one-bin-is-no-test", std::isnan(gpu_test::chi2_test({10}, {1}).p) ? 1 : 0, 1, 0);
}

GPU_TEST(harness, cosine)
{
    const vec3<float> n = to_float(tilted);
    const direction_grid grid = gpu_test::grid_around(tilted);
    const std::int64_t samples = ctx.samples(chi2_samples);
    const auto counts = gpu_test::device_histogram(cosine_sampler{n, grid}, 7101, samples, grid.cells());
    const auto masses = gpu_test::cell_masses(cosine_density_in_grid{n, grid}, grid.nu, grid.nv);
    double mass = 0;
    const gpu_test::chi2_outcome chi2 = gpu_test::chi2_against(counts, masses, mass);
    ctx.chi2("cosine-chi2", chi2.statistic, chi2.dof, chi2.p);
    ctx.exact("cosine-cell-masses-sum", mass, 1, 1e-5);
    ctx.count("cosine-zero-density-draws", gpu_test::device_count(cosine_zero_density{n}, 7101, samples), 0,
              gpu_test::drawn(samples));
    const gpu_test::mean_estimate integral =
        gpu_test::device_mean(cosine_over_uniform{n}, 7102, ctx.samples(mean_samples));
    ctx.mean("cosine-integral", integral.mean, integral.standard_error, 1);
    ctx.count("cosine-integral-left-out", integral.left_out, 0, gpu_test::drawn(ctx.samples(mean_samples)));

    const auto planted = gpu_test::device_histogram(planted_sampler{n, grid}, 7103, samples, grid.cells());
    const gpu_test::chi2_outcome wrong = gpu_test::chi2_against(planted, masses, mass);
    ctx.must_reject("planted-cos^1.05-chi2", wrong.statistic, wrong.dof, wrong.p);
}

// --- The uniform sphere ----------------------------------------------------------------------------------------------

struct uniform_sampler
{
    direction_grid grid;
    __device__ int operator()(gpu::pcg32& rng) const { return grid.cell(gpu::random_unit_vector<float>(rng)); }
};

struct uniform_density_in_grid
{
    direction_grid grid;
    __device__ double operator()(double, double) const
    {
        return gpu::uniform_sphere_density<float> * grid.jacobian();
    }
};

GPU_TEST(harness, uniform)
{
    const direction_grid grid = gpu_test::grid_around({0, 0, 1});
    const auto counts =
        gpu_test::device_histogram(uniform_sampler{grid}, 7111, ctx.samples(chi2_samples), grid.cells());
    const auto masses = gpu_test::cell_masses(uniform_density_in_grid{grid}, grid.nu, grid.nv);
    double mass = 0;
    const gpu_test::chi2_outcome chi2 = gpu_test::chi2_against(counts, masses, mass);
    ctx.chi2("uniform-chi2", chi2.statistic, chi2.dof, chi2.p);
    ctx.exact("uniform-cell-masses-sum", mass, 1, 1e-6);
}

// --- A sphere light --------------------------------------------------------------------------------------------------

gpu::light<float> sphere_light(const gpu::vec3d& center, double radius)
{
    gpu::light<float> l{};
    l.kind = gpu::light_shape::sphere;
    l.center = to_float(center);
    l.radius = static_cast<float>(radius);
    l.weight = 1;
    l.cdf = 1;
    l.twin = -1;
    return l;
}

struct sphere_sampler
{
    gpu::light<float> l;
    vec3<float> origin;
    direction_grid grid;
    __device__ int operator()(gpu::pcg32& rng) const { return grid.cell(gpu::sphere_light_random(l, origin, rng)); }
};

struct sphere_density_in_grid
{
    gpu::light<float> l;
    vec3<float> origin;
    direction_grid grid;
    __device__ double operator()(double u, double v) const
    {
        return gpu::sphere_light_pdf(l, origin, grid.direction(u, v), false) * grid.jacobian();
    }
};

struct sphere_over_uniform
{
    gpu::light<float> l;
    vec3<float> origin;
    __device__ double operator()(gpu::pcg32& rng) const
    {
        return four_pi * gpu::sphere_light_pdf(l, origin, gpu::random_unit_vector<float>(rng), false);
    }
};

// 1/p over the light's own directions: its solid angle Ω, the same for every direction of a uniform cone
struct sphere_inverse_density
{
    gpu::light<float> l;
    vec3<float> origin;
    __device__ double operator()(gpu::pcg32& rng) const
    {
        const float p = gpu::sphere_light_pdf(l, origin, gpu::sphere_light_random(l, origin, rng), false);
        return p > 0 ? 1.0 / p : nan("");
    }
};

struct sphere_zero_density
{
    gpu::light<float> l;
    vec3<float> origin;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        return !(gpu::sphere_light_pdf(l, origin, gpu::sphere_light_random(l, origin, rng), false) > 0.0f);
    }
};

// The light from a point outside it, near (it fills a cone of half-angle 42°) or far (5.7°)
void test_sphere_light(gpu_test::context& ctx, const char* name, double distance, std::uint64_t seed)
{
    const gpu::vec3d origin{0.25, -0.5, 1};
    const gpu::vec3d to_center = gpu::scale(distance / std::sqrt(14.0), gpu::vec3d{2, 3, -1});
    const gpu::vec3d center{origin.x + to_center.x, origin.y + to_center.y, origin.z + to_center.z};
    constexpr double radius = 1;
    const gpu::light<float> l = sphere_light(center, radius);
    const vec3<float> o = to_float(origin);
    // The cone in double from the float numbers the GPU has
    const gpu::vec3d c{l.center.x - o.x, l.center.y - o.y, l.center.z - o.z};
    const double d2 = gpu::dot(c, c);
    const double cos_max = std::sqrt(1 - double(l.radius) * l.radius / d2);
    const double solid_angle = 2 * std::numbers::pi * (1 - cos_max);
    const direction_grid grid = gpu_test::grid_around(c, cos_max, 1);

    const std::int64_t samples = ctx.samples(chi2_samples);
    const auto counts = gpu_test::device_histogram(sphere_sampler{l, o, grid}, seed, samples, grid.cells());
    const auto masses = gpu_test::cell_masses(sphere_density_in_grid{l, o, grid}, grid.nu, grid.nv);
    double mass = 0;
    const gpu_test::chi2_outcome chi2 = gpu_test::chi2_against(counts, masses, mass);
    const std::string prefix = std::string("sphere-light-") + name;
    ctx.chi2(prefix + "-chi2", chi2.statistic, chi2.dof, chi2.p);
    ctx.exact(prefix + "-cell-masses-sum", mass, 1, 1e-5);
    // Directions at the cone's very edge, where the hit test of the density can round to a miss
    const std::int64_t drawn = gpu_test::drawn(samples);
    ctx.count(prefix + "-zero-density-draws", gpu_test::device_count(sphere_zero_density{l, o}, seed, samples),
              drawn / 100000, drawn);
    const gpu_test::mean_estimate integral =
        gpu_test::device_mean(sphere_over_uniform{l, o}, seed + 1, ctx.samples(mean_samples));
    ctx.mean(prefix + "-integral", integral.mean, integral.standard_error, 1);
    const std::int64_t mean_drawn = gpu_test::drawn(ctx.samples(mean_samples));
    ctx.count(prefix + "-integral-left-out", integral.left_out, 0, mean_drawn);
    const gpu_test::mean_estimate inverse =
        gpu_test::device_mean(sphere_inverse_density{l, o}, seed + 2, ctx.samples(mean_samples));
    ctx.exact(prefix + "-solid-angle", inverse.mean / solid_angle, 1, 1e-5);
    // 1/p is left out where the light's own direction got the density 0: the edge draws counted above
    ctx.count(prefix + "-solid-angle-left-out", inverse.left_out, mean_drawn / 100000, mean_drawn);
}

GPU_TEST(harness, sphere_light)
{
    test_sphere_light(ctx, "near", 1.5, 7121);
    test_sphere_light(ctx, "far", 10, 7131);
}

// --- A quad light ----------------------------------------------------------------------------------------------------

// The quad's plane in double, to find where a drawn direction hits it: (a, b) of q + a·u + b·v
struct quad_plane
{
    vec3<double> origin, q, u, v, normal, alpha_axis, beta_axis;
    double area;
};

struct quad_sampler
{
    gpu::light<float> l;
    vec3<float> origin;
    quad_plane plane;
    int cells_per_side;
    __device__ int operator()(gpu::pcg32& rng) const
    {
        const vec3<float> d = gpu::quad_light_random(l, origin, rng);
        const vec3<double> e{d.x, d.y, d.z};
        const double t = gpu::dot(plane.normal, plane.q - plane.origin) / gpu::dot(plane.normal, e);
        const vec3<double> p = plane.origin + t * e - plane.q;
        const double a = gpu::dot(p, plane.alpha_axis);
        const double b = gpu::dot(p, plane.beta_axis);
        // A hair past an edge, by the float direction's rounding, is in the edge's cell
        if (!(a >= -1e-5 && a <= 1 + 1e-5 && b >= -1e-5 && b <= 1 + 1e-5))
        {
            return -1;
        }
        const int i = ::min(cells_per_side - 1, ::max(0, static_cast<int>(a * cells_per_side)));
        const int j = ::min(cells_per_side - 1, ::max(0, static_cast<int>(b * cells_per_side)));
        return j * cells_per_side + i;
    }
};

// The density of the hit point (a, b): p(ω) times the solid angle per unit of a·b, area·cos θ / r². 1 everywhere if
// quad_light_pdf is the density of quad_light_random.
struct quad_density_in_grid
{
    gpu::light<float> l;
    vec3<float> origin;
    quad_plane plane;
    __device__ double operator()(double a, double b) const
    {
        const vec3<double> to = plane.q + a * plane.u + b * plane.v - plane.origin;
        const double r2 = gpu::dot(to, to);
        const vec3<double> unit = (1 / ::sqrt(r2)) * to;
        const vec3<float> d{static_cast<float>(unit.x), static_cast<float>(unit.y), static_cast<float>(unit.z)};
        return gpu::quad_light_pdf(l, origin, d) * plane.area * ::fabs(gpu::dot(unit, plane.normal)) / r2;
    }
};

struct quad_over_uniform
{
    gpu::light<float> l;
    vec3<float> origin;
    __device__ double operator()(gpu::pcg32& rng) const
    {
        return four_pi * gpu::quad_light_pdf(l, origin, gpu::random_unit_vector<float>(rng));
    }
};

struct quad_inverse_density
{
    gpu::light<float> l;
    vec3<float> origin;
    __device__ double operator()(gpu::pcg32& rng) const
    {
        const float p = gpu::quad_light_pdf(l, origin, gpu::quad_light_random(l, origin, rng));
        return p > 0 ? 1.0 / p : nan("");
    }
};

struct quad_zero_density
{
    gpu::light<float> l;
    vec3<float> origin;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        return !(gpu::quad_light_pdf(l, origin, gpu::quad_light_random(l, origin, rng)) > 0.0f);
    }
};

// The solid angle of the triangle (a, b, c) seen from the origin of the vectors: Van Oosterom and Strackee's formula
// (IEEE Trans. Biomed. Eng., 1983), tan(Ω/2) = |a·(b × c)| / (|a||b||c| + (a·b)|c| + (a·c)|b| + (b·c)|a|)
double triangle_solid_angle(const gpu::vec3d& a, const gpu::vec3d& b, const gpu::vec3d& c)
{
    const double la = std::sqrt(gpu::dot(a, a));
    const double lb = std::sqrt(gpu::dot(b, b));
    const double lc = std::sqrt(gpu::dot(c, c));
    const double numerator = std::abs(gpu::dot(a, gpu::cross(b, c)));
    const double denominator = la * lb * lc + gpu::dot(a, b) * lc + gpu::dot(a, c) * lb + gpu::dot(b, c) * la;
    return 2 * std::atan2(numerator, denominator);
}

GPU_TEST(harness, quad_light)
{
    // A parallelogram, tilted, seen from a point in front of it
    const gpu::quad_desc desc{.q = {-0.5, 1, -0.3}, .u = {1.2, 0.1, 0}, .v = {0, 0.3, 0.9}};
    const gpu::vec3d origin_d{0.1, -0.4, 0.2};
    gpu::light<float> l{};
    l.kind = gpu::light_shape::quad;
    l.quad = gpu::make_quad<float>(desc);
    l.u = to_float(desc.u);
    l.v = to_float(desc.v);
    const gpu::vec3d n = gpu::cross(desc.u, desc.v);
    const double area = std::sqrt(gpu::dot(n, n));
    l.area = static_cast<float>(area);
    l.weight = 1;
    l.cdf = 1;
    l.twin = -1;
    const vec3<float> o = to_float(origin_d);

    // The plane from the float numbers the GPU has, in double
    const auto d = [](const vec3<float>& f) { return vec3<double>{f.x, f.y, f.z}; };
    const auto dd = [](const gpu::vec3d& v) { return vec3<double>{v.x, v.y, v.z}; };
    const gpu::vec3d w = gpu::scale(1 / gpu::dot(n, n), n);
    quad_plane plane{.origin = d(o), .q = d(l.quad.q), .u = d(l.u), .v = d(l.v),
                     .normal = dd(gpu::scale(1 / area, n)), .alpha_axis = dd(gpu::cross(desc.v, w)),
                     .beta_axis = dd(gpu::cross(w, desc.u)), .area = area};
    constexpr int side = 50;

    const std::int64_t samples = ctx.samples(chi2_samples);
    const auto counts = gpu_test::device_histogram(quad_sampler{l, o, plane, side}, 7141, samples, side * side);
    const auto masses = gpu_test::cell_masses(quad_density_in_grid{l, o, plane}, side, side);
    double mass = 0;
    const gpu_test::chi2_outcome chi2 = gpu_test::chi2_against(counts, masses, mass);
    ctx.chi2("quad-light-chi2", chi2.statistic, chi2.dof, chi2.p);
    ctx.exact("quad-light-cell-masses-sum", mass, 1, 1e-5);
    const std::int64_t drawn = gpu_test::drawn(samples);
    ctx.count("quad-light-zero-density-draws", gpu_test::device_count(quad_zero_density{l, o}, 7141, samples),
              drawn / 100000, drawn);
    const gpu_test::mean_estimate integral =
        gpu_test::device_mean(quad_over_uniform{l, o}, 7142, ctx.samples(mean_samples));
    ctx.mean("quad-light-integral", integral.mean, integral.standard_error, 1);
    const std::int64_t mean_drawn = gpu_test::drawn(ctx.samples(mean_samples));
    ctx.count("quad-light-integral-left-out", integral.left_out, 0, mean_drawn);

    // Its solid angle, two triangles from the origin
    const gpu::vec3d corner{desc.q.x - origin_d.x, desc.q.y - origin_d.y, desc.q.z - origin_d.z};
    const gpu::vec3d c1{corner.x + desc.u.x, corner.y + desc.u.y, corner.z + desc.u.z};
    const gpu::vec3d c3{corner.x + desc.v.x, corner.y + desc.v.y, corner.z + desc.v.z};
    const gpu::vec3d c2{c1.x + desc.v.x, c1.y + desc.v.y, c1.z + desc.v.z};
    const double solid_angle = triangle_solid_angle(corner, c1, c2) + triangle_solid_angle(corner, c2, c3);
    const gpu_test::mean_estimate inverse =
        gpu_test::device_mean(quad_inverse_density{l, o}, 7143, ctx.samples(mean_samples));
    ctx.mean("quad-light-solid-angle", inverse.mean, inverse.standard_error, solid_angle);
    ctx.count("quad-light-solid-angle-left-out", inverse.left_out, mean_drawn / 100000, mean_drawn);
}
}
