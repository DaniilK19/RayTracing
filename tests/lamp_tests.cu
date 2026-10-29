// Suites lamps and media: the light sampled at every bounce and weighed against the material's own samples
// (next-event estimation and multiple importance sampling, Step 76), and shadow rays through smoke and fog (Step 77).
// Written from the spec alone (out/m4/spec/D.md sections 1-8, shared.md 0-5): the implementation's files were not read,
// and every reference here is the test's own:
//   - the lamps of the light set (build_lamps) against D.md's membership rules and fields, rebuilt here (L0);
//   - a light sample lands where it says, on its lamp's front, with the density lamp_hit_pdf gives that hit (T1);
//   - the lamps' densities against closed forms: E_unif[4π·p] = 1 and E[1/p] = Ω, the rectangle's Ω by
//     inclusion-exclusion and the cap's 2π(1 − cos θmax) (T2);
//   - Veach's condition (W1) on the weights, w_L + w_B = 1 for a direction either technique draws (T3), and the power
//     heuristic's limits (W1 in the table below);
//   - the estimator at one vertex against Lambert's polygon formula (1760; Arvo 1995) and a sphere's irradiance
//     π·L·sin²θmax·cos β (T4);
//   - the depth and counting rules on a toy scene rendered through gpu::render (T6), and through smoke (T6m);
//   - the shadow ray's any-hit search against a loop over every primitive (T7);
//   - the media's transmittance against medium_collision's own free flights and against chords in closed form (T5).
// Each suite has planted faults (X1-X3) that the same checks must reject.
//
// Step criteria (D.md §9; $GT = GpuTests, $PS = PfmStats, $S = -w 480 --height 270):
//   Step 76: $GT lamps exits 0. compare.ps1 -Scene cornell -A mixture@books:ref -B mis:3,4 -Spp 1024 and -A mis:3,4
//     -B nee:5,6, the same for light ($S): tests, controls and the RMS rule pass (shared.md §12's re-run rule);
//     $PS noise trimmed σ_mis < σ_mixture on both scenes; ratios, rays and shadow rays per sample into the commit. The
//     11 media-free book scenes -p mis -w 64 -s 16: $PS stats --finite; the 7 sky-lit ones print "(no lights: the sky
//     is the light)" and a books' kernel; light, cornell, aluminum, glass print "Kernel: quality" and a "Shadow rays:"
//     line. Viewer = GPU: cornell -p mis 160×90, 16 spp, seed 3, identical PFMs. HOST: the nee and mis help lines;
//     cornell -p mis --fp64 and final -p mis refused, exit 1. Sanitizer: cornell -p mis, light -p nee, $GT lamps
//     --quick; CARRY (base mode); recorded: cornell, light -p mis $S -s 64 --seed 3.
//   Step 77: $GT lamps media exits 0. compare.ps1 -Scene final -A mixture@books:ref -B mis:3,4 -Spp 1024 and -A mis:3,4
//     -B nee:5,6 pass; σ_mis < σ_mixture. smoke -p mis -w 64 -s 16 finite. HOST: 76's two refusals gone. Sanitizer:
//     final -p mis, smoke -p nee, $GT media --quick. CARRY: cornell and light -p mis identical to 76; recorded: final
//     -p mis $S -s 64 --seed 3.
//
// Decisions where the spec leaves room:
//   1. T1: light's sphere and quad can hide each other from some points of P_light. A sample whose ray meets the other
//      lamp first is checked against its own lamp alone (hit_quad/hit_sphere on the tree's primitive) and counted INFO
//      ("blocked"); "the hit is ls.skip's front" holds for the others. A miss, a back hit or |t − distance| beyond
//      10^-4·distance is an "edge" (≤ 10^-4 of the samples drawn); a bad density, distance, direction, lamp id or
//      radiance, and a density off lamp_hit_pdf's by 10^-4, are violations (0 allowed).
//   2. T1 also checks two contracts of §3.3 the table doesn't list: the lamp picked is lamp k = min(⌊ξN⌋, N − 1) of the
//      first number (t1-pick), and a lamp sample draws exactly 3 numbers, true or false (t1-draws, §10.2).
//   3. T1b: P_light excludes D² ≤ 4.04 around the sphere lamp, so a tenth of its points are drawn inside that lamp
//      (radius 1.98), where D² < 0.9999r² must give false.
//   4. T3(a) asks W1 only where it applies: directions with f = 0 (below a matte surface) and directions whose traced
//      hit isn't the sampled lamp's front are skipped. T3's lambertian normals face the centre of a lamp picked
//      uniformly. T3 traces with bvh_tracer::closest; T3p (inside smoke) with closest_hit, the surfaces alone, so that
//      a particle doesn't hide the lamp: the media's T is common to both techniques and cancels from W1 (§2.1).
//   5. T4, T2 and T2c take their lamps by hand from §3.2 (independent of build_lamps); T1, T3, T3p take build_lamps',
//      T6 the renderer's own.
//   6. T6d (rays per sample within 0.2%) is reported as INFO under --quick: at 4096 samples the rays per sample have an
//      SE of ~1%. T6b is one-sided: must_reject with p = P(|Z| ≥ z) for z > 0, 1 otherwise. T6c allows
//      max(2, ⌊10^-4·N_cam⌋).
//   7. T7 (beyond the spec): besides §6.2's band around t_max, rays within float rounding of another
//      decision of hit_sphere/hit_quad (a grazing sphere, a quad's edge, a root at t_min = 0.001) are excluded and
//      counted, with their mismatches, as INFO. nvcc may contract the dot products of the reference's copy of those
//      functions differently from occluded's (shared.md §13.8's issue), which flips such a decision now and then
//      (estimated ~1 per 10^6 rays). Both runs (leaf sizes 1 and 8) are checks of their own, in 4 calls of 2.5·10^5
//      rays each.
//   8. W1: bsdf_weight(bsdf_only, ...) = 1 for every input (D.md's table); light_weight(power) and bsdf_weight(power)
//      equal power_weight within 2.4·10^-7 (2 ulp at 1) rather than bitwise; P(∞, ∞) and P(a, NaN) only in [0, 1].
//   9. T5c's "no random number drawn" holds by D2's signature (transmittance takes no generator): nothing to run.
//  10. T5d uses axis-aligned directions: |d|² = 1 and d·oc are then exact whatever nvcc fuses, so transmittance and the
//      oracle find the boundary's roots bit for bit, and the 0.0001 exit rule decides alike in both.
//  11. T5v's "exact": 0 exactly when blocked, else |vis − T| ≤ 10^-6·T (one function inlined twice may round apart);
//      rays within 10^-3·t_max of the wall's hit are left out.
//  12. X3's fault is t_in = line_in with no clamp at all (max(t_in, 0) alone would restore the clamp to within 0.001).
//  13. MR is the box turned 30° about y first, then 20° about x: p' = R_x(20°)·R_y(30°)·p.
//  14. A check that a test's own closed forms reproduce D.md §5's numbers (lamps.references) guards the references.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "bsdf.cuh"
#include "bvh.cuh"
#include "camera.cuh"
#include "constant_medium.cuh"
#include "gpu_test.cuh"
#include "gpu_test.h"
#include "lamps.cuh"
#include "light_sampler.cuh"
#include "material.cuh"
#include "mis.cuh"
#include "pdf.cuh"
#include "quad.cuh"
#include "quality_view.cuh"
#include "random.cuh"
#include "renderer.h"
#include "shadow.cuh"
#include "sphere.cuh"
#include "statistics.h"
#include "test_scene.h"
#include "tracer.cuh"
#include "vec3.cuh"

// The kernels of this file, laid out as the harness's (gpu_test.cuh): 4096 threads, thread t drawing from
// sample_generator(seed, t, 0), per_thread(n) samples each, but with several results per sample
namespace lamp_kernels
{
// K counts of events: f(rng, tally) adds to tally[k] for each event of its sample
template <int K, class F>
__global__ void tally_kernel(F f, std::uint64_t seed, int count, unsigned long long* tallies)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    gpu::pcg32 rng = gpu::sample_generator(seed, static_cast<std::uint64_t>(t), 0);
    unsigned long long local[K] = {};
    for (int i = 0; i < count; ++i)
    {
        f(rng, local);
    }
    for (int k = 0; k < K; ++k)
    {
        atomicAdd(&tallies[k], local[k]);
    }
}

// K sums: each thread's own at sums[K·t + k], added on the host in the threads' order, the same on every run
template <int K, class F>
__global__ void sums_kernel(F f, std::uint64_t seed, int count, double* sums)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    gpu::pcg32 rng = gpu::sample_generator(seed, static_cast<std::uint64_t>(t), 0);
    double local[K] = {};
    for (int i = 0; i < count; ++i)
    {
        f(rng, local);
    }
    for (int k = 0; k < K; ++k)
    {
        sums[K * t + k] = local[k];
    }
}

// out[i] = f(i): a handful of cases, one thread each
template <class F, class T>
__global__ void map_kernel(F f, int n, T* out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = f(i);
    }
}
}

namespace
{
using gpu::vec3;
using dvec = gpu::vec3<double>;
using gpu::light_combination;

constexpr std::int64_t full_samples = 1'000'000;
constexpr float infinite = std::numeric_limits<float>::infinity();
constexpr light_combination combinations[] = {light_combination::bsdf_only, light_combination::lights_only,
                                              light_combination::power_heuristic};

const char* mode_name(light_combination c)
{
    return c == light_combination::bsdf_only ? "material" : c == light_combination::lights_only ? "nee" : "mis";
}

std::string number(double x)
{
    char text[32];
    std::snprintf(text, sizeof text, "%.6g", x);
    return text;
}

// --- The host's side of the kernels ----------------------------------------------------------------------------------

template <int K, class F>
std::array<std::int64_t, K> device_tally(const F& f, std::uint64_t seed, std::int64_t n)
{
    gpu_test::device_buffer<unsigned long long> tallies(static_cast<std::size_t>(K));
    lamp_kernels::tally_kernel<K><<<gpu_test::test_blocks, gpu_test::test_block_size>>>(
        f, seed, gpu_test::per_thread(n), tallies.get());
    gpu_test::finish_launch("a tally kernel");
    const std::vector<unsigned long long> t = tallies.download();
    std::array<std::int64_t, K> out{};
    for (std::size_t k = 0; k < out.size(); ++k)
    {
        out[k] = static_cast<std::int64_t>(t[k]);
    }
    return out;
}

template <int K, class F>
std::array<double, K> device_sums(const F& f, std::uint64_t seed, std::int64_t n)
{
    gpu_test::device_buffer<double> sums(static_cast<std::size_t>(K) * gpu_test::test_threads);
    lamp_kernels::sums_kernel<K><<<gpu_test::test_blocks, gpu_test::test_block_size>>>(
        f, seed, gpu_test::per_thread(n), sums.get());
    gpu_test::finish_launch("a sums kernel");
    const std::vector<double> s = sums.download();
    std::array<double, K> out{};
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        out[i % K] += s[i];
    }
    return out;
}

template <class T, class F>
std::vector<T> device_map(const F& f, int n)
{
    gpu_test::device_buffer<T> out(static_cast<std::size_t>(n));
    lamp_kernels::map_kernel<<<(n + 127) / 128, 128>>>(f, n, out.get());
    gpu_test::finish_launch("a map kernel");
    return out.download();
}

template <std::size_t K>
void add(std::array<std::int64_t, K>& sum, const std::array<std::int64_t, K>& more)
{
    for (std::size_t k = 0; k < K; ++k)
    {
        sum[k] += more[k];
    }
}

// --- Vectors in double, on the host ----------------------------------------------------------------------------------

dvec to_double(const vec3<float>& v) { return {v.x, v.y, v.z}; }
dvec to_double(const gpu::vec3d& v) { return {v.x, v.y, v.z}; }
dvec cross(const dvec& a, const dvec& b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double length(const dvec& v) { return std::sqrt(gpu::dot(v, v)); }
dvec unit(const dvec& v) { return (1 / length(v)) * v; }

// The scenes' own random numbers, on the host: SplitMix64 (Steele, Lea and Flood, 2014), seeded with a literal
struct host_random
{
    std::uint64_t state;

    std::uint64_t next()
    {
        std::uint64_t z = state += 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    double uniform(double lo, double hi) { return lo + (hi - lo) * static_cast<double>(next() >> 11) * 0x1p-53; }
    dvec direction()
    {
        const double z = uniform(-1, 1);
        const double phi = uniform(0, 2 * std::numbers::pi);
        const double r = std::sqrt(std::max(0.0, 1 - z * z));
        return {r * std::cos(phi), r * std::sin(phi), z};
    }
};

// --- The references, in double ---------------------------------------------------------------------------------------

// Lambert's polygon formula (Lambert 1760; Arvo, "Applications of irradiance tensors", 1995): the projected solid angle
// Ω⊥ = ½·|Σ γ_i (m_i·n)| of a polygon wholly above the horizon of n at x, γ_i the angle between the unit vectors to
// two consecutive corners and m_i the unit normal of the plane through them and x. A white lambertian point under a
// polygon of radiance L reflects L·Ω⊥/π.
double projected_solid_angle(const dvec& x, const dvec& n, const std::vector<dvec>& corners)
{
    double sum = 0;
    for (std::size_t i = 0; i < corners.size(); ++i)
    {
        const dvec a = unit(corners[i] - x);
        const dvec b = unit(corners[(i + 1) % corners.size()] - x);
        const dvec c = cross(a, b);
        const double s = length(c);
        sum += std::atan2(s, gpu::dot(a, b)) * gpu::dot(c, n) / s;
    }
    return 0.5 * std::abs(sum);
}

std::vector<dvec> corners_of(const gpu::quad_desc& q)
{
    const dvec p = to_double(q.q);
    const dvec u = to_double(q.u);
    const dvec v = to_double(q.v);
    return {p, p + u, p + u + v, p + v};
}

// The same integral by quadrature over the parallelogram, ∫ cos θ_x·cos θ_q / r² dA: smooth while x is off the quad, so
// 8×8 panels of a 16-point Gauss-Legendre rule give it to ~10^-12. An independent check of the formula above.
double projected_solid_angle_by_quadrature(const dvec& x, const dvec& n, const gpu::quad_desc& quad)
{
    const gpu_test::gauss_rule rule = gpu_test::make_gauss_rule(16);
    constexpr int panels = 8;
    const dvec q = to_double(quad.q);
    const dvec u = to_double(quad.u);
    const dvec v = to_double(quad.v);
    const dvec normal = cross(u, v);
    const double area = length(normal);
    const dvec unit_normal = (1 / area) * normal;
    double sum = 0;
    for (int pa = 0; pa < panels; ++pa)
    {
        for (int pb = 0; pb < panels; ++pb)
        {
            for (int i = 0; i < rule.order; ++i)
            {
                for (int j = 0; j < rule.order; ++j)
                {
                    const double a = (pa + 0.5 * (rule.x[i] + 1)) / panels;
                    const double b = (pb + 0.5 * (rule.x[j] + 1)) / panels;
                    const dvec w = q + a * u + b * v - x;
                    const double r2 = gpu::dot(w, w);
                    sum += rule.w[i] * rule.w[j] * gpu::dot(n, w) * std::abs(gpu::dot(unit_normal, w)) / (r2 * r2);
                }
            }
        }
    }
    return sum * area * 0.25 / (panels * panels);
}

// Howell's catalog of configuration factors, B-3: a point under a corner of a parallel rectangle A·h × B·h at height h
double howell_b3(double a, double b)
{
    const double sa = std::sqrt(1 + a * a);
    const double sb = std::sqrt(1 + b * b);
    return (a / sa * std::atan(b / sa) + b / sb * std::atan(a / sb)) / (2 * std::numbers::pi);
}

// The solid angle of a rectangle q + a·u + b·v (u ⊥ v) seen from x, by inclusion-exclusion of the four rectangles from
// the foot of the perpendicular: F(X, Y) = atan(XY / (d·√(X² + Y² + d²))), d the distance to its plane
double rectangle_solid_angle(const dvec& x, const dvec& q, const dvec& u, const dvec& v)
{
    const double lu = length(u);
    const double lv = length(v);
    const dvec eu = (1 / lu) * u;
    const dvec ev = (1 / lv) * v;
    const dvec w = q - x;
    const double d = std::abs(gpu::dot(w, cross(eu, ev)));
    const double x0 = gpu::dot(w, eu);
    const double y0 = gpu::dot(w, ev);
    const auto f = [d](double X, double Y) { return std::atan(X * Y / (d * std::sqrt(X * X + Y * Y + d * d))); };
    return f(x0 + lu, y0 + lv) - f(x0, y0 + lv) - f(x0 + lu, y0) + f(x0, y0);
}

// A sphere's solid angle from outside: 2π(1 − √(1 − r²/D²))
double sphere_solid_angle(const dvec& x, const dvec& centre, double radius)
{
    const dvec c = centre - x;
    return 2 * std::numbers::pi * (1 - std::sqrt(1 - radius * radius / gpu::dot(c, c)));
}

// A white lambertian point under a sphere of radiance 1 wholly above its horizon: Ω⊥/π = sin²θmax·cos β
double sphere_reflected(const dvec& x, const dvec& n, const dvec& centre, double radius)
{
    const dvec c = centre - x;
    const double d2 = gpu::dot(c, c);
    return radius * radius / d2 * gpu::dot(n, c) / std::sqrt(d2);
}

// --- Scenes and lamps ------------------------------------------------------------------------------------------------

int add_material(gpu::scene_desc& s, gpu::material_kind kind, const gpu::vec3d& rgb)
{
    s.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = rgb});
    s.materials.push_back({.kind = kind, .texture = static_cast<int>(s.textures.size()) - 1});
    return static_cast<int>(s.materials.size()) - 1;
}

// The books' lamps (D.md §5.5): light's quad and sphere, L = 4; cornell's quad, L = 15
gpu::quad_desc light_quad() { return {.q = {3, 1, -2}, .u = {2, 0, 0}, .v = {0, 2, 0}}; }
gpu::sphere_desc light_sphere() { return {.center = {0, 7, 0}, .radius = 2}; }
gpu::quad_desc cornell_quad() { return {.q = {343, 554, 332}, .u = {-130, 0, 0}, .v = {0, 0, -105}}; }

// The setups of T1 and T3: the lamps alone, so that nothing but a lamp stands between a point and a lamp
gpu::scene_desc light_setup()
{
    gpu::scene_desc s;
    const int lamp = add_material(s, gpu::material_kind::diffuse_light, {4, 4, 4});
    gpu::quad_desc q = light_quad();
    q.material = lamp;
    gpu::sphere_desc ball = light_sphere();
    ball.material = lamp;
    s.quads.push_back(q);
    s.spheres.push_back(ball);
    return s;
}

gpu::scene_desc cornell_setup()
{
    gpu::scene_desc s;
    const int lamp = add_material(s, gpu::material_kind::diffuse_light, {15, 15, 15});
    gpu::quad_desc q = cornell_quad();
    q.material = lamp;
    s.quads.push_back(q);
    return s;
}

// D.md §3.1: a lamp of the set is a quad or a still sphere whose material is diffuse_light with a solid texture
bool solid_lamp(const gpu::scene_desc& s, int material, gpu::vec3d& rgb)
{
    if (material < 0 || material >= std::ssize(s.materials))
    {
        return false;
    }
    const gpu::material_desc& m = s.materials[static_cast<std::size_t>(material)];
    if (m.kind != gpu::material_kind::diffuse_light || m.texture < 0 || m.texture >= std::ssize(s.textures))
    {
        return false;
    }
    const gpu::texture_desc& t = s.textures[static_cast<std::size_t>(m.texture)];
    rgb = t.rgb;
    return t.kind == gpu::texture_kind::solid;
}

// D.md §3.2's fields, in double rounded to float
gpu::emitter quad_emitter(const gpu::quad_desc& q, int id, const gpu::vec3d& rgb)
{
    gpu::emitter e{};
    e.corner = gpu::to_real<float>(q.q);
    const gpu::vec3d n = gpu::cross(q.u, q.v);
    e.size = static_cast<float>(std::sqrt(gpu::dot(n, n)));
    e.u = gpu::to_real<float>(q.u);
    e.kind = static_cast<int>(gpu::light_shape::quad);
    e.v = gpu::to_real<float>(q.v);
    e.primitive = id;
    e.normal = gpu::make_quad<float>(q).normal;
    e.unused0 = 0;
    e.radiance = gpu::to_real<float>(rgb);
    e.unused1 = 0;
    return e;
}

gpu::emitter sphere_emitter(const gpu::sphere_desc& s, int id, const gpu::vec3d& rgb)
{
    gpu::emitter e{};
    e.corner = gpu::to_real<float>(s.center);
    e.size = static_cast<float>(s.radius);
    e.u = {};
    e.kind = static_cast<int>(gpu::light_shape::sphere);
    e.v = {};
    e.primitive = id;
    e.normal = {};
    e.unused0 = 0;
    e.radiance = gpu::to_real<float>(rgb);
    e.unused1 = 0;
    return e;
}

// The light set as D.md §3.1-§3.2 define it, made by the test: what build_lamps must give (in any order)
gpu::lamp_table reference_lamps(const gpu::scene_desc& s, std::span<const int> sphere_ids,
                                std::span<const int> quad_ids, int surfaces)
{
    gpu::lamp_table t;
    t.lamp_of.assign(static_cast<std::size_t>(surfaces), -1);
    for (std::size_t k = 0; k < s.spheres.size(); ++k)
    {
        const gpu::sphere_desc& ball = s.spheres[k];
        const bool still = ball.displacement.x == 0 && ball.displacement.y == 0 && ball.displacement.z == 0;
        gpu::vec3d rgb;
        if (still && solid_lamp(s, ball.material, rgb))
        {
            t.lamp_of[static_cast<std::size_t>(sphere_ids[k])] = static_cast<int>(t.lamps.size());
            t.lamps.push_back(sphere_emitter(ball, sphere_ids[k], rgb));
        }
    }
    for (std::size_t k = 0; k < s.quads.size(); ++k)
    {
        gpu::vec3d rgb;
        if (solid_lamp(s, s.quads[k].material, rgb))
        {
            t.lamp_of[static_cast<std::size_t>(quad_ids[k])] = static_cast<int>(t.lamps.size());
            t.lamps.push_back(quad_emitter(s.quads[k], quad_ids[k], rgb));
        }
    }
    return t;
}

// A scene's surfaces on the GPU and its lamps, as the quality_view of a test kernel (shared.md 3.11): the lamps
// build_lamps makes, or the test's own (by_hand)
class lamp_scene
{
public:
    lamp_scene(const gpu::scene_desc& scene, bool by_hand, int max_leaf_size = 1)
        : device_(scene, max_leaf_size),
          table_(by_hand ? reference_lamps(scene, device_.sphere_ids(), device_.quad_ids(), device_.surface_count())
                         : gpu::build_lamps(scene, device_.sphere_ids(), device_.quad_ids())),
          lamps_(table_.lamps),
          lamp_of_(table_.lamp_of)
    {
    }

    gpu::quality_view view(light_combination c) const
    {
        const int count = static_cast<int>(table_.lamps.size());
        gpu::quality_view v{};
        v.base = device_.view();
        v.microfacets = nullptr;
        v.lamps = {.lamps = lamps_.get(), .lamp_of = lamp_of_.get(), .count = count};
        v.environment = {};
        v.light_pmf = count > 0 ? static_cast<float>(1.0 / count) : 0.0f; // §3.3: float(1/N), divided in double
        v.combination = c;
        return v;
    }
    const gpu::lamp_table& table() const { return table_; }

private:
    gpu_test::device_scene device_;
    gpu::lamp_table table_;
    gpu_test::device_buffer<gpu::emitter> lamps_;
    gpu_test::device_buffer<int> lamp_of_;
};

// --- Device helpers --------------------------------------------------------------------------------------------------

__device__ inline bool finite(const vec3<float>& v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z); }
__device__ inline bool same_bits(float a, float b) { return __float_as_uint(a) == __float_as_uint(b); }
__device__ inline bool same_bits(const vec3<float>& a, const vec3<float>& b)
{
    return same_bits(a.x, b.x) && same_bits(a.y, b.y) && same_bits(a.z, b.z);
}
__device__ inline bool positive(const vec3<float>& c) { return c.x > 0.0f || c.y > 0.0f || c.z > 0.0f; }

// One number per statement: their order is the statements'
__device__ inline float uniform_in(gpu::pcg32& rng, float lo, float hi)
{
    return lo + (hi - lo) * gpu::random_real<float>(rng);
}

__device__ inline vec3<float> point_in_box(gpu::pcg32& rng, const vec3<float>& lo, const vec3<float>& hi)
{
    const float x = uniform_in(rng, lo.x, hi.x);
    const float y = uniform_in(rng, lo.y, hi.y);
    const float z = uniform_in(rng, lo.z, hi.z);
    return {x, y, z};
}

__device__ inline vec3<float> point_in_ball(gpu::pcg32& rng, const vec3<float>& centre, float radius)
{
    const vec3<float> d = gpu::random_unit_vector<float>(rng);
    const float r = radius * cbrtf(gpu::random_real<float>(rng));
    return centre + r * d;
}

// D.md §6's point sets. 0, P_light: [−10, 10] × [0, 10] × [−10, 10] outside D² ≤ 4.04 of the sphere lamp (the points
// behind the quad included); 1, P_cornell: [0, 555]³ (the points above the lamp's plane, its back, included); 2: inside
// the smoke MS of the media suite (T3p)
__device__ inline vec3<float> setup_point(int where, gpu::pcg32& rng)
{
    if (where == 1)
    {
        return point_in_box(rng, {0.0f, 0.0f, 0.0f}, {555.0f, 555.0f, 555.0f});
    }
    if (where == 2)
    {
        return point_in_ball(rng, {0.0f, 0.0f, 0.0f}, 1.9f);
    }
    for (;;) // rejects less than 1% of the box
    {
        const vec3<float> x = point_in_box(rng, {-10.0f, 0.0f, -10.0f}, {10.0f, 10.0f, 10.0f});
        if ((x - vec3<float>{0.0f, 7.0f, 0.0f}).length_squared() > 4.04f)
        {
            return x;
        }
    }
}

__device__ inline vec3<float> lamp_centre(const gpu::emitter& e)
{
    return e.kind == static_cast<int>(gpu::light_shape::quad) ? e.corner + 0.5f * e.u + 0.5f * e.v : e.corner;
}

// Primitive g of the tree as closest_hit sees it: a moving sphere where it is at the ray's time
__device__ inline gpu::sphere<float> sphere_at(const gpu::bvh_view<float>& tree, int g, float time)
{
    return g < tree.moving_base ? tree.spheres[g] : tree.moving_spheres[g - tree.moving_base].at(time);
}

// The ray against primitive g alone, over (0.001, ∞)
__device__ inline bool hit_primitive(const gpu::bvh_view<float>& tree, int g, const gpu::ray<float>& r, float& t,
                                     bool& front)
{
    if (g >= tree.quad_base)
    {
        return gpu::hit_quad(tree.quads[g - tree.quad_base], r, 0.001f, infinite, t, front);
    }
    return gpu::hit_sphere(sphere_at(tree, g, r.time), r, 1.0f / r.direction.length_squared(), 0.001f, infinite,
                           false, t, front);
}

// The index in lamps of the lamp whose front the hit is, or -1 (no hit, a particle, a back, no lamp of the set)
__device__ inline int front_lamp(const gpu::quality_view& v, const gpu::bvh_hit<float>& hit)
{
    if (hit.index < 0 || hit.index >= v.base.media.base || !hit.entering)
    {
        return -1;
    }
    const gpu::material<float>& m = v.base.materials[v.base.primitive_materials[hit.index]];
    return m.kind == gpu::material_kind::diffuse_light ? v.lamps.lamp_of[hit.index] : -1;
}

// =====================================================================================================================
// Suite lamps (Step 76)
// =====================================================================================================================

// --- The references reproduce D.md §5's numbers ----------------------------------------------------------------------

GPU_TEST(lamps, references)
{
    // Lambert's formula under the unit square at height 1 over a corner, against Howell's B-3 and the spec's value
    const gpu::quad_desc square{.q = {0, 0, 1}, .u = {0, 1, 0}, .v = {1, 0, 0}};
    const dvec origin{0, 0, 0};
    const dvec up{0, 0, 1};
    const double omega_square = projected_solid_angle(origin, up, corners_of(square));
    ctx.exact("reference-lambert-square", omega_square, 0.4352099, 1e-6);
    ctx.exact("reference-howell-b3", omega_square / std::numbers::pi, howell_b3(1, 1), 1e-12);
    ctx.exact("reference-howell-b3-value", howell_b3(1, 1), 0.1385316, 1e-6);
    // Lambert's formula against a quadrature, tilted and off-centre (T4's configuration b)
    const gpu::quad_desc tilted{.q = {-0.5, -0.4, 1.2}, .u = {0, 1, 0}, .v = {1.2, 0, 0}};
    const dvec x_b{0.3, -0.2, 0};
    const dvec n_b = unit(dvec{0.2, 0.1, 1});
    ctx.exact("reference-lambert-quadrature", projected_solid_angle(x_b, n_b, corners_of(tilted)),
              projected_solid_angle_by_quadrature(x_b, n_b, tilted), 1e-9);
    // The solid angles of §5.5
    const auto omega_of = [](const gpu::quad_desc& q, const dvec& x) {
        return rectangle_solid_angle(x, to_double(q.q), to_double(q.u), to_double(q.v));
    };
    ctx.exact("reference-omega-light-quad", omega_of(light_quad(), {4, 2, 3}), 0.1538841, 1e-6);
    // 2π(1 − √0.92) = 0.2565657 (D.md §5 said 0.2565671: corrected at Step 76's integration)
    ctx.exact("reference-omega-light-sphere", sphere_solid_angle({4, 2, 3}, {0, 7, 0}, 2), 0.2565657, 1e-6);
    ctx.exact("reference-omega-light-sphere-near", sphere_solid_angle({0, 7, 2.5}, {0, 7, 0}, 2), 2.5132741, 1e-6);
    ctx.exact("reference-omega-cornell", omega_of(cornell_quad(), {278, 100, 278}), 0.065124, 1e-6);
    // T4's expected values (§6.1)
    const double l_a = 4 * omega_square / std::numbers::pi;
    ctx.exact("reference-t4-a", l_a, 0.5541264, 1e-6);
    ctx.exact("reference-t4-c", 4 * sphere_reflected(origin, up, {0.5, 0.3, 3}, 0.8), 0.269055, 1e-6);
    ctx.exact("reference-t4-d", l_a + 4 * sphere_reflected(origin, up, {-1.5, -1.2, 3}, 0.8), 0.724017, 1e-6);
}

// --- L0: the light set -----------------------------------------------------------------------------------------------

// How many of an emitter's ten fields differ from the expected one's, bit for bit
int field_mismatches(const gpu::emitter& a, const gpu::emitter& b)
{
    const auto same = [](const auto& x, const auto& y) { return std::memcmp(&x, &y, sizeof x) == 0; };
    return (same(a.corner, b.corner) ? 0 : 1) + (same(a.size, b.size) ? 0 : 1) + (same(a.u, b.u) ? 0 : 1) +
           (same(a.kind, b.kind) ? 0 : 1) + (same(a.v, b.v) ? 0 : 1) + (same(a.primitive, b.primitive) ? 0 : 1) +
           (same(a.normal, b.normal) ? 0 : 1) + (same(a.unused0, b.unused0) ? 0 : 1) +
           (same(a.radiance, b.radiance) ? 0 : 1) + (same(a.unused1, b.unused1) ? 0 : 1);
}

GPU_TEST(lamps, membership)
{
    // Spheres: a still solid lamp, a moving one, a matte one, a checkered lamp, and glass the lights list names. Quads:
    // a tilted solid lamp, a matte floor, and a solid lamp that emits nothing (still a lamp of the set: its texture is
    // solid).
    gpu::scene_desc s;
    const int lamp = add_material(s, gpu::material_kind::diffuse_light, {2, 3, 4});
    const int matte = add_material(s, gpu::material_kind::lambertian, {0.5, 0.5, 0.5});
    s.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = {1, 0, 0}});
    s.textures.push_back({.kind = gpu::texture_kind::checker, .inv_scale = 2, .even = 1, .odd = 2});
    s.materials.push_back({.kind = gpu::material_kind::diffuse_light, .texture = 3});
    const int checkered = static_cast<int>(s.materials.size()) - 1;
    s.materials.push_back({.kind = gpu::material_kind::dielectric, .refraction_index = 1.5});
    const int glass = static_cast<int>(s.materials.size()) - 1;
    const int black = add_material(s, gpu::material_kind::diffuse_light, {0, 0, 0});
    s.spheres.push_back({.center = {0, 1, 0}, .radius = 0.5, .material = lamp});
    s.spheres.push_back({.center = {2, 1, 0}, .radius = 0.5, .material = lamp, .displacement = {1, 0, 0}});
    s.spheres.push_back({.center = {-2, 1, 0}, .radius = 0.5, .material = matte});
    s.spheres.push_back({.center = {0, 3, 0}, .radius = 0.7, .material = checkered});
    s.spheres.push_back({.center = {0, -1, 2}, .radius = 1, .material = glass});
    s.quads.push_back({.q = {1, 2, 3}, .u = {1, 0.5, 0}, .v = {0, 0.3, 1.2}, .material = lamp});
    s.quads.push_back({.q = {-4, 0, -4}, .u = {8, 0, 0}, .v = {0, 0, 8}, .material = matte});
    s.quads.push_back({.q = {-3, 0, -1}, .u = {0, 0, 2}, .v = {1.5, 0, 0}, .material = black});
    s.lights.push_back({.kind = gpu::light_shape::sphere, .center = {0, -1, 2}, .radius = 1, .weight = 1});

    const gpu::bvh<float> tree = gpu::build_bvh<float>(s.spheres, s.quads, 1);
    const int surfaces = static_cast<int>(tree.primitive_materials.size());
    const gpu::lamp_table built = gpu::build_lamps(s, tree.sphere_ids, tree.quad_ids);
    const gpu::lamp_table expected = reference_lamps(s, tree.sphere_ids, tree.quad_ids, surfaces);
    ctx.exact("membership-lamps", static_cast<double>(built.lamps.size()), 3, 0);
    ctx.exact("membership-count-lamps", gpu::count_lamps(s), 3, 0);
    ctx.exact("membership-lamp-of-size", static_cast<double>(built.lamp_of.size()), surfaces, 0);

    // Every field, keyed by primitive (the order of the lamps is not specified)
    std::int64_t mismatches = 0;
    for (const gpu::emitter& e : expected.lamps)
    {
        const auto copies = std::ranges::count(built.lamps, e.primitive, &gpu::emitter::primitive);
        const auto found = std::ranges::find(built.lamps, e.primitive, &gpu::emitter::primitive);
        mismatches += copies == 1 ? field_mismatches(*found, e) : 10;
    }
    ctx.count("membership-fields", mismatches, 0, 10 * static_cast<std::int64_t>(expected.lamps.size()));

    // lamp_of: the index of the lamp with that primitive, -1 for every other surface
    std::int64_t wrong = 0;
    for (int g = 0; g < surfaces; ++g)
    {
        const auto i = static_cast<std::size_t>(g);
        const int k = i < built.lamp_of.size() ? built.lamp_of[i] : -2;
        const bool names_it = k >= 0 && k < std::ssize(built.lamps) &&
                              built.lamps[static_cast<std::size_t>(k)].primitive == g;
        wrong += (expected.lamp_of[i] >= 0 ? names_it : k == -1) ? 0 : 1;
    }
    ctx.count("membership-lamp-of", wrong, 0, surfaces);
}

// --- W1: the weights at the extremes ---------------------------------------------------------------------------------

struct weight_row
{
    float power;
    float light[3]; // by combination: bsdf_only, lights_only, power_heuristic
    float bsdf[3];
};

// Pair i: a = values[i % n], b = values[i / n]
struct weights_of
{
    const float* values;
    int n;
    __device__ weight_row operator()(int i) const
    {
        const float a = values[i % n];
        const float b = values[i / n];
        const light_combination c[3] = {light_combination::bsdf_only, light_combination::lights_only,
                                        light_combination::power_heuristic};
        weight_row row{};
        row.power = gpu::power_weight(a, b);
        for (int k = 0; k < 3; ++k)
        {
            row.light[k] = gpu::light_weight(c[k], a, b);
            row.bsdf[k] = gpu::bsdf_weight(c[k], a, b);
        }
        return row;
    }
};

GPU_TEST(lamps, weights)
{
    // −1, 0, 10^k for k = −30, −27, ..., 30, +∞ and NaN: 25 values, 625 pairs
    std::vector<float> values{-1.0f, 0.0f};
    for (int k = -30; k <= 30; k += 3)
    {
        values.push_back(static_cast<float>(std::pow(10.0, k)));
    }
    values.push_back(infinite);
    values.push_back(std::numeric_limits<float>::quiet_NaN());
    const int n = static_cast<int>(values.size());
    const gpu_test::device_buffer<float> device_values(values);
    const std::vector<weight_row> rows = device_map<weight_row>(weights_of{device_values.get(), n}, n * n);

    std::int64_t checked = 0;
    std::int64_t violations = 0;
    std::string first;
    const auto check = [&](bool ok, const char* rule, float a, float b) {
        ++checked;
        if (!ok)
        {
            ++violations;
            if (first.empty())
            {
                first = std::string(rule) + " at (" + number(a) + ", " + number(b) + ")";
            }
        }
    };
    const auto in_unit = [](float w) { return std::isfinite(w) && w >= 0 && w <= 1; };
    constexpr float ulps = 2.4e-7f; // 2 ulp at 1: the same formula, fused differently
    for (int i = 0; i < n * n; ++i)
    {
        const float a = values[static_cast<std::size_t>(i % n)];
        const float b = values[static_cast<std::size_t>(i / n)];
        const weight_row& w = rows[static_cast<std::size_t>(i)];
        const float reverse = rows[static_cast<std::size_t>((i % n) * n + i / n)].power; // P(b, a)
        const bool a_positive = a > 0 && std::isfinite(a);
        const bool b_positive = b > 0 && std::isfinite(b);
        // P's limits (§2.2): computed as the ratio b/a, no square of a density
        check(in_unit(w.power), "P in [0, 1]", a, b);
        if (!(a > 0))
        {
            check(w.power == 0, "P(a, b) = 0 for a not > 0", a, b);
        }
        if (a_positive && b == 0)
        {
            check(w.power == 1, "P(a, 0) = 1", a, b);
        }
        if (a_positive && b == infinite)
        {
            check(w.power == 0, "P(a, inf) = 0", a, b);
        }
        if (a_positive && a == b)
        {
            check(w.power == 0.5f, "P(a, a) = 1/2", a, b);
        }
        if (a_positive && b_positive)
        {
            check(std::abs(static_cast<double>(w.power) + reverse - 1) <= 1e-6, "P(a, b) + P(b, a) = 1", a, b);
        }
        // The table of §2.2, for the three combinations
        const bool defined = !std::isnan(b) && !(a == infinite && b == infinite);
        for (int k = 0; k < 3; ++k)
        {
            check(in_unit(w.light[k]) && in_unit(w.bsdf[k]), "weights in [0, 1]", a, b);
            if (combinations[k] == light_combination::power_heuristic)
            {
                if (defined)
                {
                    check(std::abs(w.light[k] - w.power) <= ulps, "light_weight(mis) = P(p_L, p_B)", a, b);
                }
            }
            else
            {
                check(w.light[k] == 1, "light_weight = 1 for material and nee", a, b);
            }
            // bsdf_weight(c, p_B = a, p_L = b)
            if (!(a > 0) || !(b > 0))
            {
                check(w.bsdf[k] == 1, "bsdf_weight = 1 where a density is not > 0", a, b);
            }
            else if (combinations[k] == light_combination::power_heuristic)
            {
                if (defined)
                {
                    check(std::abs(w.bsdf[k] - w.power) <= ulps, "bsdf_weight(mis) = P(p_B, p_L)", a, b);
                }
            }
            else
            {
                check(w.bsdf[k] == (combinations[k] == light_combination::lights_only ? 0.0f : 1.0f),
                      "bsdf_weight: 0 for nee, 1 for material", a, b);
            }
        }
    }
    ctx.count("weights", violations, 0, checked);
    if (!first.empty())
    {
        ctx.info("weights-first-violation", first);
    }
}

// --- T1: a light sample is where it says -----------------------------------------------------------------------------

enum t1_tally : int
{
    t1_sampled, // sample_light returned true
    t1_wrong,   // a violation
    t1_edge,    // a miss of its lamp, a back hit, or a distance off by more than 10^-4
    t1_blocked, // the other lamp in front of the sampled point: checked against its own lamp alone
    t1_count,
};

struct t1_sample
{
    gpu::quality_view v;
    int where;
    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally) const
    {
        const vec3<float> x = setup_point(where, rng);
        gpu::light_sample ls;
        if (!gpu::sample_light(v, x, rng, ls))
        {
            return; // T1b checks where that may happen
        }
        ++tally[t1_sampled];
        // The sample's own numbers, and the lamp it names: a lamp of the set, with that lamp's one color
        const bool numbers = isfinite(ls.pdf) && ls.pdf > 0.0f && isfinite(ls.distance) && ls.distance > 0.0f &&
                             finite(ls.wi) && fabsf(sqrtf(ls.wi.length_squared()) - 1.0f) <= 1e-5f;
        const int k = ls.skip >= 0 && ls.skip < v.base.media.base ? v.lamps.lamp_of[ls.skip] : -1;
        const bool lamp = k >= 0 && k < v.lamps.count && v.lamps.lamps[k].primitive == ls.skip &&
                          same_bits(ls.le, v.base.materials[v.base.primitive_materials[ls.skip]].albedo);
        if (!numbers || !lamp)
        {
            ++tally[t1_wrong];
            return;
        }
        // The ray along it meets that lamp's front at the sample's distance
        const gpu::ray<float> r{x, ls.wi};
        const gpu::bvh_tracer tracer{v};
        const gpu::bvh_hit<float> hit = tracer.closest(r, gpu::spawn{-1, false}, rng);
        float t = hit.t;
        bool front = hit.entering;
        if (hit.index != ls.skip)
        {
            // Another lamp before the sampled point: the sample is checked against its own lamp alone
            const bool blocked = hit.index >= 0 && hit.t < (1.0f - 1e-4f) * ls.distance;
            if (!blocked || !hit_primitive(v.base.tree, ls.skip, r, t, front))
            {
                ++tally[t1_edge];
                return;
            }
            ++tally[t1_blocked];
        }
        if (!front || !(fabsf(t - ls.distance) <= 1e-4f * ls.distance))
        {
            ++tally[t1_edge];
            return;
        }
        // And the light technique's density of that hit, as the integrator asks for it, is the sample's
        const float p = gpu::lamp_hit_pdf(v, ls.skip, x, ls.wi, t * sqrtf(ls.wi.length_squared()));
        if (!(fabsf(p - ls.pdf) < 1e-4f * ls.pdf))
        {
            ++tally[t1_wrong];
        }
    }
};

// T1b: false only where §3.4 and §3.5 say, the pick §3.3 says, and 3 numbers drawn
struct t1_false_sample
{
    gpu::quality_view v;
    int where;
    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally) const
    {
        ++tally[0];
        if (v.lamps.count <= 0)
        {
            ++tally[1];
            return;
        }
        const vec3<float> x = where == 0 && gpu::random_real<float>(rng) < 0.1f
                                  ? point_in_ball(rng, {0.0f, 7.0f, 0.0f}, 1.98f) // inside the sphere lamp
                                  : setup_point(where, rng);
        // The lamp the first number picks: k = min(⌊ξN⌋, N − 1)
        gpu::pcg32 copy = rng;
        const int n = v.lamps.count;
        const int k = ::min(static_cast<int>(gpu::random_real<float>(copy) * static_cast<float>(n)), n - 1);
        const gpu::emitter& e = v.lamps.lamps[k];
        bool must_fail = false;
        bool must_pass = false;
        if (e.kind == static_cast<int>(gpu::light_shape::quad))
        {
            // Behind the plane by more than 10^-3: the back, always false; in front by more: never
            const double s = (double(x.x) - e.corner.x) * e.normal.x + (double(x.y) - e.corner.y) * e.normal.y +
                             (double(x.z) - e.corner.z) * e.normal.z;
            must_fail = s < -1e-3;
            must_pass = s > 1e-3;
        }
        else
        {
            const double dx = double(e.corner.x) - x.x;
            const double dy = double(e.corner.y) - x.y;
            const double dz = double(e.corner.z) - x.z;
            const double d2 = dx * dx + dy * dy + dz * dz;
            const double r2 = double(e.size) * e.size;
            must_pass = d2 > 1.0001 * r2;
            must_fail = d2 < 0.9999 * r2;
        }
        gpu::light_sample ls;
        const bool sampled = gpu::sample_light(v, x, rng, ls);
        if ((sampled && must_fail) || (!sampled && must_pass))
        {
            ++tally[1];
        }
        if (sampled && ls.skip != e.primitive)
        {
            ++tally[2];
        }
        // 1 number for the pick, 2 for the lamp, in every call (§3.3, §3.6)
        copy.next();
        copy.next();
        const bool same_state = rng.next() == copy.next() && rng.next() == copy.next();
        if (!same_state)
        {
            ++tally[3];
        }
    }
};

GPU_TEST(lamps, t1)
{
    const std::int64_t n = ctx.samples(full_samples);
    const std::int64_t drawn = gpu_test::drawn(n);
    const char* names[] = {"light", "cornell"};
    for (int where = 0; where < 2; ++where)
    {
        const lamp_scene lamps(where == 0 ? light_setup() : cornell_setup(), false);
        const std::string name(names[where]);
        ctx.exact("t1-" + name + "-lamps", static_cast<double>(lamps.table().lamps.size()), where == 0 ? 2 : 1, 0);
        const auto t = device_tally<t1_count>(t1_sample{lamps.view(light_combination::power_heuristic), where},
                                              7611 + where, n);
        ctx.count("t1-" + name, t[t1_wrong], 0, t[t1_sampled]);
        ctx.count("t1-" + name + "-edges", t[t1_edge], (drawn + 9999) / 10000, t[t1_sampled]); // 10^-4, rounded up
        ctx.info("t1-" + name + "-blocked", std::to_string(t[t1_blocked]) + " of " + std::to_string(t[t1_sampled]) +
                                                " samples behind the other lamp, checked against their own");
        const auto f = device_tally<4>(t1_false_sample{lamps.view(light_combination::power_heuristic), where},
                                       7613 + where, n);
        ctx.count("t1-false-" + name, f[1], 0, f[0]);
        ctx.count("t1-pick-" + name, f[2], 0, f[0]);
        ctx.count("t1-draws-" + name, f[3], 0, f[0]);
    }
}

// --- T2: the lamps' densities against closed forms -------------------------------------------------------------------

// A lamp seen from a point: its emitter (by hand, §3.2) and its primitive for the test's own hit test
struct lamp_geometry
{
    gpu::emitter e;
    gpu::quad<float> face;   // a quad's
    gpu::sphere<float> ball; // a sphere's
    bool quad;
    vec3<float> x;
};

struct lamp_case
{
    const char* name;
    lamp_geometry g;
    double omega; // its solid angle from x, in double from the float numbers
};

lamp_case quad_case(const char* name, const gpu::quad_desc& q, const gpu::vec3d& x)
{
    lamp_case c{.name = name};
    c.g.e = quad_emitter(q, 0, {1, 1, 1});
    c.g.face = gpu::make_quad<float>(q);
    c.g.quad = true;
    c.g.x = gpu::to_real<float>(x);
    c.omega = rectangle_solid_angle(to_double(c.g.x), to_double(c.g.e.corner), to_double(c.g.e.u),
                                    to_double(c.g.e.v));
    return c;
}

lamp_case sphere_case(const char* name, const gpu::sphere_desc& s, const gpu::vec3d& x)
{
    lamp_case c{.name = name};
    c.g.e = sphere_emitter(s, 0, {1, 1, 1});
    c.g.ball = {c.g.e.corner, c.g.e.size};
    c.g.quad = false;
    c.g.x = gpu::to_real<float>(x);
    c.omega = sphere_solid_angle(to_double(c.g.x), to_double(c.g.e.corner), c.g.e.size);
    return c;
}

// 4π·p(ω)·[ω meets the lamp's front] over uniform ω: 1 for a density of the lamp's directions
struct t2_uniform
{
    lamp_geometry g;
    __device__ double operator()(gpu::pcg32& rng) const
    {
        const vec3<float> w = gpu::random_unit_vector<float>(rng);
        const gpu::ray<float> r{g.x, w};
        float t;
        bool front;
        const bool hit = g.quad ? gpu::hit_quad(g.face, r, 0.001f, infinite, t, front)
                                : gpu::hit_sphere(g.ball, r, 1.0f / w.length_squared(), 0.001f, infinite, false, t,
                                                  front);
        if (!hit || !front)
        {
            return 0.0;
        }
        return 4 * std::numbers::pi * gpu::lamp_pdf(g.e, g.x, w, t * sqrtf(w.length_squared()));
    }
};

// 1/p over the lamp's own samples: its solid angle Ω; NaN (left out) for a false return
struct t2_inverse_density
{
    lamp_geometry g;
    __device__ double operator()(gpu::pcg32& rng) const
    {
        gpu::lamp_sample s;
        if (!gpu::sample_lamp(g.e, g.x, rng, s) || !(s.pdf > 0.0f))
        {
            return nan("");
        }
        return 1.0 / s.pdf;
    }
};

// A sphere's density is uniform in its cone: every sample's 1/p is Ω, to 10^-5
struct t2_sphere_off
{
    lamp_geometry g;
    double omega;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        gpu::lamp_sample s;
        if (!gpu::sample_lamp(g.e, g.x, rng, s))
        {
            return true;
        }
        return !(::fabs(1.0 / s.pdf - omega) <= 1e-5 * omega);
    }
};

// T2c: 1/pdf over sample_light's draws (false: 0) = Σ Ω_k over the set
struct t2_set
{
    gpu::quality_view v;
    vec3<float> x;
    __device__ double operator()(gpu::pcg32& rng) const
    {
        gpu::light_sample ls;
        return gpu::sample_light(v, x, rng, ls) ? 1.0 / ls.pdf : 0.0;
    }
};

GPU_TEST(lamps, t2)
{
    const std::int64_t n = ctx.samples(full_samples);
    const std::int64_t drawn = gpu_test::drawn(n);
    const lamp_case cases[] = {quad_case("light-quad", light_quad(), {4, 2, 3}),
                               sphere_case("light-sphere", light_sphere(), {4, 2, 3}),
                               sphere_case("light-sphere-near", light_sphere(), {0, 7, 2.5}),
                               quad_case("cornell", cornell_quad(), {278, 100, 278})};
    std::uint64_t seed = 7621;
    for (const lamp_case& c : cases)
    {
        const std::string name(c.name);
        const gpu_test::mean_estimate u = gpu_test::device_mean(t2_uniform{c.g}, seed++, n);
        ctx.mean("t2-unif-" + name, u.mean, u.standard_error, 1);
        ctx.count("t2-unif-" + name + "-nan", u.left_out, 0, drawn);
        if (c.g.quad)
        {
            const gpu_test::mean_estimate inverse = gpu_test::device_mean(t2_inverse_density{c.g}, seed++, n);
            ctx.mean("t2-omega-" + name, inverse.mean, inverse.standard_error, c.omega);
            ctx.count("t2-omega-" + name + "-left-out", inverse.left_out, 0, drawn);
        }
        else
        {
            ctx.count("t2-omega-" + name, gpu_test::device_count(t2_sphere_off{c.g, c.omega}, seed++, n), 0, drawn);
        }
    }
    const lamp_scene light(light_setup(), true);
    const gpu_test::mean_estimate set = gpu_test::device_mean(
        t2_set{light.view(light_combination::power_heuristic), {4.0f, 2.0f, 3.0f}}, 7629, n);
    ctx.mean("t2-set", set.mean, set.standard_error, cases[0].omega + cases[1].omega);
}

// --- T3: Veach's condition on the weights ----------------------------------------------------------------------------

enum t3_tally : int
{
    t3_tested, // directions W1 applies to
    t3_wrong,  // |w_L + w_B − 1| too large, a weight outside [0, 1], or a lamp's front without a density
    t3_far,    // |w_L + w_B − 1| > 10^-2 (X1)
    t3_pdf,    // a particle's density not uniform_sphere_density<float> (T3p)
    t3_count,
};

// The lobe at x: a matte surface of a random normal, toward the centre of a lamp of the set for half the points, or a
// particle of smoke
struct lobe_at
{
    gpu::bsdf b;
    gpu::hit_record<float> rec;
    gpu::ray<float> in;
};

__device__ inline lobe_at make_lobe(int kind, const gpu::quality_view& v, const vec3<float>& x, gpu::pcg32& rng)
{
    lobe_at l{};
    gpu::material<float> m{};
    l.rec.p = x;
    l.rec.front_face = true;
    if (kind == 0)
    {
        m.kind = gpu::material_kind::lambertian;
        m.albedo = {1.0f, 1.0f, 1.0f};
        vec3<float> n = gpu::random_unit_vector<float>(rng);
        if (gpu::random_real<float>(rng) < 0.5f && v.lamps.count > 0)
        {
            const int k = ::min(static_cast<int>(gpu::random_real<float>(rng) * static_cast<float>(v.lamps.count)),
                                v.lamps.count - 1);
            if (gpu::dot(n, lamp_centre(v.lamps.lamps[k]) - x) < 0.0f)
            {
                n = -n;
            }
        }
        l.rec.normal = n;
        l.in = {x + n, -n};
    }
    else
    {
        m.kind = gpu::material_kind::isotropic;
        m.albedo = {0.9f, 0.9f, 0.9f};
        l.rec.normal = {1.0f, 0.0f, 0.0f}; // a particle's (surface_at)
        const vec3<float> d = gpu::random_unit_vector<float>(rng);
        l.in = {x - d, d};
    }
    l.b = gpu::make_bsdf(m, nullptr, m.albedo, l.rec, l.in.direction);
    return l;
}

struct t3_sample
{
    gpu::quality_view v;
    int where;        // setup_point's set
    int lobe;         // 0 lambertian, 1 isotropic
    bool light_drawn; // (a) a light sample's direction, (b) the material's own
    bool planted;     // X1: p_L′ = lamp_pdf, the pmf forgotten on the hit side
    bool surfaces;    // trace the surfaces alone (T3p: the particles don't hide the lamps)

    __device__ gpu::bvh_hit<float> trace(const gpu::ray<float>& r, gpu::pcg32& rng) const
    {
        if (surfaces)
        {
            return gpu::closest_hit<float, gpu::kernel_kind::all_books>(v.base.tree, r, -1, false);
        }
        const gpu::bvh_tracer tracer{v};
        return tracer.closest(r, gpu::spawn{-1, false}, rng);
    }

    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally) const
    {
        const vec3<float> x = setup_point(where, rng);
        const lobe_at l = make_lobe(lobe, v, x, rng);
        const float uniform = gpu::uniform_sphere_density<float>;
        const light_combination c = v.combination;
        float w_light = 0;
        float w_bsdf = 0;
        if (light_drawn)
        {
            gpu::light_sample ls;
            if (!gpu::sample_light(v, x, rng, ls))
            {
                return;
            }
            float p_bsdf = 0;
            const vec3<float> f = gpu::eval_bsdf(l.b, ls.wi, p_bsdf);
            if (lobe == 1 && !same_bits(p_bsdf, uniform))
            {
                ++tally[t3_pdf];
            }
            if (!(p_bsdf > 0.0f) || !positive(f))
            {
                return; // f = 0: no light to weigh
            }
            const gpu::bvh_hit<float> hit = trace({x, ls.wi}, rng);
            if (hit.index != ls.skip || !hit.entering)
            {
                return; // the other lamp in front, or an edge: T1's business
            }
            const float distance = hit.t * sqrtf(ls.wi.length_squared());
            const float p_light = planted ? gpu::lamp_pdf(v.lamps.lamps[v.lamps.lamp_of[hit.index]], x, ls.wi, distance)
                                          : gpu::lamp_hit_pdf(v, hit.index, x, ls.wi, distance);
            w_light = gpu::light_weight(c, ls.pdf, p_bsdf);
            w_bsdf = gpu::bsdf_weight(c, p_bsdf, p_light);
        }
        else
        {
            gpu::bsdf_sample bs;
            if (!gpu::sample_bsdf(l.b, l.in, l.rec, rng, bs))
            {
                return;
            }
            float p_eval = 0;
            gpu::eval_bsdf(l.b, bs.wi, p_eval);
            if (lobe == 1 && !(same_bits(p_eval, uniform) && same_bits(bs.pdf, uniform)))
            {
                ++tally[t3_pdf];
            }
            const gpu::bvh_hit<float> hit = trace({x, bs.wi}, rng);
            if (front_lamp(v, hit) < 0)
            {
                return; // nothing a light sample could have drawn
            }
            // The integrator's question at a lamp hit (§2.3, 2.): the previous vertex, the unit direction, the distance
            const float p_light = gpu::lamp_hit_pdf(v, hit.index, x, gpu::unit_vector(bs.wi),
                                                    hit.t * sqrtf(bs.wi.length_squared()));
            if (!(p_light > 0.0f) || !isfinite(p_light))
            {
                ++tally[t3_tested];
                ++tally[t3_wrong]; // a lamp's front must have a density
                return;
            }
            // The sample's own density in w_B, eval_bsdf's in w_L, as the integrator pairs them (§10.7)
            w_bsdf = gpu::bsdf_weight(c, bs.pdf, p_light);
            w_light = gpu::light_weight(c, p_light, p_eval);
        }
        ++tally[t3_tested];
        const float sum = w_light + w_bsdf;
        // lights_only: 1 + 0, exactly; the power heuristic: P(a, b) + P(b, a), to rounding
        if (!(w_light >= 0.0f && w_light <= 1.0f && w_bsdf >= 0.0f && w_bsdf <= 1.0f) ||
            !(c == light_combination::lights_only ? sum == 1.0f : fabsf(sum - 1.0f) < 1e-5f))
        {
            ++tally[t3_wrong];
        }
        if (!(fabsf(sum - 1.0f) <= 1e-2f))
        {
            ++tally[t3_far];
        }
    }
};

// T3 and T3p: (a) and (b), each over n samples, for nee and mis; returns the tallies by combination
std::array<std::array<std::int64_t, t3_count>, 2> test_weights(const lamp_scene& lamps, int where, int lobe,
                                                               bool surfaces, std::uint64_t seed, std::int64_t n)
{
    std::array<std::array<std::int64_t, t3_count>, 2> total{};
    for (int k = 0; k < 2; ++k)
    {
        const light_combination c = k == 0 ? light_combination::lights_only : light_combination::power_heuristic;
        for (const bool light_drawn : {true, false})
        {
            add(total[static_cast<std::size_t>(k)],
                device_tally<t3_count>(t3_sample{lamps.view(c), where, lobe, light_drawn, false, surfaces}, seed++, n));
        }
    }
    return total;
}

GPU_TEST(lamps, t3)
{
    const std::int64_t n = ctx.samples(full_samples);
    const char* setups[] = {"light", "cornell"};
    const char* lobes[] = {"lambertian", "isotropic"};
    std::uint64_t seed = 7630;
    for (int where = 0; where < 2; ++where)
    {
        const lamp_scene lamps(where == 0 ? light_setup() : cornell_setup(), false);
        for (int lobe = 0; lobe < 2; ++lobe)
        {
            const auto total = test_weights(lamps, where, lobe, false, seed, n);
            seed += 4;
            for (int k = 0; k < 2; ++k)
            {
                const auto& t = total[static_cast<std::size_t>(k)];
                const std::string name = std::string("t3-") + setups[where] + "-" + lobes[lobe] + "-" +
                                         (k == 0 ? "nee" : "mis");
                ctx.count(name, t[t3_wrong], 0, t[t3_tested]);
                ctx.info(name + "-tested", std::to_string(t[t3_tested]) + " directions from both techniques");
            }
        }
    }
    // X1, the planted fault: the light setup's matte points, mis, the hit's density without the pmf
    const lamp_scene light(light_setup(), false);
    const auto planted = device_tally<t3_count>(
        t3_sample{light.view(light_combination::power_heuristic), 0, 0, true, true, false}, 7646, n);
    ctx.count("planted-pmf", planted[t3_far] > 0 ? 0 : 1, 0, 1);
    ctx.info("planted-pmf-far", std::to_string(planted[t3_far]) + " of " + std::to_string(planted[t3_tested]) +
                                    " directions with |w_L + w_B - 1| > 1e-2");
}

// --- T4: the estimator at one vertex ---------------------------------------------------------------------------------

// §6.1: a white lambertian point x (from −1) with normal n under lamps of L = 4, no sky, the BVH holding the lamps
// alone: the integrator's §2.3 restricted to the direct light
struct t4_estimator
{
    gpu::quality_view v;
    vec3<float> x;
    vec3<float> n;
    bool planted; // X2: the lamp a material's ray finds counted in full under nee (counted twice)

    __device__ double operator()(gpu::pcg32& rng) const
    {
        gpu::material<float> m{};
        m.kind = gpu::material_kind::lambertian;
        m.albedo = {1.0f, 1.0f, 1.0f};
        gpu::hit_record<float> rec{};
        rec.p = x;
        rec.normal = n;
        rec.front_face = true;
        const gpu::ray<float> in{x + n, -n};
        const gpu::bsdf b = gpu::make_bsdf(m, nullptr, m.albedo, rec, in.direction);
        const gpu::bvh_tracer tracer{v};
        const light_combination c = v.combination;
        vec3<float> sum{};
        if (c != light_combination::bsdf_only)
        {
            gpu::light_sample ls;
            if (gpu::sample_light(v, x, rng, ls))
            {
                float p_bsdf = 0;
                const vec3<float> f = gpu::eval_bsdf(b, ls.wi, p_bsdf);
                if (positive(f) && ls.pdf > 0.0f)
                {
                    const vec3<float> contribution = (gpu::light_weight(c, ls.pdf, p_bsdf) / ls.pdf) * (f * ls.le);
                    if (finite(contribution) && positive(contribution))
                    {
                        const gpu::ray<float> shadow{x, ls.wi, 0.0f};
                        sum = sum + tracer.visibility(shadow, gpu::spawn{-1, false}, ls.distance, ls.skip) *
                                        contribution;
                    }
                }
            }
        }
        gpu::bsdf_sample bs;
        if (gpu::sample_bsdf(b, in, rec, rng, bs))
        {
            const gpu::ray<float> out{x, bs.wi};
            const gpu::bvh_hit<float> hit = tracer.closest(out, gpu::spawn{-1, false}, rng);
            if (front_lamp(v, hit) >= 0)
            {
                const float p_light = gpu::lamp_hit_pdf(v, hit.index, x, gpu::unit_vector(bs.wi),
                                                        hit.t * sqrtf(bs.wi.length_squared()));
                const float w = planted ? 1.0f
                                        : gpu::bsdf_weight(c, c != light_combination::bsdf_only ? bs.pdf : 0.0f,
                                                           p_light);
                const vec3<float> le = v.base.materials[v.base.primitive_materials[hit.index]].albedo;
                sum = sum + w * (bs.weight * le);
            }
        }
        return (double(sum.x) + sum.y + sum.z) / 3;
    }
};

struct t4_config
{
    const char* name;
    gpu::scene_desc scene;
    vec3<float> x;
    vec3<float> n;
    double expected; // L_o
};

// §6.1's configurations, every lamp of L = 4
std::vector<t4_config> t4_configs()
{
    const gpu::vec3d radiance{4, 4, 4};
    const gpu::quad_desc square{.q = {0, 0, 1}, .u = {0, 1, 0}, .v = {1, 0, 0}}; // normal −z, over x's corner
    std::vector<t4_config> configs;
    const auto quad_lamp = [&](t4_config& c, gpu::quad_desc q) {
        q.material = add_material(c.scene, gpu::material_kind::diffuse_light, radiance);
        c.scene.quads.push_back(q);
        return 4 * projected_solid_angle(to_double(c.x), to_double(c.n), corners_of(q)) / std::numbers::pi;
    };
    const auto sphere_lamp = [&](t4_config& c, const gpu::vec3d& centre, double radius) {
        c.scene.spheres.push_back({.center = centre,
                                   .radius = radius,
                                   .material = add_material(c.scene, gpu::material_kind::diffuse_light, radiance)});
        return 4 * sphere_reflected(to_double(c.x), to_double(c.n), to_double(centre), radius);
    };
    {
        t4_config c{.name = "a", .x = {0.0f, 0.0f, 0.0f}, .n = {0.0f, 0.0f, 1.0f}};
        c.expected = quad_lamp(c, square);
        configs.push_back(c);
    }
    {
        const dvec n = unit(dvec{0.2, 0.1, 1});
        t4_config c{.name = "b",
                    .x = {0.3f, -0.2f, 0.0f},
                    .n = {static_cast<float>(n.x), static_cast<float>(n.y), static_cast<float>(n.z)}};
        c.expected = quad_lamp(c, {.q = {-0.5, -0.4, 1.2}, .u = {0, 1, 0}, .v = {1.2, 0, 0}});
        configs.push_back(c);
    }
    {
        t4_config c{.name = "c", .x = {0.0f, 0.0f, 0.0f}, .n = {0.0f, 0.0f, 1.0f}};
        c.expected = sphere_lamp(c, {0.5, 0.3, 3}, 0.8);
        configs.push_back(c);
    }
    {
        // The square and a sphere whose cone doesn't overlap it: light_pmf ½
        t4_config c{.name = "d", .x = {0.0f, 0.0f, 0.0f}, .n = {0.0f, 0.0f, 1.0f}};
        const double square_light = quad_lamp(c, square);
        c.expected = square_light + sphere_lamp(c, {-1.5, -1.2, 3}, 0.8);
        configs.push_back(c);
    }
    return configs;
}

GPU_TEST(lamps, t4)
{
    const std::int64_t n = ctx.samples(full_samples);
    const std::int64_t drawn = gpu_test::drawn(n);
    const std::vector<t4_config> configs = t4_configs();
    std::uint64_t seed = 7650;
    for (const t4_config& c : configs)
    {
        const lamp_scene lamps(c.scene, true);
        for (const light_combination mode : combinations)
        {
            const gpu_test::mean_estimate m =
                gpu_test::device_mean(t4_estimator{lamps.view(mode), c.x, c.n, false}, seed++, n);
            const std::string name = std::string("t4-") + c.name + "-" + mode_name(mode);
            ctx.mean(name, m.mean, m.standard_error, c.expected);
            ctx.count(name + "-nan", m.left_out, 0, drawn);
        }
    }
    // X2, the planted fault: (a) under nee with the lamps the material's rays find counted too
    const lamp_scene lamps(configs[0].scene, true);
    const gpu_test::mean_estimate m = gpu_test::device_mean(
        t4_estimator{lamps.view(light_combination::lights_only), configs[0].x, configs[0].n, true}, 7662, n);
    const double z = (m.mean - configs[0].expected) / m.standard_error;
    ctx.must_reject("planted-double", z, 0, gpu_test::normal_two_sided_p(z), gpu_test::normal_two_sided_p(8));
}

// --- T6: the depth rule and the counting rule through gpu::render ----------------------------------------------------

// §8's toy scene: a floor and a wall of grey 0.8 under a lamp of L = 4 facing the floor, black sky, the camera at
// (0, 1, 2) through the viewport x ∈ [−1, 1], y ∈ [0, 2] at z = 0
gpu::scene_desc toy_scene(int depth, light_combination c, int size, int sqrt_spp)
{
    gpu::scene_desc s;
    const int grey = add_material(s, gpu::material_kind::lambertian, {0.8, 0.8, 0.8});
    const int lamp = add_material(s, gpu::material_kind::diffuse_light, {4, 4, 4});
    s.quads.push_back({.q = {-1, 0, -1}, .u = {0, 0, 2}, .v = {2, 0, 0}, .material = grey});     // floor, normal +y
    s.quads.push_back({.q = {-1, 0, -1}, .u = {2, 0, 0}, .v = {0, 2, 0}, .material = grey});     // wall, normal +z
    s.quads.push_back({.q = {-0.5, 2, -0.5}, .u = {1, 0, 0}, .v = {0, 0, 1}, .material = lamp}); // lamp, normal −y
    const double step = 2.0 / size;
    s.camera = {.image_width = size,
                .image_height = size,
                .sqrt_spp = sqrt_spp,
                .max_depth = depth,
                .center = {0, 1, 2},
                .pixel00_loc = {-1 + step / 2, 2 - step / 2, 0},
                .pixel_delta_u = {step, 0, 0},
                .pixel_delta_v = {0, -step, 0},
                .defocus = false,
                .sky_bottom = {0, 0, 0},
                .sky_top = {0, 0, 0},
                .sampling = gpu::sampling_strategy::material};
    s.quality.combination = c;
    return s;
}

// The pixels' differences B − A (channel averages): their mean and its standard error over the pixels
gpu_test::mean_estimate pixel_difference(const gpu::render_result& a, const gpu::render_result& b)
{
    double sum = 0;
    double squares = 0;
    const std::size_t pixels = a.pixels.size() / 3;
    for (std::size_t p = 0; p < pixels; ++p)
    {
        const double d = (b.pixels[3 * p] + b.pixels[3 * p + 1] + b.pixels[3 * p + 2] - a.pixels[3 * p] -
                          a.pixels[3 * p + 1] - a.pixels[3 * p + 2]) /
                         3;
        sum += d;
        squares += d * d;
    }
    return gpu_test::from_sums(sum, squares, static_cast<std::int64_t>(pixels));
}

GPU_TEST(lamps, t6)
{
    const bool quick = ctx.quick();
    const int size = quick ? 32 : 64;
    const int sqrt_spp = quick ? 2 : 16;
    const std::int64_t camera_rays = static_cast<std::int64_t>(size) * size * sqrt_spp * sqrt_spp;
    const auto render = [&](int depth, light_combination c, std::uint64_t seed) {
        return gpu::render(toy_scene(depth, c, size, sqrt_spp), {.seed = seed, .quality_kernel = true});
    };
    // [d − 2][combination]
    const gpu::render_result r[2][3] = {
        {render(2, combinations[0], 7671), render(2, combinations[1], 7672), render(2, combinations[2], 7673)},
        {render(3, combinations[0], 7674), render(3, combinations[1], 7675), render(3, combinations[2], 7676)}};
    // A second material render per depth, so that nee and mis are each compared with a material image of their own:
    // with one, a material render far from its mean would show as the same offset in both checks
    const gpu::render_result second_material[2] = {render(2, combinations[0], 7677), render(3, combinations[0], 7678)};

    std::int64_t not_finite = 0;
    std::int64_t values = 0;
    const auto count_finite = [&](const gpu::render_result& image) {
        for (const double x : image.pixels)
        {
            not_finite += std::isfinite(x) ? 0 : 1;
        }
        values += static_cast<std::int64_t>(image.pixels.size());
    };
    for (const auto& by_depth : r)
    {
        for (const gpu::render_result& image : by_depth)
        {
            count_finite(image);
        }
    }
    for (const gpu::render_result& image : second_material)
    {
        count_finite(image);
    }
    ctx.count("t6-finite", not_finite, 0, values);

    for (int d = 0; d < 2; ++d)
    {
        const std::string depth = "d" + std::to_string(d + 2);
        const gpu::render_result& material = r[d][0];
        for (int k = 1; k < 3; ++k)
        {
            const gpu::render_result& image = r[d][k];
            const std::string mode = mode_name(combinations[k]);
            // T6a: the same expected image as the material's (nee against one material render, mis against the other)
            const gpu_test::mean_estimate diff = pixel_difference(k == 1 ? material : second_material[d], image);
            ctx.mean("t6-" + mode + "-" + depth, diff.mean, diff.standard_error, 0);
            // T6c: one shadow ray per continuation, at every non-delta vertex with rays_left > 1
            const auto rays = static_cast<std::int64_t>(image.rays);
            const auto shadow = static_cast<std::int64_t>(image.shadow_rays);
            const std::int64_t off = shadow > rays - camera_rays ? shadow - (rays - camera_rays)
                                                                 : (rays - camera_rays) - shadow;
            ctx.count("t6-counting-" + depth + "-" + mode, off,
                      std::max<std::int64_t>(2, static_cast<std::int64_t>(1e-4 * static_cast<double>(camera_rays))),
                      camera_rays);
            // T6d: the continuation is the material's own sample in all three: the same rays per sample
            const double ratio = static_cast<double>(image.rays) / static_cast<double>(material.rays);
            if (quick)
            {
                ctx.info("t6-rays-" + depth + "-" + mode, "rays per sample / material's = " + number(ratio) +
                                                              " (no verdict at --quick's sample count)");
            }
            else
            {
                ctx.exact("t6-rays-" + depth + "-" + mode, ratio, 1, 0.002);
            }
            ctx.info("t6-per-sample-" + depth + "-" + mode,
                     number(static_cast<double>(image.rays) / camera_rays) + " rays, " +
                         number(static_cast<double>(image.shadow_rays) / camera_rays) + " shadow rays");
        }
    }
    // nee against mis, the two renders of light samples, at each depth
    for (int d = 0; d < 2; ++d)
    {
        const gpu_test::mean_estimate diff = pixel_difference(r[d][1], r[d][2]);
        ctx.mean("t6-nee-mis-d" + std::to_string(d + 2), diff.mean, diff.standard_error, 0);
    }
    // T6b: the power of T6a, made explicit: a light sample at the last vertex would add M_3 − M_2 to N_2
    const gpu_test::mean_estimate power = pixel_difference(r[0][0], r[1][0]);
    const double z = power.mean / power.standard_error;
    ctx.must_reject("t6-power", z, 0, z > 0 ? gpu_test::normal_two_sided_p(z) : 1.0, gpu_test::normal_two_sided_p(8));
    // W2: the material's own samples alone trace no shadow ray
    ctx.exact("counting-material",
              static_cast<double>(r[0][0].shadow_rays + r[1][0].shadow_rays + second_material[0].shadow_rays +
                                  second_material[1].shadow_rays),
              0, 0);
}

// --- T7: occluded against a loop over every primitive ----------------------------------------------------------------

// 200 primitives in [−10, 10]³: 80 still spheres (radius 0.2 to 2), 40 moving ones (displacement up to 2 per axis), 80
// parallelograms of sides 0.5 to 4 in random directions (at least 11.5° apart)
gpu::scene_desc t7_scene()
{
    host_random rng{7680};
    gpu::scene_desc s;
    const int matte = add_material(s, gpu::material_kind::lambertian, {0.5, 0.5, 0.5});
    const auto point = [&rng] {
        const double x = rng.uniform(-10, 10);
        const double y = rng.uniform(-10, 10);
        const double z = rng.uniform(-10, 10);
        return gpu::vec3d{x, y, z};
    };
    for (int i = 0; i < 120; ++i)
    {
        gpu::sphere_desc ball{.center = point(), .material = matte};
        ball.radius = rng.uniform(0.2, 2);
        if (i >= 80)
        {
            const double x = rng.uniform(-2, 2);
            const double y = rng.uniform(-2, 2);
            const double z = rng.uniform(-2, 2);
            ball.displacement = {x, y, z};
        }
        s.spheres.push_back(ball);
    }
    for (int i = 0; i < 80; ++i)
    {
        const gpu::vec3d q = point();
        const dvec a = rng.direction();
        dvec b = rng.direction();
        while (length(cross(a, b)) < 0.2)
        {
            b = rng.direction();
        }
        const double lu = rng.uniform(0.5, 4);
        const double lv = rng.uniform(0.5, 4);
        s.quads.push_back({.q = q, .u = {lu * a.x, lu * a.y, lu * a.z}, .v = {lv * b.x, lv * b.y, lv * b.z},
                           .material = matte});
    }
    return s;
}

struct quad_sides
{
    vec3<float> u;
    vec3<float> v;
};

// closest_hit's test of primitive g over (0.001, ∞) with its self-intersection rules (shared.md §6): false where the
// rules skip it or the ray misses it. For t_max < ∞ the test over (0.001, t_max) is this one and t < t_max (a sphere,
// open) or t ≤ t_max (a quad, closed): neither function's choice of t depends on t_max.
__device__ inline bool first_hit(const gpu::bvh_view<float>& tree, int g, const gpu::ray<float>& r, float inverse_a,
                                 int from, bool inward, float& t, bool& quad)
{
    bool front;
    quad = g >= tree.quad_base;
    if (quad)
    {
        return g != from && gpu::hit_quad(tree.quads[g - tree.quad_base], r, 0.001f, infinite, t, front);
    }
    const bool own = g == from;
    if (own && !inward)
    {
        return false;
    }
    return gpu::hit_sphere(sphere_at(tree, g, r.time), r, inverse_a, 0.001f, infinite, own, t, front);
}

// Whether primitive g's test is within float rounding of another answer for this segment: a sphere the line grazes, a
// hit at a quad's edge, a root at t_min. The tolerances scale with the numbers the tests subtract (the distances from
// the origin), a few float steps of them.
__device__ inline bool near_boundary(const gpu::bvh_view<float>& tree, int g, const gpu::ray<float>& r,
                                     float inverse_a, int from, bool inward, float t_max)
{
    if (g >= tree.quad_base)
    {
        if (g == from)
        {
            return false;
        }
        const gpu::quad<float> q = tree.quads[g - tree.quad_base];
        const float denominator = gpu::dot(q.normal, r.direction);
        if (fabsf(denominator) < 1e-6f)
        {
            return true;
        }
        const vec3<float> origin_to_q = q.q - r.origin;
        const float t = gpu::dot(q.normal, origin_to_q) / denominator;
        const float size = sqrtf(origin_to_q.length_squared()) + fabsf(t);
        if (fabsf(t - 0.001f) <= 1e-6f * (1.0f + size) / fabsf(denominator))
        {
            return true;
        }
        if (!(t > 0.0005f && t <= 1.0002f * t_max))
        {
            return false;
        }
        const vec3<float> p = t * r.direction - origin_to_q;
        const float alpha = gpu::dot(p, q.alpha_axis);
        const float beta = gpu::dot(p, q.beta_axis);
        const float ea = 1e-6f * (1.0f + size) * sqrtf(q.alpha_axis.length_squared());
        const float eb = 1e-6f * (1.0f + size) * sqrtf(q.beta_axis.length_squared());
        const bool inside = alpha >= -ea && alpha <= 1.0f + ea && beta >= -eb && beta <= 1.0f + eb;
        const bool edge = fabsf(alpha) <= ea || fabsf(alpha - 1.0f) <= ea || fabsf(beta) <= eb ||
                          fabsf(beta - 1.0f) <= eb;
        return inside && edge;
    }
    const bool own = g == from;
    if (own && !inward)
    {
        return false;
    }
    const gpu::sphere<float> s = sphere_at(tree, g, r.time);
    const vec3<float> oc = s.center - r.origin;
    const float h = gpu::dot(r.direction, oc) * inverse_a;
    if (!(h > -s.radius - 1.0f && h < t_max + s.radius + 1.0f))
    {
        return false; // far from the segment
    }
    const vec3<float> l = oc - h * r.direction;
    const float r2 = s.radius * s.radius;
    const float discriminant = r2 - l.length_squared();
    const float size = sqrtf(oc.length_squared()) + fabsf(h);
    if (fabsf(discriminant) <= 1e-6f * (1.0f + size) * (1.0f + s.radius) + 1e-5f * r2)
    {
        return true;
    }
    if (discriminant < 0.0f)
    {
        return false;
    }
    const float half = sqrtf(discriminant * inverse_a);
    const float tolerance = 1e-6f * (1.0f + size);
    return (!own && fabsf(h - half - 0.001f) <= tolerance) || fabsf(h + half - 0.001f) <= tolerance;
}

enum t7_tally : int
{
    t7_rays,
    t7_occluded,       // the reference's answer was "occluded"
    t7_band,           // a hit within 10^-4·t_max of t_max (§6.2's band)
    t7_near,           // outside the band, within rounding of another decision (decision 7)
    t7_mismatch,       // occluded's answer differs, outside both
    t7_band_mismatch,  // ... inside the band
    t7_near_mismatch,  // ... inside the rounding zone
    t7_count,
};

struct t7_ray
{
    gpu::bvh_view<float> tree;
    const quad_sides* sides;
    int count;

    __device__ int random_primitive(gpu::pcg32& rng) const
    {
        return ::min(static_cast<int>(gpu::random_real<float>(rng) * static_cast<float>(count)), count - 1);
    }

    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally) const
    {
        // Half the rays start on a primitive (from = its id), half anywhere (from = −1)
        const float time = gpu::random_real<float>(rng);
        int from = -1;
        vec3<float> origin{};
        vec3<float> outward{};
        if (gpu::random_real<float>(rng) < 0.5f)
        {
            from = random_primitive(rng);
            if (from >= tree.quad_base)
            {
                const int i = from - tree.quad_base;
                const float a = gpu::random_real<float>(rng);
                const float b = gpu::random_real<float>(rng);
                origin = tree.quads[i].q + a * sides[i].u + b * sides[i].v;
            }
            else
            {
                const gpu::sphere<float> s = sphere_at(tree, from, time);
                outward = gpu::random_unit_vector<float>(rng);
                origin = s.center + s.radius * outward;
            }
        }
        else
        {
            origin = point_in_box(rng, {-10.0f, -10.0f, -10.0f}, {10.0f, 10.0f, 10.0f});
        }
        const vec3<float> d = gpu::random_unit_vector<float>(rng);
        const bool inward = from >= 0 && from < tree.quad_base && gpu::dot(d, outward) < 0.0f;
        float t_max = infinite;
        if (gpu::random_real<float>(rng) < 0.8f)
        {
            t_max = 0.01f * expf(gpu::random_real<float>(rng) * logf(5000.0f)); // log-uniform in [0.01, 50)
        }
        const gpu::ray<float> r{origin, d, time};
        const float inverse_a = 1.0f / d.length_squared();

        // skip: none (60%), a random primitive (20%), or the closest hit (20%)
        const float pick = gpu::random_real<float>(rng);
        int skip = -1;
        if (pick >= 0.8f)
        {
            float closest = infinite;
            for (int g = 0; g < count; ++g)
            {
                float t;
                bool quad;
                if (first_hit(tree, g, r, inverse_a, from, inward, t, quad) && t < closest)
                {
                    closest = t;
                    skip = g;
                }
            }
        }
        else if (pick >= 0.6f)
        {
            skip = random_primitive(rng);
        }

        bool reference = false;
        bool band = false;
        bool close_call = false;
        for (int g = 0; g < count; ++g)
        {
            if (g == skip)
            {
                continue;
            }
            float t;
            bool quad;
            if (first_hit(tree, g, r, inverse_a, from, inward, t, quad))
            {
                reference = reference || (quad ? t <= t_max : t < t_max);
                band = band || (isfinite(t_max) && fabsf(t - t_max) <= 1e-4f * t_max);
            }
            close_call = close_call || near_boundary(tree, g, r, inverse_a, from, inward, t_max);
        }
        const bool found = gpu::occluded(tree, r, t_max, from, inward, skip);
        ++tally[t7_rays];
        tally[t7_occluded] += reference ? 1 : 0;
        tally[t7_band] += band ? 1 : 0;
        tally[t7_near] += !band && close_call ? 1 : 0;
        if (found != reference)
        {
            ++tally[band ? t7_band_mismatch : close_call ? t7_near_mismatch : t7_mismatch];
        }
    }
};

void test_occluded(gpu_test::context& ctx, int max_leaf_size, std::uint64_t seed, const std::string& name)
{
    const gpu::scene_desc scene = t7_scene();
    const gpu_test::device_scene device(scene, max_leaf_size);
    // Each quad's sides by its place in the tree, to put points on it
    std::vector<quad_sides> sides(scene.quads.size());
    const int quad_base = device.view().tree.quad_base;
    for (std::size_t k = 0; k < scene.quads.size(); ++k)
    {
        sides[static_cast<std::size_t>(device.quad_ids()[k] - quad_base)] = {gpu::to_real<float>(scene.quads[k].u),
                                                                             gpu::to_real<float>(scene.quads[k].v)};
    }
    const gpu_test::device_buffer<quad_sides> device_sides(sides);
    const t7_ray f{device.view().tree, device_sides.get(), device.surface_count()};
    // 10^6 rays in 4 launches: each ray loops over the 200 primitives three times
    const int calls = ctx.quick() ? 1 : 4;
    const std::int64_t per_call = ctx.samples(full_samples) / calls;
    std::array<std::int64_t, t7_count> t{};
    for (int c = 0; c < calls; ++c)
    {
        add(t, device_tally<t7_count>(f, seed + static_cast<std::uint64_t>(c), per_call));
    }
    ctx.count(name, t[t7_mismatch], 0, t[t7_rays]);
    ctx.info(name + "-cases", std::to_string(t[t7_occluded]) + " of " + std::to_string(t[t7_rays]) +
                                  " rays occluded; in the t_max band " + std::to_string(t[t7_band]) + " (" +
                                  std::to_string(t[t7_band_mismatch]) + " mismatched), within rounding of another " +
                                  "decision " + std::to_string(t[t7_near]) + " (" +
                                  std::to_string(t[t7_near_mismatch]) + " mismatched)");
}

GPU_TEST(lamps, t7)
{
    test_occluded(ctx, 1, 7681, "t7-occluded-leaf1");
    test_occluded(ctx, 8, 7691, "t7-occluded-leaf8");
}

// =====================================================================================================================
// Suite media (Step 77)
// =====================================================================================================================

// The media of D.md §7, centred on the origin: MS a sphere of radius 2, ρ = 0.5; MSm the same moving by (1, 0, 0); MB
// the box [−1.5, 1.5] × [−1, 1] × [−2, 2], ρ = 0.3; MR that box turned 30° about y, then 20° about x, ρ = 0.8; MO MS
// and MB together
enum class medium_set
{
    ms,
    msm,
    mb,
    mr,
    mo,
};

gpu::medium_desc sphere_medium(const gpu::vec3d& displacement, int material)
{
    gpu::medium_desc m{.boundary = gpu::boundary_shape::sphere, .neg_inv_density = -2, .material = material};
    m.sphere = {.center = {0, 0, 0}, .radius = 2, .displacement = displacement};
    return m;
}

// A box's six faces (the books' box)
std::vector<gpu::quad_desc> box_quads(const gpu::vec3d& lo, const gpu::vec3d& hi)
{
    const double dx = hi.x - lo.x;
    const double dy = hi.y - lo.y;
    const double dz = hi.z - lo.z;
    return {{.q = {lo.x, lo.y, hi.z}, .u = {dx, 0, 0}, .v = {0, dy, 0}},
            {.q = {hi.x, lo.y, hi.z}, .u = {0, 0, -dz}, .v = {0, dy, 0}},
            {.q = {hi.x, lo.y, lo.z}, .u = {-dx, 0, 0}, .v = {0, dy, 0}},
            {.q = {lo.x, lo.y, lo.z}, .u = {0, 0, dz}, .v = {0, dy, 0}},
            {.q = {lo.x, hi.y, hi.z}, .u = {dx, 0, 0}, .v = {0, 0, -dz}},
            {.q = {lo.x, lo.y, lo.z}, .u = {dx, 0, 0}, .v = {0, 0, dz}}};
}

// R_x(x°)·R_y(y°) of a point or a vector
gpu::vec3d turned(const gpu::vec3d& p, double y_degrees, double x_degrees)
{
    const double a = y_degrees * std::numbers::pi / 180;
    const double b = x_degrees * std::numbers::pi / 180;
    const gpu::vec3d r{std::cos(a) * p.x + std::sin(a) * p.z, p.y, -std::sin(a) * p.x + std::cos(a) * p.z};
    return {r.x, std::cos(b) * r.y - std::sin(b) * r.z, std::sin(b) * r.y + std::cos(b) * r.z};
}

gpu::medium_desc box_medium(double density, bool turn, int material)
{
    std::vector<gpu::quad_desc> quads = box_quads({-1.5, -1, -2}, {1.5, 1, 2});
    if (turn)
    {
        for (gpu::quad_desc& q : quads)
        {
            q = {.q = turned(q.q, 30, 20), .u = turned(q.u, 30, 20), .v = turned(q.v, 30, 20)};
        }
    }
    return {.boundary = gpu::boundary_shape::convex_quads,
            .quads = quads,
            .neg_inv_density = -1 / density,
            .material = material};
}

gpu::scene_desc media_scene(medium_set which)
{
    gpu::scene_desc s;
    const int smoke = add_material(s, gpu::material_kind::isotropic, {0.9, 0.9, 0.9});
    switch (which)
    {
    case medium_set::ms:
        s.media.push_back(sphere_medium({0, 0, 0}, smoke));
        break;
    case medium_set::msm:
        s.media.push_back(sphere_medium({1, 0, 0}, smoke));
        break;
    case medium_set::mb:
        s.media.push_back(box_medium(0.3, false, smoke));
        break;
    case medium_set::mr:
        s.media.push_back(box_medium(0.8, true, smoke));
        break;
    case medium_set::mo:
        s.media.push_back(sphere_medium({0, 0, 0}, smoke));
        s.media.push_back(box_medium(0.3, false, smoke));
        break;
    }
    return s;
}

constexpr medium_set medium_sets[] = {medium_set::ms, medium_set::msm, medium_set::mb, medium_set::mr, medium_set::mo};
const char* const medium_names[] = {"ms", "msm", "mb", "mr", "mo"};

// T5's rays: 60% from 6 away (outside every medium at every time), aimed at a point within 2.5 of the centre, so that
// most cross; 40% from within 0.9 of the centre (inside every medium); t_max ∞ for half, uniform in (0.5, 10) for the
// rest; the time uniform in [0, 1)
struct media_ray
{
    gpu::ray<float> r;
    float t_max;
    bool inside;
};

__device__ inline media_ray random_media_ray(gpu::pcg32& rng)
{
    media_ray m{};
    const float time = gpu::random_real<float>(rng);
    m.inside = gpu::random_real<float>(rng) < 0.4f;
    vec3<float> origin;
    vec3<float> direction;
    if (m.inside)
    {
        origin = point_in_ball(rng, {0.0f, 0.0f, 0.0f}, 0.9f);
        direction = gpu::random_unit_vector<float>(rng);
    }
    else
    {
        origin = 6.0f * gpu::random_unit_vector<float>(rng);
        direction = gpu::unit_vector(point_in_ball(rng, {0.0f, 0.0f, 0.0f}, 2.5f) - origin);
    }
    m.t_max = gpu::random_real<float>(rng) < 0.5f ? infinite : uniform_in(rng, 0.5f, 10.0f);
    m.r = {origin, direction, time};
    return m;
}

// The oracle: medium_collision for every medium, with numbers of its own each, as closest_event calls it; the ray
// survives if none collides
__device__ inline bool survives(const gpu::media_view<float>& media, const gpu::ray<float>& r, float t_max,
                                gpu::pcg32& rng)
{
    const float inverse_a = 1.0f / r.direction.length_squared();
    const vec3<float> inverse{gpu::safe_inverse(r.direction.x), gpu::safe_inverse(r.direction.y),
                              gpu::safe_inverse(r.direction.z)};
    const vec3<float> origin_scaled = r.origin * inverse;
    bool collided = false;
    for (int k = 0; k < media.count; ++k)
    {
        float t;
        const bool hit = gpu::medium_collision(media.media[k], media.boundary_quads, r, inverse_a, inverse,
                                               origin_scaled, 0.001f, t_max, rng, t);
        collided = collided || hit;
    }
    return !collided;
}

// X3's fault: the chord from where the line enters the volume, behind the origin too (t_in = line_in, no clamp)
__device__ inline float planted_transmittance(const gpu::medium<float>& m, const gpu::quad<float>* quads,
                                              const gpu::ray<float>& r, float t_max)
{
    float line_in;
    float line_out;
    if (!gpu::boundary_crossings(m, quads, r, 1.0f / r.direction.length_squared(), line_in, line_out))
    {
        return 1.0f;
    }
    const float t_out = fminf(line_out, t_max);
    if (!(line_in < t_out))
    {
        return 1.0f;
    }
    return expf((t_out - line_in) * sqrtf(r.direction.length_squared()) / m.neg_inv_density);
}

// The oracle's chance of surviving one medium, in closed form: the segment exactly as medium_collision makes it
// (constant_medium.cuh: the box, the boundary's crossings and its 0.0001 rule, the clamps), and e^(−ρ·length)
__device__ inline double segment_survival(const gpu::medium<float>& m, const gpu::quad<float>* quads,
                                          const gpu::ray<float>& r, float t_max)
{
    const float t_min = 0.001f;
    const float inverse_a = 1.0f / r.direction.length_squared();
    const vec3<float> inverse{gpu::safe_inverse(r.direction.x), gpu::safe_inverse(r.direction.y),
                              gpu::safe_inverse(r.direction.z)};
    const vec3<float> origin_scaled = r.origin * inverse;
    if (gpu::box_entry(m.box_min, m.box_max, inverse, origin_scaled, t_min, t_max) == infinite)
    {
        return 1;
    }
    float line_in;
    float line_out;
    if (!gpu::boundary_crossings(m, quads, r, inverse_a, line_in, line_out))
    {
        return 1;
    }
    float t_in = fmaxf(line_in, t_min);
    const float t_out = fminf(line_out, t_max);
    if (t_in >= t_out)
    {
        return 1;
    }
    t_in = fmaxf(t_in, 0.0f);
    const float inside = (t_out - t_in) * sqrtf(r.direction.length_squared());
    return ::exp(double(inside) / double(m.neg_inv_density));
}

// T5: Σ(survived − T), Σ T(1 − T): z = the first / √(the second)
struct t5_ray
{
    gpu::media_view<float> media;
    bool planted; // X3: the inside rays' T without the t_min clamp (MS)
    __device__ void operator()(gpu::pcg32& rng, double* sums) const
    {
        const media_ray m = random_media_ray(rng);
        float t = gpu::transmittance(media, m.r, m.t_max);
        if (planted && m.inside)
        {
            t = planted_transmittance(media.media[0], media.boundary_quads, m.r, m.t_max);
        }
        const bool survived = survives(media, m.r, m.t_max, rng);
        sums[0] += (survived ? 1.0 : 0.0) - t;
        sums[1] += double(t) * (1.0 - t);
    }
};

// --- T5b: chords in closed form --------------------------------------------------------------------------------------

struct chord_case
{
    gpu::ray<float> r;
    float t_max;
};

struct transmittance_of
{
    gpu::media_view<float> media;
    const chord_case* cases;
    __device__ float operator()(int i) const { return gpu::transmittance(media, cases[i].r, cases[i].t_max); }
};

GPU_TEST(media, chord)
{
    const gpu_test::device_scene ms(media_scene(medium_set::ms));
    constexpr double rho = 0.5;
    constexpr double radius = 2;
    struct expectation
    {
        std::string name;
        double value;
        double tolerance; // relative; 0: exactly
    };
    std::vector<chord_case> cases;
    std::vector<expectation> expected;
    // From (−10, b, 0) along +x through the whole sphere: T = exp(−ρ·2√(R² − b²)), b as the GPU has it
    const std::pair<double, const char*> impacts[] = {{0, "chord-b0"},        {0.25, "chord-b0.25R"},
                                                      {0.5, "chord-b0.5R"},   {0.75, "chord-b0.75R"},
                                                      {0.95, "chord-b0.95R"}, {0.995, "chord-b0.995R"}};
    for (const auto& [fraction, name] : impacts)
    {
        const float b = static_cast<float>(fraction * radius);
        cases.push_back({{{-10.0f, b, 0.0f}, {1.0f, 0.0f, 0.0f}}, infinite});
        expected.push_back({name, std::exp(-rho * 2 * std::sqrt(radius * radius - double(b) * b)), 1e-5});
    }
    cases.push_back({{{-10.0f, static_cast<float>(1.001 * radius), 0.0f}, {1.0f, 0.0f, 0.0f}}, infinite});
    expected.push_back({"chord-miss", 1, 0});
    // From the centre outward: the segment starts at t_min = 0.001
    cases.push_back({{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}, infinite});
    expected.push_back({"chord-from-centre", std::exp(-rho * (radius - 0.001)), 1e-5});
    // From outside, stopping at the centre (t_max = 10)
    cases.push_back({{{-10.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}, 10.0f});
    expected.push_back({"chord-to-centre", std::exp(-rho * radius), 1e-5});
    // Directions that aren't unit (the camera's rays aren't): t counts lengths of d, the chord stays the world's
    cases.push_back({{{-10.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}}, infinite});
    expected.push_back({"chord-long-direction", std::exp(-rho * 2 * radius), 1e-5});
    cases.push_back({{{-10.0f, 0.0f, 0.0f}, {0.5f, 0.0f, 0.0f}}, 20.0f});
    expected.push_back({"chord-short-direction-to-centre", std::exp(-rho * radius), 1e-5});

    const gpu_test::device_buffer<chord_case> device_cases(cases);
    const std::vector<float> t =
        device_map<float>(transmittance_of{ms.view().media, device_cases.get()}, static_cast<int>(cases.size()));
    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        if (expected[i].tolerance == 0)
        {
            ctx.exact(expected[i].name, t[i], expected[i].value, 0);
        }
        else
        {
            ctx.exact(expected[i].name, t[i] / expected[i].value, 1, expected[i].tolerance);
        }
    }
}

// --- T5c: the limits -------------------------------------------------------------------------------------------------

// Rays that meet no medium: from 8 away going away, or toward the media but ending at least 3.5 from the centre (every
// medium lies within 3 of it at every time)
struct t5c_clear
{
    gpu::media_view<float> media;
    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally) const
    {
        const float time = gpu::random_real<float>(rng);
        const vec3<float> origin = 8.0f * gpu::random_unit_vector<float>(rng);
        vec3<float> d;
        float t_max;
        if (gpu::random_real<float>(rng) < 0.5f)
        {
            d = gpu::random_unit_vector<float>(rng);
            if (gpu::dot(d, origin) < 0.0f)
            {
                d = -d;
            }
            t_max = gpu::random_real<float>(rng) < 0.5f ? infinite : uniform_in(rng, 0.5f, 10.0f);
        }
        else
        {
            d = gpu::unit_vector(point_in_ball(rng, {0.0f, 0.0f, 0.0f}, 2.0f) - origin);
            t_max = uniform_in(rng, 0.5f, 4.5f);
        }
        const gpu::ray<float> r{origin, d, time};
        ++tally[0];
        if (!(gpu::transmittance(media, r, t_max) == 1.0f))
        {
            ++tally[1];
        }
    }
};

// T ∈ (0, 1] on T5's rays
struct t5c_range
{
    gpu::media_view<float> media;
    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally) const
    {
        const media_ray m = random_media_ray(rng);
        const float t = gpu::transmittance(media, m.r, m.t_max);
        ++tally[0];
        if (!(t > 0.0f && t <= 1.0f))
        {
            ++tally[1];
        }
    }
};

// MO's T is MS's times MB's: the media are independent
struct t5c_overlap
{
    gpu::media_view<float> ms;
    gpu::media_view<float> mb;
    gpu::media_view<float> mo;
    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally) const
    {
        const media_ray m = random_media_ray(rng);
        const double both = gpu::transmittance(mo, m.r, m.t_max);
        const double product = double(gpu::transmittance(ms, m.r, m.t_max)) * gpu::transmittance(mb, m.r, m.t_max);
        ++tally[0];
        if (!(::fabs(both - product) <= 1e-6 * product))
        {
            ++tally[1];
        }
    }
};

GPU_TEST(media, limits)
{
    const std::int64_t n = ctx.samples(full_samples);
    std::array<std::int64_t, 2> clear{};
    std::array<std::int64_t, 2> range{};
    for (int k = 0; k < 5; ++k)
    {
        const gpu_test::device_scene scene(media_scene(medium_sets[k]));
        add(clear, device_tally<2>(t5c_clear{scene.view().media}, 7711 + static_cast<std::uint64_t>(k), n));
        add(range, device_tally<2>(t5c_range{scene.view().media}, 7716 + static_cast<std::uint64_t>(k), n));
    }
    ctx.count("limits-clear", clear[1], 0, clear[0]);
    ctx.count("limits-range", range[1], 0, range[0]);
    const gpu_test::device_scene ms(media_scene(medium_set::ms));
    const gpu_test::device_scene mb(media_scene(medium_set::mb));
    const gpu_test::device_scene mo(media_scene(medium_set::mo));
    const auto overlap =
        device_tally<2>(t5c_overlap{ms.view().media, mb.view().media, mo.view().media}, 7721, n);
    ctx.count("limits-overlap", overlap[1], 0, overlap[0]);
}

// --- T5d: at the boundary's 0.0001 rule ------------------------------------------------------------------------------

// MS rays passing b = R(1 − δ) from the centre, δ log-uniform in [10^-8, 10^-4], along an axis (±x, ±y, ±z): then
// |d|² = 1 and d·oc are exact however nvcc fuses them, and transmittance and the oracle get the same roots, bit for bit
struct t5d_ray
{
    gpu::media_view<float> media;
    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally) const
    {
        const int axis = ::min(static_cast<int>(gpu::random_real<float>(rng) * 6.0f), 5);
        const float sign = axis % 2 == 0 ? 1.0f : -1.0f;
        vec3<float> d{};
        vec3<float> e1{};
        vec3<float> e2{};
        switch (axis / 2)
        {
        case 0:
            d = {sign, 0.0f, 0.0f};
            e1 = {0.0f, 1.0f, 0.0f};
            e2 = {0.0f, 0.0f, 1.0f};
            break;
        case 1:
            d = {0.0f, sign, 0.0f};
            e1 = {0.0f, 0.0f, 1.0f};
            e2 = {1.0f, 0.0f, 0.0f};
            break;
        default:
            d = {0.0f, 0.0f, sign};
            e1 = {1.0f, 0.0f, 0.0f};
            e2 = {0.0f, 1.0f, 0.0f};
            break;
        }
        const float delta = 1e-8f * expf(gpu::random_real<float>(rng) * logf(1e4f));
        const float b = 2.0f * (1.0f - delta);
        float s;
        float c;
        sincospif(2.0f * gpu::random_real<float>(rng), &s, &c);
        const gpu::ray<float> r{-10.0f * d + (b * c) * e1 + (b * s) * e2, d};
        const double t = gpu::transmittance(media, r, infinite);
        const double expected = segment_survival(media.media[0], media.boundary_quads, r, infinite);
        ++tally[0];
        if (!(::fabs(t - expected) <= 1e-5 * expected))
        {
            ++tally[1];
        }
        tally[2] += expected < 1 ? 1 : 0; // the rule kept a chord
    }
};

GPU_TEST(media, grazing)
{
    const gpu_test::device_scene ms(media_scene(medium_set::ms));
    const auto t = device_tally<3>(t5d_ray{ms.view().media}, 7731, ctx.samples(full_samples));
    ctx.count("grazing", t[1], 0, t[0]);
    ctx.info("grazing-chords", std::to_string(t[2]) + " of " + std::to_string(t[0]) +
                                   " rays kept a chord, the others none");
}

// --- T5 and X3 -------------------------------------------------------------------------------------------------------

GPU_TEST(media, t5)
{
    const std::int64_t n = ctx.samples(full_samples);
    const auto rays = static_cast<double>(gpu_test::drawn(n));
    for (int k = 0; k < 5; ++k)
    {
        const gpu_test::device_scene scene(media_scene(medium_sets[k]));
        const auto s = device_sums<2>(t5_ray{scene.view().media, false}, 7701 + static_cast<std::uint64_t>(k), n);
        // z = Σ(survived − T) / √(Σ T(1 − T)), as a mean of survived − T with its standard error
        ctx.mean(std::string("t5-") + medium_names[k], s[0] / rays, std::sqrt(s[1]) / rays, 0);
    }
    const gpu_test::device_scene ms(media_scene(medium_set::ms));
    const auto planted = device_sums<2>(t5_ray{ms.view().media, true}, 7706, n);
    const double z = planted[0] / std::sqrt(planted[1]);
    ctx.must_reject("planted-tmin", z, 0, gpu_test::normal_two_sided_p(z), gpu_test::normal_two_sided_p(8));
}

// --- T3p: particles take light samples -------------------------------------------------------------------------------

GPU_TEST(media, particles)
{
    // light's two lamps (pmf ½) and the smoke MS around the points
    gpu::scene_desc s = light_setup();
    const int smoke = add_material(s, gpu::material_kind::isotropic, {0.9, 0.9, 0.9});
    s.media.push_back(sphere_medium({0, 0, 0}, smoke));
    const lamp_scene lamps(s, false);
    const std::int64_t n = ctx.samples(full_samples);
    const auto total = test_weights(lamps, 2, 1, true, 7741, n);
    std::int64_t pdf = 0;
    for (int k = 0; k < 2; ++k)
    {
        const auto& t = total[static_cast<std::size_t>(k)];
        const std::string name = std::string("t3-particles-") + (k == 0 ? "nee" : "mis");
        ctx.count(name, t[t3_wrong], 0, t[t3_tested]);
        ctx.info(name + "-tested", std::to_string(t[t3_tested]) + " directions from both techniques");
        pdf += t[t3_pdf];
    }
    ctx.count("t3-particles-pdf", pdf, 0, 4 * gpu_test::drawn(n));
}

// --- T5v: the shadow ray through smoke -------------------------------------------------------------------------------

struct t5v_ray
{
    gpu::quality_view v;
    gpu::quad<float> wall;
    int wall_id;
    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally) const
    {
        // From x = −5 toward the wall at x = 3, through the smoke, hitting the wall far from its edges
        const float y = uniform_in(rng, -0.5f, 0.5f);
        const float z = uniform_in(rng, -0.5f, 0.5f);
        const float dy = uniform_in(rng, -0.1f, 0.1f);
        const float dz = uniform_in(rng, -0.1f, 0.1f);
        const gpu::ray<float> r{{-5.0f, y, z}, gpu::unit_vector(vec3<float>{1.0f, dy, dz})};
        const float t_max = uniform_in(rng, 0.5f, 16.0f);
        const int skip = gpu::random_real<float>(rng) < 0.3f ? wall_id : -1;
        float t;
        bool front;
        const bool on_wall = gpu::hit_quad(wall, r, 0.001f, infinite, t, front);
        if (on_wall && fabsf(t - t_max) <= 1e-3f * t_max)
        {
            return; // too close to call
        }
        ++tally[0];
        const bool blocked = skip != wall_id && on_wall && t <= t_max;
        const gpu::bvh_tracer tracer{v};
        const float visible = tracer.visibility(r, gpu::spawn{-1, false}, t_max, skip);
        const float expected = blocked ? 0.0f : gpu::transmittance(v.base.media, r, t_max);
        const bool ok = blocked ? visible == 0.0f : fabsf(visible - expected) <= 1e-6f * expected;
        if (!ok)
        {
            ++tally[1];
        }
        tally[2] += !blocked && expected < 1.0f ? 1 : 0; // unblocked through the smoke: T itself is checked
    }
};

GPU_TEST(media, visibility)
{
    gpu::scene_desc s;
    const int smoke = add_material(s, gpu::material_kind::isotropic, {0.9, 0.9, 0.9});
    const int matte = add_material(s, gpu::material_kind::lambertian, {0.5, 0.5, 0.5});
    const gpu::quad_desc wall{.q = {3, -3, -3}, .u = {0, 6, 0}, .v = {0, 0, 6}, .material = matte};
    s.quads.push_back(wall);
    s.media.push_back(sphere_medium({0, 0, 0}, smoke));
    const gpu_test::device_scene scene(s);
    gpu::quality_view v{};
    v.base = scene.view();
    v.microfacets = nullptr;
    v.lamps = {};
    v.environment = {};
    v.light_pmf = 0;
    v.combination = light_combination::power_heuristic;
    const auto t = device_tally<3>(t5v_ray{v, gpu::make_quad<float>(wall), scene.quad_ids()[0]}, 7751, 10'000);
    ctx.count("visibility", t[1], 0, t[0]);
    ctx.info("visibility-through-smoke", std::to_string(t[2]) + " of " + std::to_string(t[0]) +
                                             " rays unblocked through the smoke");
}

// --- T6m: the integrator through smoke -------------------------------------------------------------------------------

// T6's toy scene with a box of smoke between the lamp and the floor, of density 1.5 (a ray across it keeps about a
// fifth of its light): particles take light samples, their shadow rays cross the smoke, and with a black smoke the
// paths that enter it end at zero throughput under nee and mis while -p material's go on carrying nothing
gpu::scene_desc toy_scene_with_smoke(int depth, light_combination c, double albedo, int size, int sqrt_spp)
{
    gpu::scene_desc s = toy_scene(depth, c, size, sqrt_spp);
    const int smoke = add_material(s, gpu::material_kind::isotropic, {albedo, albedo, albedo});
    s.media.push_back({.boundary = gpu::boundary_shape::convex_quads,
                       .quads = box_quads({-0.5, 0.2, -0.6}, {0.5, 1.2, 0.4}),
                       .neg_inv_density = -1 / 1.5,
                       .material = smoke});
    return s;
}

// T6a, T6c and T6d through trace_quality_ray with media: nee and mis against two material renders of their own, the
// counting rule (one shadow ray per continuation: none from a black particle, whose path ends), and the same rays per
// sample as the material's in the grey smoke
GPU_TEST(media, integrator)
{
    const bool quick = ctx.quick();
    const int size = quick ? 32 : 64;
    const int sqrt_spp = quick ? 2 : 16;
    const std::int64_t camera_rays = static_cast<std::int64_t>(size) * size * sqrt_spp * sqrt_spp;
    struct smoke_case
    {
        double albedo;
        const char* name;
    };
    const smoke_case smokes[] = {{0.9, "grey"}, {0.0, "black"}};
    std::uint64_t seed = 7760;
    std::int64_t not_finite = 0;
    std::int64_t values = 0;
    std::int64_t material_shadow_rays = 0;
    for (const smoke_case& smoke : smokes)
    {
        for (int depth = 2; depth <= 3; ++depth)
        {
            const auto render = [&](light_combination c) {
                return gpu::render(toy_scene_with_smoke(depth, c, smoke.albedo, size, sqrt_spp),
                                   {.seed = seed++, .quality_kernel = true});
            };
            const gpu::render_result material[2] = {render(light_combination::bsdf_only),
                                                    render(light_combination::bsdf_only)};
            const gpu::render_result sampled[2] = {render(light_combination::lights_only),
                                                   render(light_combination::power_heuristic)};
            for (const gpu::render_result* image : {&material[0], &material[1], &sampled[0], &sampled[1]})
            {
                for (const double x : image->pixels)
                {
                    not_finite += std::isfinite(x) ? 0 : 1;
                }
                values += static_cast<std::int64_t>(image->pixels.size());
            }
            material_shadow_rays += static_cast<std::int64_t>(material[0].shadow_rays + material[1].shadow_rays);
            const std::string tag = std::string(smoke.name) + "-d" + std::to_string(depth);
            for (int k = 0; k < 2; ++k)
            {
                const gpu::render_result& image = sampled[k];
                const std::string mode = mode_name(k == 0 ? light_combination::lights_only
                                                          : light_combination::power_heuristic);
                const gpu_test::mean_estimate diff = pixel_difference(material[k], image);
                ctx.mean("t6m-" + mode + "-" + tag, diff.mean, diff.standard_error, 0);
                const auto rays = static_cast<std::int64_t>(image.rays);
                const auto shadow = static_cast<std::int64_t>(image.shadow_rays);
                const std::int64_t continuations = rays - camera_rays;
                ctx.count("t6m-counting-" + tag + "-" + mode,
                          shadow > continuations ? shadow - continuations : continuations - shadow,
                          std::max<std::int64_t>(2, static_cast<std::int64_t>(1e-4 * static_cast<double>(camera_rays))),
                          camera_rays);
                const double ratio = static_cast<double>(image.rays) / static_cast<double>(material[k].rays);
                if (smoke.albedo > 0 && !quick)
                {
                    ctx.exact("t6m-rays-" + tag + "-" + mode, ratio, 1, 0.002);
                }
                else
                {
                    ctx.info("t6m-rays-" + tag + "-" + mode, "rays per sample / material's = " + number(ratio));
                }
            }
        }
    }
    ctx.count("t6m-finite", not_finite, 0, values);
    ctx.exact("t6m-counting-material", static_cast<double>(material_shadow_rays), 0, 0);
}
}
