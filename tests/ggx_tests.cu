// Suites ggx and ggx-energy: the GGX rough conductor of ggx.cuh and its two lobes in bsdf.cuh, conductor and
// smooth_conductor (Step 78), its energy compensation (Step 79), and the GGX lobes' weights in multiple importance
// sampling under the lamps (Step 81). A rough metal reflects off microfacets whose
// normals h follow the GGX distribution D (Walter et al. 2007), masked by the height-correlated Smith term G2 (Heitz
// 2014); ggx_sample draws the visible normals (Heitz 2018) by spherical caps (Dupuy & Benyoub 2023). The quality
// kernel's multiple importance sampling weighs ggx_sample's density against ggx_eval's for the same direction, so the
// two must agree to the digits float allows, and both must be the GGX model, not merely each other's match. The spec
// is out/m4/spec/B.md, sections 5.1 and 5.2, whose numbers (G1-G17, E1-E7) the checks cite.
//
//   - Integrals by deterministic quadrature on the GPU, summed in double, over the half vector with
//     θh = atan(α·tan(πt/2)), t and φ at 2048 × 512 midpoints: with it D(h) cos θh dωh is exactly ¼ sin(πt) dt dφ for
//     every α, so the narrowest lobe is integrated as well as the widest. ∫ D cos θh dωh = 1 (G2), the weak white
//     furnace ∫ G1(v) max(0, v·h) D(h) / (n·v) dωh = 1 (G1, G13), the single-scattering albedo E_ss(μ, α) with F = 1
//     against ggx.md's independent table (G8) and its limit α → 0 (G9); a direction l = reflect(-v, h) weighs
//     dωl = 4 (v·h) dωh.
//   - The draws by Pearson's χ² test: l on the upper hemisphere plus one cell for the draws below it (α ≥ 0.1), and the
//     half vector unit(v + l) in stretched slope space (α ≤ 0.05), where a narrow lobe spreads over the whole grid
//     (G3); the planted fault, α = 0.31 tested against the density of 0.30, must be rejected (G16).
//   - Every draw: weight = eval/pdf and the drawn pdf = ggx_pdf (G4); finite, unit, 0 ≤ weight ≤ F (G10); exactly two
//     random numbers (G11). The share of false returns plus ∫ pdf over the upper hemisphere is 1 (G5); E_ss as the
//     mean weight equals E_ss by quadrature (G7); f(v, l) = f(l, v) (G6).
//   - The lobes: make_bsdf's choice between the smooth and the rough conductor at min_ggx_alpha, the smooth mirror with
//     Schlick's F (G12), the rough lobe = ggx.cuh's functions bit for bit (G14), validate's refusals (G15).
//   - The formulas of B.md §2.3 at random arguments against double (the formula-* checks, not numbered in the spec):
//     they pin the α / α² conventions of ggx_d, ggx_g1, ggx_g2 and smith_a, which the integrals can't see (a D of
//     width α² instead of α still integrates to 1).
//   - Under the lamps (G18, Step 81): Veach's conditions on the weights of a light sample and of the lobe's own sample,
//     the conductor and the gloss at a vertex lit by cornell's lamp or light's two: w_L + w_B = 1 wherever both
//     techniques can draw the direction (W1), and a technique that can't draw it has density 0 (W2), under nee and
//     mis. The conductor's two densities of one direction, ggx_sample's and ggx_pdf's, meet in the weights there; a
//     density 5% too wide must be rejected. (The gloss's sample takes its density from gloss_eval itself, so W1 can't
//     see an error in it: the gloss suite's χ², masses and pdf checks do.)
//   - ggx-energy: single scattering loses energy to the light that bounces between microfacets, up to 70% at α = 1.
//     Turquin et al. 2019 scale f by 1 + F0 (1 - E)/E, E = E_ss(μ, r) with F = 1, read from a 32 × 32 table that this
//     suite generates (--write FILE, the file ggx_energy.cuh) by the same quadrature and checks (--check FILE, or the
//     compiled-in table: E1, E2, E6). With F0 = 1 the compensated conductor reflects everything: E_ms = 1 (E3), which
//     a table shifted by one node must fail (E7); the factor itself (E4) and the lobe's weight and pdf (E5).
//
// Decisions where the spec leaves room (B.md §8 has the spec writer's; these are this file's):
//   1. --quick: the quadrature verdicts B.md skips (G1, G2, G8, G9, G13, the resolution check) become INFO lines; the
//      bound "E_ss ≤ 1 + 10⁻⁶" keeps only the mean weights (a 256 × 64 quadrature may round above 1).
//   2. The slope χ² puts a draw whose unit(v + l) has v·h ≤ 0 in the overflow cell. Only rounding makes one (at the
//      silhouette of the visible normals, where v + l is tiny and its direction loose), but a cell of probability 0
//      lies right beside the silhouette in three cases (α = 0.01 at 89°, 0.05 at 80° and 89°), and one such draw there
//      would be an "impossible" sample, p = 0.
//   3. The l grid's cell masses: the order-8 Gauss rule (gpu_test.cuh) on 2 × 2 sub-cells of every cell.
//   4. The checks the spec names once (ggx/finite, ggx/draws, ggx/smooth, ggx/energy-limit) and G7's bound "both ≤
//      1 + 10⁻⁶" are counts over all cases; the per-case checks are named <α>-<θ> (θ in degrees).
//   5. G10's F is Schlick's at v·h with h = unit(v + l) recomputed in double from the drawn l.
//   6. A false return must leave l as (0, 0, 0) or the reflected direction: l is preset to NaN before every call, so a
//      return that doesn't set it counts against G10 instead of passing as a draw below the horizon.
//   7. G12's boundary is min_ggx_alpha as ggx.cuh defines it (just below: smooth; at it: rough), which G13's remedy may
//      raise; its value is printed. The side entry kind none behaves as no side entry (B.md §2.2): checked too.
//   8. G15 also renders scenes validate must accept (a rough conductor, r = 0, r = 1, a side entry none), so that a
//      validate refusing everything can't pass; the refusals render 1×1 pixel, one sphere.
//   9. The resolution check (1024 × 256 against 2048 × 512) covers G1 and G2 within 10⁻⁵ (B.md §5.1) for α ≥ 0.01,
//      whose integrands are smooth in (t, φ), and E_ss within 2·10⁻⁴, a tenth of G8's tolerance: its integrand,
//      as ∫ pdf's, stops at the horizon, where the midpoint error is O(1/N) (measured 9.5·10⁻⁵); α = 10⁻³ is INFO.
//  10. G17's patch averages: E_ss(μ) over the patch's area, μ from the camera's own rays (B.md §6.2), E_ss
//      interpolated linearly between quadratures at μ = 0.96, 0.965, ..., 1.
//  11. G18 (Step 81's T3 case, written once the lamps existed): decisions 18-23.
//  12. G4's tolerances and G10's F follow the half vector h' = unit(v + l) that ggx_eval and ggx_pdf recompute from the
//      drawn l. B.md §8.4 estimated its rounding at v·h ≈ 1; v + l has the length 2 v·h and carries l's rounding, so
//      h' is off the drawn h by about 10⁻⁷/(v·h): 60 times more at θv = 89°, where a narrow lobe has v·h ≈ 0.017 (a
//      correct sampler failed B.md's flat tolerances there, by estimate). So the pdf's τ(α) = max(10⁻⁵, 10⁻⁶/α)
//      becomes max(10⁻⁵, 10⁻⁶/(α v·h')), the same at v·h = 1, and the weight's 10⁻⁵ gains what F(v·h') may differ
//      from F(v·h): Schlick's d ln F/dμ times 5·10⁻⁷/(v·h') (D cancels in eval/pdf). E5's weight check too.
//  13. The table file (B.md §5.2.1) writes each value as {:.9g} with ".0" added to a number without a point or an
//      exponent: "{:.9g}f" turns row 0's 1 into "1f", which is no C++ literal. A row's comment has a line of its own
//      and its 32 values come 8 per line (120 columns).
//  14. --check reads the literals between ggx_energy_table's "= {" and its "}", with the comments left out: B.md's
//      "after the first {" would start at namespace gpu's brace, and the comments hold numbers (0.001, 0/31). Not 1024
//      of them, or anything else in the list, is an error (exit 2). A relative FILE that isn't there is looked for from
//      the current folder up (find_repo_file), so src\gpu\ggx_energy.cuh works from the build folder; --write's folder
//      likewise.
//  15. --write writes the regenerated table and runs E2-E7 (E1 has nothing to compare with yet): E3 and E5 still test
//      the compiled-in table, so the first --write over a placeholder reports E3's failures until the rebuild. Its INFO
//      line gives the largest difference from the compiled-in table (0: no rebuild needed). --check also prints that
//      difference between the compiled-in table and FILE, and E6 checks both.
//  16. E7 shifts the compiled-in table (the one E3 used) and computes E_ss/E' from the quadrature of the
//      uncompensated conductor; "rejected" is a count of the fault accepted (max |E_ms - 1| ≤ 10⁻² over θv ≤ 80°).
//  17. E4 adds a check not in B.md's list: ggx_compensation equals B.md §3.8's bilinear lookup of the compiled-in
//      table, 1 + F0 (1 - E)/E, to 10⁻⁵ (the review's "the table's axes and bilinear lookup"), at 65 × 65 points and
//      the 32 × 32 nodes. E5 draws 10⁵ samples in each case of the §5.1 grid (B.md gives no count).
//  18. G18's cases go beyond B.md's two (cornell's lamp; the conductor at α = 0.09 over aluminium's F0, the gloss at
//      α = 0.04 over ρ = 0.73; mis): both setups of the lamps suite's T3, cornell's lamp (pmf 1) and light's quad and
//      sphere (pmf ½), × the conductor (aluminium) at the ggx grid's α and 0.09 × the gloss (ρ = 0.73) at the gloss
//      suite's α and 0.04 × nee and mis, from 10⁶ light-drawn and 10⁶ lobe-drawn directions per case (one draw for both
//      combinations: only the weights differ). B.md's ggx/t3-cornell-conductor and ggx/t3-cornell-gloss are the checks
//      t3-cornell-conductor-0.09-mis and t3-cornell-gloss-0.04-mis. An α below min_ggx_alpha (should G13's remedy
//      raise it) makes the delta mirror, with nothing to weigh: INFO. t3-<setup>-lamps checks build_lamps' count.
//  19. The points: B.md's box [100, 450] × [50, 400] × [100, 450] under cornell's lamp, D.md §6's P_light under light's
//      (behind its quad too, where a lobe's ray finds the quad's back). The normal faces the centre of a lamp picked
//      uniformly; wo is uniform above n at half the points, and at the other half aimed: the mirror image about n of
//      the direction to a uniform point of that lamp (when above n), so that the lobe's peak meets the lamp. With wo
//      uniform alone a narrow lobe would be tested almost only where p_L ≫ p_B or p_B ≫ p_L, where the sum of the
//      weights can't see a density's error (its slope 2r²/(1 + r²)² in ln r, r = p_L/p_B, vanishes there).
//  20. Under mis |Σw − 1| ≤ 10⁻⁵ + τ with decision 12's τ = max(10⁻⁵, 10⁻⁶/(α v·h')), h' = unit(wo + wi), for both
//      techniques and both lobes: B.md's τ(α) at v·h' = 1. The sum moves by at most ½|ln(p/p')| (plus rounding) when
//      one technique's density enters it as p and p', so G4's bound on ggx_sample's against ggx_pdf's implies it.
//      Under nee Σw = 1 exactly. A technique's weight counts only where its density is positive (W2): for a
//      light-drawn direction Σw = w_L + [p_B > 0]·w_B, so one the lobe can't draw (p_B = 0 with f > 0, counted INFO)
//      needs w_L = 1 alone. A lobe-drawn ray at a lamp's front must find p_L′ > 0, finite (the light sampler draws
//      every point of a front in view; without its density the light would count twice): a violation in both modes.
//  21. W2's checks, t3-<setup>-<lobe>-w2 over every α and both techniques: a drawn direction has a positive, finite
//      density in its technique (the light sample's pdf; the lobe's, which isn't delta); a light-drawn direction more
//      than 10⁻⁶ below the lobe's horizon gets f = 0 and density 0 from eval_bsdf (the lobe's draws there are
//      absorbed); a lobe-drawn ray at the back of light's quad gets lamp_hit_pdf 0, light_weight 0 under mis and
//      bsdf_weight 1 under both. Non-finite f or densities count here too.
//  22. A case whose light-drawn or lobe-drawn directions tested nothing counts one violation: a check of nothing
//      mustn't pass. Rays that find another lamp first or slip past the sampled lamp's edge are left out (the lamps
//      suite's T1 tests them).
//  23. The must-fail t3-planted-wider-pdf: cornell's conductor at α = 0.3, lobe-drawn, mis, w_L from ggx_pdf at 1.05α
//      instead of eval_bsdf's density; rejected iff some direction has |Σw − 1| > 10⁻³ (50 times the tolerance), a
//      count as the lamps suite's X1.
//
// Step 78's criteria (B.md §7; Claude runs them; $S = -w 480 --height 270):
//   $GT ggx: exit 0; it prints E_ss(1, α) and the patch averages (ggx.furnace-values-*).
//   $G ggx-furnace $S -s 256 --seed 3 -o gf.pfm, then
//     $PS stats gf.pfm --rect 0 0 479 109 --rect 0 161 479 269 --equal 1   and   --rect 66 133 70 137 --equal 1:
//       at least 99.99% of the values exactly 1.0;
//     $PS stats gf.pfm --max 1.000001: none above;
//     each GGX patch, --rect 152 133 156 137 (α 0.05), 238 133 242 137 (0.3), 323 133 327 137 (0.6),
//       409 133 413 137 (1): mean (channel average) = E_ss(1, α) ± (4 SE + 0.003), and this suite's E_ss(1, α)
//       within 2·10⁻³ of B.md §4 (G8);
//     -p mis: the same bytes (no lamps, no map: the fallback to material).
//   HOST: ggx-furnace under "Test scenes:"; errors for ggx-furnace --fp64, -p mixture, --all-books-kernel (exit 1, one
//   Error: line). Sanitizer: ggx-furnace and $GT ggx --quick. CARRY recorded: ggx-furnace $S -s 64 --seed 3.
// Step 79's (B.md §7):
//   $GT ggx-energy --write src\gpu\ggx_energy.cuh once (over the placeholder), rebuild, then
//   $GT ggx-energy --check src\gpu\ggx_energy.cuh: exit 0 (and $GT ggx: G14 now with the factor).
//   ggx-furnace $S -s 256 --seed 3: every GGX patch mean = 1 ± (4 SE + 0.005); the mirror patch and both background
//   rectangles byte-identical to Step 78's render: $PS same W\78\gf.pfm W\79\gf.pfm --rect 66 133 70 137
//   --rect 0 0 479 109 --rect 0 161 479 269. CARRY: the lamp scenes identical; ggx-furnace re-recorded.
//   Sanitizer: ggx-furnace and $GT ggx-energy --quick.
// Step 81's (B.md §7):
//   $GT ggx gloss: exit 0 (G18: Σw = 1 to 10⁻⁵ + τ with these lobes under cornell's and light's lamps; its counts keep
//   their verdicts under --quick).
//   compare.ps1 -Scene cornell-ggx -A material:1,2 -B mis:3,4 -Spp 1024: tests, controls and the RMS rule pass;
//   σ_mis < σ_material ($PS noise, trimmed).
//   veach $S -s 256 -p mis, seeds 3 and 4: the control exits 0, $PS stats --finite; no cross-strategy test on veach.
//   Help: "Scenes beyond the books:" lists cornell-ggx and veach (allowed diff); Tab unchanged until Step 84.
//   Sanitizer: cornell-ggx, veach (and $GT ggx --quick, G18 included). CARRY recorded: cornell-ggx and veach -p mis $S
//   -s 64 --seed 3.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "bsdf.cuh"
#include "bvh.cuh"
#include "ggx.cuh"
#include "ggx_energy.cuh"
#include "gpu_test.cuh"
#include "gpu_test.h"
#include "host_services.h"
#include "lamps.cuh"
#include "light_sampler.cuh"
#include "material.cuh"
#include "mis.cuh"
#include "quality_view.cuh"
#include "random.cuh"
#include "renderer.h"
#include "statistics.h"
#include "test_scene.h"
#include "tracer.cuh"
#include "vec3.cuh"

namespace
{
using gpu::vec3;
using gpu_test::direction_grid;

constexpr double pi = std::numbers::pi;

// --- The cases (B.md §5.1) -------------------------------------------------------------------------------------------

constexpr std::int64_t case_samples = 1'000'000; // per case: χ², weights, densities, false returns, E_ss
constexpr std::int64_t pair_samples = 100'000;   // reciprocity: pairs (v, l) per α
constexpr std::int64_t random_cases = 10'000;    // the formulas' and the lobes' random arguments

constexpr std::array<double, 7> grid_alphas{1e-3, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0};
constexpr std::array<int, 5> grid_thetas{0, 30, 60, 80, 89};
constexpr std::size_t grid_size = grid_alphas.size() * grid_thetas.size();

// ggx.md's E_ss(θv, α) with F = 1 (B.md §4), at θv = 0°, 45°, 80°, 89°
constexpr std::array<double, 4> table_alphas{0.05, 0.3, 0.6, 1.0};
constexpr std::array<int, 4> table_thetas{0, 45, 80, 89};
constexpr double energy_table[4][4] = {{0.9973, 0.9958, 0.9532, 0.9358},
                                       {0.8771, 0.8457, 0.8485, 0.9702},
                                       {0.5906, 0.6084, 0.7784, 0.9540},
                                       {0.3069, 0.3770, 0.6681, 0.9290}};

// Seeds: a case's draws (every check about them sees the same samples), its E_ss by the mean weight, the rest
constexpr std::uint64_t draw_seed = 7800;        // + case index, 0 to 34
constexpr std::uint64_t energy_seed = 7840;      // + case index
constexpr std::uint64_t reciprocity_seed = 7875; // + α index, 0 to 6
constexpr std::uint64_t smooth_seed = 7882;      // + 0, 1
constexpr std::uint64_t make_seed = 7884;
constexpr std::uint64_t lobe_seed = 7885;        // + 0, 1, 2
constexpr std::uint64_t formula_seed = 7888;     // + the formula, 0 to 7
constexpr std::uint64_t planted_seed = 7896;
constexpr std::uint64_t validate_seed = 7897;    // the renders' seed

// F0 of the sampling checks, and measured aluminium (Epic's table, B.md §4) for the lobes'
__host__ __device__ inline vec3<float> test_f0() { return {1.0f, 0.5f, 0.2f}; }
__host__ __device__ inline vec3<float> aluminium() { return {0.913f, 0.921f, 0.925f}; }
__host__ __device__ inline vec3<float> white() { return {1.0f, 1.0f, 1.0f}; }

std::string number(double x) { return std::format("{}", x); } // 0.001, 0.05, 1: the names' α
std::string case_name(double alpha, int theta) { return std::format("{}-{}", number(alpha), theta); }
// The grid's case k: α index k / 5, θ index k % 5
std::string grid_name(std::size_t k)
{
    return case_name(grid_alphas[k / grid_thetas.size()], grid_thetas[k % grid_thetas.size()]);
}
double cos_degrees(int theta) { return theta == 0 ? 1.0 : std::cos(theta * pi / 180); }

// --- Vectors ---------------------------------------------------------------------------------------------------------

__host__ __device__ inline vec3<double> to_double(const vec3<float>& v) { return {v.x, v.y, v.z}; }
__host__ __device__ inline vec3<float> to_float(const vec3<double>& v)
{
    return {static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)};
}
__host__ __device__ inline vec3<double> cross_d(const vec3<double>& a, const vec3<double>& b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__host__ __device__ inline vec3<double> normalized(const vec3<double>& v) { return (1 / ::sqrt(gpu::dot(v, v))) * v; }
gpu::vec3d to_vec3d(const vec3<double>& v) { return {v.x, v.y, v.z}; }

__device__ inline float quiet_nan() { return __int_as_float(0x7fc00000); }
__device__ inline vec3<float> nan_vector() { return {quiet_nan(), quiet_nan(), quiet_nan()}; }
__device__ inline bool finite(const vec3<float>& v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z); }
__device__ inline bool is_zero(const vec3<float>& v) { return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f; }
__device__ inline bool bits_equal(float a, float b) { return __float_as_uint(a) == __float_as_uint(b); }
__device__ inline bool bits_equal(const vec3<float>& a, const vec3<float>& b)
{
    return bits_equal(a.x, b.x) && bits_equal(a.y, b.y) && bits_equal(a.z, b.z);
}
__host__ __device__ inline float channel(const vec3<float>& c, int i) { return i == 0 ? c.x : i == 1 ? c.y : c.z; }
// |a - b| ≤ rel·|b| per channel (a NaN fails)
__device__ inline bool close(const vec3<float>& a, const vec3<float>& b, double rel)
{
    for (int i = 0; i < 3; ++i)
    {
        const double x = channel(a, i);
        const double y = channel(b, i);
        if (!(::fabs(x - y) <= rel * ::fabs(y)))
        {
            return false;
        }
    }
    return true;
}
__device__ inline bool close(double a, double b, double rel, double abs = 0)
{
    return ::fabs(a - b) <= rel * ::fabs(b) + abs;
}

// The normal's frame of B.md §5.1: n = unit(0.3, -0.5, 0.8), t1 = unit(n × z), t2 = n × t1
struct frame
{
    vec3<double> n, t1, t2;
};

frame shading_frame()
{
    const vec3<double> n = normalized({0.3, -0.5, 0.8});
    const vec3<double> t1 = normalized(cross_d(n, {0, 0, 1}));
    return {n, t1, cross_d(n, t1)};
}

// --- References in double, from the float arguments (B.md §2.3, §3) --------------------------------------------------

__host__ __device__ inline double smith_a_ref(double c, double alpha2) { return ::sqrt(c * c * (1 - alpha2) + alpha2); }
__host__ __device__ inline double schlick_ref(double f0, double mu)
{
    const double x = ::fmax(0.0, 1 - mu);
    return f0 + (1 - f0) * (x * x * x * x * x);
}
__host__ __device__ inline double ggx_d_ref(const vec3<double>& n, const vec3<double>& h, double alpha2)
{
    const double c = gpu::dot(n, h);
    if (!(c > 0))
    {
        return 0;
    }
    const vec3<double> s = cross_d(n, h);
    const double q = alpha2 * c * c + gpu::dot(s, s);
    return alpha2 / (pi * q * q);
}
// f·cos θl of the conductor and the density of l, or false where ggx_eval returns 0 and 0
struct eval_ref
{
    double f[3];
    double pdf;
};
__device__ inline bool ggx_eval_ref(const vec3<float>& nf, const vec3<float>& vf, const vec3<float>& lf, float alpha,
                                    const vec3<float>& f0, eval_ref& out)
{
    const vec3<double> n = to_double(nf);
    const vec3<double> v = to_double(vf);
    const vec3<double> l = to_double(lf);
    const double cv = gpu::dot(n, v);
    const double cl = gpu::dot(n, l);
    const vec3<double> s = v + l;
    if (!(cv > 0 && cl > 0 && gpu::dot(s, s) > 0))
    {
        return false;
    }
    const vec3<double> h = normalized(s);
    const double a2 = static_cast<double>(alpha) * alpha;
    const double d = ggx_d_ref(n, h, a2);
    const double av = smith_a_ref(cv, a2);
    const double al = smith_a_ref(cl, a2);
    const double vh = gpu::dot(v, h);
    for (int i = 0; i < 3; ++i)
    {
        out.f[i] = schlick_ref(channel(f0, i), vh) * d * cl / (2 * (cl * av + cv * al));
    }
    out.pdf = d / (2 * (cv + av));
    return true;
}

// --- The quadrature over the half vector (B.md §5.1) -----------------------------------------------------------------

// One integration's frame, view direction and width; the float members are what the functions get
struct quad_case
{
    vec3<double> n, t1, t2; // orthonormal, n the normal
    vec3<double> v;         // unit
    vec3<float> nf, vf;
    float alpha;
    float roughness; // r = √α
};

quad_case make_case(double alpha, double mu)
{
    const frame f = shading_frame();
    const vec3<double> v = mu * f.n + std::sqrt(std::max(0.0, 1 - mu * mu)) * f.t1; // μ = 1: v = n exactly
    quad_case k;
    k.n = f.n;
    k.t1 = f.t1;
    k.t2 = f.t2;
    k.v = v;
    k.nf = to_float(f.n);
    k.vf = to_float(v);
    k.alpha = static_cast<float>(alpha);
    k.roughness = static_cast<float>(std::sqrt(alpha));
    return k;
}

// A node: h in float, l = reflect(-v, h) in float (or 0 where v·h ≤ 0), and the node's solid angles
struct h_node
{
    vec3<float> h;
    vec3<float> l;
    double ch;        // n·h
    double vh;        // v·h
    double d_omega_h; // sin θh (dθh/dt) Δt Δφ
    double d_omega_l; // 4 (v·h) dωh where v·h > 0, else 0
};

constexpr int quad_values = 4; // integrals per pass
using integrals = std::array<double, quad_values>;

struct resolution
{
    int rows;    // t
    int columns; // φ
};
constexpr resolution full_resolution{2048, 512};
constexpr resolution half_resolution{1024, 256};
constexpr resolution quick_resolution{256, 64};

// One thread per row t of one case (blockIdx.y), its φ in order: each row's sums land in sums, the host adds the rows
// in order, so the result is the same on every run
template <class F>
__global__ void h_quadrature_kernel(F f, const quad_case* cases, int rows, int columns, double* sums)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
    {
        return;
    }
    const quad_case k = cases[blockIdx.y];
    const double alpha = k.alpha;
    const double tangent = ::tan(0.5 * pi * (row + 0.5) / rows); // tan(πt/2)
    const double at = alpha * tangent;
    const double cos_h = 1 / ::sqrt(1 + at * at);
    const double sin_h = at * cos_h;
    const double dtheta_dt = alpha * 0.5 * pi * (1 + tangent * tangent) / (1 + at * at);
    const double d_phi = 2 * pi / columns;
    const double d_omega_h = sin_h * dtheta_dt * d_phi / rows;
    // cos φ and sin φ at the midpoints by rotation, (k + ½)Δφ
    const double step_cos = ::cos(d_phi);
    const double step_sin = ::sin(d_phi);
    double phi_cos = ::cos(0.5 * d_phi);
    double phi_sin = ::sin(0.5 * d_phi);
    double sum[quad_values] = {};
    for (int column = 0; column < columns; ++column)
    {
        const vec3<double> h = (sin_h * phi_cos) * k.t1 + (sin_h * phi_sin) * k.t2 + cos_h * k.n;
        h_node node;
        node.h = to_float(h);
        node.ch = cos_h;
        node.vh = gpu::dot(k.v, h);
        node.d_omega_h = d_omega_h;
        if (node.vh > 0)
        {
            node.l = to_float((2 * node.vh) * h - k.v);
            node.d_omega_l = 4 * node.vh * d_omega_h;
        }
        else
        {
            node.l = {};
            node.d_omega_l = 0;
        }
        f(k, node, sum);
        const double next_cos = phi_cos * step_cos - phi_sin * step_sin;
        phi_sin = phi_sin * step_cos + phi_cos * step_sin;
        phi_cos = next_cos;
    }
    for (int i = 0; i < quad_values; ++i)
    {
        sums[(static_cast<std::size_t>(blockIdx.y) * rows + row) * quad_values + i] = sum[i];
    }
}

// The integrals of every case, in launches of about 16 million nodes (a few milliseconds each)
template <class F>
std::vector<integrals> h_quadrature(const F& f, const std::vector<quad_case>& cases, resolution r)
{
    std::vector<integrals> out(cases.size(), integrals{});
    const std::size_t nodes = static_cast<std::size_t>(r.rows) * static_cast<std::size_t>(r.columns);
    const std::size_t batch = std::max<std::size_t>(1, (std::size_t{1} << 24) / nodes);
    for (std::size_t first = 0; first < cases.size(); first += batch)
    {
        const std::size_t count = std::min(batch, cases.size() - first);
        const auto begin = cases.begin() + static_cast<std::ptrdiff_t>(first);
        const gpu_test::device_buffer<quad_case> on_gpu(
            std::vector<quad_case>(begin, begin + static_cast<std::ptrdiff_t>(count)));
        const gpu_test::device_buffer<double> sums(count * static_cast<std::size_t>(r.rows) * quad_values);
        const dim3 blocks(static_cast<unsigned>((r.rows + 127) / 128), static_cast<unsigned>(count));
        h_quadrature_kernel<<<blocks, 128>>>(f, on_gpu.get(), r.rows, r.columns, sums.get());
        gpu_test::finish_launch("an h quadrature");
        const std::vector<double> s = sums.download();
        for (std::size_t c = 0; c < count; ++c)
        {
            for (std::size_t row = 0; row < static_cast<std::size_t>(r.rows); ++row)
            {
                for (std::size_t i = 0; i < quad_values; ++i)
                {
                    out[first + c][i] += s[(c * static_cast<std::size_t>(r.rows) + row) * quad_values + i];
                }
            }
        }
    }
    return out;
}

// [0] the weak white furnace (G1), [1] ∫ D cos θh (G2), [2] ∫ pdf over the upper hemisphere (G5), [3] E_ss, F = 1
struct conductor_integrands
{
    __device__ void operator()(const quad_case& k, const h_node& node, double (&sum)[quad_values]) const
    {
        const float alpha2 = k.alpha * k.alpha;
        const float cv = gpu::dot(k.nf, k.vf);
        const double d = gpu::ggx_d(k.nf, node.h, alpha2);
        sum[0] += static_cast<double>(gpu::ggx_g1(cv, alpha2)) * ::fmax(0.0, node.vh) * d / cv * node.d_omega_h;
        sum[1] += d * node.ch * node.d_omega_h;
        if (node.d_omega_l > 0)
        {
            sum[2] += static_cast<double>(gpu::ggx_pdf(k.nf, k.vf, node.l, k.alpha)) * node.d_omega_l;
            float pdf = 0;
            sum[3] += static_cast<double>(gpu::ggx_eval(k.nf, k.vf, node.l, k.alpha, white(), pdf).y) * node.d_omega_l;
        }
    }
};

// --- The draws -------------------------------------------------------------------------------------------------------

// One call of ggx_sample, with l, the weight and the pdf preset to NaN (decision 6)
struct ggx_draw
{
    vec3<float> n, v;
    float alpha;
    vec3<float> f0;
    __device__ bool operator()(gpu::pcg32& rng, vec3<float>& l, vec3<float>& weight, float& pdf) const
    {
        l = nan_vector();
        weight = nan_vector();
        pdf = quiet_nan();
        return gpu::ggx_sample(n, v, alpha, f0, rng, l, weight, pdf);
    }
};

// A false return as the checks expect it: a finite direction (the reflection, below the horizon or with v·h ≤ 0). B.md
// §2.3 lets a degenerate half vector return l = (0, 0, 0), but G10 and G3 require none ("no l = 0")
__device__ inline bool sound_false_return(const vec3<float>& l) { return finite(l) && !is_zero(l); }

// The half vector that ggx_eval and ggx_pdf recompute from a drawn l, h' = unit(v + l), and v·h' (decision 12). v + l
// has the length 2 v·h, and carries l's rounding (about 10⁻⁷ per component): h' is off the drawn h by about
// 10⁻⁷/(v·h) radians, 60 times B.md's estimate at θv = 89°, where a narrow lobe has v·h ≈ cos 89° = 0.017.
struct recomputed_half
{
    double vh;    // v·h' (= |v + l|/2 for a reflection)
    double error; // a bound on |v·h' - v·h| and on the angle between h' and h: 5·10⁻⁷/(v·h), five times the estimate
};

__device__ inline recomputed_half recompute_half(const vec3<float>& vf, const vec3<float>& lf)
{
    const vec3<double> v = to_double(vf);
    const vec3<double> s = v + to_double(lf);
    const double vh = gpu::dot(v, normalized(s));
    return {vh, 5e-7 / vh};
}

// d ln F / dμ of Schlick's F in size: 5(1 - F0)(1 - μ)⁴ / F
__device__ inline double schlick_slope(double f0, double mu)
{
    const double x = ::fmax(0.0, 1 - mu);
    return 5 * (1 - f0) * (x * x * x * x) / schlick_ref(f0, mu);
}

// A true return as it must be: l unit, the pdf finite and > 0, 0 ≤ weight ≤ F(v·h) + 10⁻⁶ per channel (G10), F at the
// recomputed h' with the slack of its error
__device__ inline bool sound_true_return(const ggx_draw& d, const vec3<float>& l, const vec3<float>& w, float pdf)
{
    if (!finite(l) || !finite(w) || !isfinite(pdf) || !(pdf > 0.0f))
    {
        return false;
    }
    const vec3<double> ld = to_double(l);
    if (!(::fabs(::sqrt(gpu::dot(ld, ld)) - 1) <= 1e-5))
    {
        return false;
    }
    const recomputed_half h = recompute_half(d.v, l);
    for (int i = 0; i < 3; ++i)
    {
        const double f0 = channel(d.f0, i);
        const double f = schlick_ref(f0, h.vh) * (1 + schlick_slope(f0, h.vh) * h.error);
        const double wi = channel(w, i);
        if (!(wi >= 0 && wi <= f + 1e-6))
        {
            return false;
        }
    }
    return true;
}

// weight = eval/pdf per channel, eval and pdf from ggx_eval at the drawn l (G4): within 10⁻⁵ of the weight, plus what
// F(v·h') may differ from the drawn F(v·h) (decision 12; D cancels in eval/pdf), plus what c_l may: n·l of a direction
// near the horizon is a difference of terms near 1, a few ulps of 1 apart in two copies that nvcc contracts
// differently, and the weight is nearly proportional to it there
__device__ inline bool weight_matches(const vec3<float>& n, const vec3<float>& vf, const vec3<float>& l,
                                      const vec3<float>& f0, const vec3<float>& weight, const vec3<float>& eval,
                                      float pdf)
{
    const recomputed_half h = recompute_half(vf, l);
    const double cl = ::fmax(gpu::dot(to_double(n), to_double(l)), 1e-30);
    for (int i = 0; i < 3; ++i)
    {
        const double w = channel(weight, i);
        const double tolerance = 1e-5 + schlick_slope(channel(f0, i), h.vh) * h.error + 2.4e-7 / cl;
        if (!(::fabs(w - static_cast<double>(channel(eval, i)) / pdf) <= tolerance * w))
        {
            return false;
        }
    }
    return true;
}

// The pdf tolerance: B.md's τ(α) = max(10⁻⁵, 10⁻⁶/α) at v·h ≈ 1, where D's peak turns h's rounding into a relative
// error of about 2·10⁻⁷/α; with h' off by 10⁻⁷/(v·h) it is max(10⁻⁵, 10⁻⁶/(α v·h)) (decision 12)
__host__ __device__ inline double pdf_tolerance(double alpha, double vh) { return ::fmax(1e-5, 1e-6 / (alpha * vh)); }

enum class draw_property
{
    weight_off, // G4: a true return whose weight isn't eval/pdf
    pdf_off,    // G4: a true return whose pdf isn't ggx_pdf's
    unsound,    // G10
    false_return,
};

struct draw_has
{
    ggx_draw draw;
    draw_property what;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        vec3<float> l, w;
        float pdf;
        const bool drawn = draw(rng, l, w, pdf);
        switch (what)
        {
        case draw_property::weight_off:
        {
            if (!drawn)
            {
                return false;
            }
            float eval_pdf = 0;
            const vec3<float> eval = gpu::ggx_eval(draw.n, draw.v, l, draw.alpha, draw.f0, eval_pdf);
            return !weight_matches(draw.n, draw.v, l, draw.f0, w, eval, eval_pdf);
        }
        case draw_property::pdf_off:
        {
            if (!drawn)
            {
                return false;
            }
            const double tolerance = pdf_tolerance(draw.alpha, recompute_half(draw.v, l).vh);
            const double p = gpu::ggx_pdf(draw.n, draw.v, l, draw.alpha);
            return !(::fabs(static_cast<double>(pdf) - p) <= tolerance * pdf);
        }
        case draw_property::unsound:
            return drawn ? !sound_true_return(draw, l, w, pdf) : !sound_false_return(l);
        case draw_property::false_return:
            return !drawn;
        }
        return true;
    }
};

// G11: after a call (true or false) the generator is where two next() take a copy
struct draw_count_off
{
    ggx_draw draw;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        gpu::pcg32 used = rng;
        gpu::pcg32 expected = rng;
        vec3<float> l, w;
        float pdf;
        draw(used, l, w, pdf);
        expected.next();
        expected.next();
        bool same = true;
        for (int i = 0; i < 4; ++i)
        {
            same = used.next() == expected.next() && same;
        }
        rng = used;
        return !same;
    }
};

// G7: E_ss as the mean weight with F = 1, a false return weighing 0
struct energy_of_draw
{
    ggx_draw draw; // f0 = (1, 1, 1)
    __device__ double operator()(gpu::pcg32& rng) const
    {
        vec3<float> l, w;
        float pdf;
        return draw(rng, l, w, pdf) ? static_cast<double>(w.y) : 0.0;
    }
};

// --- χ² grids --------------------------------------------------------------------------------------------------------

// The l grid: the upper hemisphere around n (the horizon its bottom border), and one cell more, index grid.cells(), for
// the false returns with a direction (below the horizon). A true return outside the grid, a NaN or a degenerate false
// return is -1: a cell of probability 0.
struct l_cell_of_draw
{
    ggx_draw draw;
    direction_grid grid;
    __device__ int operator()(gpu::pcg32& rng) const
    {
        vec3<float> l, w;
        float pdf;
        if (draw(rng, l, w, pdf))
        {
            return grid.cell(l);
        }
        return sound_false_return(l) ? grid.cells() : -1;
    }
};

struct l_density_in_grid
{
    vec3<float> n, v;
    float alpha;
    direction_grid grid;
    __device__ double operator()(double u, double w) const
    {
        return static_cast<double>(gpu::ggx_pdf(n, v, grid.direction(u, w), alpha)) * grid.jacobian();
    }
};

// The masses of a grid's cells by the order-8 rule on su × sv sub-cells of each (decision 3)
template <class D>
std::vector<double> refined_masses(const D& density, int nu, int nv, int su, int sv)
{
    const std::vector<double> fine = gpu_test::cell_masses(density, nu * su, nv * sv);
    std::vector<double> masses(static_cast<std::size_t>(nu) * nv, 0.0);
    for (int j = 0; j < nv * sv; ++j)
    {
        for (int i = 0; i < nu * su; ++i)
        {
            masses[static_cast<std::size_t>(j / sv) * nu + i / su] +=
                fine[static_cast<std::size_t>(j) * nu * su + i];
        }
    }
    return masses;
}

// The l grid's masses (G3, G16), with the cell below the horizon at the end: 1 - the grid's
std::vector<double> l_masses(const quad_case& k, const direction_grid& grid, bool quick)
{
    const int s = quick ? 1 : 2;
    std::vector<double> masses = refined_masses(l_density_in_grid{k.nf, k.vf, k.alpha, grid}, grid.nu, grid.nv, s, s);
    double sum = 0;
    for (const double m : masses)
    {
        sum += m;
    }
    masses.push_back(1 - sum);
    return masses;
}

direction_grid l_grid(const quad_case& k) { return gpu_test::grid_around(to_vec3d(k.n), 0, 1); }

// Stretched slope space (G3, α ≤ 0.05): h → (x, y) = (-h·t1, -h·t2) / (h·n) / α on 64 × 64 cells over [-4, 4]², one
// overflow cell (index 4096) after them
struct slope_space
{
    vec3<double> n, t1, t2;
    double alpha;
    static constexpr int side = 64;
    static constexpr double extent = 4;
    static constexpr int overflow = side * side;

    __device__ int cell(const vec3<double>& h) const // h of any length
    {
        const double hn = gpu::dot(h, n);
        if (!(hn > 0))
        {
            return overflow;
        }
        const double x = -gpu::dot(h, t1) / hn / alpha;
        const double y = -gpu::dot(h, t2) / hn / alpha;
        if (!(x >= -extent && x < extent && y >= -extent && y < extent))
        {
            return overflow;
        }
        const int i = ::min(side - 1, static_cast<int>((x + extent) / (2 * extent) * side));
        const int j = ::min(side - 1, static_cast<int>((y + extent) / (2 * extent) * side));
        return j * side + i;
    }
    // The unit half vector at stretched slopes (x, y)
    __device__ vec3<double> half_vector(double x, double y) const
    {
        return normalized((-alpha * x) * t1 + (-alpha * y) * t2 + n);
    }
};

struct slope_cell_of_draw
{
    ggx_draw draw;
    slope_space space;
    __device__ int operator()(gpu::pcg32& rng) const
    {
        vec3<float> l, w;
        float pdf;
        draw(rng, l, w, pdf);
        if (!sound_false_return(l))
        {
            return -1; // NaN or degenerate: impossible
        }
        const vec3<double> v = to_double(draw.v);
        const vec3<double> h = v + to_double(l);
        // At the silhouette, where the density is 0, rounding may leave v·h ≤ 0: the overflow cell (decision 2)
        return gpu::dot(v, h) > 0 ? space.cell(h) : slope_space::overflow;
    }
};

// The visible normals' density in stretched slopes: D_v(h) cos³θh α², D_v = G1(v) max(0, v·h) D(h) / (n·v)
// (Heitz 2018), on [0, 1]² (the grid's Jacobian 8²)
struct slope_density_in_grid
{
    slope_space space;
    vec3<float> nf, vf;
    float alpha;
    __device__ double operator()(double u, double w) const
    {
        const double x = (2 * u - 1) * slope_space::extent;
        const double y = (2 * w - 1) * slope_space::extent;
        const vec3<double> h = space.half_vector(x, y);
        const double cos_h = gpu::dot(h, space.n);
        const float alpha2 = alpha * alpha;
        const float cv = gpu::dot(nf, vf);
        const double visible = static_cast<double>(gpu::ggx_g1(cv, alpha2)) * ::fmax(0.0, gpu::dot(to_double(vf), h)) *
                               gpu::ggx_d(nf, to_float(h), alpha2) / cv;
        const double a = alpha;
        return visible * cos_h * cos_h * cos_h * a * a * (4 * slope_space::extent * slope_space::extent);
    }
};

std::vector<double> slope_masses(const slope_space& space, const quad_case& k)
{
    std::vector<double> masses = gpu_test::cell_masses(slope_density_in_grid{space, k.nf, k.vf, k.alpha},
                                                       slope_space::side, slope_space::side);
    double sum = 0;
    for (const double m : masses)
    {
        sum += m;
    }
    masses.push_back(1 - sum); // the overflow cell
    return masses;
}

// --- One case's draws ------------------------------------------------------------------------------------------------

struct case_draws
{
    gpu_test::chi2_outcome chi2;
    bool slope = false;
    std::int64_t drawn = 0;
    std::int64_t weight_off = 0;
    std::int64_t pdf_off = 0;
    std::int64_t unsound = 0;
    std::int64_t false_returns = 0;
    std::int64_t count_off = 0;
    gpu_test::mean_estimate energy;
};

case_draws draw_case(const gpu_test::context& ctx, const quad_case& k, std::size_t index)
{
    const std::int64_t n = ctx.samples(case_samples);
    const std::uint64_t seed = draw_seed + index;
    const ggx_draw draw{k.nf, k.vf, k.alpha, test_f0()};
    case_draws out;
    out.drawn = gpu_test::drawn(n);
    out.slope = k.alpha <= 0.05f;
    double total = 0;
    if (out.slope)
    {
        const slope_space space{k.n, k.t1, k.t2, static_cast<double>(k.alpha)};
        const auto counts =
            gpu_test::device_histogram(slope_cell_of_draw{draw, space}, seed, n, slope_space::overflow + 1);
        out.chi2 = gpu_test::chi2_against(counts, slope_masses(space, k), total);
    }
    else
    {
        const direction_grid grid = l_grid(k);
        const auto counts = gpu_test::device_histogram(l_cell_of_draw{draw, grid}, seed, n, grid.cells() + 1);
        out.chi2 = gpu_test::chi2_against(counts, l_masses(k, grid, ctx.quick()), total);
    }
    out.weight_off = gpu_test::device_count(draw_has{draw, draw_property::weight_off}, seed, n);
    out.pdf_off = gpu_test::device_count(draw_has{draw, draw_property::pdf_off}, seed, n);
    out.unsound = gpu_test::device_count(draw_has{draw, draw_property::unsound}, seed, n);
    out.false_returns = gpu_test::device_count(draw_has{draw, draw_property::false_return}, seed, n);
    out.count_off = gpu_test::device_count(draw_count_off{draw}, seed, n);
    out.energy = gpu_test::device_mean(energy_of_draw{{k.nf, k.vf, k.alpha, white()}}, energy_seed + index, n);
    return out;
}

// --- The furnace's patches (B.md §6.2, §6.3; G17) --------------------------------------------------------------------

// The μ of the quadratures that patch averages interpolate between: 0.96, 0.965, ..., 1
constexpr int patch_mu_count = 9;
double patch_mu(int i) { return 0.96 + 0.005 * i; }

struct furnace_patch
{
    double alpha;
    double x;   // the sphere's centre (x, 0, 0)
    int first;  // its patch's columns; rows 133..137
    int last;
};
constexpr std::array<furnace_patch, 4> ggx_furnace_patches{
    {{0.05, -2, 152, 156}, {0.3, 0, 238, 242}, {0.6, 2, 323, 327}, {1.0, 4, 409, 413}}};

// The mean of E(μ) over the patch's area, μ from the camera's rays: lookfrom (0, 0, 60) toward -z, 480 × 270, vfov 6°,
// a point at depth d at s = (240 + (X/d)K, 135 - (Y/d)K), K = 270 / (2 tan 3°); the sphere of radius 0.4 at (x, 0, 0).
// E is interpolated linearly between the values at patch_mu(i). Returns the average and the smallest μ met.
std::array<double, 2> patch_average(const furnace_patch& p, const std::vector<double>& e)
{
    const double k = 270 / (2 * std::tan(3 * pi / 180));
    constexpr int per_pixel = 40;
    const vec3<double> origin{0, 0, 60};
    const vec3<double> centre{p.x, 0, 0};
    constexpr double radius = 0.4;
    double sum = 0;
    double smallest = 1;
    int count = 0;
    for (int j = 0; j < 5 * per_pixel; ++j)
    {
        const double sy = 133 + (j + 0.5) / per_pixel;
        for (int i = 0; i < (p.last - p.first + 1) * per_pixel; ++i)
        {
            const double sx = p.first + (i + 0.5) / per_pixel;
            const vec3<double> d = normalized({(sx - 240) / k, -(sy - 135) / k, -1});
            const vec3<double> oc = origin - centre;
            const double b = gpu::dot(oc, d);
            const double disc = b * b - (gpu::dot(oc, oc) - radius * radius);
            if (!(disc >= 0))
            {
                continue;
            }
            const vec3<double> hit = origin + (-b - std::sqrt(disc)) * d;
            const double mu = -gpu::dot((1 / radius) * (hit - centre), d);
            const double x = std::clamp((mu - patch_mu(0)) / 0.005, 0.0, patch_mu_count - 1.0);
            const int i0 = std::min(static_cast<int>(x), patch_mu_count - 2);
            const double fx = x - i0;
            sum += (1 - fx) * e[static_cast<std::size_t>(i0)] + fx * e[static_cast<std::size_t>(i0) + 1];
            smallest = std::min(smallest, mu);
            ++count;
        }
    }
    return {count > 0 ? sum / count : std::numeric_limits<double>::quiet_NaN(), smallest};
}

// --- G1-G5, G7-G11, G13, G16, G17 ------------------------------------------------------------------------------------

GPU_TEST(ggx, conductor)
{
    const bool quick = ctx.quick();
    const resolution r = quick ? quick_resolution : full_resolution;

    // The grid's 35 cases first, then G8's θv = 45°, G13's min_ggx_alpha and G17's μ near 1
    std::vector<quad_case> cases;
    for (const double alpha : grid_alphas)
    {
        for (const int theta : grid_thetas)
        {
            cases.push_back(make_case(alpha, cos_degrees(theta)));
        }
    }
    const std::size_t table_first = cases.size();
    for (const double alpha : table_alphas)
    {
        cases.push_back(make_case(alpha, cos_degrees(45)));
    }
    const std::size_t min_alpha_first = cases.size();
    for (const int theta : grid_thetas)
    {
        cases.push_back(make_case(gpu::min_ggx_alpha, cos_degrees(theta)));
    }
    const std::size_t patch_first = cases.size();
    for (const furnace_patch& p : ggx_furnace_patches)
    {
        for (int i = 0; i < patch_mu_count; ++i)
        {
            cases.push_back(make_case(p.alpha, patch_mu(i)));
        }
    }
    const std::vector<integrals> q = h_quadrature(conductor_integrands{}, cases, r);
    const auto grid_index = [](std::size_t a, std::size_t t) { return a * grid_thetas.size() + t; };

    std::vector<case_draws> draws;
    for (std::size_t k = 0; k < grid_size; ++k)
    {
        draws.push_back(draw_case(ctx, cases[k], k));
    }

    // G1, G2: the weak white furnace and the NDF's normalization; α = 10⁻³ is INFO
    const auto unit_integral = [&](const std::string& name, double value, double alpha) {
        if (quick || alpha < 0.01)
        {
            ctx.info(name, std::format("{:.9g}{}", value, quick ? " (quick: no verdict)" : " (α = 10⁻³: no verdict)"));
        }
        else
        {
            ctx.exact(name, value, 1, 1e-4);
        }
    };
    for (std::size_t a = 0; a < grid_alphas.size(); ++a)
    {
        for (std::size_t t = 0; t < grid_thetas.size(); ++t)
        {
            unit_integral("weak-furnace-" + case_name(grid_alphas[a], grid_thetas[t]), q[grid_index(a, t)][0],
                          grid_alphas[a]);
        }
    }
    for (std::size_t a = 0; a < grid_alphas.size(); ++a)
    {
        unit_integral("ndf-" + number(grid_alphas[a]), q[grid_index(a, 0)][1], grid_alphas[a]);
    }

    // G3: 20 l tests (α ≥ 0.1) and 15 slope tests (α ≤ 0.05): T = 35
    for (std::size_t k = 0; k < grid_size; ++k)
    {
        const std::string kind = draws[k].slope ? "slope-chi2-" : "l-chi2-";
        const std::string name = kind + grid_name(k);
        ctx.chi2(name, draws[k].chi2.statistic, draws[k].chi2.dof, draws[k].chi2.p);
    }

    // G4: every true return
    for (std::size_t k = 0; k < grid_size; ++k)
    {
        const std::string name = grid_name(k);
        ctx.count("weight-eval-" + name, draws[k].weight_off, 0, draws[k].drawn);
        ctx.count("pdf-" + name, draws[k].pdf_off, 0, draws[k].drawn);
    }

    // G5: the false returns' share plus ∫ pdf over the upper hemisphere, SE binomial (at least 10⁻⁷)
    for (std::size_t k = 0; k < grid_size; ++k)
    {
        const std::string name = grid_name(k);
        const double p = static_cast<double>(draws[k].false_returns) / static_cast<double>(draws[k].drawn);
        const double se = std::max(1e-7, std::sqrt(p * (1 - p) / static_cast<double>(draws[k].drawn)));
        ctx.mean("mass-" + name, p + q[k][2], se, 1);
    }

    // G7: E_ss as the mean weight against E_ss by quadrature (SE at least 10⁻⁶); both at most 1 + 10⁻⁶
    std::int64_t above_one = 0;
    std::int64_t left_out = 0;
    double largest = 0;
    for (std::size_t k = 0; k < grid_size; ++k)
    {
        const std::string name = grid_name(k);
        const gpu_test::mean_estimate& e = draws[k].energy;
        ctx.mean("energy-" + name, e.mean, std::max(1e-6, e.standard_error), q[k][3]);
        above_one += e.mean <= 1 + 1e-6 ? 0 : 1;
        above_one += quick || q[k][3] <= 1 + 1e-6 ? 0 : 1;
        largest = std::max({largest, e.mean, quick ? 0.0 : q[k][3]});
        left_out += e.left_out;
    }
    ctx.count("energy-at-most-1", above_one, 0, static_cast<std::int64_t>(quick ? grid_size : 2 * grid_size));
    ctx.info("energy-largest", std::format("{:.9g}", largest));
    ctx.count("energy-nan-weights", left_out, 0, static_cast<std::int64_t>(grid_size) * draws[0].drawn);

    // G8: ggx.md's table
    for (std::size_t a = 0; a < table_alphas.size(); ++a)
    {
        const std::size_t grid_a = static_cast<std::size_t>(
            std::ranges::find(grid_alphas, table_alphas[a]) - grid_alphas.begin());
        for (std::size_t t = 0; t < table_thetas.size(); ++t)
        {
            const int theta = table_thetas[t];
            const std::size_t index =
                theta == 45 ? table_first + a
                            : grid_index(grid_a, static_cast<std::size_t>(
                                                     std::ranges::find(grid_thetas, theta) - grid_thetas.begin()));
            const std::string name = "energy-table-" + case_name(table_alphas[a], theta);
            if (quick)
            {
                ctx.info(name, std::format("{:.9g} (quick: no verdict)", q[index][3]));
            }
            else
            {
                ctx.exact(name, q[index][3], energy_table[a][t], 2e-3);
            }
        }
    }

    // G9: E_ss → 1 as α → 0, at α = 10⁻³
    {
        std::int64_t short_of_one = 0;
        std::string values;
        for (std::size_t t = 0; t < grid_thetas.size(); ++t)
        {
            const double e = q[grid_index(0, t)][3];
            short_of_one += e >= (grid_thetas[t] <= 80 ? 0.9999 : 0.998) ? 0 : 1;
            values += std::format("{}{}°: {:.7f}", t == 0 ? "" : ", ", grid_thetas[t], e);
        }
        if (quick)
        {
            ctx.info("energy-limit", values + " (quick: no verdict)");
        }
        else
        {
            ctx.count("energy-limit", short_of_one, 0, static_cast<std::int64_t>(grid_thetas.size()));
            ctx.info("energy-limit-values", "E_ss at α = 0.001: " + values);
        }
    }

    // G10, G11 over every case
    std::int64_t unsound = 0;
    std::int64_t count_off = 0;
    std::int64_t all = 0;
    for (const case_draws& d : draws)
    {
        unsound += d.unsound;
        count_off += d.count_off;
        all += d.drawn;
    }
    ctx.count("finite", unsound, 0, all);
    ctx.count("draws", count_off, 0, all);

    // G13: the weak furnace at min_ggx_alpha, in float, within 10⁻³ (else the plan raises the threshold)
    ctx.info("min-ggx-alpha", std::format("{}", gpu::min_ggx_alpha));
    for (std::size_t t = 0; t < grid_thetas.size(); ++t)
    {
        const std::string name = std::format("min-alpha-ok-{}", grid_thetas[t]);
        const double value = q[min_alpha_first + t][0];
        if (quick)
        {
            ctx.info(name, std::format("{:.9g} (quick: no verdict)", value));
        }
        else
        {
            ctx.exact(name, value, 1, 1e-3);
        }
    }

    // The quadrature's resolution (B.md §5.1, decision 9): 1024 × 256 against 2048 × 512
    if (quick)
    {
        ctx.info("quadrature-resolution", "quick: not compared");
    }
    else
    {
        const std::vector<quad_case> grid_cases(cases.begin(), cases.begin() + static_cast<std::ptrdiff_t>(grid_size));
        const std::vector<integrals> half = h_quadrature(conductor_integrands{}, grid_cases, half_resolution);
        // G1's and G2's integrands are smooth: within 10⁻⁵. E_ss's stops at the horizon, where its midpoint error is
        // O(1/N): within a tenth of G8's 2·10⁻³.
        double worst = 0;
        double worst_energy = 0;
        for (std::size_t k = grid_thetas.size(); k < grid_size; ++k) // α ≥ 0.01
        {
            for (const std::size_t i : {std::size_t{0}, std::size_t{1}})
            {
                const double d = std::abs(half[k][i] - q[k][i]);
                worst = std::isnan(d) || d > worst ? d : worst; // a NaN stays
            }
            const double d = std::abs(half[k][3] - q[k][3]);
            worst_energy = d <= worst_energy ? worst_energy : d;
        }
        ctx.exact("quadrature-resolution", worst, 0, 1e-5);
        ctx.exact("quadrature-resolution-energy", worst_energy, 0, 2e-4);
    }

    // G16: the planted fault, draws at α = 0.31, θv = 45°, against the l grid's masses at α = 0.30
    {
        const quad_case right = cases[table_first + 1]; // α = 0.3, 45°
        const quad_case wrong = make_case(0.31, cos_degrees(45));
        const direction_grid grid = l_grid(right);
        const auto counts = gpu_test::device_histogram(
            l_cell_of_draw{{wrong.nf, wrong.vf, wrong.alpha, test_f0()}, grid}, planted_seed,
            ctx.samples(case_samples), grid.cells() + 1);
        double total = 0;
        const gpu_test::chi2_outcome planted = gpu_test::chi2_against(counts, l_masses(right, grid, quick), total);
        ctx.must_reject("planted-alpha", planted.statistic, planted.dof, planted.p);
    }

    // G17: E_ss(1, α) and the furnace's patch averages
    for (std::size_t p = 0; p < ggx_furnace_patches.size(); ++p)
    {
        const furnace_patch& patch = ggx_furnace_patches[p];
        std::vector<double> e;
        for (int i = 0; i < patch_mu_count; ++i)
        {
            e.push_back(q[patch_first + p * patch_mu_count + static_cast<std::size_t>(i)][3]);
        }
        const auto [average, smallest] = patch_average(patch, e);
        ctx.info("furnace-values-" + number(patch.alpha),
                 std::format("E_ss(1, α) = {:.6f}; ggx-furnace's patch [{}..{}] × [133..137], single scattering "
                             "alone (before Step 79's compensation), averages {:.6f} (μ ≥ {:.4f})",
                             e.back(), patch.first, patch.last, average, smallest));
    }
}

// --- G6: reciprocity -------------------------------------------------------------------------------------------------

// Pairs (v, l) uniform on the upper hemisphere: f(v, l) = ggx_eval(n, v, l)/c_l against f(l, v) = ggx_eval(n, l, v)/c_v,
// relative 10⁻⁴: swapping v and l rounds the same float formula in another order, and about 3 pairs in 1000 differ
// by more than 8 ulps (10⁻⁶); a formula that isn't reciprocal (a cosine missing or doubled, G1(v) alone) is off by
// far more
struct reciprocity_off
{
    vec3<float> n;
    float alpha;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        vec3<float> v = gpu::random_unit_vector<float>(rng);
        vec3<float> l = gpu::random_unit_vector<float>(rng);
        v = gpu::dot(n, v) < 0.0f ? -v : v;
        l = gpu::dot(n, l) < 0.0f ? -l : l;
        const float cv = gpu::dot(n, v);
        const float cl = gpu::dot(n, l);
        if (!(cv > 0.0f && cl > 0.0f))
        {
            return false;
        }
        float pdf = 0;
        const vec3<float> a = gpu::ggx_eval(n, v, l, alpha, test_f0(), pdf);
        const vec3<float> b = gpu::ggx_eval(n, l, v, alpha, test_f0(), pdf);
        for (int i = 0; i < 3; ++i)
        {
            const double fa = static_cast<double>(channel(a, i)) / cl;
            const double fb = static_cast<double>(channel(b, i)) / cv;
            if (!(::fabs(fa - fb) <= ::fmax(1e-4 * ::fmax(::fabs(fa), ::fabs(fb)), 1e-30)))
            {
                return true;
            }
        }
        return false;
    }
};

GPU_TEST(ggx, reciprocity)
{
    const vec3<float> n = to_float(shading_frame().n);
    const std::int64_t pairs = ctx.samples(pair_samples);
    for (std::size_t a = 0; a < grid_alphas.size(); ++a)
    {
        ctx.count("reciprocity-" + number(grid_alphas[a]),
                  gpu_test::device_count(reciprocity_off{n, static_cast<float>(grid_alphas[a])},
                                         reciprocity_seed + a, pairs),
                  0, gpu_test::drawn(pairs));
    }
}

// --- G12, G14 and make_bsdf: the lobes -------------------------------------------------------------------------------

// A random hit: a unit normal, an incoming direction against it of length 0.5 to 3 (as the bsdf suite's)
struct hit_case
{
    gpu::hit_record<float> rec;
    gpu::ray<float> in;
};

__device__ inline hit_case random_hit(gpu::pcg32& rng)
{
    hit_case h{};
    h.rec.p = {0.5f, 1.0f, -2.0f};
    h.rec.normal = gpu::random_unit_vector<float>(rng);
    h.rec.front_face = true;
    vec3<float> d = gpu::random_unit_vector<float>(rng);
    d = gpu::dot(d, h.rec.normal) > 0.0f ? -d : d;
    const float length = 0.5f + 2.5f * gpu::random_real<float>(rng);
    h.in = {h.rec.p - length * d, length * d};
    return h;
}

__device__ inline vec3<float> random_color(gpu::pcg32& rng)
{
    const float r = gpu::random_real<float>(rng);
    const float g = gpu::random_real<float>(rng);
    const float b = gpu::random_real<float>(rng);
    return {r, g, b};
}

__device__ inline gpu::material<float> metal(const vec3<float>& albedo, float fuzz = 0)
{
    gpu::material<float> m{};
    m.kind = gpu::material_kind::metal;
    m.albedo = albedo;
    m.fuzz = fuzz;
    return m;
}

__device__ inline bool generators_equal(gpu::pcg32 a, gpu::pcg32 b)
{
    bool same = true;
    for (int i = 0; i < 4; ++i)
    {
        same = a.next() == b.next() && same;
    }
    return same;
}

// G12: a conductor side entry below min_ggx_alpha is the perfect mirror with Schlick's F, a delta lobe
struct smooth_off
{
    float alpha;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const hit_case h = random_hit(rng);
        const vec3<float> f0 = random_color(rng);
        const gpu::microfacet side{gpu::microfacet_kind::conductor, alpha, sqrtf(alpha)};
        const gpu::bsdf b = gpu::make_bsdf(metal(f0), &side, f0, h.rec, h.in.direction);
        bool ok = b.kind == gpu::lobe::smooth_conductor && gpu::is_delta(b) && bits_equal(b.albedo, f0) &&
                  bits_equal(b.n, h.rec.normal);
        gpu::pcg32 used = rng;
        gpu::bsdf_sample s{nan_vector(), nan_vector(), quiet_nan(), false};
        ok = gpu::sample_bsdf(b, h.in, h.rec, used, s) && ok;
        const vec3<double> n = to_double(b.n);
        const vec3<double> wo = to_double(b.wo);
        const vec3<double> wi = to_double(s.wi);
        ok = ok && ::fabs(::sqrt(gpu::dot(wi, wi)) - 1) <= 1e-6;
        ok = ok && ::fabs(gpu::dot(n, wi) - gpu::dot(n, wo)) <= 1e-6;
        ok = ok && ::fabs(gpu::dot(wi, cross_d(n, wo))) <= 1e-6;
        ok = ok && close(s.weight, gpu::schlick(b.albedo, gpu::dot(b.n, b.wo)), 0x1p-22);
        ok = ok && s.pdf == 0.0f && s.delta;
        ok = generators_equal(used, rng) && ok; // no random number drawn
        float pdf = 1;
        const vec3<float> e = gpu::eval_bsdf(b, gpu::random_unit_vector<float>(rng), pdf);
        ok = ok && pdf == 0.0f && is_zero(e);
        pdf = 1;
        const vec3<float> mirror = gpu::eval_bsdf(b, s.wi, pdf);
        return !(ok && pdf == 0.0f && is_zero(mirror));
    }
};

// make_bsdf's fields (B.md §2.2): α = 0 and just below min_ggx_alpha → smooth; min_ggx_alpha and 0.3 → rough; a side
// entry of kind none on a metal → the books' metal, on a matte surface → lambertian
struct make_off
{
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const hit_case h = random_hit(rng);
        const vec3<float> f0 = random_color(rng);
        const vec3<float> c = random_color(rng); // the color surface_color gave
        const int which = static_cast<int>(gpu::random_real<float>(rng) * 6) % 6;
        const float alphas[] = {0.0f, nextafterf(gpu::min_ggx_alpha, 0.0f), gpu::min_ggx_alpha, 0.3f, 0.3f, 0.3f};
        const float alpha = alphas[which];
        const gpu::microfacet side{which < 4 ? gpu::microfacet_kind::conductor : gpu::microfacet_kind::none, alpha,
                                   sqrtf(alpha)};
        gpu::material<float> m = metal(f0);
        if (which == 5)
        {
            m.kind = gpu::material_kind::lambertian;
        }
        const gpu::bsdf b = gpu::make_bsdf(m, &side, c, h.rec, h.in.direction);
        const vec3<float> wo = -gpu::unit_vector(h.in.direction);
        // wo, the direction toward the viewer, is computed for the GGX lobes only (the books' lobes don't read it)
        bool ok = bits_equal(b.n, h.rec.normal) && (which >= 4 || fabsf(gpu::dot(b.wo, wo) - 1.0f) <= 1e-6f);
        switch (which)
        {
        case 0:
        case 1:
            ok = ok && b.kind == gpu::lobe::smooth_conductor && bits_equal(b.albedo, f0);
            break;
        case 2:
        case 3:
            ok = ok && b.kind == gpu::lobe::conductor && bits_equal(b.albedo, f0) && bits_equal(b.alpha, alpha) &&
                 !gpu::is_delta(b);
            break;
        case 4:
            ok = ok && b.kind == gpu::lobe::books_delta && b.books.kind == m.kind && bits_equal(b.books.albedo, f0) &&
                 bits_equal(b.books.fuzz, m.fuzz);
            break;
        default:
            ok = ok && b.kind == gpu::lobe::lambertian && bits_equal(b.albedo, c);
            break;
        }
        return !ok;
    }
};

// G14: the conductor lobe from make_bsdf is ggx.cuh's functions with the lobe's own n, wo, α and F0 on equal copies of
// the generator: the same direction and pdf bit for bit, the weight and f·cos θ times Turquin's factor (Step 79) to
// 10⁻⁶
struct lobe_off
{
    float alpha;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const hit_case h = random_hit(rng);
        const gpu::microfacet side{gpu::microfacet_kind::conductor, alpha, sqrtf(alpha)};
        const gpu::bsdf b = gpu::make_bsdf(metal(aluminium()), &side, aluminium(), h.rec, h.in.direction);
        if (b.kind != gpu::lobe::conductor || gpu::is_delta(b))
        {
            return true;
        }
        gpu::pcg32 a = rng;
        gpu::pcg32 c = rng;
        gpu::bsdf_sample s{nan_vector(), nan_vector(), quiet_nan(), true};
        const bool drawn = gpu::sample_bsdf(b, h.in, h.rec, a, s);
        vec3<float> l = nan_vector();
        vec3<float> w = nan_vector();
        float pdf = quiet_nan();
        const bool sampled = gpu::ggx_sample(b.n, b.wo, b.alpha, b.albedo, c, l, w, pdf);
        bool same = drawn == sampled && generators_equal(a, c);
        const vec3<float> factor = gpu::ggx_compensation(b.albedo, gpu::dot(b.n, b.wo), side.roughness);
        if (drawn && sampled)
        {
            same = same && bits_equal(s.wi, l) && close(s.weight, w * factor, 1e-6) && bits_equal(s.pdf, pdf) &&
                   !s.delta;
        }
        // eval at the drawn direction (true or false) and at a random one, above or below the horizon
        const vec3<float> directions[] = {l, gpu::random_unit_vector<float>(rng)};
        for (const vec3<float>& d : directions)
        {
            if (!finite(d))
            {
                continue;
            }
            float lobe_pdf = quiet_nan();
            float ggx_pdf = quiet_nan();
            const vec3<float> e = gpu::eval_bsdf(b, d, lobe_pdf);
            const vec3<float> g = gpu::ggx_eval(b.n, b.wo, d, b.alpha, b.albedo, ggx_pdf);
            same = same && close(e, g * factor, 1e-6) && bits_equal(lobe_pdf, ggx_pdf);
        }
        rng = a;
        return !same;
    }
};

GPU_TEST(ggx, lobes)
{
    const std::int64_t n = ctx.samples(random_cases);
    const std::int64_t drawn = gpu_test::drawn(n);
    std::int64_t smooth = 0;
    smooth += gpu_test::device_count(smooth_off{0.0f}, smooth_seed, n);
    smooth += gpu_test::device_count(smooth_off{5e-4f}, smooth_seed + 1, n);
    ctx.count("smooth", smooth, 0, 2 * drawn);
    ctx.count("make", gpu_test::device_count(make_off{}, make_seed, n), 0, drawn);
    constexpr double lobe_alphas[] = {0.01, 0.09, 0.3};
    for (std::size_t a = 0; a < 3; ++a)
    {
        ctx.count("lobe-consistency-" + number(lobe_alphas[a]),
                  gpu_test::device_count(lobe_off{static_cast<float>(lobe_alphas[a])}, lobe_seed + a, n), 0, drawn);
    }
}

// --- The formulas of B.md §2.3 against double ------------------------------------------------------------------------

enum class formula
{
    cross,
    smith_a,
    g1,
    g2,
    d,
    schlick,
    eval,
    guards,
};

// A direction around n at the cosine c, at a random azimuth
__device__ inline vec3<float> at_cosine(const vec3<float>& n, float c, gpu::pcg32& rng)
{
    float sin_phi;
    float cos_phi;
    sincospif(2 * gpu::random_real<float>(rng), &sin_phi, &cos_phi);
    const float s = sqrtf(fmaxf(0.0f, 1 - c * c));
    return gpu::around(n, vec3<float>{s * cos_phi, s * sin_phi, c});
}

__device__ inline float random_alpha(gpu::pcg32& rng, float smallest)
{
    return smallest * powf(1 / smallest, gpu::random_real<float>(rng)); // log-uniform in [smallest, 1)
}

struct formula_off
{
    formula which;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        switch (which)
        {
        case formula::cross:
        {
            const vec3<float> a = 2.0f * random_color(rng) - white();
            const vec3<float> b = 2.0f * random_color(rng) - white();
            const vec3<float> c = gpu::cross(a, b);
            const vec3<double> e = cross_d(to_double(a), to_double(b));
            const double scale = ::sqrt(gpu::dot(to_double(a), to_double(a)) * gpu::dot(to_double(b), to_double(b)));
            return !(::fabs(c.x - e.x) <= 1e-6 * scale && ::fabs(c.y - e.y) <= 1e-6 * scale &&
                     ::fabs(c.z - e.z) <= 1e-6 * scale);
        }
        case formula::smith_a:
        {
            const float c = gpu::random_real<float>(rng);
            const float alpha = random_alpha(rng, 1e-3f);
            const float alpha2 = alpha * alpha;
            return !close(gpu::smith_a(c, alpha2), smith_a_ref(c, alpha2), 1e-6);
        }
        case formula::g1:
        {
            const float c = 1 - gpu::random_real<float>(rng); // (0, 1]
            const float alpha2 = random_alpha(rng, 1e-3f) * random_alpha(rng, 1e-3f);
            return !close(gpu::ggx_g1(c, alpha2), 2 * c / (c + smith_a_ref(c, alpha2)), 1e-6);
        }
        case formula::g2:
        {
            const float cv = 1 - gpu::random_real<float>(rng);
            const float cl = 1 - gpu::random_real<float>(rng);
            const float alpha = random_alpha(rng, 1e-3f);
            const float alpha2 = alpha * alpha;
            const double ref =
                2.0 * cv * cl / (cl * smith_a_ref(cv, alpha2) + static_cast<double>(cv) * smith_a_ref(cl, alpha2));
            return !close(gpu::ggx_g2(cv, cl, alpha2), ref, 1e-6);
        }
        case formula::d:
        {
            // n·h from 0.05 to 0.95, where float's |n × h|² has 10⁻⁶ of its value at worst; and n·h < 0: 0
            const vec3<float> n = gpu::random_unit_vector<float>(rng);
            const bool below = gpu::random_real<float>(rng) < 0.1f;
            const float c = below ? -0.9f * gpu::random_real<float>(rng) - 0.01f
                                  : 0.05f + 0.9f * gpu::random_real<float>(rng);
            const vec3<float> h = at_cosine(n, c, rng);
            const float alpha = random_alpha(rng, 0.01f);
            const float alpha2 = alpha * alpha;
            const float d = gpu::ggx_d(n, h, alpha2);
            return below ? d != 0.0f : !close(d, ggx_d_ref(to_double(n), to_double(h), alpha2), 2e-5);
        }
        case formula::schlick:
        {
            const vec3<float> f0 = random_color(rng);
            const float mu = 1.05f * gpu::random_real<float>(rng); // past 1 too: (1 - μ) taken as max(0, 1 - μ)
            const vec3<float> f = gpu::schlick(f0, mu);
            for (int i = 0; i < 3; ++i)
            {
                if (!close(channel(f, i), schlick_ref(channel(f0, i), mu), 1e-6, 1e-7))
                {
                    return true;
                }
            }
            return false;
        }
        case formula::eval:
        {
            // v and l at cosines from 0.1 to 1 and α from 0.1 to 1, where float's D is within 10⁻⁵ of double's
            const vec3<float> n = gpu::random_unit_vector<float>(rng);
            const vec3<float> v = at_cosine(n, 0.1f + 0.9f * gpu::random_real<float>(rng), rng);
            const vec3<float> l = at_cosine(n, 0.1f + 0.9f * gpu::random_real<float>(rng), rng);
            const float alpha = random_alpha(rng, 0.1f);
            const vec3<float> f0 = random_color(rng);
            eval_ref ref;
            if (!ggx_eval_ref(n, v, l, alpha, f0, ref))
            {
                return false;
            }
            float pdf = 0;
            const vec3<float> f = gpu::ggx_eval(n, v, l, alpha, f0, pdf);
            bool ok = close(pdf, ref.pdf, 1e-4) && close(gpu::ggx_pdf(n, v, l, alpha), ref.pdf, 1e-4);
            for (int i = 0; i < 3; ++i)
            {
                ok = ok && close(channel(f, i), ref.f[i], 1e-4);
            }
            return !ok;
        }
        case formula::guards:
        {
            // v or l (or both) below the horizon: f = 0 and pdf = 0
            const vec3<float> n = gpu::random_unit_vector<float>(rng);
            const int below = 1 + static_cast<int>(gpu::random_real<float>(rng) * 3) % 3; // 1: v, 2: l, 3: both
            const float cv = (below & 1 ? -1.0f : 1.0f) * (0.01f + 0.99f * gpu::random_real<float>(rng));
            const float cl = (below & 2 ? -1.0f : 1.0f) * (0.01f + 0.99f * gpu::random_real<float>(rng));
            const vec3<float> v = at_cosine(n, cv, rng);
            const vec3<float> l = at_cosine(n, cl, rng);
            const float alpha = random_alpha(rng, 1e-3f);
            float pdf = 1;
            const vec3<float> f = gpu::ggx_eval(n, v, l, alpha, test_f0(), pdf);
            return !(is_zero(f) && pdf == 0.0f && gpu::ggx_pdf(n, v, l, alpha) == 0.0f);
        }
        }
        return true;
    }
};

GPU_TEST(ggx, formulas)
{
    const std::int64_t n = ctx.samples(random_cases);
    constexpr std::pair<formula, const char*> formulas[] = {
        {formula::cross, "cross"},     {formula::smith_a, "smith-a"}, {formula::g1, "g1"},
        {formula::g2, "g2"},           {formula::d, "d"},             {formula::schlick, "schlick"},
        {formula::eval, "eval-pdf"},   {formula::guards, "guards"}};
    for (const auto& [which, name] : formulas)
    {
        ctx.count(std::string("formula-") + name,
                  gpu_test::device_count(formula_off{which}, formula_seed + static_cast<std::uint64_t>(which), n), 0,
                  gpu_test::drawn(n));
    }
}

// --- G15: validate's refusals, through gpu::render -------------------------------------------------------------------

// One sphere of the given material in front of a 1×1-pixel camera, under a white sky, with the given side entries
gpu::scene_desc one_sphere(gpu::material_desc material, std::vector<gpu::microfacet_desc> side,
                           std::size_t material_count = 1)
{
    gpu::scene_desc scene;
    scene.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = {0.5, 0.5, 0.5}});
    material.texture = 0;
    for (std::size_t i = 0; i < material_count; ++i)
    {
        scene.materials.push_back(material);
    }
    scene.spheres.push_back({.center = {0, 0, -3}, .radius = 1, .material = 0});
    scene.camera = {.image_width = 1,
                    .image_height = 1,
                    .sqrt_spp = 1,
                    .max_depth = 4,
                    .center = {0, 0, 0},
                    .pixel00_loc = {0, 0, -1},
                    .pixel_delta_u = {0.01, 0, 0},
                    .pixel_delta_v = {0, -0.01, 0},
                    .sky_bottom = {1, 1, 1},
                    .sky_top = {1, 1, 1},
                    .sampling = gpu::sampling_strategy::material};
    scene.quality.microfacets = std::move(side);
    return scene;
}

// 1 when gpu::render throws a std::exception with a message, 0 when it renders
double refused(const gpu::scene_desc& scene)
{
    try
    {
        gpu::render(scene, {.seed = validate_seed});
    }
    catch (const std::exception& error)
    {
        return std::string_view(error.what()).empty() ? 0 : 1;
    }
    return 0;
}

GPU_TEST(ggx, validate)
{
    using gpu::material_kind;
    using gpu::microfacet_kind;
    const gpu::material_desc mirror{.kind = material_kind::metal, .albedo = {0.9, 0.9, 0.9}, .fuzz = 0};
    const gpu::material_desc fuzzy{.kind = material_kind::metal, .albedo = {0.9, 0.9, 0.9}, .fuzz = 0.2};
    const gpu::material_desc matte{.kind = material_kind::lambertian};
    const gpu::material_desc glass{.kind = material_kind::dielectric, .refraction_index = 1.5};
    const auto conductor = [](double r) { return gpu::microfacet_desc{microfacet_kind::conductor, r}; };
    const auto gloss = [](double r) { return gpu::microfacet_desc{microfacet_kind::gloss, r}; };
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    constexpr double infinity = std::numeric_limits<double>::infinity();

    ctx.exact("validate-refuses-fewer-entries", refused(one_sphere(mirror, {conductor(0.3)}, 2)), 1, 0);
    ctx.exact("validate-refuses-more-entries", refused(one_sphere(mirror, {conductor(0.3), conductor(0.3)})), 1, 0);
    ctx.exact("validate-refuses-conductor-on-matte", refused(one_sphere(matte, {conductor(0.3)})), 1, 0);
    ctx.exact("validate-refuses-conductor-on-glass", refused(one_sphere(glass, {conductor(0.3)})), 1, 0);
    ctx.exact("validate-refuses-conductor-on-fuzzy-metal", refused(one_sphere(fuzzy, {conductor(0.3)})), 1, 0);
    ctx.exact("validate-refuses-gloss-on-metal", refused(one_sphere(mirror, {gloss(0.3)})), 1, 0);
    ctx.exact("validate-refuses-roughness-nan", refused(one_sphere(mirror, {conductor(nan)})), 1, 0);
    ctx.exact("validate-refuses-roughness-infinite", refused(one_sphere(mirror, {conductor(infinity)})), 1, 0);
    ctx.exact("validate-refuses-roughness-negative", refused(one_sphere(mirror, {conductor(-0.1)})), 1, 0);
    ctx.exact("validate-refuses-roughness-above-1", refused(one_sphere(mirror, {conductor(1.5)})), 1, 0);
    ctx.exact("validate-refuses-gloss-below-min-alpha", refused(one_sphere(matte, {gloss(0.03)})), 1, 0); // α 9·10⁻⁴
    // And what it must render (decision 8)
    ctx.exact("validate-accepts-conductor", refused(one_sphere(mirror, {conductor(0.3)})), 0, 0);
    ctx.exact("validate-accepts-smooth-conductor", refused(one_sphere(mirror, {conductor(0)})), 0, 0);
    ctx.exact("validate-accepts-roughness-1", refused(one_sphere(mirror, {conductor(1)})), 0, 0);
    ctx.exact("validate-accepts-none", refused(one_sphere(matte, {gpu::microfacet_desc{}})), 0, 0);
}

// --- G18 (Step 81): Veach's conditions on the weights, with the GGX lobes under the lamps ----------------------------
//
// At a vertex of a rough metal or a glossy coat the path takes a light sample and a sample of the lobe, the next ray
// (D.md §2.1), and mis.cuh weighs each against the other technique's density of the same direction. Veach's
// multi-sample estimator (1997, eq. 9.4, p. 260) is unbiased iff (W1) w_L + w_B = 1 wherever f·Le ≠ 0 and both
// techniques can draw the direction, and (W2) a technique that can't draw it has weight 0, so there the other's weight
// is 1 alone. The lamps suite's T3 checks this with the books' lobes, whose two densities are one formula; the
// conductor computes its density twice, ggx_sample's from the drawn half vector and ggx_pdf's from unit(v + l), and
// enters the weights of one direction with both: W1 holds only as far as they agree. (The gloss's sample takes its
// density from gloss_eval at the drawn direction, as eval_bsdf does: its W1 holds by construction, and an error in
// that density is the gloss suite's to find, by its χ² and masses.) As D.md §6's T3 (and §10.7's pairing):
//   (a) a direction the light sampler draws: w_L(p_L, p_B) + w_B(p_B, p_L′), p_B eval_bsdf's density, p_L′ the
//       lamp_hit_pdf of the traced hit;
//   (b) a direction the lobe draws that reaches a lamp's front: w_B(p_B,sample, p_L′) + w_L(p_L′, p_B,eval), as the
//       integrator pairs them (the sample's own density at the lamp hit, eval_bsdf's at the light sample).
// Under the power heuristic |Σw − 1| ≤ 10⁻⁵ + τ (B.md G18), under lights_only Σw = 1 exactly.

constexpr std::int64_t t3_samples = 1'000'000; // per case and technique: light-drawn, lobe-drawn
constexpr std::uint64_t t3_seed = 8100;        // + 2·case + technique: 28 cases (2 setups × 14 lobes), 8100 to 8155
constexpr std::uint64_t t3_planted_seed = 8160;

// The conductor at the ggx grid's α and B.md's 0.09 (aluminium); the gloss at the gloss suite's α and B.md's 0.04
// (ρ = 0.73)
constexpr std::array<double, 8> t3_conductor_alphas{1e-3, 0.01, 0.05, 0.09, 0.1, 0.3, 0.6, 1.0};
constexpr std::array<double, 6> t3_gloss_alphas{0.01, 0.04, 0.1, 0.3, 0.6, 1.0};

// The books' lamps (D.md §5.5), alone in the BVH, so that nothing else stands between a point and a lamp: cornell's
// quad of L = 15 (one lamp, pmf 1), or light's quad and sphere of L = 4 (pmf ½)
gpu::scene_desc t3_lamps(bool cornell)
{
    gpu::scene_desc s;
    const double radiance = cornell ? 15 : 4;
    s.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = {radiance, radiance, radiance}});
    s.materials.push_back({.kind = gpu::material_kind::diffuse_light, .texture = 0});
    if (cornell)
    {
        s.quads.push_back({.q = {343, 554, 332}, .u = {-130, 0, 0}, .v = {0, 0, -105}, .material = 0});
    }
    else
    {
        s.quads.push_back({.q = {3, 1, -2}, .u = {2, 0, 0}, .v = {0, 2, 0}, .material = 0});
        s.spheres.push_back({.center = {0, 7, 0}, .radius = 2, .material = 0});
    }
    return s;
}

// The lamps on the GPU as the quality kernel's view holds them (shared.md 3.11): the BVH of device_scene, the light set
// of build_lamps, light_pmf = float(1/N) divided in double (D.md §3.3)
class t3_scene
{
public:
    explicit t3_scene(bool cornell)
        : scene_(t3_lamps(cornell)),
          device_(scene_),
          table_(gpu::build_lamps(scene_, device_.sphere_ids(), device_.quad_ids())),
          lamps_(table_.lamps),
          lamp_of_(table_.lamp_of)
    {
    }

    int count() const { return static_cast<int>(table_.lamps.size()); }

    gpu::quality_view view() const
    {
        gpu::quality_view v{};
        v.base = device_.view();
        v.microfacets = nullptr; // the test makes the lobe itself, passing its side entry to make_bsdf
        v.lamps = {.lamps = lamps_.get(), .lamp_of = lamp_of_.get(), .count = count()};
        v.environment = {};
        v.light_pmf = count() > 0 ? static_cast<float>(1.0 / count()) : 0.0f;
        v.combination = gpu::light_combination::power_heuristic; // unread: each weight takes its combination
        return v;
    }

private:
    gpu::scene_desc scene_;
    gpu_test::device_scene device_;
    gpu::lamp_table table_;
    gpu_test::device_buffer<gpu::emitter> lamps_;
    gpu_test::device_buffer<int> lamp_of_;
};

// The vertex's material, as make_bsdf gets it
struct t3_lobe
{
    gpu::material<float> m; // a metal of fuzz 0 (conductor) or a matte surface (gloss)
    gpu::microfacet side;   // its side entry: kind, α = float(r²), r = float(r), from r in double (B.md §2.1)
    vec3<float> c;          // the color surface_color gave: the gloss's ρ
    gpu::lobe expected;     // what make_bsdf must make of them
};

t3_lobe make_t3_lobe(bool gloss, double alpha)
{
    const double r = std::sqrt(alpha);
    t3_lobe l{};
    l.side = {gloss ? gpu::microfacet_kind::gloss : gpu::microfacet_kind::conductor, static_cast<float>(r * r),
              static_cast<float>(r)};
    if (gloss)
    {
        l.m.kind = gpu::material_kind::lambertian;
        l.m.albedo = {0.73f, 0.73f, 0.73f};
        l.expected = gpu::lobe::gloss;
    }
    else
    {
        l.m.kind = gpu::material_kind::metal;
        l.m.albedo = aluminium();
        l.m.fuzz = 0;
        l.expected = gpu::lobe::conductor;
    }
    l.c = l.m.albedo;
    return l;
}

enum t3_tally : int
{
    t3_tested,      // f·Le ≠ 0 and both techniques have a density: W1
    t3_one_density, // f·Le ≠ 0, but the lobe's density is 0 (a light-drawn direction): w_L alone must be 1 (W2)
    t3_nee_wrong,   // lights_only: Σw not exactly 1, a weight outside [0, 1], a lamp's front without p_L′
    t3_mis_wrong,   // power heuristic: |Σw − 1| > 10⁻⁵ + τ, a weight outside [0, 1], a lamp's front without p_L′
    t3_w2_wrong,    // a density where its technique can't draw the direction, or none where it does (W2)
    t3_far,         // power heuristic: |Σw − 1| > 10⁻³ (the planted fault's verdict)
    t3_tallies,
};

__device__ inline bool positive(const vec3<float>& c) { return c.x > 0.0f || c.y > 0.0f || c.z > 0.0f; }
__device__ inline bool unit_interval(float w) { return w >= 0.0f && w <= 1.0f; } // a NaN fails
__device__ inline gpu::light_combination t3_mode(int c) // 0: nee, 1: mis
{
    return c == 0 ? gpu::light_combination::lights_only : gpu::light_combination::power_heuristic;
}

// One number per statement: their order is the statements'
__device__ inline float uniform_in(gpu::pcg32& rng, float lo, float hi)
{
    return lo + (hi - lo) * gpu::random_real<float>(rng);
}

// B.md G18's points under cornell's lamp, [100, 450] × [50, 400] × [100, 450]; D.md §6's P_light under light's,
// [−10, 10] × [0, 10] × [−10, 10] outside D² ≤ 4.04 of the sphere lamp (the points behind the quad included)
__device__ inline vec3<float> t3_point(bool cornell, gpu::pcg32& rng)
{
    const vec3<float> lo = cornell ? vec3<float>{100.0f, 50.0f, 100.0f} : vec3<float>{-10.0f, 0.0f, -10.0f};
    const vec3<float> hi = cornell ? vec3<float>{450.0f, 400.0f, 450.0f} : vec3<float>{10.0f, 10.0f, 10.0f};
    for (;;) // rejects less than 1% of light's box, none of cornell's
    {
        const float x = uniform_in(rng, lo.x, hi.x);
        const float y = uniform_in(rng, lo.y, hi.y);
        const float z = uniform_in(rng, lo.z, hi.z);
        const vec3<float> p{x, y, z};
        if (cornell || (p - vec3<float>{0.0f, 7.0f, 0.0f}).length_squared() > 4.04f)
        {
            return p;
        }
    }
}

__device__ inline vec3<float> lamp_centre(const gpu::emitter& e)
{
    return e.kind == static_cast<int>(gpu::light_shape::quad) ? e.corner + 0.5f * e.u + 0.5f * e.v : e.corner;
}

// A point of lamp e for two numbers: q + a·u + b·v on a quad, the point at z = 1 − 2a, φ = 2πb on a sphere
__device__ inline vec3<float> lamp_point(const gpu::emitter& e, float a, float b)
{
    if (e.kind == static_cast<int>(gpu::light_shape::quad))
    {
        return e.corner + a * e.u + b * e.v;
    }
    const float z = 1.0f - 2.0f * a;
    const float s = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float sin_phi;
    float cos_phi;
    sincospif(2.0f * b, &sin_phi, &cos_phi);
    return e.corner + e.size * vec3<float>{s * cos_phi, s * sin_phi, z};
}

struct t3_sample
{
    gpu::quality_view v; // the lamps alone; the combination is each weight's argument, so one draw serves both
    t3_lobe lobe;
    bool cornell;     // the point set and lamps: cornell's, else light's
    bool light_drawn; // (a) the light sampler's direction, (b) the lobe's
    bool planted;     // (b), mis: w_L from the density of a conductor 5% wider than the lobe (the must-fail)

    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally, float& largest) const
    {
        // The point; the normal toward the centre of a lamp picked uniformly (B.md G18); the view direction uniform
        // above n, or for half the points aimed: the mirror image about n of the direction to a point of that lamp,
        // so that the lobe's peak meets it (decision 19)
        const vec3<float> x = t3_point(cornell, rng);
        const int count = v.lamps.count;
        const int k = ::min(static_cast<int>(gpu::random_real<float>(rng) * static_cast<float>(count)), count - 1);
        const gpu::emitter& e = v.lamps.lamps[k];
        vec3<float> n = gpu::random_unit_vector<float>(rng);
        n = gpu::dot(n, lamp_centre(e) - x) < 0.0f ? -n : n;
        vec3<float> wo = gpu::random_unit_vector<float>(rng);
        wo = gpu::dot(wo, n) < 0.0f ? -wo : wo;
        const bool aimed = gpu::random_real<float>(rng) < 0.5f;
        const float a = gpu::random_real<float>(rng);
        const float b = gpu::random_real<float>(rng);
        if (aimed)
        {
            const vec3<float> l = gpu::unit_vector(lamp_point(e, a, b) - x);
            const float c = gpu::dot(n, l);
            wo = c > 0.0f ? (2.0f * c) * n - l : wo;
        }
        gpu::hit_record<float> rec{};
        rec.p = x;
        rec.normal = n;
        rec.front_face = true;
        const gpu::ray<float> in{x + wo, -wo};
        const gpu::bsdf vertex = gpu::make_bsdf(lobe.m, &lobe.side, lobe.c, rec, in.direction);
        if (vertex.kind != lobe.expected)
        {
            ++tally[t3_nee_wrong]; // no density to weigh: G12, L9 say why
            ++tally[t3_mis_wrong];
            return;
        }
        const gpu::bvh_tracer tracer{v};
        float w_light[2] = {};
        float w_bsdf[2] = {};
        vec3<float> wi{};
        bool one_density = false;
        if (light_drawn)
        {
            gpu::light_sample ls;
            if (!gpu::sample_light(v, x, rng, ls))
            {
                return; // the back of light's quad: no sample
            }
            if (!finite(ls.wi) || !(ls.pdf > 0.0f) || !isfinite(ls.pdf))
            {
                ++tally[t3_w2_wrong]; // W2: a technique draws only where its density is positive
                return;
            }
            float p_bsdf = quiet_nan();
            const vec3<float> f = gpu::eval_bsdf(vertex, ls.wi, p_bsdf);
            if (!finite(f) || !(f.x >= 0.0f && f.y >= 0.0f && f.z >= 0.0f) || !(p_bsdf >= 0.0f) ||
                !isfinite(p_bsdf))
            {
                ++tally[t3_w2_wrong];
                return;
            }
            if (gpu::dot(to_double(vertex.n), to_double(ls.wi)) < -1e-6)
            {
                // Below the lobe's horizon its draws are absorbed: the material can't draw the direction, so its
                // density is 0 (W2), and its f too: f·Le = 0, W1 doesn't apply
                tally[t3_w2_wrong] += is_zero(f) && p_bsdf == 0.0f ? 0 : 1;
                return;
            }
            if (!positive(f))
            {
                return; // at the horizon: no light to weigh
            }
            const gpu::bvh_hit<float> hit = tracer.closest({x, ls.wi}, gpu::spawn{-1, false}, rng);
            if (hit.index != ls.skip || !hit.entering)
            {
                return; // the other lamp in front, or a ray past the lamp's edge: the lamps suite's T1
            }
            // As the integrator asks at a lamp hit (D.md §2.3, 2.): the ray's origin, unit direction and distance. A
            // p_L′ of 0 there gives w_B = 1 beside w_L: Σw > 1
            const float p_light = gpu::lamp_hit_pdf(v, hit.index, x, gpu::unit_vector(ls.wi),
                                                    hit.t * sqrtf(ls.wi.length_squared()));
            one_density = !(p_bsdf > 0.0f);
            for (int c = 0; c < 2; ++c)
            {
                w_light[c] = gpu::light_weight(t3_mode(c), ls.pdf, p_bsdf);
                // W2: no weight for the material where it can't draw the direction; then w_L alone must be 1
                w_bsdf[c] = one_density ? 0.0f : gpu::bsdf_weight(t3_mode(c), p_bsdf, p_light);
            }
            wi = ls.wi;
        }
        else
        {
            gpu::bsdf_sample s{nan_vector(), nan_vector(), quiet_nan(), true};
            if (!gpu::sample_bsdf(vertex, in, rec, rng, s))
            {
                return; // absorbed: below the horizon
            }
            if (s.delta || !finite(s.wi) || !finite(s.weight) || !(s.pdf > 0.0f) || !isfinite(s.pdf))
            {
                ++tally[t3_w2_wrong]; // W2: a GGX lobe has a density, positive where it draws
                return;
            }
            if (!positive(s.weight))
            {
                return; // f = 0: no light to weigh
            }
            float p_eval = quiet_nan();
            gpu::eval_bsdf(vertex, s.wi, p_eval);
            if (planted)
            {
                p_eval = gpu::ggx_pdf(vertex.n, vertex.wo, s.wi, 1.05f * vertex.alpha);
            }
            const gpu::bvh_hit<float> hit = tracer.closest({x, s.wi}, gpu::spawn{-1, false}, rng);
            if (hit.index < 0)
            {
                return; // no lamp: no light (the BVH holds the lamps alone)
            }
            const float p_light = gpu::lamp_hit_pdf(v, hit.index, x, gpu::unit_vector(s.wi),
                                                    hit.t * sqrtf(s.wi.length_squared()));
            if (!hit.entering)
            {
                // The back of light's quad, which glows on its front only: no light sample reaches it (sample_lamp
                // gives none from behind), so its density is 0, the light's weight 0, and the material's ray counts in
                // full (W2; Le = 0 there, so no image depends on it)
                const bool sound = p_light == 0.0f && gpu::light_weight(t3_mode(1), p_light, p_eval) == 0.0f &&
                                   gpu::bsdf_weight(t3_mode(0), s.pdf, p_light) == 1.0f &&
                                   gpu::bsdf_weight(t3_mode(1), s.pdf, p_light) == 1.0f;
                tally[t3_w2_wrong] += sound ? 0 : 1;
                return;
            }
            if (!(p_light > 0.0f) || !isfinite(p_light))
            {
                // A lamp's front in view, which the light sampler draws too: without its density the material's ray
                // would count in full beside the light sample's share, the light twice, in both modes
                ++tally[t3_tested];
                ++tally[t3_nee_wrong];
                ++tally[t3_mis_wrong];
                return;
            }
            for (int c = 0; c < 2; ++c)
            {
                w_bsdf[c] = gpu::bsdf_weight(t3_mode(c), s.pdf, p_light);
                w_light[c] = gpu::light_weight(t3_mode(c), p_light, p_eval);
            }
            wi = s.wi;
        }
        ++tally[one_density ? t3_one_density : t3_tested];
        // lights_only: 1 + 0, exactly
        const float nee_sum = w_light[0] + w_bsdf[0];
        tally[t3_nee_wrong] += unit_interval(w_light[0]) && unit_interval(w_bsdf[0]) && nee_sum == 1.0f ? 0 : 1;
        // The power heuristic: P(a, b) + P(b, a′) = 1 up to rounding and ½|ln(a/a′)|, a and a′ the one technique's two
        // densities (decision 20)
        const float off = fabsf(w_light[1] + w_bsdf[1] - 1.0f);
        bool close_to_one = off <= 2e-5f; // the smallest tolerance, 10⁻⁵ + 10⁻⁵: no h' to compute
        if (!close_to_one && off == off)
        {
            close_to_one = off <= 1e-5 + pdf_tolerance(vertex.alpha, recompute_half(vertex.wo, wi).vh);
        }
        tally[t3_mis_wrong] += unit_interval(w_light[1]) && unit_interval(w_bsdf[1]) && close_to_one ? 0 : 1;
        tally[t3_far] += off <= 1e-3f ? 0 : 1;
        largest = fmaxf(largest, off); // fmaxf passes over a NaN, which the tallies count
    }
};

template <class F>
__global__ void t3_kernel(F f, std::uint64_t seed, int count, unsigned long long* tallies, unsigned int* largest)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    gpu::pcg32 rng = gpu::sample_generator(seed, static_cast<std::uint64_t>(t), 0);
    unsigned long long local[t3_tallies] = {};
    float most = 0;
    for (int i = 0; i < count; ++i)
    {
        f(rng, local, most);
    }
    for (int k = 0; k < t3_tallies; ++k)
    {
        atomicAdd(&tallies[k], local[k]);
    }
    atomicMax(largest, __float_as_uint(most)); // a float ≥ 0 orders as its bits
}

struct t3_outcome
{
    std::array<std::int64_t, t3_tallies> tally{};
    float largest = 0; // the power heuristic's largest |Σw − 1| over the directions tested
    std::int64_t tested() const { return tally[t3_tested] + tally[t3_one_density]; }
};

// The harness's layout (gpu_test.cuh): 4096 threads, thread t drawing from sample_generator(seed, t, 0)
template <class F>
t3_outcome run_t3(const F& f, std::uint64_t seed, std::int64_t n)
{
    const gpu_test::device_buffer<unsigned long long> tallies(static_cast<std::size_t>(t3_tallies));
    const gpu_test::device_buffer<unsigned int> largest(1);
    t3_kernel<<<gpu_test::test_blocks, gpu_test::test_block_size>>>(f, seed, gpu_test::per_thread(n), tallies.get(),
                                                                     largest.get());
    gpu_test::finish_launch("a T3 kernel");
    t3_outcome out;
    const std::vector<unsigned long long> t = tallies.download();
    for (std::size_t k = 0; k < out.tally.size(); ++k)
    {
        out.tally[k] = static_cast<std::int64_t>(t[k]);
    }
    const unsigned int bits = largest.download()[0];
    std::memcpy(&out.largest, &bits, sizeof bits);
    return out;
}

GPU_TEST(ggx, t3)
{
    const std::int64_t n = ctx.samples(t3_samples);
    struct lobe_case
    {
        bool gloss;
        double alpha;
    };
    std::vector<lobe_case> lobes;
    for (const double alpha : t3_conductor_alphas)
    {
        lobes.push_back({false, alpha});
    }
    for (const double alpha : t3_gloss_alphas)
    {
        lobes.push_back({true, alpha});
    }
    std::uint64_t seed = t3_seed;
    for (const bool cornell : {true, false})
    {
        const std::string setup = cornell ? "cornell" : "light";
        const t3_scene lamps(cornell);
        ctx.exact("t3-" + setup + "-lamps", lamps.count(), cornell ? 1 : 2, 0);
        if (lamps.count() == 0)
        {
            seed += 2 * lobes.size(); // nothing to aim at: the check above fails
            continue;
        }
        std::int64_t w2_wrong[2] = {}; // conductor, gloss
        std::int64_t w2_drawn[2] = {};
        for (const lobe_case& lc : lobes)
        {
            const std::string name = "t3-" + setup + (lc.gloss ? "-gloss-" : "-conductor-") + number(lc.alpha);
            const t3_lobe lobe = make_t3_lobe(lc.gloss, lc.alpha);
            const std::uint64_t case_seed = seed;
            seed += 2;
            if (!lc.gloss && lobe.side.alpha < gpu::min_ggx_alpha)
            {
                ctx.info(name, "below min_ggx_alpha: the smooth mirror, a delta lobe with nothing to weigh");
                continue;
            }
            const t3_outcome a = run_t3(t3_sample{lamps.view(), lobe, cornell, true, false}, case_seed, n);
            const t3_outcome b = run_t3(t3_sample{lamps.view(), lobe, cornell, false, false}, case_seed + 1, n);
            // A technique that tested no direction would pass by testing nothing: one violation (decision 22)
            const std::int64_t empty = (a.tested() == 0 ? 1 : 0) + (b.tested() == 0 ? 1 : 0);
            ctx.count(name + "-nee", a.tally[t3_nee_wrong] + b.tally[t3_nee_wrong] + empty, 0,
                      a.tested() + b.tested());
            ctx.count(name + "-mis", a.tally[t3_mis_wrong] + b.tally[t3_mis_wrong] + empty, 0,
                      a.tested() + b.tested());
            ctx.info(name + "-tested",
                     std::format("{} light-drawn directions ({} of them the lobe can't draw), {} lobe-drawn ones at a "
                                 "lamp's front; mis: largest |w_L + w_B - 1| = {:.3g} (light-drawn) and {:.3g} "
                                 "(lobe-drawn)",
                                 a.tested(), a.tally[t3_one_density], b.tested(), a.largest, b.largest));
            const std::size_t kind = lc.gloss ? 1 : 0;
            w2_wrong[kind] += a.tally[t3_w2_wrong] + b.tally[t3_w2_wrong];
            w2_drawn[kind] += 2 * gpu_test::drawn(n);
        }
        ctx.count("t3-" + setup + "-conductor-w2", w2_wrong[0], 0, w2_drawn[0]);
        ctx.count("t3-" + setup + "-gloss-w2", w2_wrong[1], 0, w2_drawn[1]);
    }

    // The planted fault (decision 23): cornell's conductor at α = 0.3, lobe-drawn, w_L from a lobe 5% wider
    const t3_scene cornell(true);
    const t3_outcome planted =
        run_t3(t3_sample{cornell.view(), make_t3_lobe(false, 0.3), true, false, true}, t3_planted_seed, n);
    ctx.count("t3-planted-wider-pdf", planted.tally[t3_far] > 0 ? 0 : 1, 0, 1);
    ctx.info("t3-planted-wider-pdf-far", std::format("{} of {} directions with |w_L + w_B - 1| > 1e-3",
                                                     planted.tally[t3_far], planted.tested()));
}

// === Suite ggx-energy (Step 79): Turquin's energy compensation and its table =========================================

static_assert(gpu::ggx_energy_size == 32, "the table of B.md 3.8 is 32 by 32");
constexpr int table_side = 32;
constexpr std::size_t table_entries = 32 * 32;
constexpr std::uint64_t compensated_lobe_seed = 7900; // + case index, 0 to 34

// The node (i, j) of the table (B.md §3.8, §5.2.1, with columns even in √μ since Step 79's integration): r = j/31,
// α = r², cos θv = (i/31)², §5.1's frame. Column 0 (μ = 0) isn't integrated: see table_values.
quad_case table_node(int i, int j)
{
    const double r = j / 31.0;
    const double root = i / 31.0;
    quad_case k = make_case(r * r, root * root);
    k.roughness = static_cast<float>(r);
    return k;
}

using table_nodes = std::vector<std::array<int, 2>>; // (i, j)

table_nodes all_nodes() // in the table's order, [j·32 + i]
{
    table_nodes nodes;
    for (int j = 0; j < table_side; ++j)
    {
        for (int i = 0; i < table_side; ++i)
        {
            nodes.push_back({i, j});
        }
    }
    return nodes;
}

table_nodes every_fourth_node() // E2's 64: i and j in 0, 4, ..., 28
{
    table_nodes nodes;
    for (int j = 0; j < table_side; j += 4)
    {
        for (int i = 0; i < table_side; i += 4)
        {
            nodes.push_back({i, j});
        }
    }
    return nodes;
}

// E_ss with F = 1, the table's value: ∫ ggx_eval(F0 = (1, 1, 1)) dωl, the green channel
struct single_scattering
{
    __device__ void operator()(const quad_case& k, const h_node& node, double (&sum)[quad_values]) const
    {
        if (node.d_omega_l > 0)
        {
            float pdf = 0;
            sum[0] += static_cast<double>(gpu::ggx_eval(k.nf, k.vf, node.l, k.alpha, white(), pdf).y) * node.d_omega_l;
        }
    }
};

// The table at the nodes listed: row j = 0, the mirror, is exactly 1 (ggx_eval has no α = 0)
std::vector<double> table_values(const table_nodes& nodes, resolution r)
{
    std::vector<quad_case> cases;
    std::vector<std::size_t> where;
    // Row 0 (the mirror) and column 0 (a view grazing the surface) are 1 exactly. At μ → 0 the view's masking Λ(v)
    // grows without bound, so G2/G1(v) → 1, and every facet the view sees reflects upward (n·l = 2(v·h)(n·h) > 0): the
    // lobe keeps all of its light, as the table's nearest nodes approach (0.994 to 0.9998 at μ = 10⁻⁴). The quadrature
    // there, in float with v·h near 10⁻⁴, would be the table's least accurate node.
    for (std::size_t n = 0; n < nodes.size(); ++n)
    {
        if (nodes[n][0] > 0 && nodes[n][1] > 0)
        {
            cases.push_back(table_node(nodes[n][0], nodes[n][1]));
            where.push_back(n);
        }
    }
    const std::vector<integrals> q = h_quadrature(single_scattering{}, cases, r);
    std::vector<double> values(nodes.size(), 1.0);
    for (std::size_t c = 0; c < cases.size(); ++c)
    {
        values[where[c]] = q[c][0];
    }
    return values;
}

// f(k) for k from 0 to count - 1, a thread each: the suite's small kernels. A template, as every kernel of the tests:
// nvcc's host stub of a kernel that isn't one, in an unnamed namespace, draws MSVC's warning C4211.
template <class F>
__global__ void for_each_index_kernel(F f, int count)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count)
    {
        f(k);
    }
}

template <class F>
void for_each_index(const F& f, int count, const char* what)
{
    for_each_index_kernel<<<(count + 127) / 128, 128>>>(f, count);
    gpu_test::finish_launch(what);
}

// The table this program was compiled with (ggx_energy.cuh), which the conductor reads: a __device__ array, copied out
// by a kernel (whatever its qualifiers: __device__ or __constant__)
struct copy_energy_table
{
    float* out;
    __device__ void operator()(int k) const { out[k] = gpu::ggx_energy_table[k]; }
};

std::vector<float> compiled_table()
{
    const gpu_test::device_buffer<float> table(table_entries);
    for_each_index(copy_energy_table{table.get()}, static_cast<int>(table_entries), "copying the energy table");
    return table.download();
}

// A float as a literal that round-trips (9 significant digits, B.md §5.2.1) and compiles: "{:.9g}" writes 1 as "1",
// and "1f" is no C++ literal, so a number without a point or an exponent gets ".0" (decision 13)
std::string float_literal(float value)
{
    std::string text = std::format("{:.9g}", value);
    if (text.find_first_of(".eEni") == std::string::npos) // "n", "i": nan, inf (never in a valid table)
    {
        text += ".0";
    }
    return text + "f";
}

// ggx_energy.cuh's text: B.md §5.2.1's header lines and array, 32 values per row j, rows in order; a row's comment on a
// line of its own and its values 8 per line, to keep the file within 120 columns
std::string table_text(const std::vector<float>& table)
{
    std::string text = "// src/gpu/ggx_energy.cuh: generated by GpuTests ggx-energy --write; do not edit.\n"
                       "// E_ss(μ, r) of the GGX conductor with F = 1 (single scattering), for Turquin's compensation "
                       "(ggx.cuh)\n"
                       "#pragma once\n"
                       "namespace gpu\n"
                       "{\n"
                       "inline constexpr int ggx_energy_size = 32;\n"
                       "// [j * 32 + i]: roughness r = j / 31 (α = r²), √(cos θv) = i / 31; row 0 "
                       "(the mirror)\n// and column 0 (a grazing view) are 1\n"
                       "static __device__ const float ggx_energy_table[ggx_energy_size * ggx_energy_size] = {\n";
    for (int j = 0; j < table_side; ++j)
    {
        text += std::format("    /* r = {}/31 */\n", j);
        for (int i = 0; i < table_side; ++i)
        {
            const float value = table[static_cast<std::size_t>(j * table_side + i)];
            text += (i % 8 == 0 ? "    " : " ") + float_literal(value) + ",";
            text += i % 8 == 7 ? "\n" : "";
        }
    }
    text += "};\n}\n";
    return text;
}

// The 1024 numbers of a table file: the literals between ggx_energy_table's "= {" and its "}", comments left out
// (decision 14: B.md's "after the first {" would start at namespace gpu's brace, and the comments hold numbers too).
// Anything else is an error, exit 2.
std::vector<float> read_table(const std::filesystem::path& file)
{
    std::ifstream in(file, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error(std::format("can't read {}", file.string()));
    }
    std::ostringstream whole;
    whole << in.rdbuf();
    const std::string text = whole.str();
    std::string code; // the text without its comments
    for (std::size_t i = 0; i < text.size();)
    {
        if (text.compare(i, 2, "/*") == 0)
        {
            const std::size_t end = text.find("*/", i + 2);
            i = end == std::string::npos ? text.size() : end + 2;
            code += ' ';
        }
        else if (text.compare(i, 2, "//") == 0)
        {
            i = text.find('\n', i);
            i = i == std::string::npos ? text.size() : i;
        }
        else
        {
            code += text[i++];
        }
    }
    const std::size_t name = code.find("ggx_energy_table");
    const std::size_t equals = name == std::string::npos ? name : code.find('=', name);
    const std::size_t open = equals == std::string::npos ? equals : code.find('{', equals);
    const std::size_t close = open == std::string::npos ? open : code.find('}', open);
    if (close == std::string::npos)
    {
        throw std::runtime_error(std::format("{}: no ggx_energy_table = {{ ... }}", file.string()));
    }
    std::vector<float> values;
    std::istringstream items(code.substr(open + 1, close - open - 1));
    for (std::string item; std::getline(items, item, ',');)
    {
        const std::size_t first = item.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
            continue; // the empty item after the last comma
        }
        item = item.substr(first, item.find_last_not_of(" \t\r\n") - first + 1);
        char* end = nullptr;
        const double value = std::strtod(item.c_str(), &end);
        const std::string rest(end);
        if (end == item.c_str() || !(rest.empty() || rest == "f" || rest == "F"))
        {
            throw std::runtime_error(std::format("{}: '{}' is no float literal", file.string(), item));
        }
        values.push_back(static_cast<float>(value));
    }
    if (values.size() != table_entries)
    {
        throw std::runtime_error(
            std::format("{}: {} numbers in ggx_energy_table, not {}", file.string(), values.size(), table_entries));
    }
    return values;
}

// --check's file: as given, or found from the current folder up (find_repo_file), so that src\gpu\ggx_energy.cuh works
// from the build folder too
std::filesystem::path check_path(const std::string& name)
{
    const std::filesystem::path path(name);
    return path.is_relative() && !std::filesystem::exists(path) ? gpu_test::host::find_repo_file(name) : path;
}

// --write's file: as given, or in the repository's folder of that name when the current folder has none
std::filesystem::path write_path(const std::string& name)
{
    const std::filesystem::path path(name);
    const std::filesystem::path folder = path.parent_path();
    if (!path.is_relative() || folder.empty() || std::filesystem::is_directory(folder))
    {
        return path;
    }
    return gpu_test::host::find_repo_file(folder.generic_string()) / path.filename();
}

// B.md §3.8's lookup in double: bilinear in (√μ, r), clamped; shift = 1 reads every μ node one node up,
// T(min(i + 1, 31), j) (E7)
double table_lookup(const std::vector<float>& t, double mu, double r, int shift = 0)
{
    const double x = std::sqrt(std::clamp(mu, 0.0, 1.0)) * 31;
    const double y = std::clamp(r, 0.0, 1.0) * 31;
    const int i0 = std::min(static_cast<int>(std::floor(x)), 30);
    const int j0 = std::min(static_cast<int>(std::floor(y)), 30);
    const double fx = x - i0;
    const double fy = y - j0;
    const auto at = [&](int i, int j) {
        return static_cast<double>(t[static_cast<std::size_t>(j * table_side + std::min(i + shift, 31))]);
    };
    return (1 - fy) * ((1 - fx) * at(i0, j0) + fx * at(i0 + 1, j0)) +
           fy * ((1 - fx) * at(i0, j0 + 1) + fx * at(i0 + 1, j0 + 1));
}

// The largest |a - b| (a NaN stays)
double largest_difference(const std::vector<float>& a, const std::vector<float>& b)
{
    double worst = 0;
    for (std::size_t k = 0; k < a.size(); ++k)
    {
        const double d = std::abs(static_cast<double>(a[k]) - b[k]);
        worst = std::isnan(worst) || !(std::isnan(d) || d > worst) ? worst : d; // a NaN stays
    }
    return worst;
}

// E6: row 0 exactly 1, every value in (0, 1 + 10⁻⁶]
std::int64_t table_faults(const std::vector<float>& t)
{
    std::int64_t faults = 0;
    for (std::size_t k = 0; k < t.size(); ++k)
    {
        const bool limit = k < static_cast<std::size_t>(table_side) || k % table_side == 0; // row 0, column 0
        faults += (limit && t[k] != 1.0f) || !(t[k] > 0 && t[k] <= 1 + 1e-6) ? 1 : 0;
    }
    return faults;
}

// E1, E2, E6, and --write / --check
GPU_TEST(ggx_energy, table)
{
    const std::string* write = ctx.argument("--write");
    const std::string* check = ctx.argument("--check");
    if (write && ctx.quick())
    {
        throw std::runtime_error("ggx-energy --write needs the full quadrature: drop --quick");
    }
    if (write && check)
    {
        throw std::runtime_error("ggx-energy takes --write or --check, not both");
    }
    const std::vector<float> compiled = compiled_table();
    std::vector<float> file_table;
    if (check)
    {
        file_table = read_table(check_path(*check));
        ctx.info("table-compiled-in",
                 std::format("largest |compiled-in - {}| = {:.3g}", *check, largest_difference(compiled, file_table)));
    }

    if (ctx.quick())
    {
        // Every kernel at 64 nodes and 256 × 64 points, no verdict (B.md §5.2: E1-E3 and E7 skip)
        const table_nodes nodes = every_fourth_node();
        const std::vector<double> values = table_values(nodes, quick_resolution);
        double worst = 0;
        for (std::size_t n = 0; n < nodes.size(); ++n)
        {
            const double d = std::abs(values[n] - compiled[static_cast<std::size_t>(nodes[n][1] * 32 + nodes[n][0])]);
            worst = std::isnan(worst) || !(std::isnan(d) || d > worst) ? worst : d; // a NaN stays
        }
        ctx.info("table", std::format("quick: 64 nodes at 256 × 64, largest |Δ| from the compiled-in table {:.3g}, "
                                      "no verdict",
                                      worst));
    }
    else
    {
        const std::vector<double> full = table_values(all_nodes(), full_resolution);
        std::vector<float> regenerated;
        for (const double v : full)
        {
            regenerated.push_back(static_cast<float>(v));
        }
        if (write)
        {
            const std::filesystem::path path = write_path(*write);
            std::ofstream out(path, std::ios::binary);
            out << table_text(regenerated);
            out.close();
            if (!out)
            {
                throw std::runtime_error(std::format("can't write {}", path.string()));
            }
            ctx.info("table-written", std::format("{}; largest |regenerated - compiled-in| = {:.3g} (rebuild if > 0)",
                                                  path.string(), largest_difference(regenerated, compiled)));
        }
        else
        {
            // E1: the regenerated table against the file's (--check) or the compiled-in one
            ctx.exact("table", largest_difference(regenerated, check ? file_table : compiled), 0, 1e-6);
        }

        // E2: half the resolution on every fourth node, within 2·10⁻⁴ (B.md said 2·10⁻⁵; the ggx suite's
        // quadrature-resolution-energy takes 2·10⁻⁴ too): E_ss's integrand stops at the horizon, where the midpoint
        // rule's error is O(1/N); measured 7.2·10⁻⁵ with the columns even in √μ
        const table_nodes nodes = every_fourth_node();
        const std::vector<double> half = table_values(nodes, half_resolution);
        double worst = 0;
        for (std::size_t n = 0; n < nodes.size(); ++n)
        {
            const double d = std::abs(half[n] - full[static_cast<std::size_t>(nodes[n][1] * 32 + nodes[n][0])]);
            worst = std::isnan(worst) || !(std::isnan(d) || d > worst) ? worst : d; // a NaN stays
        }
        ctx.exact("table-accuracy", worst, 0, 2e-4);
    }

    // E6, on the compiled-in table (and the file's)
    ctx.count("mirror-row", table_faults(compiled), 0, static_cast<std::int64_t>(table_entries));
    if (check)
    {
        ctx.count("mirror-row-file", table_faults(file_table), 0, static_cast<std::int64_t>(table_entries));
    }
}

// [0] E_ms = ∫ eval_bsdf of the conductor from make_bsdf (metal (1, 1, 1), side entry {conductor, r², r}),
// [1] E_ss = ∫ ggx_eval with F0 = (1, 1, 1)
struct compensated_integrands
{
    __device__ void operator()(const quad_case& k, const h_node& node, double (&sum)[quad_values]) const
    {
        if (!(node.d_omega_l > 0))
        {
            return;
        }
        gpu::hit_record<float> rec{};
        rec.normal = k.nf;
        rec.front_face = true;
        const gpu::microfacet side{gpu::microfacet_kind::conductor, k.alpha, k.roughness};
        const gpu::bsdf b = gpu::make_bsdf(metal(white()), &side, white(), rec, -k.vf);
        float pdf = 0;
        sum[0] += static_cast<double>(gpu::eval_bsdf(b, node.l, pdf).y) * node.d_omega_l;
        sum[1] += static_cast<double>(gpu::ggx_eval(k.nf, k.vf, node.l, k.alpha, white(), pdf).y) * node.d_omega_l;
    }
};

// E3, E7
GPU_TEST(ggx_energy, compensated)
{
    const bool quick = ctx.quick();
    std::vector<quad_case> cases;
    for (const double alpha : grid_alphas)
    {
        for (const int theta : grid_thetas)
        {
            cases.push_back(make_case(alpha, cos_degrees(theta)));
        }
    }
    const std::vector<integrals> q =
        h_quadrature(compensated_integrands{}, cases, quick ? quick_resolution : full_resolution);
    const std::vector<float> compiled = compiled_table();
    double planted = 0; // E7: the largest |E_ms - 1| with the table shifted one μ node, θv ≤ 80°
    for (std::size_t k = 0; k < grid_size; ++k)
    {
        const std::string name = "compensated-" + grid_name(k);
        const int theta = grid_thetas[k % grid_thetas.size()];
        if (cases[k].alpha < gpu::min_ggx_alpha)
        {
            ctx.info(name, "the delta lobe below min_ggx_alpha: nothing to integrate");
            continue;
        }
        const double e = q[k][0];
        if (quick)
        {
            ctx.info(name, std::format("{:.9g} (quick: no verdict)", e));
        }
        else if (theta <= 80)
        {
            ctx.exact(name, e, 1, 3e-3);
        }
        else
        {
            // Grazing: the table's coarseness at α → 1 is reported, with a sanity bound (B.md §8.7)
            ctx.info(name, std::format("{:.6f} at θv = 89°", e));
            ctx.exact(name, e, 1, 2e-2);
        }
        if (theta <= 80)
        {
            const double shifted = q[k][1] / table_lookup(compiled, cos_degrees(theta), cases[k].roughness, 1);
            const double d = std::abs(shifted - 1);
            planted = d <= planted ? planted : d;
        }
    }
    if (quick)
    {
        ctx.info("planted-shift", "quick: no verdict");
    }
    else
    {
        ctx.info("planted-shift-largest", std::format("{:.6f}", planted));
        ctx.count("planted-shift", planted > 1e-2 ? 0 : 1, 0, 1); // the fault accepted: max |E_ms - 1| ≤ 10⁻²
    }
}

// The points of E4 and of the lookup check: (μ, r) on 65 × 65 points of [0, 1]² for three F0, then the table's nodes
struct factor_point
{
    float mu;
    float r;
    vec3<float> f0;
};

struct compensation_at
{
    const factor_point* points;
    vec3<float>* factors;
    __device__ void operator()(int k) const
    {
        factors[k] = gpu::ggx_compensation(points[k].f0, points[k].mu, points[k].r);
    }
};

// E4: ggx_compensation finite and ≥ 1, exactly 1 where F0 = 0; and (not in B.md's list) equal to B.md §3.8's lookup of
// the compiled-in table, 1 + F0 (1 - E)/E, to 10⁻⁵
GPU_TEST(ggx_energy, factor)
{
    std::vector<factor_point> points;
    const vec3<float> f0s[] = {{0.0f, 0.04f, 0.5f}, {0.913f, 1.0f, 0.0f}, {0.04f, 0.5f, 1.0f}};
    for (const vec3<float>& f0 : f0s)
    {
        for (int b = 0; b <= 64; ++b)
        {
            for (int a = 0; a <= 64; ++a)
            {
                points.push_back({static_cast<float>((a / 64.0) * (a / 64.0)), b / 64.0f, f0}); // even in √μ
            }
        }
    }
    for (int j = 0; j < table_side; ++j)
    {
        for (int i = 0; i < table_side; ++i)
        {
            points.push_back({static_cast<float>((i / 31.0) * (i / 31.0)), j / 31.0f, f0s[2]});
        }
    }
    const gpu_test::device_buffer<factor_point> on_gpu(points);
    const gpu_test::device_buffer<vec3<float>> out(points.size());
    for_each_index(compensation_at{on_gpu.get(), out.get()}, static_cast<int>(points.size()), "ggx_compensation");
    const std::vector<vec3<float>> factors = out.download();
    const std::vector<float> compiled = compiled_table();
    std::int64_t faults = 0;
    std::int64_t off = 0;
    for (std::size_t k = 0; k < points.size(); ++k)
    {
        const double e = table_lookup(compiled, points[k].mu, points[k].r);
        for (int c = 0; c < 3; ++c)
        {
            const double f0 = channel(points[k].f0, c);
            const double f = channel(factors[k], c);
            faults += std::isfinite(f) && f >= 1 && (f0 != 0 || f == 1) ? 0 : 1;
            const double expected = 1 + f0 * (1 - e) / e;
            off += std::abs(f - expected) <= 1e-5 * expected ? 0 : 1;
        }
    }
    ctx.count("factor", faults, 0, static_cast<std::int64_t>(3 * points.size()));
    ctx.count("factor-lookup", off, 0, static_cast<std::int64_t>(3 * points.size()));
}

// E5: the compensated conductor through bsdf.cuh, weight = eval/pdf and eval_bsdf's pdf = ggx_pdf (G4's tolerance)
struct compensated_lobe_off
{
    quad_case k;
    bool pdf_check;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        gpu::hit_record<float> rec{};
        rec.normal = k.nf;
        rec.front_face = true;
        const gpu::microfacet side{gpu::microfacet_kind::conductor, k.alpha, k.roughness};
        const gpu::bsdf b = gpu::make_bsdf(metal(test_f0()), &side, test_f0(), rec, -k.vf);
        gpu::bsdf_sample s{nan_vector(), nan_vector(), quiet_nan(), true};
        if (!gpu::sample_bsdf(b, {k.vf, -k.vf}, rec, rng, s))
        {
            return false;
        }
        float pdf = quiet_nan();
        const vec3<float> e = gpu::eval_bsdf(b, s.wi, pdf);
        if (pdf_check)
        {
            // ggx_pdf is ggx_eval's density, within G4's tolerance (decision 12): the two inlined copies may round the
            // half vector apart, which a narrow lobe's D turns into a relative error of up to 10⁻⁶/(α v·h)
            const double expected = gpu::ggx_pdf(b.n, b.wo, s.wi, b.alpha);
            const double tolerance = pdf_tolerance(b.alpha, recompute_half(b.wo, s.wi).vh);
            return !(isfinite(pdf) && ::fabs(pdf - expected) <= tolerance * ::fabs(expected));
        }
        return !weight_matches(b.n, b.wo, s.wi, b.albedo, s.weight, e, pdf);
    }
};

GPU_TEST(ggx_energy, lobe)
{
    const std::int64_t n = ctx.samples(pair_samples);
    std::int64_t weight_off = 0;
    std::int64_t pdf_off = 0;
    std::int64_t all = 0;
    for (std::size_t a = 0; a < grid_alphas.size(); ++a)
    {
        for (std::size_t t = 0; t < grid_thetas.size(); ++t)
        {
            const std::size_t k = a * grid_thetas.size() + t;
            const quad_case c = make_case(grid_alphas[a], cos_degrees(grid_thetas[t]));
            if (c.alpha < gpu::min_ggx_alpha)
            {
                continue;
            }
            weight_off += gpu_test::device_count(compensated_lobe_off{c, false}, compensated_lobe_seed + k, n);
            pdf_off += gpu_test::device_count(compensated_lobe_off{c, true}, compensated_lobe_seed + k, n);
            all += gpu_test::drawn(n);
        }
    }
    ctx.count("lobe-weight-eval", weight_off, 0, all);
    ctx.count("lobe-pdf", pdf_off, 0, all);
}
}
