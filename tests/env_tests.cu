// Suites env-map and env-sampling: the HDR environment map as the quality kernel sees it (Step 82), and as a light it
// samples by its brightness (Step 83, the second half of this file, with its own notes and criteria). Both are
// written from the spec alone (out/m4/spec/C.md, sections 5.1 and 5.2; shared.md 0-5), against the D2 declarations.
//
// Suite env-map. On the host, build_environment's
// tables (environment_data.h): the texels' radiance exactly as the file gave it, each cell's density Y/S per
// steradian, the rows' cosines, and the refusals. On the GPU, the lookups (environment.cuh, background.cuh): the one
// map from a direction to a cell, its pinned orientation (a mirrored map would light a scene from the wrong side and
// still look plausible), the nearest texel and the bilinear one. Every check is exact or a count: the map isn't sampled
// until Step 83 (suite env-sampling). The spec is out/m4/spec/C.md, section 5.1 (checks M1-M13).
//
// The maps: the two assets as host::load_image gives them; r300, 300×150 texels whose channels are 10^U each, U
// uniform in [-3, 3], with 1% of the texels black (a host SplitMix64 seeded 8200): up to six decades between
// neighbours, the hardest case for a filter; k64, 64×32 texels of 1.0; b64, 64×32 of 0; u1, one texel (0.3, 0.6, 0.9);
// and for the orientation checks, 1024×512 texels of 1.0. The device views are made by hand from the tables, as C.md
// §2.1 says quality.cu makes them (cos and sin of the yaw in double, rounded to float).
//
// Decisions where the spec leaves room, or can't hold as written in float (for Claude to confirm):
// 1. M6(a), bilinear = nearest at a texel's centre: the spec's tolerance, 10^-5 of max(nearest, 10^-6·max texel),
//    can't hold next to a much brighter neighbour. The filter computes x = u·W - 0.5 from atan2f of a float direction,
//    so at a centre fx is not 0 but up to about 3·10^-4 of a texel (atan2f's few ulp of φ' ≈ 7·10^-7 rad are
//    1.2·10^-4 of a texel at W = 1024, and u and u·W round by 6·10^-5 more); a neighbour 10^6 times brighter (r300)
//    then moves the value by far more than 10^-5 of it. The verdict adds the position error's term:
//    |F - N| ≤ 10^-5·max(N, 10^-6·M) + 2·10^-3·D, D the largest difference between the texel and its eight
//    neighbours (wrapped in u, clamped in θ), per channel. A half-texel offset (fx = 0.5) still moves most texels by
//    D/2, 250 times that. An INFO line counts the samples beyond the spec's term alone. The nearest value must also be
//    the centre's own texel, bitwise.
// 2. M6(c), the seam: the lerp's step between φ' = π - δ and -π + δ is exactly 2·10^-3·ΔT, but atan2f near ±π is good
//    to about 7·10^-7 rad, 3% of δ (2.1·10^-5 rad at W = 300), and u and x round by 2·10^-5 of a texel on each side:
//    the jump can exceed 2·10^-3·ΔT by 10-15%. The verdict takes 3·10^-3·ΔT + 10^-6·M, ΔT the largest
//    |T(W-1, j) - T(0, j)| of the two rows the θ interpolation blends; a missing wrap jumps by ΔT itself, 300 times
//    the bound. An INFO line counts the samples beyond 2·10^-3.
// 3. M6(d), the rows past the first and last centres: on r300, 10^4 directions with θ within 0.9 of half a row from
//    either pole (never near the row's centre, where rounding could blend the next row by a hair). The reference is the
//    row's own u-lerp in double from the same float direction; the tolerance is 10^-5·max(|R|, 10^-6·M) + 2·10^-3·D,
//    D the largest of the three column differences around the lerp, for the reason of decision 1.
// 4. M6(e) on r300 at yaw 0, 10^4 directions; M7 on r300 at yaw 37° and scale 2.5 (every member of the view in play),
//    the camera's sky from (1, 1, 1) at the bottom to (0.5, 0.7, 1) at the top.
// 5. M5 and M13 use the 1024×512 map of 1.0 (the mapping reads no texel; independent of the assets). M13 also prints
//    the mismatches of the unmirrored test-side mapping through the same checker (INFO, 0 expected), so a broken
//    checker can't make the must-fail pass.
// 6. M10: one check per case. The 2049×1024 and 2048×1025 maps are complete (constant 0.5, 3WH floats), so their size
//    is their only fault. Added from §2.1's list: 1×0, scale NaN, scale +∞, yaw +∞. A refusal is std::invalid_argument
//    with a non-empty what(); any other outcome fails. The two maps it accepts must come back with their dimensions.
// 7. M1 skips b64 (S = 0, every pdf 0: M11 checks it). Its ΣΩ check is named normalization-<map>-solid-angle. Both
//    sums are Neumaier's compensated sums, so the reference is as exact as double allows.
// 8. M3: S is recomputed with a compensated sum; total within 10^-12 relative (exactly 0 for b64); pdf within one ulp
//    (bit patterns, non-negative floats) of float(Y/S); S = 0: every pdf 0.
// 9. M4 under --quick: 10^4 cells spread evenly over each map (cell k·N/10^4).
// 10. M9's "zero texels" are texels with all three channels 0. M11 is split into one host check and four device
//    counts; M12 also checks that bins.size() is 2^log2_bins, or 0 with log2_bins 0 (true in Step 82 and 83 alike).
// 11. Map names in check names are lowercase: studio, kloofendal, r300, k64, u1, b64.
// 12. Interface assumed from D2 (C.md §1): everything in namespace gpu; build_environment and the tables in
//    environment_data.h, the lookups in environment.cuh, background() and the sampling functions in background.cuh.
//
// The step's criteria (C.md §7, Step 82; Claude runs them; $S = -w 480 --height 270):
//   1. $GT env-map: exit 0.
//   2. $G env-furnace $S -s 64 --seed 3 -o ef.pfm; $G furnace $S -s 64 --seed 3 --quality-kernel -o f.pfm;
//      $PS same ef.pfm f.pfm --ulps 1 --max-share 0.0001 (identical bytes expected).
//   3. studio and sunlit -p material -w 160 --height 90 -s 16: $PS stats --finite, and two runs identical (SHA-256).
//   4. Get-FileHash -Algorithm SHA256 of both assets equals C.md §4; git check-attr binary assets/*.hdr: set.
//   5. HOST: env-furnace, studio, sunlit in the lists; studio --fp64 and studio -p mixture refused (C.md §7), exit 1.
//   6. Sanitizer: env-furnace, studio, $GT env-map --quick. CARRY recorded: env-furnace, studio -p material $S -s 64
//      --seed 3.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "background.cuh"
#include "bsdf.cuh"
#include "bvh.cuh"
#include "camera.cuh"
#include "environment.cuh"
#include "environment_data.h"
#include "gpu_test.cuh"
#include "gpu_test.h"
#include "host_services.h"
#include "lamps.cuh"
#include "light_sampler.cuh"
#include "material.cuh"
#include "mis.cuh"
#include "pdf.cuh"
#include "quality_view.cuh"
#include "random.cuh"
#include "renderer.h"
#include "statistics.h"
#include "test_scene.h"
#include "tracer.cuh"

namespace
{
using gpu::environment_view;
using gpu::light_combination;
using gpu::vec3;

constexpr const char* studio_file = "assets/studio_small_09_1k.hdr";
constexpr const char* kloofendal_file = "assets/kloofendal_48d_partly_cloudy_puresky_1k.hdr";
constexpr std::uintmax_t studio_bytes = 1'615'248;
constexpr std::uintmax_t kloofendal_bytes = 1'435'119;

constexpr std::int64_t small_samples = 10'000;
constexpr std::int64_t constant_samples = 1'000'000;

// --- Host helpers ----------------------------------------------------------------------------------------------------

template <class... Args>
std::string printed(const char* format, Args... args)
{
    char buffer[256];
    std::snprintf(buffer, sizeof buffer, format, args...);
    return buffer;
}

std::uint32_t bits(float x) { return std::bit_cast<std::uint32_t>(x); }

// Neumaier's compensated sum (Neumaier 1974: Kahan's, kept right when a term is larger than the sum so far). The
// rounding errors of half a million additions stay at about one rounding of the result, so a reference sum is as exact
// as double allows, whatever order the implementation added its terms in.
class exact_sum
{
public:
    void add(double x)
    {
        const double t = sum_ + x;
        compensation_ += std::abs(sum_) >= std::abs(x) ? (sum_ - t) + x : (x - t) + sum_;
        sum_ = t;
    }
    double value() const { return sum_ + compensation_; }

private:
    double sum_ = 0;
    double compensation_ = 0;
};

// Rec. 709's luminance, in double, as C.md §2.1 defines Y
double luminance(float r, float g, float b) { return 0.2126 * r + 0.7152 * g + 0.0722 * b; }

// The solid angle of a cell of row j from double cosines: Ω_j = (2π/W)(cos(πj/H) - cos(π(j + 1)/H))
std::vector<double> row_solid_angles(int width, int height)
{
    std::vector<double> omega(static_cast<std::size_t>(height));
    for (int j = 0; j < height; ++j)
    {
        omega[static_cast<std::size_t>(j)] = 2 * std::numbers::pi / width *
                                             (std::cos(std::numbers::pi * j / height) -
                                              std::cos(std::numbers::pi * (j + 1) / height));
    }
    return omega;
}

gpu::image_desc constant_image(int width, int height, float r, float g, float b)
{
    gpu::image_desc image{.width = width, .height = height};
    const std::size_t texels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    image.rgb.resize(3 * texels);
    for (std::size_t k = 0; k < texels; ++k)
    {
        image.rgb[3 * k] = r;
        image.rgb[3 * k + 1] = g;
        image.rgb[3 * k + 2] = b;
    }
    return image;
}

// SplitMix64 (Steele, Lea and Flood 2014) on the host: the synthetic map's numbers, the same with every compiler
struct host_generator
{
    std::uint64_t state;
    std::uint64_t next()
    {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31u);
    }
    double uniform() { return static_cast<double>(next() >> 11u) * 0x1p-53; }
};

// r300: per texel one draw for black (1%), then three channels of 10^U, U uniform in [-3, 3]
gpu::image_desc r300_image()
{
    constexpr int width = 300;
    constexpr int height = 150;
    gpu::image_desc image{.width = width, .height = height};
    image.rgb.resize(3 * static_cast<std::size_t>(width) * height);
    host_generator g{8200};
    for (std::size_t k = 0; k < static_cast<std::size_t>(width) * height; ++k)
    {
        const bool black = g.uniform() < 0.01;
        for (std::size_t c = 0; c < 3; ++c)
        {
            const double value = std::pow(10.0, -3 + 6 * g.uniform());
            image.rgb[3 * k + c] = black ? 0.0f : static_cast<float>(value);
        }
    }
    return image;
}

gpu::image_desc load_asset(const char* file)
{
    return gpu_test::host::load_image(gpu_test::host::find_repo_file(file));
}

// A map, the description it was built from, and build_environment's tables of it
struct map_case
{
    std::string name;
    gpu::environment_desc desc;
    gpu::environment_tables tables;
};

map_case make_case(std::string name, gpu::image_desc image)
{
    map_case m;
    m.name = std::move(name);
    m.desc.image = std::move(image);
    m.tables = gpu::build_environment(m.desc);
    return m;
}

// Each map is made once per run (a throw leaves it unmade: the next call tries again)
const gpu::image_desc& studio_image()
{
    static const gpu::image_desc image = load_asset(studio_file);
    return image;
}
const gpu::image_desc& kloofendal_image()
{
    static const gpu::image_desc image = load_asset(kloofendal_file);
    return image;
}
const map_case& studio()
{
    static const map_case m = make_case("studio", studio_image());
    return m;
}
const map_case& kloofendal()
{
    static const map_case m = make_case("kloofendal", kloofendal_image());
    return m;
}
const map_case& r300()
{
    static const map_case m = make_case("r300", r300_image());
    return m;
}
const map_case& k64()
{
    static const map_case m = make_case("k64", constant_image(64, 32, 1, 1, 1));
    return m;
}
const map_case& b64()
{
    static const map_case m = make_case("b64", constant_image(64, 32, 0, 0, 0));
    return m;
}
const map_case& u1()
{
    static const map_case m = make_case("u1", constant_image(1, 1, 0.3f, 0.6f, 0.9f));
    return m;
}
// The orientation checks' map: 1024×512, as the assets
const map_case& pinned_map()
{
    static const map_case m = make_case("p1024", constant_image(1024, 512, 1, 1, 1));
    return m;
}

std::vector<const map_case*> all_maps() { return {&studio(), &kloofendal(), &r300(), &k64(), &u1(), &b64()}; }

// The tables have the sizes their dimensions call for (the device kernels index them by those)
bool shaped(const gpu::environment_tables& t)
{
    if (t.width < 1 || t.height < 1)
    {
        return false;
    }
    const std::size_t texels = static_cast<std::size_t>(t.width) * static_cast<std::size_t>(t.height);
    return t.texels.size() == texels && t.row_cos.size() == static_cast<std::size_t>(t.height) + 1;
}

// The largest value of any channel of the map (the floor of the relative tolerances)
float largest_value(const gpu::environment_tables& t)
{
    float largest = 0;
    for (const gpu::env_texel& texel : t.texels)
    {
        largest = std::max({largest, texel.r, texel.g, texel.b});
    }
    return largest;
}

// The tables on the GPU and the view of them that quality.cu's holder makes (C.md §2.1), for any yaw and scale
class device_environment
{
public:
    explicit device_environment(const gpu::environment_tables& t)
        : texels_(t.texels), bins_(t.bins), row_cos_(t.row_cos), width_(t.width), height_(t.height),
          log2_bins_(t.log2_bins)
    {
    }

    environment_view view(double yaw_degrees = 0, double scale = 1) const
    {
        environment_view v;
        v.texels = texels_.get();
        v.bins = bins_.size() > 0 ? bins_.get() : nullptr;
        v.row_cos = row_cos_.get();
        v.width = width_;
        v.height = height_;
        v.log2_bins = log2_bins_;
        const double alpha = yaw_degrees * std::numbers::pi / 180;
        v.cos_yaw = static_cast<float>(std::cos(alpha));
        v.sin_yaw = static_cast<float>(std::sin(alpha));
        v.scale = static_cast<float>(scale);
        return v;
    }

private:
    gpu_test::device_buffer<gpu::env_texel> texels_;
    gpu_test::device_buffer<gpu::alias_bin> bins_;
    gpu_test::device_buffer<float> row_cos_;
    int width_;
    int height_;
    int log2_bins_;
};

// --- Device helpers --------------------------------------------------------------------------------------------------

// Every index k in [0, count) once, over a grid-stride loop: how many of them p(k) finds wrong
template <class P>
__global__ void index_count_kernel(P p, std::int64_t count, unsigned long long* found)
{
    unsigned long long n = 0;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < count; k += stride)
    {
        n += p(k) ? 1 : 0;
    }
    atomicAdd(found, n);
}

template <class P>
std::int64_t count_indices(const P& p, std::int64_t count)
{
    gpu_test::device_buffer<unsigned long long> found(1);
    const auto blocks = static_cast<int>(
        std::clamp<std::int64_t>((count + gpu_test::test_block_size - 1) / gpu_test::test_block_size, 1, 256));
    index_count_kernel<<<blocks, gpu_test::test_block_size>>>(p, count, found.get());
    gpu_test::finish_launch("an index kernel");
    return static_cast<std::int64_t>(found.download()[0]);
}

// One thread runs f() (a handful of lookups whose results the host compares)
template <class F>
__global__ void single_kernel(F f)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        f();
    }
}

template <class F>
void run_once(const F& f, const char* what)
{
    single_kernel<<<1, 1>>>(f);
    gpu_test::finish_launch(what);
}

__device__ inline bool bits_equal(float a, float b) { return __float_as_uint(a) == __float_as_uint(b); }
__device__ inline bool bits_equal(const vec3<float>& a, const vec3<float>& b)
{
    return bits_equal(a.x, b.x) && bits_equal(a.y, b.y) && bits_equal(a.z, b.z);
}
__device__ inline float channel(const vec3<float>& v, int c) { return c == 0 ? v.x : c == 1 ? v.y : v.z; }
__device__ inline int wrap_column(int i, int width)
{
    const int r = i % width;
    return r < 0 ? r + width : r;
}
__device__ inline int clamp_row(int j, int height) { return ::min(::max(j, 0), height - 1); }
// Channel c of texel (i, j), in double
__device__ inline double texel_channel(const environment_view& e, int i, int j, int c)
{
    const gpu::env_texel& t = e.texels[static_cast<std::size_t>(j) * e.width + i];
    return c == 0 ? t.r : c == 1 ? t.g : t.b;
}
// The direction at azimuth φ' and polar angle θ in map space (C.md §2.2's inverse), from double, rounded to float
__device__ inline vec3<float> map_direction(double phi, double theta)
{
    const double s = ::sin(theta);
    return {static_cast<float>(s * ::cos(phi)), static_cast<float>(::cos(theta)), static_cast<float>(s * ::sin(phi))};
}
// A random direction of random length (the lookups normalize nothing: any non-zero length gives the same cell)
__device__ inline vec3<float> random_direction(gpu::pcg32& rng)
{
    const vec3<float> unit = gpu::random_unit_vector<float>(rng);
    const float length = 0.25f + 4.0f * gpu::random_real<float>(rng);
    return length * unit;
}

// --- The mappings the orientation checker takes ----------------------------------------------------------------------

// The implementation's: environment.cuh
struct implementation_mapping
{
    __device__ vec3<float> to_map(const environment_view& e, const vec3<float>& d) const { return gpu::to_map(e, d); }
    __device__ vec3<float> from_map(const environment_view& e, const vec3<float>& m) const
    {
        return gpu::from_map(e, m);
    }
    __device__ int cell(const environment_view& e, const vec3<float>& d) const { return gpu::env_cell(e, d); }
};

// The spec's (C.md §2.2) in double, test-side; mirrored: the planted fault u' = 0.5 - φ'/(2π), the books' sphere_uv's
// direction, which would show the map back to front
struct reference_mapping
{
    bool mirrored = false;

    __device__ vec3<double> to_map_exact(const environment_view& e, const vec3<float>& d) const
    {
        const double c = e.cos_yaw;
        const double s = e.sin_yaw;
        return {c * d.x + s * d.z, static_cast<double>(d.y), -s * d.x + c * d.z};
    }
    __device__ vec3<float> to_map(const environment_view& e, const vec3<float>& d) const
    {
        const vec3<double> m = to_map_exact(e, d);
        return {static_cast<float>(m.x), static_cast<float>(m.y), static_cast<float>(m.z)};
    }
    __device__ vec3<float> from_map(const environment_view& e, const vec3<float>& m) const
    {
        const double c = e.cos_yaw;
        const double s = e.sin_yaw;
        return {static_cast<float>(c * m.x - s * m.z), m.y, static_cast<float>(s * m.x + c * m.z)};
    }
    __device__ int cell(const environment_view& e, const vec3<float>& d) const
    {
        const vec3<double> m = to_map_exact(e, d);
        const double phi = ::atan2(m.z, m.x);
        const double u = mirrored ? 0.5 - phi / (2 * std::numbers::pi) : 0.5 + phi / (2 * std::numbers::pi);
        const double t = ::atan2(::sqrt(m.x * m.x + m.z * m.z), m.y) / std::numbers::pi;
        const int i = ::min(::max(static_cast<int>(::floor(u * e.width)), 0), e.width - 1);
        const int j = ::min(::max(static_cast<int>(::floor(t * e.height)), 0), e.height - 1);
        return j * e.width + i;
    }
};

// --- M9: the assets --------------------------------------------------------------------------------------------------

GPU_TEST(env_map, assets)
{
    struct asset
    {
        const char* name;
        const char* file;
        std::uintmax_t bytes;
    };
    for (const asset& a : {asset{"studio", studio_file, studio_bytes}, asset{"kloofendal", kloofendal_file,
                                                                             kloofendal_bytes}})
    {
        const std::string prefix = std::string("assets-") + a.name;
        std::filesystem::path file;
        try
        {
            file = gpu_test::host::find_repo_file(a.file);
        }
        catch (const std::exception&)
        {
        }
        ctx.count(prefix + "-found", file.empty() ? 1 : 0, 0, 1);
        if (file.empty())
        {
            continue;
        }
        ctx.exact(prefix + "-bytes", static_cast<double>(std::filesystem::file_size(file)),
                  static_cast<double>(a.bytes), 0);
        const gpu::image_desc& image = std::string_view(a.name) == "studio" ? studio_image() : kloofendal_image();
        ctx.exact(prefix + "-width", image.width, 1024, 0);
        ctx.exact(prefix + "-height", image.height, 512, 0);
        const std::size_t texels = static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height);
        std::int64_t bad = image.rgb.size() == 3 * texels ? 0 : 1;
        std::int64_t black = 0;
        for (std::size_t k = 0; bad == 0 && k < texels; ++k)
        {
            const float r = image.rgb[3 * k];
            const float g = image.rgb[3 * k + 1];
            const float b = image.rgb[3 * k + 2];
            for (const float v : {r, g, b})
            {
                bad += std::isfinite(v) && v >= 0 ? 0 : 1;
            }
            black += r == 0 && g == 0 && b == 0 ? 1 : 0;
        }
        ctx.count(prefix + "-values", bad, 0, static_cast<std::int64_t>(3 * texels));
        ctx.info(prefix + "-values", printed("%lld texels of zero in all three channels (0 expected)",
                                             static_cast<long long>(black)));
    }
}

// --- M10: what build_environment refuses -----------------------------------------------------------------------------

// 1 when build_environment refuses env as C.md §2.1 says: std::invalid_argument with a message
int refuses(const gpu::environment_desc& env)
{
    try
    {
        (void)gpu::build_environment(env);
    }
    catch (const std::invalid_argument& e)
    {
        return std::string_view(e.what()).empty() ? 0 : 1;
    }
    catch (...)
    {
        return 0;
    }
    return 0;
}

// 1 when it accepts env and gives tables of its dimensions
int accepts(const gpu::environment_desc& env)
{
    try
    {
        const gpu::environment_tables t = gpu::build_environment(env);
        return t.width == env.image.width && t.height == env.image.height && shaped(t) ? 1 : 0;
    }
    catch (...)
    {
        return 0;
    }
}

gpu::environment_desc filled(int width, int height, float value)
{
    gpu::environment_desc env;
    env.image = constant_image(width, height, value, value, value);
    return env;
}

GPU_TEST(env_map, cap)
{
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr float inf = std::numeric_limits<float>::infinity();
    // One at a time: the two maps past the cap are 25 MB of floats each, and complete: their size is their only fault
    ctx.count("cap-2049x1024", 1 - refuses(filled(2049, 1024, 0.5f)), 0, 1);
    ctx.count("cap-2048x1025", 1 - refuses(filled(2048, 1025, 0.5f)), 0, 1);
    gpu::environment_desc empty;
    empty.image = gpu::image_desc{.width = 0, .height = 1};
    ctx.count("cap-0x1", 1 - refuses(empty), 0, 1);
    empty.image = gpu::image_desc{.width = 1, .height = 0};
    ctx.count("cap-1x0", 1 - refuses(empty), 0, 1);
    // A small map with one bad value, at texel 2's green
    for (const auto& [name, value] : {std::pair{"cap-nan", nan}, std::pair{"cap-negative", -1.0f},
                                      std::pair{"cap-infinity", inf}})
    {
        gpu::environment_desc env = filled(4, 2, 0.5f);
        env.image.rgb[7] = value;
        ctx.count(name, 1 - refuses(env), 0, 1);
    }
    gpu::environment_desc short_rgb = filled(4, 2, 0.5f);
    short_rgb.image.rgb.pop_back();
    ctx.count("cap-rgb-size", 1 - refuses(short_rgb), 0, 1);
    for (const auto& [name, scale] : {std::pair{"cap-scale-negative", -1.0}, std::pair{"cap-scale-nan", double(nan)},
                                      std::pair{"cap-scale-infinity", double(inf)}})
    {
        gpu::environment_desc env = filled(4, 2, 0.5f);
        env.scale = scale;
        ctx.count(name, 1 - refuses(env), 0, 1);
    }
    for (const auto& [name, yaw] : {std::pair{"cap-yaw-nan", double(nan)}, std::pair{"cap-yaw-infinity", double(inf)}})
    {
        gpu::environment_desc env = filled(4, 2, 0.5f);
        env.yaw_degrees = yaw;
        ctx.count(name, 1 - refuses(env), 0, 1);
    }
    ctx.count("cap-accepts-2048x1024", 1 - accepts(filled(2048, 1024, 0.5f)), 0, 1);
    ctx.count("cap-accepts-u1", 1 - accepts(u1().desc), 0, 1);
}

// --- M1-M3: the tables on the host -----------------------------------------------------------------------------------

GPU_TEST(env_map, normalization)
{
    for (const map_case* m : {&studio(), &kloofendal(), &r300(), &k64(), &u1()})
    {
        const std::string name = "normalization-" + m->name;
        const gpu::environment_tables& t = m->tables;
        if (!shaped(t))
        {
            ctx.count(name + "-shape", 1, 0, 1);
            continue;
        }
        const std::vector<double> omega = row_solid_angles(t.width, t.height);
        exact_sum pdf_sum;
        exact_sum omega_sum;
        exact_sum float_sum; // Ω from the float row_cos the sampler will use
        for (int j = 0; j < t.height; ++j)
        {
            const std::size_t row = static_cast<std::size_t>(j);
            const double float_omega =
                2 * std::numbers::pi / t.width * (double{t.row_cos[row]} - double{t.row_cos[row + 1]});
            for (int i = 0; i < t.width; ++i)
            {
                const double pdf = t.texels[row * static_cast<std::size_t>(t.width) + static_cast<std::size_t>(i)].pdf;
                pdf_sum.add(pdf * omega[row]);
                omega_sum.add(omega[row]);
                float_sum.add(pdf * float_omega);
            }
        }
        ctx.exact(name, pdf_sum.value(), 1, 1e-6);
        ctx.exact(name + "-solid-angle", omega_sum.value(), 4 * std::numbers::pi, 1e-12 * 4 * std::numbers::pi);
        ctx.info(name, printed("with the float row_cos the sum is %.12f (|difference| %.3g)", float_sum.value(),
                               std::abs(float_sum.value() - 1)));
    }
}

GPU_TEST(env_map, row_cos)
{
    for (const map_case* m : all_maps())
    {
        const gpu::environment_tables& t = m->tables;
        const std::size_t rows = static_cast<std::size_t>(std::max(t.height, 0)) + 1;
        std::int64_t wrong = 0;
        if (t.height < 1 || t.row_cos.size() != rows)
        {
            wrong = static_cast<std::int64_t>(rows);
        }
        else
        {
            for (std::size_t j = 0; j < rows; ++j)
            {
                const double angle = std::numbers::pi * static_cast<double>(j) / t.height;
                wrong += bits(t.row_cos[j]) == bits(static_cast<float>(std::cos(angle))) ? 0 : 1;
            }
            wrong += t.row_cos.front() == 1.0f ? 0 : 1;
            wrong += t.row_cos.back() == -1.0f ? 0 : 1;
        }
        ctx.count("row-cos-" + m->name, wrong, 0, static_cast<std::int64_t>(rows));
    }
}

GPU_TEST(env_map, texels)
{
    for (const map_case* m : all_maps())
    {
        const std::string name = "texels-" + m->name;
        const gpu::environment_tables& t = m->tables;
        const gpu::image_desc& image = m->desc.image;
        const std::size_t texels = static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height);
        if (t.width != image.width || t.height != image.height || !shaped(t))
        {
            ctx.count(name + "-shape", 1, 0, 1);
            continue;
        }
        // The radiance unchanged, bit for bit (the scale is applied at lookup)
        std::int64_t rgb_wrong = 0;
        for (std::size_t k = 0; k < texels; ++k)
        {
            const gpu::env_texel& x = t.texels[k];
            rgb_wrong += bits(x.r) == bits(image.rgb[3 * k]) && bits(x.g) == bits(image.rgb[3 * k + 1]) &&
                                 bits(x.b) == bits(image.rgb[3 * k + 2])
                             ? 0
                             : 1;
        }
        ctx.count(name + "-rgb", rgb_wrong, 0, static_cast<std::int64_t>(texels));
        // S = Σ Y·Ω, recomputed
        const std::vector<double> omega = row_solid_angles(t.width, t.height);
        exact_sum total;
        for (std::size_t k = 0; k < texels; ++k)
        {
            const gpu::env_texel& x = t.texels[k];
            total.add(luminance(x.r, x.g, x.b) * omega[k / static_cast<std::size_t>(t.width)]);
        }
        const double s = total.value();
        if (s > 0)
        {
            ctx.exact(name + "-total", t.total / s, 1, 1e-12);
        }
        else
        {
            ctx.exact(name + "-total", t.total, 0, 0);
        }
        // pdf = float(Y/S), within one ulp (S's last bits depend on the order of the additions); 0 for a black map
        std::int64_t pdf_wrong = 0;
        for (std::size_t k = 0; k < texels; ++k)
        {
            const gpu::env_texel& x = t.texels[k];
            if (!(s > 0))
            {
                pdf_wrong += x.pdf == 0.0f ? 0 : 1;
                continue;
            }
            const float expected = static_cast<float>(luminance(x.r, x.g, x.b) / s);
            // Non-negative floats are ordered as their bit patterns: one apart is one ulp apart
            const bool valid = std::isfinite(x.pdf) && !std::signbit(x.pdf);
            const std::int64_t distance = std::int64_t{bits(x.pdf)} - std::int64_t{bits(expected)};
            pdf_wrong += valid && distance >= -1 && distance <= 1 ? 0 : 1;
        }
        ctx.count(name + "-pdf", pdf_wrong, 0, static_cast<std::int64_t>(texels));
    }
}

// --- M4: every cell's centre comes back to its cell ------------------------------------------------------------------

// Cell c's centre in (φ', cos θ), the float row_cos's midpoint, turned to the world by from_map, then env_cell
struct round_trip_mismatch
{
    environment_view e;
    std::int64_t cells;
    std::int64_t tested; // every cell, or 10^4 of them spread evenly (--quick)
    __device__ bool operator()(std::int64_t k) const
    {
        const auto c = static_cast<int>(k * cells / tested);
        const int i = c % e.width;
        const int j = c / e.width;
        const double phi = 2 * std::numbers::pi * (i + 0.5) / e.width - std::numbers::pi;
        const double cos_theta = 0.5 * (static_cast<double>(e.row_cos[j]) + static_cast<double>(e.row_cos[j + 1]));
        const double sin_theta = ::sqrt(::fmax(0.0, (1 - cos_theta) * (1 + cos_theta)));
        const vec3<float> m{static_cast<float>(sin_theta * ::cos(phi)), static_cast<float>(cos_theta),
                            static_cast<float>(sin_theta * ::sin(phi))};
        return gpu::env_cell(e, gpu::from_map(e, m)) != c;
    }
};

GPU_TEST(env_map, round_trip)
{
    for (const map_case* m : {&studio(), &kloofendal(), &r300(), &k64()})
    {
        const gpu::environment_tables& t = m->tables;
        if (!shaped(t))
        {
            ctx.count("round-trip-" + m->name + "-shape", 1, 0, 1);
            continue;
        }
        const device_environment device(t);
        const std::int64_t cells = static_cast<std::int64_t>(t.width) * t.height;
        const std::int64_t tested = ctx.samples(cells);
        for (const int yaw : {0, 37})
        {
            ctx.count("round-trip-" + m->name + "-" + std::to_string(yaw),
                      count_indices(round_trip_mismatch{device.view(yaw), cells, tested}, tested), 0, tested);
        }
    }
}

// --- M5, M13: the pinned orientation, and the mirror it must reject --------------------------------------------------

struct pinned_case
{
    vec3<float> direction; // in the world, or in map space when through_from_map
    int yaw90;             // 0: the view at yaw 0; 1: at yaw 90°
    int through_from_map;  // 1: from_map turns the direction to the world first
    int expected;          // the cell j·W + i, or the row j when row_only
    int row_only;
};

constexpr int max_pinned_cases = 16;
struct pinned_set
{
    pinned_case cases[max_pinned_cases];
    int count;
};

vec3<float> host_map_direction(double phi, double theta)
{
    const double s = std::sin(theta);
    return {static_cast<float>(s * std::cos(phi)), static_cast<float>(std::cos(theta)),
            static_cast<float>(s * std::sin(phi))};
}

// C.md M5's directions on a W×H map: half a cell past +x, +z and -z (and half a cell before them, half a row above the
// equator), straight up and down, and at yaw 90° the cell past -z in map space, through from_map
pinned_set pinned_directions(int width, int height)
{
    const double pi = std::numbers::pi;
    pinned_set set{};
    const auto add = [&set](const pinned_case& c) { set.cases[set.count++] = c; };
    const double azimuths[] = {0, pi / 2, -pi / 2};
    const int columns[] = {width / 2, 3 * width / 4, width / 4};
    for (int k = 0; k < 3; ++k)
    {
        add({host_map_direction(azimuths[k] + pi / width, pi / 2 + pi / (2 * height)), 0, 0,
             (height / 2) * width + columns[k], 0});
        add({host_map_direction(azimuths[k] - pi / width, pi / 2 - pi / (2 * height)), 0, 0,
             (height / 2 - 1) * width + columns[k] - 1, 0});
    }
    add({{0, 1, 0}, 0, 0, 0, 1});
    add({{0, -1, 0}, 0, 0, height - 1, 1});
    add({host_map_direction(-pi / 2 + pi / width, pi / 2 + pi / (2 * height)), 1, 1, (height / 2) * width + width / 4,
         0});
    return set;
}

template <class Mapping>
struct pinned_run
{
    Mapping mapping;
    environment_view yaw0;
    environment_view yaw90;
    pinned_set set;
    int* cells;
    float* turned; // to_map((1, 0, 0)) at yaw 90°
    __device__ void operator()() const
    {
        for (int k = 0; k < set.count; ++k)
        {
            const pinned_case& c = set.cases[k];
            const environment_view& e = c.yaw90 ? yaw90 : yaw0;
            const vec3<float> d = c.through_from_map ? mapping.from_map(e, c.direction) : c.direction;
            cells[k] = mapping.cell(e, d);
        }
        const vec3<float> m = mapping.to_map(yaw90, vec3<float>{1.0f, 0.0f, 0.0f});
        turned[0] = m.x;
        turned[1] = m.y;
        turned[2] = m.z;
    }
};

// The checker: how many of the pinned cases a mapping gets wrong (the turned +x counts as one more case)
template <class Mapping>
std::int64_t pinned_mismatches(const Mapping& mapping, const environment_view& yaw0, const environment_view& yaw90,
                               const pinned_set& set)
{
    gpu_test::device_buffer<int> cells(static_cast<std::size_t>(set.count));
    gpu_test::device_buffer<float> turned(3);
    run_once(pinned_run<Mapping>{mapping, yaw0, yaw90, set, cells.get(), turned.get()}, "the pinned directions");
    const std::vector<int> found = cells.download();
    const std::vector<float> m = turned.download();
    const int size = yaw0.width * yaw0.height;
    std::int64_t mismatches = 0;
    for (int k = 0; k < set.count; ++k)
    {
        const pinned_case& c = set.cases[k];
        const int cell = found[static_cast<std::size_t>(k)];
        const int got = cell >= 0 && cell < size ? (c.row_only ? cell / yaw0.width : cell) : -1;
        mismatches += got == c.expected ? 0 : 1;
    }
    const double dx = m[0];
    const double dy = m[1];
    const double dz = m[2] + 1.0;
    mismatches += std::sqrt(dx * dx + dy * dy + dz * dz) <= 1e-6 ? 0 : 1;
    return mismatches;
}

GPU_TEST(env_map, pinned)
{
    const map_case& m = pinned_map();
    const device_environment device(m.tables);
    const pinned_set set = pinned_directions(m.tables.width, m.tables.height);
    ctx.count("pinned", pinned_mismatches(implementation_mapping{}, device.view(0), device.view(90), set), 0,
              set.count + 1);
}

// --- M6: the bilinear filter -----------------------------------------------------------------------------------------

// (a) At texel (i, j)'s centre (u, t) = ((i + ½)/W, (j + ½)/H), yaw 0: the nearest value is the texel's own, and the
// filtered one equals it within decision 1's tolerance (strict: the spec's term alone, for the INFO count)
struct centre_violation
{
    environment_view e;
    float largest;
    bool strict;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const auto i = static_cast<int>(rng.next() % static_cast<unsigned>(e.width));
        const auto j = static_cast<int>(rng.next() % static_cast<unsigned>(e.height));
        const double u = (i + 0.5) / e.width;
        const double t = (j + 0.5) / e.height;
        const vec3<float> d = map_direction(2 * std::numbers::pi * (u - 0.5), std::numbers::pi * t);
        float pdf = 0;
        const vec3<float> nearest = gpu::env_radiance(e, d, pdf);
        const vec3<float> filtered = gpu::env_radiance_filtered(e, d);
        bool bad = false;
        for (int c = 0; c < 3; ++c)
        {
            const double own = texel_channel(e, i, j, c);
            const double n = channel(nearest, c);
            const double f = channel(filtered, c);
            if (!isfinite(n) || !isfinite(f) || n != own)
            {
                bad = true;
                continue;
            }
            double spread = 0; // the largest difference to one of the eight neighbours
            for (int dj = -1; dj <= 1; ++dj)
            {
                for (int di = -1; di <= 1; ++di)
                {
                    const int column = wrap_column(i + di, e.width);
                    const double other = texel_channel(e, column, clamp_row(j + dj, e.height), c);
                    spread = ::fmax(spread, ::fabs(other - own));
                }
            }
            const double base = 1e-5 * ::fmax(n, 1e-6 * largest);
            const double allowed = strict ? base : base + 2e-3 * spread;
            bad = bad || !(::fabs(f - n) <= allowed);
        }
        return bad;
    }
};

// (b) A constant map: both lookups exactly the constant, 1, in every direction
struct constant_violation
{
    environment_view e;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const vec3<float> d = random_direction(rng);
        float pdf = 0;
        const vec3<float> n = gpu::env_radiance(e, d, pdf);
        const vec3<float> f = gpu::env_radiance_filtered(e, d);
        return !(n.x == 1.0f && n.y == 1.0f && n.z == 1.0f && f.x == 1.0f && f.y == 1.0f && f.z == 1.0f);
    }
};

// (c) The seam: at φ' = π - δ and -π + δ, δ = 10^-3 of a cell, the same θ: the values differ by the lerp's 2δ step only
struct seam_violation
{
    environment_view e;
    float largest;
    double factor; // 3·10^-3 for the verdict, 2·10^-3 (the spec's) for the INFO count
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const double theta = std::numbers::pi * gpu::random_real<float>(rng);
        const double delta = 1e-3 * 2 * std::numbers::pi / e.width;
        const vec3<float> a = gpu::env_radiance_filtered(e, map_direction(std::numbers::pi - delta, theta));
        const vec3<float> b = gpu::env_radiance_filtered(e, map_direction(-std::numbers::pi + delta, theta));
        // The two rows the θ interpolation blends
        const int j0 = static_cast<int>(::floor(theta / std::numbers::pi * e.height - 0.5));
        const int rows[] = {clamp_row(j0, e.height), clamp_row(j0 + 1, e.height)};
        bool bad = false;
        for (int c = 0; c < 3; ++c)
        {
            double jump = 0;
            for (const int j : rows)
            {
                jump = ::fmax(jump, ::fabs(texel_channel(e, e.width - 1, j, c) - texel_channel(e, 0, j, c)));
            }
            const double x = channel(a, c);
            const double y = channel(b, c);
            bad = bad || !isfinite(x) || !isfinite(y) || !(::fabs(x - y) <= factor * jump + 1e-6 * largest);
        }
        return bad;
    }
};

// (d) Above the first row's centre and below the last's, the rows clamp: the value is that row's own u-lerp
struct pole_row_violation
{
    environment_view e;
    float largest;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const double phi = 2 * std::numbers::pi * gpu::random_real<float>(rng) - std::numbers::pi;
        const double s = 0.9 * (gpu::random_real<float>(rng) + 0x1p-25); // (0, 0.9) of half a row from the pole
        const bool top = gpu::random_real<float>(rng) < 0.5f;
        const double half_row = std::numbers::pi / (2 * e.height);
        const vec3<float> d = map_direction(phi, top ? s * half_row : std::numbers::pi - s * half_row);
        const vec3<float> f = gpu::env_radiance_filtered(e, d);
        const int row = top ? 0 : e.height - 1;
        // The row's u-lerp from the same float direction, in double (C.md §2.4's x = u·W - 0.5, wrapped)
        const double u = 0.5 + ::atan2(static_cast<double>(d.z), static_cast<double>(d.x)) / (2 * std::numbers::pi);
        const double x = u * e.width - 0.5;
        const double x0 = ::floor(x);
        const double fx = x - x0;
        const int i0 = wrap_column(static_cast<int>(x0), e.width);
        const int i1 = wrap_column(i0 + 1, e.width);
        bool bad = false;
        for (int c = 0; c < 3; ++c)
        {
            const double a = texel_channel(e, i0, row, c);
            const double b = texel_channel(e, i1, row, c);
            const double expected = a + fx * (b - a);
            const double spread =
                ::fmax(::fmax(::fabs(a - texel_channel(e, wrap_column(i0 - 1, e.width), row, c)), ::fabs(b - a)),
                       ::fabs(texel_channel(e, wrap_column(i1 + 1, e.width), row, c) - b));
            const double value = channel(f, c);
            const double allowed = 1e-5 * ::fmax(::fabs(expected), 1e-6 * largest) + 2e-3 * spread;
            bad = bad || !isfinite(value) || !(::fabs(value - expected) <= allowed);
        }
        return bad;
    }
};

// (e) The scale multiplies both lookups and leaves the density alone
struct scale_violation
{
    environment_view one;    // scale 1
    environment_view scaled; // the same tables, scale 2.5
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const vec3<float> d = random_direction(rng);
        float pdf_one = 0;
        float pdf_scaled = 0;
        const vec3<float> n1 = gpu::env_radiance(one, d, pdf_one);
        const vec3<float> n2 = gpu::env_radiance(scaled, d, pdf_scaled);
        const vec3<float> f1 = gpu::env_radiance_filtered(one, d);
        const vec3<float> f2 = gpu::env_radiance_filtered(scaled, d);
        bool bad = !bits_equal(pdf_one, pdf_scaled);
        for (int c = 0; c < 3; ++c)
        {
            const double nearest = 2.5 * channel(n1, c);
            const double filtered = 2.5 * channel(f1, c);
            bad = bad || !(::fabs(channel(n2, c) - nearest) <= 1e-6 * nearest) ||
                  !(::fabs(channel(f2, c) - filtered) <= 1e-6 * filtered);
        }
        return bad;
    }
};

GPU_TEST(env_map, bilinear)
{
    const std::int64_t n = ctx.samples(small_samples);
    const std::int64_t drawn = gpu_test::drawn(n);
    const map_case& r = r300();
    if (!shaped(r.tables) || !shaped(studio().tables) || !shaped(k64().tables))
    {
        ctx.count("bilinear-shape", 1, 0, 1);
        return;
    }
    const float r_largest = largest_value(r.tables);
    {
        const device_environment device(r.tables);
        const environment_view e = device.view();
        ctx.count("bilinear-centres-r300", gpu_test::device_count(centre_violation{e, r_largest, false}, 8201, n), 0,
                  drawn);
        ctx.info("bilinear-centres-r300",
                 printed("%lld of %lld beyond the spec's 1e-5 term alone",
                         static_cast<long long>(gpu_test::device_count(centre_violation{e, r_largest, true}, 8201, n)),
                         static_cast<long long>(drawn)));
        ctx.count("bilinear-seam", gpu_test::device_count(seam_violation{e, r_largest, 3e-3}, 8204, n), 0, drawn);
        ctx.info("bilinear-seam",
                 printed("%lld of %lld beyond the spec's 2e-3 factor",
                         static_cast<long long>(gpu_test::device_count(seam_violation{e, r_largest, 2e-3}, 8204, n)),
                         static_cast<long long>(drawn)));
        ctx.count("bilinear-pole-rows", gpu_test::device_count(pole_row_violation{e, r_largest}, 8205, n), 0, drawn);
        ctx.count("bilinear-scale", gpu_test::device_count(scale_violation{e, device.view(0, 2.5)}, 8206, n), 0, drawn);
    }
    {
        const map_case& s = studio();
        const float s_largest = largest_value(s.tables);
        const device_environment device(s.tables);
        const environment_view e = device.view();
        ctx.count("bilinear-centres-studio", gpu_test::device_count(centre_violation{e, s_largest, false}, 8202, n), 0,
                  drawn);
        ctx.info("bilinear-centres-studio",
                 printed("%lld of %lld beyond the spec's 1e-5 term alone",
                         static_cast<long long>(gpu_test::device_count(centre_violation{e, s_largest, true}, 8202, n)),
                         static_cast<long long>(drawn)));
    }
    {
        const device_environment device(k64().tables);
        const std::int64_t m = ctx.samples(constant_samples);
        ctx.count("bilinear-constant", gpu_test::device_count(constant_violation{device.view(37)}, 8203, m), 0,
                  gpu_test::drawn(m));
    }
}

// --- M7: background() ------------------------------------------------------------------------------------------------

gpu::camera<float> sky_camera()
{
    gpu::camera<float> cam{};
    cam.sky_bottom = {1.0f, 1.0f, 1.0f};
    cam.sky_top = {0.5f, 0.7f, 1.0f};
    return cam;
}

gpu::quality_view view_with(const environment_view& e)
{
    gpu::quality_view v{};
    v.environment = e;
    v.light_pmf = 0;
    v.combination = gpu::light_combination::bsdf_only;
    return v;
}

// With a map: filtered rays see env_radiance_filtered, the others env_radiance (bitwise, one load); without one, the
// camera's sky, bitwise. The filtered value within 4 ulps plus 10⁻⁴ of the spread of the four texels it blends: two
// inlined copies of the lookup may contract the direction's rotation apart, which moves the blend's fraction by a few
// 10⁻⁵ of a texel, and on r300 neighbours differ up to 10⁶ times (as M6(a))
// The largest minus the smallest channel value of the 2 × 2 texels the bilinear lookup blends for d, times the scale
__device__ inline float corner_spread(const environment_view& e, const vec3<float>& d)
{
    float u;
    float t;
    gpu::map_coordinates(gpu::to_map(e, d), u, t);
    const int x0 = static_cast<int>(floorf(u * static_cast<float>(e.width) - 0.5f));
    const int y0 = static_cast<int>(floorf(t * static_cast<float>(e.height) - 0.5f));
    float lo = INFINITY;
    float hi = -INFINITY;
    for (int dy = 0; dy < 2; ++dy)
    {
        for (int dx = 0; dx < 2; ++dx)
        {
            const int i = ((x0 + dx) % e.width + e.width) % e.width;
            const int j = gpu::clamp_index(y0 + dy, e.height);
            const auto& texel = e.texels[j * e.width + i];
            lo = fminf(lo, fminf(texel.r, fminf(texel.g, texel.b)));
            hi = fmaxf(hi, fmaxf(texel.r, fmaxf(texel.g, texel.b)));
        }
    }
    return e.scale * (hi - lo);
}

struct background_violation
{
    gpu::quality_view v;
    gpu::camera<float> cam;
    int which; // 0: both, 1: the nearest texel only, 2: the filtered lookup only
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const vec3<float> d = random_direction(rng);
        const vec3<float> filtered = gpu::background(v, cam, d, true);
        const vec3<float> nearest = gpu::background(v, cam, d, false);
        if (v.environment.texels == nullptr)
        {
            const vec3<float> sky = cam.sky(d);
            return !(bits_equal(filtered, sky) && bits_equal(nearest, sky));
        }
        float pdf = 0;
        const vec3<float> expected = gpu::env_radiance_filtered(v.environment, d);
        const float spread = corner_spread(v.environment, d);
        const auto close = [spread](float a, float b) {
            return fabsf(a - b) <= 4.8e-7f * fmaxf(fabsf(a), fabsf(b)) + 1e-4f * spread;
        };
        const bool filtered_ok = close(filtered.x, expected.x) && close(filtered.y, expected.y) &&
                                 close(filtered.z, expected.z);
        const bool nearest_ok = bits_equal(nearest, gpu::env_radiance(v.environment, d, pdf));
        return which == 1 ? !nearest_ok : which == 2 ? !filtered_ok : !(filtered_ok && nearest_ok);
    }
};

GPU_TEST(env_map, background)
{
    const std::int64_t n = ctx.samples(small_samples);
    const gpu::camera<float> cam = sky_camera();
    const map_case& r = r300();
    if (!shaped(r.tables))
    {
        ctx.count("background-shape", 1, 0, 1);
        return;
    }
    const device_environment device(r.tables);
    ctx.count("background-map-nearest",
              gpu_test::device_count(background_violation{view_with(device.view(37, 2.5)), cam, 1}, 8207, n), 0,
              gpu_test::drawn(n));
    ctx.count("background-map-filtered",
              gpu_test::device_count(background_violation{view_with(device.view(37, 2.5)), cam, 2}, 8207, n), 0,
              gpu_test::drawn(n));
    ctx.count("background-no-map", gpu_test::device_count(background_violation{view_with({}), cam, 0}, 8208, n), 0,
              gpu_test::drawn(n));
}

// --- M8: the sun of kloofendal_48d -----------------------------------------------------------------------------------

GPU_TEST(env_map, sun)
{
    const gpu::environment_tables& t = kloofendal().tables;
    if (!shaped(t))
    {
        ctx.count("sun-shape", 1, 0, 1);
        return;
    }
    std::size_t brightest = 0;
    double largest = -1;
    for (std::size_t k = 0; k < t.texels.size(); ++k)
    {
        const double y = luminance(t.texels[k].r, t.texels[k].g, t.texels[k].b);
        if (y > largest)
        {
            largest = y;
            brightest = k;
        }
    }
    const auto j = static_cast<int>(brightest / static_cast<std::size_t>(t.width));
    const auto i = static_cast<int>(brightest % static_cast<std::size_t>(t.width));
    const double elevation = 90 - (j + 0.5) * 180.0 / t.height;
    ctx.exact("sun", elevation, 48, 0.5);
    ctx.info("sun", printed("row %d, column %d (luminance %.6g): elevation %.3f degrees, azimuth phi' %.3f degrees in "
                            "map space",
                            j, i, largest, elevation, (i + 0.5) * 360.0 / t.width - 180));
}

// --- M11: a black map ------------------------------------------------------------------------------------------------

// what: 0 environment_sampled, 1 environment_pdf, 2 sample_environment (false, the generator untouched), 3 background
struct black_violation
{
    gpu::quality_view v;
    gpu::camera<float> cam;
    int what;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const vec3<float> d = random_direction(rng);
        const environment_view& e = v.environment;
        switch (what)
        {
        case 0:
            return gpu::environment_sampled(e);
        case 1:
            return gpu::environment_pdf(e, d) != 0.0f;
        case 2:
        {
            const gpu::pcg32 before = rng;
            vec3<float> wi{};
            vec3<float> le{};
            float pdf = 0;
            const bool drawn = gpu::sample_environment(e, rng, wi, le, pdf);
            gpu::pcg32 a = before;
            gpu::pcg32 b = rng;
            const bool untouched = a.next() == b.next() && a.next() == b.next();
            return drawn || !untouched;
        }
        default:
        {
            const vec3<float> f = gpu::background(v, cam, d, true);
            const vec3<float> n = gpu::background(v, cam, d, false);
            return !(f.x == 0.0f && f.y == 0.0f && f.z == 0.0f && n.x == 0.0f && n.y == 0.0f && n.z == 0.0f);
        }
        }
    }
};

GPU_TEST(env_map, black)
{
    const gpu::environment_tables& t = b64().tables;
    if (!shaped(t))
    {
        ctx.count("black-shape", 1, 0, 1);
        return;
    }
    std::int64_t wrong = (t.total == 0 ? 0 : 1) + (t.bins.empty() ? 0 : 1) + (t.log2_bins == 0 ? 0 : 1);
    for (const gpu::env_texel& x : t.texels)
    {
        wrong += x.pdf == 0.0f ? 0 : 1;
    }
    ctx.count("black-tables", wrong, 0, static_cast<std::int64_t>(t.texels.size()) + 3);
    const std::int64_t n = ctx.samples(small_samples);
    const device_environment device(t);
    const gpu::quality_view v = view_with(device.view());
    const gpu::camera<float> cam = sky_camera(); // not black: a background that falls back to the sky shows
    const char* names[] = {"black-sampled", "black-pdf", "black-draw", "black-background"};
    for (int what = 0; what < 4; ++what)
    {
        ctx.count(names[what], gpu_test::device_count(black_violation{v, cam, what}, 8209, n), 0, gpu_test::drawn(n));
    }
}

// --- M12: which maps are sampled -------------------------------------------------------------------------------------

struct sampled_flag
{
    environment_view e;
    int* out;
    __device__ void operator()() const { *out = gpu::environment_sampled(e) ? 1 : 0; }
};

GPU_TEST(env_map, sampled_flag)
{
    for (const map_case* m : all_maps())
    {
        const gpu::environment_tables& t = m->tables;
        if (!shaped(t))
        {
            ctx.count("sampled-flag-" + m->name, 1, 0, 1);
            continue;
        }
        const device_environment device(t);
        gpu_test::device_buffer<int> flag(1);
        run_once(sampled_flag{device.view(), flag.get()}, "environment_sampled");
        const bool sampled = flag.download()[0] != 0;
        const bool has_bins = !t.bins.empty();
        const bool bins_sized = has_bins ? t.log2_bins >= 1 && t.log2_bins < 31 &&
                                               t.bins.size() == (std::size_t{1} << static_cast<unsigned>(t.log2_bins))
                                         : t.log2_bins == 0;
        const int wrong = (sampled == has_bins ? 0 : 1) + (t.total == 0 && has_bins ? 1 : 0) + (bins_sized ? 0 : 1);
        ctx.count("sampled-flag-" + m->name, wrong, 0, 3);
    }
}

// --- M13: the must-fail ----------------------------------------------------------------------------------------------

GPU_TEST(env_map, planted_mirror)
{
    const map_case& m = pinned_map();
    const device_environment device(m.tables);
    const pinned_set set = pinned_directions(m.tables.width, m.tables.height);
    const environment_view yaw0 = device.view(0);
    const environment_view yaw90 = device.view(90);
    const std::int64_t mirrored = pinned_mismatches(reference_mapping{true}, yaw0, yaw90, set);
    ctx.count("planted-mirror", mirrored == 0 ? 1 : 0, 0, 1);
    ctx.info("planted-mirror",
             printed("the mirrored mapping misses %lld of %d pinned cases; the unmirrored test-side one %lld (0 "
                     "expected)",
                     static_cast<long long>(mirrored), set.count + 1,
                     static_cast<long long>(pinned_mismatches(reference_mapping{false}, yaw0, yaw90, set))));
}

// =====================================================================================================================
// Suite env-sampling (Step 83): the map as a light, sampled by its brightness
// =====================================================================================================================
//
// sample_environment picks a cell c with the alias table (Vose 1991) over the cells' weights w_c = Y_c·Ω_c, then a
// direction uniform in the cell's rectangle of (φ', cos θ), where dω = dφ'·d(cos θ): the density per steradian is
// p = P_c/Ω_c = Y_c/S, constant over the cell, and every sample's L/p has the luminance S (C.md §2.3). The checks
// (C.md §5.2, N1-N13, and the brief's additions):
//   - the table on the host: its size, Vose's invariants, and the probabilities it reconstructs against w/S (N1);
//   - 10^7 samples per map on the GPU, each checked on its own: the density returned is environment_pdf's of the
//     direction (N2) and the radiance env_radiance's; the direction lies in the cell the first two numbers choose
//     from the table; Y(L)/p = S (N3); 4 numbers drawn (N7); every number finite (N8). The same samples binned on a
//     grid that cuts across the cells and on one made of whole cells for Pearson's χ² test (N4), and against the
//     mirrored map's counts, which must be rejected (N13);
//   - Monte Carlo means against the map's own sums: E[1/p] = the solid angle of the cells with light, and the
//     irradiance E[L̄·cos⁺/p] of a plane facing up (per cell in closed form), of a tilted plane under a constant map
//     (π) and under two hemispheres (π(1 ± cos β)/2); a density that misses each row's own solid angle is rejected by
//     the same mean;
//   - a constant map's density, 1/(4π) exactly (N5); one bright texel that takes every sample (N6);
//   - the map as a light of the set: environment_sampled for maps with and without light and without a table; Veach's
//     W1 with the map alone (N9); with a quad lamp beside it (light_pmf ½) the light sample's fields and the source
//     frequencies (N10), W1 across both sources (N11), and the irradiance of a point lit by both against Lambert's
//     formula (N12); the map's density without the pmf on the material's side is rejected by the W1 check.
// Maps: env-map's studio, kloofendal, r300, k64, u1, b64, and s64 (64×32 of 0, texel (17, 9) = (100, 100, 100)), h48
// (48×24, channels 2^U with U uniform in [-2, 2], the 4 rows nearest -y black and 10% of the others, SplitMix64
// seeded 8300) and two (1×2: the upper hemisphere (1, 0.8, 0.6), the lower (0.2, 0.1, 0.05)).
//
// Decisions where the spec leaves room (for Claude to confirm):
// 1. h48 and two are the test's own: E[1/p] needs a map whose density has a bounded range (the assets have cells
//    sampled far less than once in 10^7: studio 197,368 cells below 2^-24, C.md §4, which hold most of E[1/p]'s
//    solid angle; its mean would be heavy-tailed), with black cells to make "the cells with light" mean something and
//    896 padded bins (1152 cells, 2048 bins); two gives the irradiance of a tilted plane in closed form.
// 2. N2 as the spec says (env_cell(wi) against the cell of wi recomputed in double, at most 10^-4 of the samples), and
//    one more check of the same samples, cell-<map>: the cell of wi in double against the cell C.md §2.3's numbers (1)
//    and (2) choose from the same generator (b = next() >> (32 - log2_bins), then ξ < q_b ? b : alias_b; the order of
//    (3) and (4) isn't used). Elsewhere than in that cell or a neighbour (|Δj| ≤ 1 and |Δi| ≤ 1 with the columns
//    wrapped, or anywhere in the same pole row: sin θ = 0 has no azimuth) is a violation; a neighbour is a border case,
//    at most 10^-3 of the samples (cell-<map>-border). Not 10^-4: the sampler's rows end at the float row_cos and its
//    vector's length is 1 to a few 10^-7, which moves cos θ near a pole by about as much; up to ~10^-3 of a polar
//    row's samples cross the row's border in double (C.md §8.5's inherent mismatch), so a map bright near a pole could
//    pass 10^-4. A wrong cell (rows or columns swapped, a padded bin returned as itself, another bin formula) misses on
//    most samples.
// 3. "Every bin index reachable" (bins-<map>): every bin b of the table was the first number's bin of some sample
//    whose direction landed in b's choice, on r300 (65,536 bins, 20,536 of them padded), k64, u1 (2 bins, one padded),
//    h48 and two. "Padded bins never returned": a padded bin returned as itself would be no cell of the map, so
//    cell-<map> counts it.
// 4. The irradiance mean runs on every map (the luminance-proportional density keeps L̄/p within S·max(L̄/Y), at most
//    4.6·S: no heavy tail); E[1/p] on h48 and two only (decision 1); N5's K64 and U1 have p constant, so their exact
//    check stands for E[1/p]. The planted fault: p' = P_c·N/(4π), the cell's probability spread as if every cell
//    covered 4π/N steradians (an equirectangular map's missing solid angle per row, the sin θ), through the E[1/p]
//    mean of h48: E[1/p'] = 4π·N_lit/N, 11% below Ω_lit there; rejected as |z| > 8.
// 5. N3 is reported as the largest |Y(L)/p - scale·S|/(scale·S) over the samples (an exact check, ≤ 10^-5); S is the
//    test's own sum in double.
// 6. N5 adds the host's texels and the device's samples that differ from uniform_sphere_density<float> in one count.
// 7. N6: s64 at yaw 37° and scale 1.5 (the radiance 150 exactly): bright-texel counts the samples outside (17, 9) and
//    its neighbours plus the true returns of another radiance (0 allowed); bright-texel-border the samples in a
//    neighbour, bright-texel-false the false returns (each at most 10^-4).
// 8. N7 is two counts: draws (every sample_environment of every run: 4 numbers) and draws-light (sample_light with
//    N10's view: 3 for the lamp, 5 for the map, judged by the first number's pick k = min(⌊2ξ⌋, 1), lamps first);
//    lamp-and-map-pick adds that the light picked is the one that number names.
// 9. N9 and N11 run under -p nee too (t3-env-nee, lamp-and-map-t3-nee: Σw == 1 exactly). Directions with f = 0 (below
//    the surface) or a black texel are skipped (W1 applies where f·Le ≠ 0), and so are N11's map directions the lamp
//    hides (counted) and lamp directions whose traced hit isn't the lamp's front (T1's business). N11's points: x
//    uniform in [-1, 2] × [-1, 2] × [-1.5, 1.5] (a sixth of them behind the lamp's plane z = 1), normals as D's T3.
//    The planted fault (planted-env-pmf): N11's light-drawn directions with the map's density on the material's side
//    without light_pmf (environment_pdf instead of environment_hit_pdf); rejected if some |Σw - 1| > 10^-2.
// 10. N10's source frequencies: the share of the true returns that came from the map, against ½ (mean, binomial SE).
//    Lamp samples are checked as D's T1 (edges at most 10^-4), map samples for distance +∞, skip -1, and the pdf ==
//    environment_hit_pdf bitwise.
// 11. N12 follows D's T4 with the map added on both sides: an escape of the material's ray sees the nearest texel with
//    the weight w_B(c, p_B, environment_hit_pdf). Its expected value from Ω⊥ = atan(1/√2)/√2 (D.md §5.1) is checked
//    against the spec's 0.9848605 (lamp-and-map-reference, 10^-6).
// 12. Yaw and scale: studio is sampled at yaw 0 and at yaw 37° with scale 2.5 (the χ² of the latter in world space,
//    its expected counts turned by the view's own float angle atan2(sin_yaw, cos_yaw)); h48 at 37° and scale 2; s64 at
//    37° and 1.5; the rest at 0 and 1.
// 13. --quick (10^4 samples): the border counts (N2, cell, N6) and bins-<map>, cells-reached-k64 need the full count:
//    printed as INFO lines; every other count and exact check keeps its verdict; means, χ² and must-rejects SKIP.
// 14. Possible spec issue, N1 on kloofendal: P' is rebuilt from the float q, so every bin that aliases a cell adds its
//    own rounding of q (up to 3·10^-8). The sun's four texels hold 41.7% of the energy: about 10^5 bins alias each, and
//    if the table was built with the unrounded q in Vose's update (p̂_g ← p̂_g + p̂_l - 1), the roundings add up to
//    ~5·10^-6 > 10^-6. Updating with the rounded q (p̂_g ← p̂_g + double(float(p̂_l)) - 1) keeps it near 10^-7. The
//    INFO line names the worst bin and how many bins alias it.
//
// The step's criteria (C.md §7, Step 83; Claude runs them; $S = -w 480 --height 270):
//   1. $GT env-sampling: exit 0 (the lamps-plus-map cases N10-N12 included).
//   2. $G env-furnace -p mis $S -s 256 --seed 3 -o efm.pfm: $PS stats --rect 238 133 242 137 channel-average mean
//      1 ± 4 SE, and the interior rectangle [200..280] × [100..170] the same; $IC of env-furnace -p material (seed 1;
//      itself ≥ 99.99% exactly 1.0, none above 1 + 10^-6) against -p mis (seed 3): exit 0.
//   3. compare.ps1 -Scene sun-test -A nee:1,2 -B mis:3,4 -Spp 1024: pass (never BSDF-only on the sun map).
//   4. compare.ps1 -Scene studio -A material:1,2 -B mis:3,4 -Spp 1024: pass; σ_mis < σ_material ($PS noise, trimmed).
//   5. sunlit -p mis $S -s 256, seeds 3 and 4: finite, the control exits 0.
//   6. Sanitizer: studio, sunlit, $GT env-sampling --quick. CARRY (all); recorded: studio, sunlit -p mis $S -s 64
//      --seed 3.

constexpr std::int64_t full_samples = 10'000'000;
constexpr std::int64_t million = 1'000'000;
constexpr std::int64_t batch_samples = 1'000'000; // per launch: a few milliseconds
constexpr double pi = std::numbers::pi;

const char* mode_name(light_combination c)
{
    return c == light_combination::bsdf_only ? "material" : c == light_combination::lights_only ? "nee" : "mis";
}

// --- The maps of this suite ------------------------------------------------------------------------------------------

gpu::image_desc s64_image()
{
    gpu::image_desc image = constant_image(64, 32, 0, 0, 0);
    const std::size_t k = 3 * (9 * std::size_t{64} + 17);
    image.rgb[k] = 100.0f;
    image.rgb[k + 1] = 100.0f;
    image.rgb[k + 2] = 100.0f;
    return image;
}

// Per texel one number for black (10%), then three channels 2^U, U uniform in [-2, 2]; the bottom 4 rows black
gpu::image_desc h48_image()
{
    constexpr int width = 48;
    constexpr int height = 24;
    gpu::image_desc image{.width = width, .height = height};
    image.rgb.resize(3 * static_cast<std::size_t>(width) * height);
    host_generator g{8300};
    for (std::size_t k = 0; k < static_cast<std::size_t>(width) * height; ++k)
    {
        const bool black = g.uniform() < 0.1 || k >= 20 * static_cast<std::size_t>(width);
        for (std::size_t c = 0; c < 3; ++c)
        {
            const double value = std::exp2(4 * g.uniform() - 2);
            image.rgb[3 * k + c] = black ? 0.0f : static_cast<float>(value);
        }
    }
    return image;
}

gpu::image_desc two_image()
{
    gpu::image_desc image{.width = 1, .height = 2};
    image.rgb = {1.0f, 0.8f, 0.6f, 0.2f, 0.1f, 0.05f};
    return image;
}

const map_case& s64()
{
    static const map_case m = make_case("s64", s64_image());
    return m;
}
const map_case& h48()
{
    static const map_case m = make_case("h48", h48_image());
    return m;
}
const map_case& two_cells()
{
    static const map_case m = make_case("two", two_image());
    return m;
}

// A constant 64 × 32 map of 1 with one texel of 10^-30: a cell whose share of a bin, about 10^-34, is far below one of
// the coin's steps (2^-24), which only the forced minimum coin keeps drawable
const map_case& dim_cell()
{
    static const map_case m = [] {
        gpu::image_desc image = constant_image(64, 32, 1, 1, 1);
        const std::size_t texel = 3 * (16 * 64 + 20);
        image.rgb[texel] = image.rgb[texel + 1] = image.rgb[texel + 2] = 1e-30f;
        return make_case("dim", std::move(image));
    }();
    return m;
}

// A map with light and its alias table, sized as C.md §2.1 says it is (the device kernels read it by those sizes)
bool has_table(const gpu::environment_tables& t)
{
    return shaped(t) && t.total > 0 && t.log2_bins >= 1 && t.log2_bins <= 21 &&
           t.bins.size() == (std::size_t{1} << static_cast<unsigned>(t.log2_bins));
}

// --- The map's cells in double ---------------------------------------------------------------------------------------

// P_c = w_c/S, w_c = Y_c·Ω_j, Ω_j from double cosines: what the table must reproduce and the samples follow
struct cell_weights
{
    std::vector<double> cosine;      // H + 1: cos(πj/H)
    std::vector<double> omega;       // H: Ω_j
    std::vector<double> probability; // W·H
    double total = 0;                // S
};

cell_weights weights_of(const gpu::environment_tables& t)
{
    cell_weights w;
    w.cosine.resize(static_cast<std::size_t>(t.height) + 1);
    for (int j = 0; j <= t.height; ++j)
    {
        w.cosine[static_cast<std::size_t>(j)] = std::cos(pi * j / t.height);
    }
    w.omega = row_solid_angles(t.width, t.height);
    w.probability.resize(t.texels.size());
    exact_sum total;
    for (std::size_t k = 0; k < t.texels.size(); ++k)
    {
        const gpu::env_texel& x = t.texels[k];
        w.probability[k] = luminance(x.r, x.g, x.b) * w.omega[k / static_cast<std::size_t>(t.width)];
        total.add(w.probability[k]);
    }
    w.total = total.value();
    for (double& p : w.probability)
    {
        p = w.total > 0 ? p / w.total : 0;
    }
    return w;
}

// The solid angle of the cells with light: E[1/p] over the samples
double lit_solid_angle(const gpu::environment_tables& t, const cell_weights& w)
{
    exact_sum sum;
    for (std::size_t k = 0; k < w.probability.size(); ++k)
    {
        if (w.probability[k] > 0)
        {
            sum.add(w.omega[k / static_cast<std::size_t>(t.width)]);
        }
    }
    return sum.value();
}

// The irradiance of a plane facing +y, ∫ L̄·cos⁺θ dω, L̄ the channel average: per cell (2π/W)·(c_j⁺² - c_{j+1}⁺²)/2
double irradiance_up(const gpu::environment_tables& t, const cell_weights& w)
{
    exact_sum sum;
    for (int j = 0; j < t.height; ++j)
    {
        const double hi = std::max(0.0, w.cosine[static_cast<std::size_t>(j)]);
        const double lo = std::max(0.0, w.cosine[static_cast<std::size_t>(j) + 1]);
        const double share = pi / t.width * (hi * hi - lo * lo);
        for (int i = 0; share > 0 && i < t.width; ++i)
        {
            const gpu::env_texel& x = t.texels[static_cast<std::size_t>(j) * t.width + i];
            sum.add((double{x.r} + x.g + x.b) / 3 * share);
        }
    }
    return sum.value();
}

double length_of(const vec3<float>& n)
{
    return std::sqrt(double{n.x} * n.x + double{n.y} * n.y + double{n.z} * n.z);
}

// Two hemispheres (a 1×2 map) and a plane tilted by β from +y: π·(L̄_top·(1 + cos β) + L̄_bottom·(1 - cos β))/2, times
// |n| (the estimator's float normal isn't exactly unit)
double irradiance_hemispheres(const gpu::environment_tables& t, const vec3<float>& n)
{
    const double length = length_of(n);
    const double c = n.y / length;
    const auto average = [&t](std::size_t k) {
        return (double{t.texels[k].r} + t.texels[k].g + t.texels[k].b) / 3;
    };
    return length * pi * (average(0) * (1 + c) + average(1) * (1 - c)) / 2;
}

// A constant map: π·L̄·|n| for any n
double irradiance_constant(const gpu::environment_tables& t, const vec3<float>& n)
{
    return length_of(n) * pi * (double{t.texels[0].r} + t.texels[0].g + t.texels[0].b) / 3;
}

// --- The expected counts (C.md §3, item 4: exact, no quadrature) -----------------------------------------------------

// (U) 101 columns of φ in [-π, π) by 50 rows of cos θ in [-1, 1], about +y in the world. Cell c of the map covers the
// rectangle [φ_i + α, φ_{i+1} + α) × (cos θ_{j+1}, cos θ_j] there (α the yaw, φ_world = φ' + α), uniformly: its
// probability P_c goes to each bin in proportion to the overlap's area. mirrored: cell (i, j)'s weight at column
// W - 1 - i (the planted fault).
constexpr int u_columns = 101;
constexpr int u_rows = 50;
constexpr int u_cells = u_columns * u_rows;

std::vector<double> unaligned_masses(const gpu::environment_tables& t, const cell_weights& w, double alpha,
                                     bool mirrored)
{
    const double column_width = 2 * pi / u_columns;
    const double row_height = 2.0 / u_rows;
    const double cell_width = 2 * pi / t.width;
    std::vector<double> masses(u_cells, 0.0);
    std::vector<std::pair<int, double>> rows; // the bins' rows a map row overlaps, with its share of the row
    for (int j = 0; j < t.height; ++j)
    {
        const double lo = w.cosine[static_cast<std::size_t>(j) + 1];
        const double hi = w.cosine[static_cast<std::size_t>(j)];
        rows.clear();
        const int r0 = std::clamp(static_cast<int>(std::floor((lo + 1) / row_height)), 0, u_rows - 1);
        const int r1 = std::clamp(static_cast<int>(std::floor((hi + 1) / row_height)), 0, u_rows - 1);
        for (int r = r0; r <= r1; ++r)
        {
            const double a = std::max(lo, -1 + r * row_height);
            const double b = std::min(hi, -1 + (r + 1) * row_height);
            if (b > a)
            {
                rows.emplace_back(r, (b - a) / (hi - lo));
            }
        }
        for (int i = 0; i < t.width; ++i)
        {
            const double p = w.probability[static_cast<std::size_t>(j) * t.width + i];
            if (!(p > 0))
            {
                continue;
            }
            const auto spread = [&](double x0, double x1) {
                const int k0 = std::clamp(static_cast<int>(std::floor((x0 + pi) / column_width)), 0, u_columns - 1);
                const int k1 = std::clamp(static_cast<int>(std::floor((x1 + pi) / column_width)), 0, u_columns - 1);
                for (int k = k0; k <= k1; ++k)
                {
                    const double a = std::max(x0, -pi + k * column_width);
                    const double b = std::min(x1, -pi + (k + 1) * column_width);
                    if (b > a)
                    {
                        for (const auto& [r, share] : rows)
                        {
                            masses[static_cast<std::size_t>(r) * u_columns + k] += p * (b - a) / cell_width * share;
                        }
                    }
                }
            };
            const int column = mirrored ? t.width - 1 - i : i;
            double start = -pi + column * cell_width + alpha;
            start -= 2 * pi * std::floor((start + pi) / (2 * pi)); // into [-π, π)
            const double end = start + cell_width;
            if (end <= pi)
            {
                spread(start, end);
            }
            else
            {
                spread(start, pi);
                spread(-pi, end - 2 * pi);
            }
        }
    }
    return masses;
}

// (A) whole cells in min(64, W) × min(32, H) blocks: column i in block i·g_x/W (64 groups of ⌊W/64⌋ or ⌈W/64⌉
// consecutive columns), row j in j·g_y/H; a block's mass is its cells' P
std::vector<double> aligned_masses(const gpu::environment_tables& t, const cell_weights& w, int groups_x,
                                   int groups_y)
{
    std::vector<double> masses(static_cast<std::size_t>(groups_x) * groups_y, 0.0);
    for (int j = 0; j < t.height; ++j)
    {
        for (int i = 0; i < t.width; ++i)
        {
            masses[static_cast<std::size_t>(j * groups_y / t.height) * groups_x + i * groups_x / t.width] +=
                w.probability[static_cast<std::size_t>(j) * t.width + i];
        }
    }
    return masses;
}

// --- N1: the alias table on the host ---------------------------------------------------------------------------------

GPU_TEST(env_sampling, alias)
{
    for (const map_case* m :
         {&studio(), &kloofendal(), &r300(), &k64(), &u1(), &s64(), &h48(), &two_cells(), &dim_cell()})
    {
        const std::string name = "alias-" + m->name;
        const gpu::environment_tables& t = m->tables;
        if (!shaped(t))
        {
            ctx.count(name + "-size", 1, 0, 1);
            continue;
        }
        // k = max(1, ⌈log2 N⌉), P = 2^k ≥ 2
        const std::int64_t cells = static_cast<std::int64_t>(t.width) * t.height;
        int k = 1;
        while ((std::int64_t{1} << k) < cells)
        {
            ++k;
        }
        const std::int64_t bins = std::int64_t{1} << k;
        const bool sized = t.log2_bins == k && static_cast<std::int64_t>(t.bins.size()) == bins;
        ctx.count(name + "-size", sized ? 0 : 1, 0, 1);
        if (!sized)
        {
            continue;
        }
        const cell_weights w = weights_of(t);
        const auto weight = [&](std::int64_t c) {
            return c >= 0 && c < cells ? w.probability[static_cast<std::size_t>(c)] : 0.0;
        };
        // Vose's invariants, and P'_c·P = q_c + Σ_{b: alias_b = c} (1 - q_b) in double from the float q
        std::vector<double> rebuilt(static_cast<std::size_t>(bins), 0.0);
        std::vector<std::int64_t> donors(static_cast<std::size_t>(bins), 0);
        std::int64_t violations = 0;
        for (std::int64_t b = 0; b < bins; ++b)
        {
            const gpu::alias_bin& x = t.bins[static_cast<std::size_t>(b)];
            const bool q_valid = x.q >= 0.0f && x.q <= 1.0f;
            const bool alias_valid = x.alias >= 0 && x.alias < cells;
            bool bad = !q_valid || !alias_valid;
            bad = bad || (weight(b) == 0 && x.q != 0.0f);        // a padded bin, a black cell: never itself
            bad = bad || (x.q < 1.0f && weight(x.alias) == 0); // its alias has light
            violations += bad ? 1 : 0;
            if (q_valid)
            {
                rebuilt[static_cast<std::size_t>(b)] += x.q;
                if (x.q < 1.0f && x.alias >= 0 && x.alias < bins)
                {
                    rebuilt[static_cast<std::size_t>(x.alias)] += 1.0 - x.q;
                    ++donors[static_cast<std::size_t>(x.alias)];
                }
            }
        }
        double worst = 0;
        std::int64_t at = 0;
        for (std::int64_t c = 0; c < bins; ++c)
        {
            const double d = std::abs(rebuilt[static_cast<std::size_t>(c)] - static_cast<double>(bins) * weight(c));
            if (!(d <= worst))
            {
                worst = d;
                at = c;
            }
        }
        // Every cell with light can be drawn, however dim (the forced minimum coin), and none without: exactly
        std::int64_t unreachable = 0;
        for (std::int64_t c = 0; c < bins; ++c)
        {
            const bool lit = weight(c) > 0;
            const bool drawable = rebuilt[static_cast<std::size_t>(c)] > 0;
            unreachable += lit != drawable ? 1 : 0;
        }
        ctx.count(name + "-reachable", unreachable, 0, bins);
        ctx.exact(name, worst, 0, 1e-6);
        ctx.info(name, printed("largest at bin %lld of %lld (P*w/S = %.9g, aliased by %lld bins)",
                               static_cast<long long>(at), static_cast<long long>(bins),
                               static_cast<double>(bins) * weight(at),
                               static_cast<long long>(donors[static_cast<std::size_t>(at)])));
        ctx.count(name + "-bins", violations, 0, bins);
    }
}

// --- Device helpers --------------------------------------------------------------------------------------------------

__device__ inline bool finite3(const vec3<float>& v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z); }
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

// Cells a and b side by side (columns wrapped), or in the same pole row (sin θ = 0 has no azimuth)
__device__ inline bool neighbours(int width, int height, int a, int b)
{
    const int ia = a % width;
    const int ja = a / width;
    const int ib = b % width;
    const int jb = b / width;
    if (::abs(ja - jb) > 1)
    {
        return false;
    }
    const int di = ::abs(ia - ib);
    return ::min(di, width - di) <= 1 || (ja == jb && (ja == 0 || ja == height - 1));
}

// Grid U's bin of a world direction (-1 for none)
__device__ inline int unaligned_bin(const vec3<float>& d)
{
    const double x = d.x;
    const double y = d.y;
    const double z = d.z;
    const double length = ::sqrt(x * x + y * y + z * z);
    if (!(length > 0) || !isfinite(length))
    {
        return -1;
    }
    const double phi = ::atan2(z, x);
    const int i = ::min(::max(static_cast<int>(::floor((phi + pi) / (2 * pi) * u_columns)), 0), u_columns - 1);
    const int j = ::min(::max(static_cast<int>(::floor((y / length + 1) / 2 * u_rows)), 0), u_rows - 1);
    return j * u_columns + i;
}

// --- N2-N8 and N4: the samples of one map ----------------------------------------------------------------------------

enum sample_tally : int
{
    st_calls,    // sample_environment calls
    st_true,     // returned true
    st_pdf,      // the returned pdf ≠ environment_pdf(e, wi), bitwise (N2)
    st_radiance, // le ≠ env_radiance(e, wi), bitwise
    st_border,   // env_cell(e, wi) ≠ the cell of wi in double (N2's border)
    st_far,      // the cell of wi in double neither the table's choice nor a neighbour of it (or no direction)
    st_near,     // a neighbour of it
    st_finite,   // N8
    st_draws,    // not 4 numbers (N7)
    st_uniform,  // pdf ≠ uniform_sphere_density<float> (N5)
    st_bright,   // a true return whose radiance isn't the bright texel's (N6)
    st_count,
};

enum sample_sum : int
{
    ss_inverse,     // 1/p (0 for a false return): E = Ω_lit
    ss_inverse2,
    ss_irradiance,  // L̄·cos⁺(n·wi)/p (0 for a false return)
    ss_irradiance2,
    ss_planted,     // 1/p' with p' = P_c·N/(4π), P_c = p·Ω_c: the planted density
    ss_planted2,
    ss_count,
};

struct sampling_setup
{
    environment_view e;
    double scale_total;      // scale·S: Y(le)/pdf of every sample
    vec3<float> normal;      // the irradiance estimator's
    const double* row_omega; // H: Ω_j from double cosines
    int groups_x;            // grid A: groups_x × groups_y blocks of whole cells
    int groups_y;
    float bright;            // the radiance of every true sample (s64); negative: not checked
};

struct sampling_buffers
{
    unsigned long long* tallies;   // st_count
    unsigned int* identity;        // the largest |Y(le)/pdf - scale·S|/(scale·S), as float bits (NaN as +∞)
    unsigned long long* unaligned; // u_cells + 1
    unsigned long long* aligned;   // groups_x·groups_y + 1
    unsigned long long* bins;      // per bin: samples that landed in its choice (nullptr: not counted)
    double* sums;                  // ss_count per thread, added over the batches
};

// Each sample once through every check: thread t of batch b draws from sample_generator(seed, t, b). (A template, as
// every kernel of the tests: nvcc's host stub of a plain __global__ function in an unnamed namespace draws MSVC's
// C4211.)
template <class Setup>
__global__ void sampling_kernel(Setup s, sampling_buffers out, std::uint64_t seed, std::uint64_t batch, int count)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    gpu::pcg32 rng = gpu::sample_generator(seed, static_cast<std::uint64_t>(t), batch);
    const environment_view& e = s.e;
    const int cells = e.width * e.height;
    const auto shift = static_cast<unsigned int>(32 - e.log2_bins);
    unsigned long long tally[st_count] = {};
    double sums[ss_count] = {};
    float worst = 0.0f;
    for (int k = 0; k < count; ++k)
    {
        // The cell C.md §2.3's numbers (1) and (2) choose, read from a copy of the generator
        gpu::pcg32 copy = rng;
        const unsigned int bin = copy.next() >> shift;
        const float coin = gpu::random_real<float>(copy);
        const gpu::alias_bin entry = e.bins[bin];
        const int chosen = coin < entry.q ? static_cast<int>(bin) : entry.alias;

        vec3<float> wi{};
        vec3<float> le{};
        float pdf = -1.0f;
        const bool sampled = gpu::sample_environment(e, rng, wi, le, pdf);
        ++tally[st_calls];
        tally[st_true] += sampled ? 1 : 0;
        // 4 numbers: the copy has drawn 2, two more and it must be where the generator is
        copy.next();
        copy.next();
        gpu::pcg32 after = rng;
        const bool four = after.next() == copy.next() && after.next() == copy.next();
        tally[st_draws] += four ? 0 : 1;

        const bool unit = finite3(wi) && fabsf(sqrtf(wi.length_squared()) - 1.0f) <= 1e-5f;
        const bool light = finite3(le) && le.x >= 0.0f && le.y >= 0.0f && le.z >= 0.0f;
        const bool density = isfinite(pdf) && pdf >= 0.0f;
        tally[st_finite] += (unit && light && density && sampled == (pdf > 0.0f)) ? 0 : 1;
        if (!finite3(wi) || !(wi.length_squared() > 0.0f))
        {
            ++tally[st_far]; // no direction, no cell
            continue;
        }
        // N2: the density through env_cell, and the radiance with it
        tally[st_pdf] += bits_equal(gpu::environment_pdf(e, wi), pdf) ? 0 : 1;
        float texel_pdf = 0.0f;
        tally[st_radiance] += bits_equal(gpu::env_radiance(e, wi, texel_pdf), le) ? 0 : 1;
        // The cell wi lies in, in double, against env_cell's and against the table's choice
        const int exact = reference_mapping{}.cell(e, wi);
        tally[st_border] += gpu::env_cell(e, wi) == exact ? 0 : 1;
        if (exact != chosen)
        {
            const bool adjacent = chosen >= 0 && chosen < cells && neighbours(e.width, e.height, exact, chosen);
            ++tally[adjacent ? st_near : st_far];
        }
        else if (out.bins != nullptr)
        {
            atomicAdd(&out.bins[bin], 1ull);
        }
        tally[st_uniform] += bits_equal(pdf, gpu::uniform_sphere_density<float>) ? 0 : 1;
        // The two grids
        const int u = unaligned_bin(wi);
        atomicAdd(&out.unaligned[u >= 0 ? u : u_cells], 1ull);
        const int i = exact % e.width;
        const int j = exact / e.width;
        atomicAdd(&out.aligned[(j * s.groups_y / e.height) * s.groups_x + i * s.groups_x / e.width], 1ull);
        if (!sampled)
        {
            continue;
        }
        // N3: Y(L)/p = scale·S
        const double y = 0.2126 * le.x + 0.7152 * le.y + 0.0722 * le.z;
        const double relative = ::fabs(y / pdf - s.scale_total) / s.scale_total;
        worst = ::fmaxf(worst, relative == relative ? static_cast<float>(relative) : gpu::infinity<float>);
        if (s.bright >= 0.0f && !(le.x == s.bright && le.y == s.bright && le.z == s.bright))
        {
            ++tally[st_bright];
        }
        // The estimators
        const double inverse = 1.0 / pdf;
        sums[ss_inverse] += inverse;
        sums[ss_inverse2] += inverse * inverse;
        const double cosine = ::fmax(0.0, static_cast<double>(s.normal.x) * wi.x +
                                              static_cast<double>(s.normal.y) * wi.y +
                                              static_cast<double>(s.normal.z) * wi.z);
        const double irradiance = (static_cast<double>(le.x) + le.y + le.z) / 3 * cosine * inverse;
        sums[ss_irradiance] += irradiance;
        sums[ss_irradiance2] += irradiance * irradiance;
        const double planted = 4 * pi / (cells * static_cast<double>(pdf) * s.row_omega[j]);
        sums[ss_planted] += planted;
        sums[ss_planted2] += planted * planted;
    }
    for (int k = 0; k < st_count; ++k)
    {
        atomicAdd(&out.tallies[k], tally[k]);
    }
    atomicMax(out.identity, __float_as_uint(worst));
    for (int k = 0; k < ss_count; ++k)
    {
        out.sums[ss_count * t + k] += sums[k];
    }
}

struct sampling_outcome
{
    std::int64_t drawn = 0;
    std::array<std::int64_t, st_count> tally{};
    float identity = 0;
    std::vector<std::uint64_t> unaligned;
    std::vector<std::uint64_t> aligned;
    std::vector<std::uint64_t> bins;
    gpu_test::mean_estimate inverse;
    gpu_test::mean_estimate irradiance;
    gpu_test::mean_estimate planted;
};

std::vector<std::uint64_t> widened(const std::vector<unsigned long long>& v) { return {v.begin(), v.end()}; }

// About n samples in launches of at most batch_samples (4096 threads each, the harness's layout)
sampling_outcome run_sampling(const sampling_setup& s, std::uint64_t seed, std::int64_t n, std::size_t bin_count)
{
    const std::int64_t batches = std::max<std::int64_t>(1, (n + batch_samples - 1) / batch_samples);
    const int per_thread = gpu_test::per_thread(n / batches);
    const auto blocks = static_cast<std::size_t>(s.groups_x) * static_cast<std::size_t>(s.groups_y);
    gpu_test::device_buffer<unsigned long long> tallies(static_cast<std::size_t>(st_count));
    gpu_test::device_buffer<unsigned int> identity(1);
    gpu_test::device_buffer<unsigned long long> unaligned(static_cast<std::size_t>(u_cells) + 1);
    gpu_test::device_buffer<unsigned long long> aligned(blocks + 1);
    gpu_test::device_buffer<unsigned long long> bins(bin_count);
    gpu_test::device_buffer<double> sums(static_cast<std::size_t>(ss_count) * gpu_test::test_threads);
    const sampling_buffers out{tallies.get(), identity.get(), unaligned.get(), aligned.get(), bins.get(), sums.get()};
    for (std::int64_t b = 0; b < batches; ++b)
    {
        sampling_kernel<sampling_setup><<<gpu_test::test_blocks, gpu_test::test_block_size>>>(
            s, out, seed, static_cast<std::uint64_t>(b), per_thread);
        gpu_test::finish_launch("the sampling kernel");
    }
    sampling_outcome o;
    o.drawn = batches * per_thread * gpu_test::test_threads;
    const std::vector<unsigned long long> t = tallies.download();
    for (std::size_t k = 0; k < o.tally.size(); ++k)
    {
        o.tally[k] = static_cast<std::int64_t>(t[k]);
    }
    o.identity = std::bit_cast<float>(identity.download()[0]);
    o.unaligned = widened(unaligned.download());
    o.aligned = widened(aligned.download());
    o.bins = widened(bins.download());
    const std::vector<double> per_thread_sums = sums.download();
    std::array<double, ss_count> total{};
    for (std::size_t i = 0; i < per_thread_sums.size(); ++i)
    {
        total[i % ss_count] += per_thread_sums[i]; // in the threads' order: the same on every run
    }
    o.inverse = gpu_test::from_sums(total[ss_inverse], total[ss_inverse2], o.drawn);
    o.irradiance = gpu_test::from_sums(total[ss_irradiance], total[ss_irradiance2], o.drawn);
    o.planted = gpu_test::from_sums(total[ss_planted], total[ss_planted2], o.drawn);
    return o;
}

// A count bounded by a share of the samples (cases at cell borders): the bound needs the full count, so under --quick
// it is printed only
void bounded_count(gpu_test::context& ctx, const std::string& name, std::int64_t found, std::int64_t drawn,
                   double share)
{
    if (ctx.quick())
    {
        ctx.info(name, printed("%lld of %lld samples (at most %g of them in the full run)",
                               static_cast<long long>(found), static_cast<long long>(drawn), share));
        return;
    }
    ctx.count(name, found, static_cast<std::int64_t>(std::ceil(share * static_cast<double>(drawn))), drawn);
}

// The entries of a histogram (its last, outside-the-grid entry aside) that no sample reached; the count needs the full
// run, so under --quick it is printed only
void every_entry(gpu_test::context& ctx, const std::string& name, const std::vector<std::uint64_t>& counts,
                 std::size_t entries)
{
    std::int64_t empty = 0;
    for (std::size_t k = 0; k < entries && k < counts.size(); ++k)
    {
        empty += counts[k] == 0 ? 1 : 0;
    }
    empty += counts.size() < entries ? static_cast<std::int64_t>(entries - counts.size()) : 0;
    if (ctx.quick())
    {
        ctx.info(name, printed("%lld of %zu never reached (all reached in the full run)",
                               static_cast<long long>(empty), entries));
        return;
    }
    ctx.count(name, empty, 0, static_cast<std::int64_t>(entries));
}

enum run_check : unsigned
{
    chi2_unaligned = 1u << 0,    // N4 on grid U
    chi2_aligned = 1u << 1,      // N4 on grid A
    mirrored_fault = 1u << 2,    // N13
    constant_density = 1u << 3,  // N5
    bin_choices = 1u << 4,       // every bin reached
    every_cell = 1u << 5,        // every cell reached (grid A = the cells)
    inverse_density = 1u << 6,   // E[1/p] = Ω_lit
    solid_angle_fault = 1u << 7, // the planted density through E[1/p]'s mean
    bright_texel = 1u << 8,      // N6
};

enum irradiance_reference : int
{
    by_cells,       // n = +y: the per-cell closed form
    by_hemispheres, // a 1×2 map
    by_constant,    // a constant map
};

struct run_spec
{
    const char* name;
    const map_case& (*map)();
    double yaw;
    double scale;
    std::int64_t samples;
    std::uint64_t seed;
    vec3<float> normal;
    irradiance_reference reference;
    unsigned checks;
};

GPU_TEST(env_sampling, samples)
{
    const vec3<float> up{0.0f, 1.0f, 0.0f};
    const double r14 = 1 / std::sqrt(14.0);
    const vec3<float> tilted{static_cast<float>(r14), static_cast<float>(2 * r14), static_cast<float>(3 * r14)};
    const vec3<float> sixty{static_cast<float>(std::sin(pi / 3)), 0.5f, 0.0f}; // β = 60° from +y
    const run_spec runs[] = {
        {"studio", &studio, 0, 1, full_samples, 8301, up, by_cells, chi2_unaligned | chi2_aligned | mirrored_fault},
        {"studio-yaw37", &studio, 37, 2.5, full_samples, 8302, up, by_cells, chi2_unaligned},
        {"kloofendal", &kloofendal, 0, 1, full_samples, 8303, up, by_cells, chi2_unaligned | chi2_aligned},
        {"r300", &r300, 0, 1, full_samples, 8304, up, by_cells, chi2_unaligned | chi2_aligned | bin_choices},
        {"k64", &k64, 0, 1, full_samples, 8305, tilted, by_constant,
         chi2_unaligned | constant_density | bin_choices | every_cell},
        {"u1", &u1, 0, 1, million, 8306, up, by_cells, constant_density | bin_choices},
        {"s64", &s64, 37, 1.5, million, 8307, up, by_cells, bright_texel},
        {"h48", &h48, 37, 2, full_samples, 8308, up, by_cells, bin_choices | inverse_density | solid_angle_fault},
        {"two", &two_cells, 0, 1, million, 8309, sixty, by_hemispheres, bin_choices | inverse_density},
    };
    std::int64_t draw_errors = 0;
    std::int64_t draw_calls = 0;
    for (const run_spec& r : runs)
    {
        const map_case& m = r.map();
        const gpu::environment_tables& t = m.tables;
        const std::string name(r.name);
        if (!has_table(t))
        {
            ctx.count("samples-" + name + "-table", 1, 0, 1);
            continue;
        }
        const cell_weights w = weights_of(t);
        const device_environment device(t);
        const environment_view e = device.view(r.yaw, r.scale);
        const gpu_test::device_buffer<double> row_omega(w.omega);
        sampling_setup s{};
        s.e = e;
        s.scale_total = static_cast<double>(e.scale) * w.total;
        s.normal = r.normal;
        s.row_omega = row_omega.get();
        s.groups_x = std::min(64, t.width);
        s.groups_y = std::min(32, t.height);
        s.bright = (r.checks & bright_texel) != 0 ? e.scale * 100.0f : -1.0f;
        const sampling_outcome o =
            run_sampling(s, r.seed, ctx.samples(r.samples), (r.checks & bin_choices) != 0 ? t.bins.size() : 0);
        const std::int64_t drawn = o.drawn;
        draw_errors += o.tally[st_draws];
        draw_calls += o.tally[st_calls];

        // Every sample: N2, the radiance, the cell, N3, N8. The returned pdf and radiance against environment_pdf and
        // env_radiance of the same direction, at most 10⁻⁶ of the samples apart: two inlined copies of the map's
        // rotation may be contracted apart, which moves a direction on a cell's border into the next cell (9 in 10⁷
        // at a yaw of 37°)
        ctx.count("pdf-" + name, o.tally[st_pdf], static_cast<std::int64_t>(1e-6 * static_cast<double>(drawn)), drawn);
        bounded_count(ctx, "pdf-" + name + "-border", o.tally[st_border], drawn, 1e-4);
        ctx.count("radiance-" + name, o.tally[st_radiance], static_cast<std::int64_t>(1e-6 * static_cast<double>(drawn)),
                  drawn);
        ctx.count("cell-" + name, o.tally[st_far], 0, drawn);
        bounded_count(ctx, "cell-" + name + "-border", o.tally[st_near], drawn, 1e-3);
        ctx.exact("identity-" + name, o.identity, 0, 1e-5);
        ctx.count("finite-" + name, o.tally[st_finite], 0, drawn);
        ctx.info("finite-" + name, printed("%lld of %lld samples returned false (a border sample in a black cell)",
                                           static_cast<long long>(drawn - o.tally[st_true]),
                                           static_cast<long long>(drawn)));
        // The irradiance, against the map's own sum
        const double irradiance = r.reference == by_cells         ? irradiance_up(t, w)
                                  : r.reference == by_hemispheres ? irradiance_hemispheres(t, r.normal)
                                                                  : irradiance_constant(t, r.normal);
        ctx.mean("irradiance-" + name, o.irradiance.mean, o.irradiance.standard_error,
                 static_cast<double>(e.scale) * irradiance);

        const double alpha = std::atan2(static_cast<double>(e.sin_yaw), static_cast<double>(e.cos_yaw));
        if ((r.checks & chi2_unaligned) != 0)
        {
            double sum = 0;
            const gpu_test::chi2_outcome c = gpu_test::chi2_against(o.unaligned, unaligned_masses(t, w, alpha, false),
                                                                    sum);
            ctx.chi2("chi2-" + name + "-u", c.statistic, c.dof, c.p);
            ctx.exact("chi2-" + name + "-u-masses", sum, 1, 1e-9);
        }
        if ((r.checks & chi2_aligned) != 0)
        {
            double sum = 0;
            const gpu_test::chi2_outcome c =
                gpu_test::chi2_against(o.aligned, aligned_masses(t, w, s.groups_x, s.groups_y), sum);
            ctx.chi2("chi2-" + name + "-a", c.statistic, c.dof, c.p);
            ctx.exact("chi2-" + name + "-a-masses", sum, 1, 1e-9);
        }
        if ((r.checks & mirrored_fault) != 0)
        {
            // N13: the same histogram against the map mirrored in u
            double sum = 0;
            const gpu_test::chi2_outcome c = gpu_test::chi2_against(o.unaligned, unaligned_masses(t, w, alpha, true),
                                                                    sum);
            ctx.must_reject("planted-mirror", c.statistic, c.dof, c.p);
        }
        if ((r.checks & constant_density) != 0)
        {
            std::int64_t wrong = 0;
            for (const gpu::env_texel& x : t.texels)
            {
                wrong += bits(x.pdf) == bits(gpu::uniform_sphere_density<float>) ? 0 : 1;
            }
            ctx.count("constant-" + name, wrong + o.tally[st_uniform], 0,
                      static_cast<std::int64_t>(t.texels.size()) + drawn);
        }
        if ((r.checks & bin_choices) != 0)
        {
            every_entry(ctx, "bins-" + name, o.bins, t.bins.size());
        }
        if ((r.checks & every_cell) != 0)
        {
            every_entry(ctx, "cells-reached-" + name, o.aligned, static_cast<std::size_t>(s.groups_x) * s.groups_y);
        }
        const double lit = lit_solid_angle(t, w);
        if ((r.checks & inverse_density) != 0)
        {
            ctx.mean("inverse-pdf-" + name, o.inverse.mean, o.inverse.standard_error, lit);
        }
        if ((r.checks & solid_angle_fault) != 0)
        {
            // The same mean's z for the planted density p' = P_c·N/(4π)
            const double z = o.planted.standard_error > 0 ? (o.planted.mean - lit) / o.planted.standard_error
                                                          : std::numeric_limits<double>::infinity();
            ctx.must_reject("planted-solid-angle", z, 0, gpu_test::normal_two_sided_p(z),
                            gpu_test::normal_two_sided_p(8));
        }
        if ((r.checks & bright_texel) != 0)
        {
            // N6: every sample in texel (17, 9) (the table chooses no other: N1), with its radiance
            ctx.count("bright-texel", o.tally[st_far] + o.tally[st_bright], 0, drawn);
            bounded_count(ctx, "bright-texel-border", o.tally[st_near], drawn, 1e-4);
            bounded_count(ctx, "bright-texel-false", drawn - o.tally[st_true], drawn, 1e-4);
        }
    }
    ctx.count("draws", draw_errors, 0, draw_calls);
}

// --- environment_sampled: maps with and without light, and without a table -------------------------------------------

// A map with light and a table is the set's one light (no lamp: light_count 1, light_pmf 1), and its densities agree;
// a map without (black, its table dropped, or no map at all) is no light: no density, no sample, no number drawn
struct sampled_violation
{
    gpu::quality_view v;
    bool light;
    __device__ bool operator()(gpu::pcg32& rng) const
    {
        const vec3<float> d = random_direction(rng);
        const environment_view& e = v.environment;
        if (gpu::environment_sampled(e) != light || gpu::light_count(v) != (light ? 1 : 0))
        {
            return true;
        }
        if (light)
        {
            const float p = gpu::environment_pdf(e, d);
            return !(isfinite(p) && p >= 0.0f) || !bits_equal(gpu::environment_hit_pdf(v, d), v.light_pmf * p);
        }
        const gpu::pcg32 before = rng;
        vec3<float> wi{};
        vec3<float> le{};
        float pdf = 0.0f;
        gpu::light_sample ls{};
        const bool drew = gpu::sample_environment(e, rng, wi, le, pdf) || gpu::sample_light(v, d, rng, ls);
        gpu::pcg32 a = before;
        gpu::pcg32 b = rng;
        const bool untouched = a.next() == b.next() && a.next() == b.next();
        return drew || !untouched || gpu::environment_pdf(e, d) != 0.0f || gpu::environment_hit_pdf(v, d) != 0.0f;
    }
};

gpu::quality_view map_alone(const environment_view& e, float light_pmf, light_combination c)
{
    gpu::quality_view v{};
    v.microfacets = nullptr;
    v.environment = e;
    v.light_pmf = light_pmf;
    v.combination = c;
    return v;
}

GPU_TEST(env_sampling, sampled)
{
    const std::int64_t n = ctx.samples(small_samples);
    std::uint64_t seed = 8311;
    for (const map_case* m : {&studio(), &kloofendal(), &r300(), &k64(), &u1(), &s64(), &h48(), &two_cells()})
    {
        if (!has_table(m->tables))
        {
            ctx.count("sampled-" + m->name, 1, 0, 1);
            ++seed;
            continue;
        }
        const device_environment device(m->tables);
        const gpu::quality_view v = map_alone(device.view(), 1.0f, light_combination::power_heuristic);
        ctx.count("sampled-" + m->name, gpu_test::device_count(sampled_violation{v, true}, seed++, n), 0,
                  gpu_test::drawn(n));
    }
    const device_environment black(b64().tables);
    ctx.count("sampled-b64",
              gpu_test::device_count(
                  sampled_violation{map_alone(black.view(), 0.0f, light_combination::power_heuristic), false}, 8319, n),
              0, gpu_test::drawn(n));
    const device_environment constant(k64().tables);
    environment_view no_table = constant.view();
    no_table.bins = nullptr;
    ctx.count("sampled-k64-no-table",
              gpu_test::device_count(
                  sampled_violation{map_alone(no_table, 0.0f, light_combination::power_heuristic), false}, 8320, n),
              0, gpu_test::drawn(n));
    ctx.count("sampled-none",
              gpu_test::device_count(sampled_violation{map_alone({}, 0.0f, light_combination::power_heuristic), false},
                                     8321, n),
              0, gpu_test::drawn(n));
}

// --- The kernels of the light-set tests ------------------------------------------------------------------------------

// K counts of events per sample, as the harness's kernels: 4096 threads, thread t from sample_generator(seed, t, 0)
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

template <int K, class F>
std::array<std::int64_t, K> device_tally(const F& f, std::uint64_t seed, std::int64_t n)
{
    gpu_test::device_buffer<unsigned long long> tallies(static_cast<std::size_t>(K));
    tally_kernel<K><<<gpu_test::test_blocks, gpu_test::test_block_size>>>(f, seed, gpu_test::per_thread(n),
                                                                          tallies.get());
    gpu_test::finish_launch("a tally kernel");
    const std::vector<unsigned long long> t = tallies.download();
    std::array<std::int64_t, K> out{};
    for (std::size_t k = 0; k < out.size(); ++k)
    {
        out[k] = static_cast<std::int64_t>(t[k]);
    }
    return out;
}

template <std::size_t K>
void add_tallies(std::array<std::int64_t, K>& sum, const std::array<std::int64_t, K>& more)
{
    for (std::size_t k = 0; k < K; ++k)
    {
        sum[k] += more[k];
    }
}

__device__ inline vec3<float> lamp_centre(const gpu::emitter& e) { return e.corner + 0.5f * e.u + 0.5f * e.v; }

// The index in lamps of the lamp whose front the hit is, or -1 (no hit, a back, no lamp of the set)
__device__ inline int front_lamp(const gpu::quality_view& v, const gpu::bvh_hit<float>& hit)
{
    if (hit.index < 0 || hit.index >= v.base.media.base || !hit.entering)
    {
        return -1;
    }
    const gpu::material<float>& m = v.base.materials[v.base.primitive_materials[hit.index]];
    return m.kind == gpu::material_kind::diffuse_light ? v.lamps.lamp_of[hit.index] : -1;
}

// A white lambertian lobe at x with normal n, the ray arriving along -n (D's T3 and T4)
struct matte_point
{
    gpu::bsdf b;
    gpu::hit_record<float> rec;
    gpu::ray<float> in;
};

__device__ inline matte_point make_matte(const vec3<float>& x, const vec3<float>& n)
{
    matte_point p{};
    gpu::material<float> m{};
    m.kind = gpu::material_kind::lambertian;
    m.albedo = {1.0f, 1.0f, 1.0f};
    p.rec.p = x;
    p.rec.normal = n;
    p.rec.front_face = true;
    p.in = {x + n, -n};
    p.b = gpu::make_bsdf(m, nullptr, m.albedo, p.rec, p.in.direction);
    return p;
}

// --- N9, N11: Veach's condition W1 with the map as a light -----------------------------------------------------------

enum t3_tally : int
{
    t3_tested,   // directions W1 applies to
    t3_wrong,    // |w_L + w_B - 1| beyond the tolerance, a weight outside [0, 1], or a lit direction without a density
    t3_far,      // |w_L + w_B - 1| > 10^-2 (the planted fault)
    t3_maps,     // light samples from the map
    t3_pdf,      // ... whose pdf isn't environment_hit_pdf's of their direction, bitwise
    t3_occluded, // ... that the lamp hides (skipped)
    t3_count,
};

struct t3_sample
{
    gpu::quality_view v;
    bool light_drawn; // (a) a light sample's direction, (b) the material's own
    bool planted;     // the map's density on the material's side without light_pmf

    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally) const
    {
        const bool lamps = v.lamps.count > 0;
        const vec3<float> x = lamps ? point_in_box(rng, {-1.0f, -1.0f, -1.5f}, {2.0f, 2.0f, 1.5f})
                                    : point_in_box(rng, {-5.0f, -5.0f, -5.0f}, {5.0f, 5.0f, 5.0f});
        // A random normal, turned toward the lamp's centre for half the points (D.md §6's T3)
        vec3<float> n = gpu::random_unit_vector<float>(rng);
        if (lamps && gpu::random_real<float>(rng) < 0.5f && gpu::dot(n, lamp_centre(v.lamps.lamps[0]) - x) < 0.0f)
        {
            n = -n;
        }
        const matte_point p = make_matte(x, n);
        const gpu::bvh_tracer tracer{v};
        const light_combination c = v.combination;
        float w_light = 0.0f;
        float w_bsdf = 0.0f;
        if (light_drawn)
        {
            gpu::light_sample ls;
            if (!gpu::sample_light(v, x, rng, ls))
            {
                return;
            }
            float p_light = 0.0f;
            if (ls.skip < 0)
            {
                // The map: a ray of the material along wi escapes (unless the lamp hides it) with this density
                const float hit_pdf = gpu::environment_hit_pdf(v, ls.wi);
                ++tally[t3_maps];
                tally[t3_pdf] += bits_equal(hit_pdf, ls.pdf) ? 0 : 1;
                p_light = planted ? gpu::environment_pdf(v.environment, ls.wi) : hit_pdf;
            }
            float p_bsdf = 0.0f;
            const vec3<float> f = gpu::eval_bsdf(p.b, ls.wi, p_bsdf);
            if (!(p_bsdf > 0.0f) || !positive(f) || !positive(ls.le))
            {
                return; // f·Le = 0: no light to weigh
            }
            if (ls.skip < 0)
            {
                if (lamps && tracer.visibility({x, ls.wi}, gpu::spawn{-1, false}, ls.distance, ls.skip) == 0.0f)
                {
                    ++tally[t3_occluded]; // its light is the lamp's, which the lamp's own technique weighs
                    return;
                }
            }
            else
            {
                const gpu::bvh_hit<float> hit = tracer.closest({x, ls.wi}, gpu::spawn{-1, false}, rng);
                if (hit.index != ls.skip || !hit.entering)
                {
                    return; // an edge: T1's business
                }
                p_light = gpu::lamp_hit_pdf(v, hit.index, x, ls.wi, hit.t * sqrtf(ls.wi.length_squared()));
            }
            w_light = gpu::light_weight(c, ls.pdf, p_bsdf);
            w_bsdf = gpu::bsdf_weight(c, p_bsdf, p_light);
        }
        else
        {
            gpu::bsdf_sample bs;
            if (!gpu::sample_bsdf(p.b, p.in, p.rec, rng, bs))
            {
                return;
            }
            float p_eval = 0.0f;
            gpu::eval_bsdf(p.b, bs.wi, p_eval);
            const gpu::bvh_hit<float> hit = lamps ? tracer.closest({x, bs.wi}, gpu::spawn{-1, false}, rng)
                                                  : gpu::bvh_hit<float>{-1, 0.0f, false};
            float p_light = 0.0f;
            if (hit.index < 0)
            {
                // An escape after a vertex with a density: the nearest texel, and the map's density with the pmf
                float texel_pdf = 0.0f;
                if (!positive(gpu::env_radiance(v.environment, bs.wi, texel_pdf)))
                {
                    return;
                }
                p_light = planted ? gpu::environment_pdf(v.environment, bs.wi) : gpu::environment_hit_pdf(v, bs.wi);
            }
            else if (front_lamp(v, hit) >= 0)
            {
                p_light = gpu::lamp_hit_pdf(v, hit.index, x, gpu::unit_vector(bs.wi),
                                            hit.t * sqrtf(bs.wi.length_squared()));
            }
            else
            {
                return; // the lamp's back: no light
            }
            if (!(p_light > 0.0f) || !isfinite(p_light))
            {
                ++tally[t3_tested];
                ++tally[t3_wrong]; // light the light technique could have drawn must have its density
                return;
            }
            // The sample's own density in w_B, eval_bsdf's in w_L, as the integrator pairs them (D.md §10.7)
            w_bsdf = gpu::bsdf_weight(c, bs.pdf, p_light);
            w_light = gpu::light_weight(c, p_light, p_eval);
        }
        ++tally[t3_tested];
        const float sum = w_light + w_bsdf;
        // lights_only: 1 + 0 exactly; the power heuristic: P(a, b) + P(b, a), to rounding
        if (!(w_light >= 0.0f && w_light <= 1.0f && w_bsdf >= 0.0f && w_bsdf <= 1.0f) ||
            !(c == light_combination::lights_only ? sum == 1.0f : fabsf(sum - 1.0f) <= 1e-5f))
        {
            ++tally[t3_wrong];
        }
        if (!(fabsf(sum - 1.0f) <= 1e-2f))
        {
            ++tally[t3_far];
        }
    }
};

// (a) and (b), n samples each
std::array<std::int64_t, t3_count> both_techniques(const gpu::quality_view& v, std::uint64_t seed, std::int64_t n)
{
    std::array<std::int64_t, t3_count> total{};
    add_tallies(total, device_tally<t3_count>(t3_sample{v, true, false}, seed, n));
    add_tallies(total, device_tally<t3_count>(t3_sample{v, false, false}, seed + 1, n));
    return total;
}

GPU_TEST(env_sampling, t3_env)
{
    const map_case& m = studio();
    if (!has_table(m.tables))
    {
        ctx.count("t3-env-table", 1, 0, 1);
        return;
    }
    const device_environment device(m.tables);
    const std::int64_t n = ctx.samples(million);
    std::int64_t pdf_wrong = 0;
    std::int64_t maps = 0;
    std::uint64_t seed = 8322;
    for (const light_combination c : {light_combination::power_heuristic, light_combination::lights_only})
    {
        const auto total = both_techniques(map_alone(device.view(), 1.0f, c), seed, n);
        seed += 2;
        const std::string name = c == light_combination::power_heuristic ? "t3-env" : "t3-env-nee";
        ctx.count(name, total[t3_wrong], 0, total[t3_tested]);
        ctx.info(name, printed("%lld directions from both techniques", static_cast<long long>(total[t3_tested])));
        pdf_wrong += total[t3_pdf];
        maps += total[t3_maps];
    }
    ctx.count("t3-env-pdf", pdf_wrong, 0, maps);
}

// --- N10-N12: a quad lamp and the map --------------------------------------------------------------------------------

// N10's lamp: q (0, 0, 1), u (0, 1, 0), v (1, 0, 0), u × v = -z, facing x below it; L = 4
gpu::scene_desc square_lamp()
{
    gpu::scene_desc s;
    s.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = {4, 4, 4}});
    s.materials.push_back({.kind = gpu::material_kind::diffuse_light, .texture = 0});
    s.quads.push_back({.q = {0, 0, 1}, .u = {0, 1, 0}, .v = {1, 0, 0}, .material = 0});
    return s;
}

// The lamp's BVH and light set (build_lamps, as quality.cu makes them) and a map: two lights, light_pmf ½
class lamp_and_map
{
public:
    lamp_and_map(const gpu::environment_tables& map, double scale)
        : scene_(square_lamp()),
          device_(scene_),
          table_(gpu::build_lamps(scene_, device_.sphere_ids(), device_.quad_ids())),
          lamps_(table_.lamps),
          lamp_of_(table_.lamp_of),
          map_(map),
          scale_(scale)
    {
    }

    gpu::quality_view view(light_combination c) const
    {
        gpu::quality_view v{};
        v.base = device_.view();
        v.microfacets = nullptr;
        v.lamps = {.lamps = lamps_.get(), .lamp_of = lamp_of_.get(), .count = lamps()};
        v.environment = map_.view(0, scale_);
        v.light_pmf = static_cast<float>(1.0 / (lamps() + 1)); // the lamps and the sampled map, in double
        v.combination = c;
        return v;
    }
    int lamps() const { return static_cast<int>(table_.lamps.size()); }
    int lamp_id() const { return table_.lamps.empty() ? -2 : table_.lamps[0].primitive; }

private:
    gpu::scene_desc scene_;
    gpu_test::device_scene device_;
    gpu::lamp_table table_;
    gpu_test::device_buffer<gpu::emitter> lamps_;
    gpu_test::device_buffer<int> lamp_of_;
    device_environment map_;
    double scale_;
};

enum lm_tally : int
{
    lm_lamp,  // true returns from the lamp
    lm_map,   // ... from the map
    lm_false, // false returns
    lm_wrong, // a violation
    lm_edge,  // a lamp sample whose traced ray misses the lamp's front, or ends off its distance by more than 10^-4
    lm_count,
};

// N10 at a fixed point: each source's sample as C.md §2.3 and D.md §3 say
struct lamp_and_map_sample
{
    gpu::quality_view v;
    vec3<float> x;
    int lamp;
    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally) const
    {
        gpu::light_sample ls;
        if (!gpu::sample_light(v, x, rng, ls))
        {
            ++tally[lm_false];
            return;
        }
        const bool numbers = isfinite(ls.pdf) && ls.pdf > 0.0f && finite3(ls.wi) &&
                             fabsf(sqrtf(ls.wi.length_squared()) - 1.0f) <= 1e-5f;
        if (ls.skip == -1)
        {
            ++tally[lm_map];
            const bool right = numbers && ls.distance == gpu::infinity<float> &&
                               bits_equal(ls.pdf, gpu::environment_hit_pdf(v, ls.wi));
            tally[lm_wrong] += right ? 0 : 1;
            return;
        }
        ++tally[lm_lamp];
        if (!numbers || ls.skip != lamp || !bits_equal(ls.le, vec3<float>{4.0f, 4.0f, 4.0f}) ||
            !(isfinite(ls.distance) && ls.distance > 0.0f))
        {
            ++tally[lm_wrong];
            return;
        }
        const gpu::bvh_tracer tracer{v};
        const gpu::bvh_hit<float> hit = tracer.closest({x, ls.wi}, gpu::spawn{-1, false}, rng);
        if (hit.index != ls.skip || !hit.entering || !(fabsf(hit.t - ls.distance) <= 1e-4f * ls.distance))
        {
            ++tally[lm_edge];
            return;
        }
        const float p = gpu::lamp_hit_pdf(v, ls.skip, x, ls.wi, hit.t * sqrtf(ls.wi.length_squared()));
        tally[lm_wrong] += fabsf(p - ls.pdf) <= 1e-4f * ls.pdf ? 0 : 1;
    }
};

// N7 for sample_light: the first number picks k = min(⌊ξ·N⌋, N - 1), the lamps first, the map last; 3 numbers for the
// lamp, 5 for the map, true or false; and the light sampled is the one picked
struct light_pick
{
    gpu::quality_view v;
    int lamp;
    __device__ void operator()(gpu::pcg32& rng, unsigned long long* tally) const
    {
        ++tally[0];
        const vec3<float> x = point_in_box(rng, {-1.0f, -1.0f, -1.5f}, {2.0f, 2.0f, 1.5f});
        gpu::pcg32 copy = rng;
        const int n = v.lamps.count + 1;
        const int k = ::min(static_cast<int>(gpu::random_real<float>(copy) * static_cast<float>(n)), n - 1);
        const bool map = k == v.lamps.count;
        gpu::light_sample ls;
        const bool sampled = gpu::sample_light(v, x, rng, ls);
        for (int i = map ? 4 : 2; i > 0; --i)
        {
            copy.next();
        }
        gpu::pcg32 after = rng;
        tally[1] += after.next() == copy.next() && after.next() == copy.next() ? 0 : 1;
        tally[2] += sampled && ls.skip != (map ? -1 : lamp) ? 1 : 0;
    }
};

GPU_TEST(env_sampling, lamp_and_map_t1)
{
    if (!has_table(studio().tables))
    {
        ctx.count("lamp-and-map-table", 1, 0, 1);
        return;
    }
    const lamp_and_map setup(studio().tables, 1.0);
    ctx.exact("lamp-and-map-lamps", setup.lamps(), 1, 0);
    if (setup.lamps() != 1)
    {
        return;
    }
    const gpu::quality_view v = setup.view(light_combination::power_heuristic);
    const std::int64_t n = ctx.samples(million);
    const std::int64_t drawn = gpu_test::drawn(n);
    const auto t = device_tally<lm_count>(lamp_and_map_sample{v, {0.3f, 0.2f, 0.0f}, setup.lamp_id()}, 8331, n);
    const std::int64_t returned = t[lm_lamp] + t[lm_map];
    const double share = returned > 0 ? static_cast<double>(t[lm_map]) / static_cast<double>(returned) : 0;
    const double binomial = std::sqrt(0.25 / static_cast<double>(std::max<std::int64_t>(1, returned)));
    ctx.mean("lamp-and-map-t1-sources", share, binomial, 0.5);
    ctx.count("lamp-and-map-t1", t[lm_wrong], 0, returned);
    ctx.count("lamp-and-map-t1-edges", t[lm_edge], (drawn + 9999) / 10000, t[lm_lamp]);
    ctx.info("lamp-and-map-t1", printed("%lld lamp and %lld map samples, %lld false returns",
                                        static_cast<long long>(t[lm_lamp]), static_cast<long long>(t[lm_map]),
                                        static_cast<long long>(t[lm_false])));
    const auto p = device_tally<3>(light_pick{v, setup.lamp_id()}, 8332, n);
    ctx.count("draws-light", p[1], 0, p[0]);
    ctx.count("lamp-and-map-pick", p[2], 0, p[0]);
}

GPU_TEST(env_sampling, lamp_and_map_t3)
{
    if (!has_table(studio().tables))
    {
        ctx.count("lamp-and-map-t3-table", 1, 0, 1);
        return;
    }
    const lamp_and_map setup(studio().tables, 1.0);
    if (setup.lamps() != 1)
    {
        ctx.count("lamp-and-map-t3-lamps", 1, 0, 1);
        return;
    }
    const std::int64_t n = ctx.samples(million);
    std::uint64_t seed = 8333;
    for (const light_combination c : {light_combination::power_heuristic, light_combination::lights_only})
    {
        const auto total = both_techniques(setup.view(c), seed, n);
        seed += 2;
        const std::string name =
            c == light_combination::power_heuristic ? "lamp-and-map-t3" : "lamp-and-map-t3-nee";
        ctx.count(name, total[t3_wrong], 0, total[t3_tested]);
        ctx.count(name + "-pdf", total[t3_pdf], 0, total[t3_maps]);
        ctx.info(name, printed("%lld directions from both techniques; %lld map directions the lamp hides, skipped",
                               static_cast<long long>(total[t3_tested]), static_cast<long long>(total[t3_occluded])));
    }
    // The planted fault: the map's density on the material's side without light_pmf
    const auto planted =
        device_tally<t3_count>(t3_sample{setup.view(light_combination::power_heuristic), true, true}, 8337, n);
    ctx.count("planted-env-pmf", planted[t3_far] > 0 ? 0 : 1, 0, 1);
    ctx.info("planted-env-pmf", printed("%lld of %lld directions with |w_L + w_B - 1| > 1e-2",
                                        static_cast<long long>(planted[t3_far]),
                                        static_cast<long long>(planted[t3_tested])));
}

// N12: D's T4 estimator at one vertex, the map added on both sides (D.md §2.3): a white lambertian point x (from -1)
// with normal n, β = 1
struct lit_point
{
    gpu::quality_view v;
    vec3<float> x;
    vec3<float> n;

    __device__ double operator()(gpu::pcg32& rng) const
    {
        const matte_point p = make_matte(x, n);
        const gpu::bvh_tracer tracer{v};
        const light_combination c = v.combination;
        vec3<float> sum{};
        if (c != light_combination::bsdf_only)
        {
            gpu::light_sample ls;
            if (gpu::sample_light(v, x, rng, ls))
            {
                float p_bsdf = 0.0f;
                const vec3<float> f = gpu::eval_bsdf(p.b, ls.wi, p_bsdf);
                if (positive(f) && ls.pdf > 0.0f)
                {
                    const vec3<float> contribution = (gpu::light_weight(c, ls.pdf, p_bsdf) / ls.pdf) * (f * ls.le);
                    if (finite3(contribution) && positive(contribution))
                    {
                        const gpu::ray<float> shadow{x, ls.wi, 0.0f};
                        sum = sum + tracer.visibility(shadow, gpu::spawn{-1, false}, ls.distance, ls.skip) *
                                        contribution;
                    }
                }
            }
        }
        gpu::bsdf_sample bs;
        if (gpu::sample_bsdf(p.b, p.in, p.rec, rng, bs))
        {
            const gpu::bvh_hit<float> hit = tracer.closest({x, bs.wi}, gpu::spawn{-1, false}, rng);
            const float p_bsdf = c != light_combination::bsdf_only ? bs.pdf : 0.0f;
            if (front_lamp(v, hit) >= 0)
            {
                const float p_light = gpu::lamp_hit_pdf(v, hit.index, x, gpu::unit_vector(bs.wi),
                                                        hit.t * sqrtf(bs.wi.length_squared()));
                const vec3<float> le = v.base.materials[v.base.primitive_materials[hit.index]].albedo;
                sum = sum + gpu::bsdf_weight(c, p_bsdf, p_light) * (bs.weight * le);
            }
            else if (hit.index < 0)
            {
                float texel_pdf = 0.0f;
                const vec3<float> le = gpu::env_radiance(v.environment, bs.wi, texel_pdf);
                sum = sum + gpu::bsdf_weight(c, p_bsdf, gpu::environment_hit_pdf(v, bs.wi)) * (bs.weight * le);
            }
        }
        return (static_cast<double>(sum.x) + sum.y + sum.z) / 3;
    }
};

GPU_TEST(env_sampling, lamp_and_map_irradiance)
{
    // L_o = c + (L - c)·Ω⊥/π: the constant map c = 0.5 everywhere but where the lamp L = 4 hides it, Ω⊥ of the unit
    // square over the corner at height 1 = atan(1/√2)/√2 (D.md §5.1)
    const double projected = std::atan(1 / std::sqrt(2.0)) / std::sqrt(2.0);
    const double expected = 0.5 + (4 - 0.5) * projected / pi;
    ctx.exact("lamp-and-map-reference", expected, 0.9848605, 1e-6);
    if (!has_table(k64().tables))
    {
        ctx.count("lamp-and-map-irradiance-table", 1, 0, 1);
        return;
    }
    const lamp_and_map setup(k64().tables, 0.5); // k64 scaled to 0.5
    if (setup.lamps() != 1)
    {
        ctx.count("lamp-and-map-irradiance-lamps", 1, 0, 1);
        return;
    }
    const std::int64_t n = ctx.samples(million);
    std::uint64_t seed = 8341;
    for (const light_combination c :
         {light_combination::bsdf_only, light_combination::lights_only, light_combination::power_heuristic})
    {
        const gpu_test::mean_estimate m =
            gpu_test::device_mean(lit_point{setup.view(c), {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}, seed++, n);
        const std::string name = std::string("lamp-and-map-irradiance-") + mode_name(c);
        ctx.mean(name, m.mean, m.standard_error, expected);
        ctx.count(name + "-nan", m.left_out, 0, gpu_test::drawn(n));
    }
}

// --- The integrator under a lamp and a map ---------------------------------------------------------------------------

// A matte floor and a wall under a quad lamp facing the floor, and the two-hemisphere map (two_image: the upper half
// brighter, a light of its own): the light set holds the lamp and the map, each picked half the time (light_pmf ½).
// Rendered through gpu::render, so the escapes' weights are the integrator's own (the radiance and the density from one
// lookup, times the pmf): -p nee and -p mis against two renders of -p material, at depths 2 and 3
gpu::scene_desc lamp_and_map_scene(int depth, gpu::light_combination c, int size, int sqrt_spp)
{
    gpu::scene_desc s;
    s.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = {0.8, 0.8, 0.8}});
    s.materials.push_back({.kind = gpu::material_kind::lambertian, .texture = 0});
    s.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = {4, 4, 4}});
    s.materials.push_back({.kind = gpu::material_kind::diffuse_light, .texture = 1});
    s.quads.push_back({.q = {-1, 0, -1}, .u = {0, 0, 2}, .v = {2, 0, 0}, .material = 0});     // floor, normal +y
    s.quads.push_back({.q = {-1, 0, -1}, .u = {2, 0, 0}, .v = {0, 2, 0}, .material = 0});     // wall, normal +z
    s.quads.push_back({.q = {-0.5, 2, -0.5}, .u = {1, 0, 0}, .v = {0, 0, 1}, .material = 1}); // lamp, normal -y
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
    s.quality.environment = gpu::environment_desc{.image = two_image()};
    return s;
}

// The pixels' channel-average differences B − A: their mean and its standard error
gpu_test::mean_estimate image_difference(const gpu::render_result& a, const gpu::render_result& b)
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

GPU_TEST(env_sampling, integrator)
{
    const bool quick = ctx.quick();
    const int size = quick ? 32 : 64;
    const int sqrt_spp = quick ? 2 : 16;
    std::uint64_t seed = 8360;
    std::int64_t not_finite = 0;
    std::int64_t values = 0;
    for (int depth = 2; depth <= 3; ++depth)
    {
        const auto render = [&](gpu::light_combination c) {
            return gpu::render(lamp_and_map_scene(depth, c, size, sqrt_spp), {.seed = seed++, .quality_kernel = true});
        };
        const gpu::render_result material[2] = {render(gpu::light_combination::bsdf_only),
                                                render(gpu::light_combination::bsdf_only)};
        const gpu::render_result sampled[2] = {render(gpu::light_combination::lights_only),
                                               render(gpu::light_combination::power_heuristic)};
        for (const gpu::render_result* image : {&material[0], &material[1], &sampled[0], &sampled[1]})
        {
            for (const double x : image->pixels)
            {
                not_finite += std::isfinite(x) ? 0 : 1;
            }
            values += static_cast<std::int64_t>(image->pixels.size());
        }
        const std::string d = "-d" + std::to_string(depth);
        const gpu_test::mean_estimate nee = image_difference(material[0], sampled[0]);
        const gpu_test::mean_estimate mis = image_difference(material[1], sampled[1]);
        ctx.mean("integrator-nee" + d, nee.mean, nee.standard_error, 0);
        ctx.mean("integrator-mis" + d, mis.mean, mis.standard_error, 0);
    }
    ctx.count("integrator-finite", not_finite, 0, values);
}
}
