// Suite env-map: the HDR environment map as the quality kernel sees it (Step 82). On the host, build_environment's
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
#include "camera.cuh"
#include "environment.cuh"
#include "environment_data.h"
#include "gpu_test.cuh"
#include "gpu_test.h"
#include "host_services.h"
#include "quality_view.cuh"
#include "random.cuh"

namespace
{
using gpu::environment_view;
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
}
