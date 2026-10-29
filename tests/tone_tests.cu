// Suite tone: the display transform beyond the books (Step 74). The tone maps none, agx and neutral and the sRGB curve
// after them (tone_curves.cuh), the dither and the bytes, the kernel that turns bands of the image's sums into those
// bytes (to_rgba8_toned, tonemap.cu), and a session's display through them (renderer.h). Every reference is a double
// implementation of the spec's equations, out/m4/spec/A.md §2.3 and §3, written from the spec alone: the device
// functions are only ever what is checked. The checks are A.md §5's T1-T18; the rules shared.md §0-§5.
//
// No χ² test (T = 0): every check is an identity within a tolerance (exact) or a count of violations (count). The
// self-check (shared.md §5.2) is T17: T1's AgX comparison, run against a reference with both matrices transposed,
// must see a difference above 10^-3.
//
// Decisions where the spec leaves a choice, taken in its spirit:
//  1. T8 (neutral-hue): u = (out − min out)/|·| is only as good as the float `out` it is made of. Each channel rounds
//     by up to 2^-24 of the largest, and PBR Neutral flattens bright colours toward white (at brightness 2^8 the
//     factor g is 0.025), so |out − min out| can be 10^-4 of the largest channel. A colour whose REFERENCE output has
//     |out − min out| below 1% of its largest channel (where rounding alone could move u by 3·10^-5) is excused and
//     counted; the count depends on the reference only (about 1% of the colours, nearly all above brightness 2^6).
//     |u − w| is the Euclidean norm; the colours are three uniform channels scaled so that the largest is the
//     brightness 2^U, U uniform in [−4, 8].
//  2. T11 (ign): the outer wrap is compared by circular distance (0.99997 and 0.00001 are 4·10^-5 apart). The inner
//     fract wraps too: where 0.06711056x + 0.00583715y lies within 10^-5 of an integer, float and double may fall on
//     opposite sides and IGN jumps by 0.98; a point there that differs is excused and counted.
//  3. T12 under --quick: the means at 64×64 pixels print INFO lines instead of verdicts (the runner skips only the
//     statistical kinds, and these are exact checks at 1024×1024); v = 0 and v = 1 still decide.
//  4. T13: the image's colours are T1's specials (all 664 of them at 64×48) followed by evenly spaced greys, colours
//     and hues; under --quick (16×12) half the pixels come from the specials. The bands are rows [0, 20) and [20, 48);
//     at 16×12, rows [0, 5) and [5, 12). "±1, counted, ≤ 2% of bytes" counts the bytes that do differ by one inside
//     the 10^-3 zone.
//  5. T14 runs the two bands in the other order the second time, so that each band's launch is seen leaving the
//     other band's sentinel bytes alone; the final bytes of the two runs must be identical.
//  6. T15 adds checks from A.md §2.1-§2.2 beyond its list: rays and shadow rays unchanged by the display, exposures
//     of −25 and +∞ refused and ±24 accepted, display(options) before a complete pass throws std::logic_error, and
//     render() in float64 with agx throws std::invalid_argument.
//  7. T7's straddling inputs are tried with the varying channel in each of the three positions.
//  8. T10(a)'s quantize(1, d) = 255 is checked for (b)'s five dithers; (c) draws v uniform in (0, 1) and d uniform in
//     [−0.5, 0.5), both as floats.
//  9. Sizes under --quick: T1 takes G 4000 + C 4335 + P 7·143 + X 664 = 10^4 inputs (C a prefix of the full run's
//     draws); T4 the points x = k/8192, k = 0…8192 (8193 of them, the same pairwise rule at a wider step).
// 10. +∞ is sanitized to the float 3.0e38f in the reference too (A.md §8.1).
// 11. Seeds (literals, never retuned): 7401 T1's colours (and T13's), 7406 T6, 7408 T8, 7410 T10, 7415 T15's renders.
//
// Interfaces as A.md §1 declares them: tone_map and display_options in renderer.h (namespace gpu, which
// tone_curves.cuh and tonemap.cuh name), gpu::tone's functions called from device code, gpu::detail::to_rgba8_toned.
// The step's image and protocol criteria (A.md §7) are at the end of this file.
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "renderer.h" // tone_map and display_options, which the two headers below name
#include "tone_curves.cuh"
#include "tonemap.cuh"

#include "gpu_test.cuh"
#include "gpu_test.h"
#include "host_services.h"

namespace tone_test_kernels
{
// One index per thread in a grid-stride loop, over a functor of the test's (outside the unnamed namespace, as
// gpu_test.cuh's kernels are)
template <class F>
__global__ void each_index(F f, int n)
{
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    for (int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); k < n; k += stride)
    {
        f(k);
    }
}
}

namespace
{
using gpu::tone_map;
using rgb_d = std::array<double, 3>;

constexpr tone_map toned_ops[] = {tone_map::none, tone_map::agx, tone_map::neutral};
constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr float float_infinity = std::numeric_limits<float>::infinity();
constexpr std::uint8_t sentinel = 0x5A;

const char* name_of(tone_map op)
{
    switch (op)
    {
    case tone_map::none:
        return "none";
    case tone_map::agx:
        return "agx";
    case tone_map::neutral:
        return "neutral";
    default:
        return "books";
    }
}

std::string number(double v)
{
    char text[64];
    std::snprintf(text, sizeof text, "%.9g", v);
    return text;
}

// At most 256 blocks of 128 threads: every array here is small (10^6 values at most), each launch well under 50 ms
template <class F>
void run_each(const F& f, int n, const char* what)
{
    if (n <= 0)
    {
        return;
    }
    const int blocks = std::min(256, (n + gpu_test::test_block_size - 1) / gpu_test::test_block_size);
    tone_test_kernels::each_index<<<blocks, gpu_test::test_block_size>>>(f, n);
    gpu_test::finish_launch(what);
}

// The largest value so far; a NaN makes it +∞ for good, so that no check passes on a NaN
void note_worst(double& worst, double v)
{
    if (!(v <= worst))
    {
        worst = std::isnan(v) ? infinity : v;
    }
}

bool near_integer(double t, double margin) { return std::abs(t - std::round(t)) <= margin; }

// --- The references: A.md §2.3 and §3, in double --------------------------------------------------------------------

double ref_clamp01(double v) { return std::min(std::max(v, 0.0), 1.0); }

// s(v) of §2.3: NaN, −0, negatives and −∞ to 0; +∞ and anything above to 3.0e38f, the float the device clamps to
double ref_sanitize(double v) { return v > 0 ? std::min(v, static_cast<double>(3.0e38f)) : 0.0; }

// The sRGB curve (IEC 61966-2-1), §3.1: input in [0, 1]
double ref_srgb(double l) { return l <= 0.0031308 ? 12.92 * l : 1.055 * std::pow(l, 1 / 2.4) - 0.055; }

// AgX's two matrices (§3.2), acting on column vectors: out_i = Σ_j M[i][j]·in_j, row 0 the red output
struct agx_matrices
{
    double inset[3][3];
    double outset[3][3];
};

const agx_matrices spec_matrices{
    {{0.842479062253094, 0.0784335999999992, 0.0792237451477643},
     {0.0423282422610123, 0.878468636469772, 0.0791661274605434},
     {0.0423756549057051, 0.0784336, 0.879142973793104}},
    {{1.19687900512017, -0.0980208811401368, -0.0990297440797205},
     {-0.0528968517574562, 1.15190312990417, -0.0989611768448433},
     {-0.0529716355144438, -0.0980434501171241, 1.15107367264116}}};

// The planted fault of T17: both matrices transposed, as a column-major mat3 read row by row would give them
agx_matrices transposed(const agx_matrices& m)
{
    agx_matrices t{};
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            t.inset[i][j] = m.inset[j][i];
            t.outset[i][j] = m.outset[j][i];
        }
    }
    return t;
}

// linlin's closed form of Troy Sobotka's sigmoid (MIT © 2024 linlin, github.com/bWFuanVzYWth/AgX):
// f(x) = 0.5 + 2d·(1 + a·|d|^b)^c, d = x − 10/16.5, with one (a, b, c) on each side of the middle grey's x
double ref_agx_curve(double x)
{
    constexpr double t = 10 / 16.5;
    const double d = x - t;
    const bool low = x <= t;
    const double a = low ? 59.507875 : 69.86278913545539;
    const double b = low ? 3.0 : 13.0 / 4;
    const double c = low ? -1.0 / 3 : -4.0 / 13;
    return 0.5 + 2 * d * std::pow(1 + a * std::pow(std::abs(d), b), c);
}

// AgX base (§3.2): inset, log2 over [−12.47393, 4.026069] (16.5 stops around middle grey), the sigmoid, outset,
// clamped; display-encoded, no OETF after it. The input is sanitized (≥ 0, finite).
rgb_d ref_agx(const rgb_d& c, const agx_matrices& m)
{
    constexpr double lowest = -12.47393;
    constexpr double highest = 4.026069;
    double y[3];
    for (int i = 0; i < 3; ++i)
    {
        const double a = m.inset[i][0] * c[0] + m.inset[i][1] * c[1] + m.inset[i][2] * c[2];
        const double x = ref_clamp01((std::log2(a) - lowest) / (highest - lowest)); // log2(0) = −∞ → 0
        y[i] = ref_agx_curve(x);
    }
    rgb_d e{};
    for (int i = 0; i < 3; ++i)
    {
        e[i] = ref_clamp01(m.outset[i][0] * y[0] + m.outset[i][1] * y[1] + m.outset[i][2] * y[2]);
    }
    return e;
}

// Khronos PBR Neutral (§3.3), linear in, linear out, for c ≥ 0
rgb_d ref_pbr_neutral(const rgb_d& c)
{
    constexpr double f90 = 0.04;
    constexpr double ks = 0.8 - f90;
    constexpr double kd = 0.15;
    const double x = std::min({c[0], c[1], c[2]});
    const double f = x <= 2 * f90 ? x - x * x / (4 * f90) : f90;
    const rgb_d shifted{c[0] - f, c[1] - f, c[2] - f};
    const double p = std::max({shifted[0], shifted[1], shifted[2]});
    if (p <= ks)
    {
        return shifted;
    }
    const double pn = 1 - (1 - ks) * (1 - ks) / (p + 1 - 2 * ks);
    const double g = 1 / (kd * (p - pn) + 1);
    rgb_d out{};
    for (int i = 0; i < 3; ++i)
    {
        out[i] = shifted[i] * (pn / p) * g + pn * (1 - g);
    }
    return out;
}

// encode(c, op) of §2.3: display-encoded values in [0, 1] for op ≠ books
rgb_d ref_encode(const rgb_d& linear, tone_map op, const agx_matrices& m)
{
    const rgb_d s{ref_sanitize(linear[0]), ref_sanitize(linear[1]), ref_sanitize(linear[2])};
    switch (op)
    {
    case tone_map::none:
        return {ref_srgb(ref_clamp01(s[0])), ref_srgb(ref_clamp01(s[1])), ref_srgb(ref_clamp01(s[2]))};
    case tone_map::agx:
        return ref_agx(s, m);
    case tone_map::neutral: {
        const rgb_d n = ref_pbr_neutral(s);
        return {ref_srgb(ref_clamp01(n[0])), ref_srgb(ref_clamp01(n[1])), ref_srgb(ref_clamp01(n[2]))};
    }
    default:
        return {std::nan(""), std::nan(""), std::nan("")}; // books never goes through encode
    }
}

// quantize(v, d) of §3.4: 0 for v not above 0, 255 from 1 on (never dithered), else ⌊255v + 0.5 + d⌋ clamped
int ref_quantize(double v, double d)
{
    if (!(v > 0))
    {
        return 0;
    }
    if (v >= 1)
    {
        return 255;
    }
    return static_cast<int>(std::clamp(std::floor(255 * v + 0.5 + d), 0.0, 255.0));
}

// Interleaved gradient noise (Jimenez 2014), §3.4, in double
double ref_fract(double v) { return v - std::floor(v); }
double ref_ign_inner(int x, int y) { return 0.06711056 * x + 0.00583715 * y; }
double ref_ign(int x, int y) { return ref_fract(52.9829189 * ref_fract(ref_ign_inner(x, y))); }

// --- The device's side: one functor per function under test ----------------------------------------------------------

struct encode_each
{
    const float* in;
    float* out;
    tone_map op;
    __device__ void operator()(int k) const
    {
        const gpu::tone::rgb c{in[3 * k], in[3 * k + 1], in[3 * k + 2]};
        const gpu::tone::rgb e = gpu::tone::encode(c, op);
        out[3 * k] = e.r;
        out[3 * k + 1] = e.g;
        out[3 * k + 2] = e.b;
    }
};

struct neutral_each
{
    const float* in;
    float* out;
    __device__ void operator()(int k) const
    {
        const gpu::tone::rgb c{in[3 * k], in[3 * k + 1], in[3 * k + 2]};
        const gpu::tone::rgb e = gpu::tone::pbr_neutral(c);
        out[3 * k] = e.r;
        out[3 * k + 1] = e.g;
        out[3 * k + 2] = e.b;
    }
};

struct curve_each
{
    const float* in;
    float* out;
    __device__ void operator()(int k) const { out[k] = gpu::tone::agx_curve(in[k]); }
};

// The curve at x = k·stride/2^20 (exact in float: k·stride ≤ 2^20)
struct curve_grid
{
    int stride;
    float* out;
    __device__ void operator()(int k) const
    {
        out[k] = gpu::tone::agx_curve(static_cast<float>(k * stride) * (1.0f / 1048576.0f));
    }
};

struct srgb_each
{
    const float* in;
    float* out;
    __device__ void operator()(int k) const { out[k] = gpu::tone::srgb_encode(in[k]); }
};

struct quantize_each
{
    const float* v;
    const float* d;
    std::uint8_t* out;
    __device__ void operator()(int k) const { out[k] = gpu::tone::quantize(v[k], d[k]); }
};

// IGN of every pixel of a width × (k / width) grid: column i, row j from the top
struct ign_grid
{
    int width;
    float* out;
    __device__ void operator()(int k) const { out[k] = gpu::tone::ign(k % width, k / width); }
};

// 2^20 points of [0, 4096)²: x = 4a + (b mod 4), y = 4b + (a mod 4) for a, b in [0, 1024), so every residue mod 4
// appears in both coordinates and the last column and row are reached; counts values outside [0, 1)
struct ign_outside
{
    unsigned long long* found;
    __device__ void operator()(int k) const
    {
        const int a = k % 1024;
        const int b = k / 1024;
        const float v = gpu::tone::ign(4 * a + (b & 3), 4 * b + (a & 3));
        if (!(v >= 0.0f && v < 1.0f))
        {
            atomicAdd(found, 1ull);
        }
    }
};

// One row j of a side × side image of the constant v, dithered by IGN: the sum of its bytes, and how many differ from
// `expected` (for v = 0 and 1, which are never dithered)
struct dither_row
{
    float v;
    int side;
    std::uint8_t expected;
    unsigned long long* sum;
    unsigned long long* off;
    __device__ void operator()(int j) const
    {
        unsigned long long s = 0;
        unsigned long long o = 0;
        for (int i = 0; i < side; ++i)
        {
            const std::uint8_t q = gpu::tone::quantize(v, gpu::tone::ign(i, j) - 0.5f);
            s += q;
            o += q == expected ? 0 : 1;
        }
        atomicAdd(sum, s);
        atomicAdd(off, o);
    }
};

// A grey v through encode and quantize(·, 0): three bytes
struct grey_bytes
{
    const float* v;
    tone_map op;
    std::uint8_t* out;
    __device__ void operator()(int k) const
    {
        const float g = v[k];
        const gpu::tone::rgb e = gpu::tone::encode(gpu::tone::rgb{g, g, g}, op);
        out[3 * k] = gpu::tone::quantize(e.r, 0.0f);
        out[3 * k + 1] = gpu::tone::quantize(e.g, 0.0f);
        out[3 * k + 2] = gpu::tone::quantize(e.b, 0.0f);
    }
};

// --- T1's inputs (A.md §5): greys, random colours, saturated hues, special values ------------------------------------

struct colour_list
{
    std::vector<float> values; // 3 per colour

    void add(float r, float g, float b) { values.insert(values.end(), {r, g, b}); }
    void append(const colour_list& other) { values.insert(values.end(), other.values.begin(), other.values.end()); }
    int size() const { return static_cast<int>(values.size() / 3); }
    rgb_d colour(int k) const
    {
        const std::size_t i = 3 * static_cast<std::size_t>(k);
        return {values[i], values[i + 1], values[i + 2]};
    }
};

// The host's seeded numbers: std::mt19937_64, whose sequence the standard fixes, and the top 53 bits of an output
class host_rng
{
public:
    explicit host_rng(std::uint64_t seed) : engine_(seed) {}
    double uniform() { return static_cast<double>(engine_() >> 11) / 9007199254740992.0; } // [0, 1)

private:
    std::mt19937_64 engine_;
};

// x moved by k floats (k < 0: down)
float ulps_away(float x, int k)
{
    const float toward = k > 0 ? float_infinity : -float_infinity;
    for (int i = 0; i < std::abs(k); ++i)
    {
        x = std::nextafter(x, toward);
    }
    return x;
}

// G: n greys 2^(−18 + 30k/(n − 1))
colour_list grey_set(int n)
{
    colour_list out;
    for (int k = 0; k < n; ++k)
    {
        const auto v = static_cast<float>(std::exp2(-18 + 30.0 * k / (n - 1)));
        out.add(v, v, v);
    }
    return out;
}

// C: each channel 2^U, U uniform in [−18, 12], replaced by 0 with probability 1/3
colour_list colour_set(int n, std::uint64_t seed)
{
    host_rng rng(seed);
    colour_list out;
    for (int k = 0; k < n; ++k)
    {
        float c[3];
        for (float& channel : c)
        {
            const double u = rng.uniform();
            const bool zero = rng.uniform() < 1.0 / 3;
            channel = zero ? 0.0f : static_cast<float>(std::exp2(-18 + 30 * u));
        }
        out.add(c[0], c[1], c[2]);
    }
    return out;
}

// P: 7 saturated hues × per_hue values s = 2^(−18 + 30k/(per_hue − 1))
colour_list saturated_set(int per_hue)
{
    colour_list out;
    for (int hue = 0; hue < 7; ++hue)
    {
        for (int k = 0; k < per_hue; ++k)
        {
            const auto s = static_cast<float>(std::exp2(-18 + 30.0 * k / (per_hue - 1)));
            const float h = s / 2;
            switch (hue)
            {
            case 0: out.add(s, 0, 0); break;
            case 1: out.add(0, s, 0); break;
            case 2: out.add(0, 0, s); break;
            case 3: out.add(s, s, 0); break;
            case 4: out.add(s, 0, s); break;
            case 5: out.add(0, s, s); break;
            default: out.add(s, h, 0); break;
            }
        }
    }
    return out;
}

// X: 664 special inputs
colour_list special_set()
{
    const float specials[] = {0.0f,
                              -0.0f,
                              -1e-30f,
                              -1.0f,
                              -float_infinity,
                              float_infinity,
                              std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::denorm_min(), // 1e-45 as a float
                              FLT_MAX,
                              1.0f / 262144, // 2^-18
                              4096.0f,       // 2^12
                              0.0031308f,
                              0.08f,
                              0.76f,
                              0.8f,
                              1.0f};
    colour_list out;
    for (const float x : specials)
    {
        out.add(x, 0.5f, 0.25f);
        out.add(0.5f, x, 0.25f);
        out.add(0.5f, 0.25f, x);
        out.add(x, x, x);
    }
    for (int k = 1; k <= 100; ++k)
    {
        for (const int sign : {-1, 1})
        {
            const float grey = ulps_away(0.0031308f, sign * k);
            out.add(grey, grey, grey);
            out.add(ulps_away(0.08f, sign * k), 0.5f, 0.3f);
            out.add(ulps_away(0.8f, sign * k), 0.3f, 0.1f);
        }
    }
    return out;
}

colour_list t1_inputs(bool quick)
{
    colour_list all = grey_set(quick ? 4000 : 40000);
    all.append(colour_set(quick ? 4335 : 49340, 7401));
    all.append(saturated_set(quick ? 143 : 1428));
    all.append(special_set());
    return all;
}

// T13's image: the specials first (at most half the pixels), then evenly spaced greys, colours and hues of T1's sets
colour_list image_colours(int pixels)
{
    const colour_list specials = special_set();
    colour_list rest = grey_set(40000);
    rest.append(colour_set(49340, 7401));
    rest.append(saturated_set(1428));
    colour_list out;
    // The k-th of `count` evenly spaced colours of a list
    const auto take = [&out](const colour_list& from, int k, int count) {
        const int i = static_cast<int>(static_cast<std::int64_t>(k) * from.size() / count);
        const std::size_t at = 3 * static_cast<std::size_t>(i);
        out.add(from.values[at], from.values[at + 1], from.values[at + 2]);
    };
    const int from_specials = std::min(specials.size(), pixels / 2);
    for (int k = 0; k < from_specials; ++k)
    {
        take(specials, k, from_specials);
    }
    const int from_rest = pixels - from_specials;
    for (int k = 0; k < from_rest; ++k)
    {
        take(rest, k, from_rest);
    }
    return out;
}

// --- Comparisons -----------------------------------------------------------------------------------------------------

struct comparison
{
    std::int64_t violations = 0; // |device − reference| above the tolerance, or a device value that isn't finite
    std::int64_t non_finite = 0;
    double max_delta = 0;        // over the finite device values
};

// Every channel of every colour: the device's float against the reference's double, on the same float input
template <class Reference>
comparison compare_colours(const colour_list& inputs, const std::vector<float>& device, double tolerance,
                           const Reference& reference)
{
    comparison c;
    for (int k = 0; k < inputs.size(); ++k)
    {
        const rgb_d expected = reference(inputs.colour(k));
        for (int channel = 0; channel < 3; ++channel)
        {
            const float v = device[3 * static_cast<std::size_t>(k) + channel];
            if (!std::isfinite(v))
            {
                ++c.non_finite;
                ++c.violations;
                continue;
            }
            const double delta = std::abs(v - expected[channel]);
            note_worst(c.max_delta, delta);
            c.violations += delta <= tolerance ? 0 : 1;
        }
    }
    return c;
}

// Values that aren't finite and in [0, 1]
std::int64_t outside_unit_interval(const std::vector<float>& values)
{
    std::int64_t n = 0;
    for (const float v : values)
    {
        n += std::isfinite(v) && v >= 0.0f && v <= 1.0f ? 0 : 1;
    }
    return n;
}

struct byte_check
{
    std::int64_t wrong = 0;      // off by more than allowed
    std::int64_t off_by_one = 0; // off by one inside the zone where that is allowed
    std::int64_t zone = 0;       // reference bytes whose 255e + 0.5 + d lies within 10^-3 of an integer
    std::int64_t alpha = 0;      // alpha bytes other than 255
    std::int64_t channels = 0;   // colour bytes compared
};

// A display's bytes (4 per pixel) against ref_quantize(ref_encode(v), IGN − 0.5): `linear` holds the kernel's float
// input after the exposure (3 per pixel), `ign` the device's IGN of each pixel. The kernel's float e is within
// 2·10^-6 of the reference's double (T1), so where 255e + 0.5 + d lies within 10^-3 of an integer the two may round to
// either side of it, and a byte there may be off by one (A.md §8.6); everywhere else they must be equal.
void check_bytes(std::span<const std::uint8_t> rgba, const std::vector<float>& linear, const std::vector<float>& ign,
                 tone_map op, byte_check& check)
{
    const std::size_t pixels = ign.size();
    if (rgba.size() != 4 * pixels || linear.size() != 3 * pixels)
    {
        check.wrong += static_cast<std::int64_t>(3 * pixels);
        check.channels += static_cast<std::int64_t>(3 * pixels);
        return;
    }
    for (std::size_t p = 0; p < pixels; ++p)
    {
        const float d = ign[p] - 0.5f; // one dither for the three channels, in float as the kernel's
        const rgb_d e = ref_encode({linear[3 * p], linear[3 * p + 1], linear[3 * p + 2]}, op, spec_matrices);
        for (std::size_t c = 0; c < 3; ++c)
        {
            const int expected = ref_quantize(e[c], d);
            const bool zone = near_integer(255 * e[c] + 0.5 + d, 1e-3);
            const int off = std::abs(static_cast<int>(rgba[4 * p + c]) - expected);
            check.zone += zone ? 1 : 0;
            if (off == 1 && zone)
            {
                ++check.off_by_one;
            }
            else if (off != 0)
            {
                ++check.wrong;
            }
        }
        check.alpha += rgba[4 * p + 3] == 255 ? 0 : 1;
        check.channels += 3;
    }
}

std::int64_t byte_differences(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b)
{
    if (a.size() != b.size())
    {
        return static_cast<std::int64_t>(std::max(a.size(), b.size()));
    }
    std::int64_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        n += a[i] == b[i] ? 0 : 1;
    }
    return n;
}

// --- T1, T2, T17: encode against the reference -----------------------------------------------------------------------

GPU_TEST(tone, encode)
{
    const colour_list inputs = t1_inputs(ctx.quick());
    const int n = inputs.size();
    const std::int64_t channels = 3 * static_cast<std::int64_t>(n);
    const gpu_test::device_buffer<float> in(inputs.values);
    const gpu_test::device_buffer<float> out(inputs.values.size());
    for (const tone_map op : toned_ops)
    {
        run_each(encode_each{in.get(), out.get(), op}, n, "encode");
        const std::vector<float> e = out.download();
        const std::string name = name_of(op);
        const comparison c = compare_colours(
            inputs, e, 2e-6, [op](const rgb_d& v) { return ref_encode(v, op, spec_matrices); });
        ctx.count(name + "-reference", c.violations, 0, channels);
        ctx.info(name + "-reference", "max |difference| " + number(c.max_delta) + " (<= 2e-06), " +
                                          std::to_string(c.non_finite) + " values not finite");
        ctx.count(name + "-range", outside_unit_interval(e), 0, channels);
        if (op == tone_map::agx)
        {
            // T17: the same comparison against a reference with the matrices transposed must see the fault
            const agx_matrices wrong = transposed(spec_matrices);
            const comparison planted = compare_colours(
                inputs, e, 2e-6, [&wrong](const rgb_d& v) { return ref_encode(v, tone_map::agx, wrong); });
            const bool fault_accepted = !(planted.max_delta > 1e-3);
            ctx.count("planted-transposed", fault_accepted ? 1 : 0, 0, 1);
            ctx.info("planted-transposed", "max |difference| " + number(planted.max_delta) + " (must exceed 0.001), " +
                                               std::to_string(planted.violations) + " channels beyond 2e-06");
        }
    }
}

// --- T3, T4, T5: AgX's curve and matrices ----------------------------------------------------------------------------

GPU_TEST(tone, agx)
{
    // T3: the anchors f(0) = 0, f(10/16.5) = 0.5 (exactly: d = 0 there), f(1) = 1
    const std::vector<float> anchors{0.0f, static_cast<float>(10 / 16.5), 1.0f};
    const gpu_test::device_buffer<float> anchor_in(anchors);
    const gpu_test::device_buffer<float> anchor_out(anchors.size());
    run_each(curve_each{anchor_in.get(), anchor_out.get()}, static_cast<int>(anchors.size()), "agx_curve");
    const std::vector<float> f = anchor_out.download();
    ctx.exact("agx-anchors-f(0)", f[0], 0, 1e-6);
    ctx.exact("agx-anchors-f(10/16.5)", f[1], 0.5, 0);
    ctx.exact("agx-anchors-f(1)", f[2], 1, 1e-6);

    // T4: monotonic to 2^-23 (one float step near 1, two below) between neighbours x_k = k/2^20
    const int stride = ctx.quick() ? 128 : 1;
    const int points = (1 << 20) / stride + 1;
    const gpu_test::device_buffer<float> curve(static_cast<std::size_t>(points));
    run_each(curve_grid{stride, curve.get()}, points, "agx_curve on a grid");
    const std::vector<float> y = curve.download();
    std::int64_t falls = 0;
    for (std::size_t k = 0; k + 1 < y.size(); ++k)
    {
        falls += std::isfinite(y[k]) && std::isfinite(y[k + 1]) && y[k + 1] >= y[k] - 0x1p-23f ? 0 : 1;
    }
    ctx.count("agx-monotonic", falls, 0, points - 1);

    // T5: the outset is the inset's inverse, in the spec's decimals
    double worst = 0;
    for (int i = 0; i < 3; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            double product = 0;
            for (int j = 0; j < 3; ++j)
            {
                product += spec_matrices.inset[i][j] * spec_matrices.outset[j][k];
            }
            note_worst(worst, std::abs(product - (i == k ? 1.0 : 0.0)));
        }
    }
    ctx.exact("agx-matrices", worst, 0, 1e-12);
}

// --- T6, T7, T8: PBR Neutral's identity region, its continuity, its hue ----------------------------------------------

// The varying value a at channel `at`, then b and c in the next two channels (cyclically)
void add_rotated(colour_list& list, int at, float a, float b, float c)
{
    float v[3];
    v[at] = a;
    v[(at + 1) % 3] = b;
    v[(at + 2) % 3] = c;
    list.add(v[0], v[1], v[2]);
}

std::vector<float> device_neutral(const colour_list& inputs)
{
    const gpu_test::device_buffer<float> in(inputs.values);
    const gpu_test::device_buffer<float> out(inputs.values.size());
    run_each(neutral_each{in.get(), out.get()}, inputs.size(), "pbr_neutral");
    return out.download();
}

// T8's colours: three uniform channels scaled so that the largest is 2^U, U uniform in [−4, 8], with
// max − min ≥ 0.01·max (on the float values)
colour_list hue_colours(int n, std::uint64_t seed)
{
    host_rng rng(seed);
    colour_list out;
    while (out.size() < n)
    {
        const double raw[3] = {rng.uniform(), rng.uniform(), rng.uniform()};
        const double brightness = std::exp2(-4 + 12 * rng.uniform());
        const double top = std::max({raw[0], raw[1], raw[2]});
        if (!(top > 0))
        {
            continue;
        }
        float c[3];
        for (int i = 0; i < 3; ++i)
        {
            c[i] = static_cast<float>(brightness * raw[i] / top);
        }
        const double hi = std::max({c[0], c[1], c[2]});
        const double lo = std::min({c[0], c[1], c[2]});
        if (hi - lo >= 0.01 * hi)
        {
            out.add(c[0], c[1], c[2]);
        }
    }
    return out;
}

// (v − min v)/|v − min v|, and |v − min v|
rgb_d chroma_direction(const rgb_d& v, double& length)
{
    const double lo = std::min({v[0], v[1], v[2]});
    const rgb_d d{v[0] - lo, v[1] - lo, v[2] - lo};
    length = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    return {d[0] / length, d[1] / length, d[2] / length};
}

GPU_TEST(tone, neutral)
{
    // T6: in [0.08, 0.8]³ the operator only takes off the 0.04 of Fresnel's F90
    {
        host_rng rng(7406);
        colour_list inputs;
        for (int k = 0; k < 10000; ++k)
        {
            float c[3];
            for (float& channel : c)
            {
                channel = static_cast<float>(0.08 + 0.72 * rng.uniform());
            }
            inputs.add(c[0], c[1], c[2]);
        }
        const std::vector<float> out = device_neutral(inputs);
        double worst = 0;
        for (int k = 0; k < inputs.size(); ++k)
        {
            const rgb_d c = inputs.colour(k);
            for (int i = 0; i < 3; ++i)
            {
                note_worst(worst, std::abs(out[3 * static_cast<std::size_t>(k) + i] - (c[i] - 0.04)));
            }
        }
        ctx.exact("neutral-identity", worst, 0, 1e-6);
    }

    // T7: pairs straddling the two seams, the min channel at x = 0.08 and the peak at p = 0.76
    {
        const auto x_low = static_cast<float>(0.08 * (1 - 1e-6));
        const auto x_high = static_cast<float>(0.08 * (1 + 1e-6));
        const auto p_low = static_cast<float>(0.8 * (1 - 1e-6));
        const auto p_high = static_cast<float>(0.8 * (1 + 1e-6));
        colour_list pairs;
        for (int at = 0; at < 3; ++at)
        {
            add_rotated(pairs, at, x_low, 0.3f, 0.5f);
            add_rotated(pairs, at, x_high, 0.3f, 0.5f);
        }
        for (int at = 0; at < 3; ++at)
        {
            add_rotated(pairs, at, p_low, 0.3f, 0.1f);
            add_rotated(pairs, at, p_high, 0.3f, 0.1f);
        }
        const std::vector<float> out = device_neutral(pairs);
        double worst_x = 0;
        double worst_p = 0;
        for (std::size_t pair = 0; pair < 6; ++pair)
        {
            for (std::size_t i = 0; i < 3; ++i)
            {
                const double delta = std::abs(static_cast<double>(out[6 * pair + i]) - out[6 * pair + 3 + i]);
                note_worst(pair < 3 ? worst_x : worst_p, delta);
            }
        }
        ctx.exact("neutral-continuity-x", worst_x, 0, 1e-5);
        ctx.exact("neutral-continuity-p", worst_p, 0, 1e-5);
    }

    // T8: out − min(out) points the way c − min(c) does
    {
        const int n = 10000;
        const colour_list inputs = hue_colours(n, 7408);
        const std::vector<float> out = device_neutral(inputs);
        double worst = 0;
        std::int64_t excused = 0;
        for (int k = 0; k < n; ++k)
        {
            const rgb_d c = inputs.colour(k);
            const rgb_d o{out[3 * static_cast<std::size_t>(k)], out[3 * static_cast<std::size_t>(k) + 1],
                          out[3 * static_cast<std::size_t>(k) + 2]};
            if (!(std::isfinite(o[0]) && std::isfinite(o[1]) && std::isfinite(o[2])))
            {
                worst = infinity;
                continue;
            }
            const rgb_d r = ref_pbr_neutral(c);
            double reference_chroma = 0;
            chroma_direction(r, reference_chroma);
            if (reference_chroma < 0.01 * std::max({r[0], r[1], r[2]}))
            {
                ++excused; // float rounding of the output alone could turn u by 3·10^-5 here (decision 1)
                continue;
            }
            double unused = 0;
            const rgb_d w = chroma_direction(c, unused);
            const rgb_d u = chroma_direction(o, unused);
            const rgb_d d{u[0] - w[0], u[1] - w[1], u[2] - w[2]};
            note_worst(worst, std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
        }
        ctx.exact("neutral-hue", worst, 0, 1e-4);
        ctx.count("neutral-hue-excused", excused, n / 10, n);
    }
}

// --- T9: the sRGB curve ----------------------------------------------------------------------------------------------

GPU_TEST(tone, srgb)
{
    const std::vector<float> inputs{0.18f, 0.0f, 1.0f, static_cast<float>(0.0031308 * (1 + 1e-6)),
                                    static_cast<float>(0.0031308 * (1 - 1e-6))};
    const gpu_test::device_buffer<float> in(inputs);
    const gpu_test::device_buffer<float> out(inputs.size());
    run_each(srgb_each{in.get(), out.get()}, static_cast<int>(inputs.size()), "srgb_encode");
    const std::vector<float> s = out.download();
    // 1.055·0.18^(1/2.4) − 0.055 = 0.4613561 (A.md §4 said 0.461374: corrected), within the float's rounding
    ctx.exact("srgb-0.18", s[0], 1.055 * std::pow(0.18, 1 / 2.4) - 0.055, 1e-6);
    ctx.exact("srgb-0", s[1], 0, 0);
    // Exactly 1: a clipped white is 255, never dithered (in float 1.055 − 0.055 is 1 − 2^-24)
    ctx.exact("srgb-1", s[2], 1, 0);
    // The two branches meet at 0.0031308 (0.0404500 from both): no jump between its neighbours
    ctx.exact("srgb-branches-meet", std::abs(static_cast<double>(s[3]) - s[4]), 0, 2e-6);
}

// --- T10: quantize ---------------------------------------------------------------------------------------------------

GPU_TEST(tone, quantize)
{
    std::vector<float> v;
    std::vector<float> d;
    std::vector<int> expected;
    const auto add = [&](float value, float dither, int byte) {
        v.push_back(value);
        d.push_back(dither);
        expected.push_back(byte);
    };
    // (a) k/255 and its two float neighbours round to k without dither
    for (int k = 1; k <= 254; ++k)
    {
        const auto x = static_cast<float>(k / 255.0);
        add(x, 0, k);
        add(std::nextafter(x, -float_infinity), 0, k);
        add(std::nextafter(x, float_infinity), 0, k);
    }
    const std::size_t exact_end = v.size();
    // (b) the ends are never dithered: nothing above 0 is 0, from 1 on is 255
    const float dithers[] = {-0.5f, -0.25f, 0.0f, 0.25f, 0.4999999f};
    const float zeros[] = {0.0f, -0.0f, -1.0f, -float_infinity, std::numeric_limits<float>::quiet_NaN()};
    const float fulls[] = {1.0f, 1.5f, float_infinity};
    for (const float dither : dithers)
    {
        for (const float x : zeros)
        {
            add(x, dither, 0);
        }
        for (const float x : fulls)
        {
            add(x, dither, 255);
        }
    }
    const std::size_t ends_end = v.size();
    // (c) 10^5 seeded pairs v in (0, 1), d in [−0.5, 0.5)
    host_rng rng(7410);
    for (int k = 0; k < 100000; ++k)
    {
        float x = 0;
        while (!(x > 0.0f && x < 1.0f))
        {
            x = static_cast<float>(rng.uniform());
        }
        float dither = 0.5f;
        while (!(dither >= -0.5f && dither < 0.5f))
        {
            dither = static_cast<float>(rng.uniform() - 0.5);
        }
        add(x, dither, -1);
    }
    const gpu_test::device_buffer<float> device_v(v);
    const gpu_test::device_buffer<float> device_d(d);
    const gpu_test::device_buffer<std::uint8_t> out(v.size());
    run_each(quantize_each{device_v.get(), device_d.get(), out.get()}, static_cast<int>(v.size()), "quantize");
    const std::vector<std::uint8_t> q = out.download();

    std::int64_t wrong_exact = 0;
    for (std::size_t i = 0; i < exact_end; ++i)
    {
        wrong_exact += q[i] == expected[i] ? 0 : 1;
    }
    std::int64_t wrong_ends = 0;
    for (std::size_t i = exact_end; i < ends_end; ++i)
    {
        wrong_ends += q[i] == expected[i] ? 0 : 1;
    }
    std::int64_t wrong_random = 0;
    std::int64_t boundary = 0;
    for (std::size_t i = ends_end; i < v.size(); ++i)
    {
        const double t = 255.0 * v[i] + 0.5 + d[i];
        const int reference = static_cast<int>(std::clamp(std::floor(t), 0.0, 255.0));
        const int off = std::abs(static_cast<int>(q[i]) - reference);
        if (near_integer(t, 1e-4))
        {
            ++boundary; // either neighbour: the device's float sum (fused or not) may land on either side
            wrong_random += off <= 1 ? 0 : 1;
        }
        else
        {
            wrong_random += off == 0 ? 0 : 1;
        }
    }
    ctx.count("quantize-k/255", wrong_exact, 0, static_cast<std::int64_t>(exact_end));
    ctx.count("quantize-ends", wrong_ends, 0, static_cast<std::int64_t>(ends_end - exact_end));
    ctx.count("quantize-random", wrong_random, 0, static_cast<std::int64_t>(v.size() - ends_end));
    ctx.info("quantize-random",
             std::to_string(boundary) + " pairs within 1e-4 of a rounding boundary (either neighbour allowed)");
}

// --- T11: interleaved gradient noise ---------------------------------------------------------------------------------

GPU_TEST(tone, ign)
{
    const gpu_test::device_buffer<unsigned long long> outside(1);
    run_each(ign_outside{outside.get()}, 1 << 20, "ign over [0, 4096)^2");
    ctx.count("ign-range", static_cast<std::int64_t>(outside.download()[0]), 0, 1 << 20);

    const int side = 64;
    const gpu_test::device_buffer<float> values(side * side);
    run_each(ign_grid{side, values.get()}, side * side, "ign");
    const std::vector<float> g = values.download();
    std::int64_t wrong = 0;
    std::int64_t excused = 0;
    double worst = 0;
    for (int y = 0; y < side; ++y)
    {
        for (int x = 0; x < side; ++x)
        {
            const float v = g[static_cast<std::size_t>(y) * side + x];
            if (!(std::isfinite(v) && v >= 0.0f && v < 1.0f))
            {
                ++wrong;
                continue;
            }
            const double delta = std::abs(v - ref_ign(x, y));
            const double distance = std::min(delta, 1 - delta); // across the wrap 0 ↔ 1
            if (distance <= 1e-4)
            {
                note_worst(worst, distance);
            }
            else if (near_integer(ref_ign_inner(x, y), 1e-5))
            {
                ++excused; // the inner fract's wrap (decision 2)
            }
            else
            {
                ++wrong;
            }
        }
    }
    ctx.count("ign-formula", wrong, 0, side * side);
    ctx.info("ign-formula", "max distance " + number(worst) + " (<= 1e-4), " + std::to_string(excused) +
                                " points excused at the inner wrap");
}

// --- T12: the dither's mean ------------------------------------------------------------------------------------------

GPU_TEST(tone, dither_mean)
{
    const int side = ctx.quick() ? 64 : 1024;
    const std::int64_t pixels = static_cast<std::int64_t>(side) * side;
    struct level
    {
        const char* name;
        float v;
    };
    const level levels[] = {{"0.2", 0.2f},
                            {"0.5", 0.5f},
                            {"0.731", 0.731f},
                            {"100.3/255", static_cast<float>(100.3 / 255)},
                            {"254.6/255", static_cast<float>(254.6 / 255)},
                            {"0", 0.0f},
                            {"1", 1.0f}};
    for (const level& l : levels)
    {
        const gpu_test::device_buffer<unsigned long long> counts(2);
        const bool never_dithered = l.v == 0.0f || l.v == 1.0f;
        const auto expected = static_cast<std::uint8_t>(l.v == 1.0f ? 255 : 0);
        run_each(dither_row{l.v, side, expected, counts.get(), counts.get() + 1}, side, "dithered rows");
        const std::vector<unsigned long long> c = counts.download();
        if (never_dithered)
        {
            ctx.count(l.v == 0.0f ? "dither-0-all-0" : "dither-1-all-255", static_cast<std::int64_t>(c[1]), 0, pixels);
            continue;
        }
        const std::string name = std::string("dither-mean-") + l.name;
        const double mean = static_cast<double>(c[0]) / static_cast<double>(pixels);
        const double target = 255.0 * l.v;
        if (ctx.quick())
        {
            ctx.info(name, number(mean) + ", expected " + number(target) +
                               " (no verdict at 64x64 pixels: SKIP (quick), decision 3)");
        }
        else
        {
            ctx.exact(name, mean, target, 0.01);
        }
    }
}

// --- T13, T14: the kernel's bytes ------------------------------------------------------------------------------------

// Bytes of rows [first, last) that no longer hold the sentinel
std::int64_t touched(const std::vector<std::uint8_t>& rgba, int width, int first, int last)
{
    std::int64_t n = 0;
    for (std::size_t i = 4 * static_cast<std::size_t>(first) * width; i < 4 * static_cast<std::size_t>(last) * width;
         ++i)
    {
        n += rgba[i] == sentinel ? 0 : 1;
    }
    return n;
}

GPU_TEST(tone, kernel_bytes)
{
    const int width = ctx.quick() ? 16 : 64;
    const int height = ctx.quick() ? 12 : 48;
    const int split = ctx.quick() ? 5 : 20; // rows [0, split): the pass in progress, one sample more than the rest
    const int pixels = width * height;
    const double scale_top = 1.0 / 17;
    const double scale_rest = 1.0 / 16;

    // The sums of 17 (top band) or 16 samples of each colour: NaN, ±∞ and −0 stay what they are
    const colour_list colours = image_colours(pixels);
    std::vector<double> sums(3 * static_cast<std::size_t>(pixels));
    for (std::size_t i = 0; i < sums.size(); ++i)
    {
        const int row = static_cast<int>(i / 3) / width;
        sums[i] = static_cast<double>(colours.values[i]) * (row < split ? 17 : 16);
    }
    const gpu_test::device_buffer<double> device_sums(sums);

    // The reference's dither: IGN of each pixel from the device's own ign
    const gpu_test::device_buffer<float> ign_values(static_cast<std::size_t>(pixels));
    run_each(ign_grid{width, ign_values.get()}, pixels, "ign of the image");
    const std::vector<float> ign = ign_values.download();

    const gpu_test::device_buffer<std::uint8_t> rgba(4 * static_cast<std::size_t>(pixels));
    const float exposures[] = {1.0f, static_cast<float>(std::exp2(0.5)), static_cast<float>(std::exp2(-3.0))};
    std::int64_t touched_bytes = 0;
    std::int64_t run_differences = 0;
    std::int64_t alpha = 0;
    std::int64_t zone = 0;
    for (const tone_map op : toned_ops)
    {
        byte_check check;
        for (const float exposure : exposures)
        {
            // The kernel's input, as A.md §2.2 computes it: the product in double rounded to float, then one float
            // multiply by the exposure
            std::vector<float> linear(sums.size());
            for (std::size_t i = 0; i < sums.size(); ++i)
            {
                const int row = static_cast<int>(i / 3) / width;
                linear[i] = static_cast<float>(sums[i] * (row < split ? scale_top : scale_rest)) * exposure;
            }
            const auto band = [&](bool top) {
                if (top)
                {
                    gpu::detail::to_rgba8_toned(device_sums.get(), width, 0, split, scale_top, op, exposure,
                                                rgba.get());
                }
                else
                {
                    gpu::detail::to_rgba8_toned(device_sums.get(), width, split, height - split, scale_rest, op,
                                                exposure, rgba.get());
                }
                gpu_test::finish_launch("to_rgba8_toned");
            };
            const auto fill_sentinel = [&] {
                gpu_test::cuda_check(cudaMemset(rgba.get(), sentinel, rgba.size()), "the sentinel bytes");
            };
            // First run: the top band, then the rest; between them the rest still holds the sentinel
            fill_sentinel();
            band(true);
            touched_bytes += touched(rgba.download(), width, split, height);
            band(false);
            const std::vector<std::uint8_t> first = rgba.download();
            // Second run (T14), the other way round
            fill_sentinel();
            band(false);
            touched_bytes += touched(rgba.download(), width, 0, split);
            band(true);
            const std::vector<std::uint8_t> second = rgba.download();
            run_differences += byte_differences(first, second);
            check_bytes(first, linear, ign, op, check);
        }
        const std::string name = std::string("kernel-bytes-") + name_of(op);
        ctx.count(name, check.wrong, 0, check.channels);
        ctx.count(name + "-off-by-one", check.off_by_one, check.channels / 50, check.channels);
        alpha += check.alpha;
        zone += check.zone;
    }
    const std::int64_t all_pixels = 9 * static_cast<std::int64_t>(pixels);
    ctx.count("kernel-bytes-alpha", alpha, 0, all_pixels);
    ctx.count("kernel-bytes-sentinel", touched_bytes, 0, 4 * all_pixels);
    ctx.info("kernel-bytes", std::to_string(zone) + " reference bytes within 1e-3 of a rounding boundary");
    ctx.count("deterministic", run_differences, 0, 4 * all_pixels);
}

// A uniform image through the kernel, every pixel (the dither's (0, 0), where ign is 0 and the dither −0.5, included):
// a clipped white (any light at least 1) is 255 under none, as is +∞ under neutral (whose white is only approached:
// p_n rounds to 1 for p above about 2·10^6), and black is 0 under all three. AgX's white isn't 255 in every channel
// (its outset's red row adds up to 0.999828), so it is checked only for black.
GPU_TEST(tone, white_and_black)
{
    const int width = ctx.quick() ? 64 : 1024;
    const int height = ctx.quick() ? 16 : 256;
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    const gpu_test::device_buffer<std::uint8_t> rgba(4 * pixels);
    struct uniform
    {
        const char* name;
        double light;
        std::uint8_t byte;
        std::vector<tone_map> ops;
    };
    const uniform cases[] = {{"white", 1.0, 255, {tone_map::none}},
                             {"over-white", 1e6, 255, {tone_map::none}},
                             {"infinite", std::numeric_limits<double>::infinity(), 255,
                              {tone_map::none, tone_map::neutral}},
                             {"black", 0.0, 0, {tone_map::none, tone_map::agx, tone_map::neutral}}};
    for (const uniform& u : cases)
    {
        const gpu_test::device_buffer<double> sums(std::vector<double>(3 * pixels, u.light));
        for (const tone_map op : u.ops)
        {
            gpu::detail::to_rgba8_toned(sums.get(), width, 0, height, 1.0, op, 1.0f, rgba.get());
            const std::vector<std::uint8_t> out = rgba.download();
            std::int64_t wrong = 0;
            for (std::size_t p = 0; p < pixels; ++p)
            {
                for (std::size_t c = 0; c < 3; ++c)
                {
                    wrong += out[4 * p + c] == u.byte ? 0 : 1;
                }
            }
            ctx.count(std::string("uniform-") + u.name + "-" + std::string(name_of(op)), wrong, 0,
                      3 * static_cast<std::int64_t>(pixels));
        }
    }
}

// --- T15: a session's display ----------------------------------------------------------------------------------------

// 16×9 pixels, one matte sphere under a sky brighter than 1 at the top, 2×2 strata (4 passes)
gpu::scene_desc tone_scene()
{
    gpu::scene_desc scene;
    scene.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = {0.6, 0.45, 0.3}});
    scene.materials.push_back({.kind = gpu::material_kind::lambertian, .texture = 0});
    scene.spheres.push_back({.center = {0, 0, -2}, .radius = 0.75, .material = 0});
    scene.camera = {.image_width = 16,
                    .image_height = 9,
                    .sqrt_spp = 2,
                    .max_depth = 8,
                    .center = {0, 0, 0},
                    .pixel00_loc = {-0.9375, 0.5, -1},
                    .pixel_delta_u = {0.125, 0, 0},
                    .pixel_delta_v = {0, -0.125, 0},
                    .sky_bottom = {0.2, 0.3, 0.5},
                    .sky_top = {4, 3, 2},
                    .sampling = gpu::sampling_strategy::material};
    return scene;
}

gpu::display_options display_of(tone_map op, double exposure)
{
    gpu::display_options d;
    d.op = op;
    d.exposure = exposure;
    return d;
}

// 1 when f throws an exception of type E, 0 when it throws another or none
template <class E, class F>
double throws(const F& f)
{
    try
    {
        f();
    }
    catch (const E&)
    {
        return 1;
    }
    catch (...)
    {
        return 0;
    }
    return 0;
}

// 1 when f returns
template <class F>
double returns(const F& f)
{
    try
    {
        f();
    }
    catch (...)
    {
        return 0;
    }
    return 1;
}

GPU_TEST(tone, session)
{
    const gpu::scene_desc scene = tone_scene();
    const int width = 16;
    const int height = 9;
    const std::int64_t bytes = 4 * width * height;
    constexpr std::uint64_t seed = 7415;

    gpu::session_options options;
    options.seed = seed;
    gpu::session s(scene, options);
    ctx.exact("session-display-needs-a-pass",
              throws<std::logic_error>([&] { s.display(display_of(tone_map::agx, 0)); }), 1, 0);
    s.render_to({4, 0});

    // (a) books is display() itself, computed again: a tone map in between overwrites the same memory
    const std::span<const std::uint8_t> shown = s.display();
    const std::vector<std::uint8_t> books_bytes(shown.begin(), shown.end());
    s.display(display_of(tone_map::agx, 0.5));
    const std::span<const std::uint8_t> books_shown = s.display(display_of(tone_map::books, 0));
    ctx.count("session-books-is-display", byte_differences(books_bytes, books_shown), 0, bytes);

    // (b) render()'s rgba: the display's bytes of the same passes with a tone map, none with books
    const std::span<const std::uint8_t> toned_shown = s.display(display_of(tone_map::agx, 0.5));
    const std::vector<std::uint8_t> toned_bytes(toned_shown.begin(), toned_shown.end());
    gpu::render_options settings;
    settings.seed = seed;
    settings.display = display_of(tone_map::agx, 0.5);
    const gpu::render_result toned = gpu::render(scene, settings);
    settings.display = display_of(tone_map::books, 0);
    const gpu::render_result plain = gpu::render(scene, settings);
    ctx.exact("session-render-rgba-size", static_cast<double>(toned.rgba.size()), static_cast<double>(bytes), 0);
    ctx.count("session-render-rgba-is-display", byte_differences(toned.rgba, toned_bytes), 0, bytes);
    ctx.exact("session-render-books-rgba-empty", static_cast<double>(plain.rgba.size()), 0, 0);

    // (c) the display changes nothing of the rendering: a session shown with a tone map after 2 passes, then rendered
    // to 4, holds the same sums and counts as s, rendered to 4 with nothing shown in between
    {
        gpu::session shown_midway(scene, options);
        shown_midway.render_to({2, 0});
        shown_midway.display(display_of(tone_map::agx, 0.5));
        shown_midway.render_to({4, 0});
        const std::vector<double> after = shown_midway.pixels();
        const std::vector<double> plain_pixels = s.pixels();
        std::int64_t pixel_differences = 0;
        if (after.size() != plain_pixels.size())
        {
            pixel_differences = static_cast<std::int64_t>(std::max(after.size(), plain_pixels.size()));
        }
        for (std::size_t i = 0; pixel_differences == 0 && i < plain_pixels.size(); ++i)
        {
            // Bit for bit: memcmp of the doubles, so that −0 against 0 or two NaNs would count
            pixel_differences += std::memcmp(&after[i], &plain_pixels[i], sizeof(double)) == 0 ? 0 : 1;
        }
        ctx.count("session-pixels-unchanged", pixel_differences, 0, static_cast<std::int64_t>(plain_pixels.size()));
        ctx.exact("session-rays-unchanged",
                  shown_midway.rays() == s.rays() && shown_midway.shadow_rays() == s.shadow_rays() ? 1 : 0, 1, 0);
    }

    // (c2) a pass in progress: its rows hold one sample more, and are divided by it. A session at 1 pass and 4 rows of
    // the second shows rows 0-3 as one at 2 passes, rows 4-8 as one at 1 pass (the sums are the same additions, the
    // scales 1/2 and 1, the dither the pixel's), byte for byte.
    {
        const auto shown_at = [&](gpu::accumulation target) {
            gpu::session at(scene, options);
            at.render_to(target);
            const std::span<const std::uint8_t> b = at.display(display_of(tone_map::agx, 0.5));
            return std::vector<std::uint8_t>(b.begin(), b.end());
        };
        const std::vector<std::uint8_t> midway = shown_at({1, 4});
        const std::vector<std::uint8_t> two = shown_at({2, 0});
        const std::vector<std::uint8_t> one = shown_at({1, 0});
        const std::size_t split = 4 * static_cast<std::size_t>(width) * 4; // 4 rows of 4-byte pixels
        const auto band = [&](const std::vector<std::uint8_t>& v, std::size_t from, std::size_t to) {
            return std::span<const std::uint8_t>(v).subspan(from, to - from);
        };
        const auto all = static_cast<std::size_t>(bytes);
        ctx.count("session-band-in-progress", byte_differences(band(midway, 0, split), band(two, 0, split)), 0,
                  static_cast<std::int64_t>(split));
        ctx.count("session-band-complete", byte_differences(band(midway, split, all), band(one, split, all)), 0,
                  static_cast<std::int64_t>(all - split));
    }

    // (d) the agx bytes against the reference: the kernel's input is float(pixel) (sums·1/4, exact) times float(√2)
    {
        const auto exposure = static_cast<float>(std::exp2(0.5));
        const std::vector<double> pixels = s.pixels();
        std::vector<float> linear(pixels.size());
        for (std::size_t i = 0; i < linear.size(); ++i)
        {
            linear[i] = static_cast<float>(pixels[i]) * exposure;
        }
        const gpu_test::device_buffer<float> ign_values(static_cast<std::size_t>(width) * height);
        run_each(ign_grid{width, ign_values.get()}, width * height, "ign of the session's image");
        byte_check check;
        check_bytes(toned_bytes, linear, ign_values.download(), tone_map::agx, check);
        ctx.count("session-agx-bytes", check.wrong, 0, check.channels);
        ctx.count("session-agx-bytes-off-by-one", check.off_by_one, check.channels / 50, check.channels);
        ctx.count("session-agx-alpha", check.alpha, 0, width * height);
    }

    // (e) the refusals of A.md §2.1, and the range's ends accepted
    ctx.exact("session-refuses-books-exposure",
              throws<std::invalid_argument>([&] { s.display(display_of(tone_map::books, 1)); }), 1, 0);
    ctx.exact("session-refuses-nan-exposure", throws<std::invalid_argument>([&] {
                  s.display(display_of(tone_map::agx, std::numeric_limits<double>::quiet_NaN()));
              }),
              1, 0);
    ctx.exact("session-refuses-infinite-exposure",
              throws<std::invalid_argument>([&] { s.display(display_of(tone_map::agx, infinity)); }), 1, 0);
    ctx.exact("session-refuses-exposure-25",
              throws<std::invalid_argument>([&] { s.display(display_of(tone_map::agx, 25)); }), 1, 0);
    ctx.exact("session-refuses-exposure--25",
              throws<std::invalid_argument>([&] { s.display(display_of(tone_map::agx, -25)); }), 1, 0);
    ctx.exact("session-takes-exposure-24", returns([&] { s.display(display_of(tone_map::neutral, 24)); }), 1, 0);
    ctx.exact("session-takes-exposure--24", returns([&] { s.display(display_of(tone_map::none, -24)); }), 1, 0);

    // Float64: the tone maps are float only; the books' display works as before
    gpu::session_options options64 = options;
    options64.arithmetic = gpu::precision::float64;
    gpu::session s64(scene, options64);
    s64.render_to({1, 0});
    ctx.exact("session-float64-refuses-agx",
              throws<std::invalid_argument>([&] { s64.display(display_of(tone_map::agx, 0)); }), 1, 0);
    ctx.exact("session-float64-display", returns([&] { s64.display(); }), 1, 0);
    gpu::render_options render64;
    render64.seed = seed;
    render64.arithmetic = gpu::precision::float64;
    render64.display = display_of(tone_map::agx, 0);
    ctx.exact("render-float64-refuses-agx", throws<std::invalid_argument>([&] { gpu::render(scene, render64); }), 1,
              0);
}

// --- T16: the bytes as the programs save them ------------------------------------------------------------------------

GPU_TEST(tone, png_round_trip)
{
    // 16×16 pixels, every byte value once per channel and each pixel's three different, so a wrong stride, row order or
    // channel order shows: byte (x + 16y + 85c) mod 256, alpha 255
    const int width = 16;
    const int height = 16;
    const int pixels = width * height;
    std::vector<std::uint8_t> rgba;
    for (int k = 0; k < pixels; ++k)
    {
        rgba.insert(rgba.end(), {static_cast<std::uint8_t>(k), static_cast<std::uint8_t>((k + 85) % 256),
                                 static_cast<std::uint8_t>((k + 170) % 256), 255});
    }
    // RayTracingGPU's bytes_to_image: b → ((b + 0.5)/256)², (2b + 1)²/2^18, exact in double
    const std::vector<double> linear = gpu_test::host::bytes_to_linear(rgba, width, height);
    std::int64_t wrong_linear = linear.size() == 3 * static_cast<std::size_t>(pixels) ? 0 : 3 * pixels;
    for (std::size_t p = 0; wrong_linear == 0 && p < static_cast<std::size_t>(pixels); ++p)
    {
        for (std::size_t c = 0; c < 3; ++c)
        {
            const double g = (rgba[4 * p + c] + 0.5) / 256;
            wrong_linear += linear[3 * p + c] == g * g ? 0 : 1;
        }
    }
    ctx.count("png-round-trip-linear", wrong_linear, 0, 3 * pixels);
    // bytes_to_image, image::save_png and image::load give every byte back
    const std::vector<std::uint8_t> back = gpu_test::host::png_round_trip(rgba, width, height);
    std::int64_t wrong_bytes = back.size() == 3 * static_cast<std::size_t>(pixels) ? 0 : 3 * pixels;
    for (std::size_t p = 0; wrong_bytes == 0 && p < static_cast<std::size_t>(pixels); ++p)
    {
        for (std::size_t c = 0; c < 3; ++c)
        {
            wrong_bytes += back[3 * p + c] == rgba[4 * p + c] ? 0 : 1;
        }
    }
    ctx.count("png-round-trip-bytes", wrong_bytes, 0, 3 * pixels);
    // A size that isn't the bytes' is refused
    ctx.exact("png-round-trip-refuses-size",
              throws<std::invalid_argument>([&] { gpu_test::host::bytes_to_linear(rgba, width, height + 1); }), 1, 0);
}

// Every scene's own display is one the renderer takes (the programs check only the options they are given): an
// exposure finite and within ±max_exposure, and none with books
GPU_TEST(tone, registry)
{
    const std::vector<gpu_test::host::scene_display> scenes = gpu_test::host::scene_displays();
    std::int64_t wrong = 0;
    std::string toned;
    for (const gpu_test::host::scene_display& s : scenes)
    {
        const gpu::display_options& d = s.display;
        const bool in_range = std::isfinite(d.exposure) && std::abs(d.exposure) <= gpu::max_exposure;
        wrong += in_range && (d.op != tone_map::books || d.exposure == 0) ? 0 : 1;
        if (d.op != tone_map::books)
        {
            toned += (toned.empty() ? "" : ", ") + s.name + " " + std::string(name_of(d.op)) + " " + number(d.exposure);
        }
    }
    ctx.count("registry-displays", wrong, 0, static_cast<std::int64_t>(scenes.size()));
    ctx.info("registry-displays", toned.empty() ? "every scene opens with books" : "tone maps: " + toned);
}

// --- T18: tonemap.md's grey bytes ------------------------------------------------------------------------------------

GPU_TEST(tone, reference_bytes)
{
    struct grey
    {
        float v;
        int byte;
    };
    struct table
    {
        tone_map op;
        int allowed; // none exact; agx and neutral ±1 (the channels round apart, A.md §8.7)
        std::vector<grey> greys;
    };
    const table tables[] = {
        {tone_map::none, 0, {{0.001f, 3}, {0.01f, 25}, {0.05f, 63}, {0.18f, 118}, {0.5f, 188}}},
        {tone_map::agx,
         1,
         {{0.001f, 4},
          {0.01f, 25},
          {0.05f, 72},
          {0.18f, 127},
          {0.5f, 173},
          {1.0f, 201},
          {2.0f, 223},
          {4.0f, 239}, // A.md §4 said 238 and 254: corrected from §3.2's equations (239.1 and 254.8 by hand)
          {16.0f, 255},
          {100.0f, 255}}},
        {tone_map::neutral, 1, {{0.18f, 105}, {1.0f, 240}, {4.0f, 253}}},
    };
    for (const table& t : tables)
    {
        std::vector<float> v;
        for (const grey& g : t.greys)
        {
            v.push_back(g.v);
        }
        const gpu_test::device_buffer<float> in(v);
        const gpu_test::device_buffer<std::uint8_t> out(3 * v.size());
        run_each(grey_bytes{in.get(), t.op, out.get()}, static_cast<int>(v.size()), "grey bytes");
        const std::vector<std::uint8_t> b = out.download();
        std::int64_t wrong = 0;
        std::string seen;
        for (std::size_t k = 0; k < t.greys.size(); ++k)
        {
            seen += (k == 0 ? "" : ", ") + number(t.greys[k].v) + " ->";
            for (std::size_t c = 0; c < 3; ++c)
            {
                wrong += std::abs(static_cast<int>(b[3 * k + c]) - t.greys[k].byte) <= t.allowed ? 0 : 1;
                seen += " " + std::to_string(b[3 * k + c]);
            }
        }
        const std::string name = std::string("reference-bytes-") + name_of(t.op);
        ctx.count(name, wrong, 0, 3 * static_cast<std::int64_t>(t.greys.size()));
        ctx.info(name, seen);
    }
}
}

// --- Step criteria (A.md §7; gates F, SASS, HOST, CARRY as final.md B) -----------------------------------------------
// Step 74 (tone_map_kernel new; quality_kernel identical to Step 73's by --prev):
//   1. $GT tone: exit 0 (T1-T18).
//   2. $G cornell -w 160 -s 16 --seed 3 -o a.png -o a.pfm, and the same with --tonemap books: both files identical
//      (SHA-256). With --tonemap agx --exposure 0.5 -o b.png -o b.pfm: b.pfm == a.pfm, b.png != a.png, a second run
//      identical (SHA-256 of both files).
//   3. cli-cases: --tonemap bogus, --exposure 1 (books), --fp64 --tonemap agx, --exposure nan, --exposure 25: exit 1,
//      one Error: line each; help gains the 2 option lines (allowed diff); the Display: line appears only with a
//      non-books display.
//   4. Sanitizer: cornell --tonemap agx at 32×18, 4 spp, and $GT tone --quick: 0 errors.
// Step 75:
//   1. Window = file: $V cornell --headless -w 160 --height 90 --strata 4 -s 16 --seed 3 --tonemap agx --exposure 0.5
//      --frames F -o v.png -o v.pfm against $G cornell -w 160 --height 90 --strata 4 -s 16 --seed 3 --tonemap agx
//      --exposure 0.5 -o g.png -o g.pfm: $PS png-same v.png g.png; the last frame in F png-same g.png;
//      $PS same v.pfm g.pfm.
//   2. Replay (guarded: no fullscreen application in front): A.md §6.4's session, then $V --headless --script r.txt
//      --frames B: every frame of A equals B (decoded); the redisplayed frames carry unchanged pass counts in r.txt;
//      P's printed command, run, reproduces P's PNG (png-same) and PFM (same).
//   3. Idle: --stats of a done session without input shows no further frames.
//   4. HOST -Tour (the GIF tour's frames and record identical; allowed diff: help lines); CARRY.
