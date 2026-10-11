// ImageCompare: are two renders the same picture, up to Monte Carlo noise?
//
// Every pixel of a render is a Monte Carlo estimate, an average of random samples, so two renders of one scene never
// agree exactly unless they use the same random numbers. What two correct renderers must agree on is the expected value
// of every pixel. The GPU renderer, for example, computes in float instead of double and draws different random
// numbers: its image differs from the CPU's in every pixel, but it must differ by noise alone. That can be tested. A
// and B are independent renders of the same scene (with different seeds, or from different renderers) at the same size,
// with the same rays per pixel and the same sampling method, saved as PFM files, which keep the renderer's linear
// values (a PNG rounds them to 8 bits after gamma correction and clamping). d = B - A is their difference in each pixel
// and channel. If A and B are the same picture, every d has the expected value 0, and an average of many of them must
// be as close to 0 as their noise allows. The program checks that in four ways, each made for a different kind of
// difference:
//   1. the whole image: the mean of d over all pixels, per channel. A small bias that covers the whole image, such as a
//      render 0.1% too dark, adds up over millions of pixels;
//   2. all tiles together: the image is cut into tiles of 32×32 pixels, each tile's mean of d is measured by its own
//      noise, and these scores are added up. The same small bias is found here even where the mean of the whole image
//      misses it, because one very noisy region (glass, say) can drown it there;
//   3. the largest tile: a difference confined to one object is diluted in the whole image, but stands out in the
//      object's tiles;
//   4. the number of tiles with |z| > 3: a weak difference spread over many tiles, perhaps with both signs (an image
//      shifted by a fraction of a pixel is brighter on one side of every edge and darker on the other).
// A sign test checks what all four assume: that the noise of A and B has the same shape (see sign_test).
// The verdict comes last, and the exit code repeats it for scripts: 0 consistent, 1 different, 2 an error.
import std;
import cli;
import image;
import vec3;

constexpr double infinity = std::numeric_limits<double>::infinity();

// --- Statistics ----------------------------------------------------------------------------------------------------

// The mean and the spread of a stream of numbers, in one pass. Welford's algorithm (MonteCarlo's sample_mean mentions
// it) updates the mean with each number, and the sum of squared deviations from it. The shortcut "mean of the squares
// minus the squared mean" subtracts two nearly equal numbers when the spread is tiny next to the mean, and here that
// happens: in a tile of a flat background, every d can be the same tiny number.
class running_stats
{
public:
    void add(double x)
    {
        ++count_;
        const double delta = x - mean_;
        mean_ += delta / static_cast<double>(count_);
        squared_deviations_ += delta * (x - mean_);
        sum_of_squares_ += x * x;
    }

    std::int64_t count() const { return count_; }
    double mean() const { return mean_; }

    // The sample variance s². It divides by N - 1, not N: the deviations are measured from the sample's own mean,
    // which lies closer to the numbers than the true mean does, so N of them would come out too small on average.
    double variance() const { return count_ > 1 ? squared_deviations_ / static_cast<double>(count_ - 1) : 0.0; }

    double rms() const { return count_ > 0 ? std::sqrt(sum_of_squares_ / static_cast<double>(count_)) : 0.0; }

private:
    std::int64_t count_ = 0;
    double mean_ = 0;
    double squared_deviations_ = 0;
    double sum_of_squares_ = 0;
};

// P(|Z| > z) for a standard normal Z: how often a normally distributed number lands more than z standard deviations
// from its mean, on either side. erfc, the complementary error function, is in the standard library.
double normal_tail(double z)
{
    return std::erfc(z / std::numbers::sqrt2);
}

// The inverse: the z with normal_tail(z) = p. normal_tail falls as z grows, so bisection finds it (the same method as
// MonteCarlo's exact_halfway_point): 60 halvings shrink [0, 40] to 40/2⁶⁰ ≈ 3·10⁻¹⁷.
double normal_quantile(double p)
{
    if (!(p > 0))
    {
        return infinity;
    }
    double low = 0;
    double high = 40; // normal_tail(40) ≈ 10⁻³⁵⁰ is below the smallest double: no p can need more
    for (int i = 0; i < 60; ++i)
    {
        const double middle = (low + high) / 2;
        if (normal_tail(middle) > p)
        {
            low = middle;
        }
        else
        {
            high = middle;
        }
    }
    return (low + high) / 2;
}

// The regularized incomplete beta function I_x(a, b) = ∫₀ˣ t^(a-1)·(1-t)^(b-1) dt / B(a, b), for Student's distribution
// below. Computed as in Numerical Recipes (section 6.4): a continued fraction, evaluated by the modified Lentz method,
// converges quickly for x < (a + 1)/(a + b + 2); above that, the symmetry I_x(a, b) = 1 - I_(1-x)(b, a) brings x
// below it.
double incomplete_beta(double a, double b, double x)
{
    if (x <= 0)
    {
        return 0;
    }
    if (x >= 1)
    {
        return 1;
    }
    if (x > (a + 1) / (a + b + 2))
    {
        return 1 - incomplete_beta(b, a, 1 - x);
    }

    // The factor in front of the fraction, x^a·(1-x)^b / B(a, b), in logarithms. The beta function
    // B(a, b) = Γ(a)·Γ(b)/Γ(a + b) is itself an ordinary number, but for a = 10⁶ (a whole image) Γ(a) and Γ(a + b) are
    // far beyond the range of doubles (Γ(172) already is), and x^a underflows to 0 when t is large; their logarithms
    // are ordinary numbers. std::lgamma is the logarithm of Γ, and log1p(-x) is log(1 - x) without losing digits when
    // x is tiny.
    const double log_front =
        a * std::log(x) + b * std::log1p(-x) + std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);

    // The fraction 1/(1 + k₁/(1 + k₂/(1 + k₃/(1 + ...)))) with k₁ = -(a + b)·x/(a + 1) and then, for m = 1, 2, ...,
    // a pair of numbers: m·(b - m)·x / ((a + 2m - 1)·(a + 2m)) and -(a + m)·(a + b + m)·x / ((a + 2m)·(a + 2m + 1)).
    // The modified Lentz method evaluates it from the top down: it keeps two ratios, c and d, and multiplies the result
    // by c·d for each new k, until that factor is 1 to the precision of doubles.
    constexpr double tiny = 1e-300; // stands in for a 0 that would be divided by
    const auto nonzero = [](double v) { return std::fabs(v) < tiny ? tiny : v; };
    double c = 1;
    double d = 1 / nonzero(1 - (a + b) * x / (a + 1));
    double fraction = d;
    // For the t of this program it takes a few dozen pairs, also for a = 10⁶: at most about 70, for t near √3, where
    // x sits at the switch to the symmetric form above and the fraction converges slowest (Numerical Recipes
    // guarantees about √max(a, b) at worst). The limit is only a safety net.
    for (int m = 1; m <= 100'000; ++m)
    {
        const std::array pair{m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                              -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))};
        for (const double k : pair)
        {
            d = 1 / nonzero(1 + k * d);
            c = nonzero(1 + k / c);
            fraction *= c * d;
        }
        if (std::fabs(c * d - 1) < 1e-15)
        {
            break;
        }
    }
    return std::exp(log_front) * fraction / a;
}

// The z that is exactly as rare as t from Student's t distribution with the given degrees of freedom (see test_mean):
// P(|Z| > z) = P(|T| > t), where P(|T| > t) = I_x(df/2, 1/2) with x = df/(df + t²). Beyond z ≈ 37 the probability is
// below the smallest double, and the leading term of Wallace's approximation (1959) takes over,
// z ≈ √(df·ln(1 + t²/df)), which tends to t as df grows. Out there a z only says "far beyond any threshold" anyway.
double normal_equivalent(double t, double degrees_of_freedom)
{
    const double p = incomplete_beta(degrees_of_freedom / 2, 0.5, degrees_of_freedom / (degrees_of_freedom + t * t));
    if (p > 0 || std::isinf(t))
    {
        return std::copysign(normal_quantile(p), t);
    }
    return std::copysign(std::sqrt(degrees_of_freedom * std::log1p(t * t / degrees_of_freedom)), t);
}

// P(K ≥ k) for a K with the binomial distribution: the number of successes in n independent trials, each a success
// with the probability q. Its terms P(K = j) = n!/(j!·(n - j)!)·q^j·(1 - q)^(n-j) are computed in logarithms, as
// the factorials of thousands overflow; the sum stops when they get too small to matter (beyond the mean, n·q,
// they fall faster and faster).
double binomial_tail(int n, double q, int k)
{
    double sum = 0;
    for (int j = std::max(k, 0); j <= n; ++j)
    {
        const double log_term = std::lgamma(n + 1.0) - std::lgamma(j + 1.0) - std::lgamma(n - j + 1.0) +
                                j * std::log(q) + (n - j) * std::log1p(-q);
        const double term = std::exp(log_term);
        sum += term;
        if (j > n * q && term < 1e-17 * sum)
        {
            break;
        }
    }
    return sum;
}

// float32, the number format of PFM files and of the GPU's arithmetic, keeps about 7 significant digits. Two renderers
// that compute the same numbers in different ways (the GPU in float; the CPU in double, saved as float) can differ in
// the last of them, and where an image has no noise at all, or almost none (a flat background; a sky seen without depth
// of field, whose color hardly changes across a pixel), such a difference would get a tiny SE and a huge z, though it
// means nothing. So a mean difference up to rounding_margin of the pixel values counts as rounding, and z measures only
// how far it goes beyond that. The margin is 1000 times below the smallest bias worth finding (0.1%), and in most
// tiles less than a thousandth of their standard error. Only the quietest regions notice it: in the sky of weekend,
// whose only noise comes from the lens, it reaches half a tile's standard error at 64 rays per pixel, so differences of
// a millionth are not looked for there either.
constexpr double rounding_margin = 1e-6;

// The test of one mean difference, of the whole image or of a tile
struct mean_test
{
    double mean = 0;           // of d
    double standard_error = 0; // of that mean
    double z = 0;              // the mean in standard errors, beyond the rounding margin
};

// How far the mean of d is from 0, measured by its standard error. level is the size of the pixel values (the mean of
// (|A| + |B|) / 2), for the rounding margin.
//
// Why the standard error is s/√N: d has a random error in every pixel, and the mean of N of them has one too, but a
// smaller one, as errors of both signs partly cancel. For N independent numbers whose standard deviations are σᵢ the
// variance of their sum is the sum of their variances, Σσᵢ², so the variance of the mean is Σσᵢ²/N², and for equal σᵢ
// its standard deviation is σ/√N: the standard error, SE (MonteCarlo shows it at work). In an image the σᵢ are unknown
// and differ from pixel to pixel (a smooth sky, noisy glass), but what SE needs is their average square, and the sample
// variance s² of the d's estimates exactly that when every d has the expected value 0. So SE = s/√N. The pixels must be
// independent: each pixel of a render gets its own random numbers, and A and B get different ones.
//
// z = mean/SE says how many standard errors the mean is from 0. By the central limit theorem, the mean of many
// independent numbers is close to normally distributed, so |z| > 3 happens by chance in 0.27% of cases and |z| > 4 in
// 0.006%. But SE is estimated from the same numbers, and with few of them it can come out too small by chance, which
// makes |z| too large: for normally distributed numbers the ratio has Student's t distribution with N - 1 degrees of
// freedom (W. S. Gosset, 1908), whose tails are wider. With N = 1024 that hardly matters (P(|t| > 4) is 0.007% instead
// of 0.006%), with N = 16 it does (0.12%), and the tiles at the image's edges can be small. So t is turned into the z
// that is exactly as rare: a |z| of 4 means the same in every tile. (When the σᵢ differ, |t| exceeds thresholds as high
// as ours no more often than Student's distribution says, as Bakirov and Székely proved in 2006, so no false alarms
// come from that.)
//
// The heavy tails of path tracing. All of this leans on the central limit theorem, which needs a mean that no single
// number dominates. Path tracing can make fireflies: rare samples hundreds of times brighter than the rest (light that
// found a small lamp through glass by chance). A tile with a firefly in it gets a large mean d, but an even larger s,
// so its z stays about ±1. The danger lies in the tiles without one: there the fireflies' share of the expected value
// is missing from the mean, and s, estimated from ordinary pixels only, is too small for the true spread. If A and B
// have noise of the same shape, this cancels: the share is missing from A and from B alike, and z stays honest (if
// anything, too small). If they don't, it doesn't. When A is sampled with a method that makes fireflies and B with one
// that doesn't, or A with 16 rays per pixel and B with 1024 (a pixel of 16 rays usually has no firefly at all, and now
// and then one that counts 64 times as much as in a pixel of 1024 rays), the tiles where A has none lean toward B with
// too small an SE. |z| is inflated, most of all in tiles that hold only a few firefly pixels, and false alarms follow,
// in any of the checks. So compare renders whose noise has the same shape: made with the same rays per pixel and the
// same sampling method, as the CPU's and the GPU's renders of a scene with the same -s are. The sign test checks that.
mean_test test_mean(const running_stats& d, double level)
{
    const double standard_error = std::sqrt(d.variance() / static_cast<double>(d.count()));
    const double beyond_rounding = std::fabs(d.mean()) - rounding_margin * level;
    double z = 0;
    if (beyond_rounding > 0)
    {
        // SE = 0: every d is the same number, larger than rounding. A difference without any noise: z is infinite.
        const double t = standard_error > 0 ? beyond_rounding / standard_error : infinity;
        z = std::copysign(normal_equivalent(t, static_cast<double>(d.count() - 1)), d.mean());
    }
    return {d.mean(), standard_error, z};
}

// Noise of the same shape: the sign test. The four checks compare means, but they assume a little more (see the heavy
// tails in test_mean): that the noise of A and B has the same shape, the same distribution in every pixel. Then B - A
// and A - B are equally likely in every pixel, and B is brighter than A in half the pixels, as a fair coin shows heads
// in half its throws. So the program counts the throws. Of the n pixels where A and B differ by more than 0.1% (in the
// channel average), B is brighter in b and A in a = n - b. For fair coins b - a is a sum of n independent steps of ±1,
// with the mean 0 and the variance n, so z = (b - a)/√n is standard normal, and no firefly can dominate it: every pixel
// counts ±1, however bright. Why only the pixels that differ by more than 0.1%: rounding can shift every pixel the same
// way by up to rounding_margin, and where the noise is about as small as that (a sky seen without depth of field), it
// would decide the signs of most pixels and move z, though the checks rightly ignore it. A thousand times the margin
// is beyond its reach, and still far below the noise of a render's pixels (a few percent at 64 rays per pixel).
//
// Noise of different shapes moves z, even where the expected values agree. A pixel of 16 rays usually misses the rare
// bright samples and is then a little darker than its expected value (now and then it has one and is much brighter),
// while a pixel of 1024 rays is close to its expected value: a render of 1024 rays is brighter than one of 16 in most
// pixels, though not on average. (Where the rare samples are dark ones, it is the other way round: in materials, a
// render of 256 rays is darker than one of 16 in 52% of the pixels.) Every pixel adds a small lean, and n pixels add it
// up, while the means hardly move: in a synthetic test with fireflies, 16 against 1024 rays per pixel at 480×270, z was
// +41 and the check of all tiles (Σz/√T: the tiles' z added up and divided by √T) only +1.3. A real difference moves
// both the same way and about as far, as both give every pixel (or tile) the same weight and count in standard errors:
// the whole picture 0.1% brighter in the same test gave z = +85 and Σz/√T = +134. So the program prints a note about
// noise of different shapes when |z| is beyond chance (5: in 6 of 10 million comparisons) and Σz/√T has moved less than
// a third as far the same way (see noise_shapes_differ).
constexpr double sign_test_threshold = 1000 * rounding_margin; // 0.1% of the pixel values

struct sign_test
{
    std::int64_t b_brighter = 0; // pixels where B is brighter than A by more than sign_test_threshold
    std::int64_t a_brighter = 0; // and the other way round

    std::int64_t differing() const { return b_brighter + a_brighter; }

    double share() const // of the differing pixels, how many have B brighter
    {
        return differing() > 0 ? static_cast<double>(b_brighter) / static_cast<double>(differing()) : 0.5;
    }

    double z() const
    {
        return differing() > 0
                   ? static_cast<double>(b_brighter - a_brighter) / std::sqrt(static_cast<double>(differing()))
                   : 0.0;
    }
};

// --- The comparison ------------------------------------------------------------------------------------------------

// The channels compared: red, green, blue, and their average, (r + g + b) / 3. The average's mean is the difference of
// the two images' mean radiances, and with all three channels in it, it is usually the least noisy.
constexpr std::array<std::string_view, 4> channel_names{"red", "green", "blue", "average"};
constexpr std::size_t average = 3; // channel_names[average] is "average"

constexpr int default_tile_size = 32; // the side of the tiles, in pixels, unless --tile says otherwise

// A tile of the image, tested on its own: a square, smaller at the right and bottom edges. Its pixels are x0 ≤ x < x1,
// y0 ≤ y < y1.
struct tile
{
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    running_stats d;      // the channel average of B - A in each pixel
    double level_sum = 0; // the sum of the pixels' (|A| + |B|) / 2, averaged over the channels
    mean_test test;

    bool has_spread() const { return d.count() >= 2; } // a single pixel has no spread to estimate SE from
    bool identical() const { return d.rms() == 0; }    // no difference at all, e.g. a black background
};

struct comparison
{
    std::array<running_stats, channel_names.size()> d; // the difference in each pixel, per channel
    std::array<double, channel_names.size()> level{};  // the mean of (|A| + |B|) / 2, for the rounding margin
    std::array<mean_test, channel_names.size()> tests;
    sign_test signs;
    std::vector<tile> tiles; // row by row, from the top left
};

comparison compare(const image& a, const image& b, int tile_size)
{
    comparison result;
    // The number of tiles across and down, rounded up: the last column and row of tiles may be partial. Written so
    // that it can't overflow, as (width + tile_size - 1) / tile_size would for a tile_size near the largest int.
    const int tiles_x = (a.width() - 1) / tile_size + 1;
    const int tiles_y = (a.height() - 1) / tile_size + 1;
    for (int ty = 0; ty < tiles_y; ++ty)
    {
        for (int tx = 0; tx < tiles_x; ++tx)
        {
            result.tiles.push_back({.x0 = tx * tile_size,
                                    .y0 = ty * tile_size,
                                    .x1 = std::min((tx + 1) * tile_size, a.width()),
                                    .y1 = std::min((ty + 1) * tile_size, a.height())});
        }
    }

    std::array<double, channel_names.size()> level_sum{};
    for (int y = 0; y < a.height(); ++y)
    {
        for (int x = 0; x < a.width(); ++x)
        {
            const color& pa = a[x, y];
            const color& pb = b[x, y];
            const color d = pb - pa;
            const color level{(std::fabs(pa.x) + std::fabs(pb.x)) / 2, (std::fabs(pa.y) + std::fabs(pb.y)) / 2,
                              (std::fabs(pa.z) + std::fabs(pb.z)) / 2};
            const std::array differences{d.x, d.y, d.z, (d.x + d.y + d.z) / 3};
            const std::array levels{level.x, level.y, level.z, (level.x + level.y + level.z) / 3};
            for (std::size_t c = 0; c < channel_names.size(); ++c)
            {
                result.d[c].add(differences[c]);
                level_sum[c] += levels[c];
            }

            const double threshold = sign_test_threshold * levels[average];
            if (differences[average] > threshold)
            {
                ++result.signs.b_brighter;
            }
            else if (differences[average] < -threshold)
            {
                ++result.signs.a_brighter;
            }

            tile& t = result.tiles[static_cast<std::size_t>(y / tile_size) * tiles_x + x / tile_size];
            t.d.add(differences[average]);
            t.level_sum += levels[average];
        }
    }

    const double pixels = static_cast<double>(result.d[0].count());
    for (std::size_t c = 0; c < channel_names.size(); ++c)
    {
        result.level[c] = level_sum[c] / pixels;
        result.tests[c] = test_mean(result.d[c], result.level[c]);
    }
    for (tile& t : result.tiles)
    {
        if (t.has_spread())
        {
            t.test = test_mean(t.d, t.level_sum / static_cast<double>(t.d.count()));
        }
    }
    return result;
}

// Two renders made with the same random numbers are identical, and their comparison would mean nothing
bool identical(const comparison& result)
{
    return std::ranges::all_of(result.d, [](const running_stats& d) { return d.rms() == 0; });
}

// --- The verdict ---------------------------------------------------------------------------------------------------

// Any test of noisy data can raise a false alarm: an extreme value that came by chance. Its rate is a choice: lower
// means fewer false alarms but a real difference must be larger to be noticed. Here, two renders of the same picture
// fail each of the four checks by chance in at most 1 of 400 comparisons, and so any of them in at most 1 of 100:
// the probability that at least one of several events happens is at most the sum of their probabilities.
constexpr double false_alarm_rate_per_check = 1.0 / 400;

// The thresholds of the four checks. Some of them look at several numbers, and the same inequality spreads their false
// alarm rate over them (Bonferroni's correction).
// 1. The whole image: four z (red, green, blue, average), so each may exceed its threshold by chance with a quarter of
//    the rate, 1/1600: |z| > 3.42.
// 2. All tiles together: Stouffer's method (1949). If the z of every tile has the standard normal distribution and the
//    tiles are independent (they have different pixels), the sum of T of them has the variance T, so Σz/√T is standard
//    normal again: |Σz/√T| > 3.02 has the rate 1/400. In this sum every tile weighs the same, however noisy it is, so a
//    small bias common to all tiles adds up even where one very noisy region dominates the mean of the whole image.
//    The same strength is its weakness: if A and B have noise of different shapes (see test_mean), every tile's z
//    leans a little the same way, and the sum adds that up too.
// 3. The largest tile: with 2040 tiles (a 1920×1080 image), about 5.5 of them have |z| > 3 by chance alone, so the
//    largest |z| is large even without any difference. Each of T tiles gets the rate/T, and the largest |z| is compared
//    with that threshold: 4.85 for 2040 tiles.
// 4. The number K of tiles with |z| > 3: a tile has |z| > 3 with the probability 0.27%, independently of the others, so
//    K has the binomial distribution, and the threshold is the smallest count k with P(K ≥ k) at most the rate. For
//    2040 tiles that is 14, while 5.5 is the mean.
struct thresholds
{
    double whole_image = 0;
    double all_tiles = 0;
    double largest_tile = 0;
    int tiles_over_3 = 0;
};

constexpr double tile_z_counted = 3; // check 4 counts the tiles with |z| above this
constexpr double tile_z_clamp = 40;  // check 2 adds up the tiles' z clamped to ±40 (see summarize)

thresholds alarm_thresholds(int tested_tiles)
{
    thresholds result{.whole_image = normal_quantile(false_alarm_rate_per_check / channel_names.size()),
                      .all_tiles = normal_quantile(false_alarm_rate_per_check)};
    if (tested_tiles > 0)
    {
        result.largest_tile = normal_quantile(false_alarm_rate_per_check / tested_tiles);
        // Up to the mean count, P(K ≥ k) is about 1/2 or more, far above the rate, so the search starts there
        const double q = normal_tail(tile_z_counted);
        result.tiles_over_3 = std::max(1, static_cast<int>(tested_tiles * q));
        while (binomial_tail(tested_tiles, q, result.tiles_over_3) > false_alarm_rate_per_check)
        {
            ++result.tiles_over_3;
        }
    }
    return result;
}

// What the tiles show: how many were tested, their z added up, how many of them have |z| > 3, and which has the largest
struct tile_summary
{
    int tested = 0;
    int identical = 0;      // tiles where A and B don't differ at all: nothing to test there
    double all_tiles_z = 0; // Σz/√T, Stouffer's combination of their z
    int over_3 = 0;
    const tile* largest = nullptr;
};

tile_summary summarize(const std::vector<tile>& tiles)
{
    tile_summary summary;
    double z_sum = 0;
    for (const tile& t : tiles)
    {
        if (!t.has_spread())
        {
            continue;
        }
        if (t.identical())
        {
            ++summary.identical;
            continue;
        }
        ++summary.tested;
        // Clamped, so that tiles of +∞ and -∞ (differences without noise, of both signs) can't add up to NaN. A tile
        // beyond ±40 fails the largest tile's check anyway.
        z_sum += std::clamp(t.test.z, -tile_z_clamp, tile_z_clamp);
        if (std::fabs(t.test.z) > tile_z_counted)
        {
            ++summary.over_3;
        }
        if (!summary.largest || std::fabs(t.test.z) > std::fabs(summary.largest->test.z))
        {
            summary.largest = &t;
        }
    }
    summary.all_tiles_z = summary.tested > 0 ? z_sum / std::sqrt(summary.tested) : 0.0;
    return summary;
}

// One of the four checks, ready to print: the value it looks at, its threshold, and whether that was exceeded
struct check
{
    std::string_view name;
    std::string_view description;
    std::string value;
    std::string limit;
    bool failed = false;
};

std::vector<check> run_checks(const comparison& result, const tile_summary& summary, const thresholds& limits)
{
    double largest_whole_image_z = 0;
    for (const mean_test& test : result.tests)
    {
        largest_whole_image_z = std::max(largest_whole_image_z, std::fabs(test.z));
    }
    std::vector<check> checks{{"whole image", "the largest |z| of its 4 rows",
                               std::format("{:.2f}", largest_whole_image_z),
                               std::format("fails above {:.2f}", limits.whole_image),
                               largest_whole_image_z > limits.whole_image}};
    if (summary.largest) // else no tile was tested: every tile of more than one pixel is identical in A and B
    {
        const double all_tiles_z = std::fabs(summary.all_tiles_z);
        checks.push_back({"all tiles", "|Σz/√T| of the T tiles", std::format("{:.2f}", all_tiles_z),
                          std::format("fails above {:.2f}", limits.all_tiles), all_tiles_z > limits.all_tiles});
        const double largest_tile_z = std::fabs(summary.largest->test.z);
        checks.push_back({"largest tile", "the largest |z| of a tile", std::format("{:.2f}", largest_tile_z),
                          std::format("fails above {:.2f}", limits.largest_tile),
                          largest_tile_z > limits.largest_tile});
        checks.push_back({"tile count", "the number of tiles with |z| > 3", std::format("{}", summary.over_3),
                          std::format("fails at {} or more", limits.tiles_over_3),
                          summary.over_3 >= limits.tiles_over_3});
    }
    return checks;
}

// Does the sign test point to noise of different shapes (see sign_test)? Its |z| must be beyond chance, while
// all_tiles_z, the Σz/√T of tiles of 32×32 pixels, which a real difference moves the same way and about as far, has
// moved less than a third as far. The tiles are that small whatever --tile says, because the comparison needs tiles
// whose pixels are about equally noisy. A large tile holds quiet and noisy pixels together, its SE is set by the
// noisiest, and a real difference in the quiet ones hardly moves its z, while the sign test, which counts every pixel
// alike, sees it in full: with tiles of 256×256 pixels, the whole picture 0.1% brighter in the synthetic test gave
// z = +85 and Σz/√T = +0.6 (+134 with 32×32). Small tiles are also safe from the clamp of ±40 in Σz/√T: even when
// all 1024 pixels of a tile point the same way, the tile adds less to the sign test than to Σz/√T.
constexpr double sign_test_beyond_chance = 5;
constexpr double sign_test_to_all_tiles = 3;

bool noise_shapes_differ(const sign_test& signs, double all_tiles_z)
{
    const double z = signs.z();
    const double all_tiles_same_way = z > 0 ? all_tiles_z : -all_tiles_z;
    return std::fabs(z) > sign_test_beyond_chance && all_tiles_same_way < std::fabs(z) / sign_test_to_all_tiles;
}

// --- Output --------------------------------------------------------------------------------------------------------

void print_whole_image(const comparison& result)
{
    std::println("\nThe difference d = B - A in each pixel, over {} pixels:", result.d[0].count());
    std::println("  {:8} {:>12} {:>11} {:>8} {:>10} {:>10}", "", "mean of d", "SE", "z", "relative", "RMS of d");
    for (std::size_t c = 0; c < channel_names.size(); ++c)
    {
        const mean_test& test = result.tests[c];
        const double relative = result.level[c] > 0 ? test.mean / result.level[c] : 0.0;
        std::println("  {:8} {:+12.8f} {:11.8f} {:+8.2f} {:+9.4f}% {:10.6f}", channel_names[c], test.mean,
                     test.standard_error, test.z, 100 * relative, result.d[c].rms());
    }
    std::println("  SE: the standard error of the mean; z ≈ mean / SE, the mean in standard errors (less a rounding\n"
                 "  margin of a millionth of the pixel values); relative: the mean as a share of the pixel values;\n"
                 "  RMS: the typical size of d, the two images' noise together");
    const sign_test& signs = result.signs;
    if (signs.differing() > 0)
    {
        std::println("  sign test: B is brighter in {:.2f}% of the {} pixels that differ by more than {:g}%,\n"
                     "  z = {:+.2f} (noise of the same shape in A and B gives 50%)",
                     100 * signs.share(), signs.differing(), 100 * sign_test_threshold, signs.z());
    }
    else
    {
        std::println("  sign test: no pixel differs by more than {:g}%", 100 * sign_test_threshold);
    }
}

void print_tiles(const tile_summary& summary, int tile_size, bool partial_tiles)
{
    std::println("\nTiles of {}×{} pixels{}: {} tested, by the channel average of d", tile_size, tile_size,
                 partial_tiles ? " (smaller at the right and bottom edges)" : "", summary.tested);
    if (summary.identical > 0)
    {
        std::println("  not tested: {} tiles identical in A and B (no noise there, e.g. a black background)",
                     summary.identical);
    }
    if (const tile* t = summary.largest)
    {
        std::println("  all tiles together: Σz/√T = {:+.2f}", summary.all_tiles_z);
        const double z = t->test.z;
        std::println("  largest |z|: {:.2f} at x {}–{}, y {}–{} ({})", std::fabs(z), t->x0, t->x1 - 1, t->y0,
                     t->y1 - 1,
                     z > 0 ? "B is brighter there" : z < 0 ? "A is brighter there" : "no difference beyond rounding");
        std::println("  tiles with |z| > {}: {}, while noise alone gives {:.1f} on average", tile_z_counted,
                     summary.over_3, summary.tested * normal_tail(tile_z_counted));
    }
}

void print_checks(const std::vector<check>& checks)
{
    // How rare a false alarm is: 1 in 400 comparisons per check, 1 in 100 for any of the four
    const auto one_in = [](double rate) { return static_cast<int>(std::lround(1 / rate)); };
    if (checks.size() == 1)
    {
        std::println("\nCheck (by chance alone it fails in at most 1 of {} comparisons):",
                     one_in(false_alarm_rate_per_check));
    }
    else
    {
        std::println("\nChecks (by chance alone each fails in at most 1 of {} comparisons, any of the {} in 1 of {}):",
                     one_in(false_alarm_rate_per_check), checks.size(),
                     one_in(static_cast<double>(checks.size()) * false_alarm_rate_per_check));
    }
    for (const check& c : checks)
    {
        std::println("  {:13} {:32} {:>8}   {:22} {}", c.name, c.description, c.value, c.limit,
                     c.failed ? "FAILED" : "passed");
    }
}

void print_shape_note(const sign_test& signs, double all_tiles_z)
{
    std::println("\nNote: the sign test is z = {:+.2f} from 50%, while the means of {}×{} tiles give Σz/√T = {:+.2f}.\n"
                 "A different picture moves both the same way and about as far; noise of different shapes, from\n"
                 "different rays per pixel or sampling methods, moves the sign test much more. The checks assume\n"
                 "noise of the same shape, so they can fail with no real difference, or miss one: compare renders\n"
                 "made with the same settings. (If they already were, the two renderers sample differently.)",
                 signs.z(), default_tile_size, default_tile_size, all_tiles_z);
}

// The map of the tiles' z: image A in gray, each tile tinted red where B is brighter (z > 0) and blue where it is
// darker. The tint grows with (|z| / threshold)² and is full at the threshold of the largest tile's check, so tiles of
// ordinary noise (|z| around 1) stay nearly gray, and a difference stands out.
image z_map(const image& a, const std::vector<tile>& tiles, double threshold)
{
    // Colors as they will look in the PNG. save_png takes the square root of every value (gamma correction, see
    // image.ixx), so a value v is stored as v², and the blending happens between the colors a person sees.
    constexpr color brighter{1.0, 0.15, 0.1};
    constexpr color darker{0.1, 0.35, 1.0};
    image map(a.width(), a.height());
    for (const tile& t : tiles)
    {
        const double strength = threshold > 0 ? std::min(std::fabs(t.test.z) / threshold, 1.0) : 0.0;
        const double tint = strength * strength;
        const color& tint_color = t.test.z > 0 ? brighter : darker;
        for (int y = t.y0; y < t.y1; ++y)
        {
            for (int x = t.x0; x < t.x1; ++x)
            {
                // Brightness as the eye weighs the channels (Rec. 709), in gamma space and a little dimmed, so that
                // a tint shows even over white
                const color& p = a[x, y];
                const double luminance = 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
                const double gray = 0.85 * std::sqrt(std::clamp(luminance, 0.0, 1.0));
                const color shown = (1 - tint) * color{gray, gray, gray} + tint * tint_color;
                map[x, y] = shown * shown;
            }
        }
    }
    return map;
}

// --- The command line ----------------------------------------------------------------------------------------------

struct options
{
    std::vector<std::filesystem::path> images; // A and B
    int tile_size = default_tile_size;
    std::optional<std::filesystem::path> zmap;
    bool help = false;
};

void print_usage()
{
    std::println(R"(Usage: ImageCompare A.pfm B.pfm [options]

Are A and B renders of the same picture that differ only by Monte Carlo noise? A and B must be
independent renders of the same scene (with different seeds, or from different renderers) at the
same size, with the same rays per pixel and the same sampling method, so that their noise has the
same shape; saved as PFM, for example:
  RayTracing materials -w 480 -s 64 -o a.pfm
  RayTracing materials -w 480 -s 64 --seed 1 -o b.pfm
  ImageCompare a.pfm b.pfm

Options:
  --tile N         the side of the square tiles that are tested one by one, in pixels (default {})
  --zmap FILE.png  save a map of the tiles' z-scores: image A in gray, tinted red where B is
                   brighter and blue where it is darker, in full color at the largest tile's
                   alarm threshold
  -h, --help       show this help

Exit code: 0 if the images are consistent, 1 if they differ, 2 on an error.)",
                 default_tile_size);
}

std::expected<options, std::string> parse_options(std::span<char* const> args)
{
    options opts;
    for (std::size_t i = 1; i < args.size(); ++i) // args[0] is the program name
    {
        const std::string_view arg = args[i];
        if (arg == "-h" || arg == "--help")
        {
            opts.help = true;
            continue;
        }
        if (!arg.starts_with('-'))
        {
            opts.images.emplace_back(arg);
            continue;
        }
        if (arg != "--tile" && arg != "--zmap")
        {
            return std::unexpected(std::format("unknown option '{}'", arg));
        }
        if (i + 1 == args.size())
        {
            return std::unexpected(std::format("{} needs a value", arg));
        }
        const std::string_view value = args[++i];
        if (arg == "--zmap")
        {
            opts.zmap = value;
            // Checked before the comparison, as the renderers check their outputs before rendering (see cli)
            if (const auto checked = check_output(*opts.zmap); !checked)
            {
                return std::unexpected(std::format("{}: {}", arg, checked.error()));
            }
            continue;
        }
        const std::optional<int> size = parse_positive_int(value);
        if (!size || *size < 2)
        {
            return std::unexpected(std::format("--tile: expected an integer of at least 2, not '{}'", value));
        }
        opts.tile_size = *size;
    }
    if (!opts.help && opts.images.size() != 2)
    {
        return std::unexpected(std::format("expected two images, A and B, not {}", opts.images.size()));
    }
    return opts;
}

// Loads a render, and refuses one with values that aren't numbers or are infinite: the renderers never write them
// (a NaN sample counts as 0), so they mean a damaged file or a broken renderer, and no statistics can be done with them
std::expected<image, std::string> load_render(const std::filesystem::path& path)
{
    std::expected<image, std::string> img = image::load_pfm(path);
    if (!img)
    {
        return std::unexpected(std::format("{} ({})", img.error(), to_utf8(path)));
    }
    std::int64_t non_finite = 0;
    for (int y = 0; y < img->height(); ++y)
    {
        for (int x = 0; x < img->width(); ++x)
        {
            const color& p = (*img)[x, y];
            for (const double value : {p.x, p.y, p.z})
            {
                if (!std::isfinite(value))
                {
                    ++non_finite;
                }
            }
        }
    }
    if (non_finite > 0)
    {
        return std::unexpected(
            std::format("{} values in {} are NaN or infinite: a damaged file or a broken renderer", non_finite,
                        to_utf8(path)));
    }
    return img;
}

int main(int argc, char* argv[])
{
    const auto opts = parse_options(std::span(argv, static_cast<std::size_t>(argc)));
    if (!opts)
    {
        std::println("Error: {}\n", opts.error());
        print_usage();
        return 2;
    }
    if (opts->help)
    {
        print_usage();
        return 0;
    }

    const auto a = load_render(opts->images[0]);
    const auto b = load_render(opts->images[1]);
    if (!a || !b)
    {
        std::println("Error: {}", !a ? a.error() : b.error());
        return 2;
    }
    std::println("A: {}: {}×{}, mean radiance {:.6f}", to_utf8(opts->images[0]), a->width(), a->height(),
                 a->mean_radiance());
    std::println("B: {}: {}×{}, mean radiance {:.6f}", to_utf8(opts->images[1]), b->width(), b->height(),
                 b->mean_radiance());
    if (a->width() != b->width() || a->height() != b->height())
    {
        std::println("Error: the images have different sizes");
        return 2;
    }
    if (static_cast<std::int64_t>(a->width()) * a->height() < 2)
    {
        std::println("Error: an image of one pixel has no spread to compare with");
        return 2;
    }

    const comparison result = compare(*a, *b, opts->tile_size);
    if (identical(result))
    {
        std::println("Error: A and B are identical: the comparison needs two independent renders (different seeds)");
        return 2;
    }
    print_whole_image(result);
    const tile_summary summary = summarize(result.tiles);
    print_tiles(summary, opts->tile_size, a->width() % opts->tile_size != 0 || a->height() % opts->tile_size != 0);
    const thresholds limits = alarm_thresholds(summary.tested);
    const std::vector<check> checks = run_checks(result, summary, limits);
    print_checks(checks);

    if (opts->zmap)
    {
        if (const auto saved = z_map(*a, result.tiles, limits.largest_tile).save(*opts->zmap); !saved)
        {
            std::println("Error: {} ({})", saved.error(), to_utf8(*opts->zmap));
            return 2;
        }
        std::println("\nMap of the tiles' z saved: {}", to_utf8(std::filesystem::absolute(*opts->zmap)));
    }

    // The note about noise of different shapes compares the sign test with Σz/√T of 32×32 tiles (see
    // noise_shapes_differ), so with another --tile those tiles are tested once more, for the note only
    double all_tiles_z = summary.all_tiles_z;
    if (opts->tile_size != default_tile_size)
    {
        const comparison small_tiles = compare(*a, *b, default_tile_size);
        all_tiles_z = summarize(small_tiles.tiles).all_tiles_z;
    }
    if (noise_shapes_differ(result.signs, all_tiles_z))
    {
        print_shape_note(result.signs, all_tiles_z);
    }

    // The names of the failed checks: std::views::filter keeps the checks whose member `failed` is true (a pointer to
    // a member works as a function, through std::invoke), and std::views::transform takes their names. Not const: a
    // filter_view finds its first element on the first begin() and remembers it, so begin() changes the view, and
    // std::ranges::empty can't call it on a const one.
    auto failed = checks | std::views::filter(&check::failed) | std::views::transform(&check::name);
    if (std::ranges::empty(failed))
    {
        std::println("\nVerdict: consistent: A and B differ by no more than noise");
        return 0;
    }
    // std::views::join_with (C++23) puts ", " between the names, and std::ranges::to collects them into a string
    std::println("\nVerdict: DIFFERENT: A and B differ by more than noise (failed: {})",
                 failed | std::views::join_with(std::string_view(", ")) | std::ranges::to<std::string>());
    return 1;
}
