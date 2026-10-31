// PfmStats: numbers about renders saved as PFM files, for the checks that ImageCompare doesn't make. ImageCompare
// decides whether two renders are the same picture up to noise; the checks of the GPU's third kernel also ask how
// noisy a render is, whether an image of a white furnace is exactly 1, and whether two renders are the same bytes.
//   stats A.pfm [--rect X0 Y0 X1 Y1 [--expect V T]]... [--equal V [--share F]] [--max V] [--finite]
//       the mean, its standard error, the smallest and the largest value of each channel, of the whole image or of
//       each rectangle (columns X0 to X1 and rows Y0 to Y1, both inclusive, row 0 at the top). --expect: the
//       rectangle's mean (the channel average) must lie within 4 standard errors + T of V. --equal: at least the share
//       F (default 0.9999) of the values of every region must be exactly V (as a float). --max: no value may exceed
//       V. --finite is always checked (see the exit codes).
//   noise A1.pfm A2.pfm [--trim F] [--tiles N]
//       the noise of one render, from two renders that differ only in their seed: d = A2 - A1 of the channel average
//       has twice the variance of one render, so σ = RMS(d)/√2. Untrimmed and with the largest share F (default 0.01)
//       of the d² left out, as fireflies and the edges of lamps make the plain RMS vary between seeds. --tiles N also
//       prints σ in tiles of N×N pixels, to see where the noise is.
//   rule T1a T1b T2a T2b A1 A2 B1 B2 [--margin M] [--trim F]
//       the RMS rule for two strategies of unequal noise: each test pair (T1a, T1b), (T2a, T2b) renders A against B
//       with independent seeds, so its expected d² is σ_A² + σ_B², which the controls A1-A2 and B1-B2 measure as
//       (R_A² + R_B²)/2. A pattern destroyed inside the tiles (a checker gone gray) passes ImageCompare's tile means
//       but not this: trimRMS(test) ≤ M·√((R_A² + R_B²)/2), M = 1.10 by default.
//   same A.pfm B.pfm [--rect X0 Y0 X1 Y1]... [--ulps N --max-share F]
//       the same floats in every pixel (of the rectangles), or at most the share F of the pixels apart, by at most N
//       ulps in any channel.
//   png-same a.png b.png
//       the same decoded pixels: the window's frames are PNGs with an alpha channel, RayTracingGPU's without one.
//   error TEST DENOISED REF1 REF2 [--min-gain G] [--max-drift F] [--tiles N] [--spp S]
//       what a denoiser removed: TEST a noisy render, DENOISED the same render denoised, REF1 and REF2 references of
//       many samples with independent seeds, clamped to [0, 10^4] as the denoiser's input is (denoise.cu). Against
//       their average R, the error of each image corrected for R's own noise: MSE' = mean((x − R)²) − v, with
//       v = mean((REF1 − REF2)²)/4, R's variance (each reference's is half the variance of their difference, and R's
//       half of that again); relMSE' the same per value, divided by R² + 0.01. Passes when the denoiser removed at
//       least G = 4 times the noisy image's relative error, relMSE'(DENOISED) ≤ relMSE'(TEST)/G, and kept the mean:
//       in every channel |mean(DENOISED) − mean(R)| ≤ F·mean(R), F = 0.02. The gain is relative because an absolute
//       one belongs to the brightest pixels: a lamp seen directly (around 15 beside a ceiling below 1) and a softbox's
//       reflections hold almost all of a denoised image's MSE', where the denoiser softens an edge, while relMSE'
//       measures each value's error relative to its own brightness (divided by R² + 0.01), as the eye sees an error.
//       Also printed: the samples the gain is worth (S = 16, TEST's samples, times it), MSE'(TEST)/MSE'(DENOISED)
//       without a verdict, and the largest drift of a tile of N×N pixels (N = 32), (mean(DENOISED) − mean(R))/(mean(R)
//       + 0.01) of the channel average. A denoised error within R's own noise (relMSE'(DENOISED) ≤ 0) is an unbounded
//       gain, which passes.
//   self-test
//       checks this program's own arithmetic on small images it writes (for ctest).
// Exit code: 0 the check passes, 1 it fails, 2 an error (a file that can't be read, sizes that differ, a NaN or an
// infinity in a render).
import std;
import cli;
import image;
import vec3;

// --- Errors and images ----------------------------------------------------------------------------------------------

// An error that ends the program with exit code 2
struct usage_error
{
    std::string message;
};

image load_pfm(const std::string& path)
{
    auto loaded = image::load_pfm(std::filesystem::path(path));
    if (!loaded)
    {
        throw usage_error{std::format("{}: {}", path, loaded.error())};
    }
    for (int y = 0; y < loaded->height(); ++y)
    {
        for (int x = 0; x < loaded->width(); ++x)
        {
            const color& c = (*loaded)[x, y];
            if (!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z))
            {
                throw usage_error{std::format("{}: pixel ({}, {}) is not finite: a broken render", path, x, y)};
            }
        }
    }
    return std::move(*loaded);
}

void check_same_size(const image& a, const image& b, std::string_view what)
{
    if (a.width() != b.width() || a.height() != b.height())
    {
        throw usage_error{std::format("{}: {}×{} and {}×{}: the images must have the same size", what, a.width(),
                                      a.height(), b.width(), b.height())};
    }
}

double channel(const color& c, int k) { return k == 0 ? c.x : k == 1 ? c.y : k == 2 ? c.z : (c.x + c.y + c.z) / 3; }

// A rectangle of pixels: columns x0 to x1 - 1, rows y0 to y1 - 1 from the top
struct rect
{
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    std::optional<std::pair<double, double>> expect; // --expect V T
};

rect whole(const image& img) { return {0, 0, img.width(), img.height(), {}}; }

void check_rect(const rect& r, const image& img)
{
    if (!(0 <= r.x0 && r.x0 < r.x1 && r.x1 <= img.width() && 0 <= r.y0 && r.y0 < r.y1 && r.y1 <= img.height()))
    {
        throw usage_error{std::format("the rectangle {} {} {} {} is not inside the {}×{} image", r.x0, r.y0,
                                      r.x1 - 1, r.y1 - 1, img.width(), img.height())};
    }
}

// --- Arguments ------------------------------------------------------------------------------------------------------

class arguments
{
public:
    explicit arguments(std::vector<std::string> list) : list_(std::move(list)) {}
    bool done() const { return next_ == list_.size(); }
    const std::string& peek() const { return list_[next_]; }
    std::string take(std::string_view what)
    {
        if (done())
        {
            throw usage_error{std::format("{} is missing", what)};
        }
        return list_[next_++];
    }
    double number(std::string_view what)
    {
        const std::string text = take(what);
        const std::optional<double> value = parse_number<double>(text);
        if (!value || !std::isfinite(*value))
        {
            throw usage_error{std::format("{}: expected a number, not '{}'", what, text)};
        }
        return *value;
    }
    int integer(std::string_view what)
    {
        const std::string text = take(what);
        const std::optional<int> value = parse_number<int>(text);
        if (!value)
        {
            throw usage_error{std::format("{}: expected an integer, not '{}'", what, text)};
        }
        return *value;
    }
    rect rectangle()
    {
        rect r;
        r.x0 = integer("--rect X0");
        r.y0 = integer("--rect Y0");
        r.x1 = integer("--rect X1") + 1; // inclusive on the command line, one past the end here
        r.y1 = integer("--rect Y1") + 1;
        return r;
    }

private:
    std::vector<std::string> list_;
    std::size_t next_ = 0;
};

// --- stats ----------------------------------------------------------------------------------------------------------

struct channel_stats
{
    double mean = 0;
    double se = 0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
};

// Mean, standard error, min and max of channel k (3: the average) over a rectangle. Welford's update, as ImageCompare's
// running_stats: a patch of a white furnace has values that are all nearly equal.
channel_stats region_stats(const image& img, const rect& r, int k)
{
    channel_stats s;
    double mean = 0;
    double m2 = 0;
    std::int64_t n = 0;
    for (int y = r.y0; y < r.y1; ++y)
    {
        for (int x = r.x0; x < r.x1; ++x)
        {
            const double v = channel(img[x, y], k);
            ++n;
            const double delta = v - mean;
            mean += delta / static_cast<double>(n);
            m2 += delta * (v - mean);
            s.min = std::min(s.min, v);
            s.max = std::max(s.max, v);
        }
    }
    s.mean = mean;
    s.se = n > 1 ? std::sqrt(m2 / static_cast<double>(n - 1) / static_cast<double>(n)) : 0;
    return s;
}

// The values of the three channels in a rectangle that equal v exactly, and those above `above`
struct value_counts
{
    std::int64_t total = 0;
    std::int64_t equal = 0;
    std::int64_t above = 0;
};

value_counts count_values(const image& img, const rect& r, double v, double above)
{
    value_counts c;
    for (int y = r.y0; y < r.y1; ++y)
    {
        for (int x = r.x0; x < r.x1; ++x)
        {
            for (int k = 0; k < 3; ++k)
            {
                const double value = channel(img[x, y], k);
                ++c.total;
                c.equal += static_cast<float>(value) == static_cast<float>(v) ? 1 : 0; // the PFM holds floats
                c.above += value > above ? 1 : 0;
            }
        }
    }
    return c;
}

int run_stats(arguments& args)
{
    const std::string path = args.take("the PFM file");
    std::vector<rect> rects;
    std::optional<double> equal;
    double share = 0.9999;
    std::optional<double> max;
    while (!args.done())
    {
        const std::string option = args.take("an option");
        if (option == "--rect")
        {
            rects.push_back(args.rectangle());
        }
        else if (option == "--expect")
        {
            if (rects.empty())
            {
                throw usage_error{"--expect follows the --rect it is about"};
            }
            const double v = args.number("--expect V");
            const double t = args.number("--expect T");
            rects.back().expect = std::pair{v, t};
        }
        else if (option == "--equal")
        {
            equal = args.number("--equal");
        }
        else if (option == "--share")
        {
            share = args.number("--share");
        }
        else if (option == "--max")
        {
            max = args.number("--max");
        }
        else if (option == "--finite")
        {
            // Always checked: a NaN or an infinity anywhere is exit code 2. Accepted, so a command can say it.
        }
        else
        {
            throw usage_error{std::format("stats: unknown option '{}'", option)};
        }
    }
    if (!(share >= 0 && share <= 1))
    {
        throw usage_error{"stats: --share is a share from 0 to 1"};
    }
    const image img = load_pfm(path);
    if (rects.empty())
    {
        rects.push_back(whole(img));
    }
    std::println("{}: {}×{}", path, img.width(), img.height());
    bool pass = true;
    constexpr std::array names{"red", "green", "blue", "average"};
    for (const rect& r : rects)
    {
        check_rect(r, img);
        std::println("region {} {} {} {} ({} pixels):", r.x0, r.y0, r.x1 - 1, r.y1 - 1,
                     static_cast<std::int64_t>(r.x1 - r.x0) * (r.y1 - r.y0));
        for (int k = 0; k < 4; ++k)
        {
            const channel_stats s = region_stats(img, r, k);
            std::println("  {:8} mean {:.9g}, SE {:.3g}, min {:.9g}, max {:.9g}", names[k], s.mean, s.se, s.min, s.max);
        }
        if (r.expect)
        {
            const channel_stats s = region_stats(img, r, 3);
            const auto [v, t] = *r.expect;
            const double allowed = 4 * s.se + t;
            const bool ok = std::abs(s.mean - v) <= allowed;
            std::println("  expected {:.9g}: off by {:.3g}, allowed 4 SE + {:.3g} = {:.3g}: {}", v, s.mean - v, t,
                         allowed, ok ? "PASS" : "FAIL");
            pass = pass && ok;
        }
        if (equal || max)
        {
            const value_counts c = count_values(img, r, equal.value_or(0), max.value_or(0));
            if (equal)
            {
                const double got = static_cast<double>(c.equal) / static_cast<double>(c.total);
                const bool ok = got >= share;
                std::println("  values exactly {:.9g}: {} of {} ({:.6f}%), at least {:.4f}% needed: {}", *equal,
                             c.equal, c.total, 100 * got, 100 * share, ok ? "PASS" : "FAIL");
                pass = pass && ok;
            }
            if (max)
            {
                const bool ok = c.above == 0;
                std::println("  values above {:.9g}: {}: {}", *max, c.above, ok ? "PASS" : "FAIL");
                pass = pass && ok;
            }
        }
    }
    return pass ? 0 : 1;
}

// --- noise and rule -------------------------------------------------------------------------------------------------

// The squared differences d² of the channel average, B - A, in a rectangle
std::vector<double> squared_differences(const image& a, const image& b, const rect& r)
{
    std::vector<double> d2;
    d2.reserve(static_cast<std::size_t>(r.x1 - r.x0) * (r.y1 - r.y0));
    for (int y = r.y0; y < r.y1; ++y)
    {
        for (int x = r.x0; x < r.x1; ++x)
        {
            const double d = channel(b[x, y], 3) - channel(a[x, y], 3);
            d2.push_back(d * d);
        }
    }
    return d2;
}

// √(mean of d²), the full RMS that ImageCompare prints in its "average" row
double rms(const std::vector<double>& d2)
{
    return d2.empty() ? 0 : std::sqrt(std::ranges::fold_left(d2, 0.0, std::plus{}) / static_cast<double>(d2.size()));
}

// The RMS of the smallest d², leaving out the largest share `trim` of them (⌊N·trim⌋ values)
double trimmed_rms(std::vector<double> d2, double trim)
{
    const auto drop = static_cast<std::size_t>(std::floor(static_cast<double>(d2.size()) * trim));
    const std::size_t keep = d2.size() - drop;
    if (keep == 0)
    {
        return 0;
    }
    std::ranges::nth_element(d2, d2.begin() + static_cast<std::ptrdiff_t>(keep - 1));
    const double sum = std::accumulate(d2.begin(), d2.begin() + static_cast<std::ptrdiff_t>(keep), 0.0);
    return std::sqrt(sum / static_cast<double>(keep));
}

struct pair_rms
{
    double full = 0;
    double trimmed = 0;
};

pair_rms pair_noise(const image& a, const image& b, double trim)
{
    const std::vector<double> d2 = squared_differences(a, b, whole(a));
    return {rms(d2), trimmed_rms(d2, trim)};
}

int run_noise(arguments& args)
{
    const std::string path1 = args.take("the first PFM file");
    const std::string path2 = args.take("the second PFM file");
    double trim = 0.01;
    int tiles = 0;
    while (!args.done())
    {
        const std::string option = args.take("an option");
        if (option == "--trim")
        {
            trim = args.number("--trim");
        }
        else if (option == "--tiles")
        {
            tiles = args.integer("--tiles");
        }
        else
        {
            throw usage_error{std::format("noise: unknown option '{}'", option)};
        }
    }
    if (!(trim >= 0 && trim <= 0.5) || tiles < 0)
    {
        throw usage_error{"noise: --trim is from 0 to 0.5, --tiles at least 1"};
    }
    const image a = load_pfm(path1);
    const image b = load_pfm(path2);
    check_same_size(a, b, "noise");
    const pair_rms r = pair_noise(a, b, trim);
    std::println("{} - {}: {}×{}, mean radiance {:.6f} and {:.6f}", path2, path1, a.width(), a.height(),
                 a.mean_radiance(), b.mean_radiance());
    std::println("RMS of d (channel average): {:.9g}; trimmed by {}: {:.9g}", r.full, trim, r.trimmed);
    std::println("sigma of one render (RMS / sqrt 2): {:.9g}; trimmed: {:.9g}", r.full / std::numbers::sqrt2,
                 r.trimmed / std::numbers::sqrt2);
    if (tiles > 0)
    {
        // σ per tile, untrimmed, as a grid: where the noise is
        const int across = (a.width() + tiles - 1) / tiles;
        const int down = (a.height() + tiles - 1) / tiles;
        std::println("sigma in tiles of {}×{} ({} across, {} down), rows from the top:", tiles, tiles, across, down);
        for (int ty = 0; ty < down; ++ty)
        {
            std::string line;
            for (int tx = 0; tx < across; ++tx)
            {
                const rect t{tx * tiles, ty * tiles, std::min(a.width(), (tx + 1) * tiles),
                             std::min(a.height(), (ty + 1) * tiles), {}};
                line += std::format(" {:9.3e}", rms(squared_differences(a, b, t)) / std::numbers::sqrt2);
            }
            std::println("{}", line);
        }
    }
    return 0;
}

int run_rule(arguments& args)
{
    std::array<std::string, 8> paths;
    constexpr std::array<std::string_view, 8> names{"T1a", "T1b", "T2a", "T2b", "A1", "A2", "B1", "B2"};
    for (std::size_t i = 0; i < paths.size(); ++i)
    {
        paths[i] = args.take(names[i]);
    }
    double margin = 1.10;
    double trim = 0.01;
    while (!args.done())
    {
        const std::string option = args.take("an option");
        if (option == "--margin")
        {
            margin = args.number("--margin");
        }
        else if (option == "--trim")
        {
            trim = args.number("--trim");
        }
        else
        {
            throw usage_error{std::format("rule: unknown option '{}'", option)};
        }
    }
    // A trim of half the pixels or more would leave out the very differences the rule is there to find
    if (!(trim >= 0 && trim <= 0.5) || !(margin >= 1))
    {
        throw usage_error{"rule: --trim is from 0 to 0.5, --margin at least 1"};
    }
    std::vector<image> images;
    for (const std::string& path : paths)
    {
        images.push_back(load_pfm(path));
        check_same_size(images.front(), images.back(), "rule");
    }
    const pair_rms t1 = pair_noise(images[0], images[1], trim);
    const pair_rms t2 = pair_noise(images[2], images[3], trim);
    const pair_rms ra = pair_noise(images[4], images[5], trim);
    const pair_rms rb = pair_noise(images[6], images[7], trim);
    const double bound = margin * std::sqrt((ra.trimmed * ra.trimmed + rb.trimmed * rb.trimmed) / 2);
    const double bound_full = margin * std::sqrt((ra.full * ra.full + rb.full * rb.full) / 2);
    std::println("controls: R_A {:.9g} (untrimmed {:.9g}), R_B {:.9g} (untrimmed {:.9g})", ra.trimmed, ra.full,
                 rb.trimmed, rb.full);
    std::println("bound: {} · sqrt((R_A² + R_B²)/2) = {:.9g} (untrimmed {:.9g})", margin, bound, bound_full);
    const bool pass1 = t1.trimmed <= bound;
    const bool pass2 = t2.trimmed <= bound;
    std::println("test 1: {:.9g} (untrimmed {:.9g}), {:.4f} of the bound: {}", t1.trimmed, t1.full, t1.trimmed / bound,
                 pass1 ? "PASS" : "FAIL");
    std::println("test 2: {:.9g} (untrimmed {:.9g}), {:.4f} of the bound: {}", t2.trimmed, t2.full, t2.trimmed / bound,
                 pass2 ? "PASS" : "FAIL");
    return pass1 && pass2 ? 0 : 1;
}

// --- same and png-same ----------------------------------------------------------------------------------------------

// The distance of two floats in units in the last place: the number of floats between them. A float's bits, read as
// an integer, count up with its value for positive numbers; the negative ones are mapped below zero so that the count
// runs on through 0 (-0 and +0 are 0 apart, the smallest numbers of either sign 2).
std::int64_t ulps_apart(float a, float b)
{
    const auto ordered = [](float f) {
        const auto bits = static_cast<std::int64_t>(std::bit_cast<std::uint32_t>(f));
        return bits < 0x80000000LL ? bits : 0x80000000LL - bits;
    };
    return std::abs(ordered(a) - ordered(b));
}

int run_same(arguments& args)
{
    const std::string path1 = args.take("the first PFM file");
    const std::string path2 = args.take("the second PFM file");
    std::vector<rect> rects;
    std::int64_t ulps = 0;
    double max_share = 0;
    while (!args.done())
    {
        const std::string option = args.take("an option");
        if (option == "--rect")
        {
            rects.push_back(args.rectangle());
        }
        else if (option == "--ulps")
        {
            ulps = args.integer("--ulps");
        }
        else if (option == "--max-share")
        {
            max_share = args.number("--max-share");
        }
        else
        {
            throw usage_error{std::format("same: unknown option '{}'", option)};
        }
    }
    const image a = load_pfm(path1);
    const image b = load_pfm(path2);
    check_same_size(a, b, "same");
    if (rects.empty())
    {
        rects.push_back(whole(a));
    }
    std::int64_t total = 0;
    std::int64_t differ = 0;
    std::int64_t too_far = 0;
    std::int64_t largest = 0;
    std::int64_t identical_pixels = 0;
    for (const rect& r : rects)
    {
        check_rect(r, a);
        for (int y = r.y0; y < r.y1; ++y)
        {
            for (int x = r.x0; x < r.x1; ++x)
            {
                bool pixel_same = true;
                for (int k = 0; k < 3; ++k)
                {
                    // The PFM's floats, back from the doubles image holds them in (exact)
                    const auto fa = static_cast<float>(channel(a[x, y], k));
                    const auto fb = static_cast<float>(channel(b[x, y], k));
                    ++total;
                    if (std::bit_cast<std::uint32_t>(fa) != std::bit_cast<std::uint32_t>(fb))
                    {
                        pixel_same = false;
                        ++differ;
                        const std::int64_t apart = ulps_apart(fa, fb);
                        largest = std::max(largest, apart);
                        too_far += apart > ulps ? 1 : 0;
                    }
                }
                identical_pixels += pixel_same ? 1 : 0;
            }
        }
    }
    // The tolerance counts pixels: a pixel with any channel apart is one pixel apart
    const std::int64_t pixels = total / 3;
    const double share = static_cast<double>(pixels - identical_pixels) / static_cast<double>(pixels);
    std::println("{} and {}: {} of {} values differ, at most {} ulps; {} of {} pixels differ ({:.6f}%)", path1, path2,
                 differ, total, largest, pixels - identical_pixels, pixels, 100 * share);
    const bool pass = too_far == 0 && share <= max_share;
    std::println("{}", differ == 0 ? "identical: PASS" : pass ? "within the tolerance: PASS" : "different: FAIL");
    return pass ? 0 : 1;
}

int run_png_same(arguments& args)
{
    const std::string path1 = args.take("the first PNG file");
    const std::string path2 = args.take("the second PNG file");
    if (!args.done())
    {
        throw usage_error{std::format("png-same: unknown option '{}'", args.peek())};
    }
    const auto load = [](const std::string& path) {
        auto loaded = image::load(std::filesystem::path(path));
        if (!loaded)
        {
            throw usage_error{std::format("{}: {}", path, loaded.error())};
        }
        return std::move(*loaded);
    };
    const image a = load(path1);
    const image b = load(path2);
    check_same_size(a, b, "png-same");
    std::int64_t differ = 0;
    for (int y = 0; y < a.height(); ++y)
    {
        for (int x = 0; x < a.width(); ++x)
        {
            // stb_image decodes a byte the same way every time, so equal bytes give equal values
            const color& p = a[x, y];
            const color& q = b[x, y];
            differ += p.x == q.x && p.y == q.y && p.z == q.z ? 0 : 1;
        }
    }
    std::println("{} and {}: {} of {} pixels differ: {}", path1, path2, differ,
                 static_cast<std::int64_t>(a.width()) * a.height(), differ == 0 ? "PASS" : "FAIL");
    return differ == 0 ? 0 : 1;
}

// --- error ----------------------------------------------------------------------------------------------------------

// relMSE's ε: a dark value's error counts relative to 0.01 rather than to its own near-0 value (the denoising
// literature's convention since Rousselle et al. 2011); the tiles' drift uses the same
constexpr double relative_epsilon = 0.01;
// The references are clamped as the denoiser's beauty is (beauty_ceiling in src/gpu/denoise.cu)
constexpr double beauty_ceiling = 1e4;

double clamp_as_beauty(double v) { return std::clamp(v, 0.0, beauty_ceiling); }

// What error TEST DENOISED REF1 REF2 measures (see the comment at the top)
struct error_figures
{
    double variance = 0; // v: R's own, mean((REF1 − REF2)²)/4
    double mse_test = 0; // MSE'
    double mse_denoised = 0;
    double rel_test = 0; // relMSE'
    double rel_denoised = 0;
    std::array<double, 3> mean_reference{}; // of R, per channel
    std::array<double, 3> mean_denoised{};
    double tile_drift = 0; // the largest drift of a tile, by its size, with its sign
    int tile_x = 0;        // that tile, counted in tiles from the top left
    int tile_y = 0;
    std::int64_t clamped = 0; // values of REF1 and REF2 that the clamp changed
};

error_figures error_of(const image& test, const image& denoised, const image& ref1, const image& ref2, int tile)
{
    error_figures f;
    double sum_v = 0;
    double sum_test = 0;
    double sum_denoised = 0;
    double sum_rel_test = 0;
    double sum_rel_denoised = 0;
    std::array<double, 3> sum_r{};
    std::array<double, 3> sum_d{};
    const int across = (test.width() + tile - 1) / tile;
    const int down = (test.height() + tile - 1) / tile;
    // Per tile: the sums of R's and DENOISED's channel averages, and its pixels
    std::vector<double> tile_r(static_cast<std::size_t>(across) * static_cast<std::size_t>(down));
    std::vector<double> tile_d(tile_r.size());
    std::vector<std::int64_t> tile_n(tile_r.size());
    for (int y = 0; y < test.height(); ++y)
    {
        for (int x = 0; x < test.width(); ++x)
        {
            double r_average = 0;
            double d_average = 0;
            for (std::size_t k = 0; k < 3; ++k)
            {
                const int c = static_cast<int>(k);
                const double a = channel(ref1[x, y], c);
                const double b = channel(ref2[x, y], c);
                const double ca = clamp_as_beauty(a);
                const double cb = clamp_as_beauty(b);
                f.clamped += (ca != a ? 1 : 0) + (cb != b ? 1 : 0);
                const double r = (ca + cb) / 2;
                const double v = (ca - cb) * (ca - cb) / 4; // this value's share of R's variance
                const double t = channel(test[x, y], c);
                const double d = channel(denoised[x, y], c);
                const double et = (t - r) * (t - r);
                const double ed = (d - r) * (d - r);
                const double scale = r * r + relative_epsilon;
                sum_v += v;
                sum_test += et;
                sum_denoised += ed;
                sum_rel_test += (et - v) / scale;
                sum_rel_denoised += (ed - v) / scale;
                sum_r[k] += r;
                sum_d[k] += d;
                r_average += r / 3;
                d_average += d / 3;
            }
            const std::size_t in_tile = static_cast<std::size_t>(y / tile) * static_cast<std::size_t>(across) +
                                        static_cast<std::size_t>(x / tile);
            tile_r[in_tile] += r_average;
            tile_d[in_tile] += d_average;
            tile_n[in_tile] += 1;
        }
    }
    const double pixels = static_cast<double>(test.width()) * test.height();
    const double values = 3 * pixels;
    f.variance = sum_v / values;
    f.mse_test = sum_test / values - f.variance;
    f.mse_denoised = sum_denoised / values - f.variance;
    f.rel_test = sum_rel_test / values;
    f.rel_denoised = sum_rel_denoised / values;
    for (std::size_t k = 0; k < 3; ++k)
    {
        f.mean_reference[k] = sum_r[k] / pixels;
        f.mean_denoised[k] = sum_d[k] / pixels;
    }
    for (std::size_t t = 0; t < tile_r.size(); ++t)
    {
        const double n = static_cast<double>(tile_n[t]);
        const double drift = (tile_d[t] / n - tile_r[t] / n) / (tile_r[t] / n + relative_epsilon);
        if (std::abs(drift) > std::abs(f.tile_drift))
        {
            f.tile_drift = drift;
            f.tile_x = static_cast<int>(t % static_cast<std::size_t>(across));
            f.tile_y = static_cast<int>(t / static_cast<std::size_t>(across));
        }
    }
    return f;
}

// The verdict's gain and what it is worth: relMSE'(TEST)/relMSE'(DENOISED) and S times it in samples, unbounded when
// the denoised error is within R's own noise (relMSE'(DENOISED) ≤ 0); and MSE'(TEST)/MSE'(DENOISED), printed without a
// verdict, unbounded the same way
struct error_gains
{
    bool passes = false;           // relMSE'(DENOISED) ≤ relMSE'(TEST)/G, multiplied out: an unbounded gain passes
    bool bounded = false;          // relMSE'(DENOISED) > 0: gain and equivalent_samples hold numbers
    double gain = 0;               // relMSE'(TEST)/relMSE'(DENOISED)
    double equivalent_samples = 0; // S · gain: the samples of TEST's kind that would have DENOISED's error
    bool mse_bounded = false;      // MSE'(DENOISED) > 0: mse_ratio holds a number
    double mse_ratio = 0;          // MSE'(TEST)/MSE'(DENOISED)
};

error_gains gains_of(const error_figures& f, double min_gain, double spp)
{
    error_gains g;
    g.passes = min_gain * f.rel_denoised <= f.rel_test;
    g.bounded = f.rel_denoised > 0;
    if (g.bounded)
    {
        g.gain = f.rel_test / f.rel_denoised;
        g.equivalent_samples = spp * g.gain;
    }
    g.mse_bounded = f.mse_denoised > 0;
    if (g.mse_bounded)
    {
        g.mse_ratio = f.mse_test / f.mse_denoised;
    }
    return g;
}

int run_error(arguments& args)
{
    const std::string test_path = args.take("TEST");
    const std::string denoised_path = args.take("DENOISED");
    const std::string ref1_path = args.take("REF1");
    const std::string ref2_path = args.take("REF2");
    double min_gain = 4;
    double max_drift = 0.02;
    int tile = 32;
    double spp = 16;
    while (!args.done())
    {
        const std::string option = args.take("an option");
        if (option == "--min-gain")
        {
            min_gain = args.number("--min-gain");
        }
        else if (option == "--max-drift")
        {
            max_drift = args.number("--max-drift");
        }
        else if (option == "--tiles")
        {
            tile = args.integer("--tiles");
        }
        else if (option == "--spp")
        {
            spp = args.number("--spp");
        }
        else
        {
            throw usage_error{std::format("error: unknown option '{}'", option)};
        }
    }
    if (!(min_gain > 0) || !(max_drift >= 0) || tile < 1 || !(spp > 0))
    {
        throw usage_error{"error: --min-gain and --spp are above 0, --max-drift at least 0, --tiles at least 1"};
    }
    const image test = load_pfm(test_path);
    const image denoised = load_pfm(denoised_path);
    const image ref1 = load_pfm(ref1_path);
    const image ref2 = load_pfm(ref2_path);
    check_same_size(test, denoised, "error");
    check_same_size(test, ref1, "error");
    check_same_size(test, ref2, "error");
    const error_figures f = error_of(test, denoised, ref1, ref2, tile);

    std::println("{} and {} against R = ({} + {})/2: {}×{}", test_path, denoised_path, ref1_path, ref2_path,
                 test.width(), test.height());
    std::println("  R's own variance v = mean((REF1 - REF2)²)/4 = {:.6e}; reference values clamped to [0, {}]: {}",
                 f.variance, beauty_ceiling, f.clamped);
    std::println("  noisy:    MSE' {:.6e}, relMSE' {:.6e}", f.mse_test, f.rel_test);
    std::println("  denoised: MSE' {:.6e}, relMSE' {:.6e}", f.mse_denoised, f.rel_denoised);
    const error_gains g = gains_of(f, min_gain, spp);
    const bool gain_ok = g.passes;
    if (g.bounded)
    {
        std::println("  gain relMSE'(noisy)/relMSE'(denoised): {:.4g}, at least {} needed: {}", g.gain, min_gain,
                     gain_ok ? "PASS" : "FAIL");
        std::println("  equivalent samples: {:.4g} ({} times the gain)", g.equivalent_samples, spp);
    }
    else
    {
        std::println("  gain relMSE'(noisy)/relMSE'(denoised): unbounded (the denoised error is within R's own "
                     "noise), at least {} needed: {}",
                     min_gain, gain_ok ? "PASS" : "FAIL");
        std::println("  equivalent samples: beyond what R can tell");
    }
    if (g.mse_bounded)
    {
        std::println("  MSE'(noisy)/MSE'(denoised), no verdict: {:.4g}", g.mse_ratio);
    }
    else
    {
        std::println("  MSE'(noisy)/MSE'(denoised), no verdict: unbounded (within R's own noise)");
    }
    constexpr std::array names{"red", "green", "blue"};
    bool drift_ok = true;
    std::string drifts;
    for (std::size_t k = 0; k < 3; ++k)
    {
        const double difference = f.mean_denoised[k] - f.mean_reference[k];
        drift_ok = drift_ok && std::abs(difference) <= max_drift * f.mean_reference[k];
        const std::string_view separator = k > 0 ? ", " : "";
        drifts += f.mean_reference[k] > 0
                      ? std::format("{}{} {:+.3f}%", separator, names[k], 100 * difference / f.mean_reference[k])
                      : std::format("{}{} {:+.3e} (R's mean is 0)", separator, names[k], difference);
    }
    std::println("  drift of the mean, denoised against R: {}; at most {}% allowed: {}", drifts, 100 * max_drift,
                 drift_ok ? "PASS" : "FAIL");
    std::println("  largest drift in tiles of {}×{}: {:+.3f}% at tile ({}, {}): columns {} to {}, rows {} to {}", tile,
                 tile, 100 * f.tile_drift, f.tile_x, f.tile_y, f.tile_x * tile,
                 std::min(test.width(), (f.tile_x + 1) * tile) - 1, f.tile_y * tile,
                 std::min(test.height(), (f.tile_y + 1) * tile) - 1);
    const bool pass = gain_ok && drift_ok;
    std::println("error: {}", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// --- self-test ------------------------------------------------------------------------------------------------------

// Small images with known numbers, written to a temporary folder and read back through the same code
int run_self_test()
{
    const std::filesystem::path folder =
        std::filesystem::temp_directory_path() /
        std::format("pfm-stats-self-test-{}", std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(folder);
    int failures = 0;
    const auto expect = [&failures](bool ok, std::string_view what) {
        std::println("{}: {}", what, ok ? "PASS" : "FAIL");
        failures += ok ? 0 : 1;
    };
    try
    {
        // a: 8×4, every value 1 but pixel (3, 2) = 3 in all channels. b: a + 0.5 in the left half.
        image a(8, 4);
        image b(8, 4);
        for (int y = 0; y < 4; ++y)
        {
            for (int x = 0; x < 8; ++x)
            {
                const double v = x == 3 && y == 2 ? 3 : 1;
                a[x, y] = color{v, v, v};
                b[x, y] = color{v + (x < 4 ? 0.5 : 0), v + (x < 4 ? 0.5 : 0), v + (x < 4 ? 0.5 : 0)};
            }
        }
        const auto save = [&folder](const image& img, std::string_view name) {
            const std::filesystem::path path = folder / name;
            if (!img.save_pfm(path))
            {
                throw usage_error{"could not write the self-test's images"};
            }
            return path.string();
        };
        const std::string pa = save(a, "a.pfm");
        const std::string pb = save(b, "b.pfm");
        const image la = load_pfm(pa);
        const image lb = load_pfm(pb);

        // The mean of a is (31 + 3)/32 = 1.0625; the left half of b is 1.5 but for the 3.5
        const channel_stats s = region_stats(la, whole(la), 3);
        expect(std::abs(s.mean - 34.0 / 32) < 1e-12 && s.min == 1 && s.max == 3, "stats: mean, min and max");
        const channel_stats left = region_stats(lb, {0, 0, 4, 4, {}}, 0);
        expect(std::abs(left.mean - (15 * 1.5 + 3.5) / 16) < 1e-12, "stats: the mean of a rectangle");
        const value_counts c = count_values(la, whole(la), 1.0, 1.0);
        expect(c.total == 96 && c.equal == 93 && c.above == 3, "stats: values equal to and above 1");

        // d = b - a is 0.5 in 16 pixels and 0 in 16: RMS √(16·0.25/32) = √0.125; trimming 1% of 32 drops nothing,
        // trimming 50% drops the 16 largest
        const pair_rms r = pair_noise(la, lb, 0.01);
        expect(r.full == std::sqrt(0.125) && r.trimmed == r.full, "noise: RMS, trimmed by 1%");
        expect(pair_noise(la, lb, 0.5).trimmed == 0, "noise: RMS, trimmed by 50%");

        expect(ulps_apart(1.0f, std::nextafter(1.0f, 2.0f)) == 1 && ulps_apart(-0.0f, 0.0f) == 0 &&
                   ulps_apart(-1e-45f, 1e-45f) == 2,
               "same: distance in ulps");
        expect(std::bit_cast<std::uint32_t>(static_cast<float>(channel(la[3, 2], 1))) ==
                   std::bit_cast<std::uint32_t>(3.0f),
               "same: floats read back exactly");

        // The commands' verdicts, as their exit codes (their own lines are printed above each one)
        const auto run = [](int (*command)(arguments&), std::vector<std::string> list) {
            arguments args(std::move(list));
            return command(args);
        };
        expect(run(run_stats, {pa, "--equal", "1", "--share", "0.96"}) == 0, "stats --equal: 93 of 96 values pass 96%");
        expect(run(run_stats, {pa, "--equal", "1", "--share", "0.97"}) == 1, "stats --equal: and fail 97%");
        expect(run(run_stats, {pa, "--max", "2"}) == 1, "stats --max: 3 values above 2 fail");
        expect(run(run_stats, {pa, "--rect", "4", "0", "7", "3", "--expect", "1", "0"}) == 0,
               "stats --rect (inclusive) --expect: the right half is 1");
        expect(run(run_same, {pa, pa}) == 0 && run(run_same, {pa, pb}) == 1, "same: identical, and different");
        expect(run(run_same, {pa, pb, "--rect", "4", "0", "7", "3"}) == 0, "same --rect: the right halves are");
        expect(run(run_rule, {pa, pb, pa, pb, pa, pb, pa, pb}) == 0, "rule: tests as noisy as the controls pass");
        expect(run(run_rule, {pa, pb, pa, pb, pa, pa, pa, pa}) == 1, "rule: tests noisier than the controls fail");

        // error, on 4×4 images of known values, every channel alike. REF1 is 1, REF2 1 but 1.5 at (0, 0): R is 1 but
        // 1.25 there, and v = (3 values of 0.5²/4)/48 = 1/256. TEST is R but 2 at (1, 1) and 0 at (2, 2): mean (TEST -
        // R)² = 6/48, MSE' = 31/256. DENOISED is R but 1.5 at (1, 1): MSE' = 0.75/48 - 1/256 = 3/256, an MSE' ratio of
        // 31/3 (printed without a verdict).
        // Its mean, 16.75/16 against R's 16.25/16, drifts by +3.08%; in tiles of 2×2 the top left one drifts by
        // (1.1875 - 1.0625)/(1.0625 + 0.01), the others not at all.
        const auto filled = [](double value) {
            image img(4, 4);
            for (int y = 0; y < 4; ++y)
            {
                for (int x = 0; x < 4; ++x)
                {
                    img[x, y] = color{value, value, value};
                }
            }
            return img;
        };
        const auto gray = [](double value) { return color{value, value, value}; };
        image r1 = filled(1);
        image r2 = filled(1);
        r2[0, 0] = gray(1.5);
        image t = filled(1);
        t[0, 0] = gray(1.25);
        t[1, 1] = gray(2);
        t[2, 2] = gray(0);
        image d = filled(1);
        d[0, 0] = gray(1.25);
        d[1, 1] = gray(1.5);
        const std::string pr1 = save(r1, "r1.pfm");
        const std::string pr2 = save(r2, "r2.pfm");
        const std::string pt = save(t, "t.pfm");
        const std::string pd = save(d, "d.pfm");
        const error_figures f = error_of(load_pfm(pt), load_pfm(pd), load_pfm(pr1), load_pfm(pr2), 2);
        expect(f.variance == 1.0 / 256 && f.mse_test == 31.0 / 256 && f.mse_denoised == 3.0 / 256 && f.clamped == 0,
               "error: R's variance and MSE' of both images");
        const double at_0_0 = -0.0625 / (1.5625 + 0.01); // ((x - R)² - (REF1 - REF2)²/4)/(R² + 0.01) at (0, 0)
        expect(std::abs(f.rel_test - 3 * (at_0_0 + 2 / 1.01) / 48) < 1e-12 &&
                   std::abs(f.rel_denoised - 3 * (at_0_0 + 0.25 / 1.01) / 48) < 1e-12,
               "error: relMSE' of both images");
        // The gain and what it is worth, as run_error prints them (gains_of): the relMSE' gain
        // (2/1.01 + at_0_0)/(0.25/1.01 + at_0_0), 16 times it in samples, and the MSE' ratio 31/3
        const double relative_gain = (2 / 1.01 + at_0_0) / (0.25 / 1.01 + at_0_0);
        const error_gains g = gains_of(f, 4, 16);
        expect(g.bounded && g.passes && std::abs(g.gain / relative_gain - 1) < 1e-12 &&
                   std::abs(g.equivalent_samples / (16 * relative_gain) - 1) < 1e-12,
               "error: the relMSE' gain and the equivalent samples, 16 times it");
        expect(g.mse_bounded && g.mse_ratio == 31.0 / 3, "error: the MSE' ratio 31/3");
        expect(f.mean_reference == std::array{16.25 / 16, 16.25 / 16, 16.25 / 16} &&
                   f.mean_denoised == std::array{16.75 / 16, 16.75 / 16, 16.75 / 16},
               "error: the means per channel");
        expect(std::abs(f.tile_drift - 0.125 / 1.0725) < 1e-12 && f.tile_x == 0 && f.tile_y == 0,
               "error: the largest drift in tiles of 2×2");
        // The references clamped as the denoiser's input: 2·10^4 in both counts as 10^4 (3 values each)
        image high = filled(1);
        high[3, 3] = gray(2e4);
        image top = filled(1);
        top[3, 3] = gray(1e4);
        const error_figures clamped = error_of(top, top, high, high, 32);
        expect(clamped.clamped == 6 && clamped.variance == 0 && clamped.mse_test == 0,
               "error: the references clamped to [0, 10^4]");
        // The verdicts: the relMSE' gain (2/1.01 + at_0_0)/(0.25/1.01 + at_0_0) = 9.34 passes 4 and 9 and fails 10,
        // which the MSE' gain 31/3 would pass; the drift of 3.08% fails 2% and passes 4%
        expect(run(run_error, {pt, pd, pr1, pr2}) == 1, "error: a mean that drifts by 3.08% fails 2%");
        expect(run(run_error, {pt, pd, pr1, pr2, "--max-drift", "0.04"}) == 0, "error: and passes 4%");
        expect(run(run_error, {pt, pd, pr1, pr2, "--max-drift", "0.04", "--min-gain", "9"}) == 0,
               "error: a relMSE' gain of 9.34 passes 9");
        expect(run(run_error, {pt, pd, pr1, pr2, "--max-drift", "0.04", "--min-gain", "10"}) == 1,
               "error: and fails 10, the gain measured in relMSE', not in MSE' (31/3)");
        // DENOISED equal to R exactly (1 but 1.25 at (0, 0)): its error lies below R's own noise, relMSE' < 0 and
        // MSE' < 0, an unbounded gain, which passes; its mean doesn't drift
        image same_as_r = filled(1);
        same_as_r[0, 0] = gray(1.25);
        const std::string pe = save(same_as_r, "r.pfm");
        const error_figures fe = error_of(load_pfm(pt), load_pfm(pe), load_pfm(pr1), load_pfm(pr2), 2);
        const error_gains ge = gains_of(fe, 4, 16);
        expect(fe.rel_denoised < 0 && !ge.bounded && ge.passes && !ge.mse_bounded,
               "error: DENOISED equal to R, relMSE' below 0: an unbounded gain that passes");
        expect(run(run_error, {pt, pe, pr1, pr2}) == 0, "error: and the command passes it");
        // Errors (exit code 2): sizes that differ, a NaN in an input
        const auto refused = [&run](std::vector<std::string> list) {
            try
            {
                run(run_error, std::move(list));
            }
            catch (const usage_error&)
            {
                return true;
            }
            return false;
        };
        image broken = filled(1);
        broken[3, 3] = gray(std::numeric_limits<double>::quiet_NaN());
        const std::string pn = save(broken, "nan.pfm");
        expect(refused({pt, pd, pr1, pa}) && refused({pn, pd, pr1, pr2}), "error: sizes that differ, a NaN");
    }
    catch (const usage_error& error)
    {
        std::println("self-test: {}", error.message);
        failures += 1;
    }
    std::error_code ignored;
    std::filesystem::remove_all(folder, ignored);
    std::println("{}", failures == 0 ? "self-test: PASS" : "self-test: FAIL");
    return failures == 0 ? 0 : 1;
}

// --- main -----------------------------------------------------------------------------------------------------------

void print_usage()
{
    std::println("Usage: PfmStats stats|noise|rule|same|png-same|error|self-test ... (see the comment at the top of "
                 "tests/pfm_stats.cpp)\n"
                 "  stats A.pfm [--rect X0 Y0 X1 Y1 [--expect V T]]... [--equal V [--share F]] [--max V]\n"
                 "  noise A1.pfm A2.pfm [--trim F] [--tiles N]\n"
                 "  rule T1a T1b T2a T2b A1 A2 B1 B2 [--margin M] [--trim F]\n"
                 "  same A.pfm B.pfm [--rect X0 Y0 X1 Y1]... [--ulps N --max-share F]\n"
                 "  png-same a.png b.png\n"
                 "  error TEST DENOISED REF1 REF2 [--min-gain 4] [--max-drift 0.02] [--tiles 32] [--spp 16]\n"
                 "  self-test\n"
                 "Exit code: 0 pass, 1 fail, 2 error.");
}

int main(int argc, char* argv[])
{
    std::vector<std::string> list(argv + std::min(argc, 2), argv + argc);
    const std::string command = argc > 1 ? argv[1] : "";
    arguments args(std::move(list));
    try
    {
        if (command == "stats")
        {
            return run_stats(args);
        }
        if (command == "noise")
        {
            return run_noise(args);
        }
        if (command == "rule")
        {
            return run_rule(args);
        }
        if (command == "same")
        {
            return run_same(args);
        }
        if (command == "png-same")
        {
            return run_png_same(args);
        }
        if (command == "error")
        {
            return run_error(args);
        }
        if (command == "self-test")
        {
            return run_self_test();
        }
        print_usage();
        return command == "-h" || command == "--help" ? 0 : 2;
    }
    catch (const usage_error& error)
    {
        std::println("Error: {}", error.message);
        return 2;
    }
    catch (const std::exception& error) // the file system's, or out of memory
    {
        std::println("Error: {}", error.what());
        return 2;
    }
}
