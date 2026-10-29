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
    std::println("Usage: PfmStats stats|noise|rule|same|png-same|self-test ... (see the comment at the top of "
                 "tests/pfm_stats.cpp)\n"
                 "  stats A.pfm [--rect X0 Y0 X1 Y1 [--expect V T]]... [--equal V [--share F]] [--max V]\n"
                 "  noise A1.pfm A2.pfm [--trim F] [--tiles N]\n"
                 "  rule T1a T1b T2a T2b A1 A2 B1 B2 [--margin M] [--trim F]\n"
                 "  same A.pfm B.pfm [--rect X0 Y0 X1 Y1]... [--ulps N --max-share F]\n"
                 "  png-same a.png b.png\n"
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
