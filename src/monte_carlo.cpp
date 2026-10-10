// MonteCarlo: the numerical experiments of the third book, "Ray Tracing: The Rest of Your Life".
// Before the ray tracer learns to choose its rays cleverly, the book studies Monte Carlo estimation on small
// problems whose exact answers are known, so the error of every estimate can be measured.
// Without arguments all experiments run in the book's order; or name the ones to run: MonteCarlo pi stratified
import std;
import random;

// An experiment in the list: a name for the command line, where it is in the book, a description for the help
// and the header, and the function that runs it (the same kind of table as scene_list in the scenes module).
struct experiment
{
    std::string_view name;
    std::string_view section; // of the third book, e.g. "2.1" (chapter 2, its first section)
    std::string_view description;
    void (*run)();
};

// Prints an estimate next to the exact value. The error alone says little: for the π estimator below an error of
// 0.01 is a great result for N = 100 samples and a sure sign of a bug for N = 10⁸. It has to be compared with
// the standard error SE, the typical error of the estimate. For an average of N independent samples SE = σ/√N
// (σ: the standard deviation of one sample), and by the central limit theorem such an average is nearly normally
// distributed, so about 68% of runs land within 1 SE of the exact value and 99.7% within 3 SE. |error/SE| < 3 is
// our test that an estimator is correct; a larger ratio points to a bug.
void print_estimate(std::string_view label, double estimate, double exact, double standard_error)
{
    const double error = estimate - exact;
    std::println("  {:10} {:.12f}  exact {:.12f}  error {:+.12f}  SE {:.12f}  error/SE {:+.2f}", label, estimate,
                 exact, error, standard_error, error / standard_error);
}

// --- Estimating π (book 3, chapter 2) --------------------------------------------------------------------------

// Is a random point of the square [-1, 1] × [-1, 1] inside the unit circle?
bool random_point_in_circle()
{
    const double x = random_double(-1, 1);
    const double y = random_double(-1, 1);
    return x * x + y * y < 1;
}

// The circle covers π·1² / 2² = π/4 of the square: the probability that a random point lands inside.
constexpr double p_inside = std::numbers::pi / 4;

// One sample of the π estimator is 4 (inside, probability p) or 0, so its mean is 4p = π and its variance
// (mean square minus squared mean) is 4²·p - (4p)² = 16·p(1 - p).
// Its standard deviation is σ = 4·√(p(1 - p)) ≈ 1.642.
// (Not constexpr: MSVC's std::sqrt isn't constexpr yet.)
const double pi_sample_sigma = 4 * std::sqrt(p_inside * (1 - p_inside));

// Book 3, section 2.1: N random points in the square; 4 × the fraction inside the circle approaches π.
// The same number is also (area of the square) × (fraction of it covered by the circle): an estimate of the
// circle's area, which for r = 1 is π. Chapter 3 builds on that view: Monte Carlo estimates areas (integrals).
void estimate_pi()
{
    seed_random(1);
    constexpr int n = 100'000; // the apostrophe only separates digits for readability (C++14)
    int inside_circle = 0;
    for (int i = 0; i < n; ++i)
    {
        if (random_point_in_circle())
        {
            ++inside_circle;
        }
    }
    std::println("  N = {} points", n);
    print_estimate("π", 4.0 * inside_circle / n, std::numbers::pi, pi_sample_sigma / std::sqrt(n));
}

// Book 3, section 2.2: a running estimate gets close to π quickly and then improves more and more slowly,
// the law of diminishing returns. How slowly is told by SE = σ/√N: 100 times more samples for an error
// only 10 times smaller.
// A single run can't show that law: its error at a given N is random too and may be close to zero by luck
// (see its error/SE column). So the run is repeated with different random numbers and we take the root mean
// square (RMS) of the errors. The mean squared error of an average of N samples is exactly σ²/N, so RMS·√N
// stays near σ for every N. The RMS of a finite number of runs is itself random, by about ±1/√(2·runs) of its
// value: with 100 runs ±7%, so we take 1000 runs (±2%) and stop at N = 10⁵ (10⁸ points in all).
void convergence()
{
    seed_random(2);
    constexpr int runs = 1000;
    constexpr std::array sample_counts{10, 100, 1'000, 10'000, 100'000};
    std::array<double, sample_counts.size()> first_run{}; // the estimates of run 0, like the book's printout
    std::array<double, sample_counts.size()> squared_error_sum{};
    for (int run = 0; run < runs; ++run)
    {
        int n = 0; // points so far
        int inside_circle = 0;
        for (std::size_t c = 0; c < sample_counts.size(); ++c)
        {
            // The run goes on up to the next N: the estimates at all N of a run come from one running count
            for (; n < sample_counts[c]; ++n)
            {
                if (random_point_in_circle())
                {
                    ++inside_circle;
                }
            }
            const double estimate = 4.0 * inside_circle / n;
            const double error = estimate - std::numbers::pi;
            squared_error_sum[c] += error * error;
            if (run == 0)
            {
                first_run[c] = estimate;
            }
        }
    }

    std::println("  exact π = {:.12f}, σ of one sample = {:.6f}\n", std::numbers::pi, pi_sample_sigma);
    std::println("  {:>9}  {:>15}  {:>16}  {:>8}  {:>18}  {:>8}  {:>7}", "N", "one run", "its error", "error/SE",
                 std::format("RMS of {} runs", runs), "σ/√N", "RMS·√N");
    // std::views::zip (C++23) walks several ranges side by side, like a zipper
    for (const auto [n, estimate, sum] : std::views::zip(sample_counts, first_run, squared_error_sum))
    {
        const double error = estimate - std::numbers::pi;
        const double standard_error = pi_sample_sigma / std::sqrt(n);
        const double rms = std::sqrt(sum / runs);
        std::println("  {:9}  {:15.12f}  {:+16.12f}  {:+8.2f}  {:18.6f}  {:8.6f}  {:7.4f}", n, estimate, error,
                     error / standard_error, rms, standard_error, rms * std::sqrt(n));
    }
}

// The book's listing 4 (pi.cc version three, section 2.3) as a function: sqrt_n × sqrt_n samples taken both ways.
// A "regular" point (the book's word) is random anywhere in the square. A stratified (jittered) one is random
// inside its own cell (i, j) of a sqrt_n × sqrt_n grid over the square: x = 2·(i + ξ)/sqrt_n - 1, where ξ is
// random in [0, 1), and the same for y with j.
struct pi_estimates
{
    double regular;
    double stratified;
};

pi_estimates estimate_pi_both_ways(int sqrt_n)
{
    int inside_circle = 0;
    int inside_circle_stratified = 0;
    for (int i = 0; i < sqrt_n; ++i)
    {
        for (int j = 0; j < sqrt_n; ++j)
        {
            if (random_point_in_circle())
            {
                ++inside_circle;
            }
            const double x = 2 * ((i + random_double()) / sqrt_n) - 1;
            const double y = 2 * ((j + random_double()) / sqrt_n) - 1;
            if (x * x + y * y < 1)
            {
                ++inside_circle_stratified;
            }
        }
    }
    const double n = static_cast<double>(sqrt_n) * sqrt_n;
    return {4.0 * inside_circle / n, 4.0 * inside_circle_stratified / n};
}

// Area of the part of the rectangle [0, a] × [0, b] (a, b ≥ 0) that lies inside the unit circle.
double area_in_circle(double a, double b)
{
    a = std::min(a, 1.0);
    b = std::min(b, 1.0);
    if (a * a + b * b <= 1)
    {
        return a * b; // the corner (a, b) is inside, so the whole rectangle is
    }
    // The arc y = √(1 - x²) crosses the top side y = b at x = c. Left of c the rectangle is inside the circle,
    // right of c (up to a) the region ends at the arc: area = c·b + ∫ from c to a of √(1 - x²) dx.
    const double c = std::sqrt(1 - b * b);
    const auto antiderivative = [](double x) { return (x * std::sqrt(1 - x * x) + std::asin(x)) / 2; };
    return c * b + antiderivative(a) - antiderivative(c);
}

// The exact standard error of the stratified estimate, to compare the measured errors with.
// The point of a cell lands in the circle with probability f = (area of the cell inside the circle) / (cell area).
// The cells are independent, so the variances of their contributions 4/N·[inside] add up:
// Var = Σ (4/N)²·f(1 - f). A cell wholly inside (f = 1) or outside (f = 0) the circle adds nothing: only the
// cells the circle passes through, fewer than 4·√N of the N, are random at all. So Var ∝ √N/N² and
// SE ∝ N^(-3/4), while random points give σ/√N ∝ N^(-1/2). That's the "better asymptotic rate" the book mentions.
struct stratified_error
{
    double standard_error;
    int boundary_cells; // cells the circle passes through
};

stratified_error stratified_standard_error(int sqrt_n)
{
    // The grid is symmetric about both axes, so we go through the cells of the quadrant [0, 1]² only and count
    // each one four times. This needs an even sqrt_n: then x = 0 and y = 0 are grid lines.
    const int m = sqrt_n / 2; // the circle's radius, in cell sides
    const double side = 2.0 / sqrt_n;
    const double n = static_cast<double>(sqrt_n) * sqrt_n;
    double variance = 0;
    int boundary_cells = 0;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            // A cell whose far corner (i + 1, j + 1) is in the circle lies wholly inside it, f = 1; one whose near
            // corner (i, j) is out lies wholly outside, f = 0. The test counts in cell sides, with integers, because
            // some corners lie exactly on the circle: for √N = 10 the corner (3, 4), as 3² + 4² = 5². In doubles
            // that corner is (0.6, 0.8), and its x² + y² can come out as 1.0000000000000002, which would count
            // a cell that only touches the circle as one it passes through.
            if ((i + 1) * (i + 1) + (j + 1) * (j + 1) <= m * m || i * i + j * j >= m * m)
            {
                continue;
            }
            const double x0 = i * side;
            const double y0 = j * side;
            const double x1 = x0 + side;
            const double y1 = y0 + side;
            // Inclusion-exclusion: the cell [x0, x1] × [y0, y1] is [0, x1] × [0, y1] minus [0, x0] × [0, y1] and
            // [0, x1] × [0, y0], plus [0, x0] × [0, y0], which was subtracted twice
            const double f = (area_in_circle(x1, y1) - area_in_circle(x0, y1) - area_in_circle(x1, y0) +
                              area_in_circle(x0, y0)) / (side * side);
            variance += 4 * (4 / n) * (4 / n) * f * (1 - f);
            boundary_cells += 4;
        }
    }
    return {std::sqrt(variance), boundary_cells};
}

// Book 3, section 2.3: stratified samples. The RMS error over many runs for both kinds of points shows how each
// one falls with N, and the exact standard errors predict it. Then one run of the book's listing 4.
void stratified()
{
    seed_random(3);
    struct grid
    {
        int sqrt_n;
        int runs; // fewer for the biggest grid: each run there takes 10⁶ points of each kind
    };
    std::println("  {:>7}  {:>8}  {:>5}  {:>14}  {:>10}  {:>15}  {:>10}  {:>14}", "√N", "N", "runs",
                 "regular: RMS", "σ/√N", "stratified: RMS", "exact SE", "boundary cells");
    for (const auto [sqrt_n, runs] : {grid{10, 1000}, grid{100, 1000}, grid{1000, 50}})
    {
        double squared_error_sum = 0;
        double squared_error_sum_stratified = 0;
        for (int run = 0; run < runs; ++run)
        {
            const pi_estimates estimates = estimate_pi_both_ways(sqrt_n);
            const double error = estimates.regular - std::numbers::pi;
            const double error_stratified = estimates.stratified - std::numbers::pi;
            squared_error_sum += error * error;
            squared_error_sum_stratified += error_stratified * error_stratified;
        }
        const double rms = std::sqrt(squared_error_sum / runs);
        const double rms_stratified = std::sqrt(squared_error_sum_stratified / runs);
        const stratified_error exact = stratified_standard_error(sqrt_n);
        std::println("  {:7}  {:8}  {:5}  {:14.7f}  {:10.7f}  {:15.7f}  {:10.7f}  {:14}", sqrt_n, sqrt_n * sqrt_n,
                     runs, rms, pi_sample_sigma / sqrt_n, rms_stratified, exact.standard_error, exact.boundary_cells);
    }
    std::println("  100× more points: regular errors fall 10×, stratified ones about 30× (N^(-3/4)).\n");

    constexpr int sqrt_n = 1000;
    std::println("  The book's listing 4 (pi.cc version three), one run with N = {}×{}:", sqrt_n, sqrt_n);
    const pi_estimates estimates = estimate_pi_both_ways(sqrt_n);
    print_estimate("regular", estimates.regular, std::numbers::pi, pi_sample_sigma / sqrt_n);
    print_estimate("stratified", estimates.stratified, std::numbers::pi,
                   stratified_standard_error(sqrt_n).standard_error);
}

// --- The list of experiments and the command line --------------------------------------------------------------

constexpr std::array experiments{
    experiment{"pi", "2.1", "estimate π from random points in a square", estimate_pi},
    experiment{"convergence", "2.2", "the law of diminishing returns: the error falls as 1/√N", convergence},
    experiment{"stratified", "2.3", "one random point in each cell of a grid (jittering) vs points anywhere",
               stratified},
};

void print_usage()
{
    std::println("Usage: MonteCarlo [experiment...]\n\n"
                 "Experiments from the third book, \"Ray Tracing: The Rest of Your Life\" (section, description):");
    for (const experiment& e : experiments)
    {
        std::println("  {:12} {:4} {}", e.name, e.section, e.description);
    }
    std::println(R"(  Without arguments all of them run, in the book's order.

Options:
  -h, --help   show this help)");
}

int main(int argc, char* argv[])
{
    // Check all arguments before running anything: a typo at the end shouldn't wait for the experiments before it.
    // drop(1) skips argv[0], the program name.
    std::vector<const experiment*> chosen;
    bool help = false;
    for (const std::string_view arg : std::span(argv, static_cast<std::size_t>(argc)) | std::views::drop(1))
    {
        if (arg == "-h" || arg == "--help")
        {
            help = true;
            continue;
        }
        // The projection &experiment::name: find compares the names, as find_scene does with the scenes
        const auto it = std::ranges::find(experiments, arg, &experiment::name);
        if (it == experiments.end())
        {
            std::println("Error: unknown {} '{}'\n", arg.starts_with('-') ? "option" : "experiment", arg);
            print_usage();
            return 1;
        }
        chosen.push_back(&*it);
    }
    if (help)
    {
        print_usage();
        return 0;
    }

#ifndef NDEBUG // NDEBUG is defined in Release and RelWithDebInfo
    std::println("This is a Debug build: the experiments run about 10 times slower than in Release.\n");
#endif

    if (chosen.empty())
    {
        for (const experiment& e : experiments)
        {
            chosen.push_back(&e);
        }
    }

    for (const experiment* e : chosen)
    {
        // Every experiment seeds the random number generator itself: its output is the same on every run and
        // doesn't depend on which experiments ran before it
        std::println("{} (section {}): {}", e->name, e->section, e->description);
        e->run();
        std::println("");
    }
}
