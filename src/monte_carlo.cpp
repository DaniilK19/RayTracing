// MonteCarlo: the numerical experiments of the third book, "Ray Tracing: The Rest of Your Life".
// Before the ray tracer learns to choose its rays cleverly, the book studies Monte Carlo estimation on small
// problems whose exact answers are known, so the error of every estimate can be measured.
// Without arguments all experiments but sphere-plot run in the book's order; or name the ones to run:
// MonteCarlo pi stratified
import std;
import hittable;
import pdf;
import quad;
import random;
import vec3;

// An experiment in the list: a name for the command line, where it is in the book, a description for the help
// and the header, and the function that runs it (the same kind of table as scene_list in the scenes module).
struct experiment
{
    std::string_view name;
    std::string_view section; // of the third book, e.g. "2.1" (chapter 2, its first section), or "4" (chapter 4,
                              // which has no sections)
    std::string_view description;
    void (*run)();
    bool in_all = true; // runs when no experiment is named on the command line
};

// Prints an estimate next to the exact value. The error alone says little: for the π estimator below an error of
// 0.01 is a great result for N = 100 samples and a sure sign of a bug for N = 10⁸. It has to be compared with
// the standard error SE, the typical error of the estimate. For an average of N independent samples SE = σ/√N
// (σ: the standard deviation of one sample), and by the central limit theorem such an average is nearly normally
// distributed, so about 68% of runs land within 1 SE of the exact value and 99.7% within 3 SE. |error/SE| < 3 is
// our test that an estimator is correct; a larger ratio points to a bug.
// An estimator can also have SE = 0 (no random error at all, section 3.7); then there is no ratio to print.
void print_estimate(std::string_view label, double estimate, double exact, double standard_error)
{
    const double error = estimate - exact;
    std::print("  {:10} {:.12f}  exact {:.12f}  error {:+.12f}  SE {:.12f}", label, estimate, exact, error,
               standard_error);
    if (standard_error > 0)
    {
        std::print("  error/SE {:+.2f}", error / standard_error);
    }
    std::println("");
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

// --- One-dimensional Monte Carlo integration (book 3, chapter 3) -----------------------------------------------

// The mean of random samples y, with its standard error. For π a sample was 4 or 0, so σ followed from p = π/4.
// Usually σ is unknown: σ² = E[y²] - E[y]², and E[y²] is another integral, as hard as the one we estimate. But the
// same samples estimate it too, as the mean of their squares. (Two remarks for the curious. The subtraction loses
// digits when σ is tiny next to the mean; Welford's algorithm avoids that, but the σ we estimate here are about as
// large as the means. And this σ² comes out (N - 1)/N times too small on average, which dividing by N - 1 instead
// of N would fix; at the N = 10⁴ and 10⁶ where we use it, that is negligible.)
class sample_mean
{
public:
    void add(double y)
    {
        ++count_;
        sum_ += y;
        sum_of_squares_ += y * y;
    }

    int count() const { return count_; }
    double mean() const { return sum_ / count_; }
    double mean_of_squares() const { return sum_of_squares_ / count_; }

    // Rounding can push the difference a hair below 0 when all samples are (nearly) equal: σ is 0 then
    double standard_deviation() const { return std::sqrt(std::max(mean_of_squares() - mean() * mean(), 0.0)); }
    double standard_error() const { return standard_deviation() / std::sqrt(count_); }

private:
    int count_ = 0;
    double sum_ = 0;
    double sum_of_squares_ = 0;
};

// Section 3.1: the average of f over [a, b] is ∫ f(x) dx / (b - a). So the integral is (b - a) × the average of f,
// and the average can be estimated from random points: one sample is y = (b - a)·f(x) with x uniform in [a, b].
sample_mean integrate_uniformly(double (*f)(double), double a, double b, int n)
{
    sample_mean samples;
    while (samples.count() < n)
    {
        const double x = random_double(a, b);
        // random_double() can return exactly 0 (very rarely), and at x = 0 ln(sin x) is -∞. Such a point is drawn
        // again: leaving out a single point of [a, b] doesn't change the probability of any interval.
        if (x == a)
        {
            continue;
        }
        samples.add((b - a) * f(x));
    }
    return samples;
}

constexpr double x_squared_integral = 8.0 / 3; // ∫ x² dx over [0, 2] = 2³/3

// sin⁵x has an antiderivative, but it takes a trick: sin⁵x = (1 - cos²x)²·sin x, and with u = cos x
// (du = -sin x dx): ∫ sin⁵x dx = -∫ (1 - 2u² + u⁴) du = -cos x + (2/3)·cos³x - (1/5)·cos⁵x.
double sin5_antiderivative(double x)
{
    const double c = std::cos(x);
    const double c3 = c * c * c;
    return -c + 2.0 / 3 * c3 - 1.0 / 5 * c3 * c * c;
}

// ln(sin x) has no antiderivative made of elementary functions. Its integral over [0, 2] is -2·ln 2 - Cl₂(4)/2
// (Cl₂ is the Clausen function); the value was computed by numerical quadrature in two ways, which agree to 15 digits.
constexpr double ln_sin_integral = -1.102222388904956;

// Book 3, section 3.2 (listings 8 to 10): three integrals over [0, 2], from N = 10⁶ uniform samples each. Monte Carlo
// treats them all the same, whether the exact answer is easy (x²), takes some work (sin⁵x) or has no formula in
// elementary functions at all (ln(sin x)).
void integrate()
{
    seed_random(4);
    constexpr int n = 1'000'000;
    struct integral
    {
        std::string_view label;
        double (*f)(double);
        double exact;
    };
    // A lambda that captures nothing converts to a plain function pointer
    const std::array integrals{
        integral{"x²", [](double x) { return x * x; }, x_squared_integral},
        integral{"sin⁵ x", [](double x) { return std::pow(std::sin(x), 5); },
                 sin5_antiderivative(2) - sin5_antiderivative(0)},
        integral{"ln(sin x)", [](double x) { return std::log(std::sin(x)); }, ln_sin_integral},
    };
    std::println("  N = {} uniform samples in [0, 2] each; the σ for the SE is estimated from the same samples", n);
    for (const auto& [label, f, exact] : integrals)
    {
        const sample_mean samples = integrate_uniformly(f, 0, 2, n);
        print_estimate(label, samples.mean(), exact, samples.standard_error());
    }
}

// The function of the book's estimate_halfway.cc (section 3.5): p(x) = e^(-x/2π)·sin²x on [0, 2π], a damped wave.
double damped_sin2(double x)
{
    const double sin_x = std::sin(x);
    return std::exp(-x / (2 * std::numbers::pi)) * sin_x * sin_x;
}

// Its exact integral from 0 to x, to check the estimates. With a = 1/(2π) and sin²t = (1 - cos 2t)/2:
// ∫₀ˣ e^(-at) dt = (1 - e^(-ax))/a and ∫₀ˣ e^(-at)·cos 2t dt = (e^(-ax)·(2 sin 2x - a cos 2x) + a)/(a² + 4)
// (to check, differentiate e^(-at)·(2 sin 2t - a cos 2t): that gives (a² + 4)·e^(-at)·cos 2t).
double damped_sin2_integral(double x)
{
    constexpr double a = 1 / (2 * std::numbers::pi);
    const double e = std::exp(-a * x);
    return (1 - e) / (2 * a) - (e * (2 * std::sin(2 * x) - a * std::cos(2 * x)) + a) / (2 * (a * a + 4));
}

// The exact halfway point: where the integral from 0 reaches half of the area. The integral grows with x (p ≥ 0), so
// bisection finds that point: halve the interval again and again, keeping the half where the integral crosses
// area/2. 60 halvings shrink [0, 2π] to 2π/2⁶⁰ ≈ 5·10⁻¹⁸, below the spacing of doubles near 2.
double exact_halfway_point(double area)
{
    double low = 0;
    double high = 2 * std::numbers::pi;
    for (int i = 0; i < 60; ++i)
    {
        const double middle = (low + high) / 2;
        if (damped_sin2_integral(middle) < area / 2)
        {
            low = middle;
        }
        else
        {
            high = middle;
        }
    }
    return low;
}

// Book 3, section 3.5 (listing 12): the point that splits the area under p(x) in half, found without integrating:
// sort the samples by x and add up their p(x) from the left until the running sum reaches half of the total; the x
// of that sample is the estimate. The book printed 2.016002314977, about 1 SE below the exact 2.038432: one run
// with other random numbers.
void halfway()
{
    seed_random(5);
    constexpr int n = 10'000;
    constexpr double length = 2 * std::numbers::pi; // of the interval [0, 2π]
    struct sample
    {
        double x;
        double p_x;
    };
    std::vector<sample> samples;
    samples.reserve(n);
    sample_mean p_values;
    for (int i = 0; i < n; ++i)
    {
        const double x = random_double(0, length);
        const double p_x = damped_sin2(x);
        samples.push_back({x, p_x});
        p_values.add(p_x);
    }

    // std::ranges::sort (C++20) takes a projection: &sample::x makes it compare the samples by their x.
    // {} stands for the default comparison, std::ranges::less.
    std::ranges::sort(samples, {}, &sample::x);
    const double half_of_total = p_values.mean() * n / 2;
    double running_sum = 0;
    double halfway_point = 0;
    for (const auto& [x, p_x] : samples)
    {
        running_sum += p_x;
        if (running_sum >= half_of_total)
        {
            halfway_point = x;
            break;
        }
    }

    // The average of p estimates ∫ p dx / 2π, and the area is 2π × the average (section 3.1)
    const double exact_area = damped_sin2_integral(length);
    std::println("  N = {} samples in [0, 2π]", n);
    print_estimate("average", p_values.mean(), exact_area / length, p_values.standard_error());
    print_estimate("area", length * p_values.mean(), exact_area, length * p_values.standard_error());

    // The standard error of the halfway point (beyond the book). Let D(t) be the sum of p(xᵢ) over the samples with
    // xᵢ ≤ t, minus half of the total; the estimate is where D crosses 0. At the exact point h each sample adds
    // +p(xᵢ)/2 to D (if xᵢ ≤ h) or -p(xᵢ)/2 (if not), on average 0, because half of the area lies left of h. So D(h)
    // is 0 give or take ½·√(N·E[p²]). Moving t past h changes D by about N·p(h)/2π per unit of t (N/2π samples per
    // unit, each adding about p(h)), so the crossing is off by about ½·√(N·E[p²]) / (N·p(h)/2π):
    // SE = 2π·√E[p²] / (2·√N·p(h)), with E[p²] estimated from the samples and p(h) taken at the estimate.
    const double halfway_standard_error =
        length * std::sqrt(p_values.mean_of_squares()) / (2 * std::sqrt(n) * damped_sin2(halfway_point));
    print_estimate("halfway", halfway_point, exact_halfway_point(exact_area), halfway_standard_error);
}

// A PDF on [0, 2] for importance sampling: its density p(x), and the inverse of its cumulative distribution function
// P(x) = ∫₀ˣ p (section 3.6). For d uniform in [0, 1), x = ICD(d) has the density p.
struct pdf_on_0_2
{
    std::string_view name;
    std::string_view formula; // of p(x), for the table
    double (*density)(double x);
    double (*icd)(double d);
    double variance; // of one sample f(x)/p(x) for f = x²: E[(f/p)²] - I², where E[(f/p)²] = ∫ f²/p dx over [0, 2]
};

// I² = (8/3)² = 64/9. The quadratic PDF's E[(f/p)²] is 64/9 too, and written the same way, so its variance is
// exactly 0 (not a rounding error like 10⁻¹⁶, whose square root would be a σ of 10⁻⁸)
constexpr double i_squared = 64.0 / 9;
constexpr double sqrt2 = std::numbers::sqrt2; // a short name for √2, for the table below

constexpr std::array importance_pdfs{
    // Uniform (listing 13): p = 1/2, P(x) = x/2, ICD(d) = 2d. E[(f/p)²] = ∫ x⁴/(1/2) dx = 2·2⁵/5 = 64/5
    pdf_on_0_2{"uniform", "1/2", [](double /*x*/) { return 0.5; }, [](double d) { return 2 * d; },
               64.0 / 5 - i_squared},
    // Two halves (listing 11), section 3.5's crude approximation of x/2: √2 splits x/2 into two halves of
    // probability 1/2, and each half gets uniform samples, so p = (1/2)/√2 below √2 and (1/2)/(2 - √2) above it.
    // The book's f(d) uses d only to choose the half and draws a second random number for x inside it. Rescaling d
    // does both with one number, and that is the ICD of this p (its P is linear on each half).
    // E[(f/p)²] = 2√2·∫₀^√2 x⁴ dx + 2(2 - √2)·∫_√2^2 x⁴ dx = 16/5 + (144 - 80√2)/5 = 32 - 16√2
    pdf_on_0_2{"two halves", "1/(2√2) | 1/(4-2√2)",
               [](double x) { return x < sqrt2 ? 0.5 / sqrt2 : 0.5 / (2 - sqrt2); },
               [](double d) { return d < 0.5 ? 2 * sqrt2 * d : sqrt2 + (2 - sqrt2) * (2 * d - 1); },
               32 - 16 * sqrt2 - i_squared},
    // Linear (listing 14): p = x/2, P(x) = x²/4, ICD(d) = √(4d). E[(f/p)²] = ∫ x⁴/(x/2) dx = ∫ 2x³ dx = 8
    pdf_on_0_2{"linear", "x/2", [](double x) { return x / 2; }, [](double d) { return std::sqrt(4 * d); },
               8 - i_squared},
    // Quadratic (listing 15): p = 3x²/8, P(x) = x³/8, and P(x) = d gives ICD(d) = ∛(8d) = 2·∛d. The book writes
    // 8·d^(1/3), which spreads the samples over [0, 8], mostly outside [0, 2]. Its estimate can't show the mistake, as
    // f/p = x²/(3x²/8) = 8/3 for every x. E[(f/p)²] = (8/3)² = 64/9: σ = 0, every sample is the exact answer.
    pdf_on_0_2{"quadratic", "3x²/8", [](double x) { return 3 * x * x / 8; },
               [](double d) { return std::cbrt(8 * d); }, 64.0 / 9 - i_squared},
};

// One estimate of ∫ x² dx over [0, 2] from n samples with the density p: the mean of f(x)/p(x) (section 3.7).
// Dividing by p weights the samples: where p is high they come often and each counts less, where p is low they come
// rarely and each counts more.
struct x_squared_estimate
{
    double value;
    double largest_x; // of the samples: all of them must lie in [0, 2]
};

x_squared_estimate estimate_x_squared(const pdf_on_0_2& pdf, int n)
{
    sample_mean samples;
    double largest_x = 0;
    while (samples.count() < n)
    {
        const double d = random_double();
        // d = 0 gives x = 0, where the linear and quadratic densities are 0 and f/p is 0/0; the book skips it too
        if (d == 0)
        {
            continue;
        }
        const double x = pdf.icd(d);
        largest_x = std::max(largest_x, x);
        samples.add(x * x / pdf.density(x));
    }
    return {samples.mean(), largest_x};
}

// Book 3, section 3.7 (listings 11 and 13 to 15): importance sampling. The same integral, ∫ x² dx over [0, 2] = 8/3,
// with samples drawn from four PDFs. All of them give the right answer on average, and the closer p is to the shape
// of f = x², the smaller σ and the errors. σ is known exactly here (the variances above), so the RMS error of many
// runs can be compared with σ/√N.
void importance()
{
    seed_random(6);
    constexpr int n = 1'000'000;
    std::println("  ∫ x² dx over [0, 2] as the mean of f(x)/p(x) = x²/p(x), with x = ICD(d) drawn with the density p.");
    std::println("  One run of N = {} samples per PDF:", n);
    for (const pdf_on_0_2& pdf : importance_pdfs)
    {
        print_estimate(pdf.name, estimate_x_squared(pdf, n).value, x_squared_integral, std::sqrt(pdf.variance / n));
    }
    std::println("  The quadratic p is proportional to f, so every sample f/p is 8/3 (but for its last bits), σ = 0.\n"
                 "  Its tiny error comes from rounding in the running sum. The sum grows to millions,\n"
                 "  and each 8/3 added to it loses its low bits.");
    std::println("  The book's last listing takes a single sample of it: {:.12f}\n",
                 estimate_x_squared(importance_pdfs.back(), 1).value);

    constexpr int runs = 1000;
    constexpr int small_n = 100;
    std::println("  The RMS error of {} runs with N = {} samples each, next to σ/√N:", runs, small_n);
    std::println("  {:10}  {:19}  {:>8}  {:>8}  {:>8}  {:>9}", "PDF", "p(x)", "σ of f/p", "σ/√N", "RMS", "largest x");
    for (const pdf_on_0_2& pdf : importance_pdfs)
    {
        double squared_error_sum = 0;
        double largest_x = 0;
        for (int run = 0; run < runs; ++run)
        {
            const x_squared_estimate estimate = estimate_x_squared(pdf, small_n);
            const double error = estimate.value - x_squared_integral;
            squared_error_sum += error * error;
            largest_x = std::max(largest_x, estimate.largest_x);
        }
        const double sigma = std::sqrt(pdf.variance);
        std::println("  {:10}  {:19}  {:8.6f}  {:8.6f}  {:8.6f}  {:9.6f}", pdf.name, pdf.formula, sigma,
                     sigma / std::sqrt(small_n), std::sqrt(squared_error_sum / runs), largest_x);
    }
}

// --- Monte Carlo integration on the sphere of directions (book 3, chapter 4) -----------------------------------

// A direction in 3D is a point on the unit sphere, and a set of directions is a patch of the sphere's surface. The
// patch's area is the set's solid angle, measured in steradians (sr); all directions together make 4π sr, the area
// of the sphere. So a PDF of directions is a density per steradian, and it integrates to 1 over the sphere: the
// uniform one is 1/(4π) everywhere. An integral over directions (like the light that reaches a point from all
// around) is estimated as before, as the mean of f(d)/p(d), only now with random directions d.

constexpr double pi = std::numbers::pi; // a short name for the formulas below

// The integrand of chapter 4: f(d) = cos²θ, where θ is the angle between the direction d and the z axis.
// For a unit vector d, cos θ = d·(0, 0, 1) = d.z, so f(d) = d.z², without computing θ.
double cos_squared(const vec3& d) { return d.z * d.z; }

// Its exact integral: the book gives 4π/3 and asks the reader to check it ("check me!"). In spherical coordinates
// (θ from the z axis, φ around it) a small patch dθ × dφ covers the solid angle dω = sin θ dθ dφ: near the poles
// the circles of constant θ are small. So
// ∫ cos²θ dω = ∫₀^2π dφ · ∫₀^π cos²θ sin θ dθ = 2π · [-cos³θ/3]₀^π = 2π · 2/3 = 4π/3.
constexpr double cos_squared_integral = 4 * pi / 3;

// A way to draw random directions: the function that draws them and their density p(d) per steradian, which must
// describe exactly how that function spreads them. (Section 10.1 makes such a pair a class, pdf.)
struct direction_sampler
{
    std::string_view name;
    vec3 (*generate)();
    double (*density)(const vec3& d);
    double variance; // of one sample f(d)/p(d) for the integrand f it is used with, worked out below
};

// The book's listing 16: random_unit_vector() and p = 1/(4π), the density of the directions it returns, named
// uniform_sphere_density in the pdf module. (The ray tracer's isotropic scatters along them as they are, with this
// density; adding one to the normal gives another density, cos θ/π, which the first two books' lambertian used.)
// By Archimedes' hat-box theorem the zone of the unit sphere between the planes z = a and z = b has the area
// 2π(b - a), like that of the cylinder around the sphere, so the z of a uniform direction is uniform in [-1, 1].
// A sample is f/p = 4π·z². E[z²] = 1/3 gives the mean 4π/3 once more, and E[z⁴] = 1/5 the variance
// (4π)²·(1/5 - 1/9) = 64π²/45: σ ≈ 3.747.
constexpr direction_sampler uniform_directions{"uniform", random_unit_vector,
                                               [](const vec3& /*d*/) { return uniform_sphere_density; },
                                               64 * pi * pi / 45};

// θ uniform in [0, π) and φ uniform in [0, 2π): a uniform point (θ, φ) of a π × 2π rectangle, turned into a unit
// vector. The directions are not uniform: the whole edge θ = 0 of the rectangle goes to a single point, the pole.
vec3 random_polar_direction()
{
    const double theta = pi * random_double();
    const double phi = 2 * pi * random_double();
    const double sin_theta = std::sin(theta);
    return {sin_theta * std::cos(phi), sin_theta * std::sin(phi), std::cos(theta)};
}

// Beyond the book, which says any coordinates will do, as long as p is a density over the sphere. With θ and φ
// uniform the density is 1/π · 1/(2π) = 1/(2π²) per unit of θ·φ, and a patch dθ × dφ covers sin θ dθ dφ steradians,
// so per steradian p = 1/(2π²·sin θ), where sin θ = √(x² + y²) for d = (x, y, z): high near the poles, where the
// directions crowd. f/p = 2π²·sin θ·cos²θ averages to (1/π)·2π²·∫₀^π cos²θ sin θ dθ = 4π/3, the right answer, and
// with ∫₀^π sin²θ·cos⁴θ dθ = π/16 (Wallis: ∫ cos⁴ - ∫ cos⁶ over [0, π] = 3π/8 - 5π/16)
// E[(f/p)²] = (1/π)·4π⁴·π/16 = π⁴/4: variance π⁴/4 - (4π/3)², σ ≈ 2.609.
constexpr direction_sampler polar_directions{
    "θ, φ", random_polar_direction,
    [](const vec3& d) { return 1 / (2 * pi * pi * std::sqrt(d.x * d.x + d.y * d.y)); },
    pi * pi * pi * pi / 4 - cos_squared_integral * cos_squared_integral};

// The same directions with the sin θ forgotten: 1/(2π²) taken for a density per steradian. Then f/p = 2π²·cos²θ, and
// E[(f/p)²] = (1/π)·4π⁴·∫₀^π cos⁴θ dθ = 3π⁴/2, while its mean is π² (see sphere() below): variance π⁴/2.
constexpr direction_sampler polar_directions_without_sin{
    "no sin θ", random_polar_direction, [](const vec3& /*d*/) { return 1 / (2 * pi * pi); }, pi * pi * pi * pi / 2};

// The mean of n samples f(d)/p(d), with the directions d drawn by the sampler.
double estimate_direction_integral(double (*f)(const vec3& d), const direction_sampler& sampler, int n)
{
    double sum = 0;
    for (int i = 0; i < n; ++i)
    {
        const vec3 d = sampler.generate();
        sum += f(d) / sampler.density(d);
    }
    return sum / n;
}

// Book 3, chapter 4 (listing 16): ∫ cos²θ dω over the sphere of directions, as the mean of f(d)/p(d) over random
// directions: the book's uniform ones, then uniform θ and φ, with the right p and with a wrong one.
void sphere()
{
    seed_random(7);
    constexpr int n = 1'000'000;
    std::println("  ∫ cos²θ dω over the unit sphere as the mean of f(d)/p(d) = d.z²/p(d), p per steradian, N = {}:", n);
    for (const direction_sampler& sampler : {uniform_directions, polar_directions})
    {
        print_estimate(sampler.name, estimate_direction_integral(cos_squared, sampler, n), cos_squared_integral,
                       std::sqrt(sampler.variance / n));
    }
    std::println("  Uniform θ and φ crowd the directions at the poles, where cos²θ is large, so σ is smaller:\n"
                 "  more samples where f is large, as with importance sampling in section 3.7.\n");

    // With a wrong p the estimate still converges, only to another number. Here the samples 2π²·cos²θ, with θ
    // uniform in [0, π), average to 2π²·1/2 = π² = ∫∫ cos²θ dθ dφ over the θ-φ rectangle.
    std::println("  Without the sin θ, p = 1/(2π²) is not a density per steradian: "
                 "over the sphere it integrates to 2/π.\n"
                 "  The estimate still converges, but to π² = ∫∫ cos²θ dθ dφ, the integral over the θ-φ rectangle:");
    const double estimate = estimate_direction_integral(cos_squared, polar_directions_without_sin, n);
    const double standard_error = std::sqrt(polar_directions_without_sin.variance / n);
    print_estimate(polar_directions_without_sin.name, estimate, cos_squared_integral, standard_error);
    print_estimate("vs π²", estimate, pi * pi, standard_error);
}

// --- Generating random directions (book 3, chapter 7) ----------------------------------------------------------

// random_unit_vector() draws points in a cube until one lands inside the ball: the rejection method. The inversion
// method of section 3.6 makes a direction from two random numbers r1, r2 in [0, 1), with no loop. If the density per
// steradian depends only on θ, the angle from the z axis, p = f(θ), then φ is uniform: φ = 2π·r1. θ has the density
// 2π·f(θ)·sin θ (the band between θ and θ + dθ covers 2π·sin θ·dθ steradians); setting its cumulative distribution
// equal to r2 and solving for cos θ gives z, and sin θ = √(1 - z²). random_cosine_direction() in vec3 is made so too.

// Uniform on the sphere, f = 1/(4π) (section 7.1): the density of θ is sin θ / 2, its cumulative distribution
// (1 - cos θ)/2 = r2, so z = 1 - 2·r2 is uniform in (-1, 1] (Archimedes' theorem in chapter 4 above says the same),
// and sin θ = 2·√(r2·(1 - r2)). The same distribution as random_unit_vector()'s.
vec3 random_sphere_direction()
{
    const double r1 = random_double();
    const double r2 = random_double();
    const double phi = 2 * pi * r1;
    const double sin_theta = 2 * std::sqrt(r2 * (1 - r2));
    return {std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, 1 - 2 * r2};
}

// Where sphere-plot writes its points: the console, or the file given with -o. The file holds only the points, what
// a plotting tool reads: on the console the experiment's header comes first, and PowerShell 5.1's > would save the
// output as UTF-16. main opens the file before any experiment runs, so a file that can't be written stops the
// program at once.
std::optional<std::string_view> sphere_plot_file_name; // none: the console
std::ofstream sphere_plot_file;

// Book 3, section 7.1 (listing 23, sphere_plot.cc): 200 directions as "x y z" lines, the input of a 3D scatter plot.
// Turned around in the plot, the points cover the sphere evenly.
void sphere_plot()
{
    seed_random(8);
    constexpr int n = 200;
    // std::ofstream is a kind of std::ostream (derived from it), so a reference to ostream can refer to either
    std::ostream& out = sphere_plot_file.is_open() ? sphere_plot_file : std::cout;
    for (int i = 0; i < n; ++i)
    {
        const vec3 d = random_sphere_direction();
        std::println(out, "{:9.6f} {:9.6f} {:9.6f}", d.x, d.y, d.z);
    }
    if (sphere_plot_file.is_open())
    {
        std::println("  {} points written to {}", n, *sphere_plot_file_name);
    }
}

// The integrand of sections 7.2 and 7.3: cos³θ = d.z³ over the hemisphere z > 0, where both samplers below draw
// their directions. Its exact integral: ∫₀^2π dφ · ∫₀^(π/2) cos³θ sin θ dθ = 2π · [-cos⁴θ/4]₀^(π/2) = 2π/4 = π/2.
double cos_cubed(const vec3& d) { return d.z * d.z * d.z; }

constexpr double cos_cubed_integral = pi / 2;

// Uniform on the hemisphere, f = 1/(2π) (section 7.2): the density of θ is sin θ, its cumulative distribution
// 1 - cos θ = r2, so z = 1 - r2, uniform in (0, 1]. (The book's cos_cubed.cc needs only z and skips φ.)
vec3 random_hemisphere_direction()
{
    const double r1 = random_double();
    const double r2 = random_double();
    const double phi = 2 * pi * r1;
    const double z = 1 - r2;
    const double sin_theta = std::sqrt(1 - z * z);
    return {std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, z};
}

// Listing 24 (cos_cubed.cc): a sample is f/p = 2π·z³ with z uniform in (0, 1]. E[z³] = 1/4 gives the mean π/2, and
// E[z⁶] = 1/7 the variance 4π²/7 - π²/4 = 9π²/28: σ ≈ 1.781.
constexpr direction_sampler uniform_hemisphere{"uniform", random_hemisphere_direction,
                                               [](const vec3& /*d*/) { return 1 / (2 * pi); }, 9 * pi * pi / 28};

// Listing 26 (cos_density.cc): random_cosine_direction() from vec3 (listing 25, section 7.3) with its density cos θ/π,
// which is cosine_density with the z axis as the normal. A sample is f/p = π·cos²θ = π·(1 - r2), uniform in (0, π]:
// the mean π/2, the variance π²/12, σ ≈ 0.907.
constexpr direction_sampler cosine_hemisphere{"cosine", random_cosine_direction,
                                              [](const vec3& d) { return cosine_density({0, 0, 1}, d); }, pi * pi / 12};

// Book 3, sections 7.2 and 7.3 (listings 24 and 26): ∫ cos³θ dω over the hemisphere, from uniform directions and
// from cosine-distributed ones. cos θ/π is closer to the shape of cos³θ than a constant, so σ is √(27/7) ≈ 1.96 times
// smaller: the same error takes 27/7 ≈ 3.9 times fewer samples (importance sampling, as in section 3.7).
// Each RMS of 1000 runs is itself random, by about ±2% (see convergence()), so the measured ratio of the two RMS
// errors scatters by about ±3% around 1.96.
void hemisphere()
{
    seed_random(9);
    constexpr int n = 1'000'000;
    constexpr std::array samplers{uniform_hemisphere, cosine_hemisphere};
    std::println("  ∫ cos³θ dω over the hemisphere z > 0 as the mean of f(d)/p(d) = d.z³/p(d), N = {}:", n);
    for (const direction_sampler& sampler : samplers)
    {
        print_estimate(sampler.name, estimate_direction_integral(cos_cubed, sampler, n), cos_cubed_integral,
                       std::sqrt(sampler.variance / n));
    }

    constexpr int runs = 1000;
    constexpr int small_n = 100;
    std::println("\n  The RMS error of {} runs with N = {} samples each, next to σ/√N:", runs, small_n);
    std::println("  {:7}  {:>8}  {:>8}  {:>8}", "p(d)", "σ of f/p", "σ/√N", "RMS");
    std::array<double, samplers.size()> rms_errors{};
    for (std::size_t i = 0; i < samplers.size(); ++i)
    {
        double squared_error_sum = 0;
        for (int run = 0; run < runs; ++run)
        {
            const double error = estimate_direction_integral(cos_cubed, samplers[i], small_n) - cos_cubed_integral;
            squared_error_sum += error * error;
        }
        rms_errors[i] = std::sqrt(squared_error_sum / runs);
        const double sigma = std::sqrt(samplers[i].variance);
        std::println("  {:7}  {:8.6f}  {:8.6f}  {:8.6f}", samplers[i].name, sigma, sigma / std::sqrt(small_n),
                     rms_errors[i]);
    }
    std::println("  With the cosine density the RMS error is {:.2f} times smaller (σ: {:.2f} times).",
                 rms_errors[0] / rms_errors[1], std::sqrt(samplers[0].variance / samplers[1].variance));
}

// --- Sampling lights directly (book 3, chapter 9) --------------------------------------------------------------

// The book warns that a wrong PDF still gives a reasonable-looking image, so here is a test beyond the book: are
// a lamp's random() and pdf_value() a matching pair, i.e. is p = pdf_value the density of the directions random()
// draws? With N directions from the point o (Ω is the solid angle the lamp covers, seen from o):
// (a) a density integrates to 1: the mean of 4π·p(d) over uniform directions d (density 1/(4π)) estimates ∫ p dω;
// (b) the mean of 1/p(d) over the lamp's own directions estimates ∫ (1/p)·p dω over the lamp, i.e. Ω;
// (c) random() never draws a direction with p = 0 (its weight 1/p would be infinite: the camera would drop it);
// (d) what the ray tracer does with them: the mean of h(d)/p(d) over the lamp's directions and the mean of
//     4π·h(d) over all the uniform directions, with 0 for each d that misses the lamp, both estimate ∫ h dω over
//     the lamp. (The misses must count: a mean over the hits only would estimate 4π·∫ h dω / Ω.) This is the
//     stricter test: (b) can pass even when the directions are spread wrongly over the lamp. h is the Lambertian
//     density around a tilted normal, so ∫ h dω is the fraction of the light a matte surface scatters that
//     reaches the lamp.
void check_light_pdf(const hittable& lamp, const point3& o, double solid_angle, int n)
{
    const vec3 normal = unit_vector(vec3{1, 2, 3});
    const auto h = [&normal](const vec3& d) { return cosine_density(normal, d); };

    sample_mean density;   // (a)
    sample_mean uniform_h; // (d)
    for (int i = 0; i < n; ++i)
    {
        const vec3 d = random_unit_vector();
        const double p = lamp.pdf_value(o, d);
        density.add(4 * pi * p);
        uniform_h.add(p > 0 ? 4 * pi * h(d) : 0.0);
    }

    sample_mean inverse_density; // (b)
    sample_mean lamp_h;          // (d)
    int zero_density = 0;        // (c)
    for (int i = 0; i < n; ++i)
    {
        const vec3 d = lamp.random(o);
        const double p = lamp.pdf_value(o, d);
        if (!(p > 0))
        {
            ++zero_density;
            continue;
        }
        inverse_density.add(1 / p);
        lamp_h.add(h(d) / p);
    }

    print_estimate("(a) ∫ p dω", density.mean(), 1, density.standard_error());
    print_estimate("(b) Ω", inverse_density.mean(), solid_angle, inverse_density.standard_error());
    std::println("  (c) directions drawn with p = 0: {} of {}", zero_density, n);
    const double difference = lamp_h.mean() - uniform_h.mean();
    const double standard_error = std::hypot(lamp_h.standard_error(), uniform_h.standard_error());
    std::println("  (d) ∫ h dω {:.12f}  uniform {:.12f}  difference {:+.12f}  SE {:.12f}  difference/SE {:+.2f}",
                 lamp_h.mean(), uniform_h.mean(), difference, standard_error, difference / standard_error);
}

// The solid angle of the square [-1, 1] × [-1, 1] in the plane z = 1, seen from o below it. A rectangle
// [0, x] × [0, y] seen from the height h above its corner (0, 0) covers F(x, y) = atan(xy / (h·√(x² + y² + h²)))
// steradians. With x and y measured from o, signed, any rectangle is a sum of four such ones with signs:
// inclusion-exclusion, as for the cells in stratified_standard_error.
double square_solid_angle(const point3& o)
{
    const double h = 1 - o.z;
    const auto f = [h](double x, double y) { return std::atan(x * y / (h * std::sqrt(x * x + y * y + h * h))); };
    const double x1 = -1 - o.x;
    const double x2 = 1 - o.x;
    const double y1 = -1 - o.y;
    const double y2 = 1 - o.y;
    return f(x2, y2) - f(x1, y2) - f(x2, y1) + f(x1, y1);
}

// Light sampling: section 9.1's density of a quad lamp, checked from two points. Seen from the middle, the square
// covers 2π/3 steradians, a third of the hemisphere (6 such squares make a cube around o).
void light_pdfs()
{
    seed_random(10);
    constexpr int n = 4'000'000;
    std::println("  p = pdf_value(o, d); N = {} uniform directions d and N drawn by the lamp's random(o).\n"
                 "  (a) the mean of 4π·p over the uniform d; (b) the mean of 1/p over the lamp's d;\n"
                 "  (d) the mean of h/p over the lamp's d, and of 4π·h over all the uniform d, with 0 where d\n"
                 "      misses the lamp (\"uniform\"); h(d) = max(0, cos θ)/π, θ the angle between d and (1, 2, 3)",
                 n);
    // A lamp needs no material here: only its shape is used, as in a scene's list of lights
    const quad square{point3{-1, -1, 1}, vec3{2, 0, 0}, vec3{0, 2, 0}, nullptr};
    for (const point3& o : {point3{0, 0, 0}, point3{0.5, 0.25, -0.5}})
    {
        const double solid_angle = square_solid_angle(o);
        std::println("\n  The square [-1, 1]² at z = 1 seen from ({}, {}, {}): Ω = {:.6f} sr", o.x, o.y, o.z,
                     solid_angle);
        check_light_pdf(square, o, solid_angle, n);
    }
}

// --- The list of experiments and the command line --------------------------------------------------------------

constexpr std::array experiments{
    experiment{"pi", "2.1", "estimate π from random points in a square", estimate_pi},
    experiment{"convergence", "2.2", "the law of diminishing returns: the error falls as 1/√N", convergence},
    experiment{"stratified", "2.3", "one random point in each cell of a grid (jittering) vs points anywhere",
               stratified},
    experiment{"integrate", "3.2", "integrals as averages of random samples: x², sin⁵ x and ln(sin x) over [0, 2]",
               integrate},
    experiment{"halfway", "3.5", "the point that halves the area under a curve, found by sorting the samples",
               halfway},
    experiment{"importance", "3.7", "∫ x² dx with four PDFs: the closer p is to f, the smaller the error", importance},
    experiment{"sphere", "4", "∫ cos²θ dω over the sphere of directions, with densities per steradian", sphere},
    // Designated initializers (C++20) name the members: .in_all = false says what the false is for
    experiment{.name = "sphere-plot",
               .section = "7.1",
               .description = "200 random directions as \"x y z\" lines, the input of a 3D scatter plot",
               .run = sphere_plot,
               .in_all = false},
    experiment{"cos-cubed", "7.2", "∫ cos³θ dω over the hemisphere: uniform vs cosine-distributed directions",
               hemisphere},
    experiment{"light-pdfs", "9.1",
               "beyond the book: does a lamp's pdf_value() describe the directions its random() draws?", light_pdfs},
};

void print_usage()
{
    std::println("Usage: MonteCarlo [experiment...] [options]\n\n"
                 "Experiments from the third book, \"Ray Tracing: The Rest of Your Life\" (section, description):");
    for (const experiment& e : experiments)
    {
        std::println("  {:12} {:4} {}", e.name, e.section, e.description);
    }
    std::println(R"(  Without arguments all of them run, in the book's order, except sphere-plot.

Options:
  -o, --output FILE  sphere-plot writes its points to FILE instead of the console: only the points,
                     the input of a plotting tool (on the console a header comes first)
  -h, --help         show this help)");
}

int main(int argc, char* argv[])
{
    // Check all arguments before running anything: a typo at the end shouldn't wait for the experiments before it
    const std::span<char* const> args(argv, static_cast<std::size_t>(argc));
    std::vector<const experiment*> chosen;
    bool help = false;
    for (std::size_t i = 1; i < args.size(); ++i) // args[0] is the program name
    {
        const std::string_view arg = args[i];
        if (arg == "-h" || arg == "--help")
        {
            help = true;
            continue;
        }
        if (arg == "-o" || arg == "--output")
        {
            if (i + 1 == args.size())
            {
                std::println("Error: {} needs a file name\n", arg);
                print_usage();
                return 1;
            }
            sphere_plot_file_name = args[++i];
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
        // An experiment named twice runs once: it seeds its own random numbers, so a second run would only print
        // the same again. std::ranges::contains is C++23.
        if (!std::ranges::contains(chosen, &*it))
        {
            chosen.push_back(&*it);
        }
    }
    if (help)
    {
        print_usage();
        return 0;
    }
    if (sphere_plot_file_name)
    {
        // Only sphere-plot writes to the file, and it never runs unless named. The projection &experiment::name
        // works on the pointers in chosen too: std::invoke turns it into e->name.
        if (!std::ranges::contains(chosen, std::string_view("sphere-plot"), &experiment::name))
        {
            std::println("Error: -o names the file for the points of sphere-plot: MonteCarlo sphere-plot -o FILE");
            return 1;
        }
        sphere_plot_file.open(std::string(*sphere_plot_file_name));
        if (!sphere_plot_file)
        {
            std::println("Error: can't write the file '{}'", *sphere_plot_file_name);
            return 1;
        }
    }

#ifndef NDEBUG // NDEBUG is defined in Release and RelWithDebInfo
    std::println("This is a Debug build: the experiments run about 10 times slower than in Release.\n");
#endif

    if (chosen.empty())
    {
        for (const experiment& e : experiments)
        {
            if (e.in_all)
            {
                chosen.push_back(&e);
            }
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
