// The statistics of GpuTests, on the host in double: Pearson's χ² test with its p-value, the normal distribution's
// two-sided p-value, and Šidák's threshold for a family of tests. Plain C++, for gpu_tests.cpp and the test files.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace gpu_test
{
// Q(a, x) = Γ(a, x) / Γ(a), the regularized upper incomplete gamma function, as in Numerical Recipes (3rd ed., §6.2):
// the series of P = 1 - Q below x = a + 1, where it converges fast, and Lentz's continued fraction of Q above, where
// it does. The p-value of χ² with k degrees of freedom is Q(k/2, χ²/2). The continued fraction keeps the tiny p-values
// of a test that must reject (10^-18 and below) accurate; 1 - P could only give 0 there.
inline double regularized_gamma_q(double a, double x)
{
    if (!(a > 0) || !(x >= 0))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 0)
    {
        return 1;
    }
    // e^-x x^a / Γ(a), in logarithms: for a of thousands the powers and Γ overflow
    const double prefactor = std::exp(-x + a * std::log(x) - std::lgamma(a));
    constexpr double epsilon = 1e-15;
    if (x < a + 1)
    {
        double term = 1 / a;
        double sum = term;
        for (int n = 1; n < 100000; ++n)
        {
            term *= x / (a + n);
            sum += term;
            if (std::abs(term) < std::abs(sum) * epsilon)
            {
                break;
            }
        }
        return std::clamp(1 - sum * prefactor, 0.0, 1.0);
    }
    constexpr double tiny = 1e-300;
    double b = x + 1 - a;
    double c = 1 / tiny;
    double d = 1 / b;
    double h = d;
    for (int i = 1; i < 100000; ++i)
    {
        const double an = -i * (i - a);
        b += 2;
        d = an * d + b;
        d = std::abs(d) < tiny ? tiny : d;
        c = b + an / c;
        c = std::abs(c) < tiny ? tiny : c;
        d = 1 / d;
        const double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1) < epsilon)
        {
            break;
        }
    }
    return std::clamp(prefactor * h, 0.0, 1.0);
}

// The probability that χ² with `dof` degrees of freedom is at least `statistic`
inline double chi2_p_value(double statistic, int dof) { return regularized_gamma_q(dof / 2.0, statistic / 2.0); }

// P(|Z| ≥ |z|) for a standard normal Z
inline double normal_two_sided_p(double z) { return std::erfc(std::abs(z) / std::sqrt(2.0)); }

// The p-value each of T independent tests must reach for the family to fail by chance with probability `alpha`:
// 1 - (1 - alpha)^(1/T) (Šidák), a little less strict than Bonferroni's alpha / T
inline double sidak_threshold(double alpha, int tests)
{
    return tests > 0 ? 1 - std::pow(1 - alpha, 1.0 / tests) : alpha;
}

struct chi2_outcome
{
    double statistic = 0;
    int dof = 0;
    double p = 0;
    int bins = 0;         // after pooling
    int pooled_cells = 0; // cells whose expected count was below the minimum, pooled into one bin
    std::int64_t samples = 0;
};

// Pearson's test of observed counts against the probabilities of their cells. Cells expected to hold fewer than
// min_expected samples are pooled into one bin, as the χ² approximation of the statistic's distribution needs counts
// that aren't tiny; if that bin is still below the minimum, it joins the smallest of the others. A cell of probability
// 0 that holds a sample makes the statistic infinite and p = 0, wherever it was pooled: that sample is impossible.
inline chi2_outcome chi2_test(const std::vector<std::uint64_t>& observed, const std::vector<double>& probabilities,
                              double min_expected = 5)
{
    chi2_outcome out;
    const std::size_t cells = std::min(observed.size(), probabilities.size());
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < cells; ++i)
    {
        total += observed[i];
    }
    out.samples = static_cast<std::int64_t>(total);
    const auto n = static_cast<double>(total);
    std::vector<std::pair<double, double>> bins; // (observed, expected)
    double pooled_observed = 0;
    double pooled_expected = 0;
    bool impossible = false;
    for (std::size_t i = 0; i < cells; ++i)
    {
        const double expected = n * std::max(0.0, probabilities[i]);
        impossible = impossible || (!(probabilities[i] > 0) && observed[i] > 0);
        if (expected < min_expected)
        {
            pooled_observed += static_cast<double>(observed[i]);
            pooled_expected += expected;
            ++out.pooled_cells;
        }
        else
        {
            bins.emplace_back(static_cast<double>(observed[i]), expected);
        }
    }
    if (out.pooled_cells > 0)
    {
        if (pooled_expected < min_expected && !bins.empty())
        {
            auto smallest = std::ranges::min_element(bins, {}, &std::pair<double, double>::second);
            smallest->first += pooled_observed;
            smallest->second += pooled_expected;
        }
        else
        {
            bins.emplace_back(pooled_observed, pooled_expected);
        }
    }
    out.bins = static_cast<int>(bins.size());
    out.dof = out.bins - 1;
    for (const auto& [o, e] : bins)
    {
        if (e > 0)
        {
            out.statistic += (o - e) * (o - e) / e;
        }
        else if (o > 0)
        {
            out.statistic = std::numeric_limits<double>::infinity();
        }
    }
    // A test of fewer than two bins tests nothing: p is NaN, which passes no check, a must-reject one neither
    if (impossible)
    {
        out.statistic = std::numeric_limits<double>::infinity();
        out.p = 0;
    }
    else
    {
        out.p = out.dof > 0 ? chi2_p_value(out.statistic, out.dof) : std::numeric_limits<double>::quiet_NaN();
    }
    return out;
}

// A mean and its standard error from the sum and the sum of squares of n samples (the variance divides by n - 1)
struct mean_estimate
{
    double mean = 0;
    double standard_error = 0;
    std::int64_t samples = 0;
    std::int64_t left_out = 0; // samples the estimator didn't apply to (device_mean: a NaN), not in the mean
};

inline mean_estimate from_sums(double sum, double sum_of_squares, std::int64_t n)
{
    mean_estimate m;
    m.samples = n;
    if (n < 2)
    {
        m.mean = n == 1 ? sum : 0;
        return m;
    }
    const auto count = static_cast<double>(n);
    m.mean = sum / count;
    const double variance = std::max(0.0, (sum_of_squares - sum * m.mean) / (count - 1));
    m.standard_error = std::sqrt(variance / count);
    return m;
}
}
