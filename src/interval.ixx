// Module interval: a range of real numbers [min, max].
export module interval;

import std;

export inline constexpr double infinity = std::numeric_limits<double>::infinity();

export struct interval
{
    double min = +infinity; // the interval is empty by default: min > max
    double max = -infinity;

    constexpr bool surrounds(double x) const { return min < x && x < max; } // strictly inside
    constexpr double clamp(double x) const { return std::clamp(x, min, max); }
};
