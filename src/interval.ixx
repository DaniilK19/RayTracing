// Module interval: a range of real numbers [min, max].
export module interval;

import std;

export inline constexpr double infinity = std::numeric_limits<double>::infinity();

export struct interval
{
    double min = +infinity; // the interval is empty by default: min > max
    double max = -infinity;

    constexpr double size() const { return max - min; }
    constexpr bool contains(double x) const { return min <= x && x <= max; } // including the ends
    constexpr bool surrounds(double x) const { return min < x && x < max; }  // strictly inside
    constexpr double clamp(double x) const { return std::clamp(x, min, max); }

    // The interval expanded by delta: half of delta on each side
    constexpr interval expand(double delta) const
    {
        const double padding = delta / 2;
        return {min - padding, max + padding};
    }
};

// The interval shifted by displacement
export constexpr interval operator+(const interval& ival, double displacement)
{
    return {ival.min + displacement, ival.max + displacement};
}

// The smallest interval containing both given ones
export constexpr interval enclose(const interval& a, const interval& b)
{
    return {std::min(a.min, b.min), std::max(a.max, b.max)};
}
