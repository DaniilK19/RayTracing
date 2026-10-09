// Module random: random numbers, the basis of the Monte Carlo method that ray tracing relies on.
export module random;

import std;

// Mersenne Twister generator with 64-bit output. Not exported: only the functions below are visible outside.
inline std::mt19937_64& engine()
{
    static std::mt19937_64 generator;
    return generator;
}

// Random number in [0, 1).
export inline double random_double()
{
    return std::uniform_real_distribution<double>(0.0, 1.0)(engine());
}

// Random number in [min, max).
export inline double random_double(double min, double max)
{
    return min + (max - min) * random_double();
}
