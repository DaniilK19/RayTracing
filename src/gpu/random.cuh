// The GPU's random numbers (see random.ixx): one small generator for every sample, and random directions.
//
// The CPU ray tracer gives each image row its own generator, an mt19937_64 seeded with the render's seed and the row's
// number (see camera::render), because each of its threads renders whole rows. The GPU runs a thread per pixel, and
// every sample of a pixel in a separate pass (see renderer.cu), so here every sample, one camera ray with all its
// bounces, gets its own generator, started from the round's seed, the pixel's index and the pass in the round (see
// session in renderer.h). Whatever
// order the GPU runs the threads in, and however the passes are grouped into launches, a sample gets the same random
// numbers, so the image is the same on every run. mt19937_64 doesn't fit this: its state takes 2.5 KB, and starting it
// takes thousands of operations, for a sample that needs a few dozen numbers.
#pragma once

#include <concepts>
#include <cstdint>

#include "vec3.cuh"

namespace gpu
{
// PCG32 (Melissa O'Neill, pcg-random.org), as her minimal C implementation: pcg32_srandom_r and pcg32_random_r. Its
// state is a 64-bit linear congruential generator, state · 6364136223846793005 + inc, whose low bits alone would be
// poor; the output is a permutation of the state that hides this ("XSH RR": an xorshift of the high bits, then a
// rotation by an amount the top 5 bits choose). It passes statistical test suites such as TestU01's BigCrush, with
// 16 bytes of state and a few instructions per number.
class pcg32
{
public:
    // pcg32_srandom_r: initstate chooses the starting point, initseq one of 2^63 sequences (each with its own odd inc)
    __device__ pcg32(std::uint64_t initstate, std::uint64_t initseq) : state_(0), inc_((initseq << 1u) | 1u)
    {
        next();
        state_ += initstate;
        next();
    }

    // pcg32_random_r: 32 random bits
    __device__ std::uint32_t next()
    {
        const std::uint64_t old_state = state_;
        state_ = old_state * 6364136223846793005ull + inc_;
        const auto xorshifted = static_cast<std::uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
        const auto rotation = static_cast<std::uint32_t>(old_state >> 59u);
        // A rotation right by `rotation` bits; O'Neill writes the left shift as (-rotation) & 31
        return (xorshifted >> rotation) | (xorshifted << ((32u - rotation) & 31u));
    }

private:
    std::uint64_t state_;
    std::uint64_t inc_;
};

// The finalizer of SplitMix64 (Steele, Lea and Flood, 2014): it scrambles a 64-bit number so that changing one input
// bit changes about half of the output bits. It is a bijection, so different inputs never give the same output: it
// turns neighboring numbers, such as pixel 7 and pixel 8, into unrelated ones.
__device__ inline std::uint64_t mix64(std::uint64_t z)
{
    z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31u);
}

// The generator of one sample: pass `pass` of pixel `pixel` (its index, row by row from the top) for the round's seed.
// The starting point and the sequence come from two different hashes of the three numbers, so two samples get the same
// numbers only if 128 bits of hashes coincide. (Two pcg32 sequences whose starting points and inc values are related
// in simple ways can give correlated numbers; after the hashes they are unrelated.)
__device__ inline pcg32 sample_generator(std::uint64_t seed, std::uint64_t pixel, std::uint64_t pass)
{
    const std::uint64_t initstate = mix64(mix64(mix64(seed) ^ pixel) ^ pass);
    const std::uint64_t initseq = mix64(mix64(mix64(seed ^ 0x9e3779b97f4a7c15ull) ^ pass) ^ pixel);
    return pcg32{initstate, initseq};
}

// A random number in [0, 1), like random_double(). A float has a 24-bit significand, so it takes the top 24 bits of one
// output: the numbers k / 2^24, every one exactly representable and equally likely. A double takes 53 bits from two
// outputs. `if constexpr` (C++17) compiles only the branch of the type at hand, so the float version has no doubles.
template <real_number Real>
__device__ Real random_real(pcg32& rng)
{
    if constexpr (std::same_as<Real, float>)
    {
        return static_cast<float>(rng.next() >> 8u) * 0x1p-24f; // a hexadecimal float literal: 1 · 2^-24
    }
    else
    {
        // Two statements: in one expression the order of the two calls would be unspecified
        const std::uint64_t high = rng.next();
        const std::uint64_t low = rng.next();
        return static_cast<double>((high << 21u) | (low >> 11u)) * 0x1p-53;
    }
}

// The random directions of vec3.ixx. They live here because on the GPU they need the sample's generator.
//
// The CPU draws random_unit_vector() and random_in_unit_disk() by rejection: points of a square or a cube until one
// falls inside the disk or the ball. On a GPU a loop that runs a random number of times is costly: the 32 threads of a
// warp execute every instruction together, so the warp keeps looping until its unluckiest thread is done, and the
// others wait. The formulas below draw the same distributions with a fixed amount of work.

// A unit vector uniformly distributed over all directions. On a sphere, z is uniform in [-1, 1] (Archimedes' hat-box
// theorem: slices of equal height have equal areas), and the angle φ around the z axis is uniform too. With z = 1 - 2u,
// the radius of the circle at height z is √(1 - z²) = √((1 - z)(1 + z)) = 2·√(u·(1 - u)), never NaN.
template <real_number Real>
__device__ vec3<Real> random_unit_vector(pcg32& rng)
{
    const Real u = random_real<Real>(rng);
    const Real phi_over_pi = Real(2) * random_real<Real>(rng); // φ = 2π·v
    Real sin_phi;
    Real cos_phi;
    sincospi(phi_over_pi, &sin_phi, &cos_phi);
    const Real r = Real(2) * sqrt(u * (Real(1) - u));
    return {r * cos_phi, r * sin_phi, Real(1) - Real(2) * u};
}

// A uniform point of the unit disk. The area within radius r grows as r², so r = √u spreads the points evenly
// (r = u would crowd them at the center).
template <real_number Real>
__device__ vec3<Real> random_in_unit_disk(pcg32& rng)
{
    const Real r = sqrt(random_real<Real>(rng));
    const Real theta_over_pi = Real(2) * random_real<Real>(rng);
    Real sin_theta;
    Real cos_theta;
    sincospi(theta_over_pi, &sin_theta, &cos_theta);
    return {r * cos_theta, r * sin_theta, Real(0)};
}

// A direction with the density cos θ / π around the z axis, as random_cosine_direction() in vec3.ixx (which explains
// the formulas): no loop there either.
template <real_number Real>
__device__ vec3<Real> random_cosine_direction(pcg32& rng)
{
    const Real r1 = random_real<Real>(rng);
    const Real r2 = random_real<Real>(rng);
    Real sin_phi;
    Real cos_phi;
    sincospi(Real(2) * r1, &sin_phi, &cos_phi);
    const Real sin_theta = sqrt(r2);
    return {cos_phi * sin_theta, sin_phi * sin_theta, sqrt(Real(1) - r2)};
}
}
