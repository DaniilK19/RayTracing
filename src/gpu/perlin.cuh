// The GPU's Perlin noise (see perlin.ixx): the CPU's tables, copied, and its noise and turbulence functions.
#pragma once

#include <cstdint>

#include "renderer.h"
#include "vec3.cuh"

namespace gpu
{
// A gradient of the lattice: 16 bytes in float, read with one load when the whole struct is copied (the fourth number
// is padding; copying only g, a vec3 of Real's own alignment, would take three loads)
template <real_number Real>
struct alignas(4 * sizeof(Real)) perlin_gradient
{
    vec3<Real> g;
    Real unused; // padding, named
};

// The tables of one perlin object, copied from the CPU's. Its constructor drew them from the CPU's random numbers while
// the scene was built, and they decide what the marble looks like, so the GPU must never draw its own: a scene is
// always built by the CPU's code (see main_gpu.cpp). The permutations of 0..255 are stored as bytes: 768 bytes instead
// of 3 KB, and the whole table, 4.8 KB in float, stays in a multiprocessor's L1 cache, which every thread that hits a
// marble surface reads.
template <real_number Real>
struct perlin_table
{
    perlin_gradient<Real> random_vectors[perlin_point_count];
    std::uint8_t perm_x[perlin_point_count];
    std::uint8_t perm_y[perlin_point_count];
    std::uint8_t perm_z[perlin_point_count];
};

// perlin::noise: the noise at p, roughly from -1 to 1. The 8 corners of p's lattice cube each contribute
// dot(gradient, p - corner), blended by trilinear interpolation with weights smoothed by the Hermite curve
// 3t² - 2t³ (see perlin.ixx). As on the CPU: the weights are smoothed, but the offsets to the corners are the raw
// fractions (u - di), and the corners come in the order di, dj, dk.
template <real_number Real>
__device__ Real perlin_noise(const perlin_table<Real>& table, const point3<Real>& p)
{
    const Real fx = floor(p.x);
    const Real fy = floor(p.y);
    const Real fz = floor(p.z);
    const int i = static_cast<int>(fx);
    const int j = static_cast<int>(fy);
    const int k = static_cast<int>(fz);
    // The position of p inside its cube, each coordinate from 0 to 1
    const Real u = p.x - fx;
    const Real v = p.y - fy;
    const Real w = p.z - fz;
    const Real uu = u * u * (Real(3) - Real(2) * u);
    const Real vv = v * v * (Real(3) - Real(2) * v);
    const Real ww = w * w * (Real(3) - Real(2) * w);
    // Each axis needs the permutation entries of the cube's two lattice planes: 6 loads instead of the 24 of the CPU's
    // form, which looks them up for each corner. & 255 is the remainder of division by 256, for negative indices too.
    const int x0 = table.perm_x[i & 255];
    const int x1 = table.perm_x[(i + 1) & 255];
    const int y0 = table.perm_y[j & 255];
    const int y1 = table.perm_y[(j + 1) & 255];
    const int z0 = table.perm_z[k & 255];
    const int z1 = table.perm_z[(k + 1) & 255];
    Real accum = 0;
    // #pragma unroll copies the loop's body for every di, dj and dk, so they are constants in each copy, and the
    // choices below between the two planes of an axis cost nothing. (With arrays indexed by di, the arrays could end up
    // in local memory, which is slower than registers.)
#pragma unroll
    for (int di = 0; di < 2; ++di)
    {
        // The CPU's weight of the corners at di along x, di·uu + (1 - di)·(1 - uu), is exactly 1 - uu for di = 0 and
        // uu for di = 1
        const Real wx = di == 0 ? Real(1) - uu : uu;
        const int px = di == 0 ? x0 : x1;
#pragma unroll
        for (int dj = 0; dj < 2; ++dj)
        {
            const Real wy = dj == 0 ? Real(1) - vv : vv;
            const int py = dj == 0 ? y0 : y1;
#pragma unroll
            for (int dk = 0; dk < 2; ++dk)
            {
                const Real wz = dk == 0 ? Real(1) - ww : ww;
                const int pz = dk == 0 ? z0 : z1;
                const perlin_gradient<Real> corner = table.random_vectors[px ^ py ^ pz];
                const vec3<Real> offset{u - Real(di), v - Real(dj), w - Real(dk)}; // from the corner to p
                accum += wx * wy * wz * dot(corner.g, offset);
            }
        }
    }
    return accum;
}

// perlin::turbulence: depth layers of noise, each twice as fine and half as strong as the one before, and the absolute
// value of their sum (not of each layer's)
template <real_number Real>
__device__ Real turbulence(const perlin_table<Real>& table, point3<Real> p, int depth)
{
    Real accum = 0;
    Real weight = 1;
    for (int layer = 0; layer < depth; ++layer)
    {
        accum += weight * perlin_noise(table, p);
        weight *= Real(0.5);
        p = Real(2) * p; // exact: doubling changes only a float's exponent
    }
    return fabs(accum);
}
}
