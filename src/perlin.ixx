// Module perlin: Perlin noise, a smooth "random" pattern for marble, clouds, wood.
export module perlin;

import std;
import random;
import vec3;

// Perlin noise is a function of a point in space that looks random but changes smoothly,
// and always returns the same value for the same point. How it works:
//   1. Every node of the integer lattice holds a random unit vector, the gradient.
//   2. Point p lies in a lattice cube. Each of the cube's 8 corners contributes dot(gradient, p − corner):
//      the noise grows in the gradient's direction and is zero at the nodes themselves.
//   3. The contributions are blended by trilinear interpolation. The weights are smoothed by the Hermite curve
//      3t² − 2t³: with plain linear interpolation the grid would be visible at the cube boundaries.
// Storing a vector for every node of an infinite lattice is impossible, so there are only 256 vectors, and the vector
// of node (i, j, k) is chosen via three random permutations of 0..255: perm_x[i] ^ perm_y[j] ^ perm_z[k].
export class perlin
{
public:
    perlin()
    {
        for (vec3& v : random_vectors_)
        {
            v = unit_vector(vec3::random(-1, 1));
        }
        generate_permutation(perm_x_);
        generate_permutation(perm_y_);
        generate_permutation(perm_z_);
    }

    // Noise at point p: a number roughly from -1 to 1.
    double noise(const point3& p) const
    {
        // The lattice cube index and the position of p inside it (fractional parts of the coordinates, from 0 to 1)
        const int i = static_cast<int>(std::floor(p.x));
        const int j = static_cast<int>(std::floor(p.y));
        const int k = static_cast<int>(std::floor(p.z));
        const double u = p.x - std::floor(p.x);
        const double v = p.y - std::floor(p.y);
        const double w = p.z - std::floor(p.z);

        // Smoothed weights: the Hermite curve has zero derivative at its ends, so the noise smoothly
        // passes from cube to cube
        const double uu = u * u * (3 - 2 * u);
        const double vv = v * v * (3 - 2 * v);
        const double ww = w * w * (3 - 2 * w);

        double accum = 0;
        for (int di = 0; di < 2; ++di)
        {
            for (int dj = 0; dj < 2; ++dj)
            {
                for (int dk = 0; dk < 2; ++dk)
                {
                    // & 255 is the remainder of division by 256, correct for negative indices too
                    // (since C++20 negative numbers are guaranteed to be two's complement)
                    const vec3& gradient =
                        random_vectors_[perm_x_[(i + di) & 255] ^ perm_y_[(j + dj) & 255] ^ perm_z_[(k + dk) & 255]];
                    const vec3 offset{u - di, v - dj, w - dk}; // from the cube corner to p
                    accum += (di * uu + (1 - di) * (1 - uu)) * (dj * vv + (1 - dj) * (1 - vv)) *
                             (dk * ww + (1 - dk) * (1 - ww)) * dot(gradient, offset);
                }
            }
        }
        return accum;
    }

    // Turbulence: a sum of depth noise layers, each next one twice as fine and twice as weak,
    // like small ripples on big waves. The absolute value of the sum gives sharp folds where it crosses zero.
    double turbulence(const point3& p, int depth) const
    {
        double accum = 0;
        point3 scaled = p;
        double weight = 1;
        for (int layer = 0; layer < depth; ++layer)
        {
            accum += weight * noise(scaled);
            weight *= 0.5;
            scaled *= 2;
        }
        return std::fabs(accum);
    }

    // The tables, for code that copies a scene into its own format: the GPU program
    static constexpr int point_count = 256;
    inline const std::array<vec3, point_count>& random_vectors() const { return random_vectors_; }
    inline const std::array<int, point_count>& perm_x() const { return perm_x_; }
    inline const std::array<int, point_count>& perm_y() const { return perm_y_; }
    inline const std::array<int, point_count>& perm_z() const { return perm_z_; }

private:
    std::array<vec3, point_count> random_vectors_;
    std::array<int, point_count> perm_x_;
    std::array<int, point_count> perm_y_;
    std::array<int, point_count> perm_z_;

    // Random permutation of 0..255: the Fisher–Yates shuffle. Go from the end and swap
    // each element with a random one among those not yet touched (including itself).
    static void generate_permutation(std::array<int, point_count>& perm)
    {
        std::ranges::iota(perm, 0); // 0, 1, 2, ..., 255 (std::ranges::iota — C++23)
        for (int i = point_count - 1; i > 0; --i)
        {
            const int target = random_int(0, i);
            std::swap(perm[i], perm[target]);
        }
    }
};
