// GpuTests' device side: GPU memory for a test, and the four kernels every sampling test is made of, each over a
// functor the test file defines (a struct with a __device__ operator(), passed by value to the kernel):
//   device_mean       the mean and its standard error of a value per sample (an estimator, such as 4π·p(ω) over
//                     uniform directions, whose expected value is known); a NaN leaves its sample out, and the
//                     samples left out are counted (mean_estimate::left_out) for the test to check;
//   device_histogram  which cell of a grid each sample falls in (-1: none), for Pearson's χ² test;
//   device_count      how many samples have a property they mustn't have (a NaN, a density of 0);
//   cell_masses       the probability of each cell of a grid over [0, 1]², by Gauss-Legendre quadrature of a density
//                     over the cell: the expected share of the samples in the cell.
// Sample i of thread t comes from gpu::sample_generator(seed, t, 0), the renderer's own generator, drawing on; the same
// seed and count draw the same samples in every kernel, so a count and a histogram are about the very same samples.
// The kernels run 4096 threads (32 blocks of 128); 10^6 samples of a light's sampler take well under a millisecond.
// Every result is the same on every run: counts are integers, and a mean's sums are added on the host in the order of
// the threads (a double atomicAdd would add them in whatever order the threads finish).
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_test.h"
#include "random.cuh"
#include "statistics.h"
#include "vec3.cuh"

namespace gpu_test
{
inline void cuda_check(cudaError_t error, const char* what)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

// The largest single allocation a test may make. The suites' budget is 64 MB (gpu_test.h); this is a backstop against a
// mistake: on Windows the driver serves a cudaMalloc larger than the GPU's memory from the system's, instead of
// failing it, and starves every program on the computer.
inline constexpr std::size_t max_test_allocation = std::size_t{1} << 30;

// An array in the GPU's memory, freed when it goes out of scope; zeroed when made
template <class T>
class device_buffer
{
public:
    explicit device_buffer(std::size_t count) : count_(count)
    {
        if (count > max_test_allocation / sizeof(T))
        {
            throw std::runtime_error("a test asked for more than 1 GB of GPU memory at once");
        }
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
        note_device_bytes(static_cast<std::int64_t>(count * sizeof(T)));
        cuda_check(cudaMemset(data_, 0, count * sizeof(T)), "cudaMemset");
    }
    explicit device_buffer(const std::vector<T>& values) : device_buffer(values.size())
    {
        cuda_check(cudaMemcpy(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice),
                   "copying to the GPU");
    }
    ~device_buffer()
    {
        cudaFree(data_);
        note_device_bytes(-static_cast<std::int64_t>(count_ * sizeof(T)));
    }
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    T* get() const { return data_; }
    std::size_t size() const { return count_; }
    std::vector<T> download() const
    {
        std::vector<T> values(count_);
        cuda_check(cudaMemcpy(values.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "copying back");
        return values;
    }

private:
    std::size_t count_;
    T* data_ = nullptr;
};

// After a launch: its start errors at once, and its own errors once it has run
inline void finish_launch(const char* what)
{
    cuda_check(cudaGetLastError(), what);
    cuda_check(cudaDeviceSynchronize(), what);
}

inline constexpr int test_blocks = 32;
inline constexpr int test_block_size = 128;
inline constexpr int test_threads = test_blocks * test_block_size;

// Samples per thread for about n samples in all (at least one); the samples drawn are threads × that
inline int per_thread(std::int64_t n) { return static_cast<int>(std::max<std::int64_t>(1, n / test_threads)); }
inline std::int64_t drawn(std::int64_t n) { return static_cast<std::int64_t>(per_thread(n)) * test_threads; }

// --- The kernels -----------------------------------------------------------------------------------------------------

// Each thread's sum, sum of squares and number of samples, at sums[3t], [3t + 1] and [3t + 2]
template <class F>
__global__ void mean_kernel(F f, std::uint64_t seed, int count, double* sums)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    gpu::pcg32 rng = gpu::sample_generator(seed, static_cast<std::uint64_t>(t), 0);
    double sum = 0;
    double sum_of_squares = 0;
    double valid = 0;
    for (int i = 0; i < count; ++i)
    {
        // A NaN marks a sample the estimator doesn't apply to (1/p of a direction drawn with density 0): left out
        const double x = f(rng);
        if (x == x)
        {
            sum += x;
            sum_of_squares += x * x;
            valid += 1;
        }
    }
    sums[3 * t] = sum;
    sums[3 * t + 1] = sum_of_squares;
    sums[3 * t + 2] = valid;
}

template <class S>
__global__ void histogram_kernel(S s, std::uint64_t seed, int count, int cells, unsigned long long* counts)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    gpu::pcg32 rng = gpu::sample_generator(seed, static_cast<std::uint64_t>(t), 0);
    for (int i = 0; i < count; ++i)
    {
        const int cell = s(rng);
        atomicAdd(&counts[cell >= 0 && cell < cells ? cell : cells], 1ull); // the last entry: outside the grid
    }
}

template <class C>
__global__ void count_kernel(C c, std::uint64_t seed, int count, unsigned long long* found)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    gpu::pcg32 rng = gpu::sample_generator(seed, static_cast<std::uint64_t>(t), 0);
    unsigned long long n = 0;
    for (int i = 0; i < count; ++i)
    {
        n += c(rng) ? 1 : 0;
    }
    atomicAdd(found, n);
}

// The Gauss-Legendre rule of `order` points on [-1, 1]: exact for polynomials of degree up to 2·order - 1
struct gauss_rule
{
    int order = 0;
    double x[16] = {};
    double w[16] = {};
};

template <class D>
__global__ void cell_mass_kernel(D density, int nu, int nv, gauss_rule rule, double* masses)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nu * nv)
    {
        return;
    }
    const int i = cell % nu;
    const int j = cell / nu;
    const double du = 1.0 / nu;
    const double dv = 1.0 / nv;
    double sum = 0;
    for (int a = 0; a < rule.order; ++a)
    {
        const double u = (i + 0.5 * (rule.x[a] + 1)) * du;
        for (int b = 0; b < rule.order; ++b)
        {
            const double v = (j + 0.5 * (rule.x[b] + 1)) * dv;
            sum += rule.w[a] * rule.w[b] * density(u, v);
        }
    }
    masses[cell] = sum * 0.25 * du * dv; // the rule's weights sum to 2 on each axis
}

// --- The host side ---------------------------------------------------------------------------------------------------

// The nodes and weights by Newton's method on the Legendre polynomial P_order (Numerical Recipes' gauleg), 1 to 16
inline gauss_rule make_gauss_rule(int order)
{
    if (!(order >= 1 && order <= 16))
    {
        throw std::invalid_argument("a Gauss-Legendre rule of 1 to 16 points");
    }
    gauss_rule rule;
    rule.order = order;
    for (int i = 0; i < order; ++i)
    {
        double z = std::cos(std::numbers::pi * (i + 0.75) / (order + 0.5));
        double dp = 0;
        for (int iteration = 0; iteration < 100; ++iteration)
        {
            double p1 = 1;
            double p2 = 0;
            for (int k = 1; k <= order; ++k)
            {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2 * k - 1) * z * p2 - (k - 1) * p3) / k;
            }
            dp = order * (z * p1 - p2) / (z * z - 1);
            const double step = p1 / dp;
            z -= step;
            if (std::abs(step) < 1e-15)
            {
                break;
            }
        }
        rule.x[i] = z;
        rule.w[i] = 2 / ((1 - z * z) * dp * dp);
    }
    return rule;
}

template <class F>
mean_estimate device_mean(const F& f, std::uint64_t seed, std::int64_t n)
{
    device_buffer<double> sums(3 * static_cast<std::size_t>(test_threads));
    mean_kernel<<<test_blocks, test_block_size>>>(f, seed, per_thread(n), sums.get());
    finish_launch("a mean kernel");
    const std::vector<double> s = sums.download();
    double sum = 0;
    double sum_of_squares = 0;
    double valid = 0;
    for (std::size_t t = 0; t < s.size(); t += 3)
    {
        sum += s[t];
        sum_of_squares += s[t + 1];
        valid += s[t + 2];
    }
    mean_estimate m = from_sums(sum, sum_of_squares, static_cast<std::int64_t>(valid));
    m.left_out = drawn(n) - m.samples;
    return m;
}

// cells + 1 counts: the last one holds the samples outside the grid
template <class S>
std::vector<std::uint64_t> device_histogram(const S& s, std::uint64_t seed, std::int64_t n, int cells)
{
    device_buffer<unsigned long long> counts(static_cast<std::size_t>(cells) + 1);
    histogram_kernel<<<test_blocks, test_block_size>>>(s, seed, per_thread(n), cells, counts.get());
    finish_launch("a histogram kernel");
    const std::vector<unsigned long long> c = counts.download();
    return {c.begin(), c.end()};
}

template <class C>
std::int64_t device_count(const C& c, std::uint64_t seed, std::int64_t n)
{
    device_buffer<unsigned long long> found(1);
    count_kernel<<<test_blocks, test_block_size>>>(c, seed, per_thread(n), found.get());
    finish_launch("a count kernel");
    return static_cast<std::int64_t>(found.download()[0]);
}

template <class D>
std::vector<double> cell_masses(const D& density, int nu, int nv, int order = 8)
{
    device_buffer<double> masses(static_cast<std::size_t>(nu) * nv);
    const int cells = nu * nv;
    cell_mass_kernel<<<(cells + test_block_size - 1) / test_block_size, test_block_size>>>(
        density, nu, nv, make_gauss_rule(order), masses.get());
    finish_launch("the cell-mass kernel");
    return masses.download();
}

// A sampler's χ² test: the histogram of n samples against the masses (scaled to sum to 1, and a last cell of
// probability 0 for the samples outside the grid), and the masses' own sum, which must be 1 for a density
inline chi2_outcome chi2_against(const std::vector<std::uint64_t>& counts, const std::vector<double>& masses,
                                 double& mass_sum)
{
    mass_sum = 0;
    for (const double m : masses)
    {
        mass_sum += m;
    }
    std::vector<double> probabilities(masses.size() + 1, 0.0);
    for (std::size_t i = 0; i < masses.size(); ++i)
    {
        probabilities[i] = masses[i] / mass_sum;
    }
    return chi2_test(counts, probabilities);
}

// --- Directions ------------------------------------------------------------------------------------------------------

// A grid over the directions around an axis: u = φ / (2π) + 1/2, the angle about the axis from -π to π, and
// v = (cos θ - c_min) / (c_max - c_min) for the cosines from c_min to c_max. Equal steps of u and v are equal solid
// angles (Archimedes), and dω = jacobian() du dv. The frame is computed on the host in double.
struct direction_grid
{
    gpu::vec3<double> x, y, z; // orthonormal, z the axis
    double c_min = -1;
    double c_max = 1;
    int nu = 101;
    int nv = 50;

    __host__ __device__ int cells() const { return nu * nv; }
    __host__ __device__ double jacobian() const { return 2 * std::numbers::pi * (c_max - c_min); }

    // The cell of a direction (of any length), or -1: a NaN, or a cosine outside [c_min, c_max] by more than 10^-4 of
    // the range (a float direction drawn at the edge of a narrow cone can round that far past it: it counts in the
    // edge's cell)
    __device__ int cell(const gpu::vec3<float>& d) const
    {
        const gpu::vec3<double> e{d.x, d.y, d.z};
        const double length = ::sqrt(gpu::dot(e, e));
        const double c = gpu::dot(e, z) / length;
        const double phi = ::atan2(gpu::dot(e, y), gpu::dot(e, x));
        const double v = (c - c_min) / (c_max - c_min);
        const double u = phi / (2 * std::numbers::pi) + 0.5;
        if (!(v >= -1e-4 && v <= 1 + 1e-4 && u >= 0 && u <= 1))
        {
            return -1;
        }
        const int i = ::min(nu - 1, static_cast<int>(u * nu));
        const int j = ::min(nv - 1, ::max(0, static_cast<int>(v * nv)));
        return j * nu + i;
    }

    // The direction at (u, v), a unit vector in float
    __device__ gpu::vec3<float> direction(double u, double v) const
    {
        const double c = c_min + v * (c_max - c_min);
        const double s = ::sqrt(::fmax(0.0, 1 - c * c));
        const double phi = (u - 0.5) * 2 * std::numbers::pi;
        const gpu::vec3<double> d = (s * ::cos(phi)) * x + (s * ::sin(phi)) * y + c * z;
        return {static_cast<float>(d.x), static_cast<float>(d.y), static_cast<float>(d.z)};
    }
};

// An orthonormal frame around a unit axis, on the host
inline direction_grid grid_around(gpu::vec3d axis, double c_min = -1, double c_max = 1, int nu = 101, int nv = 50)
{
    const double length = std::sqrt(gpu::dot(axis, axis));
    axis = gpu::scale(1 / length, axis);
    const gpu::vec3d helper = std::abs(axis.x) < 0.9 ? gpu::vec3d{1, 0, 0} : gpu::vec3d{0, 1, 0};
    gpu::vec3d x = gpu::cross(helper, axis);
    x = gpu::scale(1 / std::sqrt(gpu::dot(x, x)), x);
    const gpu::vec3d y = gpu::cross(axis, x);
    direction_grid g;
    g.x = {x.x, x.y, x.z};
    g.y = {y.x, y.y, y.z};
    g.z = {axis.x, axis.y, axis.z};
    g.c_min = c_min;
    g.c_max = c_max;
    g.nu = nu;
    g.nv = nv;
    return g;
}
}
