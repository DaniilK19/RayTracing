// The GPU renderer (see renderer.h): the kernels, and the host code that prepares the scene and runs the one it needs.
//
// A CUDA program runs on two processors. Host code is ordinary C++ on the CPU; here it copies the scene into the GPU's
// memory and launches a kernel: a __global__ function that the GPU runs in a great many threads at once, one per pixel.
// nvcc splits this file: it compiles the device code (the kernels and the __device__ functions) for the GPU and hands
// the host code to the usual C++ compiler, MSVC here. The threads are grouped into blocks of 16×8 pixels, and the
// blocks into a grid that covers the image. The GPU spreads the blocks over its multiprocessors,
// and each multiprocessor runs the threads of its blocks in warps of 32: a warp executes one instruction for all of its
// threads at once. Each multiprocessor keeps dozens of warps in flight and switches to another whenever one waits for
// memory; that is how a GPU hides the slowness of memory.
//
// The GPU's memory spaces, as this renderer uses them: the scene (read-only, see the kernels) and the image's sums
// live in global memory, the GPU's own gigabytes (cudaMalloc); the kernel's arguments, such as the camera, go to
// constant memory, which is cached and hands one value to a whole warp at once when all its threads read the same
// address, as they do here; each thread's variables live in its registers, and an array it indexes with a variable
// (the BVH stack) in its local memory, which is cached global memory.
//
// Every pixel's samples are traced in passes: a pass traces one more sample of every pixel. Each pixel adds up its
// samples in double, pass after pass, and the image is their average. See launch_planner for how passes are grouped
// into kernel launches.
#include "renderer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "bvh.cuh"
#include "camera.cuh"
#include "constant_medium.cuh"
#include "device_memory.h"
#include "display.cuh"
#include "material.cuh"
#include "quality.cuh"
#include "random.cuh"
#include "scene.cuh"
#include "sphere.cuh"
#include "texture.cuh"
#include "tonemap.cuh"
#include "vec3.cuh"

namespace gpu
{
// An unnamed namespace: what is inside is visible only in this file
namespace
{
// The numbers below were chosen by measuring the weekend scene (485 spheres) and the materials scene (5 spheres) at
// 1920×1080 with 484 rays per pixel in float32 on the test GPU.

// Threads per block. A block of neighboring pixels keeps a warp's rays close together, so they tend to visit the same
// BVH nodes and hit the same materials. 16×8 and 8×8 were equally fast, 16×16 and 32×4 slower by 2-3%, and 8×4 by 20%
// (a multiprocessor runs at most 16 blocks at once, so blocks of 32 threads give it half as many threads as it could
// hold with the first book's kernel's 64 registers per thread).
constexpr int block_width = 16;
constexpr int block_height = 8;

// A warp is 32 threads, so a block should be a whole number of warps; warp_sum below also relies on it
static_assert(block_width * block_height % 32 == 0);

// The BVH's leaves hold one primitive, as the CPU's hold one object: on the weekend scene that was faster than leaves
// of up to 2, 4 or 8 spheres (by 5, 7 and 20%). But for a handful of spheres a tree costs more than it saves: testing
// them all was faster in every scene of 8 spheres measured (parts of the weekend scene, with and without its ground)
// and about as fast at 16, and the materials scene renders 1.7 times faster as a single leaf. So a scene of at most 8
// shapes (spheres and quads) gets a single leaf (one per kind under a root when it has shapes of several kinds: see
// add_subtree in bvh.cuh).
constexpr int bvh_leaf_size = 1;
constexpr int small_scene_shapes = 8;

// The first band of rows of a view's first pass, in samples (see launch_planner). A session's first view starts with
// Step 59's probe, so that render() plans its launches exactly as before (and its timings stay comparable). A later
// view of the same session (the window's camera moved) starts with a larger band, so that a moving camera's frames
// need fewer launches: at 1280×720 in float, the whole first pass is one. The last view's times say nothing about the
// new one (a look from the sky down into glass can be dozens of times slower per sample), so every band of a later
// view's first pass is bounded by its samples alone: no larger than the probe, smaller in float64, whose samples are
// the slowest. (Growing by the speed measured, as the first view's bands do, was measured to go wrong: two bands on
// cheap rows of sky, then a band four times as large on rows of glass, in float64 at 1080p a launch of 350 ms.) At the
// slowest per-sample cost measured, about 0.4 µs inside a glass ball in float64, the probe takes about 0.1 s.
constexpr double first_view_probe_samples = 1 << 16;
template <real_number Real>
constexpr double next_view_probe_samples = sizeof(Real) == sizeof(double) ? 1 << 18 : 1 << 20;

// A CUDA event: a mark in the GPU's queue of work. When the GPU gets to it, it notes the time.
class event
{
public:
    // Waiting for an event (synchronize below) is a busy loop by default: the thread asks the GPU again and again and
    // keeps one CPU core at 100% for the whole render. With cudaEventBlockingSync it sleeps until the GPU signals
    // instead, but a sleeping thread may wake up late, and the GPU waits for the next launch meanwhile. On a quiet
    // machine that cost 0.3 ms per launch; with the other cores busy, float32 renders took up to 30% longer. So
    // RayTracingGPU, which only renders, spins (wait_mode::spin), and the window, whose core has other work, sleeps.
    explicit event(wait_mode wait = wait_mode::spin)
    {
        check(cudaEventCreateWithFlags(&event_, wait == wait_mode::sleep ? cudaEventBlockingSync : cudaEventDefault),
              "creating a CUDA event");
    }
    ~event() { cudaEventDestroy(event_); }
    event(const event&) = delete;
    event& operator=(const event&) = delete;

    void record() { check(cudaEventRecord(event_), "recording a CUDA event"); }

    // Waits until the GPU has done all the work queued before the event. A kernel's own errors (a bad memory access,
    // or Windows stopping it, see launch_planner) come back here.
    void synchronize() { check(cudaEventSynchronize(event_), "rendering on the GPU"); }

    // The GPU's time from one recorded event to another, in seconds
    static double seconds_between(const event& start, const event& stop)
    {
        float milliseconds = 0;
        check(cudaEventElapsedTime(&milliseconds, start.event_, stop.event_), "timing the GPU");
        return milliseconds / 1000.0;
    }

private:
    cudaEvent_t event_ = nullptr;
};

// Memory on the host that the GPU copies into directly ("pinned": the system never moves it), owned like device_array.
// A copy from the GPU into ordinary memory goes through a pinned buffer of the driver's, so this saves a copy.
template <typename T>
class pinned_array
{
public:
    pinned_array() = default;
    explicit pinned_array(std::size_t size) : size_(size)
    {
        if (size_ > 0)
        {
            check(cudaMallocHost(&data_, size_ * sizeof(T)), "allocating pinned host memory");
        }
    }
    ~pinned_array() { cudaFreeHost(data_); } // nothing to do for nullptr
    pinned_array(const pinned_array&) = delete;
    pinned_array& operator=(const pinned_array&) = delete;
    pinned_array(pinned_array&& other) noexcept
        : size_(std::exchange(other.size_, 0)), data_(std::exchange(other.data_, nullptr))
    {
    }
    pinned_array& operator=(pinned_array&& other) noexcept
    {
        std::swap(size_, other.size_);
        std::swap(data_, other.data_);
        return *this;
    }

    T* data() const { return data_; }
    std::size_t size() const { return size_; }

private:
    std::size_t size_ = 0;
    T* data_ = nullptr;
};

// The work of one launch: passes [first_pass, first_pass + pass_count) of rows [first_row, first_row + row_count),
// counted within one round of strata, whose seed the launch gets with it (see session in renderer.h)
struct launch_work
{
    int first_pass;
    int pass_count;
    int first_row;
    int row_count;
};

// Decides how much work each kernel launch of a view gets. Windows resets a GPU when one kernel runs longer than about
// 2 seconds (TDR, Timeout Detection and Recovery: the render fails, and on a GPU that drives a display the screen
// flickers). So each launch should take about launch_seconds (see render_options), the budget. But not much less: every
// launch costs some microseconds to start and to wait for, and at its end the GPU is partly idle while its last warps
// finish. How long a sample takes varies a lot: between scenes, between precisions, and between the rows of one image
// (a ray into the sky is done at once, while one that hits glass bounces on). So the first pass measures it. It starts
// with a small band of rows (the probe) and goes on with bands sized by the speed of the last one, but at most twice as
// large, in case the rows ahead are slower. It notes how long each row took, and the later passes are planned from
// these times: as many whole passes per launch as fit in the budget or, when not even one fits (a large image in
// float64), bands of rows that fit. The image doesn't depend on this grouping: every sample has its own random numbers
// (see random.cuh), and every pixel adds up its samples in the order of the passes.
//
// Taking a launch's time as proportional to its rows holds only while the launch has more blocks than the GPU runs at
// once (a wave, see resident_blocks). The blocks of a smaller launch all run side by side, so it takes at least as long
// as its slowest thread, however few rows it has. So a band is never much smaller than a wave: min_rows_ is the most
// rows of whole blocks that fit in one wave, and at least one row of blocks. Without this floor, a small image with
// very long paths went wrong: at 64×64 pixels in float64, where some paths are 20000 rays long, a pass took 0.15 s,
// whether in one launch or in a band of any size. Each smaller band made its rows look slower, so the next band was
// smaller still, down to one row per launch: 9 passes took 29 s instead of 1.2 s. (The floor is rounded down to whole
// rows of blocks, as a band of a wave and a bit takes two waves' time when the paths are that long.) When even a wave
// takes longer than launch_seconds, bands stay a wave long: no longer than the first band of pass 0 already was, so the
// floor doesn't bring a launch any closer to Windows' 2-second limit.
class launch_planner
{
public:
    // resident_blocks: how many of the kernel's blocks the GPU runs at once (see resident_blocks); round_passes: the
    // passes of a round of strata, sqrt_spp² (see session in renderer.h); probe_samples: the size of the first pass's
    // first band, and band_samples the most samples any band of the first pass may have (see first_view_probe_samples)
    launch_planner(int width, int height, int resident_blocks, std::int64_t round_passes, double probe_samples,
                   double band_samples)
        : width_(width),
          height_(height),
          min_rows_(wave_rows(width, height, resident_blocks)),
          round_passes_(round_passes),
          probe_samples_(probe_samples),
          band_samples_(band_samples),
          row_seconds_(static_cast<std::size_t>(height))
    {
    }

    // Where the view's image is: its complete passes, and the rows of the next one
    accumulation state() const { return {pass_, row_}; }

    // The round of strata the next launch renders in
    std::int64_t round() const { return pass_ / round_passes_; }

    // A view that renders nothing (max_depth 0) gets there without a launch
    void skip_to(accumulation target)
    {
        pass_ = target.passes;
        row_ = target.rows;
    }

    // The next launch: as much work as the budget allows, by the times measured so far, and none beyond `target`, which
    // must be ahead of state(). Its first_pass counts within the round: a launch never spans two rounds, whose samples
    // draw from different seeds (see round()).
    launch_work next(double budget, accumulation target) const
    {
        // The rows this pass may render: all, unless it is the target's unfinished one
        const int end_row = pass_ < target.passes ? height_ : target.rows;
        if (pass_ == 0)
        {
            const double samples =
                last_samples_ > 0 ? std::min({budget * last_speed_, 2 * last_samples_, band_samples_}) : probe_samples_;
            const double rows = std::min(std::floor(samples / width_), static_cast<double>(height_));
            return band(static_cast<int>(rows), end_row);
        }
        if (row_ == 0 && pass_ < target.passes && pass_seconds_ <= budget)
        {
            const double passes =
                std::min({std::floor(budget / pass_seconds_), static_cast<double>(target.passes - pass_),
                          static_cast<double>(round_passes_ - pass_ % round_passes_)});
            return {static_cast<int>(pass_ % round_passes_), static_cast<int>(passes), 0, height_};
        }
        int rows = 1;
        double seconds = row_seconds_[static_cast<std::size_t>(row_)];
        while (row_ + rows < height_ && seconds + row_seconds_[static_cast<std::size_t>(row_ + rows)] <= budget)
        {
            seconds += row_seconds_[static_cast<std::size_t>(row_ + rows)];
            ++rows;
        }
        return band(rows, end_row);
    }

    void finished(const launch_work& work, double seconds)
    {
        // A timer reading of 0 would make the next launch unlimited, so no launch counts as shorter than 1 µs
        seconds = std::max(seconds, 1e-6);
        last_samples_ = static_cast<double>(width_) * work.row_count * work.pass_count;
        last_speed_ = last_samples_ / seconds;
        const double row_seconds = seconds / (static_cast<double>(work.pass_count) * work.row_count);
        std::fill_n(row_seconds_.begin() + work.first_row, work.row_count, row_seconds);

        row_ += work.row_count;
        if (row_ == height_)
        {
            pass_seconds_ = std::accumulate(row_seconds_.begin(), row_seconds_.end(), 0.0);
            row_ = 0;
            pass_ += work.pass_count;
        }
    }

private:
    // The most rows of whole blocks that fit in one wave, but at least one row of blocks and at most the image
    static int wave_rows(int width, int height, int resident_blocks)
    {
        const int blocks_across = (width + block_width - 1) / block_width; // as the grid of a launch
        return std::min(block_height * std::max(1, resident_blocks / blocks_across), height);
    }

    // One pass of a band of `rows` rows from row_, but of at least min_rows_ rows and at most the rows up to end_row
    launch_work band(int rows, int end_row) const
    {
        return {static_cast<int>(pass_ % round_passes_), 1, row_,
                std::min(std::max(rows, min_rows_), end_row - row_)};
    }

    int width_;
    int height_;
    int min_rows_; // the fewest rows a band has, unless fewer are left: a wave's worth (see above)
    std::int64_t round_passes_;
    double probe_samples_;
    double band_samples_;
    std::int64_t pass_ = 0; // the first pass not yet complete
    int row_ = 0;           // its first row not yet rendered
    double last_samples_ = 0;
    double last_speed_ = 0;          // samples per second
    std::vector<double> row_seconds_; // how long one pass of each row took when it was last rendered
    double pass_seconds_ = 0;         // their sum: how long a whole pass takes
};

// camera::without_nan: a NaN channel of a sample counts as 0. NaN is the only value that isn't equal to itself.
template <real_number Real>
__device__ double without_nan(Real x)
{
    return static_cast<double>(x == x ? x : Real(0));
}

// The sum of a value over the 32 threads of a warp, in 5 steps. __shfl_down_sync reads a register of the thread
// `offset` lanes further in the same warp, without going through memory: after the steps for 16, 8, 4, 2 and 1, lane 0
// holds the sum of all 32. All the warp's threads must take part (the mask 0xffffffff names them all).
__device__ unsigned long long warp_sum(unsigned long long value)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// What a thread of either kernel does: it traces the launch's passes of one pixel and adds them to the pixel's sums.
//
// The thread traces its samples one after another in a single loop, one ray per iteration: when a sample ends, the
// next one starts in the next iteration. The paths of a warp's threads have different lengths (one ray to the sky,
// or a dozen bounces inside glass), and a warp keeps running until its last thread is done. Written as two nested
// loops, over samples and over bounces, each sample would wait for the longest path of the warp; with one loop a thread
// whose sample ended early goes on with its next one, and only its last sample waits for the others. With one pass per
// launch, where this can't help, the weekend scene took 1.6 times as long.
//
// __forceinline__ makes sure the compiler copies this function into each kernel rather than calling it: a call would
// pass the scene and the camera through memory.
template <real_number Real, kernel_kind Kind>
__device__ __forceinline__ void render_pixel(const camera<Real>& cam, const kernel_scene<Real, Kind>& scene,
                                             std::uint64_t seed, const launch_work& work, double* sums,
                                             unsigned long long* ray_count)
{
    // This thread's pixel: blockIdx is the block's position in the grid, blockDim the block's size and threadIdx the
    // thread's position in its block. The grid covers the launch's rows.
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int j = work.first_row + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    unsigned long long rays = 0; // rays traced: calls of closest_hit, as the CPU counts its world.hit calls
    // The grid is made of whole blocks, so some threads fall beyond the image's right edge or the launch's last row.
    // They don't return early: they still take part in warp_sum below.
    if (i < cam.image_width && j < work.first_row + work.row_count)
    {
        const std::uint64_t pixel =
            static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(cam.image_width) + static_cast<std::uint64_t>(i);
        double* const pixel_sum = sums + 3 * pixel;
        double red = pixel_sum[0];
        double green = pixel_sum[1];
        double blue = pixel_sum[2];

        int pass = work.first_pass;
        const int end_pass = work.first_pass + work.pass_count;
        pcg32 rng = sample_generator(seed, pixel, static_cast<std::uint64_t>(pass));
        path<Real> p = start_sample<Real, Kind>(cam, scene, i, j, pass, rng);
        while (true)
        {
            ++rays;
            color<Real> sample;
            if (!trace<Real, Kind>(p, scene, cam, rng, sample))
            {
                continue; // the path goes on
            }
            // Thousands of samples are added up, so the sums are kept in double, even when the samples are floats
            red += without_nan(sample.x);
            green += without_nan(sample.y);
            blue += without_nan(sample.z);
            if (++pass == end_pass)
            {
                break;
            }
            rng = sample_generator(seed, pixel, static_cast<std::uint64_t>(pass));
            p = start_sample<Real, Kind>(cam, scene, i, j, pass, rng);
        }
        pixel_sum[0] = red;
        pixel_sum[1] = green;
        pixel_sum[2] = blue;
    }

    // One atomic addition per warp: atomicAdd lets many threads add to one number safely, but the GPU performs those
    // to the same address one at a time, so 32 times fewer of them is faster
    rays = warp_sum(rays);
    if ((threadIdx.y * blockDim.x + threadIdx.x) % warpSize == 0)
    {
        atomicAdd(ray_count, rays);
    }
}

// The two kernels: __global__ functions, which the CPU launches and the GPU runs in many threads. __launch_bounds__
// tells the compiler the block size, so it can plan the registers for that many threads.
//
// Why two. A kernel takes one number of registers for all its threads, fixed when it is compiled: as many as the most
// demanding point of its code needs (where the most values are alive at once), whether a scene's paths ever get there
// or not. And the more registers a thread takes, the fewer threads fit on a multiprocessor (see
// blocks_per_multiprocessor). Each feature of the second and third books (textures, moving spheres, smoke, lamps...)
// adds code to the kernel, and it may add registers that the first book's scenes would pay for without using them
// (the solid and checker textures added none, moving spheres and images a few, the marble's Perlin noise the most:
// every render prints the registers of the kernel it ran, in its Kernel: line). So the scenes that need nothing beyond
// the first book get a kernel of their own, which traces Step 46's code (trace_first_book_ray), and every other scene
// gets the kernel for all three books. Both run render_pixel, each with its own Kind, and `if constexpr` (see trace in
// camera.cuh) compiles into each only its own code.
//
// They also get the scene differently. The first book's kernel takes Step 46's parameters: each of its arrays is a
// pointer marked __restrict__, a promise to the compiler that while the kernel runs, the memory it points to is reached
// through that pointer alone. Then writing the sums can't change the scene, so the compiler may load the scene through
// the GPU's read-only data path (the texture cache's: LDG.E.CONSTANT in the machine code) and keep what it loaded in
// registers. The compiler honors that promise only on a kernel's own parameters: the kernel for all three books gets
// its many arrays in one struct (scene_view), and the same pointers marked __restrict__ inside it were tried: they
// still gave ordinary loads (LDG.E), so they aren't marked. On the test GPU the read-only path measured no faster, as
// its L1 cache serves ordinary loads as well, but older GPUs gained from it.
template <real_number Real>
__global__ void __launch_bounds__(block_width * block_height)
    first_book_kernel(const camera<Real> cam, const bvh_node<Real>* __restrict__ nodes, const int node_count,
                      const sphere<Real>* __restrict__ spheres, const int* __restrict__ primitive_materials,
                      const material<Real>* __restrict__ materials, const std::uint64_t seed, const launch_work work,
                      double* __restrict__ sums, unsigned long long* __restrict__ ray_count)
{
    const first_book_scene<Real> scene{.tree = {.nodes = nodes, .spheres = spheres, .node_count = node_count},
                                       .primitive_materials = primitive_materials,
                                       .materials = materials};
    render_pixel<Real, kernel_kind::first_book>(cam, scene, seed, work, sums, ray_count);
}

// __launch_bounds__ gives the block size alone. Its optional second number, the blocks a multiprocessor should hold at
// once, makes the compiler fit the kernel into fewer registers and keep the rest in memory. Measured on the float
// kernel for all three books (95 registers; the double kernel, 164, is a check mode and isn't capped): final, cornell,
// perlin and smoke at 960 pixels wide with 4096 rays per pixel, medians of 5 runs on the test GPU. 6 blocks (at most 80
// registers) were as fast on average, perlin 6% faster and final 3.5% slower. 7 blocks (72), timed at 480 wide with
// 1024 rays per pixel, were 7% slower on average (final, cornell and smoke 11-12% slower, perlin 6% faster). Two ways
// to take registers from the marble's noise didn't lower the count: a call to turbulence instead of its inlined body
// (96 registers, 1% slower on average, perlin 6% faster and cornell 5% slower), and perlin_noise's outer loop, over di,
// kept rolled with #pragma unroll 1 (96, 4% slower, at 480 wide with 1024 rays per pixel). Nor did reading a quad's
// record field by field as hit_quad needs it, instead of the whole record first (2% slower). So the kernel takes the
// registers the compiler chooses.
template <real_number Real>
__global__ void __launch_bounds__(block_width * block_height)
    all_books_kernel(const camera<Real> cam, const scene_view<Real> scene, const std::uint64_t seed,
                     const launch_work work, double* __restrict__ sums, unsigned long long* __restrict__ ray_count)
{
    render_pixel<Real, kernel_kind::all_books>(cam, scene, seed, work, sums, ray_count);
}

// The kernel of the given kind, for the CUDA runtime's questions about it
template <real_number Real, kernel_kind Kind>
constexpr auto kernel_function()
{
    if constexpr (Kind == kernel_kind::first_book)
    {
        return first_book_kernel<Real>;
    }
    else
    {
        return all_books_kernel<Real>;
    }
}

// How many blocks of the kernel of the given kind one multiprocessor runs at once. It holds as many blocks as its
// resources allow. A multiprocessor of the test GPU holds up to 1536 threads, but they share its 65536 registers, and
// the first book's kernel takes 64 registers per thread in float and 96 in double: so it holds 8 of the kernel's blocks
// of 128 threads in float and 5 in double, 1024 and 640 threads. The CUDA runtime computes this from the compiled
// kernel; the share of the 1536 threads it allows, 67% and 42% here, is called the kernel's occupancy.
template <real_number Real, kernel_kind Kind>
int blocks_per_multiprocessor()
{
    int blocks = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel_function<Real, Kind>(),
                                                        block_width * block_height, 0),
          "computing the kernel's occupancy");
    return blocks;
}

// How many blocks of the kernel the whole GPU runs at once: a wave (see launch_planner). (The 1536 threads per
// multiprocessor that cudaGetDeviceProperties reports would make the first book's waves 1.5 times too large in float
// and 2.4 times in double.)
template <real_number Real, kernel_kind Kind>
int resident_blocks()
{
    int multiprocessors = 0;
    check(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0),
          "reading the CUDA device's properties");
    return multiprocessors * blocks_per_multiprocessor<Real, Kind>();
}

// The kernel as the CUDA runtime describes it (see kernel_info in renderer.h)
template <real_number Real, kernel_kind Kind>
kernel_info describe_kernel()
{
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, kernel_function<Real, Kind>()), "reading the kernel's attributes");
    int max_threads = 0;
    check(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, 0),
          "reading the CUDA device's properties");
    return {.name = Kind == kernel_kind::first_book ? "first book" : "all books",
            .registers = attributes.numRegs,
            .local_bytes = static_cast<int>(attributes.localSizeBytes),
            .blocks_per_multiprocessor = blocks_per_multiprocessor<Real, Kind>(),
            .threads_per_block = block_width * block_height,
            .max_threads_per_multiprocessor = max_threads};
}

template <real_number Real>
camera<Real> to_camera(const camera_desc& desc)
{
    // The pixel grid relative to the camera center, subtracted in double before it is rounded to Real: the kernel
    // builds its rays from relative vectors only (see camera::get_ray in camera.cuh)
    const vec3d center_to_pixel00{.x = desc.pixel00_loc.x - desc.center.x,
                                  .y = desc.pixel00_loc.y - desc.center.y,
                                  .z = desc.pixel00_loc.z - desc.center.z};
    return {.image_width = desc.image_width,
            .image_height = desc.image_height,
            .sqrt_spp = desc.sqrt_spp,
            .max_depth = desc.max_depth,
            .recip_sqrt_spp = static_cast<Real>(1.0 / desc.sqrt_spp),
            .center = to_real<Real>(desc.center),
            .center_to_pixel00 = to_real<Real>(center_to_pixel00),
            .pixel_delta_u = to_real<Real>(desc.pixel_delta_u),
            .pixel_delta_v = to_real<Real>(desc.pixel_delta_v),
            .defocus = desc.defocus,
            .defocus_disk_u = to_real<Real>(desc.defocus_disk_u),
            .defocus_disk_v = to_real<Real>(desc.defocus_disk_v),
            .sky_bottom = to_real<Real>(desc.sky_bottom),
            .sky_top = to_real<Real>(desc.sky_top)};
}

// Whether a material has a texture: a lambertian's albedo, a lamp's light, or the albedo of smoke's particles
bool has_texture(const material_desc& desc)
{
    return desc.kind == material_kind::lambertian || desc.kind == material_kind::diffuse_light ||
           desc.kind == material_kind::isotropic;
}

// Whether a material's texture is one solid color
bool solid_texture(const scene_desc& scene, const material_desc& desc)
{
    return has_texture(desc) && scene.textures[static_cast<std::size_t>(desc.texture)].kind == texture_kind::solid;
}

// The materials in Real. A texture of one color becomes the material's albedo: there the first book's kernel reads a
// lambertian's (see scatter in material.cuh), and the kernel for all three books takes it without looking up a
// texture (a lamp's light too).
template <real_number Real>
std::vector<material<Real>> to_materials(const scene_desc& scene)
{
    std::vector<material<Real>> materials;
    for (const material_desc& desc : scene.materials)
    {
        const vec3d albedo =
            solid_texture(scene, desc) ? scene.textures[static_cast<std::size_t>(desc.texture)].rgb : desc.albedo;
        materials.push_back({.kind = desc.kind,
                             .albedo = to_real<Real>(albedo),
                             .fuzz = static_cast<Real>(desc.fuzz),
                             .refraction_index = static_cast<Real>(desc.refraction_index)});
    }
    return materials;
}

// Whether a texture reads the texture coordinates: an image does, and so does a checker one of whose textures does
bool needs_uv(const scene_desc& scene, int index)
{
    const texture_desc& t = scene.textures[static_cast<std::size_t>(index)];
    switch (t.kind)
    {
    case texture_kind::image:
        return true;
    case texture_kind::checker:
        return needs_uv(scene, t.even) || needs_uv(scene, t.odd); // smaller indices: the recursion ends (validate)
    default:
        return false;
    }
}

// What the kernel for all three books needs about each material besides the struct to_materials makes (see
// material_texture in material.cuh)
std::vector<material_texture> to_material_textures(const scene_desc& scene)
{
    std::vector<material_texture> shading;
    for (const material_desc& desc : scene.materials)
    {
        const bool textured = has_texture(desc) && !solid_texture(scene, desc);
        shading.push_back(
            {.texture_index = textured ? desc.texture : -1,
             .needs_uv = textured && needs_uv(scene, desc.texture),
             .needs_exact_point =
                 textured && scene.textures[static_cast<std::size_t>(desc.texture)].kind == texture_kind::checker});
    }
    return shading;
}

template <real_number Real>
std::vector<texture<Real>> to_textures(const std::vector<texture_desc>& descs)
{
    std::vector<texture<Real>> textures;
    int noises = 0; // the marble textures so far: see to_noises
    for (const texture_desc& desc : descs)
    {
        textures.push_back({.kind = desc.kind,
                            .even = desc.even,
                            .odd = desc.odd,
                            .table = desc.kind == texture_kind::noise ? noises++ : desc.image,
                            .rgb = to_real<Real>(desc.rgb),
                            .inv_scale = static_cast<Real>(desc.inv_scale)});
    }
    return textures;
}

// The marble textures' numbers (see noise_texture in texture.cuh), in the order of the textures that are marble: their
// index in this array is what to_textures stores in a noise texture's table
template <real_number Real>
std::vector<noise_texture<Real>> to_noises(const scene_desc& scene)
{
    std::vector<noise_texture<Real>> noises;
    for (const texture_desc& t : scene.textures)
    {
        if (t.kind != texture_kind::noise)
        {
            continue;
        }
        const perlin_desc& tables = scene.perlins[static_cast<std::size_t>(t.perlin)];
        noise_texture<Real> marble{}; // value-initialized: the gradients' padding is zeros
        for (std::size_t i = 0; i < tables.random_vectors.size(); ++i)
        {
            marble.perlin.random_vectors[i].g = to_real<Real>(tables.random_vectors[i]);
            marble.perlin.perm_x[i] = static_cast<std::uint8_t>(tables.perm_x[i]);
            marble.perlin.perm_y[i] = static_cast<std::uint8_t>(tables.perm_y[i]);
            marble.perlin.perm_z[i] = static_cast<std::uint8_t>(tables.perm_z[i]);
        }
        marble.scale = static_cast<Real>(t.scale);
        noises.push_back(marble);
    }
    return noises;
}

// The pixels of all images in one array, one after another, and where each image is in it
struct images_on_device
{
    std::vector<image_info> infos;
    std::vector<texel> texels;
};

images_on_device to_images(const std::vector<image_desc>& images)
{
    images_on_device result;
    for (const image_desc& desc : images)
    {
        result.infos.push_back({.offset = static_cast<int>(result.texels.size()),
                                .width = desc.width,
                                .height = desc.height,
                                .unused = 0});
        for (std::size_t i = 0; i < desc.rgb.size(); i += 3)
        {
            result.texels.push_back({.r = desc.rgb[i], .g = desc.rgb[i + 1], .b = desc.rgb[i + 2], .unused = 0});
        }
    }
    return result;
}

// Rejects a view the kernels can't render (see validate)
void validate_camera(const camera_desc& camera)
{
    if (camera.image_width < 1 || camera.image_height < 1)
    {
        throw std::runtime_error("the image must be at least one pixel wide and high");
    }
    // The grid of one launch has at most 65535 blocks vertically (display.cu's blocks are 8 rows high too)
    if (camera.image_height > 65535 * block_height)
    {
        throw std::runtime_error("the image is too tall for the GPU renderer");
    }
    // A round's passes are numbered with ints: at most 46340² = 2,147,395,600 of them
    if (camera.sqrt_spp < 1 || camera.sqrt_spp > max_sqrt_spp)
    {
        throw std::runtime_error("the side of a pixel's grid of samples must be from 1 to 46340");
    }
}

// Rejects what the kernels can't render, before anything is copied to the GPU. Every index the kernels follow is
// checked here: on the GPU an index out of range may read whatever lies there without any error, or stop the kernel.
void validate(const scene_desc& scene)
{
    const camera_desc& camera = scene.camera;
    validate_camera(camera);
    // The permutations are stored as bytes on the GPU (see perlin_table in perlin.cuh)
    for (const perlin_desc& tables : scene.perlins)
    {
        for (const auto* perm : {&tables.perm_x, &tables.perm_y, &tables.perm_z})
        {
            if (std::ranges::any_of(*perm, [](int entry) { return entry < 0 || entry >= perlin_point_count; }))
            {
                throw std::runtime_error("a Perlin permutation has an entry outside 0..255");
            }
        }
    }
    // The texels of all images are numbered with ints (see image_info)
    std::size_t texel_count = 0;
    for (const image_desc& image : scene.images)
    {
        if (image.width < 1 || image.height < 1 ||
            image.rgb.size() != 3 * static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height))
        {
            throw std::runtime_error("an image texture's pixels don't match its size");
        }
        texel_count += image.rgb.size() / 3;
    }
    if (texel_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("the images have too many pixels for the GPU renderer");
    }
    const auto texture_count = std::ssize(scene.textures);
    for (std::ptrdiff_t i = 0; i < texture_count; ++i)
    {
        const texture_desc& t = scene.textures[static_cast<std::size_t>(i)];
        if (t.kind == texture_kind::checker)
        {
            // So that the kernel's loop down nested checkers ends (see texture_value in texture.cuh)
            if (t.even < 0 || t.even >= i || t.odd < 0 || t.odd >= i)
            {
                throw std::runtime_error("a checker texture must refer to textures that come before it");
            }
        }
        else if (t.kind == texture_kind::image)
        {
            if (t.image < 0 || t.image >= std::ssize(scene.images))
            {
                throw std::runtime_error("an image texture refers to an image that doesn't exist");
            }
        }
        else if (t.kind == texture_kind::noise)
        {
            if (t.perlin < 0 || t.perlin >= std::ssize(scene.perlins))
            {
                throw std::runtime_error("a marble texture refers to Perlin tables that don't exist");
            }
        }
        else if (t.kind != texture_kind::solid)
        {
            throw std::runtime_error("unknown texture kind");
        }
    }
    // Only diffuse_light emits light, and it scatters none: so a path ends where it finds light, and the kernel keeps
    // no running sum of it (see trace_ray in camera.cuh). A material that both emitted and scattered would need one.
    for (const material_desc& m : scene.materials)
    {
        if (has_texture(m))
        {
            if (m.texture < 0 || m.texture >= texture_count)
            {
                throw std::runtime_error("a material refers to a texture that doesn't exist");
            }
        }
        else if (m.kind != material_kind::metal && m.kind != material_kind::dielectric)
        {
            throw std::runtime_error("unknown material kind");
        }
    }
    // The lights and mixture strategies need a shape to aim at (the CPU's effective_sampling falls back to material
    // without one), each with a chance of being aimed at and a size
    if ((camera.sampling == sampling_strategy::lights || camera.sampling == sampling_strategy::mixture) &&
        scene.lights.empty())
    {
        throw std::runtime_error("the lights and mixture strategies need a lights list that isn't empty");
    }
    for (const light_desc& l : scene.lights)
    {
        const vec3d n = cross(l.quad.u, l.quad.v);
        const bool has_size = l.kind == light_shape::quad ? dot(n, n) > 0 : l.radius > 0;
        if (!(l.weight > 0 && l.weight <= 1) || !has_size)
        {
            throw std::runtime_error("a light must have a weight in (0, 1] and a size");
        }
    }
    for (const sphere_desc& s : scene.spheres)
    {
        if (s.material < 0 || s.material >= std::ssize(scene.materials))
        {
            throw std::runtime_error("a sphere refers to a material that doesn't exist");
        }
    }
    for (const quad_desc& q : scene.quads)
    {
        if (q.material < 0 || q.material >= std::ssize(scene.materials))
        {
            throw std::runtime_error("a quad refers to a material that doesn't exist");
        }
    }
    for (const medium_desc& m : scene.media)
    {
        if (m.material < 0 || m.material >= std::ssize(scene.materials) ||
            scene.materials[static_cast<std::size_t>(m.material)].kind != material_kind::isotropic)
        {
            throw std::runtime_error("a volume of smoke must refer to an isotropic material");
        }
        if (m.boundary == boundary_shape::convex_quads && (m.quads.empty() || !is_convex(m.quads)))
        {
            throw std::runtime_error("a volume of smoke must be bounded by one sphere or by quads of a convex shape");
        }
    }
    // Global ids are ints (see bvh in bvh.cuh), and so are the indices of the boundaries' quads
    std::size_t boundary_quads = 0;
    for (const medium_desc& m : scene.media)
    {
        boundary_quads += m.quads.size();
    }
    const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (scene.spheres.size() + scene.quads.size() + scene.media.size() > limit || boundary_quads > limit)
    {
        throw std::runtime_error("the scene has too many shapes for the GPU renderer");
    }

    // The quality kernel's own data (scene_desc::quality)
    const quality_desc& quality = scene.quality;
    // --- light sampling (track D): one of the three combinations
    if (quality.combination != light_combination::bsdf_only && quality.combination != light_combination::lights_only &&
        quality.combination != light_combination::power_heuristic)
    {
        throw std::runtime_error("unknown light combination");
    }
    // --- microfacets (track B): one side entry per material, or none
    if (!quality.microfacets.empty())
    {
        if (quality.microfacets.size() != scene.materials.size())
        {
            throw std::runtime_error("the materials' GGX side entries must be one per material");
        }
        if (std::ranges::any_of(quality.microfacets,
                                [](const microfacet_desc& m) { return m.kind != microfacet_kind::none; }))
        {
            throw std::runtime_error("rough metals and glossy coats aren't implemented yet");
        }
    }
    // --- environment (track C)
    if (quality.environment)
    {
        throw std::runtime_error("environment maps aren't implemented yet");
    }
}

// Whether a sphere moves while the shutter is open
bool moves(const sphere_desc& s)
{
    return s.displacement.x != 0 || s.displacement.y != 0 || s.displacement.z != 0;
}

// Whether something moves: a sphere, or a sphere that bounds a volume of smoke (the CPU passes the ray, with its time,
// on to the boundary). Then camera rays get a random time (see start_sample in camera.cuh).
bool motion(const scene_desc& scene)
{
    return std::ranges::any_of(scene.spheres, moves) || std::ranges::any_of(scene.media, [](const medium_desc& m) {
               return m.boundary == boundary_shape::sphere && moves(m.sphere);
           });
}

// A box in double rounded outward to Real: the box's corners in a medium (see bvh_builder::add_subtree for nodes)
template <real_number Real>
void round_outward(const build_box& box, point3<Real>& box_min, point3<Real>& box_max)
{
    box_min = {round_down<Real>(box.min[0]), round_down<Real>(box.min[1]), round_down<Real>(box.min[2])};
    box_max = {round_up<Real>(box.max[0]), round_up<Real>(box.max[1]), round_up<Real>(box.max[2])};
}

// The volumes of smoke in Real, and their boundaries' quads, one after another in one array (see medium in
// constant_medium.cuh). Each volume gets a conservative box: its sphere's sweep, or the union of its quads' boxes.
template <real_number Real>
struct media_on_device
{
    std::vector<medium<Real>> media;
    std::vector<quad<Real>> boundary_quads;
};

template <real_number Real>
media_on_device<Real> to_media(const scene_desc& scene)
{
    media_on_device<Real> result;
    for (const medium_desc& desc : scene.media)
    {
        medium<Real> m{}; // value-initialized: the fields the boundary's kind doesn't use are zeros
        m.neg_inv_density = static_cast<Real>(desc.neg_inv_density);
        m.material = desc.material;
        build_box box;
        if (desc.boundary == boundary_shape::sphere)
        {
            m.sphere = {.center = to_real<Real>(desc.sphere.center),
                        .radius = static_cast<Real>(desc.sphere.radius),
                        .displacement = to_real<Real>(desc.sphere.displacement),
                        .unused = 0};
            box = sweep_box(m.sphere);
        }
        else
        {
            m.first_quad = static_cast<int>(result.boundary_quads.size());
            m.quad_count = static_cast<int>(desc.quads.size());
            for (const quad_desc& q : desc.quads)
            {
                result.boundary_quads.push_back(make_quad<Real>(q));
                box.enclose(quad_box(q));
            }
        }
        round_outward(box, m.box_min, m.box_max);
        result.media.push_back(m);
    }
    return result;
}

// The global id of the world's primitive with exactly the numbers of a light, its twin, or -1 if it has none: a quad
// with the same corner and sides, or a sphere that stands still with the same center and radius (a moving one has no
// twin: the light aims at time 0, where the bounce's point isn't). A lights-list copy is built by the same function as
// the object in the world (see scene in scenes.ixx) and baked through the same instances, so the numbers agree bit for
// bit.
template <real_number Real>
int twin_of(const light_desc& desc, const scene_desc& scene, const bvh<Real>& tree)
{
    if (desc.kind == light_shape::quad)
    {
        for (std::size_t i = 0; i < scene.quads.size(); ++i)
        {
            const quad_desc& q = scene.quads[i];
            if (q.q == desc.quad.q && q.u == desc.quad.u && q.v == desc.quad.v)
            {
                return tree.quad_ids[i];
            }
        }
        return -1;
    }
    for (std::size_t i = 0; i < scene.spheres.size(); ++i)
    {
        const sphere_desc& s = scene.spheres[i];
        if (!moves(s) && s.center == desc.center && s.radius == desc.radius)
        {
            return tree.sphere_ids[i];
        }
    }
    return -1;
}

// The lights list in Real (see light in lights.cuh), for the strategies that aim at it: each quad made as the BVH's
// quads are, its area and the running sums of the weights computed in double and rounded once, and each light's twin
template <real_number Real>
std::vector<light<Real>> to_lights(const scene_desc& scene, const bvh<Real>& tree)
{
    std::vector<light<Real>> lights;
    if (scene.camera.sampling != sampling_strategy::lights && scene.camera.sampling != sampling_strategy::mixture)
    {
        return lights; // the other strategies don't aim at them
    }
    double cdf = 0;
    for (const light_desc& desc : scene.lights)
    {
        cdf += desc.weight;
        light<Real> l{};
        if (desc.kind == light_shape::quad)
        {
            const vec3d n = cross(desc.quad.u, desc.quad.v);
            l.quad = make_quad<Real>(desc.quad);
            l.u = to_real<Real>(desc.quad.u);
            l.area = static_cast<Real>(std::sqrt(dot(n, n))); // the CPU's n.length()
            l.v = to_real<Real>(desc.quad.v);
        }
        else
        {
            l.center = to_real<Real>(desc.center);
            l.radius = static_cast<Real>(desc.radius);
        }
        l.cdf = static_cast<Real>(cdf);
        l.weight = static_cast<Real>(desc.weight);
        l.kind = desc.kind;
        l.twin = twin_of(desc, scene, tree);
        lights.push_back(l);
    }
    return lights;
}

// Whether the first book's kernel can render the scene (see first_book_kernel): it has spheres that stand still and no
// other shapes, every material is a lambertian of one color, a metal or a dielectric (so the sky is the only light),
// and diffuse bounces sample as the material scatters
bool first_book_scene(const scene_desc& scene)
{
    return scene.quads.empty() && scene.media.empty() && std::ranges::none_of(scene.spheres, moves) &&
           scene.camera.sampling == sampling_strategy::material &&
           std::ranges::all_of(scene.materials, [&](const material_desc& m) {
               return (m.kind == material_kind::lambertian && solid_texture(scene, m)) ||
                      m.kind == material_kind::metal || m.kind == material_kind::dielectric;
           });
}

// The scene in the GPU's memory, in Real: the arrays the kernels read, owned here (so that they are freed however
// render() ends), and the scene_view that tells a kernel where they are
template <real_number Real>
class device_scene
{
public:
    explicit device_scene(const scene_desc& scene)
        : device_scene(scene, build_bvh<Real>(scene.spheres, scene.quads, leaf_size(scene)), to_images(scene.images),
                       to_media<Real>(scene))
    {
    }

    scene_view<Real> view() const
    {
        return {.motion = motion_,
                .sampling = sampling_,
                .tree = {.nodes = nodes_.data(),
                         .spheres = spheres_.data(),
                         .node_count = node_count_,
                         .moving_spheres = moving_spheres_.data(),
                         .moving_base = moving_base_,
                         .quads = quads_.data(),
                         .quad_base = quad_base_},
                .media = {.media = media_.data(),
                          .boundary_quads = boundary_quads_.data(),
                          .count = media_count_,
                          .base = media_base_},
                .primitive_materials = primitive_materials_.data(),
                .sphere_frames = sphere_frames_.data(),
                .materials = materials_.data(),
                .material_textures = material_textures_.data(),
                .textures = {.textures = textures_.data(),
                             .images = images_.data(),
                             .texels = texels_.data(),
                             .noises = noises_.data()},
                .lights = {.lights = lights_.data(), .count = static_cast<int>(lights_.size())}};
    }

    // The global id of each sphere and quad of scene_desc, by its index there (bvh::sphere_ids, quad_ids): the quality
    // kernel's lamps are found by them
    std::span<const int> sphere_ids() const { return sphere_ids_; }
    std::span<const int> quad_ids() const { return quad_ids_; }

private:
    device_scene(const scene_desc& scene, const bvh<Real>& tree, const images_on_device& images,
                 const media_on_device<Real>& media)
        : nodes_(tree.nodes),
          node_count_(static_cast<int>(tree.nodes.size())),
          spheres_(tree.spheres),
          moving_spheres_(tree.moving_spheres),
          moving_base_(tree.moving_base),
          quads_(tree.quads),
          quad_base_(tree.quad_base),
          media_(media.media),
          boundary_quads_(media.boundary_quads),
          media_count_(static_cast<int>(media.media.size())),
          media_base_(static_cast<int>(tree.primitive_materials.size())),
          primitive_materials_(tree.primitive_materials),
          sphere_frames_(tree.sphere_frames),
          materials_(to_materials<Real>(scene)),
          material_textures_(to_material_textures(scene)),
          textures_(to_textures<Real>(scene.textures)),
          images_(images.infos),
          texels_(images.texels),
          noises_(to_noises<Real>(scene)),
          lights_(to_lights<Real>(scene, tree)),
          motion_(motion(scene)),
          sampling_(scene.camera.sampling),
          sphere_ids_(tree.sphere_ids),
          quad_ids_(tree.quad_ids)
    {
    }

    static int leaf_size(const scene_desc& scene)
    {
        const std::size_t shapes = scene.spheres.size() + scene.quads.size();
        return shapes <= static_cast<std::size_t>(small_scene_shapes) ? small_scene_shapes : bvh_leaf_size;
    }

    device_array<bvh_node<Real>> nodes_;
    int node_count_;
    device_array<sphere<Real>> spheres_;
    device_array<moving_sphere<Real>> moving_spheres_;
    int moving_base_;
    device_array<quad<Real>> quads_;
    int quad_base_;
    device_array<medium<Real>> media_;
    device_array<quad<Real>> boundary_quads_;
    int media_count_;
    int media_base_; // the global id of the first volume's particles: right after the surface primitives
    device_array<int> primitive_materials_;
    device_array<sphere_frame<Real>> sphere_frames_;
    device_array<material<Real>> materials_;
    device_array<material_texture> material_textures_;
    device_array<texture<Real>> textures_;
    device_array<image_info> images_;
    device_array<texel> texels_;
    device_array<noise_texture<Real>> noises_;
    device_array<light<Real>> lights_;
    bool motion_;
    sampling_strategy sampling_;
    std::vector<int> sphere_ids_;
    std::vector<int> quad_ids_;
};

// The kernel of the given kind on one launch's work. <<<grid, block>>> is CUDA's syntax for a launch: the kernel runs
// in grid.x × grid.y blocks of block.x × block.y threads each. The launch only queues the kernel and returns at once.
template <real_number Real, kernel_kind Kind>
void launch_kernel(const camera<Real>& cam, const scene_view<Real>& scene, std::uint64_t seed, const launch_work& work,
                   double* sums, unsigned long long* ray_count)
{
    const dim3 block(block_width, block_height);
    const dim3 grid(static_cast<unsigned>((cam.image_width + block_width - 1) / block_width),
                    static_cast<unsigned>((work.row_count + block_height - 1) / block_height));
    if constexpr (Kind == kernel_kind::first_book)
    {
        first_book_kernel<Real><<<grid, block>>>(cam, scene.tree.nodes, scene.tree.node_count, scene.tree.spheres,
                                                 scene.primitive_materials, scene.materials, seed, work, sums,
                                                 ray_count);
    }
    else
    {
        all_books_kernel<Real><<<grid, block>>>(cam, scene, seed, work, sums, ray_count);
    }
    // A launch returns no error code. If the kernel couldn't even start (an invalid grid or block size, say),
    // cudaGetLastError reports it (check clears every error it reports, so this one is the launch's); errors while
    // it runs come back when the host waits for it
    check(cudaGetLastError(), "launching the kernel");
}

// Whether the scene has what only the quality kernel renders: a GGX side entry, an environment map
bool needs_quality_kernel(const scene_desc& scene)
{
    return std::ranges::any_of(scene.quality.microfacets,
                               [](const microfacet_desc& m) { return m.kind != microfacet_kind::none; }) ||
           scene.quality.environment.has_value();
}

// The kernel is chosen by what the scene contains, and by the options that ask for another one. The quality kernel
// renders a scene that needs it, a combination of light samples, or any scene when asked (--quality-kernel); the books'
// two kernels render the rest, as before.
bool uses_quality_kernel(const scene_desc& scene, const session_options& options)
{
    return options.quality_kernel || needs_quality_kernel(scene) ||
           scene.quality.combination != light_combination::bsdf_only;
}

bool uses_first_book_kernel(const scene_desc& scene, const session_options& options)
{
    return !uses_quality_kernel(scene, options) && !options.all_books_kernel && first_book_scene(scene);
}

// What a display transform can't be (see session::display in renderer.h), checked by the session's display(options)
// and by render() before it renders anything: an exposure that isn't a finite number of stops within the limit, an
// exposure for the books' bytes, and a tone map beyond the books in float64 (what is new computes in float only)
void check_display(const display_options& options, precision arithmetic)
{
    if (!(std::isfinite(options.exposure) && std::fabs(options.exposure) <= max_exposure))
    {
        throw std::invalid_argument(std::format("an exposure is a finite number of stops from {} to {}", -max_exposure,
                                                max_exposure));
    }
    if (options.op == tone_map::books && options.exposure != 0)
    {
        throw std::invalid_argument("the books' bytes take no exposure: it needs another tone map");
    }
    if (options.op != tone_map::books && arithmetic == precision::float64)
    {
        throw std::invalid_argument("a tone map other than books is a new display transform, which works in float "
                                    "only");
    }
}
}

// What a session holds, behind its pointer (see session in renderer.h). The precision is chosen at run time, so the
// session holds the interface and in<Real> the rest: the scene in the GPU's memory in Real, and the view.
struct session::impl
{
    template <real_number Real>
    class in;

    virtual ~impl() = default;
    virtual void set_camera(const camera_desc& camera) = 0;
    virtual const camera_desc& camera() const = 0;
    virtual accumulation state() const = 0;
    virtual render_report render_to(accumulation target, const std::function<void(std::int64_t)>& on_pass) = 0;
    virtual render_report render_for(double seconds, std::int64_t pass_limit) = 0;
    virtual std::vector<double> pixels() const = 0;
    virtual std::span<const std::uint8_t> display() = 0;
    virtual std::span<const std::uint8_t> display(const display_options& options) = 0;
    virtual std::uint64_t rays() const = 0;
    virtual std::uint64_t shadow_rays() const = 0;
    virtual bool has_view() const = 0;
    virtual double kernel_seconds() const = 0;
    virtual const kernel_info& kernel() const = 0;
};

template <real_number Real>
class session::impl::in final : public session::impl
{
public:
    in(const scene_desc& scene, const session_options& options)
        : options_(options),
          sampling_(scene.camera.sampling),
          device_(scene),
          quality_(uses_quality_kernel(scene, options)),
          first_book_(uses_first_book_kernel(scene, options)),
          kernel_(chosen_kernel_info()),
          resident_blocks_(chosen_resident_blocks()),
          ray_count_(2) // the rays traced, and the quality kernel's shadow rays
    {
        // The quality kernel computes in float only (session's constructor refuses it in double)
        if constexpr (std::same_as<Real, float>)
        {
            if (quality_)
            {
                quality_scene_ = detail::upload_quality(scene, device_.sphere_ids(), device_.quad_ids());
            }
        }
        start_view(scene.camera, first_view_probe_samples, std::numeric_limits<double>::infinity());
    }

    void set_camera(const camera_desc& camera) override
    {
        validate_camera(camera);
        if (camera.sampling != sampling_)
        {
            throw std::invalid_argument("a session's view must keep the scene's sampling strategy");
        }
        start_view(camera, next_view_probe_samples<Real>, next_view_probe_samples<Real>);
    }

    const camera_desc& camera() const override { return camera_; }
    accumulation state() const override { return planner().state(); }

    render_report render_to(accumulation target, const std::function<void(std::int64_t)>& on_pass) override
    {
        check_target(target);
        render_report report;
        if (cam_.max_depth <= 0)
        {
            skip_to(target);
        }
        while (planner().state() < target)
        {
            const std::int64_t passes_before = planner().state().passes;
            run(planner().next(options_.launch_seconds, target), report);
            if (on_pass && planner().state().passes > passes_before)
            {
                on_pass(planner().state().passes);
            }
        }
        report.state = planner().state();
        return report;
    }

    render_report render_for(double seconds, std::int64_t pass_limit) override
    {
        if (!(seconds >= 0 && seconds <= 3600))
        {
            throw std::invalid_argument("render_for needs a time from 0 to 3600 seconds");
        }
        const accumulation target{pass_limit, 0};
        check_view();
        check_range(target);
        render_report report;
        if (cam_.max_depth <= 0)
        {
            skip_to(target);
        }
        while (planner().state() < target)
        {
            // The view's first pass is completed whatever the time; after it, launches fill the time that is left,
            // except a last bit too short to be worth a launch. A launch ends waiting for its slowest blocks with the
            // GPU partly idle, for about a block's time, so the call ends rather than make a launch shorter than a
            // quarter of a full one (the call's time, or launch_seconds if that is less). The window's frames on the
            // second book's final scene used to end with such a launch: they render 6% more samples per second so.
            const bool first_pass = planner().state().passes == 0;
            const double left = seconds - report.gpu_seconds;
            const double full_launch = std::min(seconds, options_.launch_seconds);
            if (!first_pass && (left <= 0 || (report.launches > 0 && left < full_launch / 4)))
            {
                break;
            }
            const double budget = first_pass ? options_.launch_seconds : std::min(options_.launch_seconds, left);
            run(planner().next(budget, target), report);
        }
        report.state = planner().state();
        return report;
    }

    std::vector<double> pixels() const override
    {
        const accumulation now = planner().state();
        if (now.passes == 0 || now.rows != 0)
        {
            throw std::logic_error("the image has no complete passes, or a pass in progress");
        }
        std::vector<double> values = sums_.to_host(values_in_view());
        // The average of the samples, as the CPU's pixel_samples_scale_
        const double scale = 1.0 / static_cast<double>(now.passes);
        for (double& value : values)
        {
            value *= scale;
        }
        return values;
    }

    std::span<const std::uint8_t> display() override
    {
        const accumulation now = planner().state();
        if (now.passes == 0)
        {
            throw std::logic_error("the image has no complete pass to show");
        }
        const int width = camera_.image_width;
        const int height = camera_.image_height;
        const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4;
        // Grown when needed, as the sums (see start_view), and each on its own: after an allocation fails, the next
        // call tries it again
        if (rgba_.size() < bytes)
        {
            rgba_ = device_array<std::uint8_t>(0);
            rgba_ = device_array<std::uint8_t>(bytes);
        }
        if (shown_.size() < bytes)
        {
            shown_ = pinned_array<std::uint8_t>();
            shown_ = pinned_array<std::uint8_t>(bytes);
        }
        // Both scales are computed here as pixels() computes its own: 1.0 / the number of samples
        const auto passes = static_cast<double>(now.passes);
        detail::to_rgba8(sums_.data(), width, 0, now.rows, 1.0 / (passes + 1), rgba_.data());
        detail::to_rgba8(sums_.data(), width, now.rows, height - now.rows, 1.0 / passes, rgba_.data());
        // A synchronous copy waits for the conversion first (they are on the same stream), sleeping under
        // wait_mode::sleep (see query_device)
        check(cudaMemcpy(shown_.data(), rgba_.data(), bytes, cudaMemcpyDeviceToHost), "copying the image from the GPU");
        return {shown_.data(), bytes};
    }

    // display() through a tone map (see session::display in renderer.h): the options are checked first, whatever the
    // image (the programs refuse them earlier, in their own words); books is display() itself
    std::span<const std::uint8_t> display(const display_options& options) override
    {
        check_display(options, std::same_as<Real, float> ? precision::float32 : precision::float64);
        if (options.op == tone_map::books)
        {
            return display();
        }
        // The tone maps are new, and what is new computes in float only: `if constexpr` keeps them out of the double
        // session's code
        if constexpr (std::same_as<Real, float>)
        {
            const accumulation now = planner().state();
            if (now.passes == 0)
            {
                throw std::logic_error("the image has no complete pass to show");
            }
            const int width = camera_.image_width;
            const int height = camera_.image_height;
            const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4;
            // The same buffers as display()'s, grown as it grows them
            if (rgba_.size() < bytes)
            {
                rgba_ = device_array<std::uint8_t>(0);
                rgba_ = device_array<std::uint8_t>(bytes);
            }
            if (shown_.size() < bytes)
            {
                shown_ = pinned_array<std::uint8_t>();
                shown_ = pinned_array<std::uint8_t>(bytes);
            }
            // 2^EV in double, rounded once to float: 0 EV multiplies by exactly 1, a whole number of stops by an exact
            // power of two
            const auto exposure_scale = static_cast<float>(std::exp2(options.exposure));
            // The two bands and their scales as display() computes them
            const auto passes = static_cast<double>(now.passes);
            detail::to_rgba8_toned(sums_.data(), width, 0, now.rows, 1.0 / (passes + 1), options.op, exposure_scale,
                                   rgba_.data());
            detail::to_rgba8_toned(sums_.data(), width, now.rows, height - now.rows, 1.0 / passes, options.op,
                                   exposure_scale, rgba_.data());
            check(cudaMemcpy(shown_.data(), rgba_.data(), bytes, cudaMemcpyDeviceToHost),
                  "copying the image from the GPU");
            return {shown_.data(), bytes};
        }
        else
        {
            throw std::invalid_argument("a tone map other than books is a new display transform, which works in float "
                                        "only");
        }
    }

    std::uint64_t rays() const override
    {
        check_view();
        return ray_count_.to_host().front();
    }
    std::uint64_t shadow_rays() const override
    {
        check_view();
        return ray_count_.to_host()[1];
    }
    double kernel_seconds() const override
    {
        check_view();
        return kernel_seconds_;
    }
    bool has_view() const override { return planner_.has_value(); }
    const kernel_info& kernel() const override { return kernel_; }

private:
    // The chosen kernel's description and blocks per wave. The quality kernel exists in float only: `if constexpr`
    // keeps it out of the double session's code entirely.
    kernel_info chosen_kernel_info() const
    {
        if constexpr (std::same_as<Real, float>)
        {
            if (quality_)
            {
                return detail::quality_kernel_info();
            }
        }
        return first_book_ ? describe_kernel<Real, kernel_kind::first_book>()
                           : describe_kernel<Real, kernel_kind::all_books>();
    }
    int chosen_resident_blocks() const
    {
        if constexpr (std::same_as<Real, float>)
        {
            if (quality_)
            {
                return detail::quality_resident_blocks();
            }
        }
        return first_book_ ? resident_blocks<Real, kernel_kind::first_book>()
                           : resident_blocks<Real, kernel_kind::all_books>();
    }

    std::size_t values_in_view() const
    {
        return static_cast<std::size_t>(camera_.image_width) * static_cast<std::size_t>(camera_.image_height) * 3;
    }

    // The image starts over: new sums, at least as large as the view needs (they grow, never shrink, so the same
    // views in turn don't allocate again), a new count of rays and a new planner. The old view is dropped before the
    // old sums are freed, which keeps the memory of the old and the new sums from being needed at once; so if a step
    // fails (the GPU's memory can't hold the new sums), the session has no view until a set_camera succeeds (see
    // planner()).
    void start_view(const camera_desc& camera, double probe_samples, double band_samples)
    {
        const gpu::camera<Real> cam = to_camera<Real>(camera);
        planner_.reset();
        const std::size_t values = static_cast<std::size_t>(camera.image_width) *
                                   static_cast<std::size_t>(camera.image_height) * 3;
        if (sums_.size() < values)
        {
            sums_ = device_array<double>(0); // frees the old sums before the new ones are allocated
            sums_ = device_array<double>(values);
        }
        sums_.fill_with_zeros(values);
        ray_count_.fill_with_zeros();
        kernel_seconds_ = 0;
        camera_ = camera;
        cam_ = cam;
        planner_.emplace(camera.image_width, camera.image_height, resident_blocks_,
                         std::int64_t{camera.sqrt_spp} * camera.sqrt_spp, probe_samples, band_samples);
    }

    // The planner of the view; none after a set_camera that failed
    launch_planner& planner()
    {
        check_view();
        return *planner_;
    }
    const launch_planner& planner() const
    {
        check_view();
        return *planner_;
    }
    void check_view() const
    {
        if (!planner_)
        {
            throw std::logic_error("the session has no view: its last set_camera failed");
        }
    }

    void check_target(accumulation target) const
    {
        check_view();
        check_range(target);
        if (target < planner().state())
        {
            throw std::invalid_argument("a render target can't be behind the image");
        }
    }

    void check_range(accumulation target) const
    {
        if (target.passes < 0 || target.rows < 0 || target.rows >= cam_.image_height)
        {
            throw std::invalid_argument("a render target must be a number of passes from 0 up, and rows of the next "
                                        "pass from 0 to below the image's height");
        }
    }

    // A path of no rays brings no light: with max_depth 0 (or less) the image stays black, as on the CPU, and needs no
    // launch
    void skip_to(accumulation target)
    {
        if (planner().state() < target)
        {
            planner().skip_to(target);
        }
    }

    // One launch: queues the kernel, waits for it and tells the planner how long it took. The launch's passes all lie
    // in one round, whose seed it renders with.
    void run(const launch_work& work, render_report& report)
    {
        const std::uint64_t seed = options_.seed + static_cast<std::uint64_t>(planner().round()) * round_seed_step;
        start_.record();
        if (quality_)
        {
            if constexpr (std::same_as<Real, float>)
            {
                detail::launch_quality(*quality_scene_, cam_, device_.view(), seed, work.first_pass, work.pass_count,
                                       work.first_row, work.row_count, sums_.data(), ray_count_.data());
            }
        }
        else if (first_book_)
        {
            launch_kernel<Real, kernel_kind::first_book>(cam_, device_.view(), seed, work, sums_.data(),
                                                         ray_count_.data());
        }
        else
        {
            launch_kernel<Real, kernel_kind::all_books>(cam_, device_.view(), seed, work, sums_.data(),
                                                        ray_count_.data());
        }
        stop_.record();
        stop_.synchronize();
        const double seconds = event::seconds_between(start_, stop_);
        kernel_seconds_ += seconds;
        ++report.launches;
        report.gpu_seconds += seconds;
        report.longest_launch_seconds = std::max(report.longest_launch_seconds, seconds);
        planner().finished(work, seconds);
        report.state = planner().state();
    }

    session_options options_;
    sampling_strategy sampling_; // the scene's
    device_scene<Real> device_;
    bool quality_;    // the quality kernel renders (float only)
    bool first_book_; // the first book's kernel renders; neither: the kernel for all three books
    detail::quality_scene_ptr quality_scene_; // the quality kernel's own arrays, if it renders
    kernel_info kernel_;
    int resident_blocks_;
    camera_desc camera_;
    gpu::camera<Real> cam_{}; // the view as the kernels take it (the bare name camera is the function above)
    device_array<double> sums_{0};
    device_array<unsigned long long> ray_count_;
    double kernel_seconds_ = 0;
    std::optional<launch_planner> planner_;
    event start_{options_.wait};
    event stop_{options_.wait};
    device_array<std::uint8_t> rgba_{0}; // the display's bytes on the GPU (see display), and their copy on the host
    pinned_array<std::uint8_t> shown_;
};

session::session(const scene_desc& scene, const session_options& options)
{
    validate(scene);
    if (!(options.launch_seconds > 0 && options.launch_seconds <= max_launch_seconds))
    {
        throw std::invalid_argument(
            std::format("a launch must be planned to take more than 0 and at most {} seconds", max_launch_seconds));
    }
    // What the quality kernel can't render (the programs say it first, in their own words)
    if (uses_quality_kernel(scene, options))
    {
        if (options.arithmetic == precision::float64)
        {
            throw std::invalid_argument("the quality kernel computes in float only");
        }
        if (options.all_books_kernel)
        {
            throw std::invalid_argument("the quality kernel and the kernel for all three books can't both render");
        }
        if (scene.camera.sampling != sampling_strategy::material)
        {
            throw std::invalid_argument("the quality kernel samples bounces as the material scatters (-p material)");
        }
    }
    check(cudaSetDevice(0), "selecting the CUDA device");
    if (options.wait == wait_mode::sleep)
    {
        unsigned flags = 0;
        check(cudaGetDeviceFlags(&flags), "reading how the CUDA device waits");
        if ((flags & cudaDeviceScheduleMask) != cudaDeviceScheduleBlockingSync)
        {
            throw std::logic_error("a session that sleeps needs a device that sleeps: query_device(wait_mode::sleep)");
        }
    }
    if (options.arithmetic == precision::float64)
    {
        impl_ = std::make_unique<impl::in<double>>(scene, options);
    }
    else
    {
        impl_ = std::make_unique<impl::in<float>>(scene, options);
    }
}

session::~session() = default;
session::session(session&&) noexcept = default;
session& session::operator=(session&&) noexcept = default;

void session::set_camera(const camera_desc& camera)
{
    impl_->set_camera(camera);
}

const camera_desc& session::camera() const
{
    return impl_->camera();
}

accumulation session::state() const
{
    return impl_->state();
}

render_report session::render_to(accumulation target, const std::function<void(std::int64_t)>& on_pass)
{
    return impl_->render_to(target, on_pass);
}

render_report session::render_for(double seconds, std::int64_t pass_limit)
{
    return impl_->render_for(seconds, pass_limit);
}

std::vector<double> session::pixels() const
{
    return impl_->pixels();
}

std::span<const std::uint8_t> session::display()
{
    return impl_->display();
}

std::span<const std::uint8_t> session::display(const display_options& options)
{
    return impl_->display(options);
}

std::uint64_t session::rays() const
{
    return impl_->rays();
}

std::uint64_t session::shadow_rays() const
{
    return impl_->shadow_rays();
}

bool session::has_view() const
{
    return impl_->has_view();
}

double session::kernel_seconds() const
{
    return impl_->kernel_seconds();
}

const kernel_info& session::kernel() const
{
    return impl_->kernel();
}

bool is_convex(const std::vector<quad_desc>& quads)
{
    // Every corner of every quad, and the largest coordinate among them, for the tolerance
    std::vector<vec3d> corners;
    double largest = 0;
    for (const quad_desc& q : quads)
    {
        for (const vec3d& c : {q.q, vec3d{q.q.x + q.u.x, q.q.y + q.u.y, q.q.z + q.u.z},
                               vec3d{q.q.x + q.v.x, q.q.y + q.v.y, q.q.z + q.v.z},
                               vec3d{q.q.x + q.u.x + q.v.x, q.q.y + q.u.y + q.v.y, q.q.z + q.u.z + q.v.z}})
        {
            corners.push_back(c);
            largest = std::max({largest, std::fabs(c.x), std::fabs(c.y), std::fabs(c.z)});
        }
    }
    const double tolerance = 1e-9 * (1 + largest);
    for (const quad_desc& q : quads)
    {
        const vec3d n = cross(q.u, q.v);
        const double length = std::sqrt(dot(n, n));
        if (!(length > 0))
        {
            return false; // a quad without area bounds nothing
        }
        const vec3d unit = scale(1 / length, n);
        bool in_front = false;
        bool behind = false;
        for (const vec3d& c : corners)
        {
            const double height = dot(unit, vec3d{c.x - q.q.x, c.y - q.q.y, c.z - q.q.z});
            in_front = in_front || height > tolerance;
            behind = behind || height < -tolerance;
        }
        if (in_front && behind)
        {
            return false; // the quad's plane cuts the shape: it isn't on the surface of a convex one
        }
    }
    return true;
}

device_info query_device(wait_mode wait)
{
    int count = 0;
    check(cudaGetDeviceCount(&count), "looking for a CUDA device");
    if (count == 0)
    {
        throw std::runtime_error("no CUDA device found");
    }
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "reading the CUDA device's properties");
    device_info info;
    info.name = properties.name;
    info.compute_capability_major = properties.major;
    info.compute_capability_minor = properties.minor;
    info.multiprocessors = properties.multiProcessorCount;
    info.memory_gib = static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
    check(cudaRuntimeGetVersion(&info.runtime_version), "reading the CUDA runtime version");
    check(cudaDriverGetVersion(&info.driver_version), "reading the CUDA driver version");

    // Start CUDA on the device now, so that render() times only the rendering. cudaSetDevice creates the device's
    // context (it does since CUDA 12), the driver's state for this program (its memory, its loaded kernels), and that
    // takes a noticeable fraction of a second.
    check(cudaSetDevice(0), "selecting the CUDA device");
    // The flag replaces the context's own, from now on. (Spinning is also what CUDA's default does, with one context.)
    check(cudaSetDeviceFlags(wait == wait_mode::sleep ? cudaDeviceScheduleBlockingSync : cudaDeviceScheduleSpin),
          "setting how the CUDA device waits");
    return info;
}

render_result render(const scene_desc& scene, const render_options& options)
{
    const auto start_time = std::chrono::steady_clock::now();
    check_display(options.display, options.arithmetic);
    session one(scene, {.seed = options.seed,
                         .arithmetic = options.arithmetic,
                         .all_books_kernel = options.all_books_kernel,
                         .quality_kernel = options.quality_kernel,
                         .launch_seconds = options.launch_seconds});
    if (options.passes < 0)
    {
        throw std::invalid_argument("a render's passes must be 0 (one round of strata) or more");
    }
    const std::int64_t passes =
        options.passes > 0 ? options.passes : std::int64_t{scene.camera.sqrt_spp} * scene.camera.sqrt_spp;
    std::function<void(std::int64_t)> on_pass;
    if (options.progress)
    {
        on_pass = [&](std::int64_t done) { options.progress(done, passes); };
    }
    one.render_to({passes, 0}, on_pass);

    render_result result;
    result.pixels = one.pixels();
    result.kernel_seconds = one.kernel_seconds();
    result.rays = one.rays();
    result.shadow_rays = one.shadow_rays();
    result.kernel = one.kernel();
    // The display's bytes after a tone map other than books, from the sums still in the GPU's memory: the bytes the
    // window shows. With books rgba stays empty: the books' bytes are the PNG's, which the programs make from the
    // pixels.
    if (options.display != display_options{})
    {
        const std::span<const std::uint8_t> bytes = one.display(options.display);
        result.rgba.assign(bytes.begin(), bytes.end());
    }
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    return result;
}
}
