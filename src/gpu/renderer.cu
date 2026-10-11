// The GPU renderer (see renderer.h): the kernel, and the host code that prepares the scene and runs the kernel.
//
// A CUDA program runs on two processors. Host code is ordinary C++ on the CPU; here it copies the scene into the GPU's
// memory and launches a kernel: a __global__ function that the GPU runs in a great many threads at once, one per pixel.
// nvcc splits this file: it compiles the device code (the kernel and the __device__ functions) for the GPU and hands
// the host code to the usual C++ compiler, MSVC here. The threads are grouped into blocks of 16×8 pixels, and the
// blocks into a grid that covers the image. The GPU spreads the blocks over its multiprocessors,
// and each multiprocessor runs the threads of its blocks in warps of 32: a warp executes one instruction for all of its
// threads at once. Each multiprocessor keeps dozens of warps in flight and switches to another whenever one waits for
// memory; that is how a GPU hides the slowness of memory.
//
// The GPU's memory spaces, as this renderer uses them: the scene (read-only, see render_kernel) and the image's sums
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
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bvh.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "random.cuh"
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
// hold with this kernel's 64 registers per thread).
constexpr int block_width = 16;
constexpr int block_height = 8;

// A warp is 32 threads, so a block should be a whole number of warps; warp_sum below also relies on it
static_assert(block_width * block_height % 32 == 0);

// The BVH's leaves hold one sphere, as the CPU's do: on the weekend scene that was faster than leaves of up to 2, 4 or
// 8 spheres (by 5, 7 and 20%). But for a handful of spheres a tree costs more than it saves: testing them all was
// faster in every scene of 8 spheres measured (parts of the weekend scene, with and without its ground) and about as
// fast at 16, and the materials scene renders 1.7 times faster as a single leaf. So a scene of at most 8 spheres gets
// a single leaf.
constexpr int bvh_leaf_size = 1;
constexpr int small_scene_spheres = 8;

// How long one kernel launch should take (see launch_planner): 100 ms would render 2.5% faster, 25 ms 5% slower
constexpr double launch_seconds = 0.05;

// Turns a CUDA error code into an exception. Most CUDA functions return a cudaError_t instead of throwing.
void check(cudaError_t result, const char* what)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(result));
    }
}

// An array in the GPU's memory that owns it, like a std::vector: cudaMalloc in the constructor and cudaFree in the
// destructor (RAII), so the memory is freed however render() ends, an exception included.
template <typename T>
class device_array
{
public:
    explicit device_array(std::size_t size) : size_(size)
    {
        if (size_ > 0)
        {
            check(cudaMalloc(&data_, size_ * sizeof(T)), "allocating GPU memory");
        }
    }

    explicit device_array(const std::vector<T>& values) : device_array(values.size())
    {
        if (size_ > 0)
        {
            check(cudaMemcpy(data_, values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice), "copying to the GPU");
        }
    }

    ~device_array() { cudaFree(data_); } // nothing to do for nullptr

    // A copy would free the same memory twice
    device_array(const device_array&) = delete;
    device_array& operator=(const device_array&) = delete;

    T* data() const { return data_; }

    void fill_with_zeros()
    {
        if (size_ > 0)
        {
            check(cudaMemset(data_, 0, size_ * sizeof(T)), "clearing GPU memory");
        }
    }

    std::vector<T> to_host() const
    {
        std::vector<T> values(size_);
        if (size_ > 0)
        {
            check(cudaMemcpy(values.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost), "copying from the GPU");
        }
        return values;
    }

private:
    std::size_t size_;
    T* data_ = nullptr;
};

// A CUDA event: a mark in the GPU's queue of work. When the GPU gets to it, it notes the time.
class event
{
public:
    // Waiting for an event (synchronize below) is a busy loop: the thread asks the GPU again and again and keeps one
    // CPU core at 100% for the whole render. With cudaEventBlockingSync it would sleep until the GPU signals instead,
    // but a sleeping thread may wake up late, and the GPU waits for the next launch meanwhile. On a quiet machine that
    // cost 0.3 ms per launch; with the other cores busy, float32 renders took up to 30% longer. A spinning thread
    // reacts at once, so the renderer keeps CUDA's default.
    event() { check(cudaEventCreate(&event_), "creating a CUDA event"); }
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

// The work of one launch: passes [first_pass, first_pass + pass_count) of rows [first_row, first_row + row_count)
struct launch_work
{
    int first_pass;
    int pass_count;
    int first_row;
    int row_count;
};

// Decides how much work each kernel launch gets. Windows resets a GPU when one kernel runs longer than about 2 seconds
// (TDR, Timeout Detection and Recovery: the render fails, and on a GPU that drives a display the screen flickers), and
// while a kernel runs on such a GPU, the desktop can't redraw. So each launch should take about launch_seconds. But not
// much less: every launch costs some microseconds to start and to wait for, and at its end the GPU is partly idle while
// its last warps finish. How long a sample takes varies a lot: between scenes, between precisions, and between the rows
// of one image (a ray into the sky is done at once, while one that hits glass bounces on). So the first pass measures
// it. It starts with a small band of rows and goes on with bands sized by the speed of the last one, but at most twice
// as large, in case the rows ahead are slower. It notes how long each row took, and the later passes are planned from
// these times: as many whole passes per launch as fit in launch_seconds or, when not even one fits (a large image in
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
    // resident_blocks: how many of the kernel's blocks the GPU runs at once (see resident_blocks)
    launch_planner(int width, int height, int passes, int resident_blocks)
        : width_(width),
          height_(height),
          passes_(passes),
          min_rows_(wave_rows(width, height, resident_blocks)),
          row_seconds_(static_cast<std::size_t>(height))
    {
    }

    bool done() const { return pass_ == passes_; }
    int passes_done() const { return pass_; }

    launch_work next() const
    {
        if (pass_ == 0)
        {
            constexpr double probe_samples = 1 << 16;
            const double samples =
                last_samples_ > 0 ? std::min(launch_seconds * last_speed_, 2 * last_samples_) : probe_samples;
            return band(static_cast<int>(std::min(std::floor(samples / width_), static_cast<double>(height_))));
        }
        if (row_ == 0 && pass_seconds_ <= launch_seconds)
        {
            const double passes =
                std::min(std::floor(launch_seconds / pass_seconds_), static_cast<double>(passes_ - pass_));
            return {pass_, static_cast<int>(passes), 0, height_};
        }
        int rows = 1;
        double seconds = row_seconds_[static_cast<std::size_t>(row_)];
        while (row_ + rows < height_ && seconds + row_seconds_[static_cast<std::size_t>(row_ + rows)] <= launch_seconds)
        {
            seconds += row_seconds_[static_cast<std::size_t>(row_ + rows)];
            ++rows;
        }
        return band(rows);
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

    // One pass of a band of `rows` rows from row_, but of at least min_rows_ rows and at most the rows left
    launch_work band(int rows) const { return {pass_, 1, row_, std::min(std::max(rows, min_rows_), height_ - row_)}; }

    int width_;
    int height_;
    int passes_;
    int min_rows_; // the fewest rows a band has, unless fewer are left: a wave's worth (see above)
    int pass_ = 0; // the first pass not yet complete
    int row_ = 0;  // its first row not yet rendered
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

// The kernel: each thread traces the launch's passes of one pixel and adds them to the pixel's sums.
// __launch_bounds__ tells the compiler the block size, so it can plan the registers for that many threads.
//
// The thread traces its samples one after another in a single loop, one ray per iteration: when a sample ends, the
// next one starts in the next iteration. The paths of a warp's threads have different lengths (one ray to the sky,
// or a dozen bounces inside glass), and a warp keeps running until its last thread is done. Written as two nested
// loops, over samples and over bounces, each sample would wait for the longest path of the warp; with one loop a thread
// whose sample ended early goes on with its next one, and only its last sample waits for the others. With one pass per
// launch, where this can't help, the weekend scene took 1.6 times as long.
//
// The scene's arrays and the sums come as pointers marked __restrict__: a promise to the compiler that while the
// kernel runs, the memory each of them points to is reached through that pointer alone. Then the writes to sums can't
// change the scene, so the compiler may load the scene through the GPU's read-only data path (the texture cache's:
// LDG.E.CONSTANT in the machine code) and keep what it loaded in registers. On the test GPU this measured no faster,
// as its L1 cache serves ordinary loads as well, but older GPUs gained from it, and the promise costs nothing.
template <real_number Real>
__global__ void __launch_bounds__(block_width * block_height)
    render_kernel(const camera<Real> cam, const bvh_node<Real>* __restrict__ nodes, const int node_count,
                  const sphere<Real>* __restrict__ spheres, const int* __restrict__ sphere_materials,
                  const material<Real>* __restrict__ materials, const std::uint64_t seed, const launch_work work,
                  double* __restrict__ sums, unsigned long long* __restrict__ ray_count)
{
    const scene_view<Real> scene{.tree = {.nodes = nodes, .spheres = spheres, .node_count = node_count},
                                 .sphere_materials = sphere_materials,
                                 .materials = materials};

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
        path<Real> p = cam.start_path(i, j, pass, rng);
        while (true)
        {
            ++rays;
            color<Real> sample;
            if (!trace_ray(p, scene, cam, rng, sample))
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
            p = cam.start_path(i, j, pass, rng);
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

// How many blocks of render_kernel<Real> the GPU runs at once: a wave (see launch_planner). Each multiprocessor holds
// as many blocks as its resources allow. A multiprocessor of the test GPU holds up to 1536 threads, but they share its
// 65536 registers, and this kernel takes 64 registers per thread in float and 96 in double: so it holds 8 of the
// kernel's blocks of 128 threads in float and 5 in double, 1024 and 640 threads. The CUDA runtime computes this from
// the compiled kernel; the share of the 1536 threads it allows, 67% and 42% here, is called the kernel's occupancy.
// (The 1536 threads per multiprocessor that cudaGetDeviceProperties reports would make a wave 1.5 times too large in
// float and 2.4 times in double.)
template <real_number Real>
int resident_blocks()
{
    int multiprocessors = 0;
    check(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0),
          "reading the CUDA device's properties");
    int blocks_per_multiprocessor = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor, render_kernel<Real>,
                                                        block_width * block_height, 0),
          "computing the kernel's occupancy");
    return multiprocessors * blocks_per_multiprocessor;
}

template <real_number Real>
vec3<Real> to_real(const vec3d& v)
{
    return {static_cast<Real>(v.x), static_cast<Real>(v.y), static_cast<Real>(v.z)};
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

template <real_number Real>
std::vector<material<Real>> to_materials(const std::vector<material_desc>& descs)
{
    std::vector<material<Real>> materials;
    for (const material_desc& desc : descs)
    {
        materials.push_back({.kind = desc.kind,
                             .albedo = to_real<Real>(desc.albedo),
                             .fuzz = static_cast<Real>(desc.fuzz),
                             .refraction_index = static_cast<Real>(desc.refraction_index)});
    }
    return materials;
}

// Rejects what the kernel can't render, before anything is copied to the GPU
void validate(const scene_desc& scene)
{
    const camera_desc& camera = scene.camera;
    if (camera.image_width < 1 || camera.image_height < 1)
    {
        throw std::runtime_error("the image must be at least one pixel wide and high");
    }
    // The grid of one launch has at most 65535 blocks vertically
    if (camera.image_height > 65535 * block_height)
    {
        throw std::runtime_error("the image is too tall for the GPU renderer");
    }
    // The passes are numbered with ints: at most 46340² = 2,147,395,600 of them
    if (camera.sqrt_spp < 1 || camera.sqrt_spp > 46340)
    {
        throw std::runtime_error("the number of rays per pixel must be from 1 to 46340²");
    }
    for (const material_desc& m : scene.materials)
    {
        if (m.kind != material_kind::lambertian && m.kind != material_kind::metal &&
            m.kind != material_kind::dielectric)
        {
            throw std::runtime_error("unknown material kind");
        }
    }
    for (const sphere_desc& s : scene.spheres)
    {
        if (s.material < 0 || s.material >= std::ssize(scene.materials))
        {
            throw std::runtime_error("a sphere refers to a material that doesn't exist");
        }
    }
}

// render() in one precision
template <real_number Real>
render_result render_in(const scene_desc& scene, const render_options& options)
{
    const auto start_time = std::chrono::steady_clock::now();
    validate(scene);
    check(cudaSetDevice(0), "selecting the CUDA device");

    const camera<Real> cam = to_camera<Real>(scene.camera);
    const int leaf_size = std::ssize(scene.spheres) <= small_scene_spheres ? small_scene_spheres : bvh_leaf_size;
    const bvh<Real> tree = build_bvh<Real>(scene.spheres, leaf_size);
    const device_array<bvh_node<Real>> nodes(tree.nodes);
    const auto node_count = static_cast<int>(tree.nodes.size());
    const device_array<sphere<Real>> spheres(tree.spheres);
    const device_array<int> sphere_materials(tree.sphere_materials);
    const device_array<material<Real>> materials(to_materials<Real>(scene.materials));

    const int width = cam.image_width;
    const int height = cam.image_height;
    const int passes = cam.sqrt_spp * cam.sqrt_spp;
    device_array<double> sums(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
    sums.fill_with_zeros();
    device_array<unsigned long long> ray_count(1);
    ray_count.fill_with_zeros();

    render_result result;
    // A path of no rays brings no light: with max_depth 0 the image stays black, as on the CPU
    if (cam.max_depth > 0)
    {
        launch_planner planner(width, height, passes, resident_blocks<Real>());
        event start;
        event stop;
        while (!planner.done())
        {
            const launch_work work = planner.next();
            const dim3 block(block_width, block_height);
            const dim3 grid(static_cast<unsigned>((width + block_width - 1) / block_width),
                            static_cast<unsigned>((work.row_count + block_height - 1) / block_height));
            start.record();
            // <<<grid, block>>> is CUDA's syntax for a launch: the kernel runs in grid.x × grid.y blocks of
            // block.x × block.y threads each. The launch only queues the kernel and returns at once; the GPU runs it
            // while the CPU goes on to stop.synchronize(), which waits for it.
            render_kernel<Real><<<grid, block>>>(cam, nodes.data(), node_count, spheres.data(),
                                                 sphere_materials.data(), materials.data(), options.seed, work,
                                                 sums.data(), ray_count.data());
            // A launch returns no error code. If the kernel couldn't even start (an invalid grid or block size, say),
            // cudaGetLastError reports it; errors while it runs come back at stop.synchronize()
            check(cudaGetLastError(), "launching the kernel");
            stop.record();
            stop.synchronize();
            const double seconds = event::seconds_between(start, stop);
            result.kernel_seconds += seconds;

            const int passes_before = planner.passes_done();
            planner.finished(work, seconds);
            if (options.progress && planner.passes_done() > passes_before)
            {
                options.progress(planner.passes_done(), passes);
            }
        }
    }

    result.pixels = sums.to_host();
    const double scale = 1.0 / passes; // the average of the samples, as the CPU's pixel_samples_scale_
    for (double& value : result.pixels)
    {
        value *= scale;
    }
    result.rays = ray_count.to_host().front();
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    return result;
}
}

device_info query_device()
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

    // Start CUDA on the device now. The first call that needs the GPU creates its context, the driver's state for this
    // program (its memory, its loaded kernels), and that takes a noticeable fraction of a second. cudaFree(nullptr)
    // frees nothing: it is the customary way to make that happen at a chosen moment, here, so that render() times
    // only the rendering.
    check(cudaSetDevice(0), "selecting the CUDA device");
    check(cudaFree(nullptr), "starting CUDA on the device");
    return info;
}

render_result render(const scene_desc& scene, const render_options& options)
{
    return options.arithmetic == precision::float64 ? render_in<double>(scene, options)
                                                    : render_in<float>(scene, options);
}
}
