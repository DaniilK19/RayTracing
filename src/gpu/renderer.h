// The GPU renderer's interface: a scene as plain data in, an image out.
//
// The renderer itself (renderer.cu and the .cuh headers next to it) is compiled by nvcc, NVIDIA's CUDA compiler, which
// can't import C++20 modules: neither ours nor `import std`. So the two sides meet in this ordinary header. renderer.cu
// includes it, and so does the GPU program, main_gpu.cpp, which builds a scene with the CPU ray tracer's own modules
// (the same scene, the same camera) and translates it into the structures below. Only plain data crosses over: no
// virtual functions and no pointers into the CPU's objects.
//
// A file that also has `import std;` must include this header BEFORE that line: MSVC accepts standard headers ahead of
// the std module in one file, but not after it.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace gpu
{
// A point, a direction or a color in double precision, like the CPU's vec3. The renderer converts it to the precision
// it computes with (see precision).
struct vec3d
{
    double x = 0;
    double y = 0;
    double z = 0;
};

// The first book's three materials (see material.ixx).
enum class material_kind
{
    lambertian, // matte: albedo
    metal,      // a blurred mirror: albedo and fuzz
    dielectric, // glass: refraction_index
};

struct material_desc
{
    material_kind kind = material_kind::lambertian;
    vec3d albedo;                // lambertian and metal: the fraction of light kept in each channel
    double fuzz = 0;             // metal: 0 is a perfect mirror; at most 1 (the metal class caps it)
    double refraction_index = 1; // dielectric: relative to the medium around it (1 / 1.5 for an air bubble in glass)
};

// A stationary sphere
struct sphere_desc
{
    vec3d center;
    double radius = 0;
    int material = 0; // an index into scene_desc::materials
};

// The camera as the CPU's camera class has set it up from its settings (see camera_frame in camera.ixx). The sample of
// pixel (i, j) in cell (s_i, s_j) of the pixel's sqrt_spp × sqrt_spp grid is the ray from the camera center, or from a
// random point center + a·defocus_disk_u + b·defocus_disk_v of the lens disk (a² + b² < 1), toward the point
// pixel00_loc + (i + dx)·pixel_delta_u + (j + dy)·pixel_delta_v, where dx = (s_i + r1) / sqrt_spp - 0.5 and
// dy = (s_j + r2) / sqrt_spp - 0.5 with random r1, r2 in [0, 1). Its direction is not normalized, and hits count
// from t = 0.001, measured in lengths of the direction, as on the CPU.
struct camera_desc
{
    int image_width = 0;
    int image_height = 0;
    int sqrt_spp = 1;   // rays per pixel: sqrt_spp²
    int max_depth = 10; // at most this many rays per path: the camera ray and its bounces
    vec3d center;
    vec3d pixel00_loc;
    vec3d pixel_delta_u;
    vec3d pixel_delta_v;
    bool defocus = false; // rays start on the lens disk (the CPU's defocus_angle > 0)
    vec3d defocus_disk_u;
    vec3d defocus_disk_v;
    vec3d sky_bottom; // the background: the gradient between these colors that the CPU's sky computes
    vec3d sky_top;
};

struct scene_desc
{
    std::vector<sphere_desc> spheres;
    std::vector<material_desc> materials;
    camera_desc camera;
};

// The number type the GPU computes with. A GeForce card does arithmetic on 32-bit floats about
// 64 times as fast as on the 64-bit doubles the CPU ray tracer uses, so float32 is the GPU's natural choice and
// float64 is there to compare with.
enum class precision
{
    float32,
    float64,
};

struct render_options
{
    std::uint64_t seed = 0; // renders with different seeds have statistically independent noise
    precision arithmetic = precision::float32;
    // Called on the host as passes complete (a pass adds one more sample, a camera ray and its bounces, to every pixel,
    // and one kernel launch may complete several), with the number of passes done so far and their total. May be
    // empty.
    std::function<void(int done, int total)> progress;
};

struct render_result
{
    std::vector<double> pixels; // width × height × 3 linear values (before gamma correction and clamping), row by row
                                // from the top: the layout of the CPU's image
    double seconds = 0;         // the whole render() call: the BVH, copying to the GPU, all passes, copying back
    double kernel_seconds = 0;  // the passes alone, timed on the GPU
    std::uint64_t rays = 0;     // rays traced: camera rays and all their bounces
};

struct device_info
{
    std::string name;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int multiprocessors = 0;
    double memory_gib = 0;
    int runtime_version = 0; // the CUDA runtime the program was built with, as CUDA writes versions: 13040 is 13.4
    int driver_version = 0;  // the newest CUDA version the installed driver supports, not the driver's own number
};

// Describes the first CUDA device and starts CUDA on it (a fraction of a second, once per program), so that render()
// times only the rendering. Throws std::runtime_error if there is no device (no NVIDIA GPU, or no driver).
device_info query_device();

// Renders the scene on the first CUDA device. Throws std::runtime_error with CUDA's message if something fails.
render_result render(const scene_desc& scene, const render_options& options);
}
