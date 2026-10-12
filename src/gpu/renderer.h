// The GPU renderer's interface: a scene as plain data in, an image out.
//
// The renderer itself (renderer.cu and the .cuh headers next to it) is compiled by nvcc, NVIDIA's CUDA compiler, which
// can't import C++20 modules: neither ours nor `import std`. So the two sides meet in this ordinary header. renderer.cu
// includes it, and so does the GPU program, main_gpu.cpp, which builds a scene with the CPU ray tracer's own modules
// (the same scene, the same camera) and translates it into the structures below. Only plain data crosses over: no
// virtual functions and no pointers into the CPU's objects.
//
// The scene is a set of tables that refer to each other by index, as the CPU's objects refer to each other by pointer:
// a sphere names its material, a material its texture, a checker two other textures, an image texture its image. Each
// struct mirrors a CPU class, with plain fields in double (an image's pixels in float, which keeps stb_image's values
// exactly: see image_desc); how the GPU stores them is the renderer's own business.
//
// A file that also has `import std;` must include this header BEFORE that line: MSVC accepts standard headers ahead of
// the std module in one file, but not after it.
#pragma once

#include <array>
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

// --- Textures (texture.ixx) -----------------------------------------------------------------------------------------

enum class texture_kind
{
    solid,   // solid_color: one color everywhere
    checker, // checker_texture: the cubes of space alternate between two other textures
    image,   // image_texture: a picture wrapped around the surface by its texture coordinates (u, v)
    noise,   // noise_texture: marble made of Perlin noise
};

// A texture: its kind and the numbers of that kind (the other fields keep their defaults). Textures refer to each
// other by their index in scene_desc::textures, and a checker's two textures always come before it: following them
// leads to smaller and smaller indices, so the GPU's loop down nested checkers must end (renderer.cu checks it).
struct texture_desc
{
    texture_kind kind = texture_kind::solid;
    vec3d rgb;            // solid: its color (solid_color::albedo)
    double inv_scale = 1; // checker: 1 / the side of its cubes, the number the CPU multiplies by (inv_scale())
    int even = -1;        // checker: the texture of the cubes whose x + y + z is even, an index into textures
    int odd = -1;         // checker: the texture of the other cubes
    int image = -1;       // image: an index into scene_desc::images
    double scale = 1;     // noise: the frequency of the marble's stripes along z (noise_texture::scale)
    int perlin = -1;      // noise: an index into scene_desc::perlins, the tables of its noise
};

// The pixels of an image texture as the CPU's image holds them: linear colors, rows from the top. They are the floats
// that stb_image loaded (image::load), so floats keep every one of them exactly.
struct image_desc
{
    int width = 0;
    int height = 0;
    std::vector<float> rgb; // width × height × 3: pixel (x, y) starts at 3·(y·width + x)
};

// The size of a perlin object's tables (perlin::point_count; main_gpu.cpp checks that the two agree)
inline constexpr int perlin_point_count = 256;

// The tables of one perlin object, which its constructor drew from the CPU's random numbers while the scene was built
// (see perlin.ixx): 256 random unit gradients, and three permutations of 0..255 that pick one of them for every point
// (i, j, k) of the lattice: random_vectors[perm_x[i & 255] ^ perm_y[j & 255] ^ perm_z[k & 255]]. Every noise_texture
// has its own.
struct perlin_desc
{
    std::array<vec3d, perlin_point_count> random_vectors{};
    std::array<int, perlin_point_count> perm_x{};
    std::array<int, perlin_point_count> perm_y{};
    std::array<int, perlin_point_count> perm_z{};
};

// --- Materials (material.ixx) ---------------------------------------------------------------------------------------

enum class material_kind
{
    lambertian,    // matte: `texture` is its albedo
    metal,         // a blurred mirror: albedo and fuzz
    dielectric,    // glass: refraction_index
    diffuse_light, // a lamp: `texture` is the light it emits, from its front side only; it reflects nothing
};

struct material_desc
{
    material_kind kind = material_kind::lambertian;
    vec3d albedo;                // metal: the fraction of light kept in each channel
    double fuzz = 0;             // metal: 0 is a perfect mirror; at most 1 (the metal class caps it)
    double refraction_index = 1; // dielectric: relative to the medium around it (1 / 1.5 for an air bubble in glass)
    int texture = -1;            // lambertian, diffuse_light: an index into scene_desc::textures (one color is a solid
                                 // texture)
};

// --- Shapes, in world coordinates (sphere.ixx, quad.ixx) ------------------------------------------------------------

// A sphere, standing still or moving
struct sphere_desc
{
    vec3d center;     // at time 0: sphere::center_at(0)
    double radius = 0;
    int material = 0; // an index into scene_desc::materials
    // Motion blur: while the shutter is open, from time 0 to 1, the center moves by this much
    // (sphere::center_at(1) - center_at(0)), and at time t it is center + t·displacement. Exactly 0 for a sphere that
    // stands still.
    vec3d displacement;
};

// A parallelogram: the points q + α·u + β·v with α and β from 0 to 1. Its front side is the one u × v points to.
struct quad_desc
{
    vec3d q;
    vec3d u;
    vec3d v;
    int material = 0; // an index into scene_desc::materials
};

// --- The camera (camera.ixx) ----------------------------------------------------------------------------------------

// How a diffuse bounce chooses the next direction: the CPU's sampling_strategy
enum class sampling_strategy
{
    uniform,  // the same density in every direction
    material, // as the material scatters
    lights,   // toward the lamps of the scene's lights list
    mixture,  // half and half
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
    vec3d sky_top;    // (both black in the scenes lit by lamps)
    // The strategy the CPU's camera renders with: effective_sampling of the one asked for, so lights and mixture come
    // only with a lights list that isn't empty
    sampling_strategy sampling = sampling_strategy::material;
};

// --- The scene, and how to render it --------------------------------------------------------------------------------

struct scene_desc
{
    std::vector<sphere_desc> spheres;
    std::vector<material_desc> materials;
    camera_desc camera;
    std::vector<texture_desc> textures;
    std::vector<image_desc> images;
    std::vector<perlin_desc> perlins;
    std::vector<quad_desc> quads;
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
    // Render with the kernel for all three books even where the first book's kernel would do (see the kernels in
    // renderer.cu): a check that the general code computes the first book's paths as Step 46 does, and a measure of
    // what its generality costs
    bool all_books_kernel = false;
};

// The kernel that rendered, as the CUDA runtime describes it: the numbers that decide how many of its threads a
// multiprocessor keeps in flight (its occupancy, see blocks_per_multiprocessor in renderer.cu)
struct kernel_info
{
    std::string name;                       // "first book" or "all books"
    int registers = 0;                      // per thread
    // per thread: the BVH stack, the scratch arrays of CUDA's math functions (sin's reduction of huge arguments takes
    // 48 bytes in double), and registers spilled to memory, if any
    int local_bytes = 0;
    int blocks_per_multiprocessor = 0;      // as many as fit at once
    int threads_per_block = 0;              // 128
    int max_threads_per_multiprocessor = 0; // 1536 on the test GPU
};

struct render_result
{
    std::vector<double> pixels; // width × height × 3 linear values (before gamma correction and clamping), row by row
                                // from the top: the layout of the CPU's image
    double seconds = 0;         // the whole render() call: the BVH, copying to the GPU, all passes, copying back
    double kernel_seconds = 0;  // the passes alone, timed on the GPU
    std::uint64_t rays = 0;     // rays traced: camera rays and all their bounces
    kernel_info kernel;
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

// Renders the scene on the first CUDA device. Throws std::runtime_error with CUDA's message if something fails, or if
// the scene is inconsistent (an index out of range, a checker before its textures...).
render_result render(const scene_desc& scene, const render_options& options);
}
