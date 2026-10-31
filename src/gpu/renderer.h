// The GPU renderer's interface: a scene as plain data in, an image out.
//
// The renderer itself (renderer.cu, display.cu and the .cuh headers next to them) is compiled by nvcc, NVIDIA's CUDA
// compiler, which can't import C++20 modules: neither ours nor `import std`. So the two sides meet in this ordinary
// header. renderer.cu includes it, and so does the gpu_scene module (gpu_scene.ixx), which builds a scene with the CPU
// ray tracer's own modules (the same scene, the same camera) and translates it into the structures below for the GPU
// programs. Only plain data crosses over: no virtual functions and no pointers into the CPU's objects.
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
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
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

    // Equal in every coordinate. "= default" (C++20) has the compiler write the comparison, member by member.
    bool operator==(const vec3d&) const = default;
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

// The size of a perlin object's tables (perlin::point_count; gpu_scene.ixx checks that the two agree)
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
    isotropic,     // the particles of smoke and fog (constant_medium): `texture` is their albedo
};

struct material_desc
{
    material_kind kind = material_kind::lambertian;
    vec3d albedo;                // metal: the fraction of light kept in each channel
    double fuzz = 0;             // metal: 0 is a perfect mirror; at most 1 (the metal class caps it)
    double refraction_index = 1; // dielectric: relative to the medium around it (1 / 1.5 for an air bubble in glass)
    int texture = -1;            // lambertian, diffuse_light, isotropic: an index into scene_desc::textures (one
                                 // color is a solid texture)
};

// --- Shapes, in world coordinates (sphere.ixx, quad.ixx) ------------------------------------------------------------

// A sphere, standing still or moving
struct sphere_desc
{
    vec3d center;     // at time 0, in the world: sphere::center_at(0), carried out of the instances around the sphere
    double radius = 0;
    int material = 0; // an index into scene_desc::materials
    // Motion blur: while the shutter is open, from time 0 to 1, the center moves by this much
    // (sphere::center_at(1) - center_at(0), turned by those instances), and at time t it is center + t·displacement.
    // Exactly 0 for a sphere that stands still.
    vec3d displacement;
    // A sphere that rotate_y instances turned: the cosine and the sine of their combined rotation about the y axis (1
    // and 0 for a sphere that wasn't turned). Its texture coordinates come from its own, unturned normal (sphere::hit),
    // which the GPU gets back from the world's normal with rotate_y::to_object's formula.
    double uv_cos = 1;
    double uv_sin = 0;
};

// A parallelogram: the points q + α·u + β·v with α and β from 0 to 1. Its front side is the one u × v points to.
struct quad_desc
{
    vec3d q;
    vec3d u;
    vec3d v;
    int material = 0; // an index into scene_desc::materials
};

// --- Smoke and fog (constant_medium.ixx) ---------------------------------------------------------------------------

// What bounds a volume of smoke. constant_medium::hit takes the first crossing of the ray's line with the boundary and
// the next one at least 0.0001 beyond it: where the line enters and leaves a convex shape, as its comment says. The GPU
// renders the convex boundaries made of the books' shapes.
enum class boundary_shape
{
    sphere,       // one sphere (medium_desc::sphere)
    convex_quads, // quads that bound a convex shape, such as a box's six faces (medium_desc::quads, see is_convex)
};

// A volume uniformly filled with particles. Rays don't stop at its boundary, which only says where the volume is: the
// boundary's shapes are not among the scene's spheres and quads, and their materials are not used.
struct medium_desc
{
    boundary_shape boundary = boundary_shape::sphere;
    sphere_desc sphere;           // boundary_shape::sphere; it may move (its displacement)
    std::vector<quad_desc> quads; // boundary_shape::convex_quads
    double neg_inv_density = 0;   // -1 / density: constant_medium's own number
    int material = 0;             // its phase function: an isotropic material in scene_desc::materials
};

// Whether the quads lie on the surface of one convex shape: for every quad, all the corners of all the quads lie on
// one side of its plane, or on it (within 10^-9 of the largest coordinate, for the rounding of the instances' turns).
// Then a line meets the quads at no more than two points, and the CPU's "first crossing, then the next one at least
// 0.0001 beyond it" are the nearest and the farthest crossing (see boundary_crossings in constant_medium.cuh).
// gpu_scene.ixx asks it to explain a scene it can't render; renderer.cu's validate asks it again.
bool is_convex(const std::vector<quad_desc>& quads);

// --- What the lights and mixture strategies aim at: the scene's lights list (scene::lights) --------------------------

enum class light_shape
{
    quad,
    sphere,
};

// A shape of the lights list. Lists inside the list, such as a box's six faces, are flattened, and weight is the chance
// that hittable_list::random aims at this shape: 1/n in a list of n, multiplied down nested lists. So the weights add
// up to 1, and the list's density is the sum of weight × each shape's density (hittable_list::pdf_value). The shapes
// are the lights list's own copies (see scene in scenes.ixx), with their instances baked like the world's, never looked
// up among the world's objects. (The renderer only notes which primitive of the world has exactly the same numbers, if
// one has: a bounce that leaves that primitive treats its copy as itself, see lights_pdf_value in lights.cuh.)
struct light_desc
{
    light_shape kind = light_shape::quad;
    quad_desc quad;    // light_shape::quad (its material is not used)
    vec3d center;      // light_shape::sphere: its center at time 0, where sphere::pdf_value and random aim
    double radius = 0; // light_shape::sphere
    double weight = 0;
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
    int sqrt_spp = 1;   // the side of a pixel's grid of samples: a round of strata is sqrt_spp² passes (see session)
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

    // The same view in every field (0.0 and -0.0 count as equal). session::set_camera always starts the image over, so
    // the window calls it only when its camera differs from session::camera(), or when the session has no view.
    bool operator==(const camera_desc&) const = default;
};

// --- Beyond the books: what the quality kernel renders besides the books' tables (quality.cu) -------------------
//
// A third kernel, for light and materials the books don't have: light sampled at every bounce, rough metals, an
// environment map. The books' two kernels stay as they are, instruction for instruction (see the kernels in
// renderer.cu): what only the third kernel reads is kept apart from the books' tables, in these structures.

// How the quality kernel combines the light it finds
enum class light_combination
{
    bsdf_only,       // -p material: the material's own samples alone
    lights_only,     // -p nee: a light sample at every bounce; light the material's samples find counts only where no
                     // light sample could have found it
    power_heuristic, // -p mis: both, weighted by Veach's power heuristic
};

enum class microfacet_kind
{
    none,      // the material as the books define it
    conductor, // a rough metal: GGX with the metal's albedo as F0 (the material must be metal, fuzz 0)
    gloss,     // a GGX coat of F0 0.04 over the material's matte color (the material must be lambertian)
};

// A material's GGX side entry: an array of its own, so material<Real> keeps the layout the books' kernels index
struct microfacet_desc
{
    microfacet_kind kind = microfacet_kind::none;
    double roughness = 0; // r in [0, 1]; the GGX width is α = r²; gloss needs α ≥ 1e-3
};

// An HDR image around the scene, seen from inside: equirectangular, +y up, row 0 the file's top (+y)
struct environment_desc
{
    image_desc image;       // linear radiance, at most 2048×1024 texels, finite and ≥ 0
    double yaw_degrees = 0; // turns the map about +y
    double scale = 1;       // multiplies its radiance; its sampling density is unchanged
};

struct quality_desc
{
    light_combination combination = light_combination::bsdf_only;
    std::vector<microfacet_desc> microfacets;    // empty, or one per scene_desc::materials entry
    std::optional<environment_desc> environment; // the quality kernel's sky instead of camera.sky_bottom/sky_top
};

// --- Beyond the books: the display transform (tonemap.cu) --------------------------------------------------------
//
// How the image's light becomes the bytes of the screen and of a PNG. The books' way stays their scenes' default: the
// square root of each channel, clamped below 1, so everything brighter than 1 is plain white, channel by channel, and a
// highlight loses its shape and its hue. The tone maps beyond the books multiply the light by 2^exposure, compress it
// into [0, 1] with a curve made for that (tone_curves.cuh), encode it for the display and round it to bytes with a
// dither. The GPU computes those bytes for the window and for RayTracingGPU's PNG alike; a PFM always keeps the linear
// values.
enum class tone_map
{
    books,   // the books' bytes: √ gamma, clamp to 0.999, ×256 truncated (display.cu, image::save_png)
    none,    // exposure, clip to [0, 1], the sRGB curve (IEC 61966-2-1): no tone curve
    agx,     // Troy Sobotka's AgX base: 16.5 stops on a log2 curve, display-encoded
    neutral, // Khronos PBR Neutral, then the sRGB curve
};

struct display_options
{
    tone_map op = tone_map::books;
    double exposure = 0; // EV: the light is multiplied by 2^exposure first; 0 with books
    bool operator==(const display_options&) const = default;
};

// The exposure's range in stops, either way: a factor of 16.8 million, far beyond what a scene needs, and float(2^EV)
// stays a normal float with all of its precision
inline constexpr double max_exposure = 24;

// --- OptiX (optional) and triangle meshes: host types, no OptiX or CUDA type ---------------------------------------

class optix_runtime; // OptiX for the whole program (optix_runtime.cuh), started by start_optix() (start_optix.h)

// What OptiX's AI denoiser computed and holds after its last call (optix_runtime::last_denoise), for the programs' and
// the tests' reports: the exposure it was given, computed over the whole image before the invoke, and its memory
struct denoise_report
{
    float intensity = 0;                // hdrIntensity (optixDenoiserComputeIntensity)
    float average_color[3] = {0, 0, 0}; // hdrAverageColor, the AOV model's (optixDenoiserComputeAverageColor)
    std::size_t state_bytes = 0;        // the denoiser's state, set up for the last image (or for its tiles)
    std::size_t scratch_bytes = 0;      // its scratch, shared by the setup, the invoke and the two computations above
    int tile = 0;                       // the side of the tiles the last image was denoised in; 0: one invoke for all
    bool valid = false;                 // false until a denoise has completed, and again after release_denoiser
};

// How a glTF texture repeats outside [0, 1]
enum class texture_wrap
{
    repeat,
    mirrored_repeat,
    clamp,
};

// A glTF image as the texture hardware reads it: 8-bit RGBA, rows from the top (glTF's (0, 0) is the top-left texel)
struct mesh_texture_desc
{
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba; // width × height × 4
    bool srgb = true;               // base color: decoded from sRGB by the hardware; metallic-roughness: linear
    texture_wrap wrap_s = texture_wrap::repeat;
    texture_wrap wrap_t = texture_wrap::repeat;
    bool nearest = false;           // glTF's magFilter NEAREST; otherwise bilinear
};

// What a triangle wears: one of the scene's materials, or glTF's metallic-roughness model (see docs/optix.md)
struct mesh_material_desc
{
    int scene_material = -1;                   // >= 0: scene_desc::materials[i] (lambertian, metal, or one a GGX side
                                               // entry completes); -1: the glTF fields below
    std::array<double, 3> base_color{1, 1, 1}; // linear, times the texture's decoded color
    double metallic = 1;                       // glTF's defaults: a rough white metal
    double roughness = 1;
    int base_color_texture = -1;               // indices into mesh_set::textures; -1: none
    int metal_rough_texture = -1;              // G roughness, B metallic (glTF's channels)
};

// One mesh in its own coordinates; its triangles share these vertices
struct mesh_desc
{
    std::vector<float> positions;           // 3 per vertex
    std::vector<float> normals;             // 3 per vertex, or empty: flat shading
    std::vector<float> uvs;                 // 2 per vertex (glTF's convention), or empty
    std::vector<std::uint32_t> indices;     // 3 per triangle, counter-clockwise seen from the front
    std::vector<std::uint32_t> material_of; // per triangle: an index into mesh_set::materials
};

// A placement of a mesh: object to world, row-major 3×4 (OptixInstance::transform's layout)
struct mesh_instance_desc
{
    int mesh = 0;
    std::array<double, 12> object_to_world{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
};

// The scene's triangles: only OptiX renders them (a session without optix_tracer refuses a scene with instances)
struct mesh_set
{
    std::vector<std::shared_ptr<const mesh_desc>> meshes; // shared with gpu_scene's model cache, never copied per build
    std::vector<mesh_instance_desc> instances;
    std::vector<mesh_material_desc> materials;
    std::vector<std::shared_ptr<const mesh_texture_desc>> textures;
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
    std::vector<medium_desc> media;
    std::vector<light_desc> lights; // the lights list's shapes: empty unless camera.sampling is lights or mixture
    quality_desc quality;           // empty for the books' scenes; a scene that needs it renders with the quality kernel
    mesh_set meshes;                // triangle meshes, which only OptiX renders (render_options::optix_tracer); empty
                                    // for every scene of the books and of the quality kernel
};

// How many lamps the quality kernel's light set holds in the scene (lamp_table.cu): its quads and its spheres that
// stand still whose material is diffuse_light with one solid color. -p nee and -p mis fall back to -p material in a
// scene with none and no environment map, as the books' strategies that aim at lamps do without a lights list.
int count_lamps(const scene_desc& scene);

// The number type the GPU computes with. A GeForce card does arithmetic on 32-bit floats about
// 64 times as fast as on the 64-bit doubles the CPU ray tracer uses, so float32 is the GPU's natural choice and
// float64 is there to compare with.
enum class precision
{
    float32,
    float64,
};

// The longest a launch may be planned to take. Planned launches can run several times longer than planned (see
// launch_planner in renderer.cu), so this stays far below the 2 seconds after which Windows stops a kernel.
inline constexpr double max_launch_seconds = 0.5;

// The largest side of a pixel's grid of samples: a round's passes are numbered with ints, at most 46340² of them
inline constexpr int max_sqrt_spp = 46340;

struct render_options
{
    // Renders with different seeds have statistically independent noise, as long as the seeds differ in their lower
    // 32 bits: a round of strata draws the random numbers of seed + round·2^32 (see round_seed_step)
    std::uint64_t seed = 0;
    precision arithmetic = precision::float32;
    // Called on the host as passes complete (a pass adds one more sample, a camera ray and its bounces, to every pixel,
    // and one kernel launch may complete several), with the number of passes done so far and their total. May be
    // empty.
    std::function<void(std::int64_t done, std::int64_t total)> progress;
    // Render with the kernel for all three books even where the first book's kernel would do (see the kernels in
    // renderer.cu): a check that the general code computes the first book's paths as Step 46 does, and a measure of
    // what its generality costs
    bool all_books_kernel = false;
    // Render with the quality kernel even a scene the books' kernels render (in float, sampled as the material
    // scatters): a check that its code computes the books' paths (the same images, bit for bit, when it was added).
    // The quality kernel refuses float64, all_books_kernel and a strategy other than material (std::invalid_argument).
    bool quality_kernel = false;
    // How long one kernel launch should take, in seconds, more than 0 and at most max_launch_seconds
    // (std::invalid_argument otherwise; see launch_planner in renderer.cu). The image doesn't depend on it, only the
    // speed: 0.1 rendered 2.5% faster than 0.05, and 0.025 5% slower.
    double launch_seconds = 0.05;
    // How many passes to render: 0 renders camera.sqrt_spp², one sample in each cell of every pixel's grid, the image
    // of the books' camera; a positive number renders exactly that many, in rounds of sqrt_spp² (see session); a
    // negative one throws std::invalid_argument
    std::int64_t passes = 0;
    // A tone map other than books fills render_result::rgba with the display's bytes after the last pass (see
    // session::display), the bytes RayTracingGPU saves as its PNG; books leaves it empty. The render doesn't depend on
    // it: the pixels and the rays are the same with any display.
    display_options display;
    bool optix_tracer = false;            // trace with OptiX (needs `optix`): float, -p material, nee or mis
    std::shared_ptr<optix_runtime> optix; // started by the program (start_optix); every session keeps it alive
    // Also the image denoised by OptiX's AI denoiser, after the last pass, into render_result::denoised and
    // denoised_rgba (needs `optix`; float only; what it refuses: see render()): biased, for the eye, while `pixels`
    // stays the image as rendered
    bool denoise = false;
};

// The kernel that rendered, as the CUDA runtime describes it: the numbers that decide how many of its threads a
// multiprocessor keeps in flight (its occupancy, see blocks_per_multiprocessor in renderer.cu)
struct kernel_info
{
    std::string name;                       // "first book", "all books", "quality" or "optix"
    int registers = 0;                      // per thread
    // per thread: the BVH stack, the scratch arrays of CUDA's math functions (sin's reduction of huge arguments takes
    // 48 bytes in double), and registers spilled to memory, if any. OptiX: its continuation stack, once the programs
    // are compiled (what replaced the BVH's stack)
    int local_bytes = 0;
    int blocks_per_multiprocessor = 0;      // as many as fit at once
    int threads_per_block = 0;              // 128
    int max_threads_per_multiprocessor = 0; // 1536 on the test GPU
    std::string programs;                   // OptiX: the embedded PTX's SHA-256, 8 hex digits; empty for CUDA kernels
};

// The kernels that run the quality kernel's integrator ("quality", "optix"): light strategies, float only
bool uses_quality_integrator(const kernel_info& kernel);

struct render_result
{
    std::vector<double> pixels; // width × height × 3 linear values (before gamma correction and clamping), row by row
                                // from the top: the layout of the CPU's image
    double seconds = 0;         // render() up to its last pass and the display's bytes: the BVH, copying to the GPU,
                                // all passes, copying back; a denoise isn't counted
    double kernel_seconds = 0;  // the passes alone, timed on the GPU
    std::uint64_t rays = 0;     // rays traced: camera rays and all their bounces
    kernel_info kernel;
    std::uint64_t shadow_rays = 0; // the quality kernel's rays toward a light, apart from `rays` (0 in the others)
    std::vector<std::uint8_t> rgba; // the display's bytes after a tone map other than the books' (empty otherwise)
    std::vector<std::uint8_t> denoised_rgba; // with denoise: the display's bytes of the denoised image
    std::vector<float> denoised;             // with denoise: the denoised linear image, 3 floats per pixel
    int denoise_tile = 0;                    // with denoise: the side of the tiles it was denoised in; 0: one invoke
    // With denoise: why the denoised image is missing (denoised and denoised_rgba are then empty), when the denoise
    // failed after the passes; empty when it ran, or wasn't asked for
    std::string denoise_error;
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

// How the host waits for the GPU (which of its waits: see query_device and session_options)
enum class wait_mode
{
    spin,  // asks the GPU again and again: reacts at once, but keeps a CPU core busy all the while (RayTracingGPU)
    sleep, // sleeps until the GPU signals: the core is free for the rest of the program (the window)
};

// Describes the first CUDA device and starts CUDA on it (a fraction of a second, once per program), so that render()
// times only the rendering. Throws std::runtime_error if there is no device (no NVIDIA GPU, or no driver).
// `wait` sets how the device's own waits wait, for the rest of the program: synchronous copies, and waits for the
// whole device. A session's waits for its launches follow its own session_options::wait.
device_info query_device(wait_mode wait = wait_mode::spin);

// Renders the scene on the first CUDA device. Throws std::runtime_error with CUDA's message if something fails, or if
// the scene is inconsistent (an index out of range, a checker before its textures...), and std::invalid_argument for
// options out of range (launch_seconds, passes, a display session::display refuses), before rendering anything.
// What denoise can't do that is known before the passes is refused before anything is rendered too: std::logic_error
// without `optix`, std::invalid_argument in float64, std::runtime_error for an image whose denoise images alone (88
// bytes per pixel) would pass the denoiser's 1024 MiB. What fails in the denoise itself, after the passes (the
// denoiser's memory plan, its creation), throws nothing: the result holds the rendered image as it would without
// denoise, and denoise_error says why its denoised twin is missing.
render_result render(const scene_desc& scene, const render_options& options);

// --- A render that goes on: the window's -------------------------------------------------------------------------

struct session_options
{
    std::uint64_t seed = 0;
    precision arithmetic = precision::float32;
    bool all_books_kernel = false;
    bool quality_kernel = false;  // see render_options
    double launch_seconds = 0.05; // how long a launch is planned to take (see render_options and launch_planner)
    // How the session waits for its launches. A CUDA event is waited for by spinning unless it is created to sleep,
    // whatever the device's flag (see query_device), so a sleeping session needs both: without the device's, its
    // copies would still spin. The session throws std::logic_error if query_device(wait_mode::sleep) hasn't set it.
    wait_mode wait = wait_mode::spin;
    bool optix_tracer = false;            // trace with OptiX (needs `optix`): float, -p material, nee or mis
    std::shared_ptr<optix_runtime> optix; // started by the program (start_optix); every session keeps it alive
};

// How far a view's image has come: `passes` complete passes, and rows [0, rows) of the next one (a pass that takes
// longer than a launch should is rendered in bands of rows from the top, see launch_planner in renderer.cu). Ordered
// by passes, then rows: "= default" has the compiler compare the members in that order.
struct accumulation
{
    std::int64_t passes = 0;
    int rows = 0;
    auto operator<=>(const accumulation&) const = default;
};

// What one call of session::render_to or render_for did
struct render_report
{
    accumulation state;                // where the image is after the call
    int launches = 0;                  // kernel launches
    double gpu_seconds = 0;            // their time on the GPU
    double longest_launch_seconds = 0; // the longest of them
};

// The seeds of a render's rounds of strata (see session): round r draws the random numbers of seed + r·round_seed_step.
// A seed given on a command line is a 32-bit number, so two different ones never share the seed of any round.
inline constexpr std::uint64_t round_seed_step = std::uint64_t{1} << 32;

// A render that goes on: the scene stays in the GPU's memory, and the image of one view gains samples on demand, a
// few at a time. render() is a session whose one view is rendered to its passes; the window keeps a session for as
// long as it shows a scene, and gives it a new camera whenever the view changes.
//
// Rounds of strata. Every pixel's samples are spread over a grid of k × k cells, k = camera().sqrt_spp: pass p of a
// round samples cell (p % k, p / k) (see start_path in camera.cuh), so a round of k² passes covers every cell once.
// The books' camera renders one round. A session renders any number of passes, round after round: pass n of a pixel
// is pass n % k² of round n / k², with the random numbers of seed + (n / k²)·round_seed_step. So each complete round is
// a whole grid of samples with noise of its own, round 0 is the books' image, and a render of N passes adds N samples
// of the same image whatever the rounds. A round that is not complete has sampled only the top rows of cells of every
// grid, and the left cells of its last row: its samples lean up, and in their last row left, by up to (k - 1) / (2k)
// of a pixel each way, less as the round fills and diluted by the complete rounds before it. A grid of k <= 3 has
// rounds of at most 9 passes, and a launch never spans two rounds, so such grids are a check, not a faster render.
//
// A session renders exactly the samples render() renders: pass n of a pixel is sample n of render(), with the same
// random numbers, and every pixel adds its samples in the order of the passes. So a view rendered to N passes, however
// many calls and launches that took, is render()'s image of N passes bit for bit (render_options::passes = N).
class session
{
public:
    // Checks the scene, builds its BVH, copies it into the GPU's memory and picks the kernel (see the kernels in
    // renderer.cu), or with optix_tracer has OptiX build its acceleration structures. The view is scene.camera. Throws
    // as render() does; std::logic_error for optix_tracer without a runtime, std::invalid_argument for what OptiX's
    // tracer can't render and for triangle meshes without it.
    session(const scene_desc& scene, const session_options& options);
    ~session();
    session(session&&) noexcept;
    session& operator=(session&&) noexcept;

    // A new view: the image starts over from no samples. The strategy must stay the scene's, as it decided the lights
    // in the GPU's memory and the kernel (another one throws std::invalid_argument); a camera the renderer can't render
    // throws std::runtime_error and keeps the old view. If the GPU fails on the way (its memory can't hold the new
    // image), the session has no view: has_view() is false, and the renders and what they report throw
    // std::logic_error until a set_camera succeeds.
    void set_camera(const camera_desc& camera);
    const camera_desc& camera() const; // the last view set successfully
    bool has_view() const;
    accumulation state() const;

    // Renders until the image is at `target`, launch after launch. on_pass, if given, is called with the number of
    // complete passes whenever a launch completes some; it must not call the session. Throws std::invalid_argument for
    // a target behind the image or with rows past the image's.
    render_report render_to(accumulation target, const std::function<void(std::int64_t passes)>& on_pass = {});
    // Renders launch after launch for about `seconds` of GPU time (0 to 3600), but never past pass_limit complete
    // passes; a pass_limit the image has reached does nothing. After a first launch, it ends early rather than make a
    // launch shorter than a quarter of a full one (`seconds`, or the session's launch_seconds if that is less). Below
    // the limit, it renders at least until the view has one complete pass, however long that takes, in launches the
    // planner sizes (see launch_planner in renderer.cu).
    // A call may end inside a later pass (state().rows > 0); pixels() then needs the pass finished
    // (render_to({passes + 1, 0})). Throws std::invalid_argument for a pass_limit out of range or a time out of range.
    render_report render_for(double seconds, std::int64_t pass_limit);

    // The image of the complete passes: the sums of the samples divided by their number, as render_result::pixels.
    // Throws std::logic_error unless the image has complete passes and no rows of the next one.
    std::vector<double> pixels() const;

    // The image as the screen shows it: 4 bytes per pixel, red, green, blue and 255, rows from the top, each channel
    // the byte the PNG files hold (the CPU's image::save_png). Rows that the pass in progress has rendered hold one
    // sample more than the others, and are divided by their own number of samples, so the image never shows a band
    // brighter or darker than the rest. Computed on the GPU and copied into memory of this session's, valid until the
    // next display() of either kind or set_camera. Throws std::logic_error unless the image has a complete pass.
    std::span<const std::uint8_t> display();
    // The image through a display transform. books gives display()'s bytes, by calling it; another tone map converts
    // the same two bands of rows, each with its own number of samples, with tonemap.cu's kernel, into the same memory.
    // Throws std::invalid_argument for an exposure that isn't a finite number within ±max_exposure, for books with an
    // exposure other than 0, and for another tone map in a float64 session (what is new computes in float only);
    // std::logic_error as display() does.
    std::span<const std::uint8_t> display(const display_options& options);

    // OptiX's AI denoiser on the image display() shows, the complete passes plus the rows of the pass in progress, each
    // band divided by its own number of samples; then the display transform: 4 bytes per pixel as display(options),
    // valid until the next display or set_camera. Float sessions with a runtime (session_options::optix, or
    // attach_optix) only: std::invalid_argument in float64, std::logic_error without a runtime, and as display() does.
    // The view's guides are computed at its first denoise and kept until set_camera; the image the last call denoised
    // (the same view, passes and rows) isn't denoised again. The images a denoise needs (88 bytes per pixel) are
    // allocated only if the GPU has them free with 256 MiB besides, else std::runtime_error with the sizes; a denoise
    // that throws keeps none of them.
    std::span<const std::uint8_t> display_denoised(const display_options& options); // float; needs optix
    // The view's last denoised image (display_denoised's floats, kept) through other display options, whatever the
    // image has rendered since, and with no denoise: the bytes display_denoised gave, or would give, for these options
    // on that image, in the same memory. std::logic_error when the view has none (none yet, set_camera since,
    // drop_denoised); std::invalid_argument as display_denoised.
    std::span<const std::uint8_t> retone_denoised(const display_options& options);
    std::vector<float> denoised_pixels() const;   // the view's last denoised floats, linear, 3 per pixel; throws as
                                                  // retone_denoised
    // The program's runtime for a float session made without one, for display_denoised only (the session keeps the
    // tracer it was made with): the window's denoiser in a CUDA session, with no new session and the image kept.
    // std::invalid_argument for none and in float64; a session that has a runtime keeps it: the same one again does
    // nothing, another throws std::logic_error.
    void attach_optix(std::shared_ptr<optix_runtime> runtime);
    // Frees what a denoise keeps on the GPU but the denoiser itself: the session's images of it and, with a runtime,
    // the denoiser's state and scratch (made again, with the guides, at the next display_denoised)
    void drop_denoised();
    double pick_distance(const std::array<double, 3>& origin, const std::array<double, 3>& direction);
                                                  // an OptiX session of a scene with meshes: the nearest
                                                  // surface's distance along the unit direction, ∞ for none

    std::uint64_t rays() const;        // rays traced since the view was set
    std::uint64_t shadow_rays() const; // the quality kernel's rays toward a light since then (see render_result)
    double kernel_seconds() const;     // the launches' time on the GPU since the view was set
    const kernel_info& kernel() const;

private:
    struct impl; // in renderer.cu, which knows the precision and CUDA's types
    std::unique_ptr<impl> impl_;
};
}
