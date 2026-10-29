// Partition gpu_scenes:environments: the GPU's own scenes lit by an environment map, an HDR photograph of the
// surroundings in place of the camera's sky (environment_desc in gpu/renderer.h): a white furnace that checks the map's
// lookups, and two scenes of rough metals and glossy coats, in a photographer's studio and in sunlight, which
// gpu_scenes.ixx puts in its list. An interface partition that exports nothing: the scenes' functions have module
// linkage, seen by the module's other units and by no importer. (An internal partition, `module gpu_scenes:...;`
// without `export`, would say so in its first line, but MSVC compiles a .ixx file only as an interface.)
//
// The two maps are Poly Haven's (polyhaven.com), CC0, which allows committing them: "You can redistribute them". They
// are the 1k .hdr files exactly as published, 1024×512 texels, about 1.5 MB each
// (https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/<id>_1k.hdr): studio_small_09, a studio lit by softboxes, and
// kloofendal_48d_partly_cloudy_puresky, a sky with the sun 48° above the horizon and its ground replaced by sky.
//
// The metals' F0 are the measured base colors of the Unreal Engine's documentation (Physically Based Materials),
// taken as linear colors.
module;

// The renderer's types (a GGX side entry), before `import std;` (see gpu_scene.ixx)
#include "gpu/renderer.h"

export module gpu_scenes:environments;

import std;
import :parts;
import camera;
import hittable;
import material;
import quad;
import scenes;
import sphere;
import texture;
import vec3;

namespace
{
// Reflectances at normal incidence (F0) of the metals, and the scenes' matte white
constexpr color gold{1.000, 0.766, 0.336};
constexpr color silver{0.972, 0.960, 0.915};
constexpr color copper{0.955, 0.637, 0.538};
constexpr color aluminum{0.913, 0.921, 0.925};
constexpr color matte_white{0.8, 0.8, 0.8};

// A rough metal: its placeholder is the books' metal with the albedo F0 and no fuzz, which the GGX side entry turns
// into a GGX conductor of this roughness (r = 0: the perfect mirror). Each sphere gets a placeholder of its own: the
// side entries are found by the placeholder's address, so two of them must never share one.
void add_conductor(gpu_scene_parts& parts, const point3& center, double radius, const color& f0, double roughness)
{
    const auto placeholder = std::make_shared<metal>(f0, 0.0);
    parts.microfacets.emplace_back(
        placeholder.get(), gpu::microfacet_desc{.kind = gpu::microfacet_kind::conductor, .roughness = roughness});
    parts.cpu.world.add(std::make_unique<sphere>(center, radius, placeholder));
}

// A glossy coat over a matte color: the placeholder is the matte lambertian, under a GGX coat of F0 0.04
void add_gloss(gpu_scene_parts& parts, const point3& center, double radius, const color& base, double roughness)
{
    const auto placeholder = std::make_shared<lambertian>(base);
    parts.microfacets.emplace_back(
        placeholder.get(), gpu::microfacet_desc{.kind = gpu::microfacet_kind::gloss, .roughness = roughness});
    parts.cpu.world.add(std::make_unique<sphere>(center, radius, placeholder));
}

// A ground of side `size`, centered under the origin at height y, facing up (u × v = +y)
std::unique_ptr<quad> ground(double size, double y, std::shared_ptr<const material> mat)
{
    return std::make_unique<quad>(point3{-size / 2, y, -size / 2}, vec3{0, 0, size}, vec3{size, 0, 0}, std::move(mat));
}

// The showcase scenes' camera: 16:9, 1920 pixels wide, 1024 samples, at most 10 rays a path
camera_settings showcase_camera(const point3& lookfrom, const point3& lookat)
{
    return {.aspect_ratio = 16.0 / 9.0,
            .image_width = 1920,
            .samples_per_pixel = 1024,
            .max_depth = 10,
            .sampling = sampling_strategy::material,
            .vfov = 30,
            .lookfrom = lookfrom,
            .lookat = lookat};
}
}

// furnace (gpu_scenes.ixx) again, its sphere and its camera, under a map of radiance 1 everywhere instead of a white
// sky. Both of the map's lookups give exactly 1 there: the nearest texel is 1, and the bilinear filter's a + s·(b − a)
// with a = b = 1 is 1 for every s. So under -p material every pixel is exactly 1.0, and the image must equal furnace's
// in the quality kernel bit for bit: the same rays, the same random numbers, the same products. The camera's sky is
// black, so a ray that missed the map would show.
gpu_scene_parts env_furnace()
{
    hittable_list world;
    world.add(std::make_unique<sphere>(point3{0, 0, 0}, 1.0, std::make_shared<lambertian>(color{1, 1, 1})));
    const camera_settings camera{.aspect_ratio = 16.0 / 9.0,
                                 .image_width = 480,
                                 .samples_per_pixel = 64,
                                 .max_depth = 10,
                                 .sampling = sampling_strategy::material,
                                 .vfov = 30,
                                 .lookfrom = {0, 0, 6},
                                 .lookat = {0, 0, 0}};
    return {.cpu = scene{std::move(world), camera}, .environment = {.source = env_source::white}};
}

// Material balls in a photographer's studio, on a grey floor: in front, gold from the perfect mirror (r = 0) to rough
// (r = 0.8), left to right; behind, silver and copper (r = 0.25), red and blue glossy coats (r = 0.2 and 0.5) and a
// matte white sphere. Softboxes give large, bright highlights of moderate contrast (the brightest 4096 texels hold 42%
// of the map's light), the light that sampling the materials alone handles well: the scene that compares -p material
// with -p mis.
//
// The back row stands in the front row's gaps as the camera sees them (±0.6 and ±1.8 at the front row's depth along
// the view, 7.23, are ±0.71 and ±2.13 at the back row's, 8.57): from 16.7° above, a sphere straight behind a front one
// would show only its top third. Four gaps hold four spheres, so the white one stands further back in the middle, where
// it rises above the front row between the copper and the red. The camera stands back far enough for the whole front
// row: its outer spheres reach 22.1° from the middle, in a frame 25.5° wide either way.
gpu_scene_parts studio()
{
    gpu_scene_parts parts;
    parts.cpu.world.add(ground(40, 0, std::make_shared<lambertian>(color{0.5, 0.5, 0.5})));
    // The front row: each sphere's x and roughness
    constexpr std::array<std::pair<double, double>, 5> front{
        {{-2.4, 0.0}, {-1.2, 0.15}, {0, 0.3}, {1.2, 0.5}, {2.4, 0.8}}};
    for (const auto& [x, roughness] : front)
    {
        add_conductor(parts, {x, 0.5, 0.6}, 0.5, gold, roughness);
    }
    add_conductor(parts, {-2.13, 0.5, -0.8}, 0.5, silver, 0.25);
    add_conductor(parts, {-0.71, 0.5, -0.8}, 0.5, copper, 0.25);
    add_gloss(parts, {0.71, 0.5, -0.8}, 0.5, color{0.7, 0.05, 0.05}, 0.2);
    add_gloss(parts, {2.13, 0.5, -0.8}, 0.5, color{0.05, 0.15, 0.6}, 0.5);
    parts.cpu.world.add(std::make_unique<sphere>(point3{0, 0.5, -2.2}, 0.5, std::make_shared<lambertian>(matte_white)));
    parts.cpu.camera = showcase_camera({0, 2.65, 7.5}, {0, 0.4, 0});
    parts.environment = {.source = env_source::asset, .asset = "studio_small_09_1k.hdr", .yaw_degrees = 0, .scale = 1};
    return parts;
}

// Three spheres on a checkered ground in the sun: matte white, copper (r = 0.35) and rough aluminum (r = 0.2), with a
// small green glossy one in front. The sun's 4 brightest texels hold 42% of the map's light in 10⁻⁴ sr: a matte
// floor's own samples find them once in about 40,000 bounces (a relative deviation of 127 per sample), so this scene
// needs the light samples. No mirror and no glass: a path from the floor through a mirror to the sun has no light
// sample to find it, and would sparkle.
//
// The ground lies 1 mm below y = 0: the checker is solid (the parity of ⌊x/0.5⌋ + ⌊y/0.5⌋ + ⌊z/0.5⌋), and on a floor
// exactly at y = 0, a layer's boundary, ⌊y/0.5⌋ would be 0 or -1 by the hit point's rounding, a noise of squares. It is
// 200 wide, so its edge falls about 1° below the horizon (the map's lower half is sky).
gpu_scene_parts sunlit()
{
    gpu_scene_parts parts;
    const auto checker = std::make_shared<checker_texture>(0.5, color{0.2, 0.2, 0.2}, color{0.7, 0.7, 0.7});
    parts.cpu.world.add(ground(200, -0.001, std::make_shared<lambertian>(checker)));
    parts.cpu.world.add(std::make_unique<sphere>(point3{-2.2, 1, 0}, 1.0, std::make_shared<lambertian>(matte_white)));
    add_conductor(parts, {0, 1, 0}, 1, copper, 0.35);
    add_conductor(parts, {2.2, 1, 0}, 1, aluminum, 0.2);
    add_gloss(parts, {1.1, 0.4, 2}, 0.4, color{0.08, 0.45, 0.12}, 0.15);
    parts.cpu.camera = showcase_camera({0, 2, 9}, {0, 0.9, 0});
    parts.environment = {.source = env_source::asset,
                         .asset = "kloofendal_48d_partly_cloudy_puresky_1k.hdr",
                         .yaw_degrees = 0,
                         .scale = 1};
    return parts;
}
