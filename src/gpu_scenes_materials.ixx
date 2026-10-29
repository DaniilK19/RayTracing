// Partition gpu_scenes:materials, the GPU's own scenes of materials beyond the books: rough metals and glossy coats
// (GGX, see gpu/ggx.cuh). The CPU's classes can't describe them, so each GGX material is a CPU placeholder that carries
// its color, completed by a side entry with its roughness (gpu_scene_parts::microfacets), found by the placeholder's
// address when the scene is translated (gpu_scene.ixx): a rough metal's placeholder is a metal of fuzz 0 whose albedo
// is F0, a glossy coat's a matte lambertian whose texture is the color under the coat. A scene with GGX materials
// renders with the quality kernel only (see the kernels in gpu/renderer.cu): the books' kernels would render the
// placeholders.
//
// Two white furnaces check the lobes alone, under a sky of radiance 1. Two scenes put them under lamps, where the
// quality kernel's light samples (-p nee, -p mis) ask the lobes for their values and densities in directions the lobes
// didn't draw: the Cornell box with a block of rough metal and a glossy one, and Veach's four plates.
module;

// The renderer's types (a GGX side entry, a display), before `import std;` (see gpu_scene.ixx)
#include "gpu/renderer.h"

export module gpu_scenes:materials;

import std;
import :parts;
import camera;
import hittable;
import instance;
import material;
import quad;
import scenes;
import sphere;
import vec3;

// The helpers are this file's own (an unnamed namespace): the other partitions of gpu_scenes may have helpers of the
// same names
namespace
{
using side_entries = std::vector<std::pair<const material*, gpu::microfacet_desc>>;

// A sphere of a GGX material: its placeholder is a material of its own, as the translator finds a side entry by its
// material's address, so two GGX materials never share a placeholder
void add_ggx_sphere(hittable_list& world, side_entries& microfacets, const point3& center, double radius,
                    std::shared_ptr<const material> placeholder, gpu::microfacet_desc side)
{
    microfacets.emplace_back(placeholder.get(), side);
    world.add(std::make_unique<sphere>(center, radius, std::move(placeholder)));
}

// A rough metal of color F0 (a metal of fuzz 0 as its placeholder)
void add_conductor(hittable_list& world, side_entries& microfacets, const point3& center, double radius,
                   const color& f0, double roughness)
{
    add_ggx_sphere(world, microfacets, center, radius, std::make_shared<const metal>(f0, 0.0),
                   {.kind = gpu::microfacet_kind::conductor, .roughness = roughness});
}

// A glossy coat over the matte color ρ (a lambertian of that color as its placeholder)
void add_gloss(hittable_list& world, side_entries& microfacets, const point3& center, double radius, const color& rho,
               double roughness)
{
    add_ggx_sphere(world, microfacets, center, radius, std::make_shared<const lambertian>(rho),
                   {.kind = gpu::microfacet_kind::gloss, .roughness = roughness});
}

// The furnaces' camera: from 60 away with a field of 6°, nearly an orthographic view, so that each sphere's centre is
// seen almost head-on (cos θv ≥ 0.969 over the 5 × 5 pixels around it), and nothing but the sky of radiance 1 around
camera_settings furnace_camera()
{
    return {.aspect_ratio = 16.0 / 9.0,
            .image_width = 480,
            .samples_per_pixel = 256,
            .max_depth = 10,
            .sampling = sampling_strategy::material,
            .vfov = 6,
            .lookfrom = {0, 0, 60},
            .lookat = {0, 0, 0},
            .background = {.bottom = {1, 1, 1}, .top = {1, 1, 1}}};
}
}

// A white furnace for the rough metals: five spheres of a metal that reflects everything (F0 = 1) under a sky of
// radiance 1, from left to right the mirror (r = 0) and GGX at α = r² = 0.05, 0.3, 0.6 and 1, widths whose single
// scattering GpuTests checks against an independent table (ggx/energy-table). A convex sphere sees nothing but the
// sky, so a pixel's expected value is the light its lobe keeps at the pixel's angle, E(μ) = ∫ f·cos θl dωl with F = 1:
// exactly 1 for the mirror, and 1 again for a GGX metal, whose single scattering alone keeps less (0.31 at α = 1 seen
// head-on) and Turquin's compensation gives back the rest, the light the facets pass on to one another (ggx.cuh).
// The spheres stand 2 apart with a radius of 0.4: from the points of a sphere's centre patch a neighbour reaches at
// most about 13° above the horizon (none at the patch's centre), where even the widest lobe sends under 0.4% of its
// light, so the patch shows its own lobe alone.
gpu_scene_parts ggx_furnace()
{
    hittable_list world;
    side_entries microfacets;
    const std::array roughness{0.0, std::sqrt(0.05), std::sqrt(0.3), std::sqrt(0.6), 1.0};
    for (std::size_t i = 0; i < roughness.size(); ++i)
    {
        add_conductor(world, microfacets, point3{-4.0 + 2.0 * static_cast<double>(i), 0, 0}, 0.4, color{1, 1, 1},
                      roughness[i]);
    }
    return {.cpu = scene{std::move(world), furnace_camera()}, .microfacets = std::move(microfacets)};
}

// The same furnace for the glossy coat: four white matte spheres (ρ = 1) under the coat, from left to right at
// α = r² = 0.01, 0.1, 0.3 and 1, at x = −3, −1, 1, 3. A pixel's expected value is the light coat and base keep
// together, E(μ) = E_coat(μ) + 0.96(1 − (1 − μ/2)⁵): head-on about 0.97 under the smoother coats and 0.94 under the
// roughest, below 1 everywhere, as the coupled diffuse term is an approximation that loses a little light and never
// creates any.
gpu_scene_parts gloss_furnace()
{
    hittable_list world;
    side_entries microfacets;
    const std::array roughness{std::sqrt(0.01), std::sqrt(0.1), std::sqrt(0.3), 1.0};
    for (std::size_t i = 0; i < roughness.size(); ++i)
    {
        add_gloss(world, microfacets, point3{-3.0 + 2.0 * static_cast<double>(i), 0, 0}, 0.4, color{1, 1, 1},
                  roughness[i]);
    }
    return {.cpu = scene{std::move(world), furnace_camera()}, .microfacets = std::move(microfacets)};
}

// --- Under the lamps -------------------------------------------------------------------------------------------------

namespace
{
// A rough metal of color F0 as a material for any shape: its placeholder, a metal of fuzz 0, with its side entry. A
// material of its own at every call, as for the spheres above.
std::shared_ptr<const material> rough_metal(side_entries& microfacets, const color& f0, double roughness)
{
    auto placeholder = std::make_shared<const metal>(f0, 0.0);
    microfacets.emplace_back(placeholder.get(),
                             gpu::microfacet_desc{.kind = gpu::microfacet_kind::conductor, .roughness = roughness});
    return placeholder;
}

// A glossy coat over the matte color ρ, the same way: its placeholder is a lambertian of that color
std::shared_ptr<const material> glossy_coat(side_entries& microfacets, const color& rho, double roughness)
{
    auto placeholder = std::make_shared<const lambertian>(rho);
    microfacets.emplace_back(placeholder.get(),
                             gpu::microfacet_desc{.kind = gpu::microfacet_kind::gloss, .roughness = roughness});
    return placeholder;
}

// The books' Cornell box without its blocks, copied from scenes.ixx (add_cornell_walls and cornell_lamp: scenes.ixx is
// the frozen CPU renderer's module, which exports its list alone): a room 555 on a side, open toward the camera, its
// left wall green and its right one red as the camera sees them, and the lamp of radiance 15, 130 × 105, one unit below
// the ceiling, facing down. In the books' order: the walls, then the lamp.
void add_cornell_room(hittable_list& world)
{
    const auto red = std::make_shared<lambertian>(color{0.65, 0.05, 0.05});
    const auto white = std::make_shared<lambertian>(color{0.73, 0.73, 0.73});
    const auto green = std::make_shared<lambertian>(color{0.12, 0.45, 0.15});
    world.add(std::make_unique<quad>(point3{555, 0, 0}, vec3{0, 555, 0}, vec3{0, 0, 555}, green));       // left
    world.add(std::make_unique<quad>(point3{0, 0, 0}, vec3{0, 555, 0}, vec3{0, 0, 555}, red));           // right
    world.add(std::make_unique<quad>(point3{0, 0, 0}, vec3{555, 0, 0}, vec3{0, 0, 555}, white));         // floor
    world.add(std::make_unique<quad>(point3{555, 555, 555}, vec3{-555, 0, 0}, vec3{0, 0, -555}, white)); // ceiling
    world.add(std::make_unique<quad>(point3{0, 0, 555}, vec3{555, 0, 0}, vec3{0, 555, 0}, white));       // back
    world.add(std::make_unique<quad>(point3{343, 554, 332}, vec3{-130, 0, 0}, vec3{0, 0, -105},
                                     std::make_shared<diffuse_light>(color{15, 15, 15})));
}

// A block of the Cornell box (scenes.ixx's tall_block and short_block): a box from the origin to `size`, turned about
// the vertical by `degrees`, then moved by `offset` (rotate_y turns about the origin, so the turn comes first)
std::unique_ptr<hittable> cornell_block(const point3& size, double degrees, const vec3& offset,
                                        std::shared_ptr<const material> mat)
{
    std::unique_ptr<hittable> block = box(point3{0, 0, 0}, size, std::move(mat));
    block = std::make_unique<rotate_y>(std::move(block), degrees);
    return std::make_unique<translate>(std::move(block), offset);
}

// The luminance Y of a linear color, by Rec. 709's weights: the sRGB primaries, which the renderer's colors are in
constexpr double luminance(const color& c)
{
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}
}

// The books' Cornell box with its two blocks made of the new materials: the tall one rough aluminum (F0 (0.913, 0.921,
// 0.925), measured, from Epic's table; r = 0.3), the short one white under a glossy coat (ρ 0.73, r = 0.2). Where the
// books' aluminum scene has a mirror, which no light sample can use, both blocks here have a density: under -p nee and
// -p mis they take a light sample at every bounce, and their values and densities at the lamp's directions enter the
// weights. So this is the scene that checks the GGX lobes under a lamp: -p mis must converge to -p material's image,
// with less noise (scripts/verify/compare.ps1).
gpu_scene_parts cornell_ggx()
{
    hittable_list world;
    side_entries microfacets;
    add_cornell_room(world);
    world.add(cornell_block(point3{165, 330, 165}, 15, vec3{265, 0, 295},
                            rough_metal(microfacets, color{0.913, 0.921, 0.925}, 0.3)));
    world.add(cornell_block(point3{165, 165, 165}, -18, vec3{130, 0, 65},
                            glossy_coat(microfacets, color{0.73, 0.73, 0.73}, 0.2)));
    // The books' camera of the Cornell box (scenes.ixx's cornell_camera), at 32² samples
    const camera_settings camera{.aspect_ratio = 1.0,
                                 .image_width = 1080,
                                 .samples_per_pixel = 1024,
                                 .max_depth = 50,
                                 .sampling = sampling_strategy::material,
                                 .vfov = 40,
                                 .lookfrom = {278, 278, -800},
                                 .lookat = {278, 278, 0}};
    return {.cpu = scene{std::move(world), camera}, .microfacets = std::move(microfacets)};
}

// Its display: AgX at 0 EV. The lamp's radiance of 15 sits just under AgX's white (2^4.03 = 16.3), and the walls' light
// around its middle grey of 0.18: the lamp gives the floor below it an irradiance of 15·130·105/554² = 0.67, a white
// wall's radiance is about 0.2 to 0.3 with the light the room passes around, and AgX shows it about as bright as the
// books' square root shows the books' Cornell box.
constexpr gpu::display_options cornell_ggx_display{gpu::tone_map::agx, 0.0};

// Veach's four plates (Veach 1997, chapter 9; Veach & Guibas 1995), the classic picture of multiple importance
// sampling: four plates of one metal, from sharp to rough, under four sphere lamps of one power and four sizes. Each of
// the two techniques fails somewhere. The material's rays rarely find a small lamp from a rough plate: a speck in a
// wide lobe. The light samples rarely fall into a sharp plate's narrow lobe when the lamp is large: most of its cone
// reflects nowhere near the camera. -p material and -p nee each show their own failure as noise in the highlights,
// and -p mis shows all sixteen with little noise, as the power heuristic keeps the better technique of each pair.
//
// Each plate is 9 × 1, its long side along x (wide enough for the largest lamp's image in the sharpest plate), and its
// normal halves the angle between the camera C and the middle of the lamps' row L̄, n = unit(unit(C − P) +
// unit(L̄ − P)): so every plate mirrors the lamps' row toward the camera, whatever its tilt, and shows the four
// highlights along its middle line, without a trial render. Its short side is v = unit(n × x̂), so u × v = 9n: its front
// faces the camera and the lamps. From the top, the farthest, to the nearest: r = 0.05, 0.1, 0.2, 0.35, the sharpest
// plate the farthest. The plates stand apart, so that none hides another's highlights or shadows them: from every point
// of a plate, the camera's mirrored rays and the shadow rays toward L̄ pass at least 0.085 above the front edge of the
// plate behind it (0.018 toward the bottom of the largest lamp), and the camera sees the four with gaps of 1.7° to 1.9°
// between them.
//
// The lamps, left to right, have radii 0.05, 0.15, 0.4 and 1 and the radiance 4/r² times a tint of luminance 1, from
// warm to cool. A sphere of radiance L glows with the power π·L·4πr² = 16π² ≈ 158 for every radius (in luminance), as
// Veach's lamps all have one power: the smaller, the brighter (1600 at r = 0.05, 4 at r = 1). Behind them stands a dark
// matte wall (albedo 0.1), wide and tall enough to fill the frame and every plate's reflection, so that the plates show
// a little of the room around their highlights instead of a black sky.
gpu_scene_parts veach_plates()
{
    hittable_list world;
    side_entries microfacets;
    const point3 eye{0, 2, 15};
    const point3 lamp_row{0, 2.5, -3.5}; // L̄
    constexpr std::array plates{std::pair{point3{0, 0.4, -2.4}, 0.05}, std::pair{point3{0, -0.5, -0.9}, 0.1},
                                std::pair{point3{0, -1.2, 0.7}, 0.2}, std::pair{point3{0, -1.7, 2.2}, 0.35}};
    const color plates_f0{0.95, 0.93, 0.88};
    for (const auto& [centre, roughness] : plates)
    {
        const vec3 n = unit_vector(unit_vector(eye - centre) + unit_vector(lamp_row - centre));
        const vec3 u{9, 0, 0};
        const vec3 v = unit_vector(cross(n, vec3{1, 0, 0}));
        world.add(std::make_unique<quad>(centre - 0.5 * u - 0.5 * v, u, v,
                                         rough_metal(microfacets, plates_f0, roughness)));
    }
    constexpr std::array lamps{std::pair{0.05, color{1.0, 0.5, 0.2}}, std::pair{0.15, color{1.0, 0.8, 0.55}},
                               std::pair{0.4, color{0.8, 0.9, 1.0}}, std::pair{1.0, color{0.5, 0.7, 1.0}}};
    for (std::size_t i = 0; i < lamps.size(); ++i)
    {
        const auto& [radius, tint] = lamps[i];
        const color radiance = (4 / (radius * radius) / luminance(tint)) * tint;
        world.add(std::make_unique<sphere>(point3{-3.75 + 2.5 * static_cast<double>(i), 2.5, -3.5}, radius,
                                           std::make_shared<diffuse_light>(radiance)));
    }
    world.add(std::make_unique<quad>(point3{-15, -10, -6}, vec3{30, 0, 0}, vec3{0, 25, 0},
                                     std::make_shared<lambertian>(color{0.1, 0.1, 0.1})));
    // Aimed at the middle of what there is to see: from C, the largest lamp's top rises 4.6° above the horizontal and
    // the nearest plate's front edge lies 17.1° below it, so a frame of 28° pitched down by 6.1° keeps about 3° above
    // and below them. The sky is black.
    const camera_settings camera{.aspect_ratio = 16.0 / 9.0,
                                 .image_width = 1920,
                                 .samples_per_pixel = 1024,
                                 .max_depth = 10,
                                 .sampling = sampling_strategy::material,
                                 .vfov = 28,
                                 .lookfrom = eye,
                                 .lookat = {0, 0.4, 0}};
    return {.cpu = scene{std::move(world), camera}, .microfacets = std::move(microfacets)};
}

// Its display: AgX one stop down, as the scene is its highlights. A sphere lamp of radiance 4/r² gives a surface that
// faces it from a distance d the irradiance π·L·(r/d)² = 4π/d² whatever its size: the wall gets about 3 from the four
// lamps, a radiance of about 0.1 near them, and the roughest plate's highlights are about 2.6, the largest lamp 4. At
// −1 EV the wall stays a dark grey around the highlights, and these keep their tints 3 stops and more below AgX's
// white (16.3 at 0 EV); the small lamps and the sharp plates' images of them, up to 1600, are white at any exposure.
constexpr gpu::display_options veach_display{gpu::tone_map::agx, -1.0};
