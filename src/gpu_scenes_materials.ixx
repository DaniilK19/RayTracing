// Partition gpu_scenes:materials, the GPU's own scenes of materials beyond the books: rough metals and glossy coats
// (GGX, see gpu/ggx.cuh). The CPU's classes can't describe them, so each GGX material is a CPU placeholder that carries
// its color, completed by a side entry with its roughness (gpu_scene_parts::microfacets), found by the placeholder's
// address when the scene is translated (gpu_scene.ixx): a rough metal's placeholder is a metal of fuzz 0 whose albedo
// is F0, a glossy coat's a matte lambertian whose texture is the color under the coat. A scene with GGX materials
// renders with the quality kernel only (see the kernels in gpu/renderer.cu): the books' kernels would render the
// placeholders.
module;

// The renderer's types (a GGX side entry), before `import std;` (see gpu_scene.ixx)
#include "gpu/renderer.h"

export module gpu_scenes:materials;

import std;
import :parts;
import camera;
import hittable;
import material;
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
