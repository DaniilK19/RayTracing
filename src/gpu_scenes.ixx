// Module gpu_scenes: the scenes the GPU programs render. They begin with the books' 13 from scenes.ixx, in their order
// and with their names, built by the CPU's own code; then come the GPU's own scenes, which RayTracing doesn't have:
// scenes.ixx is the frozen CPU renderer's, and its scene_list stays the CPU program's. A scene of the GPU's own is
// still built from the CPU's classes (spheres, quads, materials, the camera's settings), so the GPU programs translate
// and show it like any other, and the window's double-click finds its objects.
export module gpu_scenes;

export import :parts;
// The scenes beyond the books, a partition per kind. An interface partition must be exported by the primary
// interface ([module.unit]/3); they export nothing of their own, and their scenes' functions are the module's.
export import :environments;
export import :materials;
export import :meshes; // exports to_mesh_desc too, which gpu_scene.ixx's translation takes

import std;
import camera;
import hittable;
import material;
import scenes;
import sphere;
import vec3;

// The book's scene I of scene_list. A function template: each instantiation is a function of its own, whose address
// the list below can hold, as it holds the GPU's own scenes' functions.
template <std::size_t I>
gpu_scene_parts book_scene()
{
    return {scene_list[I].build()};
}

// A white furnace: a matte sphere of albedo 1 under a sky of radiance 1 everywhere. Sampled as the material scatters
// (its strategy), every path ends in the sky carrying all of its light, so every pixel is exactly 1.0, the sphere's and
// the background's alike, whatever the samples: a renderer that loses or adds energy anywhere shows it in every pixel,
// with no noise to hide it. In float too, exactly: the sky's (1 - a)·1 + a·1 rounds to 1 for every a; the sphere
// absorbs nothing (albedo 1 multiplies exactly); a cosine bounce leaves at least 2^-12 above the surface (the largest
// random number is 1 - 2^-24, see random_cosine_direction), far beyond rounding, so none is dropped below it; and it
// leaves outward, so closest_hit's rule for the sphere a ray leaves never lets it meet the sphere again. (Under
// -p uniform the bounces are weighted by 4 cos θ, and the image is 1 only on average.)
gpu_scene_parts furnace()
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
                                 .lookat = {0, 0, 0},
                                 .background = {.bottom = {1, 1, 1}, .top = {1, 1, 1}}};
    return {scene{std::move(world), camera}};
}

// The list: the books' scenes (an index sequence 0, 1, ..., 12 expands into one entry per scene), then the GPU's own.
// consteval, so the whole array is made at compile time.
template <std::size_t... I>
consteval auto make_scene_list(std::index_sequence<I...>)
{
    return std::array{
        gpu_scene_entry{scene_list[I].name, scene_list[I].description, scene_role::book, &book_scene<I>}...,
        gpu_scene_entry{"furnace", "a white sphere under a white sky: every pixel exactly 1 (a check of the renderers)",
                        scene_role::test, &furnace},
        gpu_scene_entry{"ggx-furnace", "white metals, a mirror and four GGX roughnesses, under a white sky: the light "
                        "each keeps", scene_role::test, &ggx_furnace},
        gpu_scene_entry{"gloss-furnace", "white matte spheres under four glossy coats, under a white sky: the light "
                        "each keeps", scene_role::test, &gloss_furnace},
        gpu_scene_entry{"cornell-ggx", "the Cornell box with a block of rough aluminum and a glossy white one",
                        scene_role::showcase, &cornell_ggx, strategy::mis, cornell_ggx_display},
        gpu_scene_entry{"veach", "Veach's four plates (1997): a metal from sharp to rough under four lamps of one "
                        "power and four sizes", scene_role::showcase, &veach_plates, strategy::mis, veach_display},
        gpu_scene_entry{"env-furnace", "the white sphere under a white environment map: every pixel exactly 1 again",
                        scene_role::test, &env_furnace},
        gpu_scene_entry{"studio", "gold, silver, copper and glossy spheres in a photo studio's light (an HDR map)",
                        scene_role::showcase, &studio, strategy::mis, studio_display},
        gpu_scene_entry{"sunlit", "matte, copper and aluminum spheres in the sunlight of an HDR sky",
                        scene_role::showcase, &sunlit, strategy::mis, sunlit_display},
        gpu_scene_entry{"sun-test", "matte spheres in the sunlight of an HDR sky: -p nee against -p mis (a check of "
                        "the light samples)", scene_role::test, &sun_test, strategy::mis},
        gpu_scene_entry{"cornell-triangles", "the Cornell box with its walls and blocks made of 34 triangles: "
                        "cornell's image again (OptiX)", scene_role::test, &cornell_triangles, strategy::mis},
        gpu_scene_entry{"mesh-furnace", "two white icospheres of triangles, one mirrored, under a white sky: every "
                        "pixel exactly 1 (OptiX)", scene_role::test, &mesh_furnace},
        gpu_scene_entry{"mesh-furnace-smooth", "a white icosphere with smooth normals under a white sky: the light "
                        "its faces absorb (OptiX)", scene_role::test, &mesh_furnace_smooth},
        gpu_scene_entry{"instances", "one triangle mesh placed four times, turned, stretched and mirrored, in a "
                        "photo studio (OptiX)", scene_role::test, &instances, strategy::mis},
        gpu_scene_entry{"instances-baked", "the same four moved into the world on the CPU: the reference of "
                        "instances (OptiX)", scene_role::test, &instances_baked, strategy::mis},
    };
}

constexpr auto gpu_scenes = make_scene_list(std::make_index_sequence<scene_list.size()>{});

export std::span<const gpu_scene_entry> gpu_scene_list()
{
    return gpu_scenes;
}

// The entry of a name, or nullptr
export const gpu_scene_entry* find_gpu_scene(std::string_view name)
{
    const auto it = std::ranges::find(gpu_scenes, name, &gpu_scene_entry::name);
    return it == gpu_scenes.end() ? nullptr : &*it;
}

// The help's list of scenes, after its "Scenes:" line: the books' scenes as RayTracing prints them, the default, then
// each other group that has scenes under its heading. A scene added to the list appears in both programs' help.
export std::string scenes_usage(std::string_view default_scene)
{
    // Each group's names in a column as wide as its longest, at least the books' 10 (RayTracing's own list)
    const auto group = [](scene_role role) {
        std::size_t width = 10;
        for (const gpu_scene_entry& entry : gpu_scenes)
        {
            width = entry.role == role ? std::max(width, entry.name.size()) : width;
        }
        std::string text;
        for (const gpu_scene_entry& entry : gpu_scenes)
        {
            if (entry.role == role)
            {
                text += std::format("  {:{}} {}\n", entry.name, width, entry.description);
            }
        }
        return text;
    };
    constexpr std::array headings{
        std::pair{scene_role::showcase, "Scenes beyond the books (the quality kernel, float):"},
        std::pair{scene_role::test, "Test scenes (checks of the renderers):"},
    };
    std::string text = group(scene_role::book) + std::format("  Default: {}.\n", default_scene);
    for (const auto& [role, heading] : headings)
    {
        if (const std::string lines = group(role); !lines.empty())
        {
            text += std::format("\n{}\n{}", heading, lines);
        }
    }
    return text;
}
