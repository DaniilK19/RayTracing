// What the project's modules give, for GpuTests' .cu files, which can't import modules (nvcc): plain functions,
// implemented with the modules in gpu_tests.cpp, and in scene_services.cpp those that build the GPU programs' scenes
// (the OptiX suites'). Host code only.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "renderer.h"

namespace gpu_test::host
{
// A folder of this run's own, removed when the program ends
std::filesystem::path temp_folder();

// The first folder holding `relative` ("assets/studio_small_09_1k.hdr", "src/gpu/ggx_energy.cuh"), walking up from the
// current folder, then from the executable's; std::runtime_error if there is none
std::filesystem::path find_repo_file(std::string_view relative);

// image::load: the floats stb_image gives (a .hdr's linear radiance, no gamma), rows from the top, copied exactly
gpu::image_desc load_image(const std::filesystem::path& file);

// The display's bytes as the linear values the programs save them as, by RayTracingGPU's own bytes_to_image
// (gpu_scene.ixx; b → ((b + 0.5)/256)²): RGB doubles, rows from the top; rgba holds 4 bytes per pixel
// (std::invalid_argument otherwise)
std::vector<double> bytes_to_linear(std::span<const std::uint8_t> rgba, int width, int height);

// bytes_to_image's image saved by image::save_png (into the temp folder), read back by image::load (which turns each
// byte k into (k/255)^2.2), and each value turned into the byte whose (k/255)^2.2 is nearest: the bytes the PNG holds,
// RGB
std::vector<std::uint8_t> png_round_trip(std::span<const std::uint8_t> rgba, int width, int height);

// Every scene of the GPU programs' list (gpu_scenes.ixx) with the display it opens with
struct scene_display
{
    std::string name;
    gpu::display_options display;
};
std::vector<scene_display> scene_displays();

// The translation of a scene's GGX side entries (gpu_scene.ixx, translate_parts), on a world of its own: a matte floor,
// a lamp, smoke, then a gloss's placeholder and a conductor's, their side entries listed in the other order
struct translation_check
{
    std::string aligned;              // empty when there is one side entry per material, each at its material's index
    bool same_entry_twice = false;    // a placeholder listed twice with the same entry: accepted
    bool two_entries_refused = false; // a placeholder with two different entries: std::logic_error
    bool unused_refused = false;      // an entry whose placeholder isn't in the world: std::logic_error
};
translation_check check_ggx_translation();

// --- The GPU programs' scenes, as RayTracingGPU builds them (scene_services.cpp) ------------------------------------

// The strategy for light a registered scene is built with, as RayTracingGPU's -p builds it
enum class scene_strategy
{
    own,      // the scene's own: its entry's, else its camera's
    material, // -p material
    mis,      // -p mis: light samples where the scene has a lamp of the light set or an environment map, else as
              // -p material (build_gpu_scene's rule)
};

// What a test changes in a registered scene, as RayTracingGPU's options do; what is 0 stays the scene's own
struct scene_request
{
    int width = 0;   // -w
    int height = 0;  // --height: the image's exact height
    int samples = 0; // -s: rounded to the books' square grid of strata, sqrt_spp² passes
    scene_strategy sampling = scene_strategy::own;
};

// The registered scene `name` as gpu_scene builds it for RayTracingGPU (build_gpu_scene: the CPU's own scene
// translated, the lights list for the strategies that aim at it, the environment map loaded from assets/);
// std::runtime_error with the program's message for a name that isn't in gpu_scene_list() or a scene the GPU can't
// render
gpu::scene_desc built_scene(std::string_view name, const scene_request& request = {});

// The scene's geometry, as the tracer check (suite optix-tracer) uploads it with tests/test_scene.h:
// every texture folded to one solid color (a checker to the mean of its two, an image to its mean, a marble to 0.5,
// the mean of noise_texture's ½(1 + sin)); the images, the Perlin tables and the lights list dropped (the camera then
// samples as the material scatters). Every primitive, medium and material is kept, and with them the lamps of the
// light set (diffuse_light of one solid color), the GGX side entries and the environment map.
gpu::scene_desc geometry_only(const gpu::scene_desc& scene);
}
