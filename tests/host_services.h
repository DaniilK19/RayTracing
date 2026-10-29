// What the project's modules give, for GpuTests' .cu files, which can't import modules (nvcc): plain functions,
// implemented in gpu_tests.cpp with the modules. Host code only.
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
}
