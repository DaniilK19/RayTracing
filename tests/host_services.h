// What the project's modules give, for GpuTests' .cu files, which can't import modules (nvcc): plain functions,
// implemented in gpu_tests.cpp with the modules. Host code only.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
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

// The display's bytes as the linear values the programs save them as (b → ((b + 0.5)/256)², see bytes_to_image in
// gpu_scene.ixx): RGB doubles, rows from the top; rgba holds 4 bytes per pixel
std::vector<double> bytes_to_linear(std::span<const std::uint8_t> rgba, int width, int height);

// Those values saved by image::save_png (into the temp folder), read back by image::load (which turns each byte k into
// (k/255)^2.2), and each value turned into the byte whose (k/255)^2.2 is nearest: the bytes the PNG holds, RGB
std::vector<std::uint8_t> png_round_trip(std::span<const std::uint8_t> rgba, int width, int height);
}
