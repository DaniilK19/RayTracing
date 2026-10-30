// What the project's modules give the mesh suites (tests/mesh_gpu_tests.cu), which nvcc compiles and which can't
// import modules: module meshes' generators and bake as the renderer's mesh_desc, a scene of the GPU programs' list as
// gpu_scene builds it, and the check of gpu_scene's model cache. Plain functions, implemented with the modules in
// tests/mesh_services.cpp (cmake/meshes.cmake adds it to GpuTests). Host code only.
#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

#include "renderer.h"

namespace gpu_test::mesh_host
{
// meshes::icosphere (radius 1 about the origin; smooth: normals = positions) and meshes::torus (about the y axis), as
// the renderer's mesh: the same floats, material_of all 0
gpu::mesh_desc icosphere(int level, bool smooth);
gpu::mesh_desc torus(double major, double minor, int around, int across);

// meshes::bake of a mesh: its vertices in world coordinates under object_to_world (row-major 3×4), its normals by the
// inverse transpose, corners 1 and 2 of every triangle swapped when the transform mirrors
gpu::mesh_desc bake(const gpu::mesh_desc& mesh, const std::array<double, 12>& object_to_world);

// The renderer's data of a scene of the GPU programs' list, as build_gpu_scene makes it without overrides (the scene's
// own camera and strategy, its lights list with it); std::runtime_error with gpu_scene's message if it can't
gpu::scene_desc built_scene(std::string_view name);

// gpu_scene's model cache (the last two models used, by canonical path, size and write time, each converted once into
// shared mesh_descs and textures), seen from outside, through the pointers each build holds and the models it read
// (gpu_build::models), in five builds:
//   1, 2  still-life twice: the second shares the first's meshes and textures, and reads nothing;
//   3     the vase alone (its file as a model path), after its file's write time moved 2 s on: read again, none of
//         its textures the first build's. The vase is the model still-life reads second, so a stale entry kept for its
//         old time would push out the bust;
//   4     still-life: reads nothing, the bust's meshes and textures still the first build's and the vase's textures the
//         third's (both models stayed);
//   5     still-life once the write time is put back: reads the vase alone, and only it (its entry of the old time went
//         with the reload, the bust stayed), which leaves the cache as it found it.
// The write time is put back whatever happens. Pointers are compared, never contents.
struct model_cache_check
{
    std::string problem;                  // why the builds couldn't run (no models, a build refused), or empty
    std::string touched;                  // the file whose write time moved, under assets/models
    std::size_t meshes = 0;               // still-life's meshes (scene_desc::meshes.meshes) in the first build
    std::size_t null_meshes = 0;          // ... null among them
    std::size_t second_meshes = 0;        // the second build's meshes
    std::size_t shared = 0;               // ... the first build's own (the same pointer)
    std::size_t textures = 0;             // the first build's textures
    std::size_t second_textures = 0;
    std::size_t shared_textures = 0;
    std::size_t second_read = 0;          // models the second build read from their files
    std::size_t third_read = 0;           // models the third build (the touched vase alone) read
    std::size_t third_meshes = 0;         // its meshes and textures
    std::size_t third_textures = 0;
    std::size_t third_old_textures = 0;   // ... the first build's among them (the vase not read again)
    std::size_t fourth_read = 0;          // models the fourth build read
    std::size_t fourth_meshes = 0;        // its meshes and textures
    std::size_t fourth_textures = 0;
    std::size_t kept_meshes = 0;          // ... the first build's among them: the bust's (the vase's are copies)
    std::size_t kept_textures = 0;        // ... the first build's: the bust's
    std::size_t reloaded_textures = 0;    // ... the third build's: the reloaded vase's
    std::string fifth_read;               // the files the fifth build read, as the scene names them, or empty
    std::size_t fifth_read_count = 0;
    bool restored = true;                 // the file's write time is its own (put back after the move, or never moved)
};
model_cache_check check_model_cache();

// The second model's material offset in still-life's mesh set: the vase's materials follow the bust's, so its
// triangles' material_of are shifted by the bust's count (gpu_scene.ixx's meshes_at). The two models are told apart by
// where they stand (the bust's instances at x = -0.55, the vase's at 0.6), and every triangle's material is compared
// with its model's own, as load_gltf reads them from the model's file here again: the factors, and each texture's
// texels, size and color space (base color sRGB, metallic-roughness linear). Any material but its model's is wrong,
// the other model's included.
struct model_materials_check
{
    std::string problem;                  // why the check couldn't run, or empty
    std::size_t materials = 0;            // the mesh set's
    std::size_t textures = 0;
    std::size_t bust_meshes = 0;          // still-life's meshes the bust's instances place
    std::size_t vase_meshes = 0;
    std::size_t both_meshes = 0;          // ... placed by both (none)
    std::size_t bust_triangles = 0;       // the triangles of those meshes, each once
    std::size_t vase_triangles = 0;
    std::size_t bust_wrong = 0;           // ... wearing a material that isn't one of their model's
    std::size_t vase_wrong = 0;
};
model_materials_check check_model_materials();
}
