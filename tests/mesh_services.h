// What the project's modules give the mesh suites (tests/mesh_gpu_tests.cu), which nvcc compiles and which can't
// import modules: module meshes' generators and bake as the renderer's mesh_desc, and a scene of the GPU programs' list
// as gpu_scene builds it. Plain functions, implemented with the modules in tests/mesh_services.cpp (cmake/meshes.cmake
// adds it to GpuTests). Host code only.
#pragma once

#include <array>
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
}
