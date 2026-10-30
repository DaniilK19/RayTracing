// tests/mesh_services.h's functions, with the modules: what the mesh suites (tests/mesh_gpu_tests.cu, nvcc) can't
// import themselves. Compiled by cl.exe into GpuTests (cmake/meshes.cmake). Host code only.

// The headers before `import std;` (see renderer.h)
#include "mesh_services.h"
#include "renderer.h"

import std;
import gpu_scene;
import meshes;

namespace gpu_test::mesh_host
{
namespace
{
// The two structs have the same five arrays (module meshes has no CUDA and no renderer, the renderer no modules)
gpu::mesh_desc to_desc(meshes::mesh m)
{
    return {.positions = std::move(m.positions),
            .normals = std::move(m.normals),
            .uvs = std::move(m.uvs),
            .indices = std::move(m.indices),
            .material_of = std::move(m.material_of)};
}

meshes::mesh to_mesh(const gpu::mesh_desc& d)
{
    return {.positions = d.positions,
            .normals = d.normals,
            .uvs = d.uvs,
            .indices = d.indices,
            .material_of = d.material_of};
}
}

gpu::mesh_desc icosphere(int level, bool smooth)
{
    return to_desc(meshes::icosphere(level, smooth));
}

gpu::mesh_desc torus(double major, double minor, int around, int across)
{
    return to_desc(meshes::torus(major, minor, around, across));
}

gpu::mesh_desc bake(const gpu::mesh_desc& mesh, const std::array<double, 12>& object_to_world)
{
    return to_desc(meshes::bake(to_mesh(mesh), object_to_world));
}

gpu::scene_desc built_scene(std::string_view name)
{
    std::expected<gpu_build, std::string> built = build_gpu_scene(name, {});
    if (!built)
    {
        throw std::runtime_error(std::format("{}: {}", name, built.error()));
    }
    return std::move(built->desc);
}
}
