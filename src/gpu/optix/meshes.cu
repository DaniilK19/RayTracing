// The scene's triangle meshes on the GPU (mesh_scene.cuh): track B's file. Step 91 uploads the meshes (the mesh tables,
// the absorber and the lengthened tables, the materials allowed on triangles, the memory plan) and builds their GASes;
// Step 92 adds the glTF materials' textures and scratch slots. Until then the holder is an empty scene, and a scene
// with meshes is refused.
#include "mesh_scene.cuh"

#include <memory>
#include <stdexcept>

namespace gpu::detail
{
// Everything the meshes own on the GPU (Steps 91 and 92); nothing yet
struct mesh_scene::state
{
};

mesh_scene::mesh_scene(optix_core::core& /*core*/, const scene_desc& scene, const quality_view& /*view*/)
    : state_(std::make_unique<state>())
{
    if (!scene.meshes.instances.empty())
    {
        throw std::logic_error("triangle meshes in OptiX come in Step 91");
    }
}

mesh_scene::~mesh_scene() = default;

bool mesh_scene::empty() const
{
    return true;
}

quality_view mesh_scene::lengthened(const quality_view& view) const
{
    return view;
}

mesh_tables mesh_scene::tables() const
{
    mesh_tables none{}; // no arrays; triangle_base INT_MAX: every id belongs to the books' primitives and particles
    none.scratch_base = -1;
    return none;
}

std::span<const optix_core::instance> mesh_scene::instances() const
{
    return {};
}

void mesh_scene::prepare_view(int /*width*/, int /*height*/)
{
}
}
