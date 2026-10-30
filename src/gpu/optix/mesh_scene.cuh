// The scene's meshes on the GPU (meshes.cu): the arrays, texture objects and GASes, and the integrator's tables
// lengthened for them (the absorber, the scratch slots, lamp_of). The OptiX scene only composes it.
//
// What it holds is track B's (meshes.cu's `state`, Steps 91 and 92), so this header doesn't change when it grows.
#pragma once
#include <cstddef>
#include <memory>
#include <span>
#include <vector>
#include "mesh_tables.h"
#include "quality_view.cuh"
#include "renderer.h"
#include "runtime_core.h"

namespace gpu::detail
{
class mesh_scene
{
public:
    // Validates the meshes (indices, materials allowed on triangles, finite transforms, ids below 2^31), plans the
    // memory (at most 2048 MiB, with 256 MiB of the GPU's left free) before allocating, uploads and builds. Throws
    // std::runtime_error.
    mesh_scene(optix_core::core& core, const scene_desc& scene, const quality_view& view);
    ~mesh_scene();
    mesh_scene(const mesh_scene&) = delete;
    mesh_scene& operator=(const mesh_scene&) = delete;
    bool empty() const; // no instance: the books' GAS stays the root
    // The view with the lengthened tables (the materials, the side entries, lamp_of) in place of the scene's; the view
    // itself when empty. Only those pointers are replaced, and the tables were filled when the meshes were uploaded,
    // from the scene's own M materials and side entries, not from the view given here: so a view lengthened before
    // comes back with the current pointers, which is how the OptiX scene follows prepare_view.
    quality_view lengthened(const quality_view& view) const;
    mesh_tables tables() const;                              // triangle_base INT_MAX when empty
    std::span<const optix_core::instance> instances() const; // ids k + 1, sbt_offset 3
    // The view's size: grows the scratch slots of glTF materials, never shrinks them. Growing makes the lengthened
    // tables again, longer, and frees the old ones, so their addresses change: after every call the OptiX scene takes
    // lengthened() and tables() again (scene.cu).
    void prepare_view(int width, int height);

private:
    struct state; // meshes.cu: the device arrays, texture objects, GASes and lengthened tables
    std::unique_ptr<state> state_;
};
}
