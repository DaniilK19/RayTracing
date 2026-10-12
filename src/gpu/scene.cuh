// The scene as the kernels see it (the CPU's scene is in scenes.ixx): where its arrays are in the GPU's memory.
#pragma once

#include "bvh.cuh"
#include "material.cuh"
#include "sphere.cuh"
#include "texture.cuh"

namespace gpu
{
// The kernel for all three books takes it as an argument, so it is copied into constant memory for the launch (see
// all_books_kernel in renderer.cu), where one read serves a whole warp when all its threads read the same member. The
// arrays the search for the closest hit reads (in tree) are apart from those only a hit reads.
template <real_number Real>
struct scene_view
{
    bvh_view<Real> tree;
    const int* primitive_materials;            // the material index of each sphere, in the BVH's order
    const sphere_frame<Real>* sphere_frames;   // each sphere's frame (see onto_sphere), in the BVH's order
    const material<Real>* materials;
    const material_texture* material_textures; // read by the kernel for all three books only, by material index
    texture_view<Real> textures;
};
}
