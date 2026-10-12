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
//
// The order of the members matters, though not to this code: the first book's kernel builds this struct too (see
// first_book_kernel), and with `motion` placed last, nvcc fused other products into fmas in |d|² of the ray's direction
// (in three places, found by comparing the PTX the two orders give), so the first book's float renders were no longer
// byte-identical with Step 46's. A float computation such as x·x + y·y + z·z can be rounded differently depending on
// which products the compiler fuses with the sum, and that choice depends on the code around it. After changing this
// struct's members or their order, compare those renders again.
template <real_number Real>
struct scene_view
{
    bool motion; // some sphere moves: camera rays get a random time (see start_sample in camera.cuh)
    bvh_view<Real> tree;
    const int* primitive_materials;            // the material index of each primitive, by its global id (see bvh)
    const sphere_frame<Real>* sphere_frames;   // each sphere's frame (see onto_sphere), by its global id
    const material<Real>* materials;
    const material_texture* material_textures; // read by the kernel for all three books only, by material index
    texture_view<Real> textures;
};
}
