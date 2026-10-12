// The scene as the kernels see it (the CPU's scene is in scenes.ixx): where its arrays are in the GPU's memory.
#pragma once

#include <type_traits>

#include "bvh.cuh"
#include "constant_medium.cuh"
#include "lights.cuh"
#include "material.cuh"
#include "sphere.cuh"
#include "texture.cuh"

namespace gpu
{
// The scene as the kernel for all three books sees it. The kernel takes it as an argument, so it is copied into
// constant memory for the launch (see all_books_kernel in renderer.cu), where one read serves a whole warp when all
// its threads read the same member. The arrays the search for the closest hit reads (in tree) are apart from those
// only a hit reads.
template <real_number Real>
struct scene_view
{
    bool motion; // some sphere moves, a shape or a volume's boundary: camera rays get a random time (see start_sample)
    sampling_strategy sampling; // how a diffuse bounce chooses its direction (see diffuse_bounce in camera.cuh)
    bvh_view<Real> tree;
    media_view<Real> media; // smoke and fog, tested after the surfaces (see closest_event in camera.cuh)
    const int* primitive_materials;            // the material index of each primitive, by its global id (see bvh)
    const sphere_frame<Real>* sphere_frames;   // each sphere's frame (see sphere_frame), by its global id
    const material<Real>* materials;
    const material_texture* material_textures; // read by the kernel for all three books only, by material index
    texture_view<Real> textures;
    lights_view<Real> lights; // what the lights strategy aims at
};

// The scene as the first book's kernel sees it: Step 46's arrays and nothing else, in a struct that never grows.
//
// The first book's kernel is meant to compute exactly what Step 46's computed, bit for bit, and that holds only as long
// as nvcc compiles its code as it did then. Not every choice nvcc makes is fixed by the source: x·x + y·y + z·z, for
// one, takes two fused multiply-adds (fmas), and whether the first one fuses x·x or y·y with the sum depends on the
// code around it, which changes the result's last bit. When the first book's kernel built a scene_view (which grows
// with every feature) instead of this struct, a new member at its end changed that choice in |d|² of the ray's
// direction (in three places, as comparing the PTX showed), and the first book's float renders changed in their last
// bits, twice. With a struct of its own, the first book's kernel no longer depends on what the other kernel needs.
template <real_number Real>
struct first_book_scene
{
    first_book_tree<Real> tree;
    const int* primitive_materials; // the material index of each sphere, by its global id (see bvh)
    const material<Real>* materials;
};

// The scene of each kind of kernel
template <real_number Real, kernel_kind Kind>
using kernel_scene = std::conditional_t<Kind == kernel_kind::first_book, first_book_scene<Real>, scene_view<Real>>;
}
