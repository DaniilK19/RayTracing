// A scene_desc's surfaces and media on the GPU, in float, as the kernel for all three books sees them: for the tests
// that trace rays against a scene (shadow rays, lamps, media), without a session. The BVH is the renderer's own
// (build_bvh in bvh.cuh), so global ids, nodes and primitives are the ones the kernels walk; the rest is the minimum
// of renderer.cu's translation those tests need. Device code: included by .cu test files only.
//
// A quality_view for a test kernel is put together by hand from view() and the test's own arrays (see shared.md 3.11).
#pragma once

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "bvh.cuh"
#include "constant_medium.cuh"
#include "gpu_test.cuh"
#include "material.cuh"
#include "renderer.h"
#include "scene.cuh"

namespace gpu_test
{
class device_scene
{
public:
    // Textures must be solid colors (folded into the materials' albedos); no images, no lights list
    explicit device_scene(const gpu::scene_desc& scene, int max_leaf_size = 1)
        : tree_(gpu::build_bvh<float>(scene.spheres, scene.quads, max_leaf_size)),
          nodes_(tree_.nodes),
          spheres_(tree_.spheres),
          moving_spheres_(tree_.moving_spheres),
          quads_(tree_.quads),
          primitive_materials_(tree_.primitive_materials),
          sphere_frames_(tree_.sphere_frames),
          materials_(materials_of(scene)),
          material_textures_(std::vector<gpu::material_texture>(scene.materials.size(), {-1, false, false})),
          media_(media_of(scene, boundary_quad_list_)),
          boundary_quads_(boundary_quad_list_)
    {
        if (!scene.images.empty() || !scene.lights.empty())
        {
            throw std::invalid_argument("a test scene has no images and no lights list");
        }
        const auto moves = [](const gpu::sphere_desc& s) {
            return s.displacement.x != 0 || s.displacement.y != 0 || s.displacement.z != 0;
        };
        bool motion = false;
        for (const gpu::sphere_desc& s : scene.spheres)
        {
            motion = motion || moves(s);
        }
        for (const gpu::medium_desc& m : scene.media)
        {
            motion = motion || (m.boundary == gpu::boundary_shape::sphere && moves(m.sphere));
        }
        view_ = {.motion = motion,
                 .sampling = gpu::sampling_strategy::material,
                 .tree = {.nodes = nodes_.get(),
                          .spheres = spheres_.get(),
                          .node_count = static_cast<int>(tree_.nodes.size()),
                          .moving_spheres = moving_spheres_.get(),
                          .moving_base = tree_.moving_base,
                          .quads = quads_.get(),
                          .quad_base = tree_.quad_base},
                 .media = {.media = media_.get(),
                           .boundary_quads = boundary_quads_.get(),
                           .count = static_cast<int>(scene.media.size()),
                           .base = surface_count()},
                 .primitive_materials = primitive_materials_.get(),
                 .sphere_frames = sphere_frames_.get(),
                 .materials = materials_.get(),
                 .material_textures = material_textures_.get(),
                 .textures = {},
                 .lights = {.lights = nullptr, .count = 0}};
    }

    const gpu::scene_view<float>& view() const { return view_; }
    std::span<const int> sphere_ids() const { return tree_.sphere_ids; } // global id of scene.spheres[k]
    std::span<const int> quad_ids() const { return tree_.quad_ids; }
    // The surface primitives' count: the size of a lamp_of array, and the global id of the first medium's particles
    int surface_count() const { return static_cast<int>(tree_.primitive_materials.size()); }
    const gpu::bvh<float>& tree() const { return tree_; } // the host BVH, for brute-force references

private:
    static std::vector<gpu::material<float>> materials_of(const gpu::scene_desc& scene)
    {
        std::vector<gpu::material<float>> result;
        for (const gpu::material_desc& m : scene.materials)
        {
            gpu::material<float> g{};
            g.kind = m.kind;
            g.fuzz = static_cast<float>(m.fuzz);
            g.refraction_index = static_cast<float>(m.refraction_index);
            if (m.kind == gpu::material_kind::metal)
            {
                g.albedo = gpu::to_real<float>(m.albedo);
            }
            else if (m.kind != gpu::material_kind::dielectric)
            {
                if (m.texture < 0 || m.texture >= std::ssize(scene.textures) ||
                    scene.textures[static_cast<std::size_t>(m.texture)].kind != gpu::texture_kind::solid)
                {
                    throw std::invalid_argument("a test scene's textures must be solid colors");
                }
                g.albedo = gpu::to_real<float>(scene.textures[static_cast<std::size_t>(m.texture)].rgb);
            }
            result.push_back(g);
        }
        return result;
    }

    // As renderer.cu's to_media: the boundary's sphere or quads, and a box around the volume over the shutter time,
    // rounded outward
    static std::vector<gpu::medium<float>> media_of(const gpu::scene_desc& scene, std::vector<gpu::quad<float>>& quads)
    {
        std::vector<gpu::medium<float>> result;
        for (const gpu::medium_desc& desc : scene.media)
        {
            gpu::medium<float> m{};
            m.neg_inv_density = static_cast<float>(desc.neg_inv_density);
            m.material = desc.material;
            gpu::build_box box;
            if (desc.boundary == gpu::boundary_shape::sphere)
            {
                m.sphere = {.center = gpu::to_real<float>(desc.sphere.center),
                            .radius = static_cast<float>(desc.sphere.radius),
                            .displacement = gpu::to_real<float>(desc.sphere.displacement),
                            .unused = 0};
                box = gpu::sweep_box(m.sphere);
            }
            else
            {
                m.first_quad = static_cast<int>(quads.size());
                m.quad_count = static_cast<int>(desc.quads.size());
                for (const gpu::quad_desc& q : desc.quads)
                {
                    quads.push_back(gpu::make_quad<float>(q));
                    box.enclose(gpu::quad_box(q));
                }
            }
            m.box_min = {gpu::round_down<float>(box.min[0]), gpu::round_down<float>(box.min[1]),
                         gpu::round_down<float>(box.min[2])};
            m.box_max = {gpu::round_up<float>(box.max[0]), gpu::round_up<float>(box.max[1]),
                         gpu::round_up<float>(box.max[2])};
            result.push_back(m);
        }
        return result;
    }

    gpu::bvh<float> tree_;
    device_buffer<gpu::bvh_node<float>> nodes_;
    device_buffer<gpu::sphere<float>> spheres_;
    device_buffer<gpu::moving_sphere<float>> moving_spheres_;
    device_buffer<gpu::quad<float>> quads_;
    device_buffer<int> primitive_materials_;
    device_buffer<gpu::sphere_frame<float>> sphere_frames_;
    device_buffer<gpu::material<float>> materials_;
    device_buffer<gpu::material_texture> material_textures_;
    std::vector<gpu::quad<float>> boundary_quad_list_; // filled by media_of, before boundary_quads_ is made
    device_buffer<gpu::medium<float>> media_;
    device_buffer<gpu::quad<float>> boundary_quads_;
    gpu::scene_view<float> view_{};
};
}
