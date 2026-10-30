// The meshes as the OptiX programs read them, uploaded by meshes.cu beside the books' tables (which never change)
#pragma once
#include <climits>
#include <cstdint>
#include <cuda_runtime.h> // float2, float3, uint3, cudaTextureObject_t
#include "material.cuh"
#include "quality_view.cuh"

namespace gpu
{
struct mesh_part // one mesh's slice of the shared arrays
{
    unsigned int first_vertex;   // into positions, normals, uvs (a triangle's indices are relative to it)
    unsigned int first_triangle; // into triangles and material_of
    unsigned int has_normals;    // 0: flat shading (glTF: "MUST calculate flat normals")
    unsigned int has_uvs;        // 0: (0, 0)
};

struct mesh_instance // one placement of a mesh: instance k of the IAS has instanceId k + 1
{
    float object_to_world[12]; // row-major 3×4, OptixInstance::transform's layout
    float normal_to_world[9];  // the inverse transpose, row-major 3×3: normals stay outside under a mirror
    int part;                  // into parts
    int first_id;              // the global id of its triangle 0
};

struct mesh_material // what a triangle wears
{
    int scene_material;                     // >= 0: an index into the scene's materials and side entries; -1: glTF
    float base_color[3];                    // linear
    float metallic;
    float roughness;
    cudaTextureObject_t base_color_texture; // valid where has_base_color_texture: sRGB-decoded, glTF's wrap
    cudaTextureObject_t metal_rough_texture;
    unsigned int has_base_color_texture;
    unsigned int has_metal_rough_texture;
};

struct mesh_tables
{
    const float3* positions;
    const float3* normals;
    const float2* uvs;
    const uint3* triangles;
    const std::uint32_t* material_of; // per triangle: into materials
    const mesh_part* parts;
    const mesh_instance* instances;
    const mesh_material* materials;
    int instance_count;
    int triangle_base = INT_MAX; // media.base + media.count: the absorber's id; triangles above it. INT_MAX: no mesh
    int absorber_material;       // the absorber's index in the lengthened materials (diffuse_light, black)
    int scratch_base;            // the first per-pixel scratch slot; -1: no glTF material in the scene
    material<float>* scratch_materials; // the view's materials and side entries, writable (the same arrays)
    microfacet* scratch_microfacets;
};
}
