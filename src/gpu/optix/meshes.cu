// The scene's triangle meshes on the GPU (mesh_scene.cuh): track B's file. The meshes are checked, their memory is
// planned before anything is allocated, then they are uploaded into the shared arrays the OptiX programs read
// (mesh_tables.h), each mesh gets a compacted triangle GAS (accel.cpp), each placement an instance of the IAS, and the
// integrator's tables are lengthened beyond the scene's own: the absorber (the through-rule's black lamp) after the
// materials and their side entries, and lamp_of up to its id. Step 92 adds glTF's materials: their textures and the
// per-pixel scratch slots.
//
// Compiled by nvcc (it fills device structs), host code only: every OptiX call is accel.cpp's (runtime_core.h).
#include "mesh_scene.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "device_memory.h"

namespace gpu::detail
{
namespace
{
// The layouts the shared arrays are copied into byte for byte from the host's floats and indices
static_assert(sizeof(float3) == 3 * sizeof(float) && sizeof(float2) == 2 * sizeof(float));
static_assert(sizeof(uint3) == 3 * sizeof(std::uint32_t));

constexpr std::size_t mib = std::size_t{1} << 20;
// The memory plan: everything a scene's meshes put on the GPU stays within 2048 MiB, and 256 MiB more must be free
// besides. On Windows a cudaMalloc beyond the GPU's memory may succeed by spilling into the computer's memory (the
// display driver pages it), which would make every launch crawl instead of failing: so the plan asks first.
constexpr std::size_t max_mesh_bytes = 2048 * mib;
constexpr std::size_t kept_free_bytes = 256 * mib;
// The IAS's mesh instances, as the loader's limit of placements: their ids k + 1 stay far below OptiX's limit of
// instance ids (OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID, 2^28 - 1 on current GPUs)
constexpr std::size_t max_instances = 1'000'000;

[[noreturn]] void refuse(const std::string& what)
{
    throw std::runtime_error("the scene's triangle meshes: " + what);
}

// Where each mesh lies in the shared arrays, and the counts the memory plan needs
struct layout
{
    std::vector<mesh_part> parts; // per mesh of mesh_set::meshes
    std::size_t vertices = 0;
    std::size_t triangles = 0;
    bool any_normals = false;
    bool any_uvs = false;
};

// A mesh's arrays must agree with each other: 3 floats per vertex, finite; normals and texture coordinates for every
// vertex or none; 3 indices per triangle, each of a vertex of the mesh; a material of the set for every triangle
void check_mesh(const mesh_desc* mesh, std::size_t i, std::size_t materials)
{
    if (mesh == nullptr)
    {
        refuse(std::format("meshes[{}] is missing", i));
    }
    const std::size_t floats = mesh->positions.size();
    if (floats == 0 || floats % 3 != 0 || floats / 3 > std::numeric_limits<std::uint32_t>::max())
    {
        refuse(std::format("meshes[{}] has {} position values, not 3 for each of 1 to 2^32 - 1 vertices", i, floats));
    }
    if (!std::ranges::all_of(mesh->positions, [](float x) { return std::isfinite(x); }))
    {
        refuse(std::format("meshes[{}] has a vertex that isn't finite", i));
    }
    const std::size_t vertices = floats / 3;
    if (!mesh->normals.empty() && mesh->normals.size() != floats)
    {
        refuse(std::format("meshes[{}] has {} normal values for {} vertices", i, mesh->normals.size(), vertices));
    }
    if (!mesh->uvs.empty() && mesh->uvs.size() != 2 * vertices)
    {
        refuse(std::format("meshes[{}] has {} texture coordinates for {} vertices", i, mesh->uvs.size(), vertices));
    }
    const std::size_t indices = mesh->indices.size();
    if (indices == 0 || indices % 3 != 0)
    {
        refuse(std::format("meshes[{}] has {} indices, not 3 for each of its triangles", i, indices));
    }
    if (const auto beyond = std::ranges::find_if(mesh->indices, [vertices](std::uint32_t v) { return v >= vertices; });
        beyond != mesh->indices.end())
    {
        refuse(std::format("meshes[{}] has the index {} of a vertex it doesn't have ({} vertices)", i, *beyond,
                           vertices));
    }
    if (mesh->material_of.size() != indices / 3)
    {
        refuse(std::format("meshes[{}] has {} triangles but {} materials for them", i, indices / 3,
                           mesh->material_of.size()));
    }
    if (std::ranges::any_of(mesh->material_of, [materials](std::uint32_t m) { return m >= materials; }))
    {
        refuse(std::format("meshes[{}] gives a triangle a material the set doesn't have ({} materials)", i,
                           materials));
    }
}

// The determinant of a row-major 3×4 matrix's linear part
double determinant(const std::array<double, 12>& m)
{
    return m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
           m[2] * (m[4] * m[9] - m[5] * m[8]);
}

// M⁻ᵀ = cof(M) / det M, row-major 3×3, in double: what carries a normal into the world, outside staying outside under
// a mirror ((Ma)×(Mb) = det M · M⁻ᵀ(a×b), see triangle_surface)
std::array<double, 9> inverse_transpose(const std::array<double, 12>& m, double det)
{
    return {(m[5] * m[10] - m[6] * m[9]) / det, (m[6] * m[8] - m[4] * m[10]) / det, (m[4] * m[9] - m[5] * m[8]) / det,
            (m[2] * m[9] - m[1] * m[10]) / det, (m[0] * m[10] - m[2] * m[8]) / det, (m[1] * m[8] - m[0] * m[9]) / det,
            (m[1] * m[6] - m[2] * m[5]) / det, (m[2] * m[4] - m[0] * m[6]) / det, (m[0] * m[5] - m[1] * m[4]) / det};
}

// What a triangle may wear: a lambertian (any books' texture) or a metal, also as a GGX side entry completes them (a
// rough metal, a glossy coat). Glass would be absorbed by the through-rule at its first face, and a lamp would need
// lamp_of entries and the light samples' skip rule for triangles: both refused, with the reason.
void check_scene_material(const scene_desc& scene, std::size_t i, int index)
{
    if (index < 0 || static_cast<std::size_t>(index) >= scene.materials.size())
    {
        refuse(std::format("materials[{}] names the scene's material {}, which doesn't exist ({} materials)", i, index,
                           scene.materials.size()));
    }
    switch (scene.materials[static_cast<std::size_t>(index)].kind)
    {
    case material_kind::lambertian:
    case material_kind::metal:
        return;
    case material_kind::dielectric:
        refuse(std::format("materials[{}] is glass, which triangles can't wear: a ray going through a face is absorbed "
                           "(the faces are opaque)",
                           i));
    case material_kind::diffuse_light:
        refuse(std::format("materials[{}] is a lamp, which triangles can't be: the light samples know the books' "
                           "lamps only",
                           i));
    case material_kind::isotropic:
        refuse(std::format("materials[{}] is the particles of smoke, which a surface can't wear", i));
    }
    refuse(std::format("materials[{}] is of an unknown kind", i));
}
}

// Everything the meshes own on the GPU, and what the tables are built from
struct mesh_scene::state
{
    // The shared arrays of mesh_tables, every mesh's slice one after another (mesh_part)
    device_array<float3> positions{0};
    device_array<float3> normals{0}; // only when a mesh has normals; a mesh without them leaves its slice unread
    device_array<float2> uvs{0};     // likewise
    device_array<uint3> triangles{0};
    device_array<std::uint32_t> material_of{0};
    device_array<mesh_part> parts{0};
    device_array<mesh_instance> instances{0};
    device_array<mesh_material> materials{0};
    // A compacted triangle GAS per mesh, and the IAS's mesh instances (instance k: id k + 1, sbtOffset 3)
    std::vector<optix_core::accel> structures;
    std::vector<optix_core::instance> placed;
    // The integrator's tables, lengthened: the scene's M materials and their side entries, then the absorber at index
    // M; lamp_of up to and including the absorber's id (only in a scene with lamps: lamp_hit_pdf reads it then only)
    device_array<material<float>> lengthened_materials{0};
    device_array<microfacet> lengthened_microfacets{0}; // only in a scene with side entries
    device_array<int> lamp_of{0};
    int instance_count = 0;
    int triangle_base = std::numeric_limits<int>::max();
    int absorber = 0;
};

mesh_scene::mesh_scene(optix_core::core& core, const scene_desc& scene, const quality_view& view)
    : state_(std::make_unique<state>())
{
    const mesh_set& set = scene.meshes;
    if (set.instances.empty())
    {
        return; // the books' GAS stays the root, and the tables keep their lengths
    }
    state& s = *state_;

    // --- What the set holds, checked, and where it goes ---------------------------------------------------------
    layout lay;
    lay.parts.reserve(set.meshes.size());
    for (std::size_t i = 0; i < set.meshes.size(); ++i)
    {
        const mesh_desc* mesh = set.meshes[i].get();
        check_mesh(mesh, i, set.materials.size());
        const std::size_t vertices = mesh->positions.size() / 3;
        const std::size_t triangles = mesh->indices.size() / 3;
        if (lay.vertices + vertices > std::numeric_limits<std::uint32_t>::max() ||
            lay.triangles + triangles > std::numeric_limits<std::uint32_t>::max())
        {
            refuse("more than 2^32 - 1 vertices or triangles in all");
        }
        lay.parts.push_back({.first_vertex = static_cast<unsigned int>(lay.vertices),
                             .first_triangle = static_cast<unsigned int>(lay.triangles),
                             .has_normals = mesh->normals.empty() ? 0u : 1u,
                             .has_uvs = mesh->uvs.empty() ? 0u : 1u});
        lay.vertices += vertices;
        lay.triangles += triangles;
        lay.any_normals = lay.any_normals || !mesh->normals.empty();
        lay.any_uvs = lay.any_uvs || !mesh->uvs.empty();
    }
    for (std::size_t i = 0; i < set.materials.size(); ++i)
    {
        const mesh_material_desc& m = set.materials[i];
        if (m.scene_material == -1)
        {
            throw std::logic_error("glTF's materials on triangles come in Step 92");
        }
        check_scene_material(scene, i, m.scene_material);
    }
    if (set.instances.size() > max_instances)
    {
        refuse(std::format("{} placements, more than {}", set.instances.size(), max_instances));
    }
    // The ids: the books' primitives, the particles, then the absorber at triangle_base, then every placement's
    // triangles in turn; a hit's id must stay an int
    s.triangle_base = view.base.media.base + view.base.media.count;
    std::vector<mesh_instance> instances;
    instances.reserve(set.instances.size());
    std::int64_t next_id = std::int64_t{s.triangle_base} + 1;
    for (std::size_t k = 0; k < set.instances.size(); ++k)
    {
        const mesh_instance_desc& placement = set.instances[k];
        if (placement.mesh < 0 || static_cast<std::size_t>(placement.mesh) >= set.meshes.size())
        {
            refuse(std::format("instances[{}] places the mesh {}, which doesn't exist ({} meshes)", k, placement.mesh,
                               set.meshes.size()));
        }
        const std::array<double, 12>& m = placement.object_to_world;
        const double det = determinant(m);
        const std::array<double, 9> n = inverse_transpose(m, det);
        if (!std::ranges::all_of(m, [](double x) { return std::isfinite(x); }) || !std::isfinite(det) || det == 0 ||
            !std::ranges::all_of(n, [](double x) { return std::isfinite(x); }))
        {
            refuse(std::format("instances[{}] has a transform that isn't finite or can't be inverted", k));
        }
        mesh_instance placed{.part = placement.mesh, .first_id = static_cast<int>(std::min<std::int64_t>(
                                                         next_id, std::numeric_limits<int>::max()))};
        std::ranges::transform(m, placed.object_to_world, [](double x) { return static_cast<float>(x); });
        std::ranges::transform(n, placed.normal_to_world, [](double x) { return static_cast<float>(x); });
        instances.push_back(placed);
        next_id += static_cast<std::int64_t>(set.meshes[static_cast<std::size_t>(placement.mesh)]->indices.size() / 3);
        if (next_id > std::int64_t{std::numeric_limits<int>::max()} + 1)
        {
            refuse("the placed triangles' ids pass 2^31 - 1: too many triangles placed");
        }
    }

    // --- The memory plan, before anything is allocated -------------------------------------------------------------
    const std::size_t scene_materials = scene.materials.size();
    const bool side_entries = view.microfacets != nullptr;
    const bool lamps = view.lamps.count > 0;
    std::vector<optix_core::triangle_mesh> inputs; // device pointers set once the arrays exist
    inputs.reserve(set.meshes.size());
    for (std::size_t i = 0; i < set.meshes.size(); ++i)
    {
        inputs.push_back({.positions = nullptr,
                          .vertex_count = static_cast<unsigned int>(set.meshes[i]->positions.size() / 3),
                          .triangles = nullptr,
                          .triangle_count = static_cast<unsigned int>(set.meshes[i]->indices.size() / 3)});
    }
    const std::size_t arrays = lay.vertices * (sizeof(float3) + (lay.any_normals ? sizeof(float3) : 0) +
                                               (lay.any_uvs ? sizeof(float2) : 0)) +
                               lay.triangles * (sizeof(uint3) + sizeof(std::uint32_t)) +
                               lay.parts.size() * sizeof(mesh_part) + instances.size() * sizeof(mesh_instance) +
                               set.materials.size() * sizeof(mesh_material);
    const std::size_t tables =
        (scene_materials + 1) * (sizeof(material<float>) + (side_entries ? sizeof(microfacet) : 0)) +
        (lamps ? (static_cast<std::size_t>(s.triangle_base) + 1) * sizeof(int) : 0);
    const std::size_t planned = arrays + tables + optix_core::triangles_bytes(core, inputs);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "asking how much of the GPU's memory is free");
    if (planned > max_mesh_bytes)
    {
        refuse(std::format("they need {} MiB of GPU memory, more than the {} MiB a scene's meshes may take",
                           (planned + mib - 1) / mib, max_mesh_bytes / mib));
    }
    if (free_bytes < planned + kept_free_bytes)
    {
        refuse(std::format("they need {} MiB of GPU memory, more than the {} MiB free on the GPU less {} MiB kept for "
                           "the rest",
                           (planned + mib - 1) / mib, free_bytes / mib, kept_free_bytes / mib));
    }

    // --- The shared arrays, every mesh's slice copied in place -------------------------------------------------------
    s.positions = device_array<float3>(lay.vertices);
    s.normals = device_array<float3>(lay.any_normals ? lay.vertices : 0);
    s.uvs = device_array<float2>(lay.any_uvs ? lay.vertices : 0);
    s.triangles = device_array<uint3>(lay.triangles);
    s.material_of = device_array<std::uint32_t>(lay.triangles);
    const auto copy_in = [](void* to, const void* from, std::size_t bytes) {
        if (bytes > 0)
        {
            check(cudaMemcpy(to, from, bytes, cudaMemcpyHostToDevice), "copying a triangle mesh to the GPU");
        }
    };
    for (std::size_t i = 0; i < set.meshes.size(); ++i)
    {
        const mesh_desc& mesh = *set.meshes[i];
        const mesh_part& part = lay.parts[i];
        copy_in(s.positions.data() + part.first_vertex, mesh.positions.data(), mesh.positions.size() * sizeof(float));
        if (part.has_normals != 0u)
        {
            copy_in(s.normals.data() + part.first_vertex, mesh.normals.data(), mesh.normals.size() * sizeof(float));
        }
        if (part.has_uvs != 0u)
        {
            copy_in(s.uvs.data() + part.first_vertex, mesh.uvs.data(), mesh.uvs.size() * sizeof(float));
        }
        copy_in(s.triangles.data() + part.first_triangle, mesh.indices.data(),
                mesh.indices.size() * sizeof(std::uint32_t));
        copy_in(s.material_of.data() + part.first_triangle, mesh.material_of.data(),
                mesh.material_of.size() * sizeof(std::uint32_t));
        inputs[i].positions = s.positions.data() + part.first_vertex;
        inputs[i].triangles = s.triangles.data() + part.first_triangle;
    }
    s.parts = device_array<mesh_part>(lay.parts);
    s.instances = device_array<mesh_instance>(instances);
    std::vector<mesh_material> materials;
    materials.reserve(set.materials.size());
    for (const mesh_material_desc& m : set.materials)
    {
        materials.push_back({.scene_material = m.scene_material});
    }
    s.materials = device_array<mesh_material>(materials);
    s.instance_count = static_cast<int>(instances.size());

    // --- A GAS per mesh, and the instances the IAS will hold ---------------------------------------------------------
    s.structures = optix_core::build_triangles(core, inputs);
    s.placed.reserve(instances.size());
    for (std::size_t k = 0; k < instances.size(); ++k)
    {
        optix_core::instance placed{.id = static_cast<unsigned int>(k + 1),
                                    .sbt_offset = 3,
                                    .handle = s.structures[static_cast<std::size_t>(instances[k].part)].handle};
        std::ranges::copy(instances[k].object_to_world, placed.transform);
        s.placed.push_back(placed);
    }

    // --- The integrator's tables, lengthened ---------------------------------------------------------------------
    // The materials (and side entries) by device copies of the scene's own, then the absorber: a diffuse_light of
    // color 0 (its side entry none). These are data, not code: the integrator reads them as it reads the scene's.
    s.absorber = static_cast<int>(scene_materials);
    s.lengthened_materials = device_array<material<float>>(scene_materials + 1);
    if (scene_materials > 0)
    {
        check(cudaMemcpy(s.lengthened_materials.data(), view.base.materials, scene_materials * sizeof(material<float>),
                         cudaMemcpyDeviceToDevice),
              "lengthening the materials for the triangle meshes");
    }
    const material<float> absorber{.kind = material_kind::diffuse_light, .albedo = {0, 0, 0}, .fuzz = 0,
                                   .refraction_index = 1};
    copy_in(s.lengthened_materials.data() + scene_materials, &absorber, sizeof absorber);
    if (side_entries)
    {
        s.lengthened_microfacets = device_array<microfacet>(scene_materials + 1);
        check(cudaMemcpy(s.lengthened_microfacets.data(), view.microfacets, scene_materials * sizeof(microfacet),
                         cudaMemcpyDeviceToDevice),
              "lengthening the GGX side entries for the triangle meshes");
        const microfacet none{.kind = microfacet_kind::none, .alpha = 0, .roughness = 0};
        copy_in(s.lengthened_microfacets.data() + scene_materials, &none, sizeof none);
    }
    // lamp_of: the scene's (one entry per surface primitive, lamp_table.cu), then -1 for the particles and for the
    // absorber, whose id lamp_hit_pdf meets when a path ends there
    if (lamps)
    {
        const std::size_t surfaces = scene.spheres.size() + scene.quads.size();
        const auto length = static_cast<std::size_t>(s.triangle_base) + 1;
        s.lamp_of = device_array<int>(length);
        check(cudaMemcpy(s.lamp_of.data(), view.lamps.lamp_of, surfaces * sizeof(int), cudaMemcpyDeviceToDevice),
              "lengthening lamp_of for the triangle meshes");
        const std::vector<int> none(length - surfaces, -1);
        copy_in(s.lamp_of.data() + surfaces, none.data(), none.size() * sizeof(int));
    }
}

mesh_scene::~mesh_scene() = default;

bool mesh_scene::empty() const
{
    return state_->instance_count == 0;
}

// The view with the three lengthened tables in place of the scene's: only those pointers are replaced, so the result
// of this function given to it again comes back unchanged (scene.cu may lengthen a view it lengthened before)
quality_view mesh_scene::lengthened(const quality_view& view) const
{
    if (empty())
    {
        return view;
    }
    quality_view v = view;
    v.base.materials = state_->lengthened_materials.data();
    if (state_->lengthened_microfacets.size() > 0)
    {
        v.microfacets = state_->lengthened_microfacets.data();
    }
    if (state_->lamp_of.size() > 0)
    {
        v.lamps.lamp_of = state_->lamp_of.data();
    }
    return v;
}

mesh_tables mesh_scene::tables() const
{
    mesh_tables t{}; // no arrays; triangle_base INT_MAX: every id belongs to the books' primitives and particles
    t.scratch_base = -1;
    if (empty())
    {
        return t;
    }
    const state& s = *state_;
    t.positions = s.positions.data();
    t.normals = s.normals.size() > 0 ? s.normals.data() : nullptr;
    t.uvs = s.uvs.size() > 0 ? s.uvs.data() : nullptr;
    t.triangles = s.triangles.data();
    t.material_of = s.material_of.data();
    t.parts = s.parts.data();
    t.instances = s.instances.data();
    t.materials = s.materials.data();
    t.instance_count = s.instance_count;
    t.triangle_base = s.triangle_base;
    t.absorber_material = s.absorber;
    t.scratch_materials = s.lengthened_materials.data();
    t.scratch_microfacets = s.lengthened_microfacets.size() > 0 ? s.lengthened_microfacets.data() : nullptr;
    return t;
}

std::span<const optix_core::instance> mesh_scene::instances() const
{
    return state_->placed;
}

// No glTF material before Step 92: no scratch slot to grow
void mesh_scene::prepare_view(int /*width*/, int /*height*/)
{
}
}
