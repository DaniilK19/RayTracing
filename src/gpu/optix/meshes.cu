// The scene's triangle meshes on the GPU (mesh_scene.cuh): track B's file. The meshes are checked, their memory is
// planned before anything is allocated, then they are uploaded, each centred on its bounds, into the shared arrays the
// OptiX programs read (mesh_tables.h), each mesh gets a compacted triangle GAS (accel.cpp), each placement an instance
// of the IAS (its translation taking the centre back), and the integrator's tables are lengthened beyond the scene's
// own: after the materials and their side entries the absorber (the through-rule's black lamp) and, in a scene with
// glTF's materials, one scratch slot per pixel; lamp_of up to the absorber's id. glTF's images become CUDA texture
// objects, read by the texture hardware in the OptiX programs only.
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
// The IAS's mesh instances, as the loader's limit of placements: their ids k + 1 stay far below the limit OptiX
// reports for instance ids (OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID)
constexpr std::size_t max_instances = 1'000'000;

[[noreturn]] void refuse(const std::string& what)
{
    throw std::runtime_error("the scene's triangle meshes: " + what);
}

using centre = std::array<double, 3>;

// Where each mesh lies in the shared arrays, and the counts the memory plan needs
struct layout
{
    std::vector<mesh_part> parts; // per mesh of mesh_set::meshes
    std::vector<centre> centres;  // likewise: what the upload subtracts from its positions (mesh_frame)
    std::size_t vertices = 0;
    std::size_t triangles = 0;
    bool any_normals = false;
    bool any_uvs = false;
};

// A mesh's arrays must agree with each other: 3 floats per vertex; normals and texture coordinates for every vertex or
// none; 3 indices per triangle, each of a vertex of the mesh, which must be finite; a material of the set for every
// triangle. A vertex no triangle uses may hold anything: the glTF loader keeps the vertices of the triangles it drops,
// a corner that isn't finite among them (gltf.ixx), and the upload stores 0 for them. Returns the centre of the used
// vertices' bounds, rounded to float (mesh_frame).
centre check_mesh(const mesh_desc* mesh, std::size_t i, std::size_t materials)
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
    std::vector<bool> used(vertices, false);
    for (const std::uint32_t v : mesh->indices)
    {
        if (v >= vertices)
        {
            refuse(std::format("meshes[{}] has the index {} of a vertex it doesn't have ({} vertices)", i, v,
                               vertices));
        }
        used[v] = true;
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
    centre lower{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity()};
    centre upper{-lower[0], -lower[1], -lower[2]};
    for (std::size_t v = 0; v < vertices; ++v)
    {
        if (!used[v])
        {
            continue;
        }
        for (std::size_t a = 0; a < 3; ++a)
        {
            const float x = mesh->positions[3 * v + a];
            if (!std::isfinite(x))
            {
                refuse(std::format("meshes[{}] has a triangle with a vertex that isn't finite (vertex {})", i, v));
            }
            lower[a] = std::min(lower[a], static_cast<double>(x));
            upper[a] = std::max(upper[a], static_cast<double>(x));
        }
    }
    centre c{};
    for (std::size_t a = 0; a < 3; ++a)
    {
        c[a] = static_cast<float>((lower[a] + upper[a]) / 2);
    }
    return c;
}

// A mesh's positions as the GPU holds them, in a frame of the mesh's own: less its centre (computed in double, then
// rounded), 0 for a vertex no triangle uses. Each placement adds the centre back (its translation grows by A·c), so
// the world doesn't change, while the coordinates OptiX tests the triangles with, and those triangle_surface
// interpolates, stay at the mesh's size wherever it lies: a model whose file puts it 1000 units from its origin,
// placed back near the world's, is tested and interpolated at its own size, not at 1000's (triangles.cuh,
// offset_origin). A centre of 0 changes nothing.
// The price: a face that lies on a coordinate plane of the world no longer interpolates that coordinate exactly. Its
// corners hold −c there, and b0·(−c) + b1·(−c) + b2·(−c), with b0 = 1 − b1 − b2 in float, rounds at c's size: the hit
// comes out a float step or two of c off the plane, on either side (cornell-triangles' white mesh, c_y = 277.5: its
// floor's y within ±2⁻¹⁴ of 0, where it was exactly 0). A books' checker texture on such a face, whose cells change
// at that plane, would then speckle hit by hit; no scene puts one on a mesh.
std::vector<float> mesh_frame(const mesh_desc& mesh, const centre& c)
{
    std::vector<float> positions(mesh.positions.size(), 0.0f);
    for (const std::uint32_t v : mesh.indices)
    {
        for (std::size_t a = 0; a < 3; ++a)
        {
            const std::size_t k = 3 * static_cast<std::size_t>(v) + a;
            positions[k] = static_cast<float>(static_cast<double>(mesh.positions[k]) - c[a]);
        }
    }
    return positions;
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

// What a triangle may wear of the scene's materials: a lambertian (any books' texture) or a metal, also as a GGX side
// entry completes them (a rough metal, a glossy coat). Glass would be absorbed by the through-rule at its first face,
// and a lamp would need lamp_of entries and the light samples' skip rule for triangles: both refused, with the reason.
void check_scene_material(const scene_desc& scene, std::size_t i, int index)
{
    if (static_cast<std::size_t>(index) >= scene.materials.size())
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

// glTF's metallic-roughness factors in their ranges (glTF 2.0, material.pbrMetallicRoughness: each in [0, 1]; a base
// color above 1 is clamped after the texture's product, so only a negative or a non-finite one is refused), and its
// textures among the set's (checked before), each decoded as its use reads it: the hardware decodes a texture's sRGB
// by its image's flag (texture_on_device), and glTF's base color is sRGB while its metallic-roughness data is linear
// (glTF 2.0, 3.9.2), so a flag that disagrees with the use would read every texel of it wrong
void check_gltf_material(const mesh_material_desc& m, std::size_t i,
                         std::span<const std::shared_ptr<const mesh_texture_desc>> textures)
{
    if (!std::ranges::all_of(m.base_color, [](double x) { return x >= 0 && std::isfinite(x); }))
    {
        refuse(std::format("materials[{}] has the base color ({}, {}, {}): each channel must be finite and at least 0",
                           i, m.base_color[0], m.base_color[1], m.base_color[2]));
    }
    const auto unit = [](double x) { return x >= 0 && x <= 1; }; // NaN fails both
    if (!unit(m.metallic))
    {
        refuse(std::format("materials[{}] has the metallic factor {}, outside 0 to 1", i, m.metallic));
    }
    if (!unit(m.roughness))
    {
        refuse(std::format("materials[{}] has the roughness factor {}, outside 0 to 1", i, m.roughness));
    }
    const struct
    {
        int texture;
        const char* use;
        bool srgb;
    } uses[2] = {{m.base_color_texture, "base color", true}, {m.metal_rough_texture, "metallic-roughness", false}};
    for (const auto& [texture, use, srgb] : uses)
    {
        if (texture < -1 || texture >= std::ssize(textures))
        {
            refuse(std::format("materials[{}] names the texture {} for its {}, which doesn't exist ({} textures; -1 "
                               "for none)",
                               i, texture, use, textures.size()));
        }
        if (texture >= 0 && textures[static_cast<std::size_t>(texture)]->srgb != srgb)
        {
            refuse(std::format("materials[{}] reads textures[{}] as its {}, which must be {}", i, texture, use,
                               srgb ? "decoded from sRGB (srgb true)" : "read linearly (srgb false)"));
        }
    }
}

// The largest sides of a 2D CUDA array the device makes (cudaMallocArray's range), asked before anything is allocated
struct texture_limits
{
    int width = 0;
    int height = 0;
};

texture_limits device_texture_limits()
{
    int device = 0;
    check(cudaGetDevice(&device), "asking which GPU the meshes go to");
    texture_limits limits;
    check(cudaDeviceGetAttribute(&limits.width, cudaDevAttrMaxTexture2DWidth, device),
          "asking the GPU's largest texture");
    check(cudaDeviceGetAttribute(&limits.height, cudaDevAttrMaxTexture2DHeight, device),
          "asking the GPU's largest texture");
    return limits;
}

bool known_wrap(texture_wrap wrap)
{
    return wrap == texture_wrap::repeat || wrap == texture_wrap::mirrored_repeat || wrap == texture_wrap::clamp;
}

// An image as the texture hardware reads it: 8-bit RGBA, width × height texels, each side within the device's limits,
// and one of glTF's three wrap modes per axis
void check_texture(const mesh_texture_desc* image, std::size_t i, const texture_limits& limits)
{
    if (image == nullptr)
    {
        refuse(std::format("textures[{}] is missing", i));
    }
    if (image->width < 1 || image->height < 1 || image->width > limits.width || image->height > limits.height)
    {
        refuse(std::format("textures[{}] is {} by {} texels: each side must be from 1 to the GPU's limit, {} by {}", i,
                           image->width, image->height, limits.width, limits.height));
    }
    if (!known_wrap(image->wrap_s) || !known_wrap(image->wrap_t))
    {
        refuse(std::format("textures[{}] has an unknown wrap mode (glTF's are repeat, mirrored repeat and clamp to "
                           "edge)",
                           i));
    }
    const std::size_t texels = static_cast<std::size_t>(image->width) * static_cast<std::size_t>(image->height);
    if (image->rgba.size() != 4 * texels)
    {
        refuse(std::format("textures[{}] has {} bytes for its {} by {} texels, not 4 a texel", i, image->rgba.size(),
                           image->width, image->height));
    }
}

cudaTextureAddressMode address_mode(texture_wrap wrap)
{
    switch (wrap)
    {
    case texture_wrap::repeat:
        return cudaAddressModeWrap;
    case texture_wrap::mirrored_repeat:
        return cudaAddressModeMirror;
    case texture_wrap::clamp:
        return cudaAddressModeClamp;
    }
    throw std::logic_error("check_texture lets only glTF's three wrap modes through");
}

// A glTF image on the GPU: one CUDA array of uchar4 texels and the texture object that reads it, both freed
// with it (RAII). The object is glTF's sampler as the texture unit's modes: its wrap per axis (REPEAT → Wrap,
// MIRRORED_REPEAT → Mirror, CLAMP_TO_EDGE → Clamp), bilinear (point for magFilter NEAREST), normalized coordinates, and
// the texels read as floats in [0, 1], decoded from sRGB by the hardware for a base color and left linear for
// metallic-roughness data. 4 bytes a texel. Row 0 is the image's top row (stb's first), where glTF's (0, 0) lies: no
// flip. No mipmaps.
class texture_on_device
{
public:
    explicit texture_on_device(const mesh_texture_desc& image)
    {
        const cudaChannelFormatDesc texel = cudaCreateChannelDesc<uchar4>();
        const auto width = static_cast<std::size_t>(image.width);
        const auto height = static_cast<std::size_t>(image.height);
        check(cudaMallocArray(&array_, &texel, width, height), "allocating a texture on the GPU");
        try
        {
            check(cudaMemcpy2DToArray(array_, 0, 0, image.rgba.data(), 4 * width, 4 * width, height,
                                      cudaMemcpyHostToDevice),
                  "copying a texture to the GPU");
            cudaResourceDesc resource{};
            resource.resType = cudaResourceTypeArray;
            resource.res.array.array = array_;
            cudaTextureDesc sampling{};
            sampling.addressMode[0] = address_mode(image.wrap_s);
            sampling.addressMode[1] = address_mode(image.wrap_t);
            sampling.filterMode = image.nearest ? cudaFilterModePoint : cudaFilterModeLinear;
            sampling.readMode = cudaReadModeNormalizedFloat;
            sampling.sRGB = image.srgb ? 1 : 0;
            sampling.normalizedCoords = 1;
            check(cudaCreateTextureObject(&object_, &resource, &sampling, nullptr), "making a texture object");
        }
        catch (...)
        {
            cudaFreeArray(array_);
            throw;
        }
    }

    ~texture_on_device()
    {
        cudaDestroyTextureObject(object_);
        cudaFreeArray(array_);
    }

    texture_on_device(const texture_on_device&) = delete;
    texture_on_device& operator=(const texture_on_device&) = delete;

    cudaTextureObject_t object() const { return object_; }

private:
    cudaArray_t array_ = nullptr;
    cudaTextureObject_t object_ = 0;
};

// One scratch slot: a material and its side entry, 36 bytes, written by triangle_surface at each glTF hit
constexpr std::size_t slot_bytes = sizeof(material<float>) + sizeof(microfacet);

// The memory plan's check, for the meshes that hold `held` bytes already and are about to allocate `more`: the two
// held at once stay within the cap, and `more` leaves 256 MiB of the GPU's memory free. What they hold is no longer
// among the free bytes cudaMemGetInfo reports, so it isn't asked for again.
void check_memory(std::size_t held, std::size_t more, const char* what)
{
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "asking how much of the GPU's memory is free");
    if (held + more > max_mesh_bytes)
    {
        refuse(std::format("{} would hold {} MiB of GPU memory at once, more than the {} MiB a scene's meshes may take",
                           what, (held + more + mib - 1) / mib, max_mesh_bytes / mib));
    }
    if (free_bytes < more + kept_free_bytes)
    {
        refuse(std::format("{} need {} MiB more of GPU memory, more than the {} MiB free on the GPU less {} MiB kept "
                           "for the rest",
                           what, (more + mib - 1) / mib, free_bytes / mib, kept_free_bytes / mib));
    }
}

void copy_in(void* to, const void* from, std::size_t bytes)
{
    if (bytes > 0)
    {
        check(cudaMemcpy(to, from, bytes, cudaMemcpyHostToDevice), "copying a triangle mesh to the GPU");
    }
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
    std::vector<std::unique_ptr<texture_on_device>> textures; // by mesh_set::textures' index
    // A compacted triangle GAS per mesh, and the IAS's mesh instances (instance k: id k + 1, sbtOffset 3)
    std::vector<optix_core::accel> structures;
    std::vector<optix_core::instance> placed;
    // The integrator's tables, lengthened: the scene's M materials and their side entries, then the absorber at index
    // M, then (glTF's materials only) one scratch slot per pixel of the view from index M + 1; lamp_of up to and
    // including the absorber's id (only in a scene with lamps: lamp_hit_pdf reads it then only)
    device_array<material<float>> lengthened_materials{0};
    device_array<microfacet> lengthened_microfacets{0}; // with side entries in the scene, or glTF's materials
    device_array<int> lamp_of{0};
    int instance_count = 0;
    int triangle_base = std::numeric_limits<int>::max();
    int absorber = 0;
    bool gltf = false;     // a triangle wears glTF's material: the slots exist
    std::size_t slots = 0; // how many
    // The bytes the meshes hold once they are built, for prepare_view's plan: the arrays, the textures, the tables,
    // the compacted GASes and the IAS scene.cu builds over them (the builds' temp buffers and the IAS's records gone)
    std::size_t held = 0;
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
    lay.centres.reserve(set.meshes.size());
    for (std::size_t i = 0; i < set.meshes.size(); ++i)
    {
        const mesh_desc* mesh = set.meshes[i].get();
        lay.centres.push_back(check_mesh(mesh, i, set.materials.size()));
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
    if (!set.textures.empty())
    {
        const texture_limits limits = device_texture_limits();
        for (std::size_t i = 0; i < set.textures.size(); ++i)
        {
            check_texture(set.textures[i].get(), i, limits);
        }
    }
    for (std::size_t i = 0; i < set.materials.size(); ++i)
    {
        const mesh_material_desc& m = set.materials[i];
        if (m.scene_material >= 0)
        {
            check_scene_material(scene, i, m.scene_material);
        }
        else if (m.scene_material == -1)
        {
            check_gltf_material(m, i, set.textures);
            s.gltf = true;
        }
        else
        {
            refuse(std::format("materials[{}] names the scene's material {} (-1: glTF's own fields)", i,
                               m.scene_material));
        }
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
        // The mesh's centre (mesh_frame) moves into the placement: M·(x + c) = A·x + (A·c + t), in double
        std::array<double, 12> m = placement.object_to_world;
        const centre& c = lay.centres[static_cast<std::size_t>(placement.mesh)];
        for (std::size_t r = 0; r < 3; ++r)
        {
            m[4 * r + 3] += m[4 * r] * c[0] + m[4 * r + 1] * c[1] + m[4 * r + 2] * c[2];
        }
        const double det = determinant(m);
        const std::array<double, 9> n = inverse_transpose(m, det);
        mesh_instance placed{.part = placement.mesh, .first_id = static_cast<int>(std::min<std::int64_t>(
                                                         next_id, std::numeric_limits<int>::max()))};
        std::ranges::transform(m, placed.object_to_world, [](double x) { return static_cast<float>(x); });
        std::ranges::transform(n, placed.normal_to_world, [](double x) { return static_cast<float>(x); });
        const auto finite = [](float x) { return std::isfinite(x); };
        if (!std::isfinite(det) || det == 0 || !std::ranges::all_of(placed.object_to_world, finite) ||
            !std::ranges::all_of(placed.normal_to_world, finite))
        {
            refuse(std::format("instances[{}] has a transform that isn't finite or can't be inverted", k));
        }
        instances.push_back(placed);
        next_id += static_cast<std::int64_t>(set.meshes[static_cast<std::size_t>(placement.mesh)]->indices.size() / 3);
        if (next_id > std::int64_t{std::numeric_limits<int>::max()} + 1)
        {
            refuse("the placed triangles' ids pass 2^31 - 1: too many triangles placed");
        }
    }
    // The scratch slots: one per pixel of the session's first view (scene.camera), from index M + 1; a larger view
    // grows them (prepare_view). A slot's index is an int, as a material's.
    const std::size_t scene_materials = scene.materials.size();
    s.absorber = static_cast<int>(scene_materials);
    s.slots = s.gltf ? static_cast<std::size_t>(scene.camera.image_width) *
                           static_cast<std::size_t>(scene.camera.image_height)
                     : 0;
    if (scene_materials + 1 + s.slots > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        refuse("too many pixels for the scratch slots of glTF's materials");
    }

    // --- The memory plan, before anything is allocated -------------------------------------------------------------
    const bool side_entries = view.microfacets != nullptr || s.gltf; // the lengthened side entries exist
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
    std::size_t texels = 0;
    for (const auto& image : set.textures)
    {
        texels += static_cast<std::size_t>(image->width) * static_cast<std::size_t>(image->height);
    }
    const std::size_t tables =
        (scene_materials + 1 + s.slots) * (sizeof(material<float>) + (side_entries ? sizeof(microfacet) : 0)) +
        (lamps ? (static_cast<std::size_t>(s.triangle_base) + 1) * sizeof(int) : 0);
    // The IAS that scene.cu builds over the books' GAS and these instances: its records, output and temp at once
    const auto placements = static_cast<unsigned int>(instances.size() + 1);
    const std::size_t planned = arrays + 4 * texels + tables + optix_core::triangles_bytes(core, inputs) +
                                optix_core::instances_bytes(core, placements);
    check_memory(0, planned, "they");

    // --- The shared arrays, every mesh's slice copied in place -------------------------------------------------------
    s.positions = device_array<float3>(lay.vertices);
    s.normals = device_array<float3>(lay.any_normals ? lay.vertices : 0);
    s.uvs = device_array<float2>(lay.any_uvs ? lay.vertices : 0);
    s.triangles = device_array<uint3>(lay.triangles);
    s.material_of = device_array<std::uint32_t>(lay.triangles);
    for (std::size_t i = 0; i < set.meshes.size(); ++i)
    {
        const mesh_desc& mesh = *set.meshes[i];
        const mesh_part& part = lay.parts[i];
        const std::vector<float> positions = mesh_frame(mesh, lay.centres[i]);
        copy_in(s.positions.data() + part.first_vertex, positions.data(), positions.size() * sizeof(float));
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
    s.instance_count = static_cast<int>(instances.size());

    // --- The textures, and the materials that read them --------------------------------------------------------------
    s.textures.reserve(set.textures.size());
    for (const auto& image : set.textures)
    {
        s.textures.push_back(std::make_unique<texture_on_device>(*image));
    }
    std::vector<mesh_material> materials;
    materials.reserve(set.materials.size());
    for (const mesh_material_desc& m : set.materials)
    {
        mesh_material worn{.scene_material = m.scene_material,
                           .base_color = {static_cast<float>(m.base_color[0]), static_cast<float>(m.base_color[1]),
                                          static_cast<float>(m.base_color[2])},
                           .metallic = static_cast<float>(m.metallic),
                           .roughness = static_cast<float>(m.roughness)};
        if (m.scene_material < 0 && m.base_color_texture >= 0)
        {
            worn.base_color_texture = s.textures[static_cast<std::size_t>(m.base_color_texture)]->object();
            worn.has_base_color_texture = 1;
        }
        if (m.scene_material < 0 && m.metal_rough_texture >= 0)
        {
            worn.metal_rough_texture = s.textures[static_cast<std::size_t>(m.metal_rough_texture)]->object();
            worn.has_metal_rough_texture = 1;
        }
        materials.push_back(worn);
    }
    s.materials = device_array<mesh_material>(materials);

    // --- A GAS per mesh, and the instances the IAS will hold ---------------------------------------------------------
    s.structures = optix_core::build_triangles(core, inputs);
    s.held = arrays + 4 * texels + tables + optix_core::instance_structure_bytes(core, placements);
    for (const optix_core::accel& structure : s.structures)
    {
        s.held += structure.bytes; // compacted, where that was smaller
    }
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
    // color 0, its side entry none. A scene with glTF's materials but no side entries of its own gets side entries
    // none for its M materials (as nullptr, make_bsdf's books' lobes) and the slots' after them. These are data, not
    // code: the integrator reads them as it reads the scene's; the slots are written before they are read.
    const std::size_t length = scene_materials + 1 + s.slots;
    s.lengthened_materials = device_array<material<float>>(length);
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
        const microfacet none{.kind = microfacet_kind::none, .alpha = 0, .roughness = 0};
        s.lengthened_microfacets = device_array<microfacet>(length);
        if (view.microfacets != nullptr)
        {
            check(cudaMemcpy(s.lengthened_microfacets.data(), view.microfacets, scene_materials * sizeof(microfacet),
                             cudaMemcpyDeviceToDevice),
                  "lengthening the GGX side entries for the triangle meshes");
            copy_in(s.lengthened_microfacets.data() + scene_materials, &none, sizeof none);
        }
        else
        {
            const std::vector<microfacet> nones(scene_materials + 1, none);
            copy_in(s.lengthened_microfacets.data(), nones.data(), nones.size() * sizeof(microfacet));
        }
    }
    // lamp_of: the scene's (one entry per surface primitive, lamp_table.cu), then -1 for the particles and for the
    // absorber, whose id lamp_hit_pdf meets when a path ends there
    if (lamps)
    {
        const std::size_t surfaces = scene.spheres.size() + scene.quads.size();
        const auto ids = static_cast<std::size_t>(s.triangle_base) + 1;
        s.lamp_of = device_array<int>(ids);
        check(cudaMemcpy(s.lamp_of.data(), view.lamps.lamp_of, surfaces * sizeof(int), cudaMemcpyDeviceToDevice),
              "lengthening lamp_of for the triangle meshes");
        const std::vector<int> none(ids - surfaces, -1);
        copy_in(s.lamp_of.data() + surfaces, none.data(), none.size() * sizeof(int));
    }
}

mesh_scene::~mesh_scene() = default;

bool mesh_scene::empty() const
{
    return state_->instance_count == 0;
}

// The view with the three lengthened tables in place of the scene's: only those pointers are replaced, so the result
// of this function given to it again comes back with the current ones (scene.cu lengthens its view again after
// prepare_view, which may move the materials and side entries)
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
    t.scratch_base = s.gltf ? s.absorber + 1 : -1;
    t.scratch_materials = s.lengthened_materials.data();
    t.scratch_microfacets = s.lengthened_microfacets.size() > 0 ? s.lengthened_microfacets.data() : nullptr;
    return t;
}

std::span<const optix_core::instance> mesh_scene::instances() const
{
    return state_->placed;
}

// A view larger than any before needs more slots: the materials and side entries are made again, longer, their first
// M + 1 entries copied over (the slots are written before they are read, so theirs aren't). The old and the new
// tables are held at once for the copy: the cap counts both beside everything else the meshes hold, while only the new
// ones must still find room among the GPU's free memory. They never shrink, so the same views in turn allocate
// nothing. Their addresses change: the OptiX scene takes lengthened() and tables() again (scene.cu).
void mesh_scene::prepare_view(int width, int height)
{
    state& s = *state_;
    const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (!s.gltf || pixels <= s.slots)
    {
        return;
    }
    const auto kept = static_cast<std::size_t>(s.absorber) + 1;
    if (kept + pixels > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        refuse("too many pixels for the scratch slots of glTF's materials");
    }
    const std::size_t longer = (kept + pixels) * slot_bytes;
    check_memory(s.held, longer, "they and the view's scratch slots");
    device_array<material<float>> materials(kept + pixels);
    device_array<microfacet> microfacets(kept + pixels);
    check(cudaMemcpy(materials.data(), s.lengthened_materials.data(), kept * sizeof(material<float>),
                     cudaMemcpyDeviceToDevice),
          "growing the scratch slots of glTF's materials");
    check(cudaMemcpy(microfacets.data(), s.lengthened_microfacets.data(), kept * sizeof(microfacet),
                     cudaMemcpyDeviceToDevice),
          "growing the scratch slots of glTF's materials");
    s.held = s.held - (kept + s.slots) * slot_bytes + longer; // the old tables go below
    s.lengthened_materials = std::move(materials);
    s.lengthened_microfacets = std::move(microfacets);
    s.slots = pixels;
}
}
