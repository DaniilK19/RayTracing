// tests/mesh_services.h's functions, with the modules: what the mesh suites (tests/mesh_gpu_tests.cu, nvcc) can't
// import themselves. Compiled by cl.exe into GpuTests (cmake/meshes.cmake). Host code only.

// The headers before `import std;` (see renderer.h)
#include "host_services.h"
#include "mesh_services.h"
#include "renderer.h"

import std;
import gltf;
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

namespace
{
// How many of `list`'s pointers are among `known`'s
template <class T>
std::size_t among(const std::vector<std::shared_ptr<const T>>& list, const std::vector<std::shared_ptr<const T>>& known)
{
    std::set<const T*> seen;
    for (const std::shared_ptr<const T>& p : known)
    {
        seen.insert(p.get());
    }
    return static_cast<std::size_t>(
        std::ranges::count_if(list, [&seen](const std::shared_ptr<const T>& p) { return seen.contains(p.get()); }));
}

// A file's write time moved on for a check, and put back by restore(), or on any other way out (an exception) when the
// check ends
class write_time_mover
{
public:
    write_time_mover(std::filesystem::path file, std::chrono::seconds by, bool& restored)
        : file_(std::move(file)), time_(std::filesystem::last_write_time(file_)), restored_(restored)
    {
        std::filesystem::last_write_time(file_, time_ + by);
        moved_ = true;
    }
    write_time_mover(const write_time_mover&) = delete;
    write_time_mover& operator=(const write_time_mover&) = delete;
    ~write_time_mover() { restore(); }

    void restore()
    {
        if (moved_)
        {
            std::error_code failed;
            std::filesystem::last_write_time(file_, time_, failed);
            restored_ = !failed;
            moved_ = false;
        }
    }

private:
    std::filesystem::path file_;
    std::filesystem::file_time_type time_;
    bool& restored_;
    bool moved_ = false;
};

// A committed model's file: its folder's .gltf or .glb, under assets/models
std::filesystem::path model_file(const std::filesystem::path& models, std::string_view folder)
{
    for (const std::filesystem::directory_entry& e : std::filesystem::directory_iterator(models / folder))
    {
        const std::filesystem::path extension = e.path().extension();
        if (e.is_regular_file() && (extension == ".gltf" || extension == ".glb"))
        {
            return e.path();
        }
    }
    throw std::runtime_error(std::format("no .gltf or .glb in assets/models/{}", folder));
}

// build_gpu_scene without overrides: a scene of the list, or a model's file; std::runtime_error with its message
gpu_build build_scene(const std::string& scene)
{
    std::expected<gpu_build, std::string> built = build_gpu_scene(scene, {});
    if (!built)
    {
        throw std::runtime_error(std::format("{}: {}", scene, built.error()));
    }
    return std::move(*built);
}
}

model_cache_check check_model_cache()
{
    model_cache_check c;
    try
    {
        const std::filesystem::path models = host::find_repo_file("assets/models");
        // The vase: still-life reads the bust first, the vase second
        const std::filesystem::path vase = model_file(models, "brass_vase_01");
        c.touched = vase.lexically_relative(models).generic_string();
        const gpu_build first = build_scene("still-life");
        const gpu_build second = build_scene("still-life");
        const gpu::mesh_set& one = first.desc.meshes;
        c.meshes = one.meshes.size();
        c.null_meshes = static_cast<std::size_t>(
            std::ranges::count_if(one.meshes, [](const std::shared_ptr<const gpu::mesh_desc>& p) { return !p; }));
        c.second_meshes = second.desc.meshes.meshes.size();
        c.shared = among(second.desc.meshes.meshes, one.meshes);
        c.textures = one.textures.size();
        c.second_textures = second.desc.meshes.textures.size();
        c.shared_textures = among(second.desc.meshes.textures, one.textures);
        c.second_read = second.models.size();
        {
            write_time_mover moved(vase, std::chrono::seconds(2), c.restored);
            const gpu_build third = build_scene(vase.string());
            c.third_read = third.models.size();
            c.third_meshes = third.desc.meshes.meshes.size();
            c.third_textures = third.desc.meshes.textures.size();
            c.third_old_textures = among(third.desc.meshes.textures, one.textures);
            const gpu_build fourth = build_scene("still-life");
            c.fourth_read = fourth.models.size();
            c.fourth_meshes = fourth.desc.meshes.meshes.size();
            c.fourth_textures = fourth.desc.meshes.textures.size();
            c.kept_meshes = among(fourth.desc.meshes.meshes, one.meshes);
            c.kept_textures = among(fourth.desc.meshes.textures, one.textures);
            c.reloaded_textures = among(fourth.desc.meshes.textures, third.desc.meshes.textures);
            moved.restore();
            const gpu_build fifth = build_scene("still-life");
            c.fifth_read_count = fifth.models.size();
            for (const model_load& m : fifth.models)
            {
                c.fifth_read += std::format("{}{}", c.fifth_read.empty() ? "" : ", ", m.file);
            }
        }
    }
    catch (const std::exception& error)
    {
        c.problem = error.what();
    }
    return c;
}

namespace
{
// A committed model as its file gives it, read again here (load_gltf), apart from gpu_scene and its cache
meshes::model read_model(const std::filesystem::path& file)
{
    std::expected<meshes::model, std::string> model = load_gltf(file);
    if (!model)
    {
        throw std::runtime_error(model.error());
    }
    return std::move(*model);
}

// Whether the mesh set's material m is one of the model's own: glTF's (no scene material), with the factors of one of
// the model's materials and its textures, the same texels, size and color space (base color sRGB, metallic-roughness
// linear), or none where it has none
bool is_own(const gpu::mesh_set& set, std::uint32_t m, const meshes::model& model)
{
    if (m >= set.materials.size())
    {
        return false;
    }
    const gpu::mesh_material_desc& d = set.materials[m];
    const auto same_texture = [&](int slot, int image, bool srgb) {
        if (image < 0)
        {
            return slot == -1;
        }
        if (slot < 0 || static_cast<std::size_t>(slot) >= set.textures.size() ||
            !set.textures[static_cast<std::size_t>(slot)] || static_cast<std::size_t>(image) >= model.images.size())
        {
            return false;
        }
        const gpu::mesh_texture_desc& t = *set.textures[static_cast<std::size_t>(slot)];
        const meshes::image& i = model.images[static_cast<std::size_t>(image)];
        return t.srgb == srgb && t.width == i.width && t.height == i.height && t.rgba == i.rgba;
    };
    return d.scene_material == -1 && std::ranges::any_of(model.materials, [&](const meshes::pbr_material& p) {
               return d.base_color == p.base_color && d.metallic == p.metallic && d.roughness == p.roughness &&
                      same_texture(d.base_color_texture, p.base_color_image, true) &&
                      same_texture(d.metal_rough_texture, p.metal_rough_image, false);
           });
}
}

model_materials_check check_model_materials()
{
    model_materials_check c;
    try
    {
        const std::filesystem::path models = host::find_repo_file("assets/models");
        const std::array<meshes::model, 2> own{read_model(model_file(models, "marble_bust_01")),
                                               read_model(model_file(models, "brass_vase_01"))};
        const gpu_build built = build_scene("still-life");
        const gpu::mesh_set& set = built.desc.meshes;
        c.materials = set.materials.size();
        c.textures = set.textures.size();
        // The meshes of each model: the bust stands at x = -0.55, the vase at x = 0.6 (gpu_scenes_meshes.ixx), and
        // unit_placement moves neither by more than a few hundredths (its box's middle moved over the origin)
        std::array<std::set<int>, 2> placed;
        for (const gpu::mesh_instance_desc& i : set.instances)
        {
            placed[i.object_to_world[3] < 0 ? 0u : 1u].insert(i.mesh);
        }
        c.bust_meshes = placed[0].size();
        c.vase_meshes = placed[1].size();
        c.both_meshes = static_cast<std::size_t>(
            std::ranges::count_if(placed[0], [&placed](int mesh) { return placed[1].contains(mesh); }));
        for (std::size_t side = 0; side < 2; ++side)
        {
            std::map<std::uint32_t, bool> own_material; // by material: the textures compared once each
            std::size_t& triangles = side == 0 ? c.bust_triangles : c.vase_triangles;
            std::size_t& wrong = side == 0 ? c.bust_wrong : c.vase_wrong;
            for (const int mesh : placed[side])
            {
                if (mesh < 0 || static_cast<std::size_t>(mesh) >= set.meshes.size() ||
                    !set.meshes[static_cast<std::size_t>(mesh)])
                {
                    throw std::runtime_error(
                        std::format("an instance places mesh {}, which the set doesn't have", mesh));
                }
                const gpu::mesh_desc& m = *set.meshes[static_cast<std::size_t>(mesh)];
                const std::size_t count = m.indices.size() / 3;
                triangles += count;
                wrong += count > m.material_of.size() ? count - m.material_of.size() : 0; // no material at all
                for (const std::uint32_t material : m.material_of)
                {
                    auto [found, added] = own_material.try_emplace(material, false);
                    if (added)
                    {
                        found->second = is_own(set, material, own[side]);
                    }
                    wrong += found->second ? 0u : 1u;
                }
            }
        }
    }
    catch (const std::exception& error)
    {
        c.problem = error.what();
    }
    return c;
}
}
