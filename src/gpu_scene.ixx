// Module gpu_scene: RayTracing's scenes as the data of the GPU renderer's interface (gpu/renderer.h), shared by the GPU
// programs. A scene is built by the CPU ray tracer's own modules, the same objects and the same camera as in
// RayTracing, and translated into plain numbers here. So the programs differ from RayTracing only in how they render,
// and the CPU stays the reference: a scene rendered by both must come out the same up to the noise.
module;

// The header includes standard headers (<vector>, <string>...), so it must come before `import std;`, here in the
// global module fragment: MSVC can merge the two copies of the standard library, the headers' and the module's, only in
// this order (see main_gpu.cpp). A header included here isn't exported with the module: a file that uses the
// renderer's types or calls it includes gpu/renderer.h itself, before its own `import std;`.
#include "gpu/renderer.h"

export module gpu_scene;

import std;
import bvh;
import camera;
import constant_medium;
import gltf;
import hittable;
import image;
import instance;
import gpu_scenes;
import material;
import meshes;
import perlin;
import quad;
import random;
import scenes;
import sphere;
import strategy_options;
import texture;
import vec3;
import view_options;

// --- The scene as the GPU renderer's data ----------------------------------------------------------------------

// The renderer can't use the CPU's objects: they are C++ classes with virtual functions, built by code the GPU
// can't run, and the GPU has its own memory. So they are translated into plain numbers (renderer.h), which the
// renderer copies into the GPU's memory. point3 and color are vec3 too.
gpu::vec3d to_gpu(const vec3& v)
{
    return {v.x, v.y, v.z};
}

// The strategy as the GPU renderer's enum (renderer.h can't import the camera module, so it has its own)
gpu::sampling_strategy to_gpu(sampling_strategy strategy)
{
    switch (strategy)
    {
    case sampling_strategy::uniform:
        return gpu::sampling_strategy::uniform;
    case sampling_strategy::material:
        return gpu::sampling_strategy::material;
    case sampling_strategy::lights:
        return gpu::sampling_strategy::lights;
    case sampling_strategy::mixture:
        return gpu::sampling_strategy::mixture;
    }
    std::unreachable(); // C++23: the switch covers every enumerator
}

// The pixel grid and the lens, as the CPU's camera computed them from the scene's settings, and the strategy it
// renders with
export gpu::camera_desc to_gpu(const camera_frame& frame, sampling_strategy sampling)
{
    return {.image_width = frame.image_width,
            .image_height = frame.image_height,
            .sqrt_spp = frame.sqrt_spp,
            .max_depth = frame.max_depth,
            .center = to_gpu(frame.center),
            .pixel00_loc = to_gpu(frame.pixel00_loc),
            .pixel_delta_u = to_gpu(frame.pixel_delta_u),
            .pixel_delta_v = to_gpu(frame.pixel_delta_v),
            .defocus = frame.defocus_angle > 0, // as in camera::get_ray: an angle of 0 means no lens
            .defocus_disk_u = to_gpu(frame.defocus_disk_u),
            .defocus_disk_v = to_gpu(frame.defocus_disk_v),
            .sky_bottom = to_gpu(frame.background.bottom),
            .sky_top = to_gpu(frame.background.top),
            .sampling = to_gpu(sampling)};
}

// The same for a strategy of the GPU programs: the camera's part of it, the books' strategy it samples bounces with
export gpu::camera_desc to_gpu(const camera_frame& frame, strategy sampling)
{
    return to_gpu(frame, entry_of(sampling).books.value_or(sampling_strategy::material));
}

// Where an object inside instances ends up in the world. The CPU moves every ray into an instance's coordinates and
// the hit back out (instance.ixx); the translation does the opposite, once: it moves the objects themselves into the
// world ("bakes" the instances), so the GPU's kernel knows only world coordinates and traces no ray twice. However
// translate and rotate_y are nested, they add up to one rigid motion: a turn about the y axis, then a shift.
//
// In exact arithmetic the two ways give the same hits: a rigid motion keeps lengths and angles, so the same t, the
// same point and normal, the same side, the same α and β of a quad. Only a sphere's texture coordinates come from its
// own coordinates, so a sphere keeps its turn for them (sphere_desc::uv_cos and uv_sin).
struct placement
{
    double cos_theta = 1; // the turn, as rotate_y keeps it: the cosine and the sine of its angle
    double sin_theta = 0;
    vec3 offset{0, 0, 0}; // then the shift

    // rotate_y::to_world: a direction, or a side of a quad, turns but doesn't move
    vec3 turn(const vec3& v) const
    {
        return {cos_theta * v.x + sin_theta * v.z, v.y, -sin_theta * v.x + cos_theta * v.z};
    }

    // A point turns, then moves: as a hit point leaves rotate_y and then translate
    point3 place(const point3& p) const { return turn(p) + offset; }

    // The placement of the object a translate holds, if the translate is placed here: the object's points move
    // first, by the translate's offset, then go where this placement puts them
    placement inside(const translate& moved) const { return {cos_theta, sin_theta, place(moved.offset())}; }

    // The placement of the object a rotate_y holds: its points turn first. Turns about one axis add up their angles:
    // cos(a + b) = cos a·cos b - sin a·sin b and sin(a + b) = sin a·cos b + cos a·sin b. For the scenes' usual
    // translate(rotate_y(...)), the walk starts from {1, 0, 0}: inside the translate {1, 0, offset} exactly, then
    // {1·c - 0·s, 0·c + 1·s} = {c, s}, the rotate_y's own numbers, bit for bit.
    placement inside(const rotate_y& turned) const
    {
        return {cos_theta * turned.cos_theta() - sin_theta * turned.sin_theta(),
                sin_theta * turned.cos_theta() + cos_theta * turned.sin_theta(), offset};
    }
};

// Calls on_shape(shape, where) for every object under `object` that holds no other objects, with the placement that
// brings it into the world. Lists (the world, a box()) and the nodes of a BVH hold objects; translate and rotate_y hold
// one and move it. Every other object goes to on_shape whole: add_shape translates a volume of smoke (add_medium walks
// its boundary with this function again) and rejects a kind the GPU doesn't know. Stops at the first error that
// on_shape returns. (An abbreviated function template, C++20: on_shape may be any callable that takes a hittable and
// a placement, here a lambda.)
std::expected<void, std::string> for_each_shape(const hittable& object, const placement& where, const auto& on_shape)
{
    // dynamic_cast asks an object what it really is: dynamic_cast<const hittable_list*>(&object) gives a pointer to
    // the list if object is one, and nullptr if it is another kind of hittable
    if (const auto* list = dynamic_cast<const hittable_list*>(&object))
    {
        for (const std::unique_ptr<hittable>& child : list->objects())
        {
            if (auto done = for_each_shape(*child, where, on_shape); !done)
            {
                return done;
            }
        }
        return {};
    }
    // A BVH built inside the world (final's ground and cloud): the GPU renderer builds a tree of its own over all of
    // the world's shapes (on the CPU, see build_bvh in gpu/bvh.cuh), so this walk takes only the objects at this
    // tree's leaves
    if (const auto* node = dynamic_cast<const bvh_node*>(&object))
    {
        if (auto done = for_each_shape(node->left(), where, on_shape); !done)
        {
            return done;
        }
        return for_each_shape(node->right(), where, on_shape);
    }
    if (const auto* moved = dynamic_cast<const translate*>(&object))
    {
        return for_each_shape(moved->object(), where.inside(*moved), on_shape);
    }
    if (const auto* turned = dynamic_cast<const rotate_y*>(&object))
    {
        return for_each_shape(turned->object(), where.inside(*turned), on_shape);
    }
    return on_shape(object, where);
}

// glTF models as the translation meets them (their cache and their conversion: "glTF models" below). A model is
// loaded once per file and shared by every build that places it.
struct loaded_model;
std::filesystem::path find_asset(std::string_view name); // "The environment map" below

// Where a mesh_ref's model is: a file under assets/models (the scene list's), or the path as given (a model file from
// the command line)
enum class model_paths
{
    assets,
    as_given,
};

// A model a build read from its file (one the cache already held isn't read again, and isn't listed): what the
// programs say of it, its warnings once per model before rendering, its counts and the time its reading took
export struct model_load
{
    std::string file;                  // as the scene or the command line names it
    std::size_t triangles = 0;         // placed: each placement's mesh counted once per placement
    std::size_t images = 0;            // decoded
    double seconds = 0;                // reading and decoding it on the CPU
    std::vector<std::string> warnings; // what the renderer can't show, one line each (gltf.ixx)
};

// The models of one build: each file is asked of the cache once per build, by the path the build finds it at, so that
// every mesh_ref of a file (busts' three) and the model file's scene (framed by model_scene_parts, then translated)
// place one model and list its reading once, whatever the cache does meanwhile (another thread's build, a file written
// during the build, a scene of more models than the cache keeps)
class build_models
{
public:
    // loads: where the models read from their files are listed (nullptr: not listed)
    explicit build_models(std::vector<model_load>* loads) : loads_(loads) {}

    // The model in `file`, from the build's own, else from the cache; `named` is the file as the scene or the command
    // line names it, for the list
    std::expected<std::shared_ptr<const loaded_model>, std::string> get(const std::filesystem::path& file,
                                                                        std::string_view named);

private:
    struct held
    {
        std::filesystem::path file;
        std::shared_ptr<const loaded_model> model;
    };
    std::vector<model_load>* loads_;
    std::vector<held> held_;
};

// Translates the CPU's scene into the GPU renderer's tables (renderer.h), or says why the GPU can't render it. It
// reads the CPU's objects where they are, through dynamic_cast and their accessors, and changes nothing.
class scene_translator
{
public:
    // microfacets: the GGX side entries of the scene's CPU materials, by address (gpu_scene_parts::microfacets)
    scene_translator(const camera_frame& frame, sampling_strategy sampling,
                     std::span<const std::pair<const material*, gpu::microfacet_desc>> microfacets)
    {
        desc_.camera = to_gpu(frame, sampling);
        // Materials are stored once per address (material_index), so an address can carry one side entry only: a scene
        // that gave one placeholder two different GGX materials would render one of them in both places, with no
        // error. That is a mistake in the scene's code, not the user's, so it is no message but an exception.
        for (const auto& [mat, side] : microfacets)
        {
            const auto [found, added] = microfacets_.emplace(mat, side);
            if (!added && (found->second.kind != side.kind || found->second.roughness != side.roughness))
            {
                throw std::logic_error("two GGX materials share one placeholder material");
            }
        }
    }

    // Every shape (sphere, quad) and every volume of smoke of the world, wherever it is in the world's lists, BVHs and
    // instances
    std::expected<void, std::string> add_world(const hittable_list& world)
    {
        return for_each_shape(world, placement{}, [this](const hittable& shape, const placement& where) {
            return add_shape(shape, where);
        });
    }

    // The scene's triangle meshes (gpu_scene_parts::meshes), after the world: see "Triangle meshes" below. Their models
    // come from the build's `models`.
    std::expected<void, std::string> add_meshes(std::span<const mesh_ref> refs, model_paths paths,
                                                build_models& models);

    // The tables. Every GGX side entry must have met its placeholder in the world: one that didn't would leave its
    // material rendered as the placeholder (a mirror, a matte surface), with no error, as two entries on one address
    // would above
    gpu::scene_desc finish() &&
    {
        if (microfacets_used_ != microfacets_.size())
        {
            throw std::logic_error("a GGX side entry's placeholder material isn't in the scene's world");
        }
        return std::move(desc_);
    }

private:
    gpu::scene_desc desc_;
    // Objects can share a material or a texture (they are shared_ptrs), and on the GPU they share it too: each one is
    // stored once, the first time something refers to it, and found again by its address. The maps are only looked
    // up, never iterated, so the tables' order depends only on the order of the world's objects, the same on every
    // run: the order of first use, except that a checker comes after the two textures it refers to.
    std::unordered_map<const material*, int> material_indices_;
    std::unordered_map<const texture*, int> texture_indices_;
    // The GGX side entries by the address of the CPU material they complete; empty for the books' scenes
    std::unordered_map<const material*, gpu::microfacet_desc> microfacets_;
    std::size_t microfacets_used_ = 0; // how many of them a material of the world has taken
    // The mesh materials (mesh_set::materials) that wear a scene material, by the scene material's index, and those of
    // glTF's placeholders (gltf_material), by its address: one entry however many meshes wear it. The mesh set's
    // textures by their image's address: each image once.
    std::unordered_map<int, std::uint32_t> mesh_material_slots_;
    std::unordered_map<const material*, std::uint32_t> gltf_slots_;
    std::unordered_map<const gpu::mesh_texture_desc*, int> mesh_texture_slots_;
    // The models placed so far, each with the index of its first mesh in mesh_set::meshes: a model placed by several
    // mesh_refs (busts) is one source, its meshes and materials added once
    struct model_source
    {
        std::shared_ptr<const loaded_model> model;
        int first_mesh;
    };
    std::vector<model_source> model_sources_;

    std::expected<void, std::string> add_shape(const hittable& shape, const placement& where);
    std::expected<void, std::string> add_medium(const constant_medium& volume, const placement& where);
    std::expected<int, std::string> material_index(const material* mat);
    std::expected<int, std::string> texture_index(const texture& tex);
    std::expected<std::uint32_t, std::string> mesh_material_slot(const material* mat);
    std::uint32_t add_gltf_material(const gpu::mesh_material_desc& factors,
                                    std::span<const std::shared_ptr<const gpu::mesh_texture_desc>> images);
    int mesh_texture_slot(const std::shared_ptr<const gpu::mesh_texture_desc>& image);
    std::expected<void, std::string> add_model(const mesh_ref& ref, const std::filesystem::path& file,
                                               build_models& models);
};

std::expected<void, std::string> scene_translator::add_shape(const hittable& shape, const placement& where)
{
    if (const auto* ball = dynamic_cast<const sphere*>(&shape))
    {
        const auto mat = material_index(ball->mat());
        if (!mat)
        {
            return std::unexpected(mat.error());
        }
        // How far the center moves while the shutter is open, from the sphere's own accessors: exactly 0 for a sphere
        // that stands still (c + 1·0 - c), which is how the renderer tells the two kinds apart. Its path is a line, so
        // in the world it starts at place(center) and moves by turn(displacement).
        desc_.spheres.push_back({.center = to_gpu(where.place(ball->center_at(0))),
                                 .radius = ball->radius(),
                                 .material = *mat,
                                 .displacement = to_gpu(where.turn(ball->center_at(1) - ball->center_at(0))),
                                 .uv_cos = where.cos_theta,
                                 .uv_sin = where.sin_theta});
        return {};
    }
    if (const auto* face = dynamic_cast<const quad*>(&shape))
    {
        const auto mat = material_index(face->mat());
        if (!mat)
        {
            return std::unexpected(mat.error());
        }
        desc_.quads.push_back({.q = to_gpu(where.place(face->q())),
                               .u = to_gpu(where.turn(face->u())),
                               .v = to_gpu(where.turn(face->v())),
                               .material = *mat});
        return {};
    }
    if (const auto* volume = dynamic_cast<const constant_medium*>(&shape))
    {
        return add_medium(*volume, where);
    }
    return std::unexpected("it has an object the GPU doesn't know (it renders spheres, quads, boxes and smoke, also in "
                           "lists, BVHs, translate and rotate_y)");
}

std::expected<void, std::string> scene_translator::add_medium(const constant_medium& volume, const placement& where)
{
    // The phase function is always an isotropic (constant_medium's constructor makes one); material_index translates
    // it like any other material
    const auto phase = material_index(&volume.phase_function());
    if (!phase)
    {
        return std::unexpected(phase.error());
    }
    gpu::medium_desc medium{.neg_inv_density = volume.neg_inv_density(), .material = *phase};
    // The boundary only says where the volume is, so its shapes are collected without their materials: the smoke's
    // boxes carry white and the final scene's spheres glass, and neither is ever used. The boundary may sit inside
    // instances of its own (the smoke scene's turned boxes).
    int spheres = 0;
    const auto collected = for_each_shape(
        volume.boundary(), where, [&](const hittable& part, const placement& at) -> std::expected<void, std::string> {
            if (const auto* ball = dynamic_cast<const sphere*>(&part))
            {
                ++spheres;
                medium.sphere = {.center = to_gpu(at.place(ball->center_at(0))),
                                 .radius = ball->radius(),
                                 .material = 0,
                                 .displacement = to_gpu(at.turn(ball->center_at(1) - ball->center_at(0)))};
                return {};
            }
            if (const auto* face = dynamic_cast<const quad*>(&part))
            {
                medium.quads.push_back({.q = to_gpu(at.place(face->q())),
                                        .u = to_gpu(at.turn(face->u())),
                                        .v = to_gpu(at.turn(face->v())),
                                        .material = 0});
                return {};
            }
            return std::unexpected("the boundary of its smoke or fog holds something other than spheres and quads");
        });
    if (!collected)
    {
        return collected;
    }
    if (spheres == 1 && medium.quads.empty())
    {
        medium.boundary = gpu::boundary_shape::sphere;
    }
    else if (spheres == 0 && !medium.quads.empty() && gpu::is_convex(medium.quads))
    {
        medium.boundary = gpu::boundary_shape::convex_quads;
    }
    else
    {
        return std::unexpected("the boundary of its smoke or fog isn't one sphere or a convex shape made of quads");
    }
    desc_.media.push_back(std::move(medium));
    return {};
}

std::expected<int, std::string> scene_translator::material_index(const material* mat)
{
    if (!mat)
    {
        return std::unexpected("an object of its world has no material");
    }
    if (const auto found = material_indices_.find(mat); found != material_indices_.end())
    {
        return found->second;
    }
    gpu::material_desc desc;
    if (const auto* matte = dynamic_cast<const lambertian*>(mat))
    {
        const auto tex = texture_index(matte->albedo_texture());
        if (!tex)
        {
            return tex;
        }
        desc = {.kind = gpu::material_kind::lambertian, .texture = *tex};
    }
    else if (const auto* mirror = dynamic_cast<const metal*>(mat))
    {
        desc = {.kind = gpu::material_kind::metal, .albedo = to_gpu(mirror->albedo()), .fuzz = mirror->fuzz()};
    }
    else if (const auto* glass = dynamic_cast<const dielectric*>(mat))
    {
        desc = {.kind = gpu::material_kind::dielectric, .refraction_index = glass->refraction_index()};
    }
    else if (const auto* lamp = dynamic_cast<const diffuse_light*>(mat))
    {
        const auto tex = texture_index(lamp->emission_texture());
        if (!tex)
        {
            return tex;
        }
        desc = {.kind = gpu::material_kind::diffuse_light, .texture = *tex};
    }
    else if (const auto* particles = dynamic_cast<const isotropic*>(mat))
    {
        const auto tex = texture_index(particles->albedo_texture());
        if (!tex)
        {
            return tex;
        }
        desc = {.kind = gpu::material_kind::isotropic, .texture = *tex};
    }
    else
    {
        return std::unexpected(
            "it has a material other than lambertian, metal, dielectric, diffuse_light and isotropic");
    }
    const auto index = static_cast<int>(desc_.materials.size());
    desc_.materials.push_back(desc);
    material_indices_.emplace(mat, index);
    // Its GGX side entry, in a scene that has any: then every material has one (`none` but for the GGX materials'
    // placeholders), in the order of the materials
    if (!microfacets_.empty())
    {
        const auto side = microfacets_.find(mat);
        microfacets_used_ += side != microfacets_.end() ? 1 : 0; // once per material: this runs at its first use only
        desc_.quality.microfacets.push_back(side != microfacets_.end() ? side->second : gpu::microfacet_desc{});
    }
    return index;
}

std::expected<int, std::string> scene_translator::texture_index(const texture& tex)
{
    if (const auto found = texture_indices_.find(&tex); found != texture_indices_.end())
    {
        return found->second;
    }
    gpu::texture_desc desc;
    if (const auto* solid = dynamic_cast<const solid_color*>(&tex))
    {
        desc = {.kind = gpu::texture_kind::solid, .rgb = to_gpu(solid->albedo())};
    }
    else if (const auto* checker = dynamic_cast<const checker_texture*>(&tex))
    {
        // Its two textures first: they get smaller indices than the checker (see texture_desc)
        const auto even = texture_index(checker->even());
        if (!even)
        {
            return even;
        }
        const auto odd = texture_index(checker->odd());
        if (!odd)
        {
            return odd;
        }
        desc = {.kind = gpu::texture_kind::checker, .inv_scale = checker->inv_scale(), .even = *even, .odd = *odd};
    }
    else if (const auto* picture = dynamic_cast<const image_texture*>(&tex))
    {
        // An image belongs to exactly one image_texture (a member), so storing each texture once stores each image once
        const image& img = picture->source_image();
        gpu::image_desc pixels{.width = img.width(), .height = img.height()};
        pixels.rgb.reserve(3 * static_cast<std::size_t>(img.width()) * static_cast<std::size_t>(img.height()));
        for (int y = 0; y < img.height(); ++y)
        {
            for (int x = 0; x < img.width(); ++x)
            {
                // Exact: image::load made these doubles from stb_image's floats
                const color& c = img[x, y];
                pixels.rgb.push_back(static_cast<float>(c.x));
                pixels.rgb.push_back(static_cast<float>(c.y));
                pixels.rgb.push_back(static_cast<float>(c.z));
            }
        }
        desc = {.kind = gpu::texture_kind::image, .image = static_cast<int>(desc_.images.size())};
        desc_.images.push_back(std::move(pixels));
    }
    else if (const auto* marble = dynamic_cast<const noise_texture*>(&tex))
    {
        // The tables of this texture's own perlin object, drawn while the scene was built: copied, never drawn again
        static_assert(perlin::point_count == gpu::perlin_point_count);
        const perlin& noise = marble->noise();
        gpu::perlin_desc tables;
        for (std::size_t i = 0; i < tables.random_vectors.size(); ++i)
        {
            tables.random_vectors[i] = to_gpu(noise.random_vectors()[i]);
            tables.perm_x[i] = noise.perm_x()[i];
            tables.perm_y[i] = noise.perm_y()[i];
            tables.perm_z[i] = noise.perm_z()[i];
        }
        desc = {.kind = gpu::texture_kind::noise,
                .scale = marble->scale(),
                .perlin = static_cast<int>(desc_.perlins.size())};
        desc_.perlins.push_back(tables);
    }
    else
    {
        return std::unexpected(
            "it has a texture other than solid_color, checker_texture, image_texture and noise_texture");
    }
    const auto index = static_cast<int>(desc_.textures.size());
    desc_.textures.push_back(desc);
    texture_indices_.emplace(&tex, index);
    return index;
}

// --- Triangle meshes ---------------------------------------------------------------------------------------------
//
// The CPU builds no triangle: a scene's meshes ride beside its world as mesh_refs (gpu_scene_parts::meshes), which
// become the renderer's mesh_set (renderer.h), each distinct source once (one acceleration structure for OptiX) and
// every mesh_ref an instance of it. A mesh made in code is one source per function and material: its triangles wear
// that material, translated as the world's are (material_index, which also brings its GGX side entry), so a mesh and
// a quad of one material share it. Which materials a triangle may wear is the upload's to check (meshes.cu): lambertian
// and metal, also completed by a side entry, and glTF's metallic-roughness materials, which a mesh made in code wears
// through a placeholder (gltf_material, gpu_scenes:meshes).

// The mesh set's texture of an image, added at its first use
int scene_translator::mesh_texture_slot(const std::shared_ptr<const gpu::mesh_texture_desc>& image)
{
    const auto [found, added] =
        mesh_texture_slots_.try_emplace(image.get(), static_cast<int>(desc_.meshes.textures.size()));
    if (added)
    {
        desc_.meshes.textures.push_back(image);
    }
    return found->second;
}

// A glTF material of the mesh set: its factors, its texture indices (into `images`, -1 for none) made the set's. An
// index `images` doesn't have, below -1 too, is the program's mistake (the loader and the scenes make both), an
// exception.
std::uint32_t scene_translator::add_gltf_material(const gpu::mesh_material_desc& factors,
                                                  std::span<const std::shared_ptr<const gpu::mesh_texture_desc>> images)
{
    gpu::mesh_material_desc m = factors;
    m.scene_material = -1;
    for (int* texture : {&m.base_color_texture, &m.metal_rough_texture})
    {
        const auto at = static_cast<std::size_t>(*texture);
        if (*texture < -1 || *texture >= static_cast<int>(images.size()) || (*texture >= 0 && !images[at]))
        {
            throw std::logic_error("a glTF material names an image it doesn't have");
        }
        *texture = *texture >= 0 ? mesh_texture_slot(images[at]) : -1;
    }
    desc_.meshes.materials.push_back(m);
    return static_cast<std::uint32_t>(desc_.meshes.materials.size() - 1);
}

// The mesh material (an index into mesh_set::materials) that wears the CPU material `mat`: glTF's for its
// placeholder, else the scene material it translates to
std::expected<std::uint32_t, std::string> scene_translator::mesh_material_slot(const material* mat)
{
    if (const auto* gltf = dynamic_cast<const gltf_material*>(mat))
    {
        if (const auto found = gltf_slots_.find(mat); found != gltf_slots_.end())
        {
            return found->second;
        }
        const std::uint32_t slot = add_gltf_material(gltf->factors(), gltf->images());
        gltf_slots_.emplace(mat, slot);
        return slot;
    }
    const auto index = material_index(mat);
    if (!index)
    {
        return std::unexpected(index.error());
    }
    const auto [found, added] =
        mesh_material_slots_.try_emplace(*index, static_cast<std::uint32_t>(desc_.meshes.materials.size()));
    if (added)
    {
        desc_.meshes.materials.push_back({.scene_material = *index});
    }
    return found->second;
}

std::expected<void, std::string> scene_translator::add_meshes(std::span<const mesh_ref> refs, model_paths paths,
                                                              build_models& models)
{
    // The meshes made in code so far, by their function and material, and their index in mesh_set::meshes
    struct generated_source
    {
        gpu::mesh_desc (*generate)();
        const material* worn;
        int mesh;
    };
    std::vector<generated_source> generated;
    for (const mesh_ref& ref : refs)
    {
        if (!ref.model.empty())
        {
            // With the system's separators throughout, as the loader's messages name the file
            const std::filesystem::path file = paths == model_paths::assets
                                                   ? find_asset(std::format("models/{}", ref.model)).make_preferred()
                                                   : std::filesystem::path(ref.model);
            if (auto added = add_model(ref, file, models); !added)
            {
                return added;
            }
            continue;
        }
        if (!ref.generated)
        {
            return std::unexpected("a mesh of its list has neither a model nor a function that makes it");
        }
        if (!ref.material)
        {
            return std::unexpected("a mesh it makes in code wears no material");
        }
        const auto same = [&ref](const generated_source& g) {
            return g.generate == ref.generated && g.worn == ref.material.get();
        };
        auto source = std::ranges::find_if(generated, same);
        if (source == generated.end())
        {
            const auto slot = mesh_material_slot(ref.material.get());
            if (!slot)
            {
                return std::unexpected(slot.error());
            }
            // A generated mesh's triangles all wear material 0 of their own: here, the one slot of its material
            gpu::mesh_desc made = ref.generated();
            made.material_of.assign(made.indices.size() / 3, *slot);
            const auto mesh = static_cast<int>(desc_.meshes.meshes.size());
            desc_.meshes.meshes.push_back(std::make_shared<const gpu::mesh_desc>(std::move(made)));
            generated.push_back({.generate = ref.generated, .worn = ref.material.get(), .mesh = mesh});
            source = std::prev(generated.end());
        }
        desc_.meshes.instances.push_back({.mesh = source->mesh, .object_to_world = ref.placement});
    }
    return {};
}

// An object of the lights list and what it holds, with `weight`, the chance of aiming at it: hittable_list::random
// picks each of a list's n objects with the chance 1/n, so a list inside the list shares its weight among its objects.
// Instances are baked as in the world (see placement), which keeps every density: a rigid motion maps the solid angle
// that a shape covers from a point onto the solid angle of the moved shape from the moved point (MonteCarlo's
// light-pdfs checks this on the CPU).
std::expected<void, std::string> add_light(const hittable& object, const placement& where, double weight,
                                           std::vector<gpu::light_desc>& lights)
{
    if (const auto* list = dynamic_cast<const hittable_list*>(&object))
    {
        const auto children = list->objects();
        if (children.empty())
        {
            // The CPU gives it the density 0 and the direction (1, 0, 0) (see hittable_list): nothing to imitate
            return std::unexpected("it holds an empty list");
        }
        for (const std::unique_ptr<hittable>& child : children)
        {
            if (auto added = add_light(*child, where, weight / static_cast<double>(children.size()), lights); !added)
            {
                return added;
            }
        }
        return {};
    }
    if (const auto* moved = dynamic_cast<const translate*>(&object))
    {
        return add_light(moved->object(), where.inside(*moved), weight, lights);
    }
    if (const auto* turned = dynamic_cast<const rotate_y*>(&object))
    {
        return add_light(turned->object(), where.inside(*turned), weight, lights);
    }
    if (const auto* face = dynamic_cast<const quad*>(&object))
    {
        if (!(cross(face->u(), face->v()).length_squared() > 0))
        {
            return std::unexpected("it holds a quad without area");
        }
        // The same numbers as add_shape gives the lamp in the world, built by the same function (see scene)
        lights.push_back({.quad = {.q = to_gpu(where.place(face->q())),
                                   .u = to_gpu(where.turn(face->u())),
                                   .v = to_gpu(where.turn(face->v())),
                                   .material = 0},
                          .weight = weight});
        return {};
    }
    if (const auto* ball = dynamic_cast<const sphere*>(&object))
    {
        if (!(ball->radius() > 0))
        {
            return std::unexpected("it holds a sphere without size");
        }
        // sphere::pdf_value and random aim at the center at time 0, a moving sphere's too
        lights.push_back({.kind = gpu::light_shape::sphere,
                          .center = to_gpu(where.place(ball->center_at(0))),
                          .radius = ball->radius(),
                          .weight = weight});
        return {};
    }
    // A BVH or a volume of smoke keeps hittable's pdf_value and random, which describe no density
    return std::unexpected("it holds an object the GPU can't aim at: only spheres and quads, also in lists, translate "
                           "and rotate_y, have a density to aim with");
}

// The shapes of the lights list, each with its chance of being aimed at, or why the GPU can't aim at them. Only the
// lights and mixture strategies need them: a scene whose list the GPU can't aim at still renders with the others.
std::expected<std::vector<gpu::light_desc>, std::string> lights_to_gpu(const hittable_list& lights)
{
    std::vector<gpu::light_desc> result;
    if (const auto added = add_light(lights, placement{}, 1.0, result); !added)
    {
        return std::unexpected(added.error());
    }
    return result;
}

// The whole scene, or why the GPU can't render it. The GPU renders spheres, standing still or moving, and quads, also
// in lists, BVHs, translate and rotate_y, with matte surfaces of solid, checker, image or marble textures, metal, glass
// and lamps, and volumes of smoke or fog: every scene of the books. sampling is the strategy the CPU would render with
// (effective_sampling). microfacets: the GGX side entries of its materials (gpu_scene_parts::microfacets); meshes: its
// triangle meshes (gpu_scene_parts::meshes), which only OptiX renders, their models found as `paths` says, through the
// build's `models`.
std::expected<gpu::scene_desc, std::string> to_gpu(
    const scene& s, const camera_frame& frame, sampling_strategy sampling,
    std::span<const std::pair<const material*, gpu::microfacet_desc>> microfacets, std::span<const mesh_ref> meshes,
    model_paths paths, build_models& models)
{
    scene_translator translator(frame, sampling, microfacets);
    if (const auto added = translator.add_world(s.world); !added)
    {
        return std::unexpected(added.error());
    }
    if (const auto added = translator.add_meshes(meshes, paths, models); !added)
    {
        return std::unexpected(added.error());
    }
    return std::move(translator).finish();
}

// A scene's parts as the renderer's tables, at their camera's own frame, without the programs' options, a lights list
// or an environment: the translation of the world, of its GGX side entries and of its meshes alone (models from
// assets/models), which GpuTests checks
export std::expected<gpu::scene_desc, std::string> translate_parts(const gpu_scene_parts& parts)
{
    build_models models(nullptr);
    return to_gpu(parts.cpu, camera(parts.cpu.camera).frame(), sampling_strategy::material, parts.microfacets,
                  parts.meshes, model_paths::assets, models);
}

// --- The environment map ---------------------------------------------------------------------------------------

// A file of the project's assets folder, found as the CPU's scenes find theirs (find_asset in scenes.ixx, which that
// module keeps to itself): in the current folder or the nearest one above it that has it, as the programs run from the
// project's root and from Visual Studio, whose current folder is out\build\x64-release. Without one, the plain relative
// path, which then doesn't open.
std::filesystem::path find_asset(std::string_view name)
{
    for (std::filesystem::path dir = std::filesystem::current_path();; dir = dir.parent_path())
    {
        const std::filesystem::path candidate = dir / "assets" / name;
        if (std::filesystem::exists(candidate))
        {
            return candidate;
        }
        if (dir == dir.parent_path())
        {
            return std::filesystem::path("assets") / name;
        }
    }
}

// The renderer's largest map, per side (gpu::max_environment_width and _height in gpu/environment_data.h, a header of
// the renderer's own)
constexpr int max_environment_width = 2048;
constexpr int max_environment_height = 1024;

// The side of the constant map made in code: small, as every texel is the same (and so is every density, 1/(4π))
constexpr int white_map_width = 64;
constexpr int white_map_height = 32;

// The environment map a scene names (gpu_scene_parts::environment) as the renderer's data: none (the camera's sky), an
// equirectangular .hdr image from the assets folder, or a constant map of radiance 1 made in code (a furnace). The
// error is the whole message, as the programs print it after "Error: ".
std::expected<std::optional<gpu::environment_desc>, std::string> load_environment(const environment_choice& choice)
{
    if (choice.source == env_source::none)
    {
        return std::optional<gpu::environment_desc>{};
    }
    gpu::environment_desc environment{.yaw_degrees = choice.yaw_degrees, .scale = choice.scale};
    gpu::image_desc& map = environment.image;
    if (choice.source == env_source::white)
    {
        map = {.width = white_map_width,
               .height = white_map_height,
               .rgb = std::vector<float>(3 * std::size_t{white_map_width} * white_map_height, 1.0f)};
        return std::optional{std::move(environment)};
    }
    // stb_image reads a Radiance .hdr (RGBE) as linear floats, with no gamma: the 2.2 that image::load's comment speaks
    // of is for 8-bit files only. It reads only the usual orientation, rows from the top (-Y H +X W), which is the
    // map's: row 0 is straight up.
    const auto loaded = image::load(find_asset(choice.asset));
    if (!loaded)
    {
        return std::unexpected(std::format("the environment map '{}' doesn't load: {}", choice.asset, loaded.error()));
    }
    const image& img = *loaded;
    if (img.width() > max_environment_width || img.height() > max_environment_height)
    {
        return std::unexpected(std::format("the environment map '{}' is {}×{}: at most {}×{}", choice.asset,
                                           img.width(), img.height(), max_environment_width, max_environment_height));
    }
    map = {.width = img.width(), .height = img.height()};
    map.rgb.reserve(3 * static_cast<std::size_t>(img.width()) * static_cast<std::size_t>(img.height()));
    for (int y = 0; y < img.height(); ++y)
    {
        for (int x = 0; x < img.width(); ++x)
        {
            // Exact: image::load made these doubles from stb_image's floats
            const color& c = img[x, y];
            for (const double value : {c.x, c.y, c.z})
            {
                if (!(std::isfinite(value) && value >= 0))
                {
                    return std::unexpected(std::format(
                        "the environment map '{}' has a texel that isn't a finite radiance ≥ 0, at ({}, {})",
                        choice.asset, x, y));
                }
                map.rgb.push_back(static_cast<float>(value));
            }
        }
    }
    return std::optional{std::move(environment)};
}

// --- glTF models: a file in place of a scene's name ------------------------------------------------------------

// A loaded model as every build reuses it (not exported): meshes::model from load_gltf is converted once per load into
// these shared pieces and dropped; a build copies the shared_ptrs into scene_desc::meshes. Its materials
// are glTF's (scene_material -1), their texture indices into `textures`; its meshes' material_of index `materials`.
struct loaded_model
{
    std::vector<std::shared_ptr<const gpu::mesh_desc>> meshes;
    std::vector<gpu::mesh_instance_desc> placements;
    std::vector<gpu::mesh_material_desc> materials;
    std::vector<std::shared_ptr<const gpu::mesh_texture_desc>> textures;
    std::array<double, 3> lower{};
    std::array<double, 3> upper{};
    std::vector<std::string> warnings;
    std::size_t triangles = 0; // placed: each placement's mesh once per placement
    // The meshes again with every material_of raised by an offset, for a scene whose mesh materials don't start with
    // this model's (still-life's second model: renderer.h's material_of indexes the whole set's materials). Made at the
    // first build that needs an offset and kept with the model, so the later builds share them as they share `meshes`.
    // A cache inside a model that is otherwise never changed: its own mutex guards it.
    mutable std::mutex shifted_mutex;
    mutable std::map<std::uint32_t, std::vector<std::shared_ptr<const gpu::mesh_desc>>> shifted;
};

namespace
{
gpu::texture_wrap to_texture_wrap(meshes::wrap wrap)
{
    switch (wrap)
    {
    case meshes::wrap::repeat:
        return gpu::texture_wrap::repeat;
    case meshes::wrap::mirrored_repeat:
        return gpu::texture_wrap::mirrored_repeat;
    case meshes::wrap::clamp:
        return gpu::texture_wrap::clamp;
    }
    std::unreachable();
}

// A model as the renderer's pieces, once per load: its vectors moved into shared meshes and images
std::shared_ptr<const loaded_model> to_loaded_model(meshes::model m)
{
    auto loaded = std::make_shared<loaded_model>();
    for (meshes::mesh& mesh : m.meshes)
    {
        loaded->meshes.push_back(std::make_shared<const gpu::mesh_desc>(to_mesh_desc(std::move(mesh))));
    }
    for (const meshes::placement& p : m.placements)
    {
        loaded->placements.push_back({.mesh = p.mesh, .object_to_world = p.object_to_world});
        loaded->triangles += loaded->meshes[static_cast<std::size_t>(p.mesh)]->indices.size() / 3;
    }
    for (const meshes::pbr_material& pbr : m.materials)
    {
        loaded->materials.push_back({.scene_material = -1,
                                     .base_color = pbr.base_color,
                                     .metallic = pbr.metallic,
                                     .roughness = pbr.roughness,
                                     .base_color_texture = pbr.base_color_image,
                                     .metal_rough_texture = pbr.metal_rough_image});
    }
    for (meshes::image& image : m.images)
    {
        loaded->textures.push_back(std::make_shared<const gpu::mesh_texture_desc>(
            gpu::mesh_texture_desc{.width = image.width,
                                   .height = image.height,
                                   .rgba = std::move(image.rgba),
                                   .srgb = image.srgb,
                                   .wrap_s = to_texture_wrap(image.wrap_s),
                                   .wrap_t = to_texture_wrap(image.wrap_t),
                                   .nearest = image.nearest}));
    }
    loaded->lower = m.lower;
    loaded->upper = m.upper;
    loaded->warnings = std::move(m.warnings);
    return loaded;
}

// The model's meshes for a scene whose mesh materials start with this model's at `offset`
std::vector<std::shared_ptr<const gpu::mesh_desc>> meshes_at(const loaded_model& model, std::uint32_t offset)
{
    if (offset == 0)
    {
        return model.meshes;
    }
    const std::lock_guard lock(model.shifted_mutex);
    if (const auto found = model.shifted.find(offset); found != model.shifted.end())
    {
        return found->second;
    }
    // Made whole before the map holds them: a copy that throws (out of memory) leaves no part of a set behind, which
    // every later build would take for the whole and place past
    std::vector<std::shared_ptr<const gpu::mesh_desc>> shifted;
    shifted.reserve(model.meshes.size());
    for (const auto& mesh : model.meshes)
    {
        gpu::mesh_desc copy = *mesh;
        for (std::uint32_t& material : copy.material_of)
        {
            material += offset;
        }
        shifted.push_back(std::make_shared<const gpu::mesh_desc>(std::move(copy)));
    }
    return model.shifted.emplace(offset, std::move(shifted)).first->second;
}

// The models used last: two, by their file's canonical path, size and time of its last write, so a scene built again
// (the window's Tab and M, a second render) reads no file again, and a file written since is read again. Only the
// model's own file (the .gltf or .glb) is watched: a .bin or a texture beside a .gltf, written since, is not seen until
// the .gltf changes too (or the program starts again). Two, so that still-life's bust and vase both stay; a hit makes
// its model the most recent, so a new model pushes out the one least recently used, never one a build just used. Host
// memory only (no GPU object), so a static is safe at exit. One build at a time reads (its mutex).
class model_cache
{
public:
    struct found
    {
        std::shared_ptr<const loaded_model> model;
        bool read = false; // from its file just now (else the cache's)
        double seconds = 0;
    };

    std::expected<found, std::string> get(const std::filesystem::path& file)
    {
        const std::lock_guard lock(mutex_);
        // The canonical path, so that two names of one file (relative and absolute, with ..\ in it) are one model.
        // Where the system can't give it (canonical fails on some network and virtual drives), the absolute path with
        // its . and .. taken out by hand (lexically_normal): the same key each time the file is named the same way, so
        // that later builds still find it.
        std::error_code error;
        key k{.path = std::filesystem::canonical(file, error)};
        if (error)
        {
            error.clear();
            k.path = std::filesystem::absolute(file, error).lexically_normal();
        }
        if (!error)
        {
            k.size = std::filesystem::file_size(k.path, error);
        }
        if (!error)
        {
            k.written = std::filesystem::last_write_time(k.path, error);
        }
        if (!error)
        {
            if (const auto known = std::ranges::find(recent_, k, &entry::first); known != recent_.end())
            {
                std::rotate(known, std::next(known), recent_.end()); // now the most recent
                return found{.model = recent_.back().second};
            }
        }
        // Read (also when the file couldn't be found: the loader says why, in its words)
        const auto start = std::chrono::steady_clock::now();
        auto model = load_gltf(file);
        if (!model)
        {
            return std::unexpected(std::move(model.error()));
        }
        found result{.model = to_loaded_model(std::move(*model)), .read = true};
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (!error)
        {
            // A file read again because it was written since replaces its old entry, which no key finds any more: kept,
            // it would push out the other model and hold the old one's memory
            std::erase_if(recent_, [&k](const entry& e) { return e.first.path == k.path; });
            recent_.emplace_back(std::move(k), result.model);
            if (recent_.size() > kept)
            {
                recent_.erase(recent_.begin());
            }
        }
        return result;
    }

private:
    struct key
    {
        std::filesystem::path path;
        std::uintmax_t size = 0;
        std::filesystem::file_time_type written{};
        bool operator==(const key&) const = default;
    };
    using entry = std::pair<key, std::shared_ptr<const loaded_model>>;
    static constexpr std::size_t kept = 2;
    std::mutex mutex_;
    std::vector<entry> recent_; // the least recently used first, at most one entry per path
};

model_cache& the_model_cache()
{
    static model_cache cache;
    return cache;
}

// What a build says of a model it read: its file as named, its counts, its warnings
model_load load_of(std::string_view file, const loaded_model& model, double seconds)
{
    return {.file = std::string(file),
            .triangles = model.triangles,
            .images = model.textures.size(),
            .seconds = seconds,
            .warnings = model.warnings};
}

// The refusal of a model without a triangle, with the loader's warnings, which say why (every primitive dropped, points
// and lines alone), each without the file's name that starts the message once: "lines.glb: it has no triangles to
// render: points and lines skipped, only triangles are rendered (2): meshes[0].primitives[0], meshes[1].primitives[0]"
std::string no_triangles(std::string_view named, const loaded_model& model)
{
    // The loader's lines all start with its name for the file and ": ", and the last of such a model's is that start
    // and "no triangles to render" (gltf.ixx's finish), which the refusal says itself: it gives the start to take off
    constexpr std::string_view none = "no triangles to render";
    std::span<const std::string> lines = model.warnings;
    std::string_view start;
    if (!lines.empty() && lines.back().ends_with(none))
    {
        start = std::string_view(lines.back()).substr(0, lines.back().size() - none.size());
        lines = lines.first(lines.size() - 1);
    }
    std::string why;
    for (std::string_view line : lines)
    {
        if (!start.empty() && line.starts_with(start))
        {
            line.remove_prefix(start.size());
        }
        why += std::format("{}{}", why.empty() ? "" : "; ", line);
    }
    return std::format("{}: it has no triangles to render{}{}", named, why.empty() ? "" : ": ", why);
}

// a·b of row-major 3×4 matrices: b first, then a
std::array<double, 12> compose(const std::array<double, 12>& a, const std::array<double, 12>& b)
{
    std::array<double, 12> c{};
    for (int r = 0; r < 3; ++r)
    {
        for (int k = 0; k < 4; ++k)
        {
            double sum = k == 3 ? a[4 * r + 3] : 0.0;
            for (int j = 0; j < 3; ++j)
            {
                sum += a[4 * r + j] * b[4 * j + k];
            }
            c[4 * r + k] = sum;
        }
    }
    return c;
}
}

// The cache's model of a file, once per build (see build_models)
std::expected<std::shared_ptr<const loaded_model>, std::string> build_models::get(const std::filesystem::path& file,
                                                                                  std::string_view named)
{
    if (const auto known = std::ranges::find(held_, file, &held::file); known != held_.end())
    {
        return known->model;
    }
    auto got = the_model_cache().get(file);
    if (!got)
    {
        return std::unexpected(std::move(got.error()));
    }
    if (got->read && loads_)
    {
        loads_->push_back(load_of(named, *got->model, got->seconds));
    }
    held_.push_back({.file = file, .model = got->model});
    return std::move(got->model);
}

// A model in the scene (a mesh_ref naming a file): its meshes, materials and images added once per build however
// many mesh_refs place it (one acceleration structure each), and every placement of the file's own nodes an instance
// at ref.placement · unit_placement · the node's transform. unit_placement (meshes.ixx) makes any model the same size
// first: its bounding sphere of radius 1, centered over the origin, its lowest point on y = 0, so a scene places it in
// the units of its own world. The model's materials go after those already in the set: its meshes' material_of are
// shifted to match where that offset isn't 0 (meshes_at).
std::expected<void, std::string> scene_translator::add_model(const mesh_ref& ref, const std::filesystem::path& file,
                                                             build_models& models)
{
    const auto got = models.get(file, ref.model);
    if (!got)
    {
        return std::unexpected(got.error());
    }
    const loaded_model& model = **got;
    if (model.placements.empty())
    {
        return std::unexpected(no_triangles(ref.model, model));
    }
    auto source = std::ranges::find(model_sources_, *got, &model_source::model);
    if (source == model_sources_.end())
    {
        const auto offset = static_cast<std::uint32_t>(desc_.meshes.materials.size());
        for (const gpu::mesh_material_desc& m : model.materials)
        {
            add_gltf_material(m, model.textures);
        }
        const auto first_mesh = static_cast<int>(desc_.meshes.meshes.size());
        std::ranges::copy(meshes_at(model, offset), std::back_inserter(desc_.meshes.meshes));
        model_sources_.push_back({.model = *got, .first_mesh = first_mesh});
        source = std::prev(model_sources_.end());
    }
    std::array<double, 12> unit;
    try
    {
        unit = meshes::unit_placement(model.lower, model.upper);
    }
    catch (const std::invalid_argument&)
    {
        return std::unexpected(std::format("{}: its triangles have no size to place", ref.model));
    }
    const std::array<double, 12> placed = compose(ref.placement, unit);
    for (const gpu::mesh_instance_desc& p : model.placements)
    {
        desc_.meshes.instances.push_back(
            {.mesh = source->first_mesh + p.mesh, .object_to_world = compose(placed, p.object_to_world)});
    }
    return {};
}

// Whether a scene argument names a glTF 2.0 file: it ends in .glb or .gltf, in any case (no scene's name has a dot)
export bool is_model_path(std::string_view argument)
{
    const auto ends_with = [argument](std::string_view suffix) {
        return argument.size() >= suffix.size() &&
               std::ranges::equal(argument.substr(argument.size() - suffix.size()), suffix, [](char a, char b) {
                   return std::tolower(static_cast<unsigned char>(a)) == b;
               });
    };
    return ends_with(".glb") || ends_with(".gltf");
}

// A model file's scene is built from the file (build_gpu_scene: model_scene_parts), never by an entry's function: this
// one only says so
gpu_scene_parts model_scene()
{
    throw std::logic_error("a glTF model's scene is built from its file, by build_gpu_scene");
}

// A model file's scene: the model unit-placed (radius 1, standing on y = 0 over the origin) on a matte grey ground 40
// wide, in the photo studio's light (studio's map), seen with a field of 30° from 4.24 away from its middle, the
// center of its bounding sphere, which then fills 2·asin(1/4.24) = 27° of the 30° whatever the model, from 8.1° above
// (0.6 up, 4.2 in front) or higher for a flat model (below). Its middle is (0, h/2, 0) for its height h once placed,
// which only the model knows, so the file is read here (the build's models hold it for the translation). A model
// without a triangle (all of them dropped, or points and lines alone) or without size is refused here, by the file's
// name, before anything divides by its size.
std::expected<gpu_scene_parts, std::string> model_scene_parts(std::string_view file, build_models& models)
{
    const auto got = models.get(std::filesystem::path(file), file);
    if (!got)
    {
        return std::unexpected(got.error());
    }
    const loaded_model& model = **got;
    if (model.placements.empty())
    {
        return std::unexpected(no_triangles(file, model));
    }
    std::array<double, 12> unit;
    try
    {
        unit = meshes::unit_placement(model.lower, model.upper);
    }
    catch (const std::invalid_argument&)
    {
        return std::unexpected(std::format("{}: its triangles have no size to place", file));
    }
    const double height = unit[5] * (model.upper[1] - model.lower[1]); // the scale times the box's height
    gpu_scene_parts parts;
    // The ground 1 mm below the model's lowest point, as busts' is: a model's flat bottom, its own floor or a model
    // flat as a rug would otherwise lie in the ground's plane, the quad hit by the books' test in the world and the
    // triangles by the hardware in their instance's space, and the closest hit would flip from pixel to pixel. Every
    // model is 1 in radius, so one gap fits all: under half a row of the 1080 at the camera's distance (the frame is
    // 2.27 tall there), and far above float's rounding of the hit distances (about 5·10⁻⁷ at 4.24)
    parts.cpu.world.add(std::make_unique<quad>(point3{-20, -0.001, -20}, vec3{0, 0, 40}, vec3{40, 0, 0},
                                               std::make_shared<lambertian>(color{0.5, 0.5, 0.5})));
    const point3 lookat{0, height / 2, 0};
    // A model at least 1 tall, its sphere's radius (its box then at most 1.73 across), is seen from 8.1° above: the
    // bust (1.57) and the vase (1.81) are. A flatter one is seen from higher, at the same distance, 8.1° at h = 1
    // rising linearly to 45° at h = 0: a rug, a coin or a sword lying down would show only its edge from 8.1° (a
    // square rug, 1.41 deep, a sliver 0.2 tall) and shows its face from 45° (1 tall); a car (h ≈ 0.55) is seen from
    // about 25°.
    vec3 from{0, 0.6, 4.2};
    if (height < 1)
    {
        const double low = std::atan2(0.6, 4.2);
        const double elevation = low + (std::numbers::pi / 4 - low) * (1 - height);
        const double distance = std::hypot(0.6, 4.2);
        from = vec3{0, distance * std::sin(elevation), distance * std::cos(elevation)};
    }
    parts.cpu.camera = {.aspect_ratio = 16.0 / 9.0,
                        .image_width = 1920,
                        .samples_per_pixel = 1024,
                        .max_depth = 10,
                        .sampling = sampling_strategy::material,
                        .vfov = 30,
                        .lookfrom = lookat + from,
                        .lookat = lookat};
    parts.environment = {.source = env_source::asset, .asset = "studio_small_09_1k.hdr", .yaw_degrees = 0, .scale = 1};
    parts.meshes = {mesh_ref{.model = file}};
    return parts;
}

// The entry of a model file's scene, outside the list (neither the help nor Tab shows it): a showcase, rendered with
// -p mis and shown through AgX at 0 EV
constexpr gpu_scene_entry model_scene_entry{.name = "model",
                                            .description = "a glTF 2.0 model, FILE.glb or FILE.gltf, in a photo studio",
                                            .role = scene_role::showcase,
                                            .build = &model_scene,
                                            .default_strategy = strategy::mis,
                                            .display = {.op = gpu::tone_map::agx, .exposure = 0}};

export const gpu_scene_entry& model_entry()
{
    return model_scene_entry;
}

// The entry of a scene's name (find_gpu_scene), or the model entry for a model file's path; nullptr for neither
export const gpu_scene_entry* scene_entry(std::string_view scene_or_path)
{
    if (const gpu_scene_entry* entry = find_gpu_scene(scene_or_path))
    {
        return entry;
    }
    return is_model_path(scene_or_path) ? &model_entry() : nullptr;
}

// --- A scene ready for the GPU ---------------------------------------------------------------------------------

// The largest side of a pixel's grid of samples, the renderer's (see gpu::max_sqrt_spp)
export constexpr int max_strata = gpu::max_sqrt_spp;

// What a program may change in a scene's settings: the width, the samples and the strategy, as RayTracing's options
// do, and the view and the strata, as the GPU programs' own options do. What is empty stays the scene's own.
export struct scene_overrides
{
    std::optional<int> image_width;
    std::optional<int> samples_per_pixel;
    std::optional<strategy> sampling; // a strategy of the GPU programs (strategy_options.ixx)
    view_options view; // the image's height and the camera (view_options.ixx)
    // Rounds of strata (see gpu::session): the side of every pixel's grid of samples. Without it the samples are
    // rounded to one grid, as the books' camera does; with it they are exactly the samples asked for, in rounds of the
    // grid.
    std::optional<int> strata;
};

// A scene translated for the GPU, and what it was made from
export struct gpu_build
{
    // The CPU's scene, with the overrides in cpu.camera: its strategy is the books' part of the one asked for, and
    // with strata its samples_per_pixel is the grid's strata² (the samples are `passes`)
    scene cpu;
    camera_frame frame;    // the pixel grid and the lens, as the CPU's camera computes them from cpu.camera
    strategy requested;    // the strategy asked for (-p), or the scene's own
    bool sampling_given;   // requested came from -p
    strategy sampling;     // the strategy rendered with: as the CPU would (effective_sampling: without lamps, material)
    gpu::scene_desc desc;  // the renderer's data, with the lights list for the strategies that aim at it
    std::int64_t passes;   // how many to render: frame.sqrt_spp², or with strata the samples asked for
    std::vector<model_load> models; // the glTF models this build read from their files (not those the cache held)
};

// Builds the scene called `name` with the CPU's code, applies the overrides and translates it, or says why the GPU
// can't render it: the whole message, as the programs print it after "Error: ". `name` may also be a glTF model's file
// (is_model_path): its scene is the model in the studio (model_scene_parts).
export std::expected<gpu_build, std::string> build_gpu_scene(std::string_view name, const scene_overrides& overrides)
{
    const gpu_scene_entry* entry = scene_entry(name);
    if (!entry)
    {
        return std::unexpected(std::format("unknown scene '{}'", name));
    }
    const bool model_file = entry == &model_entry();
    // The camera divides by the width and takes the square root of the samples (see camera), so it needs both
    // positive: the programs' options check that, and so does this function, for callers without options
    if (overrides.image_width.value_or(1) < 1 || overrides.samples_per_pixel.value_or(1) < 1)
    {
        return std::unexpected("the image width and the samples per pixel must be greater than zero");
    }
    // The renderer's limit on the grid (see validate in gpu/renderer.cu)
    if (overrides.strata && !(*overrides.strata >= 1 && *overrides.strata <= max_strata))
    {
        return std::unexpected(std::format("the strata must be from 1 to {}", max_strata));
    }
    // Building a scene draws random numbers from this thread's generator (the first book's cover places its spheres at
    // random, a marble texture shuffles its Perlin tables), so a second build in the same process would make another
    // scene. Starting the generator over gives every build the scene of a fresh process, the scene RayTracing renders.
    seed_random(std::mt19937_64::default_seed);
    std::vector<model_load> models_read;
    build_models models(&models_read);
    gpu_scene_parts parts;
    if (model_file)
    {
        auto made = model_scene_parts(name, models);
        if (!made)
        {
            return std::unexpected(made.error()); // the loader's message: the file, then what is wrong with it
        }
        parts = std::move(*made);
    }
    else
    {
        parts = entry->build();
    }
    scene& s = parts.cpu;
    s.camera.image_width = overrides.image_width.value_or(s.camera.image_width);
    s.camera.samples_per_pixel = overrides.samples_per_pixel.value_or(s.camera.samples_per_pixel);
    const int samples = s.camera.samples_per_pixel;
    if (overrides.strata)
    {
        s.camera.samples_per_pixel = *overrides.strata * *overrides.strata; // the camera's grid: strata × strata
    }
    // The strategy: the one asked for, or the scene's own (its entry's, else its camera's); the camera renders its
    // books' part
    const strategy requested =
        overrides.sampling.value_or(entry->default_strategy.value_or(from_books(s.camera.sampling)));
    s.camera.sampling = entry_of(requested).books.value_or(sampling_strategy::material);
    // After the width: an exact height sets the aspect ratio from it
    if (const auto applied = apply(overrides.view, s.camera); !applied)
    {
        return std::unexpected(applied.error());
    }

    // The CPU's camera computes the pixel grid and the lens from the settings; the GPU renders with its numbers
    const camera_frame frame = camera(s.camera).frame();
    // The strategy the CPU would render with, as in RayTracing: lights and mixture fall back to material in a scene
    // without lamps
    const sampling_strategy sampling = effective_sampling(s.camera.sampling, s.lights);

    auto desc = to_gpu(s, frame, sampling, parts.microfacets, parts.meshes,
                       model_file ? model_paths::as_given : model_paths::assets, models);
    if (!desc)
    {
        return std::unexpected(std::format("the GPU can't render '{}': {}.", name, desc.error()));
    }
    // The lights list, for the strategies that aim at it
    if (sampling == sampling_strategy::lights || sampling == sampling_strategy::mixture)
    {
        auto lights = lights_to_gpu(s.lights);
        if (!lights)
        {
            return std::unexpected(
                std::format("the GPU can't aim at the lights list of '{}': {}. Try -p material or -p uniform.", name,
                            lights.error()));
        }
        desc->lights = std::move(*lights);
    }
    // The environment map the scene names, if any: the quality kernel's sky in place of the camera's
    auto environment = load_environment(parts.environment);
    if (!environment)
    {
        return std::unexpected(environment.error());
    }
    desc->quality.environment = std::move(*environment);
    // The quality kernel's light samples (-p nee, -p mis): a combination of its own, while its camera samples bounces
    // as the material scatters (the books' part of the strategy, above). A scene with no lamp of the light set
    // (gpu/lamps.cuh) and no environment map has nothing to sample: it renders as -p material, with RayTracing's note,
    // in the kernel the scene picks, as the CPU's effective_sampling falls back from lights and mixture without a
    // lights list. (The scene's environment map must be in desc->quality by here, so that a scene lit by the sky alone
    // samples it.)
    strategy rendered = from_books(sampling);
    if ((requested == strategy::nee || requested == strategy::mis) &&
        (gpu::count_lamps(*desc) > 0 || desc->quality.environment))
    {
        desc->quality.combination = requested == strategy::nee ? gpu::light_combination::lights_only
                                                               : gpu::light_combination::power_heuristic;
        rendered = requested;
    }
    const std::int64_t passes = overrides.strata ? samples : std::int64_t{frame.sqrt_spp} * frame.sqrt_spp;
    return gpu_build{.cpu = std::move(s),
                     .frame = frame,
                     .requested = requested,
                     .sampling_given = overrides.sampling.has_value(),
                     .sampling = rendered,
                     .desc = std::move(*desc),
                     .passes = passes,
                     .models = std::move(models_read)};
}

// Whether the quality kernel renders the scene (see the kernels in renderer.cu): asked for, or needed by what the scene
// has (light samples, rough metals, an environment map)
export bool renders_with_quality_kernel(const gpu_build& build, kernel_choice kernel)
{
    const gpu::quality_desc& q = build.desc.quality;
    return kernel == kernel_choice::quality || q.combination != gpu::light_combination::bsdf_only ||
           q.environment.has_value() || std::ranges::any_of(q.microfacets, [](const gpu::microfacet_desc& m) {
               return m.kind != gpu::microfacet_kind::none;
           });
}

// The strategy the integrator's rules check: the one -p gave as given (a books' strategy but the material's is refused,
// even where a scene without lamps would fall back to it), a scene's own as it renders
strategy checked_strategy(const gpu_build& build)
{
    return build.sampling_given ? build.requested : build.sampling;
}

// What a scene with triangle meshes can't be rendered with, as the programs say it after "Error: ", before CUDA or
// OptiX starts: only OptiX traces triangles, in float, with the quality kernel's integrator, which samples the
// materials. So a scene with meshes renders with OptiX whatever the options ask (no --optix needed), and refuses the
// options that pick another renderer or float64, and the books' strategies but the material's. `scene` is the scene's
// name, or the model's file, for the message.
export std::expected<void, std::string> check_meshes(const gpu_build& build, std::string_view scene,
                                                     kernel_choice kernel, bool fp64)
{
    if (build.desc.meshes.instances.empty())
    {
        return {};
    }
    const std::string_view option = fp64                                ? "--fp64"
                                    : kernel == kernel_choice::quality   ? "--quality-kernel"
                                    : kernel == kernel_choice::all_books ? "--all-books-kernel"
                                                                         : "";
    if (!option.empty())
    {
        return std::unexpected(
            std::format("'{}' has triangle meshes, which only OptiX renders (in float): drop {}", scene, option));
    }
    if (!integrator_renders(checked_strategy(build)))
    {
        return std::unexpected(std::format("OptiX samples the materials: use -p {}", integrator_strategies()));
    }
    return {};
}

// What the quality kernel can't render, as the programs say it after "Error: ", before anything is rendered (the
// renderer refuses it too, in its own words): float64, the kernel for all three books, and the books' strategies but
// the material's. With kernel_choice::optix (--optix) the rules of OptiX's programs, which run the quality kernel's
// integrator on every scene: float64, and the books' strategies but the material's (so a scene whose own strategy is
// one of them needs -p). Another option that picks a renderer beside --optix is the programs' to refuse. `scene` is
// the scene's name, for the message.
export std::expected<void, std::string> check_kernel(const gpu_build& build, std::string_view scene,
                                                     kernel_choice kernel, bool fp64)
{
    if (kernel == kernel_choice::optix)
    {
        if (fp64)
        {
            return std::unexpected("--optix renders with OptiX, which computes in float only: drop --fp64");
        }
        if (!integrator_renders(checked_strategy(build)))
        {
            return std::unexpected(std::format("OptiX samples the materials: use -p {}", integrator_strategies()));
        }
        return {};
    }
    if (!renders_with_quality_kernel(build, kernel))
    {
        return {};
    }
    // What the scene has that only the quality kernel renders, if anything
    const gpu::quality_desc& q = build.desc.quality;
    const auto has = [&q](gpu::microfacet_kind kind) {
        return std::ranges::any_of(q.microfacets, [kind](const gpu::microfacet_desc& m) { return m.kind == kind; });
    };
    const std::string_view reason = q.environment                          ? "an environment map"
                                    : has(gpu::microfacet_kind::conductor) ? "a rough metal"
                                    : has(gpu::microfacet_kind::gloss)     ? "a glossy coat"
                                                                           : "";
    // What asked for the quality kernel: the option, a strategy of its own given with -p, or the scene
    const std::string what = kernel == kernel_choice::quality ? std::string("--quality-kernel")
                             : build.sampling_given && !entry_of(build.requested).books
                                 ? std::format("-p {}", strategy_name(build.requested))
                             : reason.empty() ? std::format("'{}'", scene)
                                              : std::format("'{}' ({})", scene, reason);
    if (fp64)
    {
        return std::unexpected(
            std::format("{} renders with the quality kernel, which computes in float only: drop --fp64", what));
    }
    if (kernel == kernel_choice::all_books)
    {
        return std::unexpected(std::format(
            "--all-books-kernel picks the books' kernel for all three books; {} needs the quality kernel", what));
    }
    // Its bounces sample as the material scatters
    if (!integrator_renders(checked_strategy(build)))
    {
        // The strategies it renders: material, and every one of its own (those that aren't the books')
        const std::string strategies = integrator_strategies();
        if (!reason.empty() && kernel != kernel_choice::quality)
        {
            return std::unexpected(std::format("'{}' has {}, which the books' kernels can't render: use -p {}", scene,
                                               reason, strategies));
        }
        return std::unexpected(std::format("the quality kernel samples the materials: use -p {}", strategies));
    }
    return {};
}

// The renderer's pixels (rows from the top, then r, g, b) in the CPU's image class, so the mean radiance and the saved
// files are computed exactly as in RayTracing
export std::expected<image, std::string> to_image(std::span<const double> pixels, int width, int height)
{
    if (pixels.size() != static_cast<std::size_t>(width) * height * 3)
    {
        return std::unexpected(
            std::format("the renderer returned {} numbers for {}×{} pixels", pixels.size(), width, height));
    }
    image img(width, height);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 3; // rows from the top, then r, g, b
            img[x, y] = color{pixels[i], pixels[i + 1], pixels[i + 2]};
        }
    }
    return img;
}

// The display's bytes (4 per pixel, red, green, blue and alpha, rows from the top) as the linear values that the CPU's
// image::save_png turns back into exactly those bytes, so a tone map's PNG is written by the books' own writer: byte b
// becomes ((b + 0.5)/256)². Everything is exact in double: (b + 0.5)/256 has 9 significant bits, its square 18;
// save_png takes the square root, which is correctly rounded and so gives back (b + 0.5)/256 exactly, multiplies it by
// 256 (b + 0.5) and truncates it to b. 255 gives 0.998046875, below the clamp at 0.999, and 0 gives 0.5/256, which
// truncates to 0. Alpha is dropped (the PNG has three channels). Throws std::invalid_argument unless rgba holds
// width × height pixels.
export image bytes_to_image(std::span<const std::uint8_t> rgba, int width, int height)
{
    const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (width < 1 || height < 1 || rgba.size() != 4 * pixels)
    {
        throw std::invalid_argument(std::format("{} bytes aren't the display of {}×{} pixels", rgba.size(), width,
                                                height));
    }
    const auto linear = [](std::uint8_t byte) {
        const double g = (byte + 0.5) / 256;
        return g * g;
    };
    image img(width, height);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 4; // rows from the top, then r, g, b, a
            img[x, y] = color{linear(rgba[i]), linear(rgba[i + 1]), linear(rgba[i + 2])};
        }
    }
    return img;
}
