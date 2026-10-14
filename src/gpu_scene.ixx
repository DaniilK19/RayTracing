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
import hittable;
import image;
import instance;
import material;
import perlin;
import quad;
import random;
import scenes;
import sphere;
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

// Translates the CPU's scene into the GPU renderer's tables (renderer.h), or says why the GPU can't render it. It
// reads the CPU's objects where they are, through dynamic_cast and their accessors, and changes nothing.
class scene_translator
{
public:
    scene_translator(const camera_frame& frame, sampling_strategy sampling) { desc_.camera = to_gpu(frame, sampling); }

    // Every shape (sphere, quad) and every volume of smoke of the world, wherever it is in the world's lists, BVHs and
    // instances
    std::expected<void, std::string> add_world(const hittable_list& world)
    {
        return for_each_shape(world, placement{}, [this](const hittable& shape, const placement& where) {
            return add_shape(shape, where);
        });
    }

    gpu::scene_desc finish() && { return std::move(desc_); }

private:
    gpu::scene_desc desc_;
    // Objects can share a material or a texture (they are shared_ptrs), and on the GPU they share it too: each one is
    // stored once, the first time something refers to it, and found again by its address. The maps are only looked
    // up, never iterated, so the tables' order depends only on the order of the world's objects, the same on every
    // run: the order of first use, except that a checker comes after the two textures it refers to.
    std::unordered_map<const material*, int> material_indices_;
    std::unordered_map<const texture*, int> texture_indices_;

    std::expected<void, std::string> add_shape(const hittable& shape, const placement& where);
    std::expected<void, std::string> add_medium(const constant_medium& volume, const placement& where);
    std::expected<int, std::string> material_index(const material* mat);
    std::expected<int, std::string> texture_index(const texture& tex);
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
// (effective_sampling).
std::expected<gpu::scene_desc, std::string> to_gpu(const scene& s, const camera_frame& frame,
                                                   sampling_strategy sampling)
{
    scene_translator translator(frame, sampling);
    if (const auto added = translator.add_world(s.world); !added)
    {
        return std::unexpected(added.error());
    }
    return std::move(translator).finish();
}

// --- A scene ready for the GPU ---------------------------------------------------------------------------------

// What a program may change in a scene's settings: the width, the samples and the strategy, as RayTracing's options
// do, and the view, as the GPU programs' own options do. What is empty stays the scene's own.
export struct scene_overrides
{
    std::optional<int> image_width;
    std::optional<int> samples_per_pixel;
    std::optional<sampling_strategy> sampling;
    view_options view; // the image's height and the camera (view_options.ixx)
};

// A scene translated for the GPU, and what it was made from
export struct gpu_build
{
    scene cpu;                  // the CPU's scene, with the overrides in cpu.camera (its strategy is the one asked for)
    camera_frame frame;         // the pixel grid and the lens, as the CPU's camera computes them from cpu.camera
    sampling_strategy sampling; // the strategy rendered with: the one the CPU would use (effective_sampling)
    gpu::scene_desc desc;       // the renderer's data, with the lights list for the strategies that aim at it
};

// Builds the scene called `name` with the CPU's code, applies the overrides and translates it, or says why the GPU
// can't render it: the whole message, as the programs print it after "Error: ".
export std::expected<gpu_build, std::string> build_gpu_scene(std::string_view name, const scene_overrides& overrides)
{
    const scene_entry* entry = find_scene(name);
    if (!entry)
    {
        return std::unexpected(std::format("unknown scene '{}'", name));
    }
    // The camera divides by the width and takes the square root of the samples (see camera), so it needs both
    // positive: the programs' options check that, and so does this function, for callers without options
    if (overrides.image_width.value_or(1) < 1 || overrides.samples_per_pixel.value_or(1) < 1)
    {
        return std::unexpected("the image width and the samples per pixel must be greater than zero");
    }
    // Building a scene draws random numbers from this thread's generator (the first book's cover places its spheres at
    // random, a marble texture shuffles its Perlin tables), so a second build in the same process would make another
    // scene. Starting the generator over gives every build the scene of a fresh process, the scene RayTracing renders.
    seed_random(std::mt19937_64::default_seed);
    scene s = entry->build();
    s.camera.image_width = overrides.image_width.value_or(s.camera.image_width);
    s.camera.samples_per_pixel = overrides.samples_per_pixel.value_or(s.camera.samples_per_pixel);
    s.camera.sampling = overrides.sampling.value_or(s.camera.sampling);
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

    auto desc = to_gpu(s, frame, sampling);
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
    return gpu_build{.cpu = std::move(s), .frame = frame, .sampling = sampling, .desc = std::move(*desc)};
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
