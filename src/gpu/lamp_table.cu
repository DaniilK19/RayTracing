// The quality kernel's light set, built on the host from the scene (see lamps.cuh): host code only, in a translation
// unit of its own like the other table builders, so that nothing here can change how nvcc compiles a kernel.
#include "lamps.cuh"

#include <cmath>
#include <cstddef>
#include <iterator>
#include <span>
#include <stdexcept>
#include <vector>

#include "quad.cuh"
#include "renderer.h"
#include "vec3.cuh"

namespace gpu
{
namespace
{
// Whether a material makes a lamp of the light set: diffuse_light with one solid color. (A textured lamp's light
// changes over its surface, which one radiance per lamp can't follow.) A lamp of radiance 0 is still one: the set is
// the scene's lamps, as they are.
bool lamp_material(const scene_desc& scene, int index)
{
    if (index < 0 || index >= std::ssize(scene.materials))
    {
        throw std::invalid_argument("a lamp refers to a material that doesn't exist");
    }
    const material_desc& m = scene.materials[static_cast<std::size_t>(index)];
    if (m.kind != material_kind::diffuse_light)
    {
        return false;
    }
    if (m.texture < 0 || m.texture >= std::ssize(scene.textures))
    {
        throw std::invalid_argument("a lamp refers to a texture that doesn't exist");
    }
    return scene.textures[static_cast<std::size_t>(m.texture)].kind == texture_kind::solid;
}

// The color of a lamp_material: the light its front emits, the number the kernels read as the material's albedo
vec3d lamp_radiance(const scene_desc& scene, int index)
{
    const material_desc& m = scene.materials[static_cast<std::size_t>(index)];
    return scene.textures[static_cast<std::size_t>(m.texture)].rgb;
}

// A moving sphere is no lamp of the set: where would a light sample aim, at a time the path doesn't know yet?
bool stands_still(const sphere_desc& s)
{
    return s.displacement.x == 0 && s.displacement.y == 0 && s.displacement.z == 0;
}

bool sphere_lamp(const scene_desc& scene, const sphere_desc& s)
{
    return lamp_material(scene, s.material) && stands_still(s);
}
}

lamp_table build_lamps(const scene_desc& scene, std::span<const int> sphere_ids, std::span<const int> quad_ids)
{
    if (sphere_ids.size() != scene.spheres.size() || quad_ids.size() != scene.quads.size())
    {
        throw std::invalid_argument("the lamps need one global id per sphere and per quad of the scene");
    }
    lamp_table table;
    // Every surface primitive's global id lies in [0, spheres + quads): the spheres' ids come first, then the quads'
    // from quad_base = the number of spheres (see bvh in bvh.cuh)
    table.lamp_of.assign(sphere_ids.size() + quad_ids.size(), -1);
    const auto add = [&table](const emitter& e) {
        const auto id = static_cast<std::size_t>(e.primitive);
        if (e.primitive < 0 || id >= table.lamp_of.size())
        {
            throw std::invalid_argument("a lamp's global id lies outside the scene's primitives");
        }
        table.lamp_of[id] = static_cast<int>(table.lamps.size());
        table.lamps.push_back(e);
    };
    for (std::size_t k = 0; k < scene.spheres.size(); ++k)
    {
        const sphere_desc& s = scene.spheres[k];
        if (!sphere_lamp(scene, s))
        {
            continue;
        }
        // The BVH's sphere: its center and radius rounded to float the same way (bvh_builder::add_sphere)
        add({.corner = to_real<float>(s.center),
             .size = static_cast<float>(s.radius),
             .u = {},
             .kind = static_cast<int>(light_shape::sphere),
             .v = {},
             .primitive = sphere_ids[k],
             .normal = {},
             .unused0 = 0,
             .radiance = to_real<float>(lamp_radiance(scene, s.material)),
             .unused1 = 0});
    }
    for (std::size_t k = 0; k < scene.quads.size(); ++k)
    {
        const quad_desc& q = scene.quads[k];
        if (!lamp_material(scene, q.material))
        {
            continue;
        }
        // The area in double, rounded once; the normal is the BVH quad's own (make_quad), so the side a light sample
        // finds glowing is exactly the side hit_quad calls the front
        const vec3d n = cross(q.u, q.v);
        add({.corner = to_real<float>(q.q),
             .size = static_cast<float>(std::sqrt(dot(n, n))),
             .u = to_real<float>(q.u),
             .kind = static_cast<int>(light_shape::quad),
             .v = to_real<float>(q.v),
             .primitive = quad_ids[k],
             .normal = make_quad<float>(q).normal,
             .unused0 = 0,
             .radiance = to_real<float>(lamp_radiance(scene, q.material)),
             .unused1 = 0});
    }
    return table;
}

int count_lamps(const scene_desc& scene)
{
    int count = 0;
    for (const sphere_desc& s : scene.spheres)
    {
        count += sphere_lamp(scene, s) ? 1 : 0;
    }
    for (const quad_desc& q : scene.quads)
    {
        count += lamp_material(scene, q.material) ? 1 : 0;
    }
    return count;
}
}
