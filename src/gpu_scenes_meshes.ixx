// Partition gpu_scenes:meshes: the GPU's own scenes with triangle meshes, which only OptiX renders (the CPU ray tracer
// builds no triangle). A scene's meshes ride beside its CPU objects as mesh_refs (gpu_scene_parts::meshes): a mesh made
// in code by a function of this file, or a glTF model under assets/models, wearing its materials, placed by a row-major
// 3×4 matrix. gpu_scene.ixx translates them, each distinct source once (one acceleration structure) and every mesh_ref
// as an instance of it. The ground, the Cornell box's walls and lamp and the camera stay the books' own objects.
//
// The test scenes check the triangles against what the quads and spheres already render: the Cornell box with its
// walls and blocks made of triangles, white furnaces of icospheres, one mesh under four instances against the same
// four baked into the world on the CPU, and glTF's materials on spheres of triangles against the scene materials
// their mapping promises. The showcases place two models of Poly Haven's in a studio, in the sun and in the Cornell
// box. An interface partition that exports only what gpu_scene.ixx's translation needs (to_mesh_desc, gltf_material)
// and what the window asks (has_triangle_meshes): the scenes' functions have module linkage, seen by the module's
// other units and by no importer (as in the other partitions).
module;

// The renderer's types (a mesh, a GGX side entry), before `import std;` (see gpu_scene.ixx)
#include "gpu/renderer.h"

export module gpu_scenes:meshes;

import std;
import :parts;
import camera;
import hittable;
import instance;
import material;
import meshes;
import quad;
import ray;
import scenes;
import texture;
import vec3;

// A mesh of the module meshes as the renderer's: the same five arrays, moved (a generated mesh is made per build and
// dropped; a glTF model's, once per load: gpu_scene.ixx)
export gpu::mesh_desc to_mesh_desc(meshes::mesh m)
{
    return {.positions = std::move(m.positions),
            .normals = std::move(m.normals),
            .uvs = std::move(m.uvs),
            .indices = std::move(m.indices),
            .material_of = std::move(m.material_of)};
}

// glTF's metallic-roughness material for a mesh made in code (pbr-test). The CPU renders no triangle, so this is a CPU
// material in name only, a carrier as the GGX placeholders are: gpu_scene.ixx's translation knows it by its type and
// makes it one of the mesh set's glTF materials, its images with it (a mesh_ref names a CPU material, and glTF's
// factors have no CPU class)
export class gltf_material final : public material
{
public:
    // factors: scene_material -1, its texture indices into `images`
    gltf_material(const gpu::mesh_material_desc& factors,
                  std::vector<std::shared_ptr<const gpu::mesh_texture_desc>> images)
        : factors_(factors), images_(std::move(images))
    {
    }

    // No CPU object wears it
    std::optional<scatter_record> scatter(const ray& /*r_in*/, const hit_record& /*rec*/) const override
    {
        return std::nullopt;
    }

    const gpu::mesh_material_desc& factors() const { return factors_; }
    std::span<const std::shared_ptr<const gpu::mesh_texture_desc>> images() const { return images_; }

private:
    gpu::mesh_material_desc factors_;
    std::vector<std::shared_ptr<const gpu::mesh_texture_desc>> images_;
};

// The helpers are this file's own (an unnamed namespace): the other partitions of gpu_scenes have helpers of the
// same names
namespace
{
using quad_sides = std::array<double, 9>; // q, u, v of a books' quad, as meshes::triangulate_quads takes them
using matrix = std::array<double, 12>;    // row-major 3×4, the translation last

// a·b: b first, then a (affine maps (A, a)(B, b) = (AB, Ab + a))
matrix compose(const matrix& a, const matrix& b)
{
    matrix c{};
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

// Turns about the axes by an angle in degrees, counter-clockwise seen from the axis's positive end (rotate_y's sense
// for y), a stretch along the axes, and a shift
matrix turn_x(double degrees)
{
    const double a = degrees * std::numbers::pi / 180;
    return {1, 0, 0, 0, 0, std::cos(a), -std::sin(a), 0, 0, std::sin(a), std::cos(a), 0};
}
matrix turn_y(double degrees)
{
    const double a = degrees * std::numbers::pi / 180;
    return {std::cos(a), 0, std::sin(a), 0, 0, 1, 0, 0, -std::sin(a), 0, std::cos(a), 0};
}
matrix turn_z(double degrees)
{
    const double a = degrees * std::numbers::pi / 180;
    return {std::cos(a), -std::sin(a), 0, 0, std::sin(a), std::cos(a), 0, 0, 0, 0, 1, 0};
}
matrix stretch(double x, double y, double z)
{
    return {x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0};
}
matrix shift(double x, double y, double z)
{
    return {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z};
}
}

// --- The Cornell box of triangles ------------------------------------------------------------------------------------

namespace
{
// The books' Cornell box (scenes.ixx's add_cornell_walls, cornell_lamp, tall_block, short_block and cornell_camera,
// which that frozen module keeps to itself): a room 555 on a side, open toward the camera, its left wall green and its
// right one red as the camera sees them, the lamp 130 × 105 one unit below the ceiling, facing down
constexpr quad_sides cornell_left{555, 0, 0, 0, 555, 0, 0, 0, 555};
constexpr quad_sides cornell_right{0, 0, 0, 0, 555, 0, 0, 0, 555};
constexpr std::array<quad_sides, 3> cornell_white_walls{{
    {0, 0, 0, 555, 0, 0, 0, 0, 555},         // floor
    {555, 555, 555, -555, 0, 0, 0, 0, -555}, // ceiling
    {0, 0, 555, 555, 0, 0, 0, 555, 0},       // back
}};
constexpr color cornell_red{0.65, 0.05, 0.05};
constexpr color cornell_white{0.73, 0.73, 0.73};
constexpr color cornell_green{0.12, 0.45, 0.15};

std::unique_ptr<quad> cornell_lamp(std::shared_ptr<const material> mat)
{
    return std::make_unique<quad>(point3{343, 554, 332}, vec3{-130, 0, 0}, vec3{0, 0, -105}, std::move(mat));
}

// The books' camera of the Cornell box, its samples and its depth too, so that a render of cornell and one of a scene
// made from it look at the very same pixels
constexpr camera_settings cornell_camera{.aspect_ratio = 1.0,
                                         .image_width = 1080,
                                         .samples_per_pixel = 1000,
                                         .max_depth = 50,
                                         .vfov = 40,
                                         .lookfrom = {278, 278, -800},
                                         .lookat = {278, 278, 0},
                                         .vup = {0, 1, 0}};

// A block's six faces in the world, baked as gpu_scene.ixx bakes the books' translate(rotate_y(box(...))): a turn by
// rotate_y's own cosine and sine (to_world's formula), then the offset. So every face is, number for number, the quad
// the GPU renders in the books' cornell, and its two triangles have that quad's corners.
std::vector<quad_sides> cornell_block_faces(const point3& size, double degrees, const vec3& offset)
{
    const rotate_y turned(box(point3{0, 0, 0}, size, nullptr), degrees);
    const double c = turned.cos_theta();
    const double s = turned.sin_theta();
    const auto turn = [c, s](const vec3& v) { return vec3{c * v.x + s * v.z, v.y, -s * v.x + c * v.z}; };
    std::vector<quad_sides> faces;
    for (const std::unique_ptr<hittable>& side : dynamic_cast<const hittable_list&>(turned.object()).objects())
    {
        const auto& face = dynamic_cast<const quad&>(*side);
        const point3 q = turn(face.q()) + offset;
        const vec3 u = turn(face.u());
        const vec3 v = turn(face.v());
        faces.push_back({q.x, q.y, q.z, u.x, u.y, u.z, v.x, v.y, v.z});
    }
    return faces;
}

// The three meshes of cornell-triangles, one per color (a generated mesh wears its mesh_ref's one material)
gpu::mesh_desc cornell_left_mesh()
{
    return to_mesh_desc(meshes::triangulate_quads(std::span(&cornell_left, 1), {}));
}

gpu::mesh_desc cornell_right_mesh()
{
    return to_mesh_desc(meshes::triangulate_quads(std::span(&cornell_right, 1), {}));
}

// The floor, the ceiling and the back wall, then the tall block (turned 15°) and the short one (turned -18°)
gpu::mesh_desc cornell_white_mesh()
{
    std::vector<quad_sides> faces(cornell_white_walls.begin(), cornell_white_walls.end());
    std::ranges::copy(cornell_block_faces(point3{165, 330, 165}, 15, vec3{265, 0, 295}), std::back_inserter(faces));
    std::ranges::copy(cornell_block_faces(point3{165, 165, 165}, -18, vec3{130, 0, 65}), std::back_inserter(faces));
    return to_mesh_desc(meshes::triangulate_quads(faces, {}));
}
}

// The books' Cornell box with its five walls and both blocks made of 34 triangles, two per quad, wearing the books'
// materials; the lamp stays a quad, in the light set. The triangles have the quads' corners and their texture
// coordinates, so the image must be cornell's within the noise (compare.ps1 against cornell's references): the scene
// that checks the triangles' hits, their normals, the origins moved off the faces and the IAS against the books' own
// quads, in a closed room where every path bounces many times.
gpu_scene_parts cornell_triangles()
{
    hittable_list world;
    world.add(cornell_lamp(std::make_shared<diffuse_light>(color{15, 15, 15})));
    hittable_list lights;
    lights.add(cornell_lamp(nullptr));
    gpu_scene_parts parts{.cpu = scene{std::move(world), cornell_camera, std::move(lights)}};
    parts.meshes = {
        mesh_ref{.generated = &cornell_left_mesh, .material = std::make_shared<lambertian>(cornell_green)},
        mesh_ref{.generated = &cornell_right_mesh, .material = std::make_shared<lambertian>(cornell_red)},
        mesh_ref{.generated = &cornell_white_mesh, .material = std::make_shared<lambertian>(cornell_white)},
    };
    return parts;
}

// --- White furnaces of triangles -------------------------------------------------------------------------------------

namespace
{
// furnace's camera (gpu_scenes.ixx) and its white sky, with a deeper path: mesh-furnace's two spheres face each other
// across a gap of 0.4, and from the middle of the gap one of them covers half the other's cosine-weighted directions
// (sin²45.6° = 0.51), so a path may bounce between them. furnace's 10 rays would end one in about 2^-9 of the paths
// that reach the gap there with no light (a pixel then below 1); 50 make that about 2^-49, never at 480×270×64.
camera_settings white_furnace_camera()
{
    return {.aspect_ratio = 16.0 / 9.0,
            .image_width = 480,
            .samples_per_pixel = 64,
            .max_depth = 50,
            .sampling = sampling_strategy::material,
            .vfov = 30,
            .lookfrom = {0, 0, 6},
            .lookat = {0, 0, 0},
            .background = {.bottom = {1, 1, 1}, .top = {1, 1, 1}}};
}

gpu::mesh_desc flat_icosphere()
{
    return to_mesh_desc(meshes::icosphere(3, false));
}

gpu::mesh_desc smooth_icosphere()
{
    return to_mesh_desc(meshes::icosphere(2, true));
}
}

// furnace with two flat icospheres of 1280 triangles each, white (albedo 1), at x = -1.2 and 1.2, the right one placed
// with the scale (-1, 1, 1): one mesh, two instances, one of them a mirror. Under -p material every pixel must be
// exactly 1.0, as in furnace: the sky is 1 everywhere, a bounce multiplies by 1 exactly, flat normals never send a ray
// through the face it leaves, and a ray never meets the face it leaves again (its origin moved off the face). A ray
// that missed a face where two triangles meet, or hit one twice, would show as a pixel off 1.
gpu_scene_parts mesh_furnace()
{
    gpu_scene_parts parts{.cpu = scene{hittable_list{}, white_furnace_camera()}};
    const auto white = std::make_shared<lambertian>(color{1, 1, 1});
    parts.meshes = {
        mesh_ref{.generated = &flat_icosphere, .material = white, .placement = shift(-1.2, 0, 0)},
        mesh_ref{.generated = &flat_icosphere,
                 .material = white,
                 .placement = compose(shift(1.2, 0, 0), stretch(-1, 1, 1))},
    };
    return parts;
}

// One smooth icosphere of 320 triangles, the same white, under the same sky. With smooth normals a bounce drawn around
// the shading normal may point through the face it leaves, which the through-rule absorbs (optix_tracer.cuh): the
// image's mean below 1 is the light that rule costs a coarse mesh (an INFO of Step 91, never a pass criterion).
gpu_scene_parts mesh_furnace_smooth()
{
    gpu_scene_parts parts{.cpu = scene{hittable_list{}, white_furnace_camera()}};
    parts.meshes = {mesh_ref{.generated = &smooth_icosphere, .material = std::make_shared<lambertian>(color{1, 1, 1})}};
    return parts;
}

// --- One mesh, four instances ----------------------------------------------------------------------------------------

namespace
{
// The mesh: a smooth level-1 icosphere stretched to the ellipsoid of half-axes 1, 0.5 and 0.8, so that a turn shows
// (its normals carried by the stretch's inverse transpose, at the length that gives them: bake)
meshes::mesh ellipsoid()
{
    return meshes::bake(meshes::icosphere(1, true), stretch(1, 0.5, 0.8));
}

gpu::mesh_desc ellipsoid_mesh()
{
    return to_mesh_desc(ellipsoid());
}

// Its four placements, 2.4 apart along x: two turns (about y, then a tilt about x and z), a stretch that isn't
// uniform, and a mirror of a tilted copy (the scale -1 along x after a turn about z: an ellipsoid is its own mirror
// image along its axes, a tilted one isn't). Their M⁻ᵀ carries the shading normals: a stretch changes their
// directions, a mirror their side.
std::array<matrix, 4> ellipsoid_placements()
{
    return {compose(shift(-3.6, 0, 0), turn_y(35)), compose(shift(-1.2, 0, 0), compose(turn_z(40), turn_x(-30))),
            compose(shift(1.2, 0, 0), stretch(0.8, 1.6, 1.1)),
            compose(shift(3.6, 0, 0), compose(stretch(-1, 1, 1), turn_z(30)))};
}

// instances-baked's meshes: the mesh moved into the world by each placement on the CPU (meshes::bake, which also
// swaps the corners under the mirror), each under an identity instance. A function per placement, as a mesh_ref holds
// a function without captures.
template <std::size_t K>
gpu::mesh_desc baked_ellipsoid()
{
    return to_mesh_desc(meshes::bake(ellipsoid(), ellipsoid_placements()[K]));
}

// The studio and its camera (gpu_scenes:environments' studio): softboxes give every curve of a glossy surface a
// highlight
constexpr environment_choice studio_light{
    .source = env_source::asset, .asset = "studio_small_09_1k.hdr", .yaw_degrees = 0, .scale = 1};

camera_settings instances_camera()
{
    return {.aspect_ratio = 16.0 / 9.0,
            .image_width = 480,
            .samples_per_pixel = 1024,
            .max_depth = 10,
            .sampling = sampling_strategy::material,
            .vfov = 30,
            .lookfrom = {0, 2, 10.5},
            .lookat = {0, 0, 0}};
}

// The glossy coat the four wear: a pale matte color under a GGX coat of r = 0.3, as a GGX material is made: a
// lambertian placeholder completed by its side entry
gpu_scene_parts glossy_parts()
{
    gpu_scene_parts parts{.cpu = scene{hittable_list{}, instances_camera()}, .environment = studio_light};
    return parts;
}

std::shared_ptr<lambertian> add_glossy_coat(gpu_scene_parts& parts)
{
    auto placeholder = std::make_shared<lambertian>(color{0.7, 0.7, 0.75});
    parts.microfacets.emplace_back(placeholder.get(),
                                   gpu::microfacet_desc{.kind = gpu::microfacet_kind::gloss, .roughness = 0.3});
    return placeholder;
}
}

// One ellipsoid mesh under four instances of the IAS, in the studio's light: the transforms on the GPU (OptiX's of
// the rays, and the instance's own of the hit point and normals in triangle_surface), the mirror's sides included.
// Compared with instances-baked (compare.ps1), which differs in nothing else.
gpu_scene_parts instances()
{
    gpu_scene_parts parts = glossy_parts();
    const auto coat = add_glossy_coat(parts);
    for (const matrix& placement : ellipsoid_placements())
    {
        parts.meshes.push_back({.generated = &ellipsoid_mesh, .material = coat, .placement = placement});
    }
    return parts;
}

// The reference of instances: the four ellipsoids baked into the world on the CPU, in double, each a mesh of its own
// under an identity instance
gpu_scene_parts instances_baked()
{
    gpu_scene_parts parts = glossy_parts();
    const auto coat = add_glossy_coat(parts);
    parts.meshes = {
        mesh_ref{.generated = &baked_ellipsoid<0>, .material = coat},
        mesh_ref{.generated = &baked_ellipsoid<1>, .material = coat},
        mesh_ref{.generated = &baked_ellipsoid<2>, .material = coat},
        mesh_ref{.generated = &baked_ellipsoid<3>, .material = coat},
    };
    return parts;
}

// --- glTF's materials, against the scene materials they map to -------------------------------------------------------

namespace
{
// The spheres: smooth level-4 icospheres (5120 triangles), one of them with texture coordinates for the textured
// material: the sphere's longitude and latitude, u = 1/2 + atan2(z, x)/2π and v = acos(y)/π (the triangles along the
// seam interpolate across it, which a texture of one color can't show)
gpu::mesh_desc pbr_sphere()
{
    return to_mesh_desc(meshes::icosphere(4, true));
}

gpu::mesh_desc pbr_textured_sphere()
{
    meshes::mesh m = meshes::icosphere(4, true);
    m.uvs.reserve(m.positions.size() / 3 * 2);
    for (std::size_t i = 0; i < m.positions.size(); i += 3)
    {
        const double x = m.positions[i];
        const double y = m.positions[i + 1];
        const double z = m.positions[i + 2];
        m.uvs.push_back(static_cast<float>(0.5 + std::atan2(z, x) / (2 * std::numbers::pi)));
        m.uvs.push_back(static_cast<float>(std::acos(std::clamp(y, -1.0, 1.0)) / std::numbers::pi));
    }
    return to_mesh_desc(std::move(m));
}

// The four spheres of factors alone, left to right: two metals, gold at r = 0.1 and silver at r = 0.5 (the F0 of
// gpu_scenes:environments' gold and silver), and two dielectrics, red at r = 0.3 and blue at r = 0.8. Metallic is 1
// or 0, so the lobe is certain and the two scenes differ only in the random numbers the glTF lobe takes.
struct pbr_factors
{
    color base;
    double metallic;
    double roughness;
};
constexpr std::array<pbr_factors, 4> pbr_spheres{{{color{1.000, 0.766, 0.336}, 1, 0.1},
                                                  {color{0.972, 0.960, 0.915}, 1, 0.5},
                                                  {color{0.7, 0.08, 0.06}, 0, 0.3},
                                                  {color{0.1, 0.25, 0.7}, 0, 0.8}}};
// The fifth: a metal of r = 0.3 whose base color is a 2×2 texture of one sRGB color (a copper's), its factor 1
constexpr std::array<std::uint8_t, 4> pbr_texel{232, 160, 110, 255};
constexpr double pbr_textured_roughness = 0.3;

// A byte of sRGB as the linear value it encodes: IEC 61966-2-1's curve, the texture hardware's decoding
double srgb_to_linear(std::uint8_t byte)
{
    const double c = byte / 255.0;
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}

// Sphere i, radius 1, 2.2 apart along x
matrix pbr_place(std::size_t i)
{
    return shift(-4.4 + 2.2 * static_cast<double>(i), 0, 0);
}

camera_settings pbr_camera()
{
    return {.aspect_ratio = 16.0 / 9.0,
            .image_width = 480,
            .samples_per_pixel = 1024,
            .max_depth = 10,
            .sampling = sampling_strategy::material,
            .vfov = 30,
            .lookfrom = {0, 2.5, 13},
            .lookat = {0, 0, 0}};
}
}

// Five spheres of triangles wearing glTF's metallic-roughness materials, in the studio's light: four of factors alone
// and one whose base color is a texture, read by the texture hardware (sRGB decoded, bilinear). Each hit writes the
// material glTF's mapping makes of them (triangles.cuh) into its pixel's scratch slot, which the integrator reads as
// any material. Compared with pbr-reference (compare.ps1): the scratch path must render the table path's image.
gpu_scene_parts pbr_test()
{
    gpu_scene_parts parts{.cpu = scene{hittable_list{}, pbr_camera()}, .environment = studio_light};
    for (std::size_t i = 0; i < pbr_spheres.size(); ++i)
    {
        const pbr_factors& f = pbr_spheres[i];
        const gpu::mesh_material_desc factors{.scene_material = -1,
                                              .base_color = {f.base.x, f.base.y, f.base.z},
                                              .metallic = f.metallic,
                                              .roughness = f.roughness};
        parts.meshes.push_back({.generated = &pbr_sphere,
                                .material = std::make_shared<gltf_material>(
                                    factors, std::vector<std::shared_ptr<const gpu::mesh_texture_desc>>{}),
                                .placement = pbr_place(i)});
    }
    gpu::mesh_texture_desc image{.width = 2, .height = 2, .srgb = true};
    for (int texel = 0; texel < 4; ++texel)
    {
        image.rgba.insert(image.rgba.end(), pbr_texel.begin(), pbr_texel.end());
    }
    const gpu::mesh_material_desc textured{.scene_material = -1,
                                           .base_color = {1, 1, 1},
                                           .metallic = 1,
                                           .roughness = pbr_textured_roughness,
                                           .base_color_texture = 0};
    parts.meshes.push_back(
        {.generated = &pbr_textured_sphere,
         .material = std::make_shared<gltf_material>(
             textured, std::vector<std::shared_ptr<const gpu::mesh_texture_desc>>{
                           std::make_shared<const gpu::mesh_texture_desc>(std::move(image))}),
         .placement = pbr_place(pbr_spheres.size())});
    return parts;
}

// pbr-test's spheres wearing the scene materials glTF's mapping promises, as milestone 4 makes them: a metal of fuzz 0
// whose albedo is F0, completed by a GGX conductor of the same roughness; a lambertian of the base color under the
// glossy coat (r at least 0.0317, the coat's floor); the textured metal with the texel's linear color as F0
gpu_scene_parts pbr_reference()
{
    gpu_scene_parts parts{.cpu = scene{hittable_list{}, pbr_camera()}, .environment = studio_light};
    const auto conductor = [&parts](const color& f0, double roughness) {
        auto placeholder = std::make_shared<metal>(f0, 0.0);
        parts.microfacets.emplace_back(
            placeholder.get(), gpu::microfacet_desc{.kind = gpu::microfacet_kind::conductor, .roughness = roughness});
        return placeholder;
    };
    const auto coat = [&parts](const color& base, double roughness) {
        auto placeholder = std::make_shared<lambertian>(base);
        parts.microfacets.emplace_back(placeholder.get(),
                                       gpu::microfacet_desc{.kind = gpu::microfacet_kind::gloss,
                                                            .roughness = std::max(roughness, 0.0317)});
        return placeholder;
    };
    for (std::size_t i = 0; i < pbr_spheres.size(); ++i)
    {
        const pbr_factors& f = pbr_spheres[i];
        std::shared_ptr<material> worn;
        if (f.metallic == 1)
        {
            worn = conductor(f.base, f.roughness);
        }
        else
        {
            worn = coat(f.base, f.roughness);
        }
        parts.meshes.push_back({.generated = &pbr_sphere, .material = std::move(worn), .placement = pbr_place(i)});
    }
    const color texel{srgb_to_linear(pbr_texel[0]), srgb_to_linear(pbr_texel[1]), srgb_to_linear(pbr_texel[2])};
    parts.meshes.push_back({.generated = &pbr_textured_sphere,
                            .material = conductor(texel, pbr_textured_roughness),
                            .placement = pbr_place(pbr_spheres.size())});
    return parts;
}

// --- The showcases: two models of Poly Haven's -----------------------------------------------------------------------

namespace
{
// The models are Poly Haven's (polyhaven.com/a/marble_bust_01, polyhaven.com/a/brass_vase_01), CC0, which allows
// committing them: their 1k glTF packages exactly as published, under assets/models, the normal maps the renderer
// ignores included, so that every file can be checked against the download (MeshTests holds each one's SHA-256). A
// marble bust, a textured dielectric (metallic 0 and a roughness map), and a brass vase, a textured metal, which shows
// a base color on a metal. A scene places a model after unit_placement has made it 1 in radius and stood it on y = 0
// over the origin (gpu_scene.ixx): the bust is then 1.57 tall and 0.83 wide, the vase 1.81 tall and 0.6 wide, each
// facing +z ("the front of a glTF asset faces +Z", glTF 2.0, 3.4).
constexpr std::string_view marble_bust = "marble_bust_01/marble_bust_01_1k.gltf";
constexpr std::string_view brass_vase = "brass_vase_01/brass_vase_01_1k.gltf";
// The bust's height once unit-placed: twice its height over the diagonal of its box, from the .gltf's POSITION min and
// max and its node's translation, 2 · 0.514943 / 0.654861
constexpr double bust_unit_height = 1.57268;

// A ground of side `size`, centered under the origin at height y, facing up (u × v = +y), as gpu_scenes:environments'
std::unique_ptr<quad> ground(double size, double y, std::shared_ptr<const material> mat)
{
    return std::make_unique<quad>(point3{-size / 2, y, -size / 2}, vec3{0, 0, size}, vec3{size, 0, 0}, std::move(mat));
}

// The showcases' camera: 16:9, 1920 wide, 1024 samples, at most 10 rays a path, a field of 30° (the books' part of the
// strategy only: the list gives them -p mis)
camera_settings showcase_camera(const point3& lookfrom, const point3& lookat)
{
    return {.aspect_ratio = 16.0 / 9.0,
            .image_width = 1920,
            .samples_per_pixel = 1024,
            .max_depth = 10,
            .sampling = sampling_strategy::material,
            .vfov = 30,
            .lookfrom = lookfrom,
            .lookat = lookat};
}

// sunlit's sky: the sun 48° above the horizon in a partly cloudy sky, the ground replaced by sky
constexpr environment_choice sunlight{.source = env_source::asset,
                                      .asset = "kloofendal_48d_partly_cloudy_puresky_1k.hdr",
                                      .yaw_degrees = 0,
                                      .scale = 1};

// A books' quad of the Cornell box's data above
std::unique_ptr<quad> cornell_quad(const quad_sides& q, std::shared_ptr<const material> mat)
{
    return std::make_unique<quad>(point3{q[0], q[1], q[2]}, vec3{q[3], q[4], q[5]}, vec3{q[6], q[7], q[8]},
                                  std::move(mat));
}
}

// The displays the showcases open with (their entries in gpu_scenes.ixx): AgX, as for studio and sunlit
inline constexpr gpu::display_options still_life_display{.op = gpu::tone_map::agx, .exposure = 0};
inline constexpr gpu::display_options busts_display{.op = gpu::tone_map::agx, .exposure = -0.5};
inline constexpr gpu::display_options cornell_bust_display{.op = gpu::tone_map::agx, .exposure = 0};

// The marble bust and the brass vase side by side on a matte grey ground in the photo studio's light, both facing the
// camera: a dielectric whose roughness varies texel by texel beside a metal whose color does. Two models, so the
// vase's materials follow the bust's in the mesh set (gpu_scene.ixx's meshes_at).
gpu_scene_parts still_life()
{
    gpu_scene_parts parts{.cpu = scene{hittable_list{}, showcase_camera({0, 1.5, 4.6}, {0, 1.0, 0})},
                          .environment = studio_light};
    parts.cpu.world.add(ground(40, 0, std::make_shared<lambertian>(color{0.5, 0.5, 0.5})));
    parts.meshes = {
        mesh_ref{.model = marble_bust, .placement = shift(-0.55, 0, 0)},
        mesh_ref{.model = brass_vase, .placement = shift(0.6, 0, 0)},
    };
    return parts;
}

// Three busts in the sun on sunlit's checkered ground, the middle one facing the camera and the outer two turned by 40°
// toward it: one model, so one acceleration structure under three instances. The ground is sunlit's, 1 mm below y = 0,
// where the busts stand (see sunlit: the solid checker would speckle a floor at y = 0).
gpu_scene_parts busts()
{
    gpu_scene_parts parts{.cpu = scene{hittable_list{}, showcase_camera({0, 1.5, 5.2}, {0, 0.8, 0})},
                          .environment = sunlight};
    const auto checker = std::make_shared<checker_texture>(0.5, color{0.2, 0.2, 0.2}, color{0.7, 0.7, 0.7});
    parts.cpu.world.add(ground(200, -0.001, std::make_shared<lambertian>(checker)));
    constexpr std::array<std::pair<double, double>, 3> placed{{{-1.4, 40}, {0, 0}, {1.4, -40}}}; // x and the turn
    for (const auto& [x, degrees] : placed)
    {
        parts.meshes.push_back({.model = marble_bust, .placement = compose(shift(x, 0, 0), turn_y(degrees))});
    }
    return parts;
}

// The books' Cornell box with the bust in place of the tall block: 330 units tall like the block, standing where the
// block's middle stands (the block's own turn of 15° applied to its center (82.5, 82.5), then its offset), turned
// 15° from facing the camera (180° + 15°: the bust faces +z, the camera looks along +z), as the block is turned. The
// short block, the walls and the lamp are the books' quads (the lamp in the light set).
gpu_scene_parts cornell_bust()
{
    hittable_list world;
    const auto white = std::make_shared<lambertian>(cornell_white);
    world.add(cornell_quad(cornell_left, std::make_shared<lambertian>(cornell_green)));
    world.add(cornell_quad(cornell_right, std::make_shared<lambertian>(cornell_red)));
    for (const quad_sides& wall : cornell_white_walls)
    {
        world.add(cornell_quad(wall, white));
    }
    world.add(cornell_lamp(std::make_shared<diffuse_light>(color{15, 15, 15})));
    std::unique_ptr<hittable> block = box(point3{0, 0, 0}, point3{165, 165, 165}, white);
    block = std::make_unique<rotate_y>(std::move(block), -18);
    world.add(std::make_unique<translate>(std::move(block), vec3{130, 0, 65}));
    hittable_list lights;
    lights.add(cornell_lamp(nullptr));
    gpu_scene_parts parts{.cpu = scene{std::move(world), cornell_camera, std::move(lights)}};
    const rotate_y tall(box(point3{0, 0, 0}, point3{165, 330, 165}, nullptr), 15);
    const double c = tall.cos_theta();
    const double s = tall.sin_theta();
    const double size = 330 / bust_unit_height;
    parts.meshes = {mesh_ref{.model = marble_bust,
                             .placement = compose(shift(c * 82.5 + s * 82.5 + 265, 0, -s * 82.5 + c * 82.5 + 295),
                                                  compose(turn_y(195), stretch(size, size, size)))}};
    return parts;
}

// Whether a scene of the list holds triangle meshes, which only OptiX renders, without building it (and so without
// reading a model's file): the window's Tab leaves these out of a session that doesn't use OptiX, so that Tab without
// OptiX goes through the same scenes as before them
export bool has_triangle_meshes(const gpu_scene_entry& entry)
{
    constexpr std::array scenes{&cornell_triangles, &mesh_furnace,  &mesh_furnace_smooth, &instances,
                                &instances_baked,   &pbr_test,      &pbr_reference,       &still_life,
                                &busts,             &cornell_bust};
    return std::ranges::contains(scenes, entry.build);
}
