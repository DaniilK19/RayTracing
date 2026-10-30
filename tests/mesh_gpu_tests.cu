// Suite optix-triangles (Step 91): triangle meshes on the GPU, tested from their specification alone. The
// implementation's files were not read (meshes.cu, the bodies of triangles.cuh, accel.cpp's meshes region,
// raygen_pick.cuh, gpu_scene's mesh translation, gpu_scenes_meshes.ixx), and every reference here is the test's
// own, in double:
//   offset-origin   offset_origin on 10^6 (point, normal) pairs at scales 10^-3 to 10^4, in a plain CUDA kernel: the
//                   point moved strictly onto the normal's side of the plane through p (evaluated in double), each
//                   coordinate by at most 2^-15·|p|∞ + 2^-15 (the offset's bound); planted: the same check on
//                   offset_origin(p, -n), whose every point it must refuse;
//   surface         triangle_surface at 10^4 random hits (eight instances: mirrors, non-uniform scales, 10^-3 and 10^2
//                   scales; smooth, flat and wild normals) on tables laid out by the test as mesh_tables.h describes
//                   them, against the surface rules recomputed in double: p within 10^-5 of the coordinates' size, the
//                   face normal and the shading normal within 10^-5, the side, u and v, the material, its texture and
//                   the id; each rule of the shading normal on a case of its own; the outside of a closed mesh under
//                   mirroring instances; absorber_surface's fields;
//   baked-twin      a mesh baked by module meshes (world vertices, corners 1 and 2 swapped under a mirror) against the
//                   same mesh under the instance: the same point, face side and front, and the same shading normal;
//   brute-force     OptiX: 10^5 rays at a level-3 icosphere and 10^5 at a 24×12 torus (each under an instance, the
//                   torus's mirrored and stretched) against every triangle in double: the same triangle for every ray
//                   farther than 10^-4 (barycentrically) from an edge, t within 10^-6 relative, barycentrics within
//                   10^-5, the id triangle_base + 1 + primitive; the near-edge rays INFO;
//   coplanar        OptiX: rays leaving random triangles of a tilted plane that has a parallel neighbour at a gap of
//                   4·2^-15·|p|∞, at |p|∞ = 1, 8 and 555, 0.01° to 90° off the plane, through optix_tracer (its offset
//                   origin, hardware tmin 0): no hit of the plane left, no miss of the neighbour, in both directions;
//                   the through-rule's absorber for rays going into the face; planted: the face's side reversed, so
//                   that the rays start behind it and must hit it;
//   surface-upload  OptiX: triangle_surface on the tables meshes.cu uploads (the instances' ids, parts, transforms and
//                   inverse transposes, the materials) against the same double reference;
//   world-ray       P7: the world ray __intersection__probe reads under an identity instance (cornell, the IAS root
//                   forced) is the ray traced, bit for bit, -0 included;
//   pick            session::pick_distance on a triangle mesh's front and back and on a books' sphere: t within 10^-6
//                   (relative) of the host's, ∞ on a miss;
//   closed-block    gpu::render (__raygen__render's loop, the session's upload) from a camera inside cornell-triangles'
//                   tall block, a closed box of 12 triangles, under -p material and -p mis: every path traces its
//                   max_depth rays, since none can leave the block, and the image is black.
// The OptiX checks reach the GPU through the fixture's runtime (optix_fixture.h): an upload of a scene_desc the test
// makes (a gpu::mesh_set of meshes from module meshes, through tests/mesh_services.h), and launch_test of the check
// module's __raygen__probe_mesh. Its creation and first compiles are left out of the 10 s budget. closed-block renders
// through the program's own raygen instead: the probe traces one ray per launch, where the paths of a raygen's loop can
// go wrong in ways one ray can't show (decision 9).
//
// The probe of meshes, __raygen__probe_mesh (tests/optix_check_programs.cu, the probes' block): one sample per launch
// index k = y·width + x of a launch of n = width·height, 20 words in at params.test.values[20k] (floats; an int as its
// float's bits), 40 out at values[20n + 40k]:
//   in   0-2 the ray's origin, 3-5 its direction, 6 its time, 7 spawn.from, 8 spawn.inward (0 or 1), 9-11 memo.facing,
//        12 the mode: 0 traces the ray with optix_tracer::closest (sample_generator(test.seed, k, 0)); 1 traces
//        nothing and asks optix_tracer::surface_of for the triangle hit of 13 the instance, 14 the primitive, 15-16 b1
//        and b2, 17 memo.lobe, in scratch slot scratch_base + k;
//   out  0 the hit's index, 1 its t, 2 entering (0 or 1); the side record: 3 instance, 4 primitive, 5-6 b1, b2,
//        7 lobe; mode 1: 8-10 rec.p, 11-13 rec.normal, 14 front_face, 15-17 outward, 18-20 texture_point, 21 u, 22 v,
//        23 material, 24 texture_index, 25 from, 26-28 memo.facing, 29 = 1; then what the integrator reads at
//        s.material: 30 the material's kind, 31-33 albedo, 34 fuzz, 35 refraction index, 36 its side entry's kind
//        (-1: no side entries), 37 alpha, 38 roughness. (39: the CUDA twin's wears_gltf.)
// The surface checks' plain kernel writes the same 40 words from the same 20 (mode 1 only), so one decoder reads both.
// Under P7's probe_intersections the probe intersection program keeps 16 words of params.test.counts per sample (its
// own block of optix_check_programs.cu says which).
//
// Decisions where the specification leaves room:
//   1. "Farther than 10^-4 (relative) from an edge" is barycentric: a ray is near an edge when the reference finds any
//      triangle crossed at t > t_min, in front of the nearest hit (or anywhere, for a miss), whose smallest barycentric
//      coordinate is within 10^-4 of 0 on either side. t's 10^-6 is relative to t, and for grazing hits (|cos θ| < 0.1
//      between the ray and the face) 10^-6·0.1/|cos θ|: the hardware's t loses digits as the ray turns parallel to the
//      face, its barycentrics don't. The brute force runs in double on the GPU (10^5 rays × 1280 triangles would take a
//      CPU second).
//   2. |Δp| ≤ 10^-5·scale with scale the hit triangle's largest world coordinate; normals within 10^-5 in each
//      component; a hit within 10^-4 of grazing its face (|cos θ| < 10^-4) has no decided side, and a shading normal
//      whose rules decide within 10^-4 of a threshold, or which comes out of a sum that cancels to under 1% of its
//      terms, isn't compared: both counted INFO.
//   3. u and v: the hand-built tables give their materials a texture that reads them (material_texture::needs_uv), so
//      they're interpolated whichever rule triangle_surface follows; on the uploaded tables (device_scene folds every
//      texture to a color) (0, 0) passes too.
//   4. The baked twin: bake leaves each vertex normal at the length M⁻ᵀ gives it, so the interpolated shading normal
//      equals the instanced one, M⁻ᵀ·Σbᵢnᵢ, under every transform, a non-uniform scale and a mirror included (made
//      unit per vertex, they would differ there: the first run of Step 91's protocols found that).
//   5. The coplanar neighbour: one mesh holds both planes (10×10 cells each, finer at the centre), tilted (normal
//      (0.30, 0.90, 0.25)), its centre near (0.7, 1, 0.02)·|p|∞ so that a coordinate crosses 0 (Wächter and Binder's
//      n/65536 branch, offset_origin's until decision 9). Rays start in the central 0.2·|p|∞ square at points
//      interpolated from the float vertices, rounded and moved by up to 2 ulps per coordinate (triangle_surface's own
//      rounding); the elevation is log-uniform in [0.01°, 90°]. 10^5 rays per scale, half each way. The planted
//      fault: 10^4 rays at |p|∞ = 8 with memo.facing reversed; at least 99% must hit the plane they leave.
//   6. P7: 4096 rays from random points inside the box toward its back half (the box is open on the camera's side, so
//      a ray toward it could meet no box), 64 from its centre and corners with ±0 coordinates in the origin and the
//      direction, and 8 with subnormal direction coordinates (INFO: a driver that flushes them to 0 reconstructs no
//      other ray). Every ray from inside must reach the probe.
//   7. The pick: directions given as unit vectors (session::pick_distance's distance along a unit direction); the
//      sphere is a books' primitive, so the pick sees the books' GAS too.
//   8. The brute force's t and barycentrics are held to float precision as well as to 10^-6 and 10^-5, whichever is
//      larger. The hardware's watertight triangle test works on the corners' coordinates relative to the ray's origin,
//      in float (in the instance's space, which OptiX's float inverse transform moves the ray into), so the point a hit
//      names can be some float steps of those coordinates away from the reference's: δ = 2^-16·(|o|∞ + t·|d|∞), 256
//      steps, is δ/(t·|d|·|cos θ|) of t and δ over the triangle's smallest altitude in barycentric units. A wrong
//      corner, triangle or convention is off by tenths. (The first run, held to 10^-5 alone, found barycentrics up to
//      2.8·10^-4 off, rays coming from 10 units away, while every ray met the reference's triangle and t agreed to
//      9·10^-7; in float steps over the altitude the differences reached 23 under the identity (the third mesh), 38
//      under a turning, scaling instance and 67 on the torus's thin triangles under a mirroring one.)
//   9. The closed block: the camera at (355, 278, 278), 7 units inside the tall block's face, looking along +x; 8×8
//      pixels, 16 samples each, max_depth 50. It was added for two defects the probe's single rays passed: nvcc hoisted
//      the barycentrics' read out of __raygen__render's loop, before any traversal (optix_tracer.cuh's decode_hit), and
//      offset_origin's step off the floor (y = 0, x and z in the hundreds) fell inside the triangle test's rounding
//      (triangles.cuh), so paths met the floor again from below. Paths then traced 3.5 and 46.5 of their 50 rays, and
//      the first found the lamp. A path may still slip through an edge where the hit point rounds outside its face,
//      which the books' quads allow too: the rays may fall short of the full count by 1% (a path ending at its first
//      bounce costs 49 of the 51200 rays of a strategy), and one value of the 192 may be above 0.
// Seeds: 9101-9113.
//
// Step criteria: $GT optix-triangles --require-optix exits 0 (≈ 3 s, the check module's cold compile apart).
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "camera.cuh"
#include "gpu_test.cuh"
#include "gpu_test.h"
#include "material.cuh"
#include "mesh_services.h"
#include "optix/embedded.h"
#include "optix/triangles.cuh"
#include "optix_fixture.h"
#include "optix_runtime.cuh"
#include "quality.cuh"
#include "quality_view.cuh"
#include "random.cuh"
#include "renderer.h"
#include "start_optix.h"
#include "statistics.h"
#include "test_scene.h"
#include "vec3.cuh"

namespace
{
// --- Vectors and transforms on the host, in double ---------------------------------------------------------------

using dvec = std::array<double, 3>;
using matrix = std::array<double, 12>; // row-major 3×4, object to world: mesh_instance_desc's and OptixInstance's

constexpr matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
constexpr double infinite = std::numeric_limits<double>::infinity();

dvec add(const dvec& a, const dvec& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
dvec sub(const dvec& a, const dvec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
dvec mul(double s, const dvec& a) { return {s * a[0], s * a[1], s * a[2]}; }
double dot(const dvec& a, const dvec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
dvec cross(const dvec& a, const dvec& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double length(const dvec& a) { return std::sqrt(dot(a, a)); }
dvec unit(const dvec& a) { return mul(1 / length(a), a); }
double max_abs(const dvec& a) { return std::max({std::abs(a[0]), std::abs(a[1]), std::abs(a[2])}); }
// Triangle i's smallest altitude, twice its area over its longest edge, of corners given as 9 doubles per triangle
double smallest_altitude(std::span<const double> corners, int i)
{
    const auto corner = [&](int c) {
        const double* const v = corners.data() + 9 * static_cast<std::size_t>(i) + 3 * c;
        return dvec{v[0], v[1], v[2]};
    };
    const dvec a = corner(0);
    const dvec b = corner(1);
    const dvec c = corner(2);
    const double longest = std::max({length(sub(b, a)), length(sub(c, b)), length(sub(a, c))});
    return length(cross(sub(b, a), sub(c, a))) / longest;
}
// The largest difference of a coordinate; NaN when either has one
double apart(const dvec& a, const dvec& b)
{
    const dvec d = sub(a, b);
    return std::isnan(d[0]) || std::isnan(d[1]) || std::isnan(d[2]) ? std::numeric_limits<double>::quiet_NaN()
                                                                      : max_abs(d);
}
bool finite(const dvec& a) { return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]); }

dvec point(const matrix& m, const dvec& p)
{
    return {m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3], m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
            m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]};
}

// a·b: b applied first
matrix compose(const matrix& a, const matrix& b)
{
    matrix c{};
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
        {
            c[4 * i + j] =
                a[4 * i] * b[j] + a[4 * i + 1] * b[4 + j] + a[4 * i + 2] * b[8 + j] + (j == 3 ? a[4 * i + 3] : 0);
        }
    }
    return c;
}
matrix compose(const matrix& a, const matrix& b, const matrix& c) { return compose(a, compose(b, c)); }

matrix translation(double x, double y, double z) { return {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z}; }
matrix scaling(double x, double y, double z) { return {x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0}; }

// Rodrigues: R = cos θ·I + sin θ·[k]× + (1 − cos θ)·k kᵀ about the unit axis k
matrix rotation(const dvec& axis, double degrees)
{
    const dvec k = unit(axis);
    const double a = degrees * std::numbers::pi / 180;
    const double c = std::cos(a);
    const double s = std::sin(a);
    matrix m{};
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            m[4 * i + j] = (i == j ? c : 0) + (1 - c) * k[i] * k[j];
        }
    }
    m[1] -= s * k[2];
    m[2] += s * k[1];
    m[4] += s * k[2];
    m[6] -= s * k[0];
    m[8] -= s * k[1];
    m[9] += s * k[0];
    return m;
}

dvec row(const matrix& m, std::size_t i) { return {m[4 * i], m[4 * i + 1], m[4 * i + 2]}; }
double determinant(const matrix& m) { return dot(row(m, 0), cross(row(m, 1), row(m, 2))); }

// M⁻ᵀ of M's 3×3 part, row-major 3×3: the cofactor matrix over the determinant, whose rows are r₁×r₂, r₂×r₀, r₀×r₁ of
// M's rows (M·cof(M)ᵀ = det M·I)
std::array<double, 9> inverse_transpose(const matrix& m)
{
    const double det = determinant(m);
    const std::array<dvec, 3> rows{cross(row(m, 1), row(m, 2)), cross(row(m, 2), row(m, 0)),
                                   cross(row(m, 0), row(m, 1))};
    std::array<double, 9> n{};
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            n[3 * i + j] = rows[i][j] / det;
        }
    }
    return n;
}

dvec times(const std::array<double, 9>& n, const dvec& v)
{
    return {n[0] * v[0] + n[1] * v[1] + n[2] * v[2], n[3] * v[0] + n[4] * v[1] + n[5] * v[2],
            n[6] * v[0] + n[7] * v[1] + n[8] * v[2]};
}

// The transform as the GPU gets it, in float, read back in double: the reference intersects what the GPU does
matrix in_float(const matrix& m)
{
    matrix f{};
    for (std::size_t i = 0; i < 12; ++i)
    {
        f[i] = static_cast<float>(m[i]);
    }
    return f;
}

std::array<float, 3> to_float(const dvec& v)
{
    return {static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])};
}
dvec of(const std::array<float, 3>& v) { return {v[0], v[1], v[2]}; }
dvec of(const float3& v) { return {v.x, v.y, v.z}; }

// A float `steps` ulps away (the bit pattern moved, toward +∞ for positive steps); 0 stays put
float ulps_away(float x, int steps)
{
    if (x == 0.0f || steps == 0)
    {
        return x;
    }
    const auto bits = std::bit_cast<std::int32_t>(x);
    return std::bit_cast<float>(bits + (x > 0 ? steps : -steps));
}

// The test's own random numbers on the host, for the rays it gives OptiX: splitmix64 (Steele, Lea and Flood, OOPSLA
// 2014), a literal seed per check, the same numbers on every run and every computer
class host_random
{
public:
    explicit host_random(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next()
    {
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * 0x1p-53; } // [0, 1)
    double between(double a, double b) { return a + (b - a) * uniform(); }
    int below(int n) { return std::min(n - 1, static_cast<int>(uniform() * n)); }
    // Uniform on the unit sphere (Archimedes: z uniform)
    dvec direction()
    {
        const double z = between(-1, 1);
        const double phi = 2 * std::numbers::pi * uniform();
        const double r = std::sqrt(std::max(0.0, 1 - z * z));
        return {r * std::cos(phi), r * std::sin(phi), z};
    }
    // Uniform on a triangle: (b1, b2), the weights of corners 1 and 2
    std::pair<double, double> barycentrics()
    {
        double a = uniform();
        double b = uniform();
        if (a + b > 1)
        {
            a = 1 - a;
            b = 1 - b;
        }
        return {a, b};
    }

private:
    std::uint64_t state_;
};

// --- Meshes laid out as mesh_tables.h says, by the test -----------------------------------------------------------

// The shared arrays of mesh_tables.h, its comments read as the layout: a part's vertices from first_vertex (normals and
// UVs at the same indices: zeros where the part has none), its triangles from first_triangle, relative to its first
// vertex, material_of by the same index; instance k with its 3×4 and inverse transpose in float and the id of its
// triangle 0, the ids running on from triangle_base + 1 in instance order
struct host_tables
{
    std::vector<gpu::mesh_desc> meshes; // each part as given
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    std::vector<uint3> triangles;
    std::vector<std::uint32_t> material_of;
    std::vector<gpu::mesh_part> parts;
    std::vector<gpu::mesh_instance> instances;
    std::vector<gpu::mesh_material> materials;
    int triangle_base = 1000; // the absorber's id: the books' primitives and particles below it
    int next_id = 1001;

    int add_part(const gpu::mesh_desc& m)
    {
        const auto vertices = static_cast<unsigned int>(m.positions.size() / 3);
        parts.push_back({.first_vertex = static_cast<unsigned int>(positions.size()),
                         .first_triangle = static_cast<unsigned int>(triangles.size()),
                         .has_normals = m.normals.empty() ? 0u : 1u,
                         .has_uvs = m.uvs.empty() ? 0u : 1u});
        for (unsigned int i = 0; i < vertices; ++i)
        {
            positions.push_back(make_float3(m.positions[3 * i], m.positions[3 * i + 1], m.positions[3 * i + 2]));
            normals.push_back(m.normals.empty()
                                  ? make_float3(0, 0, 0)
                                  : make_float3(m.normals[3 * i], m.normals[3 * i + 1], m.normals[3 * i + 2]));
            uvs.push_back(m.uvs.empty() ? make_float2(0, 0) : make_float2(m.uvs[2 * i], m.uvs[2 * i + 1]));
        }
        for (std::size_t t = 0; t < m.indices.size() / 3; ++t)
        {
            triangles.push_back(make_uint3(m.indices[3 * t], m.indices[3 * t + 1], m.indices[3 * t + 2]));
            material_of.push_back(m.material_of.empty() ? 0u : m.material_of[t]);
        }
        meshes.push_back(m);
        return static_cast<int>(parts.size()) - 1;
    }

    int add_instance(int part, const matrix& object_to_world)
    {
        gpu::mesh_instance in{};
        const matrix m = in_float(object_to_world);
        const std::array<double, 9> n = inverse_transpose(object_to_world);
        for (std::size_t i = 0; i < 12; ++i)
        {
            in.object_to_world[i] = static_cast<float>(m[i]);
        }
        for (std::size_t i = 0; i < 9; ++i)
        {
            in.normal_to_world[i] = static_cast<float>(n[i]);
        }
        in.part = part;
        in.first_id = next_id;
        next_id += static_cast<int>(meshes[static_cast<std::size_t>(part)].indices.size() / 3);
        instances.push_back(in);
        return static_cast<int>(instances.size()) - 1;
    }

    std::size_t triangle_count(int instance) const
    {
        return meshes[static_cast<std::size_t>(instances[static_cast<std::size_t>(instance)].part)].indices.size() / 3;
    }
};

// A scene material of the hand-built tables: its index (the absorber after them, then the scratch slots)
gpu::mesh_material scene_material(int index)
{
    gpu::mesh_material m{};
    m.scene_material = index;
    return m;
}

// What an untouched scratch slot holds: a material no mapping writes (the glTF mapping writes metal and lambertian)
constexpr gpu::material<float> slot_sentinel{.kind = gpu::material_kind::isotropic,
                                             .albedo = {-7.0f, -7.0f, -7.0f},
                                             .fuzz = -7.0f,
                                             .refraction_index = -7.0f};
constexpr gpu::microfacet facet_sentinel{.kind = gpu::microfacet_kind::none, .alpha = -7.0f, .roughness = -7.0f};

// The tables on the GPU, and beside them what triangle_surface reads of the integrator's: the scene's materials, their
// texture indices and side entries, then the absorber's entry and `slots` scratch slots (entry M the absorber, then one
// slot per pixel), all through one optix_params
class device_tables
{
public:
    device_tables(const host_tables& t, std::span<const gpu::material<float>> scene_materials,
                  std::span<const gpu::material_texture> textures, int slots)
        : positions_(t.positions),
          normals_(t.normals),
          uvs_(t.uvs),
          triangles_(t.triangles),
          material_of_(t.material_of),
          parts_(t.parts),
          instances_(t.instances),
          materials_(t.materials),
          view_materials_(with_slots(scene_materials, slots)),
          view_facets_(facets_with_slots(scene_materials.size(), slots)),
          material_textures_(std::vector<gpu::material_texture>(textures.begin(), textures.end())),
          scene_count_(static_cast<int>(scene_materials.size())),
          slots_(slots),
          instance_count_(static_cast<int>(t.instances.size())),
          triangle_base_(t.triangle_base)
    {
    }

    gpu::optix_params params() const
    {
        gpu::optix_params p{};
        p.view.base.materials = view_materials_.get();
        p.view.base.material_textures = material_textures_.get();
        p.view.microfacets = view_facets_.get();
        p.meshes.positions = positions_.get();
        p.meshes.normals = normals_.get();
        p.meshes.uvs = uvs_.get();
        p.meshes.triangles = triangles_.get();
        p.meshes.material_of = material_of_.get();
        p.meshes.parts = parts_.get();
        p.meshes.instances = instances_.get();
        p.meshes.materials = materials_.get();
        p.meshes.instance_count = instance_count_;
        p.meshes.triangle_base = triangle_base_;
        p.meshes.absorber_material = scene_count_;
        p.meshes.scratch_base = slots_ > 0 ? scene_count_ + 1 : -1;
        p.meshes.scratch_materials = view_materials_.get();
        p.meshes.scratch_microfacets = view_facets_.get();
        return p;
    }
    int absorber() const { return scene_count_; }

private:
    static std::vector<gpu::material<float>> with_slots(std::span<const gpu::material<float>> scene, int slots)
    {
        std::vector<gpu::material<float>> all(scene.begin(), scene.end());
        all.push_back({.kind = gpu::material_kind::diffuse_light, .albedo = {}, .fuzz = 0, .refraction_index = 1});
        all.insert(all.end(), static_cast<std::size_t>(slots), slot_sentinel);
        return all;
    }
    static std::vector<gpu::microfacet> facets_with_slots(std::size_t scene, int slots)
    {
        std::vector<gpu::microfacet> all(scene + 1, gpu::microfacet{.kind = gpu::microfacet_kind::none});
        all.insert(all.end(), static_cast<std::size_t>(slots), facet_sentinel);
        return all;
    }

    gpu_test::device_buffer<float3> positions_;
    gpu_test::device_buffer<float3> normals_;
    gpu_test::device_buffer<float2> uvs_;
    gpu_test::device_buffer<uint3> triangles_;
    gpu_test::device_buffer<std::uint32_t> material_of_;
    gpu_test::device_buffer<gpu::mesh_part> parts_;
    gpu_test::device_buffer<gpu::mesh_instance> instances_;
    gpu_test::device_buffer<gpu::mesh_material> materials_;
    gpu_test::device_buffer<gpu::material<float>> view_materials_;
    gpu_test::device_buffer<gpu::microfacet> view_facets_;
    gpu_test::device_buffer<gpu::material_texture> material_textures_;
    int scene_count_;
    int slots_;
    int instance_count_;
    int triangle_base_;
};

// --- The probe's words (the layout in the file's header) -----------------------------------------------------------

constexpr int probe_in_words = 20;
constexpr int probe_out_words = 40;
constexpr int probe_batch = 1 << 15; // samples per launch: 7.9 MB of words

struct probe_sample
{
    std::array<float, 3> origin{};
    std::array<float, 3> direction{};
    int from = -1;
    bool inward = false;
    std::array<float, 3> facing{};
    int mode = 0; // 0: closest; 1: surface_of the triangle hit below
    int instance = 0;
    unsigned int primitive = 0;
    float b1 = 0;
    float b2 = 0;
    float lobe = 0;
};

struct probe_answer
{
    int index = -1;
    float t = 0;
    bool entering = false;
    int instance = 0;
    unsigned int primitive = 0;
    float b1 = 0;
    float b2 = 0;
    float lobe = 0;
    bool surfaced = false; // mode 1 ran surface_of
    std::array<float, 3> p{};
    std::array<float, 3> normal{};
    bool front = false;
    std::array<float, 3> outward{};
    std::array<float, 3> texture_point{};
    float u = 0;
    float v = 0;
    int material = -1;
    int texture_index = -1;
    int from = -1;
    std::array<float, 3> facing{};
    int kind = -1; // the material at s.material
    std::array<float, 3> albedo{};
    float fuzz = 0;
    float refraction_index = 0;
    int facet_kind = -1; // its side entry, -1 without side entries
    float alpha = 0;
    float roughness = 0;
    bool wears_gltf = false; // the CUDA twin's word 39
};

float word(int i) { return std::bit_cast<float>(i); }
float word(unsigned int i) { return std::bit_cast<float>(i); }
int int_of(float f) { return std::bit_cast<int>(f); }

void encode(const probe_sample& s, float* w)
{
    w[0] = s.origin[0];
    w[1] = s.origin[1];
    w[2] = s.origin[2];
    w[3] = s.direction[0];
    w[4] = s.direction[1];
    w[5] = s.direction[2];
    w[6] = 0.0f;
    w[7] = word(s.from);
    w[8] = s.inward ? 1.0f : 0.0f;
    w[9] = s.facing[0];
    w[10] = s.facing[1];
    w[11] = s.facing[2];
    w[12] = word(s.mode);
    w[13] = word(s.instance);
    w[14] = word(s.primitive);
    w[15] = s.b1;
    w[16] = s.b2;
    w[17] = s.lobe;
}

probe_answer decode(const float* w)
{
    probe_answer a;
    a.index = int_of(w[0]);
    a.t = w[1];
    a.entering = int_of(w[2]) != 0;
    a.instance = int_of(w[3]);
    a.primitive = std::bit_cast<unsigned int>(w[4]);
    a.b1 = w[5];
    a.b2 = w[6];
    a.lobe = w[7];
    a.p = {w[8], w[9], w[10]};
    a.normal = {w[11], w[12], w[13]};
    a.front = int_of(w[14]) != 0;
    a.outward = {w[15], w[16], w[17]};
    a.texture_point = {w[18], w[19], w[20]};
    a.u = w[21];
    a.v = w[22];
    a.material = int_of(w[23]);
    a.texture_index = int_of(w[24]);
    a.from = int_of(w[25]);
    a.facing = {w[26], w[27], w[28]};
    a.surfaced = int_of(w[29]) == 1;
    a.kind = int_of(w[30]);
    a.albedo = {w[31], w[32], w[33]};
    a.fuzz = w[34];
    a.refraction_index = w[35];
    a.facet_kind = int_of(w[36]);
    a.alpha = w[37];
    a.roughness = w[38];
    a.wears_gltf = int_of(w[39]) == 1;
    return a;
}

// --- Device code ---------------------------------------------------------------------------------------------------

// The 40 words of a surface (the probe's mode 1), as the CUDA twin writes them
__device__ void write_surface(const gpu::optix_params& p, const gpu::bvh_hit<float>& hit,
                              const gpu::triangle_memo& memo, const gpu::surface<float>& s, bool gltf, float* out)
{
    out[0] = __int_as_float(hit.index);
    out[1] = hit.t;
    out[2] = __int_as_float(hit.entering ? 1 : 0);
    out[3] = __int_as_float(memo.instance);
    out[4] = __uint_as_float(memo.primitive);
    out[5] = memo.b1;
    out[6] = memo.b2;
    out[7] = memo.lobe;
    out[8] = s.rec.p.x;
    out[9] = s.rec.p.y;
    out[10] = s.rec.p.z;
    out[11] = s.rec.normal.x;
    out[12] = s.rec.normal.y;
    out[13] = s.rec.normal.z;
    out[14] = __int_as_float(s.rec.front_face ? 1 : 0);
    out[15] = s.outward.x;
    out[16] = s.outward.y;
    out[17] = s.outward.z;
    out[18] = s.texture_point.x;
    out[19] = s.texture_point.y;
    out[20] = s.texture_point.z;
    out[21] = s.u;
    out[22] = s.v;
    out[23] = __int_as_float(s.material);
    out[24] = __int_as_float(s.texture_index);
    out[25] = __int_as_float(s.from);
    out[26] = memo.facing.x;
    out[27] = memo.facing.y;
    out[28] = memo.facing.z;
    out[29] = __int_as_float(1);
    const gpu::material<float> m = p.view.base.materials[s.material];
    out[30] = __int_as_float(static_cast<int>(m.kind));
    out[31] = m.albedo.x;
    out[32] = m.albedo.y;
    out[33] = m.albedo.z;
    out[34] = m.fuzz;
    out[35] = m.refraction_index;
    if (p.view.microfacets != nullptr)
    {
        const gpu::microfacet f = p.view.microfacets[s.material];
        out[36] = __int_as_float(static_cast<int>(f.kind));
        out[37] = f.alpha;
        out[38] = f.roughness;
    }
    else
    {
        out[36] = __int_as_float(-1);
    }
    out[39] = __int_as_float(gltf ? 1 : 0);
}

// triangles.cuh's triangle_surface in a plain CUDA kernel, on tables the test lays out: the probe's mode 1 without
// OptiX, the same words in and out, n samples, sample k in scratch slot scratch_base + k
template <int Unused>
__global__ void surface_kernel(gpu::optix_params p, float* values, int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
    {
        return;
    }
    const float* const in = values + static_cast<std::size_t>(probe_in_words) * k;
    float* const out =
        values + static_cast<std::size_t>(probe_in_words) * n + static_cast<std::size_t>(probe_out_words) * k;
    gpu::triangle_memo memo;
    memo.facing = {in[9], in[10], in[11]};
    memo.instance = __float_as_int(in[13]);
    memo.primitive = __float_as_uint(in[14]);
    memo.b1 = in[15];
    memo.b2 = in[16];
    memo.lobe = in[17];
    const gpu::ray<float> r{.origin = {in[0], in[1], in[2]}, .direction = {in[3], in[4], in[5]}, .time = in[6]};
    const int id = p.meshes.instances[memo.instance].first_id + static_cast<int>(memo.primitive);
    const gpu::bvh_hit<float> hit{.index = id, .t = 1.0f, .entering = false};
    const bool gltf = gpu::wears_gltf(p.meshes, memo);
    const gpu::surface<float> s = gpu::triangle_surface(p, r, hit, memo, p.meshes.scratch_base + k);
    write_surface(p, hit, memo, s, gltf, out);
}

// absorber_surface for n rays (origin 0-2, direction 3-5 of 6 floats each): 12 words out per ray at 6n + 12k: rec.p
// 0-2, rec.normal 3-5, front_face 6, material 7, from 8, texture_index 9
template <int Unused>
__global__ void absorber_kernel(gpu::mesh_tables m, float* values, int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
    {
        return;
    }
    const float* const in = values + 6 * k;
    float* const out = values + 6 * n + 12 * k;
    const gpu::ray<float> r{.origin = {in[0], in[1], in[2]}, .direction = {in[3], in[4], in[5]}, .time = 0.0f};
    const gpu::surface<float> s = gpu::absorber_surface(m, r);
    out[0] = s.rec.p.x;
    out[1] = s.rec.p.y;
    out[2] = s.rec.p.z;
    out[3] = s.rec.normal.x;
    out[4] = s.rec.normal.y;
    out[5] = s.rec.normal.z;
    out[6] = __int_as_float(s.rec.front_face ? 1 : 0);
    out[7] = __int_as_float(s.material);
    out[8] = __int_as_float(s.from);
    out[9] = __int_as_float(s.texture_index);
}

// offset-origin's tallies: [0] a point not strictly on n's side, [1] a coordinate moved beyond the offset's bound,
// [2] a coordinate not finite, [3] the planted offset (toward -n) on n's side: one it failed to refuse, [4] coordinates
// below 1/32 drawn, [5] coordinates of ±1/32 drawn: the other branch of Wächter and Binder's offset and its switch,
// which offset_origin followed until decision 9, kept as edge cases
constexpr int offset_tallies = 6;

// One (point, normal) pair: the point's scale 10^-3 to 10^4, log-uniform; each coordinate uniform in [-s, s], or with
// probability 1/8 below 1/32 (the old n/65536 branch), or with probability 1/8 one of the old switch's own floats: 0,
// -0, ±1/32, ±the float below 1/32. The normal uniform on the sphere, or with probability 1/8 along an axis, or 1/8
// nearly along one (its other coordinates 10^-3 and 10^-6, which the old offset rounded to 0 steps).
__device__ void offset_pair(gpu::pcg32& rng, gpu::point3<float>& p, gpu::vec3<float>& n)
{
    const double s = ::pow(10.0, -3.0 + 7.0 * gpu::random_real<double>(rng));
    float c[3];
    for (int i = 0; i < 3; ++i)
    {
        const double u = gpu::random_real<double>(rng);
        const double w = 2 * gpu::random_real<double>(rng) - 1;
        if (u < 0.75)
        {
            c[i] = static_cast<float>(s * w);
        }
        else if (u < 0.875)
        {
            c[i] = static_cast<float>(w / 32);
        }
        else
        {
            const float below = ::nextafterf(0.03125f, 0.0f);
            const float edges[6] = {0.0f, -0.0f, 0.03125f, -0.03125f, below, -below};
            c[i] = edges[::min(5, static_cast<int>(6 * gpu::random_real<float>(rng)))];
        }
    }
    p = {c[0], c[1], c[2]};
    const double kind = gpu::random_real<double>(rng);
    if (kind < 0.75)
    {
        n = gpu::random_unit_vector<float>(rng);
        return;
    }
    const int axis = ::min(2, static_cast<int>(3 * gpu::random_real<float>(rng)));
    const double sign = gpu::random_real<float>(rng) < 0.5f ? -1.0 : 1.0;
    double v[3] = {0, 0, 0};
    v[axis] = sign;
    if (kind >= 0.875)
    {
        v[(axis + 1) % 3] = 1e-3 * (2 * gpu::random_real<double>(rng) - 1);
        v[(axis + 2) % 3] = 1e-6 * (2 * gpu::random_real<double>(rng) - 1);
    }
    const double l = ::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    n = {static_cast<float>(v[0] / l), static_cast<float>(v[1] / l), static_cast<float>(v[2] / l)};
}

// (o - p)·n in double: the side of the plane through p that o lies on
__device__ double side_of(const gpu::point3<float>& o, const gpu::point3<float>& p, const gpu::vec3<float>& n)
{
    return (static_cast<double>(o.x) - p.x) * n.x + (static_cast<double>(o.y) - p.y) * n.y +
           (static_cast<double>(o.z) - p.z) * n.z;
}

template <int Unused>
__global__ void offset_kernel(std::uint64_t seed, int count, unsigned long long* tallies, double* worst)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    gpu::pcg32 rng = gpu::sample_generator(seed, static_cast<std::uint64_t>(t), 0);
    unsigned long long local[offset_tallies] = {};
    double largest = 0;
    for (int i = 0; i < count; ++i)
    {
        gpu::point3<float> p;
        gpu::vec3<float> n;
        offset_pair(rng, p, n);
        const gpu::point3<float> o = gpu::offset_origin(p, n);
        const double size = ::fmax(::fabs(static_cast<double>(p.x)),
                                   ::fmax(::fabs(static_cast<double>(p.y)), ::fabs(static_cast<double>(p.z))));
        const double bound = 0x1p-15 * size + 0x1p-15;
        const float pc[3] = {p.x, p.y, p.z};
        const float oc[3] = {o.x, o.y, o.z};
        bool too_far = false;
        bool unfinite = false;
        for (int k = 0; k < 3; ++k)
        {
            const double moved = ::fabs(static_cast<double>(oc[k]) - pc[k]);
            unfinite = unfinite || !::isfinite(oc[k]);
            too_far = too_far || !(moved <= bound);
            largest = ::isfinite(moved) ? ::fmax(largest, moved / bound) : largest;
            local[4] += ::fabsf(pc[k]) < 0.03125f ? 1 : 0;
            local[5] += ::fabsf(pc[k]) == 0.03125f ? 1 : 0;
        }
        local[0] += side_of(o, p, n) > 0 ? 0 : 1;
        local[1] += too_far ? 1 : 0;
        local[2] += unfinite ? 1 : 0;
        local[3] += side_of(gpu::offset_origin(p, -n), p, n) > 0 ? 1 : 0;
    }
    for (int k = 0; k < offset_tallies; ++k)
    {
        atomicAdd(&tallies[k], local[k]);
    }
    worst[t] = largest;
}

// Every world triangle (9 doubles each) against rays in the probe's input layout, in double: Möller and Trumbore's
// test (1997), the nearest crossing beyond t_min. near_edge: some triangle crossed in front of it (anywhere, for a
// miss) has its smallest barycentric coordinate within 10^-4 of 0, on either side.
struct brute_answer
{
    int triangle;
    int near_edge;
    double t;
    double b1;
    double b2;
    double cosine; // |cos θ| between the ray and the hit face
};

template <int Unused>
__global__ void brute_force_kernel(const double* corners, int triangles, const float* rays, int n, double t_min,
                                   brute_answer* answers)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
    {
        return;
    }
    const float* const r = rays + static_cast<std::size_t>(probe_in_words) * k;
    const double o[3] = {r[0], r[1], r[2]};
    const double d[3] = {r[3], r[4], r[5]};
    brute_answer best{.triangle = -1, .near_edge = 0, .t = gpu::infinity<double>, .b1 = 0, .b2 = 0, .cosine = 0};
    double near_t = gpu::infinity<double>;
    for (int i = 0; i < triangles; ++i)
    {
        const double* const c = corners + 9 * static_cast<std::size_t>(i);
        const double e1[3] = {c[3] - c[0], c[4] - c[1], c[5] - c[2]};
        const double e2[3] = {c[6] - c[0], c[7] - c[1], c[8] - c[2]};
        const double pv[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
        const double det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
        if (det == 0)
        {
            continue; // parallel to the plane
        }
        const double inv = 1 / det;
        const double tv[3] = {o[0] - c[0], o[1] - c[1], o[2] - c[2]};
        const double u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * inv;
        const double qv[3] = {tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2],
                              tv[0] * e1[1] - tv[1] * e1[0]};
        const double v = (d[0] * qv[0] + d[1] * qv[1] + d[2] * qv[2]) * inv;
        const double t = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv;
        if (!(t > t_min))
        {
            continue;
        }
        const double margin = ::fmin(1 - u - v, ::fmin(u, v));
        if (::fabs(margin) < 1e-4)
        {
            near_t = ::fmin(near_t, t);
        }
        if (margin >= 0 && t < best.t)
        {
            const double nx = e1[1] * e2[2] - e1[2] * e2[1];
            const double ny = e1[2] * e2[0] - e1[0] * e2[2];
            const double nz = e1[0] * e2[1] - e1[1] * e2[0];
            best = {.triangle = i,
                    .near_edge = 0,
                    .t = t,
                    .b1 = u,
                    .b2 = v,
                    .cosine = ::fabs(d[0] * nx + d[1] * ny + d[2] * nz) /
                              (::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) * ::sqrt(nx * nx + ny * ny + nz * nz))};
        }
    }
    best.near_edge = near_t <= best.t * (1 + 1e-9) ? 1 : 0;
    answers[k] = best;
}

// --- Host helpers ---------------------------------------------------------------------------------------------------

// The check module's PTX as the runtime takes it: always this span (the runtime keeps one module per PTX address)
std::span<const unsigned char> check_module()
{
    return {optix_check_ptx, optix_check_ptx_size};
}

// Runs the plain kernel on samples in the probe's layout and decodes its answers
std::vector<probe_answer> surfaces_in_cuda(const device_tables& tables, std::span<const probe_sample> samples)
{
    const auto n = static_cast<int>(samples.size());
    std::vector<float> words(static_cast<std::size_t>(probe_in_words + probe_out_words) * samples.size(), 0.0f);
    for (std::size_t k = 0; k < samples.size(); ++k)
    {
        encode(samples[k], words.data() + probe_in_words * k);
    }
    const gpu_test::device_buffer<float> values(words);
    surface_kernel<0><<<(n + 127) / 128, 128>>>(tables.params(), values.get(), n);
    gpu_test::finish_launch("the triangle_surface kernel");
    const std::vector<float> out = values.download();
    std::vector<probe_answer> answers;
    answers.reserve(samples.size());
    for (std::size_t k = 0; k < samples.size(); ++k)
    {
        answers.push_back(decode(out.data() + probe_in_words * samples.size() + probe_out_words * k));
    }
    return answers;
}

// A scene the renderer accepts as plain data: a camera that passes its checks, a matte and a metal material on a solid
// texture, and the mesh_set's materials 0 and 1 wearing them
gpu::camera_desc small_camera()
{
    return {.image_width = 8,
            .image_height = 8,
            .sqrt_spp = 1,
            .max_depth = 4,
            .center = {0, 0, 10},
            .pixel00_loc = {-1, 1, 0},
            .pixel_delta_u = {0.25, 0, 0},
            .pixel_delta_v = {0, -0.25, 0},
            .sky_bottom = {1, 1, 1},
            .sky_top = {0.5, 0.7, 1},
            .sampling = gpu::sampling_strategy::material};
}

gpu::scene_desc mesh_scene(std::vector<gpu::mesh_desc> meshes, const std::vector<std::pair<int, matrix>>& placements)
{
    gpu::scene_desc s;
    s.camera = small_camera();
    s.textures = {{.kind = gpu::texture_kind::solid, .rgb = {0.5, 0.5, 0.5}}};
    s.materials = {{.kind = gpu::material_kind::lambertian, .texture = 0},
                   {.kind = gpu::material_kind::metal, .albedo = {0.8, 0.8, 0.8}, .fuzz = 0.2}};
    s.meshes.materials = {{.scene_material = 0}, {.scene_material = 1}};
    for (gpu::mesh_desc& m : meshes)
    {
        s.meshes.meshes.push_back(std::make_shared<const gpu::mesh_desc>(std::move(m)));
    }
    for (const auto& [mesh, m] : placements)
    {
        s.meshes.instances.push_back({.mesh = mesh, .object_to_world = m});
    }
    return s;
}

// A scene in OptiX as a session would hold it, for launch_test: the books' tables (test_scene.h: solid colors, no
// lights list, the renderer's own BVH and ids), the quality kernel's arrays, and the runtime's upload over them.
// Declared in that order, so the upload goes first.
struct optix_upload
{
    gpu_test::device_scene books;
    gpu::detail::quality_scene_ptr quality;
    std::unique_ptr<gpu::detail::optix_scene> scene;
    int triangle_base; // media.base + media.count: the absorber's id, the triangles' ids above it

    optix_upload(gpu::optix_runtime& runtime, const gpu::scene_desc& desc,
                 const gpu::optix_upload_options& options = {})
        : books(desc),
          quality(gpu::detail::upload_quality(desc, books.sphere_ids(), books.quad_ids())),
          scene(runtime.upload(desc, books.view(), *quality, books.sphere_ids(), books.quad_ids(), options)),
          triangle_base(books.surface_count() + static_cast<int>(desc.media.size()))
    {
    }
};

// The first launch of the probe's pipeline in a process (with the books' own intersection programs, and with P7's
// probe in their place) compiles the check module and makes the pipeline: left out of the budget
bool probe_pipeline_made[2] = {false, false};

// Every sample through __raygen__probe_mesh, in launches of at most probe_batch; `words`, when given, receives the
// probe intersection program's 16 words per sample (P7)
std::vector<probe_answer> probe(gpu_test::context& ctx, gpu::detail::optix_scene& scene,
                                std::span<const probe_sample> samples, std::uint64_t seed, bool probing = false,
                                std::vector<unsigned long long>* words = nullptr)
{
    std::vector<probe_answer> answers;
    answers.reserve(samples.size());
    for (std::size_t first = 0; first < samples.size(); first += probe_batch)
    {
        const std::size_t n = std::min<std::size_t>(probe_batch, samples.size() - first);
        std::vector<float> host(static_cast<std::size_t>(probe_in_words + probe_out_words) * n, 0.0f);
        for (std::size_t k = 0; k < n; ++k)
        {
            encode(samples[first + k], host.data() + probe_in_words * k);
        }
        const gpu_test::device_buffer<float> values(host);
        const gpu_test::device_buffer<unsigned long long> counts(words ? 16 * n : 0);
        scene.prepare_view(static_cast<int>(n), 1); // the scratch slots of samples 0 to n - 1
        const auto launch = [&] {
            scene.launch_test(check_module(), "__raygen__probe_mesh", static_cast<int>(n), 1, gpu::camera<float>{},
                              {.counts = counts.get(), .values = values.get(), .seed = seed});
            gpu_test::finish_launch("a launch of __raygen__probe_mesh");
        };
        if (!probe_pipeline_made[probing ? 1 : 0])
        {
            gpu_test::optix::excluded_seconds(ctx, launch);
            probe_pipeline_made[probing ? 1 : 0] = true;
        }
        else
        {
            launch();
        }
        const std::vector<float> out = values.download();
        for (std::size_t k = 0; k < n; ++k)
        {
            answers.push_back(decode(out.data() + probe_in_words * n + probe_out_words * k));
        }
        if (words)
        {
            const std::vector<unsigned long long> w = counts.download();
            words->insert(words->end(), w.begin(), w.end());
        }
    }
    return answers;
}

// The process's OptiX for check `check`, its render module compiled (outside the budget), or null after the fixture's
// SKIP or FAIL
std::shared_ptr<gpu::optix_runtime> optix_for(gpu_test::context& ctx, std::string_view check)
{
    std::shared_ptr<gpu::optix_runtime> runtime = gpu_test::optix::runtime(ctx, check);
    if (runtime)
    {
        gpu_test::optix::excluded_seconds(ctx, [&runtime] { runtime->compile_programs(); });
    }
    return runtime;
}

// An OptiX check's body: a refusal or a CUDA error (the probe's raygen missing from the check module, an upload that
// throws) is that check's FAIL with the message, and the suite goes on
template <class F>
void guarded(gpu_test::context& ctx, std::string_view check, F&& body)
{
    try
    {
        std::forward<F>(body)();
    }
    catch (const std::exception& error)
    {
        ctx.fail(check, std::format("threw: {}", error.what()));
    }
}

// --- triangle_surface's rules, recomputed in double -------------------------------------------------------------

struct expected_surface
{
    dvec p{};
    dvec ng{};
    dvec facing{};
    dvec ns{};
    bool front = false;
    double u = 0;
    double v = 0;
    int material = -1;
    int texture_index = -1;
    int id = 0;
    double scale = 0;       // the hit triangle's largest world coordinate: p's tolerance is 10^-5 of it
    bool grazing = false;   // |cos θ| between ng and the ray below 10^-4: the side isn't decided by the floats
    bool ns_close = false;  // a rule of the shading normal decides within 10^-4, or its sum cancels below 1% of terms
    bool fallback = false;  // ns = facing by one of the rules
};

// The hit of instance `instance`'s triangle `primitive` at (b1, b2) by a ray of direction d, from the floats the
// tables hold (positions, normals, UVs, the instance's 3×4 and inverse transpose)
expected_surface expected(const host_tables& t, std::span<const gpu::material_texture> textures, int instance,
                          unsigned int primitive, float b1, float b2, const dvec& d)
{
    const gpu::mesh_instance& in = t.instances[static_cast<std::size_t>(instance)];
    const gpu::mesh_part& part = t.parts[static_cast<std::size_t>(in.part)];
    const uint3 tri = t.triangles[part.first_triangle + primitive];
    const std::array<unsigned int, 3> corner{part.first_vertex + tri.x, part.first_vertex + tri.y,
                                             part.first_vertex + tri.z};
    matrix m{};
    std::array<double, 9> n{};
    for (std::size_t i = 0; i < 12; ++i)
    {
        m[i] = in.object_to_world[i];
    }
    for (std::size_t i = 0; i < 9; ++i)
    {
        n[i] = in.normal_to_world[i];
    }
    const std::array<double, 3> w{1.0 - b1 - b2, b1, b2};
    std::array<dvec, 3> p{};
    dvec local{};
    expected_surface e;
    for (std::size_t i = 0; i < 3; ++i)
    {
        p[i] = of(t.positions[corner[i]]);
        local = add(local, mul(w[i], p[i]));
        e.scale = std::max(e.scale, max_abs(point(m, p[i])));
    }
    e.p = point(m, local);
    e.ng = unit(times(n, cross(sub(p[1], p[0]), sub(p[2], p[0]))));
    const dvec dir = unit(d);
    const double cosine = dot(e.ng, dir);
    e.grazing = std::abs(cosine) < 1e-4;
    e.front = cosine < 0;
    e.facing = e.front ? e.ng : mul(-1, e.ng);
    e.fallback = true;
    e.ns = e.facing;
    if (part.has_normals != 0)
    {
        dvec sum{};
        double terms = 0;
        for (std::size_t i = 0; i < 3; ++i)
        {
            const dvec ni = times(n, of(t.normals[corner[i]]));
            sum = add(sum, mul(w[i], ni));
            terms += std::abs(w[i]) * length(ni);
        }
        const double l = length(sum);
        if (finite(sum) && l > 0)
        {
            dvec ns = mul(1 / l, sum);
            ns = e.front ? ns : mul(-1, ns);
            const double toward_face = dot(ns, e.facing);
            const double toward_ray = -dot(ns, dir);
            e.ns_close = std::abs(toward_face) < 1e-4 || std::abs(toward_ray) < 1e-4 || l < 0.01 * terms;
            if (toward_face > 0 && toward_ray > 0)
            {
                e.ns = ns;
                e.fallback = false;
            }
        }
    }
    if (part.has_uvs != 0)
    {
        for (std::size_t i = 0; i < 3; ++i)
        {
            const float2 uv = t.uvs[corner[i]];
            e.u += w[i] * uv.x;
            e.v += w[i] * uv.y;
        }
    }
    const gpu::mesh_material& mm = t.materials[t.material_of[part.first_triangle + primitive]];
    e.material = mm.scene_material;
    e.texture_index = e.material >= 0 && static_cast<std::size_t>(e.material) < textures.size()
                          ? textures[static_cast<std::size_t>(e.material)].texture_index
                          : -1;
    e.id = in.first_id + static_cast<int>(primitive);
    return e;
}

// The comparisons of a set of surfaces against their references, counted by kind
struct surface_tally
{
    std::int64_t samples = 0;
    std::int64_t grazing = 0;
    std::int64_t ns_close = 0;
    std::int64_t fallbacks = 0;
    std::int64_t point = 0;
    std::int64_t front = 0;
    std::int64_t facing = 0;
    std::int64_t normal = 0;
    std::int64_t uv = 0;
    std::int64_t ids = 0;
    std::int64_t copies = 0;
    double worst_point = 0; // in units of the scale
    double worst_facing = 0;
    double worst_normal = 0;
    double worst_uv = 0;

    void add(const expected_surface& e, const probe_answer& a, bool uv_may_be_zero)
    {
        ++samples;
        grazing += e.grazing ? 1 : 0;
        ns_close += e.ns_close ? 1 : 0;
        fallbacks += e.fallback ? 1 : 0;
        const double dp = apart(of(a.p), e.p) / e.scale;
        worst_point = std::isnan(dp) ? dp : std::max(worst_point, dp);
        point += dp <= 1e-5 ? 0 : 1;
        if (!e.grazing)
        {
            front += a.front == e.front ? 0 : 1;
            const double df = apart(of(a.outward), e.facing);
            worst_facing = std::isnan(df) ? df : std::max(worst_facing, df);
            facing += df <= 1e-5 ? 0 : 1;
            if (!e.ns_close)
            {
                const double dn = apart(of(a.normal), e.ns);
                worst_normal = std::isnan(dn) ? dn : std::max(worst_normal, dn);
                normal += dn <= 1e-5 ? 0 : 1;
            }
        }
        const double du = std::max(std::abs(a.u - e.u) / std::max(1.0, std::abs(e.u)),
                                   std::abs(a.v - e.v) / std::max(1.0, std::abs(e.v)));
        const bool zero = uv_may_be_zero && a.u == 0.0f && a.v == 0.0f;
        worst_uv = zero || std::isnan(du) ? worst_uv : std::max(worst_uv, du);
        uv += du <= 1e-5 || zero ? 0 : 1;
        ids += a.material == e.material && a.texture_index == e.texture_index && a.from == e.id ? 0 : 1;
        const bool same_point = std::bit_cast<std::array<std::uint32_t, 3>>(a.texture_point) ==
                                std::bit_cast<std::array<std::uint32_t, 3>>(a.p);
        const bool same_face = std::bit_cast<std::array<std::uint32_t, 3>>(a.facing) ==
                               std::bit_cast<std::array<std::uint32_t, 3>>(a.outward);
        copies += same_point && same_face && a.surfaced ? 0 : 1;
    }

    void report(gpu_test::context& ctx, const std::string& prefix) const
    {
        ctx.count(prefix + "-point", point, 0, samples);
        ctx.count(prefix + "-front", front, 0, samples - grazing);
        ctx.count(prefix + "-face-normal", facing, 0, samples - grazing);
        ctx.count(prefix + "-shading-normal", normal, 0, samples - grazing);
        ctx.count(prefix + "-uv", uv, 0, samples);
        ctx.count(prefix + "-material-texture-id", ids, 0, samples);
        ctx.count(prefix + "-copies", copies, 0, samples); // texture_point = rec.p, memo.facing = outward, bit for bit
        ctx.info(prefix + "-worst",
                 std::format("|dp| {:.3g} of the scale, face normal {:.3g}, shading normal {:.3g}, uv {:.3g} (all <= "
                             "1e-5); {} grazing hits and {} shading normals near a rule's threshold left out, {} "
                             "fallbacks to the face's normal, of {}",
                             worst_point, worst_facing, worst_normal, worst_uv, grazing, ns_close, fallbacks, samples));
    }
};

// A random hit of the tables: an instance, a triangle of its part, uniform barycentrics, a direction from either side
probe_sample random_hit(host_random& random, const host_tables& t, int instance)
{
    probe_sample s;
    s.mode = 1;
    s.instance = instance;
    s.primitive = static_cast<unsigned int>(random.below(static_cast<int>(t.triangle_count(instance))));
    const auto [b1, b2] = random.barycentrics();
    s.b1 = static_cast<float>(b1);
    s.b2 = static_cast<float>(b2);
    s.direction = to_float(random.direction());
    s.origin = {0.0f, 0.0f, 0.0f};
    return s;
}

// --- The meshes of the checks -----------------------------------------------------------------------------------

// 24 separate triangles with corners in [-1, 1]³ (none thinner than a 20th of its longest side squared), UVs in
// [-2, 3], and vertex normals of every kind triangle_surface's rules name: most within 60° of the face, some against
// it, some random, some zero and some NaN
gpu::mesh_desc wild_mesh(std::uint64_t seed)
{
    host_random random(seed);
    gpu::mesh_desc m;
    while (m.indices.size() < 3 * 24)
    {
        std::array<dvec, 3> c{};
        for (dvec& corner : c)
        {
            corner = {random.between(-1, 1), random.between(-1, 1), random.between(-1, 1)};
        }
        const dvec face = cross(sub(c[1], c[0]), sub(c[2], c[0]));
        const double longest = std::max({length(sub(c[1], c[0])), length(sub(c[2], c[1])), length(sub(c[0], c[2]))});
        if (length(face) < 0.05 * longest * longest)
        {
            continue;
        }
        const dvec ng = unit(face);
        const auto first = static_cast<std::uint32_t>(m.positions.size() / 3);
        for (const dvec& corner : c)
        {
            const std::array<float, 3> p = to_float(corner);
            m.positions.insert(m.positions.end(), p.begin(), p.end());
            const double kind = random.uniform();
            dvec n{};
            if (kind < 0.6)
            {
                n = unit(add(ng, mul(0.8 * random.uniform(), random.direction()))); // within ~53° of ng
            }
            else if (kind < 0.7)
            {
                n = mul(-1, ng);
            }
            else if (kind < 0.85)
            {
                n = random.direction();
            }
            else if (kind < 0.95)
            {
                n = {0, 0, 0};
            }
            else
            {
                n = {std::numeric_limits<double>::quiet_NaN(), 0, 1};
            }
            const std::array<float, 3> nf = to_float(n);
            m.normals.insert(m.normals.end(), nf.begin(), nf.end());
            m.uvs.push_back(static_cast<float>(random.between(-2, 3)));
            m.uvs.push_back(static_cast<float>(random.between(-2, 3)));
        }
        m.indices.insert(m.indices.end(), {first, first + 1, first + 2});
        m.material_of.push_back(0);
    }
    return m;
}

// Materials 0 and 1 by turns, so that a triangle's material_of matters
gpu::mesh_desc alternating(gpu::mesh_desc m)
{
    for (std::size_t t = 0; t < m.material_of.size(); ++t)
    {
        m.material_of[t] = static_cast<std::uint32_t>(t % 2);
    }
    return m;
}

// One triangle (0, 0, 0), (1, 0, 0), (0, 1, 0): its counter-clockwise side +z. Normals: all three `n`, or none
gpu::mesh_desc one_triangle(std::span<const float> normals = {})
{
    gpu::mesh_desc m{.positions = {0, 0, 0, 1, 0, 0, 0, 1, 0},
                     .normals = {},
                     .uvs = {0, 0, 1, 0, 0, 1},
                     .indices = {0, 1, 2},
                     .material_of = {0}};
    if (!normals.empty())
    {
        m.normals.assign(normals.begin(), normals.end());
    }
    return m;
}

// The scene materials of the hand-built tables: matte (with texture 3, which reads u and v) and metal; their texture
// indices
const std::vector<gpu::material<float>> hand_materials{
    {.kind = gpu::material_kind::lambertian, .albedo = {0.5f, 0.5f, 0.5f}, .fuzz = 0, .refraction_index = 1},
    {.kind = gpu::material_kind::metal, .albedo = {0.8f, 0.8f, 0.8f}, .fuzz = 0.2f, .refraction_index = 1}};
const std::vector<gpu::material_texture> hand_textures{
    {.texture_index = 3, .needs_uv = true, .needs_exact_point = false},
    {.texture_index = -1, .needs_uv = false, .needs_exact_point = false}};

// =====================================================================================================================
// Suite optix-triangles
// =====================================================================================================================

// offset_origin's side and size rule, and the planted fault: the same check on the offset toward -n
GPU_TEST(optix_triangles, offset_origin)
{
    const std::int64_t n = ctx.samples(1'000'000);
    gpu_test::device_buffer<unsigned long long> tallies(offset_tallies);
    gpu_test::device_buffer<double> worst(gpu_test::test_threads);
    offset_kernel<0><<<gpu_test::test_blocks, gpu_test::test_block_size>>>(9101, gpu_test::per_thread(n),
                                                                           tallies.get(), worst.get());
    gpu_test::finish_launch("the offset_origin kernel");
    const std::vector<unsigned long long> t = tallies.download();
    const std::vector<double> w = worst.download();
    const std::int64_t drawn = gpu_test::drawn(n);
    ctx.count("offset-side", static_cast<std::int64_t>(t[0]), 0, drawn);
    ctx.count("offset-size", static_cast<std::int64_t>(t[1]), 0, drawn);
    ctx.count("offset-finite", static_cast<std::int64_t>(t[2]), 0, drawn);
    ctx.info("offset-size-largest",
             std::format("the largest move is {:.4f} of 2^-15 |p|max + 2^-15; {} coordinates below 1/32 and {} of "
                         "+-1/32 among {}",
                         *std::ranges::max_element(w), t[4], t[5], 3 * drawn));
    // The planted fault: offset_origin(p, -n) moves every point to the other side, and the side check must say so of
    // every pair: one it lets pass is a check that can't see a wrong side
    ctx.count("planted-reversed-normal", static_cast<std::int64_t>(t[3]), 0, drawn);
}

// triangle_surface against its rules in double on hand-built tables: random hits, then the shading normal's rules one
// by one, the outside of closed meshes under mirrors, and absorber_surface
GPU_TEST(optix_triangles, surface)
{
    host_tables t;
    t.materials = {scene_material(0), scene_material(1)};
    const int smooth = t.add_part(alternating(gpu_test::mesh_host::icosphere(1, true)));
    const int flat = t.add_part(alternating(gpu_test::mesh_host::icosphere(1, false)));
    const int wild = t.add_part(alternating(wild_mesh(9102)));
    const int ring = t.add_part(alternating(gpu_test::mesh_host::torus(1, 0.35, 12, 6)));
    const std::vector<int> random_instances{
        t.add_instance(smooth, identity),
        t.add_instance(wild, compose(translation(2, -1, 0.5), rotation({1, 2, 3}, 40))),
        t.add_instance(smooth, compose(translation(-3, 0, 1), rotation({0, 1, 1}, 25), scaling(-1.5, 1.5, 1.5))),
        t.add_instance(flat, compose(translation(0, 4, -2), rotation({1, 0, 1}, 70), scaling(0.2, 3, 1))),
        t.add_instance(wild, compose(translation(1, 1, 1), scaling(1, -0.7, 1.3), rotation({2, 1, 0}, 15))),
        t.add_instance(ring, compose(translation(400, -250, 555), rotation({0, 0, 1}, 33), scaling(100, 100, 100))),
        t.add_instance(wild, compose(translation(1e-3, 0, 0), scaling(1e-3, 1e-3, 1e-3))),
        t.add_instance(ring, scaling(-1, 1, 1))};
    // The shading normal's rules, each on the triangle of one_triangle() hit at (1/4, 1/4)
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 9> zero{};
    const std::array<float, 9> against{0, 0, -1, 0, 0, -1, 0, 0, -1};
    const std::array<float, 9> tilted{0.9f, 0, 0.43588989f, 0.9f, 0, 0.43588989f, 0.9f, 0, 0.43588989f};
    const std::array<float, 9> leaning{0, 0.6f, 0.8f, 0, 0.6f, 0.8f, 0, 0.6f, 0.8f};
    const std::array<float, 9> with_nan{nan, 0, 1, 0, 0, 1, 0, 0, 1};
    const int rule_parts[6] = {t.add_part(one_triangle()), t.add_part(one_triangle(zero)),
                               t.add_part(one_triangle(with_nan)), t.add_part(one_triangle(against)),
                               t.add_part(one_triangle(tilted)), t.add_part(one_triangle(leaning))};
    std::array<int, 6> rule_instances{};
    for (std::size_t i = 0; i < 6; ++i)
    {
        rule_instances[i] = t.add_instance(rule_parts[i], identity);
    }
    const int mirrored_leaning = t.add_instance(rule_parts[5], scaling(-1, 1, 1));
    // Closed meshes under mirrors: the outside stays outside
    const int ball = t.add_part(gpu_test::mesh_host::icosphere(2, true));
    const std::array<matrix, 2> mirrors{compose(translation(5, 0, 0), rotation({0, 1, 1}, 25), scaling(-1.5, 1.5, 1.5)),
                                        compose(translation(-5, 2, 0), scaling(0.6, -1.4, 1), rotation({1, 1, 0}, 50))};
    const std::array<int, 2> mirror_instances{t.add_instance(ball, mirrors[0]), t.add_instance(ball, mirrors[1])};
    const device_tables tables(t, hand_materials, hand_textures, 0);

    // Random hits
    host_random random(9103);
    std::vector<probe_sample> samples;
    const auto count = static_cast<std::size_t>(ctx.samples(10'000));
    for (std::size_t k = 0; k < count; ++k)
    {
        const int pick = random.below(static_cast<int>(random_instances.size()));
        samples.push_back(random_hit(random, t, random_instances[static_cast<std::size_t>(pick)]));
    }
    // The rules: (name, instance, direction, whether ns falls back to facing)
    struct rule
    {
        std::string_view name;
        int instance;
        dvec direction;
        bool falls_back;
    };
    const std::array<rule, 9> rules{{{"no-normals", rule_instances[0], {0.3, 0.2, -1}, true},
                                     {"zero-normals", rule_instances[1], {0.3, 0.2, -1}, true},
                                     {"nan-normal", rule_instances[2], {0.3, 0.2, -1}, true},
                                     {"against-the-face", rule_instances[3], {0.3, 0.2, -1}, true},
                                     {"away-from-the-ray", rule_instances[4], {0.95, 0, -0.31}, true},
                                     {"interpolated", rule_instances[5], {0.1, 0.2, -1}, false},
                                     {"back-hit-negated", rule_instances[5], {0.1, -0.2, 1}, false},
                                     {"mirror-front", mirrored_leaning, {-0.1, 0.2, -1}, false},
                                     {"mirror-back", mirrored_leaning, {-0.1, -0.2, 1}, false}}};
    const std::size_t first_rule = samples.size();
    for (const rule& r : rules)
    {
        probe_sample s;
        s.mode = 1;
        s.instance = r.instance;
        s.b1 = 0.25f;
        s.b2 = 0.25f;
        s.direction = to_float(r.direction);
        samples.push_back(s);
    }
    // Rays from outside the closed meshes: toward the centre, turned by up to ~30°
    const std::size_t first_outside = samples.size();
    for (std::size_t k = 0; k < 2000; ++k)
    {
        probe_sample s = random_hit(random, t, mirror_instances[k % 2]);
        samples.push_back(s);
    }
    std::vector<dvec> centres;
    for (std::size_t k = first_outside; k < samples.size(); ++k)
    {
        probe_sample& s = samples[k];
        const matrix& m = mirrors[(k - first_outside) % 2];
        const dvec centre = point(in_float(m), {0, 0, 0});
        const expected_surface e = expected(t, hand_textures, s.instance, s.primitive, s.b1, s.b2, {0, 0, 1});
        const dvec inward = unit(sub(centre, e.p));
        s.direction = to_float(unit(add(inward, mul(0.5 * random.uniform(), random.direction()))));
        centres.push_back(centre);
    }

    const std::vector<probe_answer> answers = surfaces_in_cuda(tables, samples);
    surface_tally random_tally;
    for (std::size_t k = 0; k < first_rule; ++k)
    {
        const probe_sample& s = samples[k];
        random_tally.add(expected(t, hand_textures, s.instance, s.primitive, s.b1, s.b2, of(s.direction)), answers[k],
                         false);
    }
    random_tally.report(ctx, "surface");

    for (std::size_t i = 0; i < rules.size(); ++i)
    {
        const probe_sample& s = samples[first_rule + i];
        const probe_answer& a = answers[first_rule + i];
        const expected_surface e = expected(t, hand_textures, s.instance, s.primitive, s.b1, s.b2, of(s.direction));
        // The reference must take the branch the case is made for, and the GPU must agree with it
        const bool reference_right = e.fallback == rules[i].falls_back && !e.grazing && !e.ns_close;
        const double dn = apart(of(a.normal), e.ns);
        const bool right =
            reference_right && dn <= 1e-5 && a.front == e.front && apart(of(a.outward), e.facing) <= 1e-5;
        ctx.count(std::format("rule-{}", rules[i].name), right ? 0 : 1, 0, 1);
        if (!right)
        {
            ctx.info(std::format("rule-{}", rules[i].name),
                     std::format("normal ({}, {}, {}), expected ({}, {}, {}); front {}, expected {}; the reference {} "
                                 "falls back",
                                 a.normal[0], a.normal[1], a.normal[2], e.ns[0], e.ns[1], e.ns[2], a.front, e.front,
                                 e.fallback ? "does" : "doesn't"));
        }
    }

    // Under a mirror, a ray from outside meets the front, and both normals point out of the closed mesh
    std::int64_t inside_out = 0;
    std::int64_t outside_tested = 0;
    for (std::size_t k = first_outside; k < samples.size(); ++k)
    {
        const probe_answer& a = answers[k];
        const dvec out = sub(of(a.p), centres[k - first_outside]);
        const expected_surface e =
            expected(t, hand_textures, samples[k].instance, samples[k].primitive, samples[k].b1, samples[k].b2,
                     of(samples[k].direction));
        if (e.grazing)
        {
            continue;
        }
        ++outside_tested;
        inside_out += a.front && dot(of(a.outward), out) > 0 && dot(of(a.normal), out) > 0 ? 0 : 1;
    }
    ctx.count("mirror-outside", inside_out, 0, outside_tested);

    // absorber_surface: diffuse_light black (the absorber's entry M), front_face false, from -1, normal -unit(d), p
    // the ray's origin
    const std::array<std::array<float, 6>, 4> rays{{{1.5f, -2.0f, 0.25f, 0.0f, 0.0f, -3.0f},
                                                     {-0.0f, 7.0f, 1e-3f, 0.3f, -0.4f, 1.2f},
                                                     {555.0f, 0.5f, -278.0f, -1.0f, 1.0f, 1e-6f},
                                                     {0.0f, 0.0f, 0.0f, 1e-3f, 2e-3f, -5e-4f}}};
    std::vector<float> words(18 * rays.size(), 0.0f);
    for (std::size_t k = 0; k < rays.size(); ++k)
    {
        std::copy(rays[k].begin(), rays[k].end(), words.begin() + 6 * static_cast<std::ptrdiff_t>(k));
    }
    const gpu_test::device_buffer<float> absorber_values(words);
    absorber_kernel<0><<<1, 32>>>(tables.params().meshes, absorber_values.get(), static_cast<int>(rays.size()));
    gpu_test::finish_launch("the absorber_surface kernel");
    const std::vector<float> out = absorber_values.download();
    std::int64_t wrong = 0;
    for (std::size_t k = 0; k < rays.size(); ++k)
    {
        const float* const w = out.data() + 6 * rays.size() + 12 * k;
        const dvec d{rays[k][3], rays[k][4], rays[k][5]};
        const bool same_origin = std::bit_cast<std::uint32_t>(w[0]) == std::bit_cast<std::uint32_t>(rays[k][0]) &&
                                 std::bit_cast<std::uint32_t>(w[1]) == std::bit_cast<std::uint32_t>(rays[k][1]) &&
                                 std::bit_cast<std::uint32_t>(w[2]) == std::bit_cast<std::uint32_t>(rays[k][2]);
        const bool right = same_origin && apart({w[3], w[4], w[5]}, mul(-1, unit(d))) <= 1e-6 && int_of(w[6]) == 0 &&
                           int_of(w[7]) == tables.absorber() && int_of(w[8]) == -1;
        wrong += right ? 0 : 1;
    }
    ctx.count("absorber", wrong, 0, std::ssize(rays));
}

// The baked twin: bake's world mesh under an identity instance against the mesh under the transform; under a mirror its
// corners 1 and 2 are swapped, so the twin of (b1, b2) is (b2, b1)
GPU_TEST(optix_triangles, baked_twin)
{
    const gpu::mesh_desc ball = gpu_test::mesh_host::icosphere(1, true);
    struct twin_case
    {
        std::string_view name;
        matrix m;
    };
    const std::array<twin_case, 3> cases{
        {{"mirror", compose(translation(1, -2, 3), rotation({0, 1, 1}, 25), scaling(-1.5, 1.5, 1.5))},
         {"mirror-stretched", compose(translation(-1, 0.5, 2), scaling(-0.6, 1.4, 1), rotation({1, 2, 0}, 35))},
         {"rotation", compose(translation(0.5, 0.5, -4), rotation({3, 1, 2}, 70))}}};
    host_random random(9104);
    for (const twin_case& c : cases)
    {
        host_tables t;
        t.materials = {scene_material(0), scene_material(1)};
        const int instanced = t.add_instance(t.add_part(ball), c.m);
        const int baked = t.add_instance(t.add_part(gpu_test::mesh_host::bake(ball, c.m)), identity);
        const device_tables tables(t, hand_materials, hand_textures, 0);
        const bool swapped = determinant(c.m) < 0;
        std::vector<probe_sample> samples;
        for (int k = 0; k < 2000; ++k)
        {
            probe_sample s = random_hit(random, t, instanced);
            probe_sample twin = s;
            twin.instance = baked;
            if (swapped)
            {
                std::swap(twin.b1, twin.b2);
            }
            samples.push_back(s);
            samples.push_back(twin);
        }
        const std::vector<probe_answer> answers = surfaces_in_cuda(tables, samples);
        std::int64_t point = 0;
        std::int64_t side = 0;
        std::int64_t normal = 0;
        std::int64_t compared = 0;
        double worst_normal = 0;
        for (std::size_t k = 0; k < samples.size(); k += 2)
        {
            const probe_answer& a = answers[k];
            const probe_answer& b = answers[k + 1];
            const expected_surface e = expected(t, hand_textures, samples[k].instance, samples[k].primitive,
                                                samples[k].b1, samples[k].b2, of(samples[k].direction));
            point += apart(of(a.p), of(b.p)) <= 1e-5 * e.scale ? 0 : 1;
            if (e.grazing)
            {
                continue;
            }
            ++compared;
            side += a.front == b.front && apart(of(a.outward), of(b.outward)) <= 1e-5 ? 0 : 1;
            if (!e.ns_close)
            {
                const double dn = apart(of(a.normal), of(b.normal));
                worst_normal = std::max(worst_normal, std::isnan(dn) ? infinite : dn);
                normal += dn <= 1e-5 ? 0 : 1;
            }
        }
        const std::string name{c.name};
        ctx.count("baked-" + name + "-point", point, 0, std::ssize(samples) / 2);
        ctx.count("baked-" + name + "-front-and-face", side, 0, compared);
        ctx.count("baked-" + name + "-shading-normal", normal, 0, compared);
        ctx.info("baked-" + name + "-shading-normal-largest", std::format("{:.3g}", worst_normal));
    }
}

// OptiX against the brute force in double: a level-3 icosphere and a 24×12 torus under instances
GPU_TEST(optix_triangles, brute_force)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = optix_for(ctx, "brute-force");
    if (!runtime)
    {
        return;
    }
    guarded(ctx, "brute-force", [&] {
        struct mesh_case
        {
            std::string_view name;
            gpu::mesh_desc mesh;
            matrix m;
            std::uint64_t seed;
        };
        const std::array<mesh_case, 3> cases{
            {{"icosphere", gpu_test::mesh_host::icosphere(3, false),
              compose(translation(0.5, -0.25, 0.3), rotation({1, 1, 2}, 37), scaling(1.5, 1.5, 1.5)), 9105},
             {"torus", gpu_test::mesh_host::torus(1, 0.4, 24, 12),
              compose(translation(-0.4, 0.2, 0.1), scaling(-1, 0.8, 1.2), rotation({2, 0, 1}, 58)), 9106},
             {"icosphere-identity", gpu_test::mesh_host::icosphere(3, false), identity, 9110}}};
        for (const mesh_case& c : cases)
        {
            const gpu::scene_desc scene = mesh_scene({c.mesh}, {{0, c.m}});
            const optix_upload upload(*runtime, scene);
            // The world triangles as the GPU's float transform places them, in double
            const matrix mf = in_float(c.m);
            const std::size_t triangles = c.mesh.indices.size() / 3;
            std::vector<double> corners(9 * triangles);
            dvec lower{infinite, infinite, infinite};
            dvec upper{-infinite, -infinite, -infinite};
            for (std::size_t i = 0; i < 3 * triangles; ++i)
            {
                const std::uint32_t v = c.mesh.indices[i];
                const dvec w =
                    point(mf, {c.mesh.positions[3 * v], c.mesh.positions[3 * v + 1], c.mesh.positions[3 * v + 2]});
                for (std::size_t a = 0; a < 3; ++a)
                {
                    corners[3 * i + a] = w[a];
                    lower[a] = std::min(lower[a], w[a]);
                    upper[a] = std::max(upper[a], w[a]);
                }
            }
            // From a sphere around the box at twice its diagonal, at points of the box grown by 5%
            const dvec centre = mul(0.5, add(lower, upper));
            const dvec half = mul(0.525, sub(upper, lower));
            const double distance = 2 * length(sub(upper, lower));
            host_random random(c.seed);
            std::vector<probe_sample> samples(static_cast<std::size_t>(ctx.samples(100'000)));
            for (probe_sample& s : samples)
            {
                const dvec origin = add(centre, mul(distance, random.direction()));
                const dvec target = {centre[0] + half[0] * random.between(-1, 1),
                                     centre[1] + half[1] * random.between(-1, 1),
                                     centre[2] + half[2] * random.between(-1, 1)};
                s.origin = to_float(origin);
                s.direction = to_float(sub(target, of(s.origin)));
            }
            const std::vector<probe_answer> answers = probe(ctx, *upload.scene, samples, c.seed);
            // The reference, on the GPU in double, from the same floats
            std::vector<float> rays(probe_in_words * samples.size(), 0.0f);
            for (std::size_t k = 0; k < samples.size(); ++k)
            {
                encode(samples[k], rays.data() + probe_in_words * k);
            }
            const gpu_test::device_buffer<float> device_rays(rays);
            const gpu_test::device_buffer<double> device_corners(corners);
            const gpu_test::device_buffer<brute_answer> device_answers(samples.size());
            const auto n = static_cast<int>(samples.size());
            brute_force_kernel<0><<<(n + 127) / 128, 128>>>(device_corners.get(), static_cast<int>(triangles),
                                                            device_rays.get(), n, 0.001, device_answers.get());
            gpu_test::finish_launch("the brute-force kernel");
            const std::vector<brute_answer> reference = device_answers.download();

            const int first_id = upload.triangle_base + 1;
            std::int64_t hits = 0;
            std::int64_t near_edge = 0;
            std::int64_t near_differ = 0;
            std::int64_t hit_or_miss = 0;
            std::int64_t triangle = 0;
            std::int64_t distance_off = 0;
            std::int64_t weights = 0;
            std::int64_t compared = 0;
            double worst_t = 0;       // |Δt|/t at |cos θ| >= 0.1
            double worst_grazing = 0; // |Δt|/t over its allowance below
            double worst_b = 0;
            double worst_b_share = 0; // the largest barycentric difference over its allowance
            for (std::size_t k = 0; k < samples.size(); ++k)
            {
                const brute_answer& r = reference[k];
                const probe_answer& a = answers[k];
                const probe_sample& sample = samples[k];
                const bool hit = a.index >= 0;
                const bool same = (r.triangle < 0 && !hit) ||
                                  (r.triangle >= 0 && hit && a.index == first_id + r.triangle && a.instance == 0 &&
                                   a.primitive == static_cast<unsigned int>(r.triangle));
                hits += r.triangle >= 0 ? 1 : 0;
                if (r.near_edge != 0)
                {
                    ++near_edge;
                    near_differ += same ? 0 : 1;
                    continue;
                }
                ++compared;
                if ((r.triangle >= 0) != hit)
                {
                    ++hit_or_miss;
                    continue;
                }
                if (!hit)
                {
                    continue;
                }
                if (!same)
                {
                    ++triangle;
                    continue;
                }
                // What float arithmetic leaves of the hit (decision 8): δ, the point's distance from the reference's,
                // along the ray as δ/(t·|d|·|cos θ|) of t, across the triangle as δ over its smallest altitude in
                // barycentric units
                const dvec d = of(sample.direction);
                const double reach = max_abs(of(sample.origin)) + r.t * max_abs(d);
                const double delta = 0x1p-16 * reach;
                const double cosine = std::max(r.cosine, 1e-12);
                // 10^-6 of t; a grazing hit's t loses digits as 1/|cos θ| (decision 1)
                const double dt = std::abs(a.t - r.t) / r.t;
                const double allowed =
                    std::max(1e-6 * std::max(1.0, 0.1 / cosine), delta / (r.t * length(d) * cosine));
                if (r.cosine >= 0.1)
                {
                    worst_t = std::isnan(dt) ? infinite : std::max(worst_t, dt);
                }
                else
                {
                    worst_grazing = std::isnan(dt) ? infinite : std::max(worst_grazing, dt / allowed);
                }
                distance_off += dt <= allowed ? 0 : 1;
                const double db = std::max(std::abs(a.b1 - r.b1), std::abs(a.b2 - r.b2));
                const double allowed_b = std::max(1e-5, delta / smallest_altitude(corners, r.triangle));
                worst_b = std::max(worst_b, db);
                worst_b_share = std::max(worst_b_share, db / allowed_b);
                weights += db <= allowed_b ? 0 : 1;
            }
            const std::string name{c.name};
            ctx.count(name + "-hit-or-miss", hit_or_miss, 0, compared);
            ctx.count(name + "-triangle", triangle, 0, compared);
            ctx.count(name + "-t", distance_off, 0, compared);
            ctx.count(name + "-barycentrics", weights, 0, compared);
            ctx.info(name + "-rays",
                     std::format("{} rays, {} hit; {} near an edge (left out), {} of them answered otherwise; the "
                                 "largest |dt|/t {:.3g} (|cos| >= 0.1), {:.3g} of the allowance at grazing hits; "
                                 "barycentrics {:.3g}, {:.3g} of their allowance",
                                 samples.size(), hits, near_edge, near_differ, worst_t, worst_grazing, worst_b,
                                 worst_b_share));
        }
    });
}

// --- The coplanar neighbour (decision 5) ----------------------------------------------------------------------------

// A tilted plane (layer 0) and its parallel copy (layer 1) at a gap of 4·2^-15·|p|∞, as one mesh: 10×10 cells a
// layer, two triangles a cell, counter-clockwise seen from +n
struct coplanar_planes
{
    static constexpr int cells = 10;
    static constexpr int layer_triangles = 2 * cells * cells;
    static constexpr int layer_vertices = (cells + 1) * (cells + 1);
    gpu::mesh_desc mesh;
    dvec centre{};
    dvec u{};
    dvec v{};
    dvec n{};
    double size = 0; // |p|∞ of the points the rays leave
    double gap = 0;
};

// The cells' edges along u and v around the centre, in units of |p|∞: finer where the rays start (|a|, |b| < 0.1)
constexpr std::array<double, coplanar_planes::cells + 1> coplanar_edges{-1.2, -0.5, -0.2, -0.1, -0.05, 0,
                                                                        0.05, 0.1,  0.2,  0.5,  1.2};

coplanar_planes coplanar(double size)
{
    coplanar_planes c;
    c.n = unit({0.30, 0.90, 0.25});
    c.u = unit(cross({0, 0, 1}, c.n));
    c.v = cross(c.n, c.u); // u × v = n
    c.centre = mul(size, {0.70, 1.00, 0.02});
    c.size = size;
    c.gap = 4 * 0x1p-15 * size;
    for (int layer = 0; layer < 2; ++layer)
    {
        for (std::size_t j = 0; j <= coplanar_planes::cells; ++j)
        {
            for (std::size_t i = 0; i <= coplanar_planes::cells; ++i)
            {
                const dvec p = add(add(c.centre, mul(size * coplanar_edges[i], c.u)),
                                   add(mul(size * coplanar_edges[j], c.v), mul(layer * c.gap, c.n)));
                const std::array<float, 3> f = to_float(p);
                c.mesh.positions.insert(c.mesh.positions.end(), f.begin(), f.end());
            }
        }
        for (std::uint32_t j = 0; j < coplanar_planes::cells; ++j)
        {
            for (std::uint32_t i = 0; i < coplanar_planes::cells; ++i)
            {
                const std::uint32_t v00 = static_cast<std::uint32_t>(layer * coplanar_planes::layer_vertices) +
                                          j * (coplanar_planes::cells + 1) + i;
                const std::uint32_t v10 = v00 + 1;
                const std::uint32_t v11 = v00 + coplanar_planes::cells + 2;
                const std::uint32_t v01 = v00 + coplanar_planes::cells + 1;
                c.mesh.indices.insert(c.mesh.indices.end(), {v00, v10, v11, v00, v11, v01});
                c.mesh.material_of.insert(c.mesh.material_of.end(), {0u, 0u});
            }
        }
    }
    return c;
}

// Where a ray leaves layer `layer` at plane coordinates (a, b) (units of |p|∞): its triangle, the point as
// triangle_surface would deliver it (interpolated from the float corners, rounded, then moved by up to 2 ulps in each
// coordinate), and the triangle's face normal from its float corners
struct start_point
{
    int triangle = 0;
    std::array<float, 3> p{};
    dvec ng{};
};

start_point start_on(const coplanar_planes& c, int layer, double a, double b, host_random& random)
{
    const auto cell_of = [](double x) {
        std::size_t i = 0;
        while (i + 2 < coplanar_edges.size() && coplanar_edges[i + 1] <= x)
        {
            ++i;
        }
        return i;
    };
    const std::size_t i = cell_of(a);
    const std::size_t j = cell_of(b);
    const double s = (a - coplanar_edges[i]) / (coplanar_edges[i + 1] - coplanar_edges[i]);
    const double t = (b - coplanar_edges[j]) / (coplanar_edges[j + 1] - coplanar_edges[j]);
    const bool lower = s >= t; // triangle (v00, v10, v11); else (v00, v11, v01)
    start_point sp;
    sp.triangle = layer * coplanar_planes::layer_triangles +
                  2 * static_cast<int>(j * coplanar_planes::cells + i) + (lower ? 0 : 1);
    const std::size_t first = 3 * static_cast<std::size_t>(sp.triangle);
    std::array<dvec, 3> corner{};
    for (std::size_t k = 0; k < 3; ++k)
    {
        const std::uint32_t vertex = c.mesh.indices[first + k];
        corner[k] = {c.mesh.positions[3 * vertex], c.mesh.positions[3 * vertex + 1], c.mesh.positions[3 * vertex + 2]};
    }
    const std::array<double, 3> w =
        lower ? std::array<double, 3>{1 - s, s - t, t} : std::array<double, 3>{1 - t, s, t - s};
    const dvec p = add(add(mul(w[0], corner[0]), mul(w[1], corner[1])), mul(w[2], corner[2]));
    sp.p = to_float(p);
    for (float& x : sp.p)
    {
        x = ulps_away(x, random.below(5) - 2);
    }
    sp.ng = unit(cross(sub(corner[1], corner[0]), sub(corner[2], corner[0])));
    return sp;
}

// Rays leaving one layer for the other at 0.01° to 90° off the plane (log-uniform), half each way; `reversed` gives
// them the face's other side (the planted fault: they start behind the face they leave)
std::vector<probe_sample> leaving_rays(const coplanar_planes& c, int triangle_base, std::size_t count,
                                       host_random& random, bool reversed = false, bool one_way = false)
{
    std::vector<probe_sample> samples;
    for (std::size_t k = 0; k < count; ++k)
    {
        const int layer = one_way ? 0 : static_cast<int>(k % 2);
        const start_point sp = start_on(c, layer, random.between(-0.1, 0.1), random.between(-0.1, 0.1), random);
        const double degrees = std::exp(random.between(std::log(0.01), std::log(90.0)));
        const double elevation = degrees * std::numbers::pi / 180;
        const double around = 2 * std::numbers::pi * random.uniform();
        const double toward = layer == 0 ? 1 : -1; // layer 0 looks up at its neighbour, layer 1 down
        const dvec along = add(mul(std::cos(around), c.u), mul(std::sin(around), c.v));
        probe_sample s;
        s.origin = sp.p;
        s.direction = to_float(add(mul(std::cos(elevation), along), mul(toward * std::sin(elevation), c.n)));
        s.from = triangle_base + 1 + sp.triangle;
        s.inward = false;
        // The side the ray came from: layer 0 was hit from above (its front, +n), layer 1 from below (its back)
        s.facing = to_float(mul(toward * (reversed ? -1 : 1), sp.ng));
        samples.push_back(s);
    }
    return samples;
}

// Rays leaving the planes, through optix_tracer's offset origin: none may hit the plane it leaves, every one must hit
// the neighbour; and the through-rule
GPU_TEST(optix_triangles, coplanar)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = optix_for(ctx, "coplanar");
    if (!runtime)
    {
        return;
    }
    guarded(ctx, "coplanar", [&] {
        host_random random(9107);
        for (const double size : {1.0, 8.0, 555.0})
        {
            const coplanar_planes c = coplanar(size);
            const optix_upload upload(*runtime, mesh_scene({c.mesh}, {{0, identity}}));
            const int first = upload.triangle_base + 1;
            const std::vector<probe_sample> samples =
                leaving_rays(c, upload.triangle_base, static_cast<std::size_t>(ctx.samples(100'000)), random);
            const std::vector<probe_answer> answers = probe(ctx, *upload.scene, samples, 9107);
            std::int64_t own = 0;      // the very triangle left
            std::int64_t own_plane = 0; // another triangle of the plane left
            std::int64_t missed = 0;   // anything but the neighbour
            for (std::size_t k = 0; k < samples.size(); ++k)
            {
                const int layer = static_cast<int>(k % 2);
                const int other_first = first + (1 - layer) * coplanar_planes::layer_triangles;
                const int index = answers[k].index;
                own += index == samples[k].from ? 1 : 0;
                own_plane += index != samples[k].from && index >= first + layer * coplanar_planes::layer_triangles &&
                                     index < first + (layer + 1) * coplanar_planes::layer_triangles
                                 ? 1
                                 : 0;
                missed += index >= other_first && index < other_first + coplanar_planes::layer_triangles ? 0 : 1;
            }
            const std::string name = std::format("coplanar-{}", size);
            ctx.count(name + "-triangle-left", own, 0, std::ssize(samples));
            ctx.count(name + "-plane-left", own_plane, 0, std::ssize(samples));
            ctx.count(name + "-neighbour-missed", missed, 0, std::ssize(samples));
            ctx.info(name + "-gap",
                     std::format("the neighbour {:.6g} away (4 2^-15 {}), the rays 0.01 to 90 degrees off the plane, "
                                 "half each way",
                                 c.gap, size));

            // The through-rule: a ray leaving a triangle inward is the absorber's, untraced
            std::vector<probe_sample> inward = leaving_rays(c, upload.triangle_base, 32, random, false, true);
            for (probe_sample& s : inward)
            {
                s.direction = {-s.direction[0], -s.direction[1], -s.direction[2]};
                s.inward = true;
            }
            const std::vector<probe_answer> absorbed = probe(ctx, *upload.scene, inward, 9107);
            std::int64_t not_absorbed = 0;
            for (const probe_answer& a : absorbed)
            {
                not_absorbed += a.index == upload.triangle_base && a.t == 0.0f ? 0 : 1;
            }
            ctx.count(name + "-through-rule", not_absorbed, 0, std::ssize(absorbed));
        }

        // The planted fault: memo.facing on the wrong side, so the offset moves each origin behind the face it leaves
        const coplanar_planes c = coplanar(8);
        const optix_upload upload(*runtime, mesh_scene({c.mesh}, {{0, identity}}));
        const std::vector<probe_sample> samples =
            leaving_rays(c, upload.triangle_base, static_cast<std::size_t>(ctx.samples(10'000)), random, true, true);
        const std::vector<probe_answer> answers = probe(ctx, *upload.scene, samples, 9107);
        std::int64_t self = 0;
        for (const probe_answer& a : answers)
        {
            self += a.index >= upload.triangle_base + 1 &&
                            a.index < upload.triangle_base + 1 + coplanar_planes::layer_triangles
                        ? 1
                        : 0;
        }
        const auto rays = std::ssize(answers);
        ctx.count("planted-reversed-face", 100 * self >= 99 * rays ? 0 : 1, 0, 1);
        ctx.info("planted-reversed-face", std::format("{} of {} rays started behind the face hit the plane they leave "
                                                      "(at least 99% must)",
                                                      self, rays));
    });
}

// triangle_surface on the tables meshes.cu uploads: the same reference, the ids from the upload's triangle_base
GPU_TEST(optix_triangles, surface_upload)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = optix_for(ctx, "surface-upload");
    if (!runtime)
    {
        return;
    }
    guarded(ctx, "surface-upload", [&] {
        gpu::mesh_desc quad{.positions = {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0},
                            .normals = {0.1f, 0, 0.995f, 0, 0.1f, 0.995f, -0.1f, 0, 0.995f, 0, -0.1f, 0.995f},
                            .uvs = {0, 0, 1, 0, 1, 1, 0, 1},
                            .indices = {0, 1, 2, 0, 2, 3},
                            .material_of = {0, 1}};
        gpu::mesh_desc flat = gpu_test::mesh_host::icosphere(1, false);
        std::ranges::fill(flat.material_of, 1u);
        const std::vector<gpu::mesh_desc> meshes{gpu_test::mesh_host::icosphere(2, true),
                                                 alternating(gpu_test::mesh_host::torus(1, 0.35, 16, 8)), quad, flat};
        const std::vector<std::pair<int, matrix>> placements{
            {0, compose(translation(-3, 0, 1), rotation({0, 1, 1}, 25), scaling(-1.5, 1.5, 1.5))},
            {1, compose(translation(0, 4, -2), rotation({1, 0, 1}, 70), scaling(0.5, 2, 1))},
            {2, compose(translation(2, -1, 0.5), rotation({1, 2, 3}, 40))},
            {3, translation(0, -3, 0)},
            {0, compose(translation(3, 3, 3), scaling(0.6, -1.4, 1), rotation({1, 1, 0}, 50))}};
        const optix_upload upload(*runtime, mesh_scene(meshes, placements));
        // The reference's view of the same meshes: the upload's ids; device_scene folds every texture to a color
        host_tables t;
        t.triangle_base = upload.triangle_base;
        t.next_id = upload.triangle_base + 1;
        t.materials = {scene_material(0), scene_material(1)};
        for (const gpu::mesh_desc& m : meshes)
        {
            t.add_part(m);
        }
        for (const auto& [mesh, m] : placements)
        {
            t.add_instance(mesh, m);
        }
        const std::vector<gpu::material_texture> folded(2, {.texture_index = -1, .needs_uv = false,
                                                            .needs_exact_point = false});
        host_random random(9108);
        std::vector<probe_sample> samples;
        for (int k = 0; k < 2000; ++k)
        {
            samples.push_back(random_hit(random, t, random.below(static_cast<int>(placements.size()))));
        }
        const std::vector<probe_answer> answers = probe(ctx, *upload.scene, samples, 9108);
        surface_tally tally;
        for (std::size_t k = 0; k < samples.size(); ++k)
        {
            const probe_sample& s = samples[k];
            tally.add(expected(t, folded, s.instance, s.primitive, s.b1, s.b2, of(s.direction)), answers[k], true);
        }
        tally.report(ctx, "upload");
    });
}

// P7: the world ray an intersection program reads, under the IAS root's identity instance, is the ray traced
GPU_TEST(optix_triangles, world_ray)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = optix_for(ctx, "world-ray");
    if (!runtime)
    {
        return;
    }
    guarded(ctx, "world-ray", [&] {
        gpu::scene_desc cornell = gpu_test::mesh_host::built_scene("cornell");
        cornell.lights.clear(); // test_scene.h's tables: no lights list (the probe reads none of the scene)
        cornell.camera.sampling = gpu::sampling_strategy::material;
        const optix_upload upload(*runtime, cornell, {.force_instance_root = true, .probe_intersections = true});
        host_random random(9109);
        std::vector<probe_sample> samples;
        // From random points inside the box toward its back half (z growing): every one meets some wall's box. The
        // box is open on the camera's side, z = 0, so a ray heading there could leave it through no box at all
        for (int k = 0; k < 4096; ++k)
        {
            probe_sample s;
            s.origin = to_float({random.between(5, 550), random.between(5, 550), random.between(5, 550)});
            dvec d = random.direction();
            d[2] = std::abs(d[2]);
            s.direction = to_float(d);
            samples.push_back(s);
        }
        // ±0 in origins and directions: from the centre, and from points on the walls' planes into the box
        const float z = 0.0f;
        const float nz = -0.0f;
        const std::array<std::array<float, 3>, 5> origins{
            {{278.0f, 278.0f, 278.0f}, {nz, 278.0f, 278.0f}, {278.0f, nz, 278.0f}, {278.0f, 278.0f, nz},
             {z, 300.5f, 100.25f}}};
        const std::array<std::array<float, 3>, 12> directions{{{1, z, nz},
                                                              {1, nz, z},
                                                              {1, nz, nz},
                                                              {nz, 1, nz},
                                                              {z, 1, nz},
                                                              {nz, nz, 1},
                                                              {0.6f, nz, 0.8f},
                                                              {nz, 0.6f, 0.8f},
                                                              {0.8f, 0.6f, nz},
                                                              {0.36f, 0.48f, 0.8f},
                                                              {1e-3f, 1, nz},
                                                              {nz, 1e-3f, 1}}};
        for (const auto& o : origins)
        {
            for (const auto& d : directions)
            {
                probe_sample s;
                s.origin = o;
                s.direction = d;
                samples.push_back(s);
            }
        }
        // Subnormal coordinates (INFO): a reconstruction that flushes them to 0 shows here first
        const std::size_t first_subnormal = samples.size();
        for (const std::uint32_t bits : {0x00011000u, 0x80011000u, 0x00000007u, 0x80000400u})
        {
            const float tiny = std::bit_cast<float>(bits);
            probe_sample s;
            s.origin = {278.0f, 278.0f, 278.0f};
            s.direction = {tiny, 1, -tiny};
            samples.push_back(s);
            s.direction = {0.6f, tiny, 0.8f};
            samples.push_back(s);
        }
        std::vector<unsigned long long> words;
        probe(ctx, *upload.scene, samples, 9109, true, &words);
        std::int64_t unprobed = 0;
        std::int64_t differ = 0;
        std::int64_t unsteady = 0;
        std::int64_t object_same = 0;
        std::int64_t probed = 0;
        std::int64_t subnormal_differ = 0;
        unsigned long long calls = 0;
        for (std::size_t k = 0; k < samples.size(); ++k)
        {
            const unsigned long long* const w = words.data() + 16 * k;
            const probe_sample& s = samples[k];
            const std::array<float, 6> traced{s.origin[0],    s.origin[1],    s.origin[2],
                                              s.direction[0], s.direction[1], s.direction[2]};
            bool same = true;
            bool object = true;
            for (std::size_t i = 0; i < 6; ++i)
            {
                same = same && w[i] == std::bit_cast<std::uint32_t>(traced[i]);
                object = object && w[6 + i] == std::bit_cast<std::uint32_t>(traced[i]);
            }
            if (k >= first_subnormal)
            {
                subnormal_differ += w[12] > 0 && same ? 0 : 1;
                continue;
            }
            if (w[12] == 0)
            {
                ++unprobed;
                continue;
            }
            ++probed;
            calls += w[12];
            differ += same ? 0 : 1;
            unsteady += w[13] > 0 ? 1 : 0;
            object_same += object ? 1 : 0;
        }
        ctx.count("world-ray-probed", unprobed, 0, static_cast<std::int64_t>(first_subnormal));
        ctx.count("world-ray-bits", differ, 0, probed);
        ctx.count("world-ray-steady", unsteady, 0, probed); // every run of the program for one ray reads one ray
        ctx.info("world-ray",
                 std::format("{} rays reached the probe, {} times in all; their object rays (the identity applied) "
                             "equal the traced floats bit for bit in {}; of the {} rays with subnormal coordinates, "
                             "{} read otherwise or unprobed",
                             probed, calls, object_same, samples.size() - first_subnormal, subnormal_differ));
    });
}

// The nearest of a ray's crossings with world triangles and spheres, in double: the host's pick
double nearest_crossing(const dvec& o, const dvec& d, std::span<const std::array<dvec, 3>> triangles,
                        std::span<const std::pair<dvec, double>> spheres)
{
    double best = infinite;
    for (const std::array<dvec, 3>& c : triangles)
    {
        const dvec e1 = sub(c[1], c[0]);
        const dvec e2 = sub(c[2], c[0]);
        const dvec pv = cross(d, e2);
        const double det = dot(e1, pv);
        if (det == 0)
        {
            continue;
        }
        const dvec tv = sub(o, c[0]);
        const double u = dot(tv, pv) / det;
        const dvec qv = cross(tv, e1);
        const double v = dot(d, qv) / det;
        const double t = dot(e2, qv) / det;
        if (u >= 0 && v >= 0 && u + v <= 1 && t > 0.001)
        {
            best = std::min(best, t);
        }
    }
    for (const auto& [centre, radius] : spheres)
    {
        const dvec oc = sub(centre, o);
        const double a = dot(d, d);
        const double h = dot(d, oc);
        const double disc = h * h - a * (dot(oc, oc) - radius * radius);
        if (disc >= 0)
        {
            for (const double t : {(h - std::sqrt(disc)) / a, (h + std::sqrt(disc)) / a})
            {
                if (t > 0.001)
                {
                    best = std::min(best, t);
                    break;
                }
            }
        }
    }
    return best;
}

// session::pick_distance (Step 91's region of renderer.cu, __raygen__pick): a triangle mesh's front and back, a books'
// sphere, a miss
GPU_TEST(optix_triangles, pick)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = optix_for(ctx, "pick");
    if (!runtime)
    {
        return;
    }
    guarded(ctx, "pick", [&] {
        const gpu::mesh_desc quad{.positions = {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0},
                                  .normals = {},
                                  .uvs = {},
                                  .indices = {0, 1, 2, 0, 2, 3},
                                  .material_of = {0, 0}};
        const matrix m = compose(translation(0.3, -0.2, -2), rotation({1, 1, 0}, 30), scaling(2, 1.5, 1));
        gpu::scene_desc scene = mesh_scene({quad}, {{0, m}});
        scene.spheres = {{.center = {5, 0, 0}, .radius = 0.5, .material = 0}};
        gpu::session session(scene, {.optix_tracer = true, .optix = runtime});
        const matrix mf = in_float(m);
        std::vector<std::array<dvec, 3>> triangles;
        for (std::size_t t = 0; t < 2; ++t)
        {
            std::array<dvec, 3> c{};
            for (std::size_t k = 0; k < 3; ++k)
            {
                const std::uint32_t v = quad.indices[3 * t + k];
                c[k] = point(mf, {quad.positions[3 * v], quad.positions[3 * v + 1], quad.positions[3 * v + 2]});
            }
            triangles.push_back(c);
        }
        const std::array<std::pair<dvec, double>, 1> spheres{{{{5, 0, 0}, 0.5}}};
        struct pick_case
        {
            std::string_view name;
            dvec origin;
            dvec toward; // a point the ray passes, or with `direction` the direction itself
            bool direction;
        };
        const std::array<pick_case, 4> cases{{{"front", {0, 0, 5}, point(mf, {0.3, -0.4, 0}), false},
                                              {"back", {0.5, 0.5, -8}, point(mf, {-0.5, 0.5, 0}), false},
                                              {"sphere", {5, 0, 5}, {5, 0, 0}, false},
                                              {"miss", {0, 0, 5}, {0, 1, 0}, true}}};
        for (const pick_case& c : cases)
        {
            const dvec d = unit(c.direction ? c.toward : sub(c.toward, c.origin));
            const double reference = nearest_crossing(c.origin, d, triangles, spheres);
            const double t = session.pick_distance({c.origin[0], c.origin[1], c.origin[2]}, {d[0], d[1], d[2]});
            const std::string name = std::format("pick-{}", c.name);
            if (std::isinf(reference))
            {
                ctx.count(name, std::isinf(t) && t > 0 ? 0 : 1, 0, 1);
                ctx.info(name, std::format("{}, expected infinity (no surface)", t));
            }
            else
            {
                ctx.exact(name, t / reference, 1, 1e-6);
            }
        }
    });
}

// The render path inside a closed mesh: cornell-triangles' tall block seen from inside through gpu::render, as
// RayTracingGPU renders it (decision 9). No ray leaves the block and no light enters it, so every path traces max_depth
// rays and finds nothing: -p material only bounces, -p mis also aims shadow rays at the lamp, which the block's faces
// (or the through-rule, for one aimed through the face it leaves) must stop.
GPU_TEST(optix_triangles, closed_block)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = optix_for(ctx, "closed-block");
    if (!runtime)
    {
        return;
    }
    guarded(ctx, "closed-block", [&] {
        gpu::scene_desc scene = gpu_test::mesh_host::built_scene("cornell-triangles");
        gpu::camera_desc& cam = scene.camera;
        cam.image_width = 8;
        cam.image_height = 8;
        cam.sqrt_spp = 4;
        cam.max_depth = 50;
        cam.center = {355, 278, 278};
        cam.pixel00_loc = {356, 278.07, 277.93};
        cam.pixel_delta_u = {0, 0, 0.02};
        cam.pixel_delta_v = {0, -0.02, 0};
        cam.defocus = false;
        const std::int64_t full = std::int64_t{8} * 8 * 16 * 50;
        const std::array<std::pair<std::string_view, gpu::light_combination>, 2> strategies{
            {{"material", gpu::light_combination::bsdf_only}, {"mis", gpu::light_combination::power_heuristic}}};
        for (const auto& [strategy, combination] : strategies)
        {
            scene.quality.combination = combination;
            const gpu::render_result r = gpu::render(scene, {.seed = 9113, .optix_tracer = true, .optix = runtime});
            const auto rays = static_cast<std::int64_t>(r.rays);
            const std::int64_t lit = std::ranges::count_if(r.pixels, [](double v) { return v != 0; }); // NaN too
            const std::string name = std::format("closed-block.{}", strategy);
            ctx.info(name, std::format("{} of the {} rays of full paths ({:.3f} per sample), {} shadow rays; {} of {} "
                                       "values above 0, the largest {:.3g}",
                                       rays, full, static_cast<double>(rays) / (8 * 8 * 16), r.shadow_rays, lit,
                                       r.pixels.size(), *std::ranges::max_element(r.pixels)));
            ctx.count(name + ".rays-missing", full - rays, full / 100, full);
            ctx.count(name + ".lit", lit, 1, static_cast<std::int64_t>(r.pixels.size()));
        }
    });
}
}
