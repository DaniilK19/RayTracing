// Suite optix-triangles (Step 91): triangle meshes on the GPU, tested against their specification: every reference
// here is the test's own, in double, written from the interfaces (mesh_tables.h's layout, the rules the comments of
// triangles.cuh, optix_tracer.cuh, mesh_scene.cuh and meshes.cu state) rather than from the code behind them; the
// two suites' refusals alone name the causes in meshes.cu's messages:
//   offset-origin   offset_origin on 10^6 (point, normal) pairs at scales 10^-3 to 10^4, in a plain CUDA kernel: the
//                   point moved strictly onto the normal's side of the plane through p (evaluated in double), each
//                   coordinate by at most 2^-15·|p|∞ + 2^-15 (the offset's bound without a reach); planted: the same
//                   check on offset_origin(p, -n), whose every point it must refuse;
//   surface         triangle_surface at 10^4 random hits (nine instances: mirrors, non-uniform scales, 10^-3 and 10^2
//                   scales, one untranslated; smooth, flat and wild normals) on tables laid out by the test as
//                   mesh_tables.h describes them, against the surface rules recomputed in double: p within 10^-5 of the
//                   coordinates' size, the face normal and the shading normal within 10^-5, the side, u and v, the
//                   material, its texture and the id; each rule of the shading normal on a case of its own; the outside
//                   of a closed mesh under mirroring instances; memo.reach within 10^-6 of its reference, each of its
//                   two terms deciding somewhere (decision 14); absorber_surface's fields;
//   baked-twin      a mesh baked by module meshes (world vertices, corners 1 and 2 swapped under a mirror) against the
//                   same mesh under the instance: the same point, face side and front, and the same shading normal;
//   brute-force     OptiX: 10^5 rays at each of three meshes (a level-3 icosphere under a turning, scaling instance and
//                   under the identity, a 24×12 torus mirrored and stretched) against every triangle in double: the
//                   same answer for every ray not near an edge (decision 1), a hit where the reference misses
//                   included; t within 10^-6 relative and barycentrics within 10^-5, or float precision (decision 8);
//                   the id triangle_base + 1 + primitive; the near-edge rays INFO;
//   coplanar        OptiX: rays leaving random triangles of a tilted plane that has a parallel neighbour at a gap of
//                   4·2^-15·|p|∞, at |p|∞ = 1, 8 and 555, 0.01° to 90° off the plane, through optix_tracer (its offset
//                   origin, hardware tmin 0): no hit of the plane left, no miss of the neighbour, in both directions;
//                   the through-rule's absorber for rays going into the face; planted: the face's side reversed, so
//                   that the rays start behind it and must hit it; and at |p|∞ = 8 the step sized by a reach given
//                   (memo.reach, as traverse passes it): 4·|p|∞, half the gap, changes nothing, 16·|p|∞, twice the gap,
//                   starts every ray beyond the neighbour, going away from both planes, so that it must meet nothing;
//   surface-upload  OptiX: triangle_surface on the tables meshes.cu uploads (the instances' ids, parts, transforms and
//                   inverse transposes, the materials; two meshes off their origins, whose centres the upload moves
//                   into turning, mirroring, non-uniform placements) against the same double reference, which keeps
//                   every mesh's own coordinates; memo.reach as in surface, from the upload's centred corners
//                   (decision 14);
//   world-ray       P7: the world ray __intersection__probe reads under an identity instance (cornell, the IAS root
//                   forced) is the ray traced, bit for bit, -0 included;
//   pick            session::pick_distance on a triangle mesh's front and back and on a books' sphere: t within 10^-6
//                   (relative) of the host's, ∞ on a miss;
//   closed-block    gpu::render (__raygen__render's loop, the session's upload) from a camera inside cornell-triangles'
//                   tall block, a closed box of 12 triangles, under -p material and -p mis: every path traces its
//                   max_depth rays, since none can leave the block, and the image is black (decision 9);
//   closed-corner   the same inside a closed cube of 555 near its corner (0, 0, 0), where the paths start on floor
//                   triangles that reach 555 at points where |p|∞ < 1: offset_origin's reach (decision 12);
//   refusals        the upload's refusals, each a std::runtime_error naming its cause, before any launch: glass, a lamp
//                   and smoke worn by a triangle, an index past the vertices, a material past the set, a corner that
//                   isn't finite, a NaN and a singular placement; and a NaN in a vertex no triangle uses, accepted.
// Suite optix-pbr (Step 92): glTF's metallic-roughness materials, from the same specification:
//   mapping         triangle_surface's glTF branch in a plain CUDA kernel, on hand-made materials (factors alone, past
//                   [0, 1], at the coat's floors, half metal on both sides of the lobe, lobe = metallic, and textures
//                   made here as meshes.cu makes them): the slot's material and side entry equal the mapping's table
//                   computed in float, bit for bit; the slot is scratch_base + the pixel, the texture -1, wears_gltf
//                   true; a scene material's hit leaves its slot alone; and the hardware's sRGB decode against
//                   IEC 61966-2-1, within half a byte's step (decision 13);
//   lobe-comparison the mapping's comparison lobe < metal in a plain CUDA kernel, the lobe drawn by the test: P(metal)
//                   = metallic over 10^6 numbers at metallic 0.3, binomial |z| <= 4; planted: the same count against
//                   0.31, which must be rejected;
//   lobe-draw       OptiX: the draw itself, closest() through the probe's mode 0: 2^18 rays straight down at a glTF
//                   triangle of metallic 0.3, one per launch index and generator, every one a hit of it, its lobe the
//                   generator's first number bit for bit, the share below 0.3f binomial |z| <= 4, planted 0.31 rejected
//                   (decision 15); a ray at a triangle wearing a scene material and a ray that misses: lobe exactly 0;
//   textures        OptiX: the texture objects meshes.cu makes, read through its upload and surface_of: every byte of
//                   a base color decoded within half a byte's step of IEC 61966-2-1 (the largest INFO),
//                   metallic-roughness read linearly, bilinear at texel centres equal to the texel within 10^-6 (and to
//                   NEAREST's), (0, 0) the top-left texel, the wrap modes (repeat, mirrored repeat, clamp) at u and
//                   v = -0.25 and 1.25, glTF's NEAREST; the mapping end to end on mesh_material_descs, bit for bit;
//                   and, once the probe has grown the tables past the 8×8 camera's 64 slots, a triangle wearing a
//                   scene material reads that material and its side entry, and the absorber (the probe's mode 2) its
//                   black lamp and no side entry;
//   refusals        the upload's refusals of glTF's materials and textures, each a std::runtime_error naming its cause,
//                   before any launch: a base color below 0 and NaN, metallic 1.5, roughness -0.1, a texture index
//                   past the set and one of -2, a scene material of -2, a base color texture that isn't sRGB and a
//                   metallic-roughness one that is; a texture missing, 0 texels wide, past the GPU's width, with a
//                   wrap mode glTF doesn't have, or with 12 bytes for 2×2 texels; and the scene they come from,
//                   accepted.
// Suite mesh-cache (Step 93), host only (GpuTests links gpu_scene; tests/mesh_services.cpp holds the modules' side):
//   still-life      two builds of still-life share their mesh_desc and texture pointers and read nothing (each model
//                   converted once per load); after the vase's file gets a new write time (put back afterwards) the
//                   vase alone reads it again, and still-life then reads nothing, the bust's pointers the first
//                   build's and the vase's the reload's (the stale entry gone, not the bust); with the time put back
//                   still-life reads the vase alone;
//   second-model    still-life's material offset: every triangle of the bust and of the vase (told apart by where
//                   they stand) wears one of its own model's materials as load_gltf reads them from the file, the
//                   factors and the textures' texels, base color sRGB and metallic-roughness linear.
// The OptiX checks reach the GPU through the fixture's runtime (optix_fixture.h): an upload of a scene_desc the test
// makes (a gpu::mesh_set of meshes from module meshes, through tests/mesh_services.h), and launch_test of the check
// module's __raygen__probe_mesh. Its creation and first compiles are left out of the 10 s budget. closed-block and
// closed-corner render through the program's own raygen instead: the probe traces one ray per launch, where the paths
// of a raygen's loop can go wrong in ways one ray can't show (decision 9).
//
// The probe of meshes, __raygen__probe_mesh (tests/optix_check_programs.cu, the probes' block): one sample per launch
// index k = y·width + x of a launch of n = width·height, 20 words in at params.test.values[20k] (floats; an int as its
// float's bits), 41 out at values[20n + 41k]:
//   in   0-2 the ray's origin, 3-5 its direction, 6 its time, 7 spawn.from, 8 spawn.inward (0 or 1), 9-11 memo.facing,
//        12 the mode: 0 traces the ray with optix_tracer::closest (sample_generator(test.seed, k, 0)); 1 traces
//        nothing and asks optix_tracer::surface_of for the triangle hit of 13 the instance, 14 the primitive, 15-16 b1
//        and b2, 17 memo.lobe, in scratch slot scratch_base + k; 2 traces nothing and asks surface_of for the
//        absorber's id; 18 memo.reach, the reach of the face a ray of mode 0 leaves;
//   out  0 the hit's index, 1 its t, 2 entering (0 or 1); the side record: 3 instance, 4 primitive, 5-6 b1, b2,
//        7 lobe; modes 1 and 2: 8-10 rec.p, 11-13 rec.normal, 14 front_face, 15-17 outward, 18-20 texture_point, 21 u,
//        22 v, 23 material, 24 texture_index, 25 from, 26-28 memo.facing, 29 = 1; then what the integrator reads at
//        s.material: 30 the material's kind, 31-33 albedo, 34 fuzz, 35 refraction index, 36 its side entry's kind
//        (-1: no side entries), 37 alpha, 38 roughness. (39: the CUDA twin's wears_gltf.) 40 memo.reach (mode 1:
//        triangle_surface's).
// The surface checks' plain kernel writes the same 41 words from the same 20 (mode 1 only), so one decoder reads both.
// The probe's memo.reach is 0 but in coplanar's reach passes: a ray it traces from a triangle then starts off the face
// by its point's size alone (the offset-origin check's form), which the coplanar planes, about as large as their
// points' coordinates, allow.
// Under P7's probe_intersections the probe intersection program keeps 16 words of params.test.counts per sample (its
// own block of optix_check_programs.cu says which).
//
// Decisions where the specification leaves room:
//   1. "Farther than 10^-4 (relative) from an edge" is barycentric: a ray is near an edge when the reference finds a
//      triangle crossed at t > t_min, no farther than the nearest hit (anywhere, for a miss), whose smallest
//      barycentric coordinate is within max(10^-4, δ/altitude) of 0 on either side, δ decision 8's float allowance at
//      that crossing and the altitude the triangle's smallest: the band the barycentrics themselves are held to, so a
//      ray that rounding may carry across an edge isn't compared. A miss without such a crossing is compared as any
//      ray is: the hardware must miss too. t's 10^-6 is relative to t, and for grazing hits (|cos θ| < 0.1
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
//      a ray toward it could meet no box), 60 from its centre and from points on the planes x = 0, y = 0 and z = 0 (5
//      origins, 12 directions) with ±0 coordinates in the origin and the direction, and 8 with subnormal direction
//      coordinates (INFO: a driver that flushes them to 0 reconstructs no other ray). Every ray from inside must reach
//      the probe.
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
//      under a turning, scaling instance and 67 on the torus's thin triangles under a mirroring one.) The near-edge
//      band of decision 1 takes the same δ at each crossing.
//   9. The closed block: the camera at (355, 278, 278), 7 units inside the tall block's face, looking along +x; 8×8
//      pixels, 16 samples each, max_depth 50. It was added for two defects the probe's single rays passed: nvcc hoisted
//      the barycentrics' read out of __raygen__render's loop, before any traversal (optix_tracer.cuh's decode_hit), and
//      offset_origin's step off the floor (y = 0, x and z in the hundreds) fell inside the triangle test's rounding
//      (triangles.cuh), so paths met the floor again from below. Paths then traced 3.5 and 46.5 of their 50 rays, and
//      the first found the lamp. A path may still slip through an edge where the hit point rounds outside its face,
//      which the books' quads allow too: the rays may fall short of the full count by 1% (a path ending at its first
//      bounce costs 49 of the 51200 rays of a strategy), and one of the 64 pixels may be lit (a channel above 0).
//  10. "Bit for bit" covers what the mapping's table names: the material's kind and albedo, a metal's fuzz 0, the side
//      entry's kind, α and r, each computed in float as the table writes it (clamp(factor·texel), metallic·B,
//      roughness·G, rough·rough, max(rough², 1e-3f), max(rough, 0.0317f)); a lambertian's fuzz and the refraction
//      indices aren't in it (INFO). The texels the expectation multiplies are the hardware's own: read back by a
//      kernel (mapping) or through a material of factors 1 on the same texture (textures).
//  11. Texture reads land exactly on the point asked for: each point is corner 0 of a triangle of its own, hit at
//      (b1, b2) = (0, 0). The wrap points sit halfway between two texels (texel coordinates -1.5 and 4.5), where the
//      hardware's 8-bit filter weight 1/2 is exact: the expected value is the two texels' mean, within 2·10^-3.
//  12. The closed corner: a cube of 555 (8 shared corners, 12 triangles, one mesh under the identity), the camera at
//      (0.9, 0.9, 0.9) looking down at the floor's points x 0.48 to 0.8, z 0.09 to 0.25 (off its diagonal and its
//      edges), 8×8 pixels, 16 samples each, max_depth 50, the sky lit so that a path leaving the cube shows, and a
//      lamp above the cube for -p mis's shadow rays. Sized by the point alone (2^-16 there), the step off the floor
//      would lie inside the rounding of its 555-unit triangles, as in decision 9's second defect; offset_origin's
//      reach sizes it by the triangle (2^-16·555). Decision 9's allowances.
//  13. The hardware's sRGB decode is held to half the step of the byte it decodes, the step taken through
//      IEC 61966-2-1's curve: a byte stands for every sRGB value within half a step of it, so a decode inside that
//      interval is as close to the image as the byte itself. The texture hardware keeps about 8 significant bits of
//      the linear value (measured: 0.00375 off at byte 251, 0.86 of its half step), so a bound of 10^-3 fails 40 of the
//      bytes from 184 up.
//  14. memo.reach's reference is computed in double from the floats the GPU holds: the corners in the mesh's frame and
//      the instance's 3×4. On the hand-built tables those are the tables' own; on an upload, meshes.cu's frame as its
//      comments describe it: c, the centre of the bounds of the vertices the mesh's triangles use, computed in double
//      and rounded to float; the corners x − c, in double, rounded to float; the placement's A rounded to float and
//      its translation t + A·c, in double, rounded to float. The GPU's reach is within 10^-6 of it (relative): each
//      world coordinate takes at most 4 roundings of terms whose sizes add up to at most 3·reach (|t|∞ ≤ in_world +
//      ‖A‖∞·in_mesh), 7.2·10^-7 of the reach, and ‖A‖∞·in_mesh takes 3, 1.8·10^-7. A term dropped, a row sum taken
//      along a column, or a centre folded in wrong moves it by percents. The mesh term is the larger on the surface
//      tables' untranslated instance and on surface-upload's icosphere moved back over the world's origin.
//  15. lobe-draw: 2^18 rays in one launch, since a launch's sample k draws from sample_generator(seed, k, 0) and a
//      second launch would repeat the first one's numbers. The share's standard error is then 9·10^-4: the planted
//      0.31 lies 11 of them from the 0.3f the draws are compared with, and is rejected (|z| ≥ 4.9) unless the share
//      strays by 6. The lobe is the generator's first number bit for bit: closest() draws it after the media's
//      numbers, and the scene has no media.
// Seeds: 9101-9111, 9113, 9114, 9120, 9122.
//
// Step criteria: $GT optix-triangles --require-optix and $GT optix-pbr --require-optix exit 0 (≈ 3 s and ≈ 1 s, the
// check module's cold compile apart); $GT mesh-cache exits 0 (a fraction of a second of CPU: at most six reads of a
// model, the first build's two, the touched vase's two and second-model's two, and six builds' studio map).
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <limits>
#include <memory>
#include <numbers>
#include <set>
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
    int scratch_base() const { return slots_ > 0 ? scene_count_ + 1 : -1; }
    std::vector<gpu::material<float>> materials() const { return view_materials_.download(); }

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
constexpr int probe_out_words = 41;
constexpr int probe_batch = 1 << 15; // samples per launch: 8 MB of words

struct probe_sample
{
    std::array<float, 3> origin{};
    std::array<float, 3> direction{};
    int from = -1;
    bool inward = false;
    std::array<float, 3> facing{};
    int mode = 0; // 0: closest; 1: surface_of the triangle hit below; 2: surface_of the absorber's id
    int instance = 0;
    unsigned int primitive = 0;
    float b1 = 0;
    float b2 = 0;
    float lobe = 0;
    float reach = 0; // memo.reach of the face a ray leaves (mode 0)
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
    float reach = 0;         // memo.reach: triangle_surface's in mode 1
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
    w[18] = s.reach;
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
    a.reach = w[40];
    return a;
}

// --- Device code ---------------------------------------------------------------------------------------------------

// The 41 words of a surface (the probe's mode 1), as the CUDA twin writes them
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
    out[40] = memo.reach;
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
// test (1997), the nearest crossing beyond t_min. near_edge (decision 1): some triangle crossed no farther than it
// (anywhere, for a miss) has its smallest barycentric coordinate within max(10^-4, δ/altitude) of 0, on either side,
// with δ = 2^-16·(|o|∞ + t·|d|∞) at that crossing (decision 8) and the triangle's smallest altitude; a miss without
// such a crossing isn't near an edge.
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
__global__ void brute_force_kernel(const double* corners, const double* altitudes, int triangles, const float* rays,
                                   int n, double t_min, brute_answer* answers)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
    {
        return;
    }
    const float* const r = rays + static_cast<std::size_t>(probe_in_words) * k;
    const double o[3] = {r[0], r[1], r[2]};
    const double d[3] = {r[3], r[4], r[5]};
    const double o_size = ::fmax(::fabs(o[0]), ::fmax(::fabs(o[1]), ::fabs(o[2])));
    const double d_size = ::fmax(::fabs(d[0]), ::fmax(::fabs(d[1]), ::fabs(d[2])));
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
        const double band = ::fmax(1e-4, 0x1p-16 * (o_size + t * d_size) / altitudes[i]);
        if (::fabs(margin) < band)
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
    // A miss is near an edge only where some crossing was: without one near_t stays ∞, which ∞·(1 + 10^-9) would pass
    best.near_edge = ::isfinite(near_t) && near_t <= best.t * (1 + 1e-9) ? 1 : 0;
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

// Every sample through __raygen__probe_mesh, in launches of at most `batch` (sample k of a launch draws from
// sample_generator(seed, k, 0), so samples of different launches share streams); `words`, when given, receives the
// probe intersection program's 16 words per sample (P7)
std::vector<probe_answer> probe(gpu_test::context& ctx, gpu::detail::optix_scene& scene,
                                std::span<const probe_sample> samples, std::uint64_t seed, bool probing = false,
                                std::vector<unsigned long long>* words = nullptr, std::size_t batch = probe_batch)
{
    std::vector<probe_answer> answers;
    answers.reserve(samples.size());
    for (std::size_t first = 0; first < samples.size(); first += batch)
    {
        const std::size_t n = std::min<std::size_t>(batch, samples.size() - first);
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

// An upload the runtime must refuse, before anything is launched: a std::runtime_error whose message names `cause`
void expect_refusal(gpu_test::context& ctx, gpu::optix_runtime& runtime, std::string_view name,
                    const gpu::scene_desc& scene, std::string_view cause)
{
    std::string message = "no exception";
    bool right = false;
    try
    {
        [[maybe_unused]] const optix_upload upload(runtime, scene);
    }
    catch (const std::runtime_error& error)
    {
        message = error.what();
        right = message.find(cause) != std::string::npos;
    }
    catch (const std::exception& error)
    {
        message = std::format("not a std::runtime_error: {}", error.what());
    }
    const std::string check = std::format("refused-{}", name);
    ctx.count(check, right ? 0 : 1, 0, 1);
    if (!right)
    {
        ctx.info(check, std::format("expected a std::runtime_error naming \"{}\"; got {}", cause, message));
    }
}

// An upload the runtime must accept: the message of what it threw, empty for none
std::string upload_error(gpu::optix_runtime& runtime, const gpu::scene_desc& scene)
{
    try
    {
        [[maybe_unused]] const optix_upload upload(runtime, scene);
    }
    catch (const std::exception& error)
    {
        return error.what();
    }
    return {};
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

// --- memo.reach, recomputed in double (decision 14) -------------------------------------------------------------

// The reach triangle_surface gives a triangle of corners p (in the mesh's frame, as the GPU holds them) under the 3×4
// m (as the GPU holds it): the larger of the corners' largest coordinate in the world and their largest coordinate
// times ‖A‖∞, the largest row sum of m's linear part; by_mesh when the second is the larger
struct reach_reference
{
    double reach = 0;
    bool by_mesh = false;
};

reach_reference reach_of(const std::array<dvec, 3>& p, const matrix& m)
{
    double in_world = 0;
    double in_mesh = 0;
    for (const dvec& corner : p)
    {
        in_world = std::max(in_world, max_abs(point(m, corner)));
        in_mesh = std::max(in_mesh, max_abs(corner));
    }
    double row_sum = 0;
    for (std::size_t r = 0; r < 3; ++r)
    {
        row_sum = std::max(row_sum, std::abs(m[4 * r]) + std::abs(m[4 * r + 1]) + std::abs(m[4 * r + 2]));
    }
    return {.reach = std::max(in_world, row_sum * in_mesh), .by_mesh = row_sum * in_mesh > in_world};
}

// The reach of instance `instance`'s triangle `primitive` on the hand-built tables: their own corners and 3×4
reach_reference held_reach(const host_tables& t, int instance, unsigned int primitive)
{
    const gpu::mesh_instance& in = t.instances[static_cast<std::size_t>(instance)];
    const gpu::mesh_part& part = t.parts[static_cast<std::size_t>(in.part)];
    const uint3 tri = t.triangles[part.first_triangle + primitive];
    const std::array<dvec, 3> p{of(t.positions[part.first_vertex + tri.x]), of(t.positions[part.first_vertex + tri.y]),
                                of(t.positions[part.first_vertex + tri.z])};
    matrix m{};
    for (std::size_t i = 0; i < 12; ++i)
    {
        m[i] = in.object_to_world[i];
    }
    return reach_of(p, m);
}

// A mesh's centre in an upload (meshes.cu, mesh_frame): the centre of the bounds of the vertices its triangles use,
// computed in double and rounded to float
dvec upload_centre(const gpu::mesh_desc& mesh)
{
    dvec lower{infinite, infinite, infinite};
    dvec upper{-infinite, -infinite, -infinite};
    for (const std::uint32_t v : mesh.indices)
    {
        for (std::size_t a = 0; a < 3; ++a)
        {
            lower[a] = std::min(lower[a], static_cast<double>(mesh.positions[3 * static_cast<std::size_t>(v) + a]));
            upper[a] = std::max(upper[a], static_cast<double>(mesh.positions[3 * static_cast<std::size_t>(v) + a]));
        }
    }
    dvec c{};
    for (std::size_t a = 0; a < 3; ++a)
    {
        c[a] = static_cast<float>((lower[a] + upper[a]) / 2);
    }
    return c;
}

// The reach of a placed mesh's triangle `primitive` in an upload, whose centre is c: the corners x − c (in double)
// rounded to float; the placement's linear part A in float, its translation t + A·c (in double) in float
reach_reference upload_reach(const gpu::mesh_desc& mesh, const dvec& c, const matrix& placement,
                             unsigned int primitive)
{
    std::array<dvec, 3> p{};
    for (std::size_t k = 0; k < 3; ++k)
    {
        const std::size_t v = mesh.indices[3 * static_cast<std::size_t>(primitive) + k];
        for (std::size_t a = 0; a < 3; ++a)
        {
            p[k][a] = static_cast<float>(static_cast<double>(mesh.positions[3 * v + a]) - c[a]);
        }
    }
    matrix m = placement;
    for (std::size_t r = 0; r < 3; ++r)
    {
        m[4 * r + 3] += m[4 * r] * c[0] + m[4 * r + 1] * c[1] + m[4 * r + 2] * c[2];
    }
    return reach_of(p, in_float(m));
}

// The reaches of a set of hits against their references: within 10^-6 (relative), and each term the larger somewhere,
// so that both are seen
struct reach_tally
{
    std::int64_t samples = 0;
    std::int64_t wrong = 0;
    std::int64_t by_mesh = 0;
    double worst = 0;

    void add(const reach_reference& e, float reach)
    {
        ++samples;
        by_mesh += e.by_mesh ? 1 : 0;
        const double off = std::abs(reach - e.reach) / e.reach;
        worst = std::isnan(off) ? off : std::max(worst, off);
        wrong += off <= 1e-6 ? 0 : 1;
    }

    void report(gpu_test::context& ctx, const std::string& prefix) const
    {
        ctx.count(prefix + "-reach", wrong, 0, samples);
        ctx.count(prefix + "-reach-both-terms", by_mesh > 0 && by_mesh < samples ? 0 : 1, 0, 1);
        ctx.info(prefix + "-reach", std::format("at most {:.3g} of the reference off (<= 1e-6); the corners in the "
                                                "mesh's frame times ||A|| decided {} of {}, those in the world the "
                                                "rest",
                                                worst, by_mesh, samples));
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
        t.add_instance(ring, scaling(-1, 1, 1)),
        // Untranslated, turned and stretched: |A·p|∞ < ‖A‖∞·|p|∞ for most corners, so the reach's mesh term decides
        t.add_instance(wild, compose(rotation({1, 2, 3}, 40), scaling(1.5, -0.5, 2)))};
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

    // memo.reach of every hit, from the tables' own corners and transforms (decision 14)
    reach_tally reach;
    for (std::size_t k = 0; k < samples.size(); ++k)
    {
        reach.add(held_reach(t, samples[k].instance, samples[k].primitive), answers[k].reach);
    }
    reach.report(ctx, "surface");

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
            std::vector<double> altitudes(triangles);
            for (std::size_t i = 0; i < triangles; ++i)
            {
                altitudes[i] = smallest_altitude(corners, static_cast<int>(i));
            }
            const gpu_test::device_buffer<float> device_rays(rays);
            const gpu_test::device_buffer<double> device_corners(corners);
            const gpu_test::device_buffer<double> device_altitudes(altitudes);
            const gpu_test::device_buffer<brute_answer> device_answers(samples.size());
            const auto n = static_cast<int>(samples.size());
            brute_force_kernel<0><<<(n + 127) / 128, 128>>>(device_corners.get(), device_altitudes.get(),
                                                            static_cast<int>(triangles), device_rays.get(), n, 0.001,
                                                            device_answers.get());
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

        // The reach traverse sizes the step with (memo.reach, the probe's word 18): δ = 2^-16·reach once the reach is
        // above |p|∞. 4·|p|∞ moves the origins half the gap off their faces, which changes nothing; 16·|p|∞ moves
        // them twice the gap, beyond the neighbour, and the rays, going away from both planes, must meet nothing.
        const auto step = static_cast<std::size_t>(ctx.samples(10'000));
        for (const bool beyond : {false, true})
        {
            std::vector<probe_sample> reached = leaving_rays(c, upload.triangle_base, step, random);
            for (probe_sample& s : reached)
            {
                s.reach = static_cast<float>((beyond ? 16 : 4) * c.size);
            }
            const std::vector<probe_answer> met = probe(ctx, *upload.scene, reached, 9107);
            const int first = upload.triangle_base + 1;
            std::int64_t own_plane = 0; // a triangle of the plane left
            std::int64_t missed = 0;    // anything but the neighbour
            std::int64_t anything = 0;  // any hit at all
            for (std::size_t k = 0; k < reached.size(); ++k)
            {
                const int layer = static_cast<int>(k % 2);
                const int index = met[k].index;
                const int other_first = first + (1 - layer) * coplanar_planes::layer_triangles;
                own_plane += index >= first + layer * coplanar_planes::layer_triangles &&
                                     index < first + (layer + 1) * coplanar_planes::layer_triangles
                                 ? 1
                                 : 0;
                missed += index >= other_first && index < other_first + coplanar_planes::layer_triangles ? 0 : 1;
                anything += index >= 0 ? 1 : 0;
            }
            if (!beyond)
            {
                ctx.count("coplanar-reach-half-gap-plane-left", own_plane, 0, std::ssize(reached));
                ctx.count("coplanar-reach-half-gap-neighbour-missed", missed, 0, std::ssize(reached));
            }
            else
            {
                ctx.count("coplanar-reach-beyond-the-neighbour-met", anything, 0, std::ssize(reached));
            }
        }
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
        // Meshes off their own origins, whose centres the upload moves into their placements (t + A·c): the quad 300
        // units off, and the flat icosphere 1.8 units off
        const auto shifted = [](gpu::mesh_desc m, const dvec& by) {
            for (std::size_t k = 0; k < m.positions.size(); ++k)
            {
                m.positions[k] = static_cast<float>(m.positions[k] + by[k % 3]);
            }
            return m;
        };
        const std::vector<gpu::mesh_desc> meshes{gpu_test::mesh_host::icosphere(2, true),
                                                 alternating(gpu_test::mesh_host::torus(1, 0.35, 16, 8)),
                                                 quad,
                                                 flat,
                                                 shifted(quad, {300, -200, 50}),
                                                 shifted(flat, {1.5, -1, 0.25})};
        const std::vector<std::pair<int, matrix>> placements{
            {0, compose(translation(-3, 0, 1), rotation({0, 1, 1}, 25), scaling(-1.5, 1.5, 1.5))},
            {1, compose(translation(0, 4, -2), rotation({1, 0, 1}, 70), scaling(0.5, 2, 1))},
            {2, compose(translation(2, -1, 0.5), rotation({1, 2, 3}, 40))},
            {3, translation(0, -3, 0)},
            {0, compose(translation(3, 3, 3), scaling(0.6, -1.4, 1), rotation({1, 1, 0}, 50))},
            // Turning, mirroring, non-uniform: a fold of t + c, or of A transposed, moves the quad by hundreds
            {4, compose(translation(-2, 1, 3), rotation({1, -2, 1}, 35), scaling(-0.8, 1.3, 0.6))},
            // The icosphere's centre taken back to the world's origin: its triangles lie near it, about 1 from their
            // mesh's centre, so the reach's mesh term decides
            {5, compose(rotation({2, -1, 1}, 50), scaling(1.2, -0.7, 2), translation(-1.5, 1, -0.25))}};
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

        // memo.reach, from the corners and placements as the upload holds them (decision 14)
        std::vector<dvec> centres;
        for (const gpu::mesh_desc& m : meshes)
        {
            centres.push_back(upload_centre(m));
        }
        reach_tally reach;
        for (std::size_t k = 0; k < samples.size(); ++k)
        {
            const auto& [mesh, m] = placements[static_cast<std::size_t>(samples[k].instance)];
            const auto i = static_cast<std::size_t>(mesh);
            reach.add(upload_reach(meshes[i], centres[i], m, samples[k].primitive), answers[k].reach);
        }
        reach.report(ctx, "upload");
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

// A closed room seen from inside through gpu::render, as RayTracingGPU renders it (decision 9), under -p material and
// -p mis: no ray leaves the room and no light enters it, so every path traces max_depth rays and finds nothing. The
// rays may fall short of the full count by 1%, and one pixel may be lit (any channel above 0, or NaN).
void closed_room(gpu_test::context& ctx, const std::shared_ptr<gpu::optix_runtime>& runtime, gpu::scene_desc scene,
                 std::string_view check, std::uint64_t seed)
{
    const gpu::camera_desc& cam = scene.camera;
    const std::int64_t samples = std::int64_t{cam.image_width} * cam.image_height * cam.sqrt_spp * cam.sqrt_spp;
    const std::int64_t full = samples * cam.max_depth;
    const std::array<std::pair<std::string_view, gpu::light_combination>, 2> strategies{
        {{"material", gpu::light_combination::bsdf_only}, {"mis", gpu::light_combination::power_heuristic}}};
    for (const auto& [strategy, combination] : strategies)
    {
        scene.quality.combination = combination;
        const gpu::render_result r = gpu::render(scene, {.seed = seed, .optix_tracer = true, .optix = runtime});
        const auto rays = static_cast<std::int64_t>(r.rays);
        const std::int64_t pixels = std::ssize(r.pixels) / 3;
        std::int64_t lit = 0;
        for (std::size_t i = 0; i + 2 < r.pixels.size(); i += 3)
        {
            lit += r.pixels[i] != 0 || r.pixels[i + 1] != 0 || r.pixels[i + 2] != 0 ? 1 : 0; // NaN too
        }
        const std::string name = std::format("{}.{}", check, strategy);
        ctx.info(name, std::format("{} of the {} rays of full paths ({:.3f} per sample), {} shadow rays; {} of {} "
                                   "pixels lit, the largest value {:.3g}",
                                   rays, full, static_cast<double>(rays) / static_cast<double>(samples),
                                   r.shadow_rays, lit, pixels, *std::ranges::max_element(r.pixels)));
        ctx.count(name + ".rays-missing", full - rays, full / 100, full);
        ctx.count(name + ".lit", lit, 1, pixels);
    }
}

// The render path inside a closed mesh: cornell-triangles' tall block seen from inside (decision 9). -p material only
// bounces, -p mis also aims shadow rays at the lamp, which the block's faces (or the through-rule, for one aimed
// through the face it leaves) must stop.
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
        closed_room(ctx, runtime, scene, "closed-block", 9113);
    });
}

// A closed cube of 555 made of 12 triangles, seen from inside near its corner (0, 0, 0) (decision 12): the paths
// start on the floor where |p|∞ < 1, on triangles that reach 555, and bounce near the corner's three faces; a lamp
// above the cube, outside it, draws -p mis's shadow rays through the ceiling
GPU_TEST(optix_triangles, closed_corner)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = optix_for(ctx, "closed-corner");
    if (!runtime)
    {
        return;
    }
    guarded(ctx, "closed-corner", [&] {
        // The cube's 8 corners (x, y, z each 0 or 555) and its 6 faces, each two triangles sharing the corners
        gpu::mesh_desc cube;
        for (std::uint32_t k = 0; k < 8; ++k)
        {
            const auto side = [k](std::uint32_t bit) { return (k & bit) != 0 ? 555.0f : 0.0f; };
            cube.positions.insert(cube.positions.end(), {side(1), side(2), side(4)});
        }
        const std::array<std::array<std::uint32_t, 4>, 6> faces{{{0, 4, 6, 2},   // x = 0
                                                                 {1, 3, 7, 5},   // x = 555
                                                                 {0, 1, 5, 4},   // y = 0, the floor
                                                                 {2, 6, 7, 3},   // y = 555
                                                                 {0, 2, 3, 1},   // z = 0
                                                                 {4, 5, 7, 6}}}; // z = 555
        for (const auto& f : faces)
        {
            cube.indices.insert(cube.indices.end(), {f[0], f[1], f[2], f[0], f[2], f[3]});
            cube.material_of.insert(cube.material_of.end(), {0u, 0u});
        }
        gpu::scene_desc scene = mesh_scene({cube}, {{0, identity}});
        scene.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = {15, 15, 15}});
        scene.materials.push_back({.kind = gpu::material_kind::diffuse_light, .texture = 1});
        scene.quads = {{.q = {200, 600, 200}, .u = {155, 0, 0}, .v = {0, 0, 155}, .material = 2}}; // facing down
        // From (0.9, 0.9, 0.9) down to the floor's points x 0.48 to 0.8, z 0.09 to 0.25 (the pixels lie halfway): off
        // the floor's diagonal x = z and its edges; the sky stays lit, so a path that leaves the cube shows
        gpu::camera_desc& cam = scene.camera;
        cam.sqrt_spp = 4;
        cam.max_depth = 50;
        cam.center = {0.9, 0.9, 0.9};
        cam.pixel00_loc = {0.7, 0.45, 0.5};
        cam.pixel_delta_u = {0.02, 0, 0};
        cam.pixel_delta_v = {0, 0, 0.01};
        cam.defocus = false;
        closed_room(ctx, runtime, scene, "closed-corner", 9114);
    });
}

// The upload's refusals (meshes.cu), each a std::runtime_error naming its cause, before anything is launched: the
// materials a triangle can't wear (glass, a lamp, smoke), an index past the mesh's vertices, a material past the set,
// a triangle's corner that isn't finite, and placements that aren't finite or can't be inverted. And what it must
// accept: a vertex no triangle uses may hold a NaN, as the glTF loader leaves one when it drops a triangle.
GPU_TEST(optix_triangles, refusals)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = optix_for(ctx, "refusals");
    if (!runtime)
    {
        return;
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto refused = [&](std::string_view name, const gpu::scene_desc& scene, std::string_view cause) {
        expect_refusal(ctx, *runtime, name, scene, cause);
    };
    // A triangle wearing each of the three kinds a triangle can't: the scene's material 2
    const std::array<std::pair<std::string_view, gpu::material_desc>, 3> worn{
        {{"glass", {.kind = gpu::material_kind::dielectric, .refraction_index = 1.5}},
         {"lamp", {.kind = gpu::material_kind::diffuse_light, .texture = 0}},
         {"smoke", {.kind = gpu::material_kind::isotropic, .texture = 0}}}};
    const std::array<std::string_view, 3> causes{"is glass", "is a lamp", "particles of smoke"};
    for (std::size_t i = 0; i < worn.size(); ++i)
    {
        gpu::mesh_desc m = one_triangle();
        m.material_of = {2};
        gpu::scene_desc scene = mesh_scene({m}, {{0, identity}});
        scene.materials.push_back(worn[i].second);
        scene.meshes.materials.push_back({.scene_material = 2});
        refused(worn[i].first, scene, causes[i]);
    }
    // The mesh's own arrays
    gpu::mesh_desc past = one_triangle();
    past.indices = {0, 1, 3};
    refused("index-past-the-vertices", mesh_scene({past}, {{0, identity}}), "of a vertex it doesn't have");
    gpu::mesh_desc unworn = one_triangle();
    unworn.material_of = {2};
    refused("material-past-the-set", mesh_scene({unworn}, {{0, identity}}), "a material the set doesn't have");
    gpu::mesh_desc corner = one_triangle();
    corner.positions[4] = nan;
    refused("corner-not-finite", mesh_scene({corner}, {{0, identity}}), "isn't finite");
    // The placements: a NaN, and a matrix that flattens the mesh onto a plane (determinant 0)
    matrix with_nan = translation(1, 2, 3);
    with_nan[5] = std::numeric_limits<double>::quiet_NaN();
    refused("transform-nan", mesh_scene({one_triangle()}, {{0, with_nan}}), "can't be inverted");
    refused("transform-singular", mesh_scene({one_triangle()}, {{0, scaling(1, 1, 0)}}), "can't be inverted");

    // Accepted: a NaN and an infinity in vertices that no triangle uses
    gpu::mesh_desc spare = one_triangle();
    spare.positions.insert(spare.positions.end(), {nan, 0, 0, 0, std::numeric_limits<float>::infinity(), 0});
    spare.uvs.insert(spare.uvs.end(), {0, 0, 0, 0});
    const std::string message = upload_error(*runtime, mesh_scene({spare}, {{0, identity}}));
    ctx.count("unused-vertex-not-finite", message.empty() ? 0 : 1, 0, 1);
    if (!message.empty())
    {
        ctx.info("unused-vertex-not-finite", std::format("the upload threw: {}", message));
    }
}

// =====================================================================================================================
// Suite optix-pbr
// =====================================================================================================================

// --- glTF's materials: the mapping in float, and textures ------------------------------------------------------------

// A glTF material of the hand-built tables (mesh_tables.h's mesh_material): factors in float, textures by handle
gpu::mesh_material gltf_material(const std::array<double, 3>& base, double metallic, double roughness,
                                 const cudaTextureObject_t* base_texture = nullptr,
                                 const cudaTextureObject_t* metal_rough_texture = nullptr)
{
    gpu::mesh_material m{};
    m.scene_material = -1;
    for (std::size_t c = 0; c < 3; ++c)
    {
        m.base_color[c] = static_cast<float>(base[c]);
    }
    m.metallic = static_cast<float>(metallic);
    m.roughness = static_cast<float>(roughness);
    m.base_color_texture = base_texture ? *base_texture : 0;
    m.metal_rough_texture = metal_rough_texture ? *metal_rough_texture : 0;
    m.has_base_color_texture = base_texture ? 1u : 0u;
    m.has_metal_rough_texture = metal_rough_texture ? 1u : 0u;
    return m;
}

// What the mapping of glTF's metallic-roughness model (its Appendix B, onto milestone 4's GGX metal and glossy coat)
// writes into a hit's slot, in float: base = clamp(factor ⊙ texel, 0, 1), metal = metallic · texel.b, rough =
// roughness · texel.g (a missing texture reads 1); lobe < metal: the GGX metal (fuzz 0, α = rough², r = rough),
// otherwise the coat over a matte base (α = max(rough², 10^-3), r = max(rough, 0.0317))
struct slot_contents
{
    gpu::material<float> material;
    gpu::microfacet facet;
};

slot_contents mapped_slot(const std::array<float, 3>& factor, float metallic, float roughness,
                          const std::array<float, 3>& base_texel, float texel_g, float texel_b, float lobe)
{
    slot_contents s{};
    std::array<float, 3> base{};
    for (std::size_t c = 0; c < 3; ++c)
    {
        base[c] = std::clamp(factor[c] * base_texel[c], 0.0f, 1.0f);
    }
    const float metal = metallic * texel_b;
    const float rough = roughness * texel_g;
    s.material.albedo = {base[0], base[1], base[2]};
    if (lobe < metal)
    {
        s.material.kind = gpu::material_kind::metal;
        s.material.fuzz = 0;
        s.facet = {.kind = gpu::microfacet_kind::conductor, .alpha = rough * rough, .roughness = rough};
    }
    else
    {
        s.material.kind = gpu::material_kind::lambertian;
        s.facet = {.kind = gpu::microfacet_kind::gloss,
                   .alpha = std::max(rough * rough, 1e-3f),
                   .roughness = std::max(rough, 0.0317f)};
    }
    return s;
}

bool same_bits(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }

// The slot as the integrator reads it against the mapping's: kind and albedo bit for bit, fuzz 0 on a metal, the side
// entry bit for bit (a lambertian's fuzz and every refraction index are no part of the mapping's table)
bool slot_matches(const probe_answer& a, const slot_contents& e)
{
    const bool metal = e.material.kind == gpu::material_kind::metal;
    return a.kind == static_cast<int>(e.material.kind) && same_bits(a.albedo[0], e.material.albedo.x) &&
           same_bits(a.albedo[1], e.material.albedo.y) && same_bits(a.albedo[2], e.material.albedo.z) &&
           (!metal || same_bits(a.fuzz, 0.0f)) && a.facet_kind == static_cast<int>(e.facet.kind) &&
           same_bits(a.alpha, e.facet.alpha) && same_bits(a.roughness, e.facet.roughness);
}

std::string describe(const probe_answer& a)
{
    return std::format("kind {}, albedo ({:.9g}, {:.9g}, {:.9g}), fuzz {:.9g}, side entry {} alpha {:.9g} r {:.9g}",
                       a.kind, a.albedo[0], a.albedo[1], a.albedo[2], a.fuzz, a.facet_kind, a.alpha, a.roughness);
}
std::string describe(const slot_contents& e)
{
    return std::format("kind {}, albedo ({:.9g}, {:.9g}, {:.9g}), side entry {} alpha {:.9g} r {:.9g}",
                       static_cast<int>(e.material.kind), e.material.albedo.x, e.material.albedo.y,
                       e.material.albedo.z, static_cast<int>(e.facet.kind), e.facet.alpha, e.facet.roughness);
}

// IEC 61966-2-1's decoding of an sRGB value c in [0, 1] to linear light
double srgb_to_linear(double c) { return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4); }

// |linear - IEC 61966-2-1's decoding of byte b| in halves of the byte's step, decoded (the larger of its two halves):
// at most 1 for a decode as close to the image as the byte itself (decision 13)
double srgb_steps_off(double linear, int b)
{
    const double c = b / 255.0;
    const double half = std::max(srgb_to_linear(c + 0.5 / 255) - srgb_to_linear(c),
                                 srgb_to_linear(c) - srgb_to_linear(c - 0.5 / 255));
    return std::abs(linear - srgb_to_linear(c)) / half;
}

// A texture object as meshes.cu says it makes them: one uchar4 cudaArray, read as normalized floats at normalized
// coordinates, glTF's wrap per axis, bilinear or nearest, sRGB-decoded or not. Freed when it goes.
class test_texture
{
public:
    test_texture(int width, int height, std::span<const std::uint8_t> rgba, bool srgb, bool nearest)
    {
        const cudaChannelFormatDesc format = cudaCreateChannelDesc<uchar4>();
        gpu_test::cuda_check(
            cudaMallocArray(&array_, &format, static_cast<std::size_t>(width), static_cast<std::size_t>(height)),
            "cudaMallocArray");
        try
        {
            const std::size_t row = 4 * static_cast<std::size_t>(width);
            gpu_test::cuda_check(cudaMemcpy2DToArray(array_, 0, 0, rgba.data(), row, row,
                                                     static_cast<std::size_t>(height), cudaMemcpyHostToDevice),
                                 "copying a texture to the GPU");
            cudaResourceDesc resource{};
            resource.resType = cudaResourceTypeArray;
            resource.res.array.array = array_;
            cudaTextureDesc desc{};
            desc.addressMode[0] = cudaAddressModeWrap;
            desc.addressMode[1] = cudaAddressModeWrap;
            desc.filterMode = nearest ? cudaFilterModePoint : cudaFilterModeLinear;
            desc.readMode = cudaReadModeNormalizedFloat;
            desc.normalizedCoords = 1;
            desc.sRGB = srgb ? 1 : 0;
            gpu_test::cuda_check(cudaCreateTextureObject(&object_, &resource, &desc, nullptr),
                                 "cudaCreateTextureObject");
        }
        catch (...)
        {
            cudaFreeArray(array_);
            throw;
        }
    }
    ~test_texture()
    {
        cudaDestroyTextureObject(object_);
        cudaFreeArray(array_);
    }
    test_texture(const test_texture&) = delete;
    test_texture& operator=(const test_texture&) = delete;

    const cudaTextureObject_t* object() const { return &object_; }

private:
    cudaArray_t array_ = nullptr;
    cudaTextureObject_t object_ = 0;
};

// tex2D at n points: what the hardware returns there
template <int Unused>
__global__ void texel_kernel(cudaTextureObject_t texture, const float2* at, float4* out, int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n)
    {
        out[k] = tex2D<float4>(texture, at[k].x, at[k].y);
    }
}

std::vector<float4> texels(const test_texture& texture, const std::vector<float2>& at)
{
    const gpu_test::device_buffer<float2> points(at);
    const gpu_test::device_buffer<float4> out(at.size());
    const auto n = static_cast<int>(at.size());
    texel_kernel<0><<<(n + 127) / 128, 128>>>(*texture.object(), points.get(), out.get(), n);
    gpu_test::finish_launch("the texel kernel");
    return out.download();
}

// One small triangle per point of a texture to read: corner 0 carries the point's (u, v), corners 1 and 2 one unit
// further in u and in v, so the hit (b1, b2) = (0, 0) reads exactly that point (u = 1·u0 + 0·u1 + 0·u2, whatever the
// order of the sum); flat, counter-clockwise from +z; the triangle wears the point's material
struct uv_point
{
    double u;
    double v;
    std::uint32_t material;
    float lobe;
};

gpu::mesh_desc uv_points(std::span<const uv_point> points)
{
    gpu::mesh_desc m;
    for (std::size_t j = 0; j < points.size(); ++j)
    {
        const auto x = static_cast<float>(j);
        const auto u = static_cast<float>(points[j].u);
        const auto v = static_cast<float>(points[j].v);
        m.positions.insert(m.positions.end(), {x, 0, 0, x + 1, 0, 0, x, 1, 0});
        m.uvs.insert(m.uvs.end(), {u, v, u + 1, v, u, v + 1});
        const auto first = static_cast<std::uint32_t>(3 * j);
        m.indices.insert(m.indices.end(), {first, first + 1, first + 2});
        m.material_of.push_back(points[j].material);
    }
    return m;
}

// The probe's samples of the points: triangle j of instance 0 at (0, 0), hit from +z, with the point's lobe
std::vector<probe_sample> point_samples(std::span<const uv_point> points)
{
    std::vector<probe_sample> samples;
    for (std::size_t j = 0; j < points.size(); ++j)
    {
        probe_sample s;
        s.mode = 1;
        s.instance = 0;
        s.primitive = static_cast<unsigned int>(j);
        s.origin = {static_cast<float>(j) + 0.25f, 0.25f, 1.0f};
        s.direction = {0, 0, -1};
        s.lobe = points[j].lobe;
        samples.push_back(s);
    }
    return samples;
}

// The mapping's comparison: n numbers drawn here as memo.lobe on a glTF triangle of metallic 0.3, each thread in its
// own slot
template <int Unused>
__global__ void lobe_kernel(gpu::optix_params p, std::uint64_t seed, int count, unsigned long long* metals)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    gpu::pcg32 rng = gpu::sample_generator(seed, static_cast<std::uint64_t>(t), 0);
    unsigned long long local = 0;
    const gpu::ray<float> r{.origin = {0.25f, 0.25f, 1.0f}, .direction = {0.0f, 0.0f, -1.0f}, .time = 0.0f};
    const gpu::bvh_hit<float> hit{.index = p.meshes.instances[0].first_id, .t = 1.0f, .entering = false};
    for (int i = 0; i < count; ++i)
    {
        gpu::triangle_memo memo;
        memo.instance = 0;
        memo.primitive = 0;
        memo.b1 = 0.25f;
        memo.b2 = 0.25f;
        memo.lobe = gpu::random_real<float>(rng);
        const gpu::surface<float> s = gpu::triangle_surface(p, r, hit, memo, p.meshes.scratch_base + t);
        local += p.view.base.materials[s.material].kind == gpu::material_kind::metal ? 1 : 0;
    }
    atomicAdd(metals, local);
}

// The first number of sample_generator(seed, k, 0) for k < n: the generator of the probe's sample k
template <int Unused>
__global__ void first_number_kernel(std::uint64_t seed, float* out, int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n)
    {
        gpu::pcg32 rng = gpu::sample_generator(seed, static_cast<std::uint64_t>(k), 0);
        out[k] = gpu::random_real<float>(rng);
    }
}

// The mapping in a plain CUDA kernel, on hand-made materials and hand-built tables (textures made here the way
// meshes.cu says it makes them): each slot against the expected structs bit for bit, the slot's index, texture -1,
// wears_gltf; a scene material's hit leaves its slot alone. And the hardware's sRGB decode against IEC 61966-2-1.
GPU_TEST(optix_pbr, mapping)
{
    // 2×2 textures read nearest: a base color in sRGB and a metallic-roughness map in linear (G roughness, B metallic)
    const std::array<std::uint8_t, 16> color_bytes{200, 40,  90,  255, 10, 250, 128, 255,
                                                   255, 255, 255, 255, 0,  0,   0,   255};
    const std::array<std::uint8_t, 16> rough_bytes{0, 51, 255, 255, 0, 102, 0, 255, 0, 153, 255, 255, 0, 204, 0, 255};
    const test_texture color(2, 2, color_bytes, true, true);
    const test_texture rough(2, 2, rough_bytes, false, true);
    const std::vector<gpu::mesh_material> gltf{
        gltf_material({0.9, 0.6, 0.3}, 1, 0.1),                                // 1: a metal, r 0.1
        gltf_material({0.9, 0.6, 0.3}, 1, 0.5),                                // 2: a rough metal
        gltf_material({0.2, 0.5, 0.8}, 0, 0.3),                                // 3: a coat over matte
        gltf_material({0.2, 0.5, 0.8}, 0, 0.8),                                // 4: a rough coat
        gltf_material({0.2, 0.5, 0.8}, 0, 0.01),                               // 5: the coat's floors
        gltf_material({0.2, 0.5, 0.8}, 0, 0),                                  // 6: roughness 0 on the coat
        gltf_material({0.9, 0.6, 0.3}, 1, 0),                                  // 7: a smooth metal (α 0)
        gltf_material({0.7, 0.7, 0.7}, 0.5, 0.4),                              // 8: half metal
        gltf_material({1.5, -0.25, 0.5}, 1, 0.5),                              // 9: a factor past [0, 1], metal
        gltf_material({1.5, -0.25, 0.5}, 0, 0.5),                              // 10: the same on the coat
        gltf_material({0.5, 0.3, 0.3}, 0.3f, 0.5),                             // 11: lobe = metallic: not a metal
        gltf_material({0.5, 0.75, 1}, 0.9, 0.6, color.object(), rough.object()), // 12: both textures
        gltf_material({1, 1, 1}, 1, 1, color.object(), rough.object())};       // 13: the textures' own values
    host_tables t;
    t.materials = {scene_material(0)};
    t.materials.insert(t.materials.end(), gltf.begin(), gltf.end());
    std::vector<uv_point> points{{0.5, 0.5, 1, 0.5f},  {0.5, 0.5, 2, 0.5f},  {0.5, 0.5, 3, 0.5f},  {0.5, 0.5, 4, 0.5f},
                                 {0.5, 0.5, 5, 0.5f},  {0.5, 0.5, 6, 0.5f},  {0.5, 0.5, 7, 0.5f},  {0.5, 0.5, 8, 0.25f},
                                 {0.5, 0.5, 8, 0.75f}, {0.5, 0.5, 9, 0.5f},  {0.5, 0.5, 10, 0.5f}, {0.5, 0.5, 11, 0.3f},
                                 {0.5, 0.5, 0, 0.5f}}; // the last: a scene material
    const std::size_t scene_point = points.size() - 1;
    std::vector<float2> centres;
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            const double u = (x + 0.5) / 2;
            const double v = (y + 0.5) / 2;
            points.push_back({u, v, 12, 0.5f});
            points.push_back({u, v, 13, 0.5f});
            centres.push_back(make_float2(static_cast<float>(u), static_cast<float>(v)));
        }
    }
    t.add_instance(t.add_part(uv_points(points)), identity);
    const std::vector<probe_sample> samples = point_samples(points);
    const device_tables tables(t, hand_materials, hand_textures, static_cast<int>(samples.size()));
    const std::vector<probe_answer> answers = surfaces_in_cuda(tables, samples);
    const std::vector<float4> color_texels = texels(color, centres);
    const std::vector<float4> rough_texels = texels(rough, centres);

    std::int64_t wrong = 0;
    std::int64_t gltf_hits = 0;
    std::string first_wrong;
    std::int64_t slot_index = 0;
    std::set<std::uint32_t> refraction_indices;
    for (std::size_t k = 0; k < points.size(); ++k)
    {
        if (k == scene_point)
        {
            continue;
        }
        ++gltf_hits;
        const gpu::mesh_material& m = t.materials[points[k].material];
        std::array<float, 3> base_texel{1, 1, 1};
        float texel_g = 1;
        float texel_b = 1;
        if (m.has_base_color_texture != 0)
        {
            const float4 c = color_texels[(k - scene_point - 1) / 2];
            base_texel = {c.x, c.y, c.z};
        }
        if (m.has_metal_rough_texture != 0)
        {
            const float4 c = rough_texels[(k - scene_point - 1) / 2];
            texel_g = c.y;
            texel_b = c.z;
        }
        const slot_contents e = mapped_slot({m.base_color[0], m.base_color[1], m.base_color[2]}, m.metallic,
                                            m.roughness, base_texel, texel_g, texel_b, points[k].lobe);
        const probe_answer& a = answers[k];
        const bool right = slot_matches(a, e) && a.wears_gltf && a.texture_index == -1;
        if (!right && first_wrong.empty())
        {
            first_wrong = std::format("point {}: {}; expected {}", k, describe(a), describe(e));
        }
        wrong += right ? 0 : 1;
        slot_index += a.material == tables.scratch_base() + static_cast<int>(k) ? 0 : 1;
        refraction_indices.insert(std::bit_cast<std::uint32_t>(a.refraction_index));
    }
    ctx.count("mapping", wrong, 0, gltf_hits);
    if (!first_wrong.empty())
    {
        ctx.info("mapping", first_wrong);
    }
    ctx.count("mapping-slot-index", slot_index, 0, gltf_hits); // s.material = scratch_base + the pixel
    std::string indices;
    for (const std::uint32_t r : refraction_indices)
    {
        indices += std::format("{}{}", indices.empty() ? "" : ", ", std::bit_cast<float>(r));
    }
    ctx.info("mapping-refraction-index", std::format("{} (no part of the mapping's table)", indices));
    // A scene material's hit: wears_gltf false, its own material, and its slot left as it was
    const probe_answer& s = answers[scene_point];
    const std::vector<gpu::material<float>> after = tables.materials();
    const gpu::material<float>& slot = after[static_cast<std::size_t>(tables.scratch_base()) + scene_point];
    const bool untouched = slot.kind == slot_sentinel.kind && same_bits(slot.albedo.x, slot_sentinel.albedo.x) &&
                           same_bits(slot.fuzz, slot_sentinel.fuzz);
    ctx.count("scene-material-hit", !s.wears_gltf && s.material == 0 && untouched ? 0 : 1, 0, 1);

    // The hardware's sRGB decode: byte i of a 256×1 texture at its texel's centre, against IEC 61966-2-1
    std::vector<std::uint8_t> ramp;
    std::vector<float2> ramp_centres;
    for (int i = 0; i < 256; ++i)
    {
        const auto b = static_cast<std::uint8_t>(i);
        ramp.insert(ramp.end(), {b, b, b, 255});
        ramp_centres.push_back(make_float2((i + 0.5f) / 256, 0.5f));
    }
    const test_texture srgb(256, 1, ramp, true, false);
    const std::vector<float4> decoded = texels(srgb, ramp_centres);
    double largest = 0;
    double largest_off = 0;
    int at = 0;
    for (int i = 0; i < 256; ++i)
    {
        const float4 d = decoded[static_cast<std::size_t>(i)];
        const double steps = std::max({srgb_steps_off(d.x, i), srgb_steps_off(d.y, i), srgb_steps_off(d.z, i)});
        largest_off = std::max(largest_off, std::abs(d.x - srgb_to_linear(i / 255.0)));
        if (!(steps <= largest))
        {
            largest = steps;
            at = i;
        }
    }
    ctx.exact("srgb-decode-hardware", largest, 0, 1);
    ctx.info("srgb-decode-hardware", std::format("over the 256 bytes the decode is at most {:.3g} of a byte's half "
                                                 "step from IEC 61966-2-1, at byte {}; the largest |difference| {:.3g}",
                                                 largest, at, largest_off));
}

// The mapping's comparison lobe < metal, on numbers the kernel draws itself (closest()'s draw is lobe-draw's): P(metal)
// = metallic (glTF's mix of metal and dielectric) over 10^6 numbers at metallic 0.3, binomial |z| <= 4; planted: the
// same count against 0.31 must be rejected
GPU_TEST(optix_pbr, lobe_comparison)
{
    host_tables t;
    t.materials = {gltf_material({0.5, 0.5, 0.5}, 0.3, 0.5)};
    t.add_instance(t.add_part(one_triangle()), identity);
    const device_tables tables(t, hand_materials, hand_textures, gpu_test::test_threads);
    const std::int64_t n = ctx.samples(1'000'000);
    gpu_test::device_buffer<unsigned long long> metals(1);
    lobe_kernel<0><<<gpu_test::test_blocks, gpu_test::test_block_size>>>(tables.params(), 9120,
                                                                         gpu_test::per_thread(n), metals.get());
    gpu_test::finish_launch("the lobe kernel");
    const auto drawn = static_cast<double>(gpu_test::drawn(n));
    const double share = static_cast<double>(metals.download()[0]) / drawn;
    // P(lobe < 0.3f) for lobe = k·2^-24 is exactly 0.3f: 0.3f = 5033165·2^-24
    const double p = 0.3f;
    ctx.mean("comparison-metal-share", share, std::sqrt(p * (1 - p) / drawn), p);
    const double wrong = 0.31;
    const double z = (share - wrong) / std::sqrt(wrong * (1 - wrong) / drawn);
    ctx.must_reject("planted-comparison-share", z, 0, gpu_test::normal_two_sided_p(z));
}

// closest()'s draw of the lobe number, through the probe's mode 0 (decision 15): rays straight down at a glTF triangle
// of metallic 0.3, one per launch index and generator, each lobe the generator's first number and below 0.3f with
// P = 0.3f; planted: the same count against 0.31 must be rejected. A ray at a triangle wearing a scene material and a
// ray that misses draw nothing: their lobe stays 0.
GPU_TEST(optix_pbr, lobe_draw)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = optix_for(ctx, "lobe-draw");
    if (!runtime)
    {
        return;
    }
    guarded(ctx, "lobe-draw", [&] {
        // Triangle 0 wears glTF's material (mesh material 2), triangle 1, two units along x, the scene's matte
        gpu::mesh_desc pair{.positions = {0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 2, 1, 0},
                            .normals = {},
                            .uvs = {},
                            .indices = {0, 1, 2, 3, 4, 5},
                            .material_of = {2, 0}};
        gpu::scene_desc scene = mesh_scene({pair}, {{0, identity}});
        scene.meshes.materials.push_back(
            {.scene_material = -1, .base_color = {0.5, 0.5, 0.5}, .metallic = 0.3, .roughness = 0.5});
        const optix_upload upload(*runtime, scene);
        const auto down_at = [](float x) {
            probe_sample s;
            s.origin = {x, 0.25f, 1.0f};
            s.direction = {0.0f, 0.0f, -1.0f};
            return s;
        };
        const auto n = static_cast<std::size_t>(ctx.samples(1 << 18));
        std::vector<probe_sample> samples(n, down_at(0.25f));
        samples.push_back(down_at(2.25f)); // the scene material's triangle
        samples.push_back(down_at(5.0f));  // nothing there
        const std::vector<probe_answer> answers =
            probe(ctx, *upload.scene, samples, 9111, false, nullptr, samples.size()); // one launch
        const gpu_test::device_buffer<float> first(n);
        first_number_kernel<0><<<static_cast<unsigned int>((n + 127) / 128), 128>>>(9111, first.get(),
                                                                                     static_cast<int>(n));
        gpu_test::finish_launch("the first-number kernel");
        const std::vector<float> numbers = first.download();

        std::int64_t not_hit = 0;
        std::int64_t other_number = 0;
        std::int64_t metals = 0;
        for (std::size_t k = 0; k < n; ++k)
        {
            not_hit += answers[k].index == upload.triangle_base + 1 ? 0 : 1;
            other_number += same_bits(answers[k].lobe, numbers[k]) ? 0 : 1;
            metals += answers[k].lobe < 0.3f ? 1 : 0;
        }
        const auto rays = static_cast<std::int64_t>(n);
        ctx.count("draw-gltf-hits", not_hit, 0, rays);
        ctx.count("draw-first-number", other_number, 0, rays);
        const double share = static_cast<double>(metals) / static_cast<double>(n);
        const double p = 0.3f; // as lobe-comparison: exactly the share of the numbers k·2^-24 below 0.3f
        ctx.mean("draw-metal-share", share, std::sqrt(p * (1 - p) / static_cast<double>(n)), p);
        const double wrong = 0.31;
        const double z = (share - wrong) / std::sqrt(wrong * (1 - wrong) / static_cast<double>(n));
        ctx.must_reject("planted-draw-share", z, 0, gpu_test::normal_two_sided_p(z));
        const probe_answer& matte = answers[n];
        ctx.count("draw-scene-material",
                  matte.index == upload.triangle_base + 2 && same_bits(matte.lobe, 0.0f) ? 0 : 1, 0, 1);
        const probe_answer& sky = answers[n + 1];
        ctx.count("draw-miss", sky.index == -1 && same_bits(sky.lobe, 0.0f) ? 0 : 1, 0, 1);
    });
}

// The texture objects meshes.cu makes, through its upload and optix_tracer::surface_of (the probe's mode 1): IEC's
// sRGB curve for every byte of a base color, linear metallic-roughness, bilinear reads at texel centres equal the
// texel, (0, 0) the top-left texel, the wrap modes at u and v = -0.25 and 1.25, glTF's NEAREST; and the mapping end to
// end on materials given as mesh_material_descs (their doubles become the floats of the expected structs). The probe
// grows the tables past the 8×8 camera's 64 slots first (prepare_view): a triangle wearing a scene material, and the
// absorber (the probe's mode 2), then read what the longer tables copied of the scene's materials and side entries.
GPU_TEST(optix_pbr, textures)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = optix_for(ctx, "textures");
    if (!runtime)
    {
        return;
    }
    guarded(ctx, "textures", [&] {
        gpu::scene_desc scene = mesh_scene({}, {});
        // A side entry of the scene's own on its matte, a glossy coat, so that a copy of the side entries shows
        scene.quality.microfacets = {{.kind = gpu::microfacet_kind::gloss, .roughness = 0.5}, {}};
        std::vector<gpu::mesh_texture_desc> textures;
        const auto add_texture = [&textures](int width, int height, std::vector<std::uint8_t> rgba, bool srgb,
                                             gpu::texture_wrap s = gpu::texture_wrap::repeat,
                                             gpu::texture_wrap t = gpu::texture_wrap::repeat, bool nearest = false) {
            textures.push_back({.width = width,
                                .height = height,
                                .rgba = std::move(rgba),
                                .srgb = srgb,
                                .wrap_s = s,
                                .wrap_t = t,
                                .nearest = nearest});
            return static_cast<int>(textures.size()) - 1;
        };
        std::vector<gpu::mesh_material_desc> materials;
        const auto add_material = [&materials](std::array<double, 3> base, double metallic, double roughness,
                                               int base_texture = -1, int metal_rough_texture = -1) {
            materials.push_back({.scene_material = -1,
                                 .base_color = base,
                                 .metallic = metallic,
                                 .roughness = roughness,
                                 .base_color_texture = base_texture,
                                 .metal_rough_texture = metal_rough_texture});
            return static_cast<std::uint32_t>(materials.size()) - 1;
        };
        // Every byte of a base color in sRGB (R = G = B = i), on a metal: the slot's albedo is the decoded byte
        std::vector<std::uint8_t> ramp;
        for (int i = 0; i < 256; ++i)
        {
            const auto b = static_cast<std::uint8_t>(i);
            ramp.insert(ramp.end(), {b, b, b, 255});
        }
        const std::uint32_t srgb_metal = add_material({1, 1, 1}, 1, 0.5, add_texture(256, 1, ramp, true));
        // A 4×4 metallic-roughness map, G = 16·(4y + x) + 8 at texel (x, y) of rows from the top, B = 255 (metal):
        // roughness 1 reads G itself into the side entry's r
        const auto grid_g = [](int x, int y) { return static_cast<std::uint8_t>(16 * (4 * y + x) + 8); };
        std::vector<std::uint8_t> grid;
        std::vector<std::uint8_t> grey_grid;
        for (int y = 0; y < 4; ++y)
        {
            for (int x = 0; x < 4; ++x)
            {
                grid.insert(grid.end(), {0, grid_g(x, y), 255, 255});
                grey_grid.insert(grey_grid.end(), {grid_g(x, y), grid_g(x, y), grid_g(x, y), 255});
            }
        }
        const std::uint32_t grid_metal = add_material({1, 1, 1}, 1, 1, -1, add_texture(4, 4, grid, false));
        // The same bytes as a base color, bilinear and nearest: at a texel's centre the two read the same
        const std::uint32_t grey_linear = add_material({1, 1, 1}, 1, 0.5, add_texture(4, 4, grey_grid, true));
        const std::uint32_t grey_nearest = add_material(
            {1, 1, 1}, 1, 0.5,
            add_texture(4, 4, grey_grid, true, gpu::texture_wrap::repeat, gpu::texture_wrap::repeat, true));
        // 4×1 and 1×4 maps of G = 20, 80, 160, 240 under each wrap mode, bilinear; and 4×1 nearest
        const std::array<std::uint8_t, 4> strip{20, 80, 160, 240};
        std::vector<std::uint8_t> strip_bytes;
        for (const std::uint8_t g : strip)
        {
            strip_bytes.insert(strip_bytes.end(), {0, g, 255, 255});
        }
        const std::array<gpu::texture_wrap, 3> wraps{gpu::texture_wrap::repeat, gpu::texture_wrap::mirrored_repeat,
                                                     gpu::texture_wrap::clamp};
        std::array<std::uint32_t, 3> along_u{};
        std::array<std::uint32_t, 3> along_v{};
        for (std::size_t w = 0; w < 3; ++w)
        {
            along_u[w] = add_material({1, 1, 1}, 1, 1, -1,
                                      add_texture(4, 1, strip_bytes, false, wraps[w], gpu::texture_wrap::repeat));
            along_v[w] = add_material({1, 1, 1}, 1, 1, -1,
                                      add_texture(1, 4, strip_bytes, false, gpu::texture_wrap::repeat, wraps[w]));
        }
        const std::uint32_t strip_nearest =
            add_material({1, 1, 1}, 1, 1, -1,
                         add_texture(4, 1, strip_bytes, false, gpu::texture_wrap::repeat, gpu::texture_wrap::repeat,
                                     true));
        // Both textures with factors, and the same textures with factors 1 (their own values): the mapping end to end
        const int color = add_texture(2, 2, {200, 40, 90, 255, 10, 250, 128, 255, 255, 255, 255, 255, 0, 0, 0, 255},
                                      true, gpu::texture_wrap::repeat, gpu::texture_wrap::repeat, true);
        const int rough = add_texture(2, 2, {0, 51, 255, 255, 0, 102, 0, 255, 0, 153, 255, 255, 0, 204, 0, 255}, false,
                                      gpu::texture_wrap::repeat, gpu::texture_wrap::repeat, true);
        const std::uint32_t textured = add_material({0.5, 0.75, 1}, 0.9, 0.6, color, rough);
        const std::uint32_t textured_plain = add_material({1, 1, 1}, 1, 1, color, rough);
        // Factors alone
        const std::array<std::uint32_t, 4> factor_only{add_material({0.9, 0.6, 0.3}, 1, 0.1),
                                                       add_material({0.2, 0.5, 0.8}, 0, 0.3),
                                                       add_material({0.2, 0.5, 0.8}, 0, 0.01),
                                                       add_material({0.7, 0.7, 0.7}, 0.5, 0.4)};

        std::vector<uv_point> points;
        for (int i = 0; i < 256; ++i)
        {
            points.push_back({(i + 0.5) / 256, 0.5, srgb_metal, 0.0f});
        }
        const std::size_t first_grid = points.size();
        for (int y = 0; y < 4; ++y)
        {
            for (int x = 0; x < 4; ++x)
            {
                const double u = (x + 0.5) / 4;
                const double v = (y + 0.5) / 4;
                points.push_back({u, v, grid_metal, 0.5f});
                points.push_back({u, v, grey_linear, 0.0f});
                points.push_back({u, v, grey_nearest, 0.0f});
            }
        }
        const std::size_t first_wrap = points.size();
        for (std::size_t w = 0; w < 3; ++w)
        {
            for (const double at : {-0.25, 1.25})
            {
                points.push_back({at, 0.5, along_u[w], 0.5f});
                points.push_back({0.5, at, along_v[w], 0.5f});
            }
        }
        const std::size_t first_nearest = points.size();
        points.push_back({0.4, 0.5, strip_nearest, 0.5f}); // texel 1.6 → texel 1
        points.push_back({0.6, 0.5, strip_nearest, 0.5f}); // texel 2.4 → texel 2
        const std::size_t first_textured = points.size();
        for (int y = 0; y < 2; ++y)
        {
            for (int x = 0; x < 2; ++x)
            {
                points.push_back({(x + 0.5) / 2, (y + 0.5) / 2, textured, 0.5f});
                points.push_back({(x + 0.5) / 2, (y + 0.5) / 2, textured_plain, 0.5f});
            }
        }
        const std::size_t first_factor = points.size();
        for (const std::uint32_t m : factor_only)
        {
            points.push_back({0.5, 0.5, m, 0.25f});
        }
        points.push_back({0.5, 0.5, factor_only[3], 0.75f});
        // Then a triangle wearing the scene's matte, and (a sample of its own, after the points') the absorber
        const std::size_t gltf_points = points.size();
        materials.push_back({.scene_material = 0});
        points.push_back({0.5, 0.5, static_cast<std::uint32_t>(materials.size() - 1), 0.5f});

        scene.meshes.meshes = {std::make_shared<const gpu::mesh_desc>(uv_points(points))};
        scene.meshes.instances = {{.mesh = 0, .object_to_world = identity}};
        scene.meshes.materials = materials;
        for (gpu::mesh_texture_desc& d : textures)
        {
            scene.meshes.textures.push_back(std::make_shared<const gpu::mesh_texture_desc>(std::move(d)));
        }
        const optix_upload upload(*runtime, scene);
        std::vector<probe_sample> samples = point_samples(points);
        probe_sample absorber;
        absorber.mode = 2;
        absorber.origin = {0.5f, 0.5f, 1.0f};
        absorber.direction = {0.0f, 0.0f, -1.0f};
        samples.push_back(absorber);
        const std::vector<probe_answer> answers = probe(ctx, *upload.scene, samples, 9122);

        // The slot of sample k is entry M + 1 + k (M scene materials, then the absorber); its texture -1
        const int scratch_base = static_cast<int>(scene.materials.size()) + 1;
        std::int64_t slot_index = 0;
        for (std::size_t k = 0; k < gltf_points; ++k)
        {
            slot_index += answers[k].material == scratch_base + static_cast<int>(k) && answers[k].texture_index == -1
                              ? 0
                              : 1;
        }
        ctx.count("slot-index", slot_index, 0, static_cast<std::int64_t>(gltf_points));

        // The tables grew from 64 slots to one per sample: what they copied of the scene's own entries. The matte's
        // triangle reads the matte (the solid 0.5 its texture folds to) and its coat (α = 0.5², r = 0.5); the
        // absorber, entry M, a black diffuse_light without a side entry, from -1
        const probe_answer& matte = answers[gltf_points];
        const bool matte_right =
            matte.material == 0 && matte.kind == static_cast<int>(gpu::material_kind::lambertian) &&
            same_bits(matte.albedo[0], 0.5f) && same_bits(matte.albedo[1], 0.5f) && same_bits(matte.albedo[2], 0.5f) &&
            matte.facet_kind == static_cast<int>(gpu::microfacet_kind::gloss) && same_bits(matte.alpha, 0.25f) &&
            same_bits(matte.roughness, 0.5f);
        ctx.count("grown-tables-scene-material", matte_right ? 0 : 1, 0, 1);
        if (!matte_right)
        {
            ctx.info("grown-tables-scene-material", std::format("material {}: {}", matte.material, describe(matte)));
        }
        const probe_answer& black = answers.back();
        const bool absorber_right =
            black.material == static_cast<int>(scene.materials.size()) &&
            black.kind == static_cast<int>(gpu::material_kind::diffuse_light) && same_bits(black.albedo[0], 0.0f) &&
            same_bits(black.albedo[1], 0.0f) && same_bits(black.albedo[2], 0.0f) &&
            black.facet_kind == static_cast<int>(gpu::microfacet_kind::none) && same_bits(black.alpha, 0.0f) &&
            same_bits(black.roughness, 0.0f) && black.from == -1 && black.surfaced;
        ctx.count("grown-tables-absorber", absorber_right ? 0 : 1, 0, 1);
        if (!absorber_right)
        {
            ctx.info("grown-tables-absorber", std::format("material {}: {}", black.material, describe(black)));
        }

        // sRGB: every byte against IEC 61966-2-1, through meshes.cu's base color texture
        double largest = 0;
        double largest_off = 0;
        int at = 0;
        std::int64_t not_metal = 0;
        for (int i = 0; i < 256; ++i)
        {
            const probe_answer& a = answers[static_cast<std::size_t>(i)];
            const double steps = std::max(
                {srgb_steps_off(a.albedo[0], i), srgb_steps_off(a.albedo[1], i), srgb_steps_off(a.albedo[2], i)});
            largest_off = std::max(largest_off, std::abs(a.albedo[0] - srgb_to_linear(i / 255.0)));
            not_metal += a.kind == static_cast<int>(gpu::material_kind::metal) ? 0 : 1;
            if (!(steps <= largest))
            {
                largest = steps;
                at = i;
            }
        }
        ctx.count("srgb-on-a-metal", not_metal, 0, 256);
        ctx.exact("srgb-decode", largest, 0, 1);
        ctx.info("srgb-decode", std::format("over the 256 bytes the decode is at most {:.3g} of a byte's half step "
                                            "from IEC 61966-2-1, at byte {}; the largest |difference| {:.3g}",
                                            largest, at, largest_off));

        // Bilinear at texel centres, linear and from the top-left: r = G(x, y)/255
        double grid_off = 0;
        double grey_apart = 0;
        double grey_off = 0;
        for (int y = 0; y < 4; ++y)
        {
            for (int x = 0; x < 4; ++x)
            {
                const std::size_t k = first_grid + 3 * static_cast<std::size_t>(4 * y + x);
                const int level = grid_g(x, y);
                const double g = level / 255.0;
                const probe_answer& metal = answers[k];
                grid_off = std::max(grid_off, metal.kind == static_cast<int>(gpu::material_kind::metal)
                                                  ? std::abs(metal.roughness - g)
                                                  : infinite);
                const probe_answer& linear = answers[k + 1];
                const probe_answer& nearest = answers[k + 2];
                grey_apart = std::max(grey_apart, apart(of(linear.albedo), of(nearest.albedo)));
                grey_off = std::max(grey_off, srgb_steps_off(linear.albedo[0], level));
            }
        }
        ctx.exact("linear-texel-centres", grid_off, 0, 1e-6); // the top-left origin: row y is the y-th from the top
        ctx.exact("bilinear-equals-nearest-at-centres", grey_apart, 0, 1e-6);
        ctx.exact("bilinear-srgb-at-centres", grey_off, 0, 1); // in halves of a byte's step (decision 13)

        // Wrap: bilinear halfway between two texels at -0.25 and 1.25 (texel coordinates -1.5 and 4.5)
        const auto g = [&strip](std::size_t i) { return strip[i] / 255.0; };
        const std::array<std::array<double, 2>, 3> wrapped{{{(g(2) + g(3)) / 2, (g(0) + g(1)) / 2},  // repeat
                                                            {(g(0) + g(1)) / 2, (g(2) + g(3)) / 2},  // mirrored repeat
                                                            {g(0), g(3)}}};                          // clamp
        const std::array<std::string_view, 3> wrap_names{"repeat", "mirrored-repeat", "clamp"};
        for (std::size_t w = 0; w < 3; ++w)
        {
            for (std::size_t side = 0; side < 2; ++side)
            {
                const std::size_t k = first_wrap + 4 * w + 2 * side;
                const std::string where = side == 0 ? "-0.25" : "1.25";
                ctx.exact(std::format("wrap-{}-u-at-{}", wrap_names[w], where), answers[k].roughness,
                          wrapped[w][side], 2e-3);
                ctx.exact(std::format("wrap-{}-v-at-{}", wrap_names[w], where), answers[k + 1].roughness,
                          wrapped[w][side], 2e-3);
            }
        }
        ctx.exact("nearest-u-0.4", answers[first_nearest].roughness, g(1), 1e-6);
        ctx.exact("nearest-u-0.6", answers[first_nearest + 1].roughness, g(2), 1e-6);

        // The mapping end to end: the factors' floats (the descs' doubles rounded) times the texels the plain material
        // reads
        std::int64_t wrong = 0;
        std::string first_wrong;
        const auto expect = [&](std::size_t k, const gpu::mesh_material_desc& d, const std::array<float, 3>& texel,
                                float texel_g, float texel_b) {
            const slot_contents e =
                mapped_slot({static_cast<float>(d.base_color[0]), static_cast<float>(d.base_color[1]),
                             static_cast<float>(d.base_color[2])},
                            static_cast<float>(d.metallic), static_cast<float>(d.roughness), texel, texel_g, texel_b,
                            points[k].lobe);
            const bool right = slot_matches(answers[k], e);
            if (!right && first_wrong.empty())
            {
                first_wrong = std::format("point {}: {}; expected {}", k, describe(answers[k]), describe(e));
            }
            wrong += right ? 0 : 1;
        };
        const std::array<std::uint8_t, 4> rough_b{255, 0, 255, 0};
        for (std::size_t c = 0; c < 4; ++c)
        {
            const std::size_t k = first_textured + 2 * c;
            const probe_answer& plain = answers[k + 1]; // factors 1: albedo = the decoded texel, r = G, B 0 or 1
            expect(k, materials[textured], plain.albedo, plain.roughness, rough_b[c] / 255.0f);
        }
        for (std::size_t k = first_factor; k < gltf_points; ++k)
        {
            expect(k, materials[points[k].material], {1, 1, 1}, 1, 1);
        }
        ctx.count("mapping-end-to-end", wrong, 0, static_cast<std::int64_t>(4 + gltf_points - first_factor));
        if (!first_wrong.empty())
        {
            ctx.info("mapping-end-to-end", first_wrong);
        }
    });
}

// A texture of `bytes` bytes, each 128, as the upload takes it
std::shared_ptr<const gpu::mesh_texture_desc> grey_texture(int width, int height, std::size_t bytes, bool srgb)
{
    return std::make_shared<const gpu::mesh_texture_desc>(
        gpu::mesh_texture_desc{.width = width, .height = height, .rgba = std::vector<std::uint8_t>(bytes, 128),
                               .srgb = srgb});
}

// The upload's refusals of glTF's materials and textures (meshes.cu), each a std::runtime_error naming its cause,
// before anything is allocated or launched, each made from one scene the upload accepts: a triangle wearing glTF's
// material 2, a base color texture 0 in sRGB and a metallic-roughness texture 1 read linearly
GPU_TEST(optix_pbr, refusals)
{
    const std::shared_ptr<gpu::optix_runtime> runtime = optix_for(ctx, "refusals");
    if (!runtime)
    {
        return;
    }
    const auto accepted = [] {
        gpu::mesh_desc m = one_triangle();
        m.material_of = {2};
        gpu::scene_desc scene = mesh_scene({m}, {{0, identity}});
        scene.meshes.materials.push_back({.scene_material = -1,
                                          .base_color = {0.5, 0.5, 0.5},
                                          .metallic = 0.5,
                                          .roughness = 0.5,
                                          .base_color_texture = 0,
                                          .metal_rough_texture = 1});
        scene.meshes.textures = {grey_texture(2, 2, 16, true), grey_texture(2, 2, 16, false)};
        return scene;
    };
    const std::string message = upload_error(*runtime, accepted());
    ctx.count("accepted", message.empty() ? 0 : 1, 0, 1);
    if (!message.empty())
    {
        ctx.info("accepted", std::format("the upload threw: {}", message));
    }
    // `change` made to the accepted scene's glTF material (mesh material 2) or its textures
    const auto refused = [&](std::string_view name, const auto& change, std::string_view cause) {
        gpu::scene_desc scene = accepted();
        change(scene.meshes.materials[2], scene.meshes.textures);
        expect_refusal(ctx, *runtime, name, scene, cause);
    };
    using textures = std::vector<std::shared_ptr<const gpu::mesh_texture_desc>>;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // The factors, the texture indices and their uses, the scene material's index
    refused("base-color-negative", [](gpu::mesh_material_desc& m, textures&) { m.base_color[1] = -0.1; },
            "has the base color");
    refused("base-color-nan", [nan](gpu::mesh_material_desc& m, textures&) { m.base_color[0] = nan; },
            "has the base color");
    refused("metallic-past-1", [](gpu::mesh_material_desc& m, textures&) { m.metallic = 1.5; },
            "the metallic factor 1.5");
    refused("roughness-below-0", [](gpu::mesh_material_desc& m, textures&) { m.roughness = -0.1; },
            "the roughness factor -0.1");
    refused("base-color-texture-past-the-set", [](gpu::mesh_material_desc& m, textures&) { m.base_color_texture = 2; },
            "the texture 2 for its base color");
    refused("metal-rough-texture-minus-2", [](gpu::mesh_material_desc& m, textures&) { m.metal_rough_texture = -2; },
            "the texture -2 for its metallic-roughness");
    refused("base-color-texture-linear",
            [](gpu::mesh_material_desc&, textures& t) { t[0] = grey_texture(2, 2, 16, false); },
            "reads textures[0] as its base color, which must be decoded from sRGB");
    refused("metal-rough-texture-srgb",
            [](gpu::mesh_material_desc&, textures& t) { t[1] = grey_texture(2, 2, 16, true); },
            "reads textures[1] as its metallic-roughness, which must be read linearly");
    refused("scene-material-minus-2", [](gpu::mesh_material_desc& m, textures&) { m.scene_material = -2; },
            "names the scene's material -2");

    // The textures themselves
    refused("texture-missing", [](gpu::mesh_material_desc&, textures& t) { t[1] = nullptr; }, "textures[1] is missing");
    refused("texture-bytes", [](gpu::mesh_material_desc&, textures& t) { t[0] = grey_texture(2, 2, 12, true); },
            "textures[0] has 12 bytes for its 2 by 2 texels");
    refused("texture-zero-wide", [](gpu::mesh_material_desc&, textures& t) { t[0] = grey_texture(0, 2, 0, true); },
            "textures[0] is 0 by 2 texels");
    refused("texture-wrap-unknown",
            [](gpu::mesh_material_desc&, textures& t) {
                gpu::mesh_texture_desc d = *t[0];
                d.wrap_t = static_cast<gpu::texture_wrap>(7);
                t[0] = std::make_shared<const gpu::mesh_texture_desc>(std::move(d));
            },
            "textures[0] has an unknown wrap mode");
    // One texel wider than the GPU's 2D textures may be (cudaDevAttrMaxTexture2DWidth), a row of them
    int device = 0;
    int widest = 0;
    gpu_test::cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    gpu_test::cuda_check(cudaDeviceGetAttribute(&widest, cudaDevAttrMaxTexture2DWidth, device),
                         "cudaDeviceGetAttribute");
    if (widest > 0 && widest < (1 << 24))
    {
        const int width = widest + 1;
        refused("texture-past-the-gpu",
                [width](gpu::mesh_material_desc&, textures& t) {
                    t[0] = grey_texture(width, 1, 4 * static_cast<std::size_t>(width), true);
                },
                std::format("textures[0] is {} by 1 texels", width));
    }
    else
    {
        ctx.info("refused-texture-past-the-gpu", std::format("not tried: the GPU's widest texture is {}", widest));
    }
}

// =====================================================================================================================
// Suite mesh-cache (Step 93): gpu_scene's model cache, on the host alone
// =====================================================================================================================

// Two builds of still-life share their meshes and textures (the same shared_ptrs: each model converted once per
// load); after the vase's file gets a new write time, the vase alone reads it again (new pointers), and still-life
// then reads nothing: the bust stayed beside the reloaded vase, the vase's stale entry gone instead of pushing the bust
// out; with the time put back, still-life reads the vase alone again (see check_model_cache). No GPU work and no OptiX
// (the translation is the host's), so it runs in a build without OptiX too, and stays in every GpuTests run (it isn't
// one of --exclude's OptiX suites).
GPU_TEST(mesh_cache, still_life)
{
    const gpu_test::mesh_host::model_cache_check c = gpu_test::mesh_host::check_model_cache();
    ctx.count("write-time-restored", c.restored ? 0 : 1, 0, 1);
    if (!c.problem.empty())
    {
        ctx.fail("model-cache", c.problem);
        return;
    }
    const auto n = [](std::size_t value) { return static_cast<std::int64_t>(value); };
    const std::int64_t meshes = n(c.meshes);
    const std::int64_t textures = n(c.textures);
    // The bust's part of still-life: its meshes and textures but the vase's, which the third build holds alone
    const std::int64_t bust_meshes = meshes - n(c.third_meshes);
    const std::int64_t bust_textures = textures - n(c.third_textures);
    ctx.count("meshes-built", c.meshes > 0 && c.null_meshes == 0 && bust_meshes > 0 && bust_textures > 0 ? 0 : 1, 0,
              1);
    ctx.count("second-build-shares-meshes", meshes - n(c.shared) + std::abs(n(c.second_meshes) - meshes), 0, meshes);
    ctx.count("second-build-shares-textures",
              textures - n(c.shared_textures) + std::abs(n(c.second_textures) - textures), 0, textures);
    ctx.count("second-build-reads-nothing", n(c.second_read), 0, 2);
    ctx.count("touched-file-loads-again", c.third_read == 1 && c.third_textures > 0 ? 0 : 1, 0, 1);
    ctx.count("touched-model-new-textures", n(c.third_old_textures), 0, n(c.third_textures));
    ctx.count("fourth-build-reads-nothing", n(c.fourth_read), 0, 2);
    ctx.count("other-model-stays-cached",
              std::abs(n(c.kept_meshes) - bust_meshes) + std::abs(n(c.kept_textures) - bust_textures), 0,
              bust_meshes + bust_textures);
    ctx.count("reloaded-model-stays-cached",
              n(c.third_textures) - n(c.reloaded_textures) + std::abs(n(c.fourth_textures) - textures) +
                  std::abs(n(c.fourth_meshes) - meshes),
              0, n(c.third_textures));
    ctx.count("stale-entry-gone",
              c.fifth_read_count == 1 && c.fifth_read.find("brass_vase_01") != std::string::npos ? 0 : 1, 0, 1);
    ctx.info("model-cache",
             std::format("{} meshes and {} textures in each build of still-life, the vase {} and {} of them; after {} "
                         "got a new write time, the vase alone read {} model(s) and still-life {}, its bust kept {} "
                         "meshes and {} textures, its vase {} textures of the reload; with the time put back "
                         "still-life read: {}",
                         c.meshes, c.textures, c.third_meshes, c.third_textures, c.touched, c.third_read,
                         c.fourth_read, c.kept_meshes, c.kept_textures, c.reloaded_textures,
                         c.fifth_read.empty() ? "nothing" : c.fifth_read));
}

// still-life's second model, the vase, wears its own materials: its triangles' material_of are shifted past the bust's
// materials (gpu_scene.ixx's meshes_at), and every triangle of both models is checked against the materials load_gltf
// reads from its model's file (the factors, and the textures' texels and color spaces), the models told apart by where
// they stand (see check_model_materials). An unshifted vase would wear the bust's marble, an index the upload accepts.
// Host only, like still-life; it reads both models' files once more (four 1k images decoded).
GPU_TEST(mesh_cache, second_model)
{
    const gpu_test::mesh_host::model_materials_check c = gpu_test::mesh_host::check_model_materials();
    if (!c.problem.empty())
    {
        ctx.fail("second-model", c.problem);
        return;
    }
    const auto n = [](std::size_t value) { return static_cast<std::int64_t>(value); };
    ctx.count("both-models-placed", (c.bust_triangles > 0 ? 0 : 1) + (c.vase_triangles > 0 ? 0 : 1), 0, 2);
    ctx.count("models-apart", n(c.both_meshes), 0, n(c.bust_meshes + c.vase_meshes));
    ctx.count("bust-wears-its-materials", n(c.bust_wrong), 0, n(c.bust_triangles));
    ctx.count("vase-wears-its-materials", n(c.vase_wrong), 0, n(c.vase_triangles));
    ctx.info("second-model", std::format("{} materials and {} textures; the bust {} mesh(es) of {} triangles, the vase "
                                         "{} mesh(es) of {} triangles",
                                         c.materials, c.textures, c.bust_meshes, c.bust_triangles, c.vase_meshes,
                                         c.vase_triangles));
}
}
