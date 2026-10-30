// Triangles as the integrator sees them: a mesh hit's surface, the ray origin moved off a face, the absorber
//
// Track B's file. Step 91: triangle_surface for the scene's materials, offset_origin and absorber_surface; Step 92 adds
// glTF's materials (wears_gltf, and the mapping into the pixel's scratch slot). No OptiX header: GpuTests' plain CUDA
// kernels call these functions too. Every number here is a float, as the quality kernel's.
#pragma once
#include "camera.cuh"
#include "optix_params.h"

namespace gpu
{
// What the OptiX tracer remembers between its calls in one step of a path (trace_quality_ray calls closest, then
// surface_of for that hit, then at most one visibility, then the next closest: quality_path.cuh:88-186): a triangle's
// hit data, which bvh_hit has no room for, and the face the next rays leave. It lives in the raygen's frame.
struct triangle_memo
{
    int instance = 0;           // closest(): the hit triangle's instance (an index into mesh_tables::instances)
    unsigned int primitive = 0; // its index in its mesh
    float b1 = 0;               // its barycentric coordinates: the weights of corners 1 and 2
    float b2 = 0;
    float lobe = 0;             // closest(): the number a glTF material chooses its lobe with
    vec3<float> facing{};       // surface_of(): the face's unit normal on the side the ray came from
};

// --- A triangle's numbers -------------------------------------------------------------------------------------------

// e1 × e2 of a triangle's edges, by a name of its own: ggx.cuh has a cross() in this namespace, and a call that saw
// both would be ambiguous
__device__ inline vec3<float> edge_cross(const vec3<float>& a, const vec3<float>& b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__device__ inline vec3<float> to_vec3(const float3& v)
{
    return {v.x, v.y, v.z};
}

// A point of the mesh's own coordinates in the world: its instance's row-major 3×4 matrix, the translation last
__device__ inline point3<float> place_point(const float (&m)[12], const point3<float>& p)
{
    return {m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
            m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]};
}

// A normal of the mesh's own coordinates in the world: the inverse transpose M⁻ᵀ, row-major 3×3; not made unit
__device__ inline vec3<float> place_normal(const float (&n)[9], const vec3<float>& v)
{
    return {n[0] * v.x + n[1] * v.y + n[2] * v.z, n[3] * v.x + n[4] * v.y + n[5] * v.z,
            n[6] * v.x + n[7] * v.y + n[8] * v.z};
}

// v made unit where it can be: false for a zero, an infinite or a NaN vector (NaN fails both comparisons)
__device__ inline bool normalized(vec3<float>& v)
{
    const float length_squared = v.length_squared();
    if (!(length_squared > 0.0f && length_squared < infinity<float>))
    {
        return false;
    }
    v = rsqrt(length_squared) * v;
    return true;
}

// The hit triangle wears a glTF material (its lobe needs a number). False until Step 92: the upload refuses glTF's
// materials until then, so every triangle wears one of the scene's.
__device__ inline bool wears_gltf(const mesh_tables& /*m*/, const triangle_memo& /*memo*/)
{
    return false;
}

// A triangle's surface, from closest()'s memo (instance, primitive, barycentrics) and the ray that hit it:
// - the point p = M·(b0·p0 + b1·p1 + b2·p2), from the barycentrics rather than o + t·d, whose error grows with t; the
//   traversal of a ray leaving a triangle starts at the moved origin (offset_origin) anyway, so its t isn't measured
//   from r.origin;
// - the geometric normal ng = unit(M⁻ᵀ·(e1 × e2)), on the side the corners run counter-clockwise from. A mirroring M
//   keeps it outside, since (Ma)×(Mb) = det M · M⁻ᵀ(a×b): neither OptiX's FLIP_TRIANGLE_FACING nor a swap of the
//   indices is needed. The ray hit the front when d·ng < 0; `facing` is ng turned toward the ray (the face it came
//   from);
// - the shading normal ns = unit(M⁻ᵀ·Σ bᵢnᵢ), reversed on a back hit (glTF's double-sided rule: the back's normal is
//   the front's reversed), and `facing` instead where the mesh has no normals (glTF: "MUST calculate flat normals"),
//   where it isn't finite or is zero, and where it leans past the face (ns·facing ≤ 0) or away from the viewer
//   (ns·d ≥ 0), which no lobe could shade;
// - outward = facing, so that the integrator's `inward` (the next direction · outward < 0) means "through the face the
//   ray came from", the through-rule's case (optix_tracer.cuh); and memo.facing = facing, the face the next rays leave;
// - the texture coordinates interpolated (0 without them), the material the triangle wears, from = the hit's id.
// An instance's M is its object_to_world, M⁻ᵀ its normal_to_world (meshes.cu computes both in double).
__device__ inline surface<float> triangle_surface(const optix_params& p, const ray<float>& r, const bvh_hit<float>& h,
                                                  triangle_memo& memo, int /*slot*/)
{
    const mesh_tables& m = p.meshes;
    const mesh_instance& instance = m.instances[memo.instance];
    const mesh_part part = m.parts[instance.part];
    const unsigned int triangle = part.first_triangle + memo.primitive;
    const uint3 corners = m.triangles[triangle];
    const unsigned int i0 = part.first_vertex + corners.x;
    const unsigned int i1 = part.first_vertex + corners.y;
    const unsigned int i2 = part.first_vertex + corners.z;
    const vec3<float> p0 = to_vec3(m.positions[i0]);
    const vec3<float> p1 = to_vec3(m.positions[i1]);
    const vec3<float> p2 = to_vec3(m.positions[i2]);
    const float b1 = memo.b1;
    const float b2 = memo.b2;
    const float b0 = 1.0f - b1 - b2;

    surface<float> s{};
    s.rec.p = place_point(instance.object_to_world, b0 * p0 + b1 * p1 + b2 * p2);
    s.texture_point = s.rec.p;
    s.from = h.index;

    // A triangle too thin for its cross product in float can't be hit (OptiX drops degenerate triangles); should one
    // be, it faces the ray
    vec3<float> ng = place_normal(instance.normal_to_world, edge_cross(p1 - p0, p2 - p0));
    if (!normalized(ng))
    {
        ng = -unit_vector(r.direction);
    }
    const bool front = dot(ng, r.direction) < 0.0f;
    const vec3<float> facing = front ? ng : -ng;

    vec3<float> ns = facing;
    if (part.has_normals != 0u)
    {
        vec3<float> shading =
            place_normal(instance.normal_to_world, b0 * to_vec3(m.normals[i0]) + b1 * to_vec3(m.normals[i1]) +
                                                       b2 * to_vec3(m.normals[i2]));
        if (normalized(shading))
        {
            shading = front ? shading : -shading;
            if (dot(shading, facing) > 0.0f && dot(shading, r.direction) < 0.0f)
            {
                ns = shading;
            }
        }
    }
    s.rec.normal = ns;
    s.rec.front_face = front;
    s.outward = facing;
    memo.facing = facing;

    if (part.has_uvs != 0u)
    {
        const float2 t0 = m.uvs[i0];
        const float2 t1 = m.uvs[i1];
        const float2 t2 = m.uvs[i2];
        s.u = b0 * t0.x + b1 * t1.x + b2 * t2.x;
        s.v = b0 * t0.y + b1 * t1.y + b2 * t2.y;
    }

    // One of the scene's materials, read as surface_at reads a primitive's: its texture's index from the books'
    // material_textures (-1 for one solid color, which the material's albedo then holds). A test's hand-made view may
    // have no such table: its materials are then taken as solid colors.
    s.material = m.materials[m.material_of[triangle]].scene_material;
    s.texture_index =
        p.view.base.material_textures != nullptr ? p.view.base.material_textures[s.material].texture_index : -1;
    return s;
}

// The absorber's surface (the through-rule, optix_tracer.cuh): a ray leaving a triangle through the face it came from
// carries no light from an opaque face. It wears the absorber material, a black diffuse_light that meshes.cu appends
// to the scene's materials, seen from its back: the integrator's lamp branch then ends the path with the light found
// so far (quality_path.cuh: c is 0; lamp_of[triangle_base] is -1, so the weight stays finite). Its normal faces the
// ray, −unit(d), as the denoiser's guides read a lamp's normal. From -1: there is nothing to leave.
__device__ inline surface<float> absorber_surface(const mesh_tables& m, const ray<float>& r)
{
    surface<float> s{};
    s.rec.p = r.origin;
    s.rec.normal = -unit_vector(r.direction);
    s.rec.front_face = false;
    s.outward = s.rec.normal;
    s.texture_point = r.origin;
    s.material = m.absorber_material;
    s.texture_index = -1;
    s.from = -1;
    return s;
}

// The origin of a ray leaving a triangle, moved off the face toward n (the face's unit normal on the side the ray goes
// to), so that the traversal, which starts there with tmin 0, can't meet the face again: p + δ·n, with
// δ = 2⁻¹⁶·max(|p|∞, 1), the same δ for every coordinate.
//
// δ follows the point's largest coordinate, not each coordinate's own size, because the hardware's triangle test does:
// it takes the corners relative to the ray's origin and shears them along the ray (Woop, Benthin and Wald,
// "Watertight Ray/Triangle Intersection", JCGT 2013), in float, so the side of a face it finds the origin on is
// rounded at the size of the largest of those coordinates, whichever axis the face looks along. Wächter and Binder's
// offset ("A Fast and Robust Method for Avoiding Self-Intersection", Ray Tracing Gems, chapter 6, 2019), which this
// function first followed, moves each coordinate by 256 of its own float steps, or by nᵢ/65536 below 1/32. On
// cornell-triangles' floor (y = 0, x and z up to 555, whose float steps are 6·10⁻⁵) that put the origin 1.5·10⁻⁵ above
// the floor, inside the test's rounding: the test found the floor again, or the tall block's bottom lying in the same
// plane, at t of 10⁻⁶ to 10⁻⁵, from below, and 15% of the paths started inside the closed block left it under the
// floor.
//
// 2⁻¹⁶·|p|∞ is 128 to 256 float steps of the largest coordinate (the old offset gave a coordinate 256 of its own). Each
// coordinate moves toward the side nᵢ points to or not at all (a sum that rounds back), and the largest component of
// a unit n, at least 1/√3, moves by more than 70 float steps of its coordinate: the result lies strictly on n's side of
// the face's plane. A coordinate moves by at most δ and half a float step, under 2⁻¹⁵·|p|∞ + 2⁻¹⁵. The price is a
// blind zone of δ along n: geometry closer than that to a face isn't seen from it. (A triangle far larger than |p|∞, a
// ground of 10⁴ under a point near the origin, would round at its own size: no scene has one.)
__device__ inline point3<float> offset_origin(const point3<float>& p, const vec3<float>& n)
{
    const float delta = 0x1p-16f * fmaxf(fmaxf(fabsf(p.x), fabsf(p.y)), fmaxf(fabsf(p.z), 1.0f));
    return {p.x + delta * n.x, p.y + delta * n.y, p.z + delta * n.z};
}
}
