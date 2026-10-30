// Module meshes: triangle meshes as the host builds them, without CUDA: what a glTF file holds once loaded, meshes made
// in code for the tests, and the placements that put them in a scene. gpu_scene turns them into the renderer's
// mesh_set (renderer.h), which only the OptiX path traces; the CPU renderer never sees a triangle.
// Conventions are glTF 2.0's: right-handed coordinates with +y up; a triangle's front is the side from which its
// corners run counter-clockwise; a texture's (0, 0) is its image's top-left corner, except on triangulate_quads'
// meshes, whose texture coordinates are the books' quads' own (α, β), which the books' textures read with (0, 0) at
// the image's bottom-left. Placements are row-major 3×4 matrices (OptixInstance::transform's layout; glTF itself
// stores 4×4 column-major ones), the last column the translation.
export module meshes;

import std;

export namespace meshes
{
enum class wrap
{
    repeat,
    mirrored_repeat,
    clamp,
};

struct image // 8-bit RGBA, rows from the top
{
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba;
    bool srgb = true; // base color; false for metallic-roughness data
    wrap wrap_s = wrap::repeat;
    wrap wrap_t = wrap::repeat;
    bool nearest = false;
};

struct pbr_material // glTF's metallic-roughness factors and textures (indices into model::images)
{
    std::array<double, 3> base_color{1, 1, 1};
    double metallic = 1;
    double roughness = 1;
    int base_color_image = -1;
    int metal_rough_image = -1;
};

struct mesh
{
    std::vector<float> positions;           // 3 per vertex
    std::vector<float> normals;             // 3 per vertex, or empty: flat
    std::vector<float> uvs;                 // 2 per vertex, or empty
    std::vector<std::uint32_t> indices;     // 3 per triangle, counter-clockwise from the front
    std::vector<std::uint32_t> material_of; // per triangle
};

struct placement
{
    int mesh = 0;
    std::array<double, 12> object_to_world{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}; // row-major 3×4
};

struct model
{
    std::vector<mesh> meshes;
    std::vector<placement> placements;
    std::vector<pbr_material> materials;  // the last: glTF's default material, for primitives that name none
    std::vector<image> images;
    std::array<double, 3> lower{};        // world bounds of every placement
    std::array<double, 3> upper{};
    std::vector<std::string> warnings;
};

// What load_gltf refuses, before it allocates for it: file_bytes also bounds each image file, every buffer file
// together, and (twice it, plus 1 MiB) the memory of the parsed JSON; placements also bounds the nodes (4 per
// placement, plus 4096); triangles also bounds the vertices (3 per triangle)
struct limits
{
    std::uintmax_t file_bytes = 256ull << 20;
    std::size_t triangles = 8'000'000;
    std::size_t placed_triangles = 64'000'000;
    std::size_t placements = 1'000'000;
    int image_side = 4096;
    std::size_t image_bytes = 512ull << 20;
};

mesh icosphere(int level, bool smooth);                         // radius 1: V = 10·4^k + 2, T = 20·4^k
mesh torus(double major, double minor, int around, int across); // seam vertices duplicated
mesh triangulate_quads(std::span<const std::array<double, 9>> q_u_v, std::span<const std::uint32_t> material_of);
mesh bake(const mesh& m, const std::array<double, 12>& object_to_world); // world vertices, normals by M^-T; indices
                                                                         // 1 and 2 swapped when det M < 0
// Uniform scale to a bounding sphere of radius 1, centered over the origin, its lowest point on y = 0
std::array<double, 12> unit_placement(const std::array<double, 3>& lower, const std::array<double, 3>& upper);
}

// --- The generators --------------------------------------------------------------------------------------------------
// Everything is computed in double and rounded to float once, when stored. Invalid arguments throw
// std::invalid_argument: the generators are called by code with constant arguments (the scene list, the tests), where
// a mistake is the program's, not the user's.

namespace meshes
{
namespace
{
using vec = std::array<double, 3>;

vec add(const vec& a, const vec& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
vec sub(const vec& a, const vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const vec& a, const vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
vec cross(const vec& a, const vec& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
vec unit(const vec& a)
{
    const double length = std::sqrt(dot(a, a));
    return {a[0] / length, a[1] / length, a[2] / length};
}

void push(std::vector<float>& out, const vec& a)
{
    out.push_back(static_cast<float>(a[0]));
    out.push_back(static_cast<float>(a[1]));
    out.push_back(static_cast<float>(a[2]));
}

vec vertex(const std::vector<float>& floats, std::size_t i)
{
    return {floats[3 * i], floats[3 * i + 1], floats[3 * i + 2]};
}

// The linear part of a row-major 3×4 matrix times a, and the whole matrix (with its translation) times a point
vec linear(const std::array<double, 12>& m, const vec& a)
{
    return {m[0] * a[0] + m[1] * a[1] + m[2] * a[2], m[4] * a[0] + m[5] * a[1] + m[6] * a[2],
            m[8] * a[0] + m[9] * a[1] + m[10] * a[2]};
}
vec point(const std::array<double, 12>& m, const vec& p)
{
    const vec l = linear(m, p);
    return {l[0] + m[3], l[1] + m[7], l[2] + m[11]};
}

double determinant(const std::array<double, 12>& m)
{
    return m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
           m[2] * (m[4] * m[9] - m[5] * m[8]);
}

// M⁻ᵀ = cof(M) / det M, row-major 3×3: the matrix normals are carried by. Under a mirror (det M < 0) it still maps an
// outward normal to the outward side of the mirrored surface, since (Ma)×(Mb) = det M · M⁻ᵀ(a×b).
std::array<double, 9> inverse_transpose(const std::array<double, 12>& m, double det)
{
    return {(m[5] * m[10] - m[6] * m[9]) / det, (m[6] * m[8] - m[4] * m[10]) / det, (m[4] * m[9] - m[5] * m[8]) / det,
            (m[2] * m[9] - m[1] * m[10]) / det, (m[0] * m[10] - m[2] * m[8]) / det, (m[1] * m[8] - m[0] * m[9]) / det,
            (m[1] * m[6] - m[2] * m[5]) / det, (m[2] * m[4] - m[0] * m[6]) / det, (m[0] * m[5] - m[1] * m[4]) / det};
}
}

// A level-k icosphere: the icosahedron with each triangle cut into four k times, the new vertices pushed out onto the
// unit sphere. Vertices are shared between triangles (the midpoint of an edge is made once, for both triangles that
// have that edge), so each level turns T triangles and E = 3T/2 edges into 4T triangles and V + E vertices: from 12
// vertices and 20 triangles, V = 10·4^k + 2 and T = 20·4^k. Flat (smooth false): no normals, so the renderer shades
// each face with its own normal (glTF's rule); smooth: the normal of a point of the unit sphere is the point itself.
mesh icosphere(int level, bool smooth)
{
    if (level < 0 || level > 9)
    {
        throw std::invalid_argument(
            std::format("icosphere: level {} is outside 0 to 9 (level 9 already has 5,242,880 triangles)", level));
    }
    // The icosahedron's 12 vertices: the cyclic permutations of (0, ±1, ±φ), φ the golden ratio, on the unit sphere
    const double phi = (1 + std::sqrt(5.0)) / 2;
    std::vector<vec> points{{-1, phi, 0}, {1, phi, 0},  {-1, -phi, 0}, {1, -phi, 0}, {0, -1, phi},  {0, 1, phi},
                            {0, -1, -phi}, {0, 1, -phi}, {phi, 0, -1},  {phi, 0, 1},  {-phi, 0, -1}, {-phi, 0, 1}};
    for (vec& p : points)
    {
        p = unit(p);
    }
    std::vector<std::array<std::uint32_t, 3>> faces{{0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
                                                    {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                                                    {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
                                                    {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};
    // Every face turned so that its corners run counter-clockwise seen from outside: its normal (e1 × e2) then has a
    // positive dot product with any of its corners, since the solid is convex and holds the origin
    for (auto& [a, b, c] : faces)
    {
        if (dot(cross(sub(points[b], points[a]), sub(points[c], points[a])), points[a]) < 0)
        {
            std::swap(b, c);
        }
    }

    for (int k = 0; k < level; ++k)
    {
        // An edge's midpoint by its two corners, the smaller index first: both triangles of the edge find the same one
        std::unordered_map<std::uint64_t, std::uint32_t> midpoints;
        midpoints.reserve(faces.size() * 3 / 2);
        const auto midpoint = [&](std::uint32_t a, std::uint32_t b) {
            const std::uint64_t key = (std::uint64_t{std::min(a, b)} << 32) | std::max(a, b);
            const auto [found, added] = midpoints.try_emplace(key, static_cast<std::uint32_t>(points.size()));
            if (added)
            {
                points.push_back(unit(add(points[a], points[b])));
            }
            return found->second;
        };
        // The four triangles keep their parent's turning: each corner triangle lists its corner, then the midpoints
        // toward the next and the previous corner, in the parent's order, and the middle one the three midpoints
        std::vector<std::array<std::uint32_t, 3>> finer;
        finer.reserve(faces.size() * 4);
        for (const auto& [a, b, c] : faces)
        {
            const std::uint32_t ab = midpoint(a, b);
            const std::uint32_t bc = midpoint(b, c);
            const std::uint32_t ca = midpoint(c, a);
            finer.push_back({a, ab, ca});
            finer.push_back({b, bc, ab});
            finer.push_back({c, ca, bc});
            finer.push_back({ab, bc, ca});
        }
        faces = std::move(finer);
    }

    mesh m;
    m.positions.reserve(points.size() * 3);
    for (const vec& p : points)
    {
        push(m.positions, p);
    }
    if (smooth)
    {
        m.normals = m.positions;
    }
    m.indices.reserve(faces.size() * 3);
    for (const auto& [a, b, c] : faces)
    {
        m.indices.insert(m.indices.end(), {a, b, c});
    }
    m.material_of.assign(faces.size(), 0);
    return m;
}

// A ring torus around the y axis: the circle of radius `major` in the xz plane, swept by a tube of radius `minor`;
// `around` segments along the ring, `across` around the tube, two triangles per cell, smooth normals and texture
// coordinates (u along the ring, v around the tube, both from 0 to 1). The point at angles θ (ring) and φ (tube) is
// p = ((R + r cos φ) cos θ, r sin φ, (R + r cos φ) sin θ), its outward normal n = (cos φ cos θ, sin φ, cos φ sin θ),
// and ∂p/∂φ × ∂p/∂θ = r (R + r cos φ) n: so a cell's corners run counter-clockwise from outside in the order
// (θ, φ) → (θ, φ + dφ) → (θ + dθ, φ + dφ). The last row and column of vertices repeat the first ones with u = 1 or
// v = 1, for the texture's seam: their angles are taken modulo the circle, so the copies hold the very same floats.
mesh torus(double major, double minor, int around, int across)
{
    if (!(minor > 0 && major > minor && std::isfinite(major)) || around < 3 || across < 3 ||
        static_cast<std::int64_t>(around) * across > 4'000'000)
    {
        throw std::invalid_argument(std::format("torus: needs 0 < minor < major and 3 to 4,000,000 cells of at least "
                                                "3 a side, not minor {}, major {}, {} × {} cells",
                                                minor, major, around, across));
    }
    mesh m;
    const auto columns = static_cast<std::size_t>(across) + 1;
    const std::size_t vertices = (static_cast<std::size_t>(around) + 1) * columns;
    m.positions.reserve(vertices * 3);
    m.normals.reserve(vertices * 3);
    m.uvs.reserve(vertices * 2);
    for (int i = 0; i <= around; ++i)
    {
        const double theta = 2 * std::numbers::pi * (i % around) / around;
        for (int j = 0; j <= across; ++j)
        {
            const double phi = 2 * std::numbers::pi * (j % across) / across;
            const double ring = major + minor * std::cos(phi);
            push(m.positions, {ring * std::cos(theta), minor * std::sin(phi), ring * std::sin(theta)});
            push(m.normals, {std::cos(phi) * std::cos(theta), std::sin(phi), std::cos(phi) * std::sin(theta)});
            m.uvs.push_back(static_cast<float>(static_cast<double>(i) / around));
            m.uvs.push_back(static_cast<float>(static_cast<double>(j) / across));
        }
    }
    const std::size_t cells = static_cast<std::size_t>(around) * across;
    m.indices.reserve(cells * 6);
    const auto at = [&](int i, int j) {
        return static_cast<std::uint32_t>(static_cast<std::size_t>(i) * columns + static_cast<std::size_t>(j));
    };
    for (int i = 0; i < around; ++i)
    {
        for (int j = 0; j < across; ++j)
        {
            m.indices.insert(m.indices.end(), {at(i, j), at(i, j + 1), at(i + 1, j + 1)});
            m.indices.insert(m.indices.end(), {at(i, j), at(i + 1, j + 1), at(i + 1, j)});
        }
    }
    m.material_of.assign(cells * 2, 0);
    return m;
}

// The books' quads (corner q, edges u and v) as two triangles each: corners q, q + u, q + u + v, q + v, triangles
// (0, 1, 2) and (0, 2, 3), both with e1 × e2 = u × v, the quad's front; texture coordinates the quad's own (α, β) of
// p = q + αu + βv, so a books' texture lies on the triangles as on the quad. Flat: no normals. Every quad has four
// vertices of its own (its texture coordinates are its own) and wears material_of[i] (all 0 when the span is empty).
mesh triangulate_quads(std::span<const std::array<double, 9>> q_u_v, std::span<const std::uint32_t> material_of)
{
    if (!material_of.empty() && material_of.size() != q_u_v.size())
    {
        throw std::invalid_argument(std::format("triangulate_quads: {} quads but {} materials", q_u_v.size(),
                                                material_of.size()));
    }
    if (q_u_v.size() > 4'000'000)
    {
        throw std::invalid_argument(std::format("triangulate_quads: {} quads, more than 4,000,000", q_u_v.size()));
    }
    mesh m;
    m.positions.reserve(q_u_v.size() * 12);
    m.uvs.reserve(q_u_v.size() * 8);
    m.indices.reserve(q_u_v.size() * 6);
    m.material_of.reserve(q_u_v.size() * 2);
    for (std::size_t i = 0; i < q_u_v.size(); ++i)
    {
        const auto& quad = q_u_v[i];
        const vec q{quad[0], quad[1], quad[2]};
        const vec u{quad[3], quad[4], quad[5]};
        const vec v{quad[6], quad[7], quad[8]};
        push(m.positions, q);
        push(m.positions, add(q, u));
        push(m.positions, add(add(q, u), v));
        push(m.positions, add(q, v));
        m.uvs.insert(m.uvs.end(), {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
        const auto first = static_cast<std::uint32_t>(4 * i);
        m.indices.insert(m.indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
        const std::uint32_t material = material_of.empty() ? 0 : material_of[i];
        m.material_of.insert(m.material_of.end(), {material, material});
    }
    return m;
}

// A mesh moved into the world once and for all: the reference an instanced copy of it must reproduce. Positions by
// the whole matrix, normals by M⁻ᵀ and made unit again (a zero normal stays zero). A baked triangle sits under an
// identity instance, where the renderer takes its outside to be e1 × e2 of the stored corners; under a mirror that is
// (Ma)×(Mb) = det M · M⁻ᵀ(a×b), the inside, while the instanced copy's is M⁻ᵀ(a×b): so when det M < 0 corners 1 and 2
// of every triangle trade places (glTF's winding rule for mirrored nodes), and the baked copy's outside, front and
// smooth normals are the instanced copy's. Texture coordinates and materials don't change.
mesh bake(const mesh& m, const std::array<double, 12>& object_to_world)
{
    const double det = determinant(object_to_world);
    if (!std::isfinite(det) || det == 0)
    {
        throw std::invalid_argument(std::format("bake: the transform is singular or not finite (determinant {})", det));
    }
    mesh out = m;
    for (std::size_t i = 0; i < m.positions.size() / 3; ++i)
    {
        const vec p = point(object_to_world, vertex(m.positions, i));
        out.positions[3 * i] = static_cast<float>(p[0]);
        out.positions[3 * i + 1] = static_cast<float>(p[1]);
        out.positions[3 * i + 2] = static_cast<float>(p[2]);
    }
    const std::array<double, 9> n = inverse_transpose(object_to_world, det);
    for (std::size_t i = 0; i < m.normals.size() / 3; ++i)
    {
        const vec a = vertex(m.normals, i);
        vec b{n[0] * a[0] + n[1] * a[1] + n[2] * a[2], n[3] * a[0] + n[4] * a[1] + n[5] * a[2],
              n[6] * a[0] + n[7] * a[1] + n[8] * a[2]};
        if (const double length = std::sqrt(dot(b, b)); length > 0)
        {
            b = {b[0] / length, b[1] / length, b[2] / length};
        }
        out.normals[3 * i] = static_cast<float>(b[0]);
        out.normals[3 * i + 1] = static_cast<float>(b[1]);
        out.normals[3 * i + 2] = static_cast<float>(b[2]);
    }
    if (det < 0)
    {
        for (std::size_t t = 0; t + 2 < out.indices.size(); t += 3)
        {
            std::swap(out.indices[t + 1], out.indices[t + 2]);
        }
    }
    return out;
}

// The box's bounding sphere (center the box's, radius half its diagonal) scaled to radius 1, its center moved over the
// origin (x = z = 0) and the box's floor (lower y) onto y = 0: a model of any size and origin then stands on the
// ground at the origin, about 2 units tall at most. The translation is −s·(the point moved to 0), so the floor lands on
// exactly 0.
std::array<double, 12> unit_placement(const std::array<double, 3>& lower, const std::array<double, 3>& upper)
{
    const double radius = 0.5 * std::hypot(upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]);
    if (!(radius > 0 && std::isfinite(radius)) || lower[0] > upper[0] || lower[1] > upper[1] || lower[2] > upper[2])
    {
        throw std::invalid_argument(std::format("unit_placement: the box ({}, {}, {}) to ({}, {}, {}) has no size",
                                                lower[0], lower[1], lower[2], upper[0], upper[1], upper[2]));
    }
    const double s = 1 / radius;
    const double center_x = (lower[0] + upper[0]) / 2;
    const double center_z = (lower[2] + upper[2]) / 2;
    return {s, 0, 0, -s * center_x, 0, s, 0, -s * lower[1], 0, 0, s, -s * center_z};
}
}
