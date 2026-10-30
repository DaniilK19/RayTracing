// MeshTests: the host side of the triangle meshes (library mesh_io: modules meshes and gltf), tested from their
// specification without a GPU and without CUDA. It checks
//   - load_gltf on small glTF 2.0 files the program writes into a temporary folder, removed at the end: JSON .gltf
//     files with base64 data: buffers, one .glb built in memory, one .gltf whose .bin and .png lie beside it in a
//     folder with a non-ASCII name; a few kB each (24 kB at most), no binary fixture. Positions, normals (or none:
//     flat), UVs, u8/u16/u32 and no indices, quantized, normalized, interleaved and sparse accessors (one over an
//     interleaved view), strips and fans, primitives sharing their vertices; the node walk (T·R·S, a parent and its
//     child, a matrix, a mirror whose face normals stay outside, a mesh shared by two nodes, the world bounds, a file
//     without scenes, one without nodes); the triangles it drops with a warning (a non-finite corner, zero area); what
//     it refuses (required compression and unknown extensions, an index past the vertices, node cycles through and off
//     the scene, sizes that wrap modulo 2^64 in cgltf's own checks, URIs that aren't UTF-8, missing files, and every
//     limit with tiny caps on both sides of its boundary: the file, a .bin and a .png beside it, the buffer files
//     together, cgltf's memory, the nodes, counts near 2^63); materials and images (the factors, glTF's default
//     material, a 2×2 PNG decoded to its exact bytes, the samplers' wrap and filter, an image used twice decoded once,
//     a material nobody wears not decoded) and a warning for each feature the renderer ignores (a texture it can't
//     read where it is worn: the factor alone);
//   - the meshes made in code: icosphere, torus, triangulate_quads, unit_placement and bake (its mirror rule).
// The committed models join in Step 93 (the TODO section near the end). Nothing here is random: no seeds.
// Each check prints one line, as GpuTests does, then a summary:
//   <group>.<check>: <what was found>, expected <what the specification says>, PASS|FAIL
// Run: `ctest -R mesh_tests` in the build folder (CTest's mesh_tests), or MeshTests itself. Exit code: 0 every check
// passes, 1 a check fails, 2 an error of the program itself (its temporary folder can't be written, or the run takes
// over 20 s: a loader that loops).
import std;
import meshes;
import gltf;

// --- The report -----------------------------------------------------------------------------------------------------

// The checks' lines, one each, counted for the summary
class report
{
public:
    // A property: what was found, and what was expected
    void check(std::string_view name, bool pass, std::string_view found, std::string_view expected)
    {
        line(name, std::format("{}, expected {}", found, expected), pass);
    }
    // A number against its exact value, within a tolerance (NaN and infinity fail)
    void exact(std::string_view name, double value, double expected, double tolerance)
    {
        const double difference = std::abs(value - expected);
        line(name,
             std::format("{:.9g}, expected {:.9g}, |difference| = {:.3g} (<= {:.3g})", value, expected, difference,
                         tolerance),
             difference <= tolerance);
    }
    // A count against its expected value
    template <std::integral A, std::integral B>
    void equal(std::string_view name, A value, B expected)
    {
        line(name, std::format("{}, expected {}", value, expected), std::cmp_equal(value, expected));
    }
    void info(std::string_view name, std::string_view text) { std::println("{}: {}, INFO", name, text); }

    // A refusal's message. The loader starts every error with its file ("<file>: meshes[3].primitives[1]: …"): one line
    // at the end counts the messages that name it
    void error_message(std::string_view check_name, bool names_file)
    {
        ++messages_;
        if (!names_file)
        {
            unnamed_.emplace_back(check_name);
        }
    }
    void error_summary()
    {
        std::string missing;
        for (const std::string& name : unnamed_)
        {
            missing += std::format("{}{}", missing.empty() ? " (not: " : ", ", name);
        }
        check("errors.name-their-file", unnamed_.empty(),
              std::format("{} of {} messages{}", messages_ - unnamed_.size(), messages_,
                          missing.empty() ? std::string() : missing + ")"),
              "every message names its file");
    }

    int checks() const { return checks_; }
    int failures() const { return failures_; }

private:
    void line(std::string_view name, std::string_view detail, bool pass)
    {
        std::println("{}: {}, {}", name, detail, pass ? "PASS" : "FAIL");
        ++checks_;
        failures_ += pass ? 0 : 1;
    }

    int checks_ = 0;
    int failures_ = 0;
    std::size_t messages_ = 0;
    std::vector<std::string> unnamed_;
};

// A failure of the test itself (a file it can't write): the run ends with exit code 2
struct setup_error
{
    std::string message;
};

// --- Vectors and matrices, in double --------------------------------------------------------------------------------

using vec = std::array<double, 3>;
using matrix = std::array<double, 12>; // row-major 3×4, as meshes::placement's object_to_world
using triangle = std::array<vec, 3>;

constexpr matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
constexpr double no_match = std::numeric_limits<double>::infinity(); // a deviation that can't be measured

vec add(const vec& a, const vec& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
vec sub(const vec& a, const vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
vec mul(double s, const vec& a) { return {s * a[0], s * a[1], s * a[2]}; }
double dot(const vec& a, const vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
vec cross(const vec& a, const vec& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double length(const vec& a) { return std::sqrt(dot(a, a)); }
double gap(const vec& a, const vec& b) { return length(sub(a, b)); }
vec unit(const vec& a) { return mul(1 / length(a), a); }

// The larger of two deviations, NaN winning (std::max drops a NaN in its second argument)
double worse(double a, double b)
{
    return std::isnan(a) || std::isnan(b) ? std::numeric_limits<double>::quiet_NaN() : std::max(a, b);
}

vec point(const matrix& m, const vec& p)
{
    return {m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3], m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
            m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]};
}
vec column(const matrix& m, std::size_t j) { return {m[j], m[4 + j], m[8 + j]}; }
double determinant(const matrix& m) { return dot(column(m, 0), cross(column(m, 1), column(m, 2))); }

// M⁻ᵀ·n from the columns cᵢ of M's 3×3 part: M⁻¹'s rows are (c₁×c₂, c₂×c₀, c₀×c₁)/det M, so these are M⁻ᵀ's columns
vec normal_to_world(const matrix& m, const vec& n)
{
    const vec c0 = column(m, 0), c1 = column(m, 1), c2 = column(m, 2);
    return mul(1 / determinant(m),
               add(add(mul(n[0], cross(c1, c2)), mul(n[1], cross(c2, c0))), mul(n[2], cross(c0, c1))));
}

matrix moved(double x, double y, double z) { return {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z}; }

// Rodrigues: R = cos θ·I + sin θ·[k]× + (1 − cos θ)·k kᵀ about the unit axis k, then the translation t
matrix rotation(const vec& axis, double degrees, const vec& t)
{
    const vec k = unit(axis);
    const double a = degrees * std::numbers::pi / 180, c = std::cos(a), s = std::sin(a);
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
    m[3] = t[0];
    m[7] = t[1];
    m[11] = t[2];
    return m;
}

// --- Meshes as the module delivers them -----------------------------------------------------------------------------

vec vertex(std::span<const float> values, std::size_t i)
{
    return {values[3 * i], values[3 * i + 1], values[3 * i + 2]};
}
std::size_t triangle_count(const meshes::mesh& m) { return m.indices.size() / 3; }
triangle corners(const meshes::mesh& m, std::size_t t)
{
    return {vertex(m.positions, m.indices[3 * t]), vertex(m.positions, m.indices[3 * t + 1]),
            vertex(m.positions, m.indices[3 * t + 2])};
}
vec face_normal(const triangle& p) { return cross(sub(p[1], p[0]), sub(p[2], p[0])); } // twice the area long
vec centroid(const triangle& p) { return mul(1.0 / 3, add(add(p[0], p[1]), p[2])); }

double area(const meshes::mesh& m)
{
    double sum = 0;
    for (std::size_t t = 0; t < triangle_count(m); ++t)
    {
        sum += length(face_normal(corners(m, t))) / 2;
    }
    return sum;
}

// What makes a mesh unusable for the checks (an index past the vertices, arrays of the wrong length), or nothing.
// `materials` > 0 also bounds material_of.
std::string mesh_problem(const meshes::mesh& m, std::size_t materials)
{
    if (m.positions.size() % 3 != 0 || m.indices.size() % 3 != 0)
    {
        return std::format("{} position floats and {} indices: not multiples of 3", m.positions.size(),
                           m.indices.size());
    }
    const std::size_t vertices = m.positions.size() / 3;
    if (!m.normals.empty() && m.normals.size() != m.positions.size())
    {
        return std::format("{} normal floats for {} vertices", m.normals.size(), vertices);
    }
    if (!m.uvs.empty() && m.uvs.size() != 2 * vertices)
    {
        return std::format("{} UV floats for {} vertices", m.uvs.size(), vertices);
    }
    if (m.material_of.size() != triangle_count(m))
    {
        return std::format("material_of has {} entries for {} triangles", m.material_of.size(), triangle_count(m));
    }
    for (const std::uint32_t i : m.indices)
    {
        if (i >= vertices)
        {
            return std::format("index {} with {} vertices", i, vertices);
        }
    }
    if (materials > 0)
    {
        for (const std::uint32_t k : m.material_of)
        {
            if (k >= materials)
            {
                return std::format("material {} of {}", k, materials);
            }
        }
    }
    return {};
}

std::string model_problem(const meshes::model& m)
{
    for (std::size_t i = 0; i < m.meshes.size(); ++i)
    {
        if (!m.meshes[i].indices.empty() && m.materials.empty())
        {
            return std::format("meshes[{}] has triangles and the model no material (not even glTF's default)", i);
        }
        if (const std::string problem = mesh_problem(m.meshes[i], m.materials.size()); !problem.empty())
        {
            return std::format("meshes[{}]: {}", i, problem);
        }
    }
    for (std::size_t i = 0; i < m.placements.size(); ++i)
    {
        const int k = m.placements[i].mesh;
        if (k < 0 || static_cast<std::size_t>(k) >= m.meshes.size())
        {
            return std::format("placements[{}] places mesh {} of {}", i, k, m.meshes.size());
        }
    }
    const int images = static_cast<int>(m.images.size());
    for (std::size_t i = 0; i < m.materials.size(); ++i)
    {
        const meshes::pbr_material& p = m.materials[i];
        if (p.base_color_image < -1 || p.base_color_image >= images || p.metal_rough_image < -1 ||
            p.metal_rough_image >= images)
        {
            return std::format("materials[{}] names images {} and {} of {}", i, p.base_color_image,
                               p.metal_rough_image, images);
        }
    }
    for (std::size_t i = 0; i < m.images.size(); ++i)
    {
        const meshes::image& img = m.images[i];
        const std::size_t texels = static_cast<std::size_t>(std::max(img.width, 0)) *
                                   static_cast<std::size_t>(std::max(img.height, 0));
        if (!img.rgba.empty() && img.rgba.size() != 4 * texels)
        {
            return std::format("images[{}]: {} bytes for {}x{} RGBA texels", i, img.rgba.size(), img.width,
                               img.height);
        }
    }
    return {};
}

// The triangles of every placement in world space, each with its material index
struct placed_triangle
{
    triangle p{};
    std::uint32_t material = 0;
};

std::vector<placed_triangle> world_triangles(const meshes::model& model)
{
    std::vector<placed_triangle> out;
    for (const meshes::placement& placement : model.placements)
    {
        const meshes::mesh& m = model.meshes[static_cast<std::size_t>(placement.mesh)];
        for (std::size_t t = 0; t < triangle_count(m); ++t)
        {
            const triangle c = corners(m, t);
            const matrix& to_world = placement.object_to_world;
            out.push_back({{point(to_world, c[0]), point(to_world, c[1]), point(to_world, c[2])}, m.material_of[t]});
        }
    }
    return out;
}

std::vector<triangle> world_geometry(const meshes::model& model)
{
    std::vector<triangle> out;
    for (const placed_triangle& t : world_triangles(model))
    {
        out.push_back(t.p);
    }
    return out;
}

std::vector<triangle> mesh_geometry(const meshes::mesh& m)
{
    std::vector<triangle> out;
    for (std::size_t t = 0; t < triangle_count(m); ++t)
    {
        out.push_back(corners(m, t));
    }
    return out;
}

// Triangles from positions (3 floats a vertex) and indices (none: one triangle per three vertices), moved by m
std::vector<triangle> make_triangles(std::span<const float> positions, std::span<const std::uint32_t> indices = {},
                                     const matrix& m = identity)
{
    std::vector<triangle> out;
    const std::size_t count = indices.empty() ? positions.size() / 3 : indices.size();
    for (std::size_t c = 0; c + 2 < count; c += 3)
    {
        triangle t{};
        for (std::size_t k = 0; k < 3; ++k)
        {
            t[k] = point(m, vertex(positions, indices.empty() ? c + k : indices[c + k]));
        }
        out.push_back(t);
    }
    return out;
}

// The largest distance between a corner and its counterpart, the triangles paired one to one in any order, each up to a
// rotation of its corners but never a reflection (the winding counts). Infinity when the counts differ or a triangle
// finds no partner within 1e-3.
double deviation(std::span<const triangle> found, std::span<const triangle> expected)
{
    if (found.size() != expected.size())
    {
        return no_match;
    }
    std::vector<bool> used(found.size(), false);
    double worst = 0;
    for (const triangle& e : expected)
    {
        double best = no_match;
        std::size_t pick = found.size();
        for (std::size_t i = 0; i < found.size(); ++i)
        {
            for (std::size_t turn = 0; turn < 3 && !used[i]; ++turn)
            {
                double d = 0;
                for (std::size_t k = 0; k < 3; ++k)
                {
                    d = worse(d, gap(found[i][(k + turn) % 3], e[k]));
                }
                if (d < best)
                {
                    best = d;
                    pick = i;
                }
            }
        }
        if (pick == found.size() || best > 1e-3)
        {
            return no_match;
        }
        used[pick] = true;
        worst = worse(worst, best);
    }
    return worst;
}

// The largest difference between a vertex attribute the loader delivered (n floats a vertex) and the file's, over the
// vertices the triangles use, matched to the file's vertices by position (the files' positions are distinct)
double attribute_deviation(const meshes::mesh& m, std::span<const float> values, std::size_t n,
                           std::span<const float> file_positions, std::span<const float> file_values)
{
    if (values.size() != n * (m.positions.size() / 3))
    {
        return no_match;
    }
    double worst = 0;
    for (const std::uint32_t i : m.indices)
    {
        const vec p = vertex(m.positions, i);
        std::size_t j = 0;
        while (j < file_positions.size() / 3 && !(gap(vertex(file_positions, j), p) <= 1e-6))
        {
            ++j;
        }
        if (j == file_positions.size() / 3)
        {
            return no_match;
        }
        for (std::size_t c = 0; c < n; ++c)
        {
            worst = worse(worst, std::abs(static_cast<double>(values[n * i + c]) - file_values[n * j + c]));
        }
    }
    return worst;
}

// Closed and consistently wound: once the vertices that share a position are welded (a seam's duplicates), every
// directed edge appears once and its reverse once. The count of directed edges that break the rule, plus the triangles
// that weld into a degenerate one.
std::size_t open_edges(const meshes::mesh& m, double weld = 1e-5)
{
    const std::size_t n = m.positions.size() / 3;
    std::vector<std::uint32_t> parent(n);
    std::iota(parent.begin(), parent.end(), std::uint32_t{0});
    const auto find = [&parent](std::uint32_t i) {
        while (parent[i] != i)
        {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    // A sweep along x: only vertices within `weld` of each other in x can share a position
    std::vector<std::uint32_t> order(parent);
    std::ranges::sort(order, {}, [&m](std::uint32_t i) { return m.positions[3 * i]; });
    for (std::size_t a = 0; a < n; ++a)
    {
        for (std::size_t b = a + 1; b < n && m.positions[3 * order[b]] - m.positions[3 * order[a]] <= weld; ++b)
        {
            if (gap(vertex(m.positions, order[a]), vertex(m.positions, order[b])) <= weld)
            {
                parent[find(order[b])] = find(order[a]);
            }
        }
    }
    std::map<std::pair<std::uint32_t, std::uint32_t>, int> edges;
    std::size_t bad = 0;
    for (std::size_t t = 0; t < triangle_count(m); ++t)
    {
        const std::uint32_t a = find(m.indices[3 * t]), b = find(m.indices[3 * t + 1]), c = find(m.indices[3 * t + 2]);
        if (a == b || b == c || c == a)
        {
            ++bad;
            continue;
        }
        ++edges[{a, b}];
        ++edges[{b, c}];
        ++edges[{c, a}];
    }
    for (const auto& [edge, count] : edges)
    {
        const auto reverse = edges.find({edge.second, edge.first});
        bad += count != 1 || reverse == edges.end() || reverse->second != 1 ? 1 : 0;
    }
    return bad;
}

// The triangles whose face normal (counter-clockwise: a × b) doesn't point away from centre(centroid)
template <class Centre>
std::size_t inward_faces(const meshes::mesh& m, const Centre& centre)
{
    std::size_t inward = 0;
    for (std::size_t t = 0; t < triangle_count(m); ++t)
    {
        const triangle c = corners(m, t);
        const vec at = centroid(c);
        inward += dot(face_normal(c), sub(at, centre(at))) > 0 ? 0 : 1;
    }
    return inward;
}

std::string_view wrap_name(meshes::wrap w)
{
    switch (w)
    {
    case meshes::wrap::repeat:
        return "repeat";
    case meshes::wrap::mirrored_repeat:
        return "mirrored_repeat";
    case meshes::wrap::clamp:
        return "clamp";
    }
    return "an unknown wrap";
}

std::string bytes_text(std::span<const std::uint8_t> bytes)
{
    std::string out;
    for (const std::uint8_t b : bytes)
    {
        out += std::format("{}{}", out.empty() ? "" : " ", b);
    }
    return out;
}

std::string describe(const meshes::image& i)
{
    return std::format("{}x{} [{}], {}, wrap {} {}, {}", i.width, i.height, bytes_text(i.rgba),
                       i.srgb ? "sRGB" : "linear", wrap_name(i.wrap_s), wrap_name(i.wrap_t),
                       i.nearest ? "nearest" : "bilinear");
}

std::string describe(const meshes::pbr_material& m)
{
    return std::format("base ({}, {}, {}), metallic {}, roughness {}, images {} {}", m.base_color[0], m.base_color[1],
                       m.base_color[2], m.metallic, m.roughness, m.base_color_image, m.metal_rough_image);
}

std::string describe(const std::vector<std::string>& warnings)
{
    if (warnings.empty())
    {
        return "no warning";
    }
    std::string out = std::format("{} warning{}:", warnings.size(), warnings.size() == 1 ? "" : "s");
    for (const std::string& w : warnings)
    {
        out += std::format(" '{}'", w);
    }
    return out;
}

// glTF 2.0 §3.9.6: a primitive without a material wears base color 1, metallic 1, roughness 1, no textures
bool is_default(const meshes::pbr_material& m)
{
    return m.base_color == std::array<double, 3>{1, 1, 1} && m.metallic == 1 && m.roughness == 1 &&
           m.base_color_image == -1 && m.metal_rough_image == -1;
}

bool same_image(const meshes::image& a, const meshes::image& b)
{
    return a.width == b.width && a.height == b.height && a.rgba == b.rgba && a.srgb == b.srgb &&
           a.wrap_s == b.wrap_s && a.wrap_t == b.wrap_t && a.nearest == b.nearest;
}

// --- Bytes: CRC-32, Adler-32, base64, PNG ---------------------------------------------------------------------------

// CRC-32 of ISO 3309 (PNG's chunks, PNG spec §5.5): the reflected polynomial 0xEDB88320, all ones in and out
std::uint32_t crc32(std::span<const std::uint8_t> bytes)
{
    std::uint32_t c = 0xFFFFFFFFu;
    for (const std::uint8_t b : bytes)
    {
        c ^= b;
        for (int k = 0; k < 8; ++k)
        {
            c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        }
    }
    return c ^ 0xFFFFFFFFu;
}

// Adler-32 (RFC 1950 §8.2): zlib's checksum of the uncompressed bytes
std::uint32_t adler32(std::span<const std::uint8_t> bytes)
{
    std::uint32_t a = 1, b = 0;
    for (const std::uint8_t x : bytes)
    {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    return b << 16 | a;
}

// base64 (RFC 4648 §4): the alphabet A–Z a–z 0–9 + /, padded with '='
std::string base64(std::span<const std::uint8_t> bytes)
{
    static constexpr std::string_view digits = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (std::size_t i = 0; i < bytes.size(); i += 3)
    {
        const std::size_t n = std::min<std::size_t>(3, bytes.size() - i);
        std::uint32_t triple = 0;
        for (std::size_t k = 0; k < 3; ++k)
        {
            triple = triple << 8 | (k < n ? bytes[i + k] : 0u);
        }
        for (std::size_t k = 0; k < 4; ++k)
        {
            out += k <= n ? digits[(triple >> (18 - 6 * k)) & 63] : '=';
        }
    }
    return out;
}

std::vector<std::uint8_t> text_bytes(std::string_view text)
{
    std::vector<std::uint8_t> out;
    for (const char c : text)
    {
        out.push_back(static_cast<std::uint8_t>(c));
    }
    return out;
}

void put_be32(std::vector<std::uint8_t>& out, std::uint32_t v) // big-endian, as PNG and zlib store numbers
{
    for (int shift = 24; shift >= 0; shift -= 8)
    {
        out.push_back(static_cast<std::uint8_t>(v >> shift));
    }
}

// A PNG of 8-bit RGB (3 channels) or RGBA (4) texels, rows from the top, written so that its bytes are known exactly
// without a compressor: the zlib stream (RFC 1950) is CMF 0x78 (deflate, 32 KiB window), FLG 0x01 (0x7801 = 31·991,
// the header's check), one final stored deflate block (RFC 1951 §3.2.4: BFINAL 1 and BTYPE 00 in one byte, LEN, its
// complement NLEN, the bytes) and the Adler-32 of the bytes; each row starts with filter type 0 (none)
std::vector<std::uint8_t> make_png(int width, int height, int channels, std::span<const std::uint8_t> texels)
{
    const std::size_t row = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
    std::vector<std::uint8_t> raw;
    for (std::size_t y = 0; y < static_cast<std::size_t>(height); ++y)
    {
        raw.push_back(0);
        const std::span<const std::uint8_t> line = texels.subspan(y * row, row);
        raw.insert(raw.end(), line.begin(), line.end());
    }
    std::vector<std::uint8_t> zlib{0x78, 0x01, 0x01};
    const std::uint32_t size = static_cast<std::uint32_t>(raw.size());
    const std::uint32_t complement = ~size & 0xFFFFu;
    zlib.push_back(static_cast<std::uint8_t>(size & 0xFFu));
    zlib.push_back(static_cast<std::uint8_t>(size >> 8 & 0xFFu));
    zlib.push_back(static_cast<std::uint8_t>(complement & 0xFFu));
    zlib.push_back(static_cast<std::uint8_t>(complement >> 8 & 0xFFu));
    zlib.insert(zlib.end(), raw.begin(), raw.end());
    put_be32(zlib, adler32(raw));

    std::vector<std::uint8_t> png{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    const auto chunk = [&png](std::string_view type, std::span<const std::uint8_t> data) {
        put_be32(png, static_cast<std::uint32_t>(data.size()));
        std::vector<std::uint8_t> covered = text_bytes(type); // the CRC covers the type and the data
        covered.insert(covered.end(), data.begin(), data.end());
        png.insert(png.end(), covered.begin(), covered.end());
        put_be32(png, crc32(covered));
    };
    std::vector<std::uint8_t> header;
    put_be32(header, static_cast<std::uint32_t>(width));
    put_be32(header, static_cast<std::uint32_t>(height));
    // 8 bits a channel; color type 6 (RGBA) or 2 (RGB); deflate; filter method 0; no interlace
    const auto color_type = static_cast<std::uint8_t>(channels == 4 ? 6 : 2);
    header.insert(header.end(), {std::uint8_t{8}, color_type, std::uint8_t{0}, std::uint8_t{0}, std::uint8_t{0}});
    chunk("IHDR", header);
    chunk("IDAT", zlib);
    chunk("IEND", {});
    return png;
}

// The test's 2×2 images, rows from the top. A: RGBA, a base color. B: RGB, metallic-roughness data, which the loader
// expands to RGBA with alpha 255 (stbi_load_from_memory(…, 4)). C: RGBA, a second base color.
constexpr std::array<std::uint8_t, 16> texels_a{255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 10, 20, 30, 255};
constexpr std::array<std::uint8_t, 12> texels_b{0, 64, 128, 0, 32, 255, 1, 2, 3, 250, 251, 252};
constexpr std::array<std::uint8_t, 16> texels_b_rgba{0, 64, 128, 255, 0, 32, 255, 255,
                                                     1, 2, 3, 255, 250, 251, 252, 255};
constexpr std::array<std::uint8_t, 16> texels_c{1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255, 4, 4, 4, 255};

const std::vector<std::uint8_t>& png_a()
{
    static const std::vector<std::uint8_t> png = make_png(2, 2, 4, texels_a);
    return png;
}
const std::vector<std::uint8_t>& png_b()
{
    static const std::vector<std::uint8_t> png = make_png(2, 2, 3, texels_b);
    return png;
}
const std::vector<std::uint8_t>& png_c()
{
    static const std::vector<std::uint8_t> png = make_png(2, 2, 4, texels_c);
    return png;
}

void write_bytes(const std::filesystem::path& file, std::span<const std::byte> bytes)
{
    std::ofstream out(file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out)
    {
        throw setup_error{std::format("can't write the test file {}", file.filename().string())};
    }
}

// --- glTF files written by the test ---------------------------------------------------------------------------------

// A glTF 2.0 file assembled from JSON pieces: the arrays (accessors, meshes, nodes, …) as their items' text, one
// buffer of bytes (a base64 data: URI in a .gltf, the BIN chunk of a .glb, or a file named by buffer_uri)
class gltf_writer
{
public:
    std::string buffer_uri; // the buffer's URI when its bytes live in a file of their own (written by the case)

    // Bytes appended to the buffer at a multiple of 4 (glTF's alignment of every accessor); returns the bufferView's
    // index. target: 34962 vertex data, 34963 indices, 0 none (images, the parts of a sparse accessor)
    int view(std::span<const std::byte> bytes, int stride = 0, int target = 0)
    {
        while (bin_.size() % 4 != 0)
        {
            bin_.push_back(0);
        }
        std::string json =
            std::format(R"({{"buffer": 0, "byteOffset": {}, "byteLength": {})", bin_.size(), bytes.size());
        for (const std::byte b : bytes)
        {
            bin_.push_back(static_cast<std::uint8_t>(b));
        }
        if (stride > 0)
        {
            json += std::format(R"(, "byteStride": {})", stride);
        }
        if (target > 0)
        {
            json += std::format(R"(, "target": {})", target);
        }
        return add("bufferViews", json + "}");
    }

    // A tightly packed float accessor of `type` (SCALAR, VEC2, VEC3, VEC4); bounds: its min and max (POSITION needs
    // them; non-finite values are left out of them, JSON has no NaN)
    int floats(std::span<const float> values, std::string_view type, bool bounds = false)
    {
        const std::size_t n = type == "SCALAR" ? 1 : type == "VEC2" ? 2 : type == "VEC3" ? 3 : 4;
        const int v = view(std::as_bytes(values), 0, 34962);
        std::string json = std::format(R"({{"bufferView": {}, "componentType": 5126, "count": {}, "type": "{}")", v,
                                       values.size() / n, type);
        if (bounds)
        {
            json += R"(, "min": )" + bound(values, n, false) + R"(, "max": )" + bound(values, n, true);
        }
        return add("accessors", json + "}");
    }

    // An index accessor of component type 5121 (UNSIGNED_BYTE), 5123 (UNSIGNED_SHORT) or 5125 (UNSIGNED_INT)
    int indices(std::span<const std::uint32_t> values, int component_type)
    {
        const int size = component_type == 5121 ? 1 : component_type == 5123 ? 2 : 4;
        std::vector<std::byte> bytes;
        for (const std::uint32_t i : values)
        {
            for (int k = 0; k < size; ++k)
            {
                bytes.push_back(static_cast<std::byte>(i >> (8 * k) & 0xFFu));
            }
        }
        const int v = view(bytes, 0, 34963);
        return add("accessors",
                   std::format(R"({{"bufferView": {}, "componentType": {}, "count": {}, "type": "SCALAR"}})", v,
                               component_type, values.size()));
    }

    // An item of a top-level array ("meshes", "nodes", "materials", …); returns its index there
    int add(std::string_view array, std::string json)
    {
        auto it = std::ranges::find(arrays_, array, &named_array::name);
        if (it == arrays_.end())
        {
            arrays_.push_back({std::string(array), {}});
            it = arrays_.end() - 1;
        }
        it->items.push_back(std::move(json));
        return static_cast<int>(it->items.size()) - 1;
    }

    void extension(std::string_view name, bool required)
    {
        if (!std::ranges::contains(used_, name))
        {
            used_.emplace_back(name);
        }
        if (required && !std::ranges::contains(required_, name))
        {
            required_.emplace_back(name);
        }
    }

    // A top-level member as JSON text ("cameras": […], "extensions": {…})
    void top(std::string member) { tops_.push_back(std::move(member)); }

    const std::vector<std::uint8_t>& bin() const { return bin_; }

    // The JSON; `glb`: its buffer without a URI (the BIN chunk)
    std::string text(bool glb) const
    {
        std::string out = R"({"asset": {"version": "2.0", "generator": "MeshTests"})";
        if (std::ranges::contains(arrays_, std::string("scenes"), &named_array::name))
        {
            out += R"(, "scene": 0)";
        }
        for (const named_array& a : arrays_)
        {
            out += std::format(R"(, "{}": [)", a.name);
            for (std::size_t i = 0; i < a.items.size(); ++i)
            {
                out += (i > 0 ? ", " : "") + a.items[i];
            }
            out += "]";
        }
        if (!bin_.empty())
        {
            out += std::format(R"(, "buffers": [{{"byteLength": {})", bin_.size());
            if (!glb)
            {
                out += R"(, "uri": ")" +
                       (buffer_uri.empty() ? "data:application/octet-stream;base64," + base64(bin_) : buffer_uri) +
                       "\"";
            }
            out += "}]";
        }
        if (!used_.empty())
        {
            out += R"(, "extensionsUsed": )" + names(used_);
        }
        if (!required_.empty())
        {
            out += R"(, "extensionsRequired": )" + names(required_);
        }
        for (const std::string& t : tops_)
        {
            out += ", " + t;
        }
        return out + "}";
    }

    // The binary container (glTF 2.0 §4.4): a 12-byte header (magic "glTF", version 2, length), a JSON chunk padded
    // with spaces and a BIN chunk padded with zeros, both to a multiple of 4
    std::vector<std::uint8_t> glb() const
    {
        std::string json = text(true);
        while (json.size() % 4 != 0)
        {
            json += ' ';
        }
        std::vector<std::uint8_t> bin = bin_;
        while (bin.size() % 4 != 0)
        {
            bin.push_back(0);
        }
        std::vector<std::uint8_t> out;
        const auto le32 = [&out](std::size_t v) {
            for (int k = 0; k < 4; ++k)
            {
                out.push_back(static_cast<std::uint8_t>(v >> (8 * k) & 0xFFu));
            }
        };
        le32(0x46546C67); // "glTF"
        le32(2);
        le32(12 + 8 + json.size() + 8 + bin.size());
        le32(json.size());
        le32(0x4E4F534A); // "JSON"
        for (const char c : json)
        {
            out.push_back(static_cast<std::uint8_t>(c));
        }
        le32(bin.size());
        le32(0x004E4942); // "BIN\0"
        out.insert(out.end(), bin.begin(), bin.end());
        return out;
    }

    // Written as a .glb or a .gltf, by the file's extension
    std::filesystem::path write(const std::filesystem::path& file) const
    {
        if (file.extension() == ".glb")
        {
            const std::vector<std::uint8_t> bytes = glb();
            write_bytes(file, std::as_bytes(std::span(bytes)));
        }
        else
        {
            const std::string json = text(false);
            write_bytes(file, std::as_bytes(std::span(json)));
        }
        return file;
    }

private:
    struct named_array
    {
        std::string name;
        std::vector<std::string> items;
    };

    static std::string bound(std::span<const float> values, std::size_t n, bool upper)
    {
        std::string out = "[";
        for (std::size_t c = 0; c < n; ++c)
        {
            double b = 0;
            bool any = false;
            for (std::size_t i = c; i < values.size(); i += n)
            {
                if (std::isfinite(values[i]))
                {
                    const double v = values[i];
                    b = !any ? v : upper ? std::max(b, v) : std::min(b, v);
                    any = true;
                }
            }
            out += std::format("{}{}", c > 0 ? ", " : "", b);
        }
        return out + "]";
    }

    static std::string names(const std::vector<std::string>& list)
    {
        std::string out = "[";
        for (std::size_t i = 0; i < list.size(); ++i)
        {
            out += std::format(R"({}"{}")", i > 0 ? ", " : "", list[i]);
        }
        return out + "]";
    }

    std::vector<std::uint8_t> bin_;
    std::vector<named_array> arrays_;
    std::vector<std::string> used_;
    std::vector<std::string> required_;
    std::vector<std::string> tops_;
};

// The triangle most files hold (counter-clockwise about +z) and its UVs
constexpr std::array<float, 9> triangle_positions{0, 0, 0, 1, 0, 0, 0, 1, 0};
constexpr std::array<float, 6> triangle_uvs{0, 0, 1, 0, 0, 1};

// A tetrahedron around the origin, every face counter-clockwise seen from outside. Four corners that aren't coplanar
// fix an affine map, so matching its world corners checks a whole transform.
constexpr std::array<float, 12> tetra_positions{1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1};
constexpr std::array<std::uint32_t, 12> tetra_indices{0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2};

// glTF's 90° turn about y, (0, sin 45°, 0, cos 45°), as a JSON exporter writes it (cgltf keeps it as floats)
constexpr std::string_view quarter_turn_y = "[0, 0.7071067811865476, 0, 0.7071067811865476]";

struct primitive_data
{
    std::span<const float> positions;
    std::span<const std::uint32_t> indices; // empty: not indexed
    int index_type = 5125;                  // 5121, 5123 or 5125
    std::span<const float> normals;         // empty: no NORMAL
    std::span<const float> uvs;             // empty: no TEXCOORD_0
    int material = -1;                      // -1: none (glTF's default material)
    int mode = 4;                           // 4 TRIANGLES, 5 TRIANGLE_STRIP, 6 TRIANGLE_FAN
};

// A mesh of the given primitives (POSITION with min and max, as glTF requires); returns its index
int add_mesh(gltf_writer& w, std::initializer_list<primitive_data> primitives)
{
    std::string list;
    for (const primitive_data& d : primitives)
    {
        std::string json = std::format(R"({{"attributes": {{"POSITION": {})", w.floats(d.positions, "VEC3", true));
        if (!d.normals.empty())
        {
            json += std::format(R"(, "NORMAL": {})", w.floats(d.normals, "VEC3"));
        }
        if (!d.uvs.empty())
        {
            json += std::format(R"(, "TEXCOORD_0": {})", w.floats(d.uvs, "VEC2"));
        }
        json += "}";
        if (!d.indices.empty())
        {
            json += std::format(R"(, "indices": {})", w.indices(d.indices, d.index_type));
        }
        if (d.material >= 0)
        {
            json += std::format(R"(, "material": {})", d.material);
        }
        if (d.mode != 4)
        {
            json += std::format(R"(, "mode": {})", d.mode);
        }
        list += (list.empty() ? "" : ", ") + json + "}";
    }
    return w.add("meshes", R"({"primitives": [)" + list + "]}");
}

// Node 0 wears mesh 0, with `members` (a transform); the default scene holds it
std::filesystem::path one_node(gltf_writer& w, const std::filesystem::path& file, std::string_view members = {})
{
    w.add("nodes", R"({"mesh": 0)" + std::string(members) + "}");
    w.add("scenes", R"({"nodes": [0]})");
    return w.write(file);
}

std::filesystem::path triangle_file(const std::filesystem::path& file)
{
    gltf_writer w;
    add_mesh(w, {{.positions = triangle_positions}});
    return one_node(w, file);
}

// One triangle placed by two nodes, at x = -2 and x = +2
std::filesystem::path shared_file(const std::filesystem::path& file)
{
    gltf_writer w;
    add_mesh(w, {{.positions = triangle_positions}});
    w.add("nodes", R"({"mesh": 0, "translation": [-2, 0, 0]})");
    w.add("nodes", R"({"mesh": 0, "translation": [2, 0, 0]})");
    w.add("scenes", R"({"nodes": [0, 1]})");
    return w.write(file);
}

std::string data_image(std::span<const std::uint8_t> png)
{
    return R"({"uri": "data:image/png;base64,)" + base64(png) + R"("})";
}

// The clean file every material, image and warning case starts from: one triangle with TEXCOORD_0, image 0 (PNG A as a
// data: URI unless `base_image` gives the image object), texture 0 of it, one material whose base color is texture 0,
// one node. It loads with no warning. A case's `feature` runs once the base's accessors, image and texture are there:
// it may add accessors, images, textures, extensions and top-level members, and it fills in these parts.
struct textured_parts
{
    int position = -1; // the base's accessors, for the feature's use
    int texcoord = -1;
    bool base_texture = true;         // the material's baseColorTexture (texture 0)
    bool texcoord_0 = true;           // the primitive's TEXCOORD_0
    std::string base_texture_members; // more members of baseColorTexture ("texCoord", "extensions")
    std::string pbr_members;          // more members of pbrMetallicRoughness ("metallicFactor")
    std::string material_members;     // more members of the material
    std::string attributes;           // more attributes of the primitive
    std::string primitive_members;    // more members of the primitive ("targets")
    std::string more_primitives;      // more primitives of the mesh, each with its leading comma
    std::string mesh_members;         // more members of the mesh ("weights")
    std::string node_members;         // more members of node 0 ("skin")
    std::vector<std::string> nodes;   // more root nodes: 1, 2, …
};
using feature_function = std::function<void(gltf_writer&, textured_parts&)>;
using image_function = std::function<std::string(gltf_writer&)>;

gltf_writer textured(const feature_function& feature = {}, const image_function& base_image = {})
{
    gltf_writer w;
    textured_parts p;
    p.position = w.floats(triangle_positions, "VEC3", true);
    p.texcoord = w.floats(triangle_uvs, "VEC2");
    w.add("images", base_image ? base_image(w) : data_image(png_a()));
    w.add("textures", R"({"source": 0})");
    if (feature)
    {
        feature(w, p);
    }
    const std::string pbr_alone = p.pbr_members.empty() ? std::string() : p.pbr_members.substr(2); // without ", "
    const std::string pbr = p.base_texture ? std::format(R"({{"baseColorTexture": {{"index": 0{}}}{}}})",
                                                         p.base_texture_members, p.pbr_members)
                                           : "{" + pbr_alone + "}";
    w.add("materials", R"({"pbrMetallicRoughness": )" + pbr + p.material_members + "}");
    std::string attributes = std::format(R"("POSITION": {})", p.position);
    if (p.texcoord_0)
    {
        attributes += std::format(R"(, "TEXCOORD_0": {})", p.texcoord);
    }
    attributes += p.attributes;
    w.add("meshes", R"({"primitives": [{"attributes": {)" + attributes + R"(}, "material": 0)" + p.primitive_members +
                        "}" + p.more_primitives + "]" + p.mesh_members + "}");
    w.add("nodes", R"({"mesh": 0)" + p.node_members + "}");
    std::string roots = "0";
    for (const std::string& node : p.nodes)
    {
        roots += std::format(", {}", w.add("nodes", node));
    }
    w.add("scenes", R"({"nodes": [)" + roots + "]}");
    return w;
}

// --- Loading, and reading what the loader delivered -----------------------------------------------------------------

using load_result = std::expected<meshes::model, std::string>;

// load_gltf; an exception escaping it is a failed check of its own (the loader returns its errors)
load_result load(report& r, std::string_view name, const std::filesystem::path& file, const meshes::limits& limits = {})
{
    try
    {
        return load_gltf(file, limits);
    }
    catch (const std::exception& error)
    {
        r.check(std::format("{}-no-exception", name), false, std::format("'{}' thrown", error.what()),
                "errors returned, not thrown");
        return std::unexpected(std::format("(thrown) {}", error.what()));
    }
}

// One line: the file loaded into a well-formed model. Returns the model, or nullptr (the case's other checks are then
// skipped)
const meshes::model* loaded(report& r, std::string_view name, const load_result& result)
{
    const std::string problem = result ? model_problem(*result) : std::format("error '{}'", result.error());
    r.check(std::format("{}-loads", name), problem.empty(), problem.empty() ? "a well-formed model" : problem,
            "a well-formed model");
    return problem.empty() ? &*result : nullptr;
}

// One line: the file refused, with an error (not a model) saying each of `words`; whether the message names the file
// is counted for errors.name-their-file
void refused(report& r, std::string_view name, const load_result& result, const std::filesystem::path& file,
             std::initializer_list<std::string_view> words = {})
{
    std::string expected = "an error";
    for (const std::string_view w : words)
    {
        expected += std::format(" saying '{}'", w);
    }
    if (result)
    {
        r.check(name, false, "a model", expected);
        return;
    }
    const std::string& message = result.error();
    const bool says = std::ranges::all_of(words, [&message](std::string_view w) { return message.contains(w); });
    r.check(name, says, std::format("error '{}'", message), expected);
    r.error_message(name, message.contains(file.filename().string()));
}

// One mesh placed once (a file of one node): one line, and whether the mesh's own checks can go on
bool one_mesh(report& r, std::string_view name, const meshes::model& m)
{
    const bool ok = m.meshes.size() == 1 && m.placements.size() == 1;
    r.check(std::format("{}-meshes", name), ok,
            std::format("{} meshes, {} placements", m.meshes.size(), m.placements.size()), "1 mesh, 1 placement");
    return ok;
}

// The model's world triangles against the expected ones, in one line
void compare(report& r, std::string_view name, const meshes::model& m, std::span<const triangle> expected,
             double tolerance)
{
    const std::vector<triangle> found = world_geometry(m);
    const double d = deviation(found, expected);
    r.check(name, d <= tolerance,
            std::format("{} triangles, corners at most {:.3g} from the expected", found.size(), d),
            std::format("{} triangles, corners within {:.3g}", expected.size(), tolerance));
}

// The model's world bounds against the box of the expected triangles
void compare_bounds(report& r, std::string_view name, const meshes::model& m, std::span<const triangle> expected,
                    double tolerance)
{
    vec lower{no_match, no_match, no_match};
    vec upper{-no_match, -no_match, -no_match};
    for (const triangle& t : expected)
    {
        for (const vec& p : t)
        {
            for (std::size_t k = 0; k < 3; ++k)
            {
                lower[k] = std::min(lower[k], p[k]);
                upper[k] = std::max(upper[k], p[k]);
            }
        }
    }
    double d = 0;
    for (std::size_t k = 0; k < 3; ++k)
    {
        d = worse(d, worse(std::abs(m.lower[k] - lower[k]), std::abs(m.upper[k] - upper[k])));
    }
    r.check(name, d <= tolerance,
            std::format("lower ({}, {}, {}), upper ({}, {}, {})", m.lower[0], m.lower[1], m.lower[2], m.upper[0],
                        m.upper[1], m.upper[2]),
            std::format("lower ({}, {}, {}), upper ({}, {}, {}) within {:.3g}", lower[0], lower[1], lower[2],
                        upper[0], upper[1], upper[2], tolerance));
}

// The material index of the world triangle at `where`, if there is one
std::optional<std::uint32_t> material_at(const meshes::model& m, const triangle& where)
{
    const std::array<triangle, 1> wanted{where};
    for (const placed_triangle& t : world_triangles(m))
    {
        const std::array<triangle, 1> one{t.p};
        if (deviation(one, wanted) <= 1e-6)
        {
            return t.material;
        }
    }
    return std::nullopt;
}

// The material of the first placed triangle
const meshes::pbr_material* first_material(const meshes::model& m)
{
    const std::vector<placed_triangle> list = world_triangles(m);
    return list.empty() ? nullptr : &m.materials[list[0].material];
}
const meshes::image* image_at(const meshes::model& m, int index)
{
    return index >= 0 && static_cast<std::size_t>(index) < m.images.size()
               ? &m.images[static_cast<std::size_t>(index)]
               : nullptr;
}
// The first material whose metallic factor is `metallic` (the cases tell their materials apart by it)
const meshes::pbr_material* material_by_metallic(const meshes::model& m, double metallic)
{
    const auto it = std::ranges::find(m.materials, metallic, &meshes::pbr_material::metallic);
    return it == m.materials.end() ? nullptr : &*it;
}

meshes::image expected_image(std::span<const std::uint8_t> rgba, bool srgb, meshes::wrap s, meshes::wrap t,
                             bool nearest)
{
    return {.width = 2,
            .height = 2,
            .rgba = std::vector<std::uint8_t>(rgba.begin(), rgba.end()),
            .srgb = srgb,
            .wrap_s = s,
            .wrap_t = t,
            .nearest = nearest};
}

void check_image(report& r, std::string_view name, const meshes::image* found, const meshes::image& expected)
{
    r.check(name, found && same_image(*found, expected), found ? describe(*found) : std::string("no such image"),
            describe(expected));
}

// A group of checks. An exception from the modules under test is a failed check of the group and the run goes on; the
// test's own setup_error ends it (exit code 2).
template <class Checks>
void group(report& r, std::string_view name, const Checks& checks)
{
    try
    {
        checks();
    }
    catch (const setup_error&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        r.check(std::format("{}.no-exception", name), false, std::format("'{}' thrown", error.what()),
                "no exception from the modules");
    }
}

// --- The test's own arithmetic --------------------------------------------------------------------------------------

void self_checks(report& r)
{
    const std::uint32_t iend = crc32(text_bytes("IEND"));
    r.check("self.crc32", iend == 0xAE426082u, std::format("CRC-32('IEND') {:08X}", iend),
            "AE426082 (every PNG's last four bytes)");
    const std::uint32_t wiki = adler32(text_bytes("Wikipedia"));
    r.check("self.adler32", wiki == 0x11E60398u, std::format("Adler-32('Wikipedia') {:08X}", wiki), "11E60398");
    // RFC 4648 §10's test vectors
    const std::array<std::pair<std::string_view, std::string_view>, 7> vectors{
        {{"", ""}, {"f", "Zg=="}, {"fo", "Zm8="}, {"foo", "Zm9v"}, {"foob", "Zm9vYg=="}, {"fooba", "Zm9vYmE="},
         {"foobar", "Zm9vYmFy"}}};
    std::string found;
    bool all = true;
    for (const auto& [text, code] : vectors)
    {
        const std::string encoded = base64(text_bytes(text));
        found += std::format("{}'{}'", found.empty() ? "" : " ", encoded);
        all = all && encoded == code;
    }
    r.check("self.base64", all, found,
            "'' 'Zg==' 'Zm8=' 'Zm9v' 'Zm9vYg==' 'Zm9vYmE=' 'Zm9vYmFy' (RFC 4648 section 10)");
    r.info("self.png-a", std::format("the 2x2 RGBA PNG as base64: {}", base64(png_a())));
}

// --- glTF: geometry and accessors -----------------------------------------------------------------------------------

void gltf_geometry(report& r, const std::filesystem::path& folder)
{
    const std::vector<triangle> one_triangle = make_triangles(triangle_positions);

    // Without NORMAL the mesh is flat (no normals: the GPU takes the face's); without a material, glTF's default
    {
        const load_result result = load(r, "gltf.flat", triangle_file(folder / "flat.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.flat", result); m && one_mesh(r, "gltf.flat", *m))
        {
            compare(r, "gltf.flat-triangle", *m, one_triangle, 0);
            r.equal("gltf.flat-normals", m->meshes[0].normals.size(), 0);
            r.check("gltf.flat-warnings", m->warnings.empty(), describe(m->warnings), "no warning");
            const std::vector<std::uint32_t>& material_of = m->meshes[0].material_of;
            const std::uint32_t k = material_of.empty() ? 0 : material_of[0];
            const bool ok = !material_of.empty() && k + 1 == m->materials.size() && is_default(m->materials[k]);
            r.check("gltf.flat-default-material", ok,
                    std::format("materials[{}] of {}: {}", k, m->materials.size(),
                                m->materials.empty() ? std::string("none") : describe(m->materials[k])),
                    "the last material, glTF's default: base (1, 1, 1), metallic 1, roughness 1, images -1 -1");
        }
    }

    // With NORMAL: the file's normals, vertex by vertex
    {
        constexpr std::array<float, 9> normals{0, 0, 1, 0.6f, 0, 0.8f, 0, 0.6f, 0.8f};
        gltf_writer w;
        add_mesh(w, {{.positions = triangle_positions, .normals = normals}});
        const load_result result = load(r, "gltf.normals", one_node(w, folder / "normals.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.normals", result); m && one_mesh(r, "gltf.normals", *m))
        {
            compare(r, "gltf.normals-triangle", *m, one_triangle, 0);
            r.exact("gltf.normals-values",
                    attribute_deviation(m->meshes[0], m->meshes[0].normals, 3, triangle_positions, normals), 0, 1e-6);
        }
    }

    // A square of two triangles indexed with each of glTF's index types, and not indexed (six vertices)
    {
        constexpr std::array<float, 12> square{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
        constexpr std::array<std::uint32_t, 6> order{0, 1, 2, 0, 2, 3};
        const std::vector<triangle> expected = make_triangles(square, order);
        for (const auto& [type, label] : {std::pair{5121, "u8"}, std::pair{5123, "u16"}, std::pair{5125, "u32"}})
        {
            gltf_writer w;
            add_mesh(w, {{.positions = square, .indices = order, .index_type = type}});
            const std::string name = std::format("gltf.indices-{}", label);
            const load_result result = load(r, name, one_node(w, folder / std::format("indices-{}.gltf", label)));
            if (const meshes::model* m = loaded(r, name, result))
            {
                compare(r, name + "-triangles", *m, expected, 0);
            }
        }
        std::vector<float> soup;
        for (const std::uint32_t i : order)
        {
            soup.insert(soup.end(), {square[3 * i], square[3 * i + 1], square[3 * i + 2]});
        }
        gltf_writer w;
        add_mesh(w, {{.positions = soup}});
        const load_result result = load(r, "gltf.indices-none", one_node(w, folder / "indices-none.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.indices-none", result))
        {
            compare(r, "gltf.indices-none-triangles", *m, expected, 0);
        }
    }

    // KHR_mesh_quantization, required (one of the two required extensions the loader allows): POSITION as signed
    // shorts in a view of stride 8 (glTF's 4-byte alignment of vertex attributes), NORMAL as normalized signed bytes
    // (stride 4; glTF §3.11: f = max(c/127, -1), so -128 → -1), TEXCOORD_0 as normalized unsigned bytes (stride 4)
    {
        constexpr std::array<std::int16_t, 12> positions{-3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0};
        constexpr std::array<std::int8_t, 12> normals{0, 0, 127, 0, 0, 127, 0, 0, -128, 0, 0, 0};
        constexpr std::array<std::uint8_t, 12> uvs{0, 0, 0, 0, 255, 0, 0, 0, 0, 51, 0, 0};
        constexpr std::array<float, 9> expected_positions{-3, 0, 0, 2, 0, 0, 0, 2, 0};
        constexpr std::array<float, 9> expected_normals{0, 0, 1, 0, 1, 0, -1, 0, 0};
        const std::array<float, 6> expected_uvs{0, 0, 1, 0, 0, 51.0f / 255.0f};
        gltf_writer w;
        w.extension("KHR_mesh_quantization", true);
        const int pv = w.view(std::as_bytes(std::span(positions)), 8, 34962);
        const int nv = w.view(std::as_bytes(std::span(normals)), 4, 34962);
        const int tv = w.view(std::as_bytes(std::span(uvs)), 4, 34962);
        const int pa = w.add("accessors", std::format(R"({{"bufferView": {}, "componentType": 5122, "count": 3, )"
                                                      R"("type": "VEC3", "min": [-3, 0, 0], "max": [2, 2, 0]}})",
                                                      pv));
        const int na = w.add("accessors", std::format(R"({{"bufferView": {}, "componentType": 5120, )"
                                                      R"("normalized": true, "count": 3, "type": "VEC3"}})",
                                                      nv));
        const int ta = w.add("accessors", std::format(R"({{"bufferView": {}, "componentType": 5121, )"
                                                      R"("normalized": true, "count": 3, "type": "VEC2"}})",
                                                      tv));
        w.add("meshes", std::format(R"({{"primitives": [{{"attributes": {{"POSITION": {}, "NORMAL": {}, )"
                                    R"("TEXCOORD_0": {}}}}}]}})",
                                    pa, na, ta));
        const load_result result = load(r, "gltf.quantized", one_node(w, folder / "quantized.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.quantized", result); m && one_mesh(r, "gltf.quantized", *m))
        {
            compare(r, "gltf.quantized-positions", *m, make_triangles(expected_positions), 0);
            r.exact("gltf.quantized-normals",
                    attribute_deviation(m->meshes[0], m->meshes[0].normals, 3, expected_positions, expected_normals),
                    0, 1e-6);
            r.exact("gltf.quantized-uvs",
                    attribute_deviation(m->meshes[0], m->meshes[0].uvs, 2, expected_positions, expected_uvs), 0, 1e-6);
        }
    }

    // POSITION and NORMAL interleaved in one buffer view of stride 24 (byte offsets 0 and 12)
    {
        constexpr std::array<float, 18> interleaved{0, 0, 0, 0, 0, 1, 1, 0, 0, 0.6f, 0, 0.8f, 0, 1, 0, 0, 0.6f, 0.8f};
        constexpr std::array<float, 9> normals{0, 0, 1, 0.6f, 0, 0.8f, 0, 0.6f, 0.8f};
        gltf_writer w;
        const int v = w.view(std::as_bytes(std::span(interleaved)), 24, 34962);
        const int pa = w.add("accessors", std::format(R"({{"bufferView": {}, "byteOffset": 0, "componentType": 5126, )"
                                                      R"("count": 3, "type": "VEC3", "min": [0, 0, 0], )"
                                                      R"("max": [1, 1, 0]}})",
                                                      v));
        const int na = w.add("accessors", std::format(R"({{"bufferView": {}, "byteOffset": 12, )"
                                                      R"("componentType": 5126, "count": 3, "type": "VEC3"}})",
                                                      v));
        w.add("meshes",
              std::format(R"({{"primitives": [{{"attributes": {{"POSITION": {}, "NORMAL": {}}}}}]}})", pa, na));
        const load_result result = load(r, "gltf.interleaved", one_node(w, folder / "interleaved.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.interleaved", result); m && one_mesh(r, "gltf.interleaved", *m))
        {
            compare(r, "gltf.interleaved-positions", *m, one_triangle, 0);
            r.exact("gltf.interleaved-normals",
                    attribute_deviation(m->meshes[0], m->meshes[0].normals, 3, triangle_positions, normals), 0, 1e-6);
        }
    }

    // A sparse POSITION accessor (glTF §3.6.2.3): the triangle from a buffer view, vertices 0 and 2 replaced through u8
    // indices; the mesh holds the substituted array, (-1, 0, 0), (1, 0, 0), (0, 5, 0)
    {
        constexpr std::array<std::uint8_t, 2> which{0, 2};
        constexpr std::array<float, 6> values{-1, 0, 0, 0, 5, 0};
        constexpr std::array<float, 9> expected{-1, 0, 0, 1, 0, 0, 0, 5, 0};
        gltf_writer w;
        const int base = w.view(std::as_bytes(std::span(triangle_positions)), 0, 34962);
        const int iv = w.view(std::as_bytes(std::span(which)));
        const int vv = w.view(std::as_bytes(std::span(values)));
        const int pa = w.add("accessors", std::format(R"({{"bufferView": {}, "componentType": 5126, "count": 3, )"
                                                      R"("type": "VEC3", "min": [-1, 0, 0], "max": [1, 5, 0], )"
                                                      R"("sparse": {{"count": 2, "indices": {{"bufferView": {}, )"
                                                      R"("componentType": 5121}}, "values": {{"bufferView": {}}}}}}})",
                                                      base, iv, vv));
        w.add("meshes", std::format(R"({{"primitives": [{{"attributes": {{"POSITION": {}}}}}]}})", pa));
        const load_result result = load(r, "gltf.sparse", one_node(w, folder / "sparse.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.sparse", result))
        {
            compare(r, "gltf.sparse-positions", *m, make_triangles(expected), 0);
        }
    }

    // The same substitutions over an interleaved base (POSITION and NORMAL in one view of stride 24): the sparse
    // values stay tightly packed, 12 bytes apart, whatever the base's stride (glTF §3.6.2.3). A view of 99s follows
    // the values, so a reader that stepped through them at the base's 24 bytes would take the second from there.
    {
        constexpr std::array<float, 18> interleaved{0, 0, 0, 0, 0, 1, 1, 0, 0, 0.6f, 0, 0.8f, 0, 1, 0, 0, 0.6f, 0.8f};
        constexpr std::array<float, 9> normals{0, 0, 1, 0.6f, 0, 0.8f, 0, 0.6f, 0.8f};
        constexpr std::array<std::uint8_t, 2> which{0, 2};
        constexpr std::array<float, 6> values{-1, 0, 0, 0, 5, 0};
        constexpr std::array<float, 6> after{99, 99, 99, 99, 99, 99};
        constexpr std::array<float, 9> expected{-1, 0, 0, 1, 0, 0, 0, 5, 0};
        gltf_writer w;
        const int base = w.view(std::as_bytes(std::span(interleaved)), 24, 34962);
        const int iv = w.view(std::as_bytes(std::span(which)));
        const int vv = w.view(std::as_bytes(std::span(values)));
        w.view(std::as_bytes(std::span(after)));
        const int pa = w.add("accessors", std::format(R"({{"bufferView": {}, "componentType": 5126, "count": 3, )"
                                                      R"("type": "VEC3", "min": [-1, 0, 0], "max": [1, 5, 0], )"
                                                      R"("sparse": {{"count": 2, "indices": {{"bufferView": {}, )"
                                                      R"("componentType": 5121}}, "values": {{"bufferView": {}}}}}}})",
                                                      base, iv, vv));
        const int na = w.add("accessors", std::format(R"({{"bufferView": {}, "byteOffset": 12, )"
                                                      R"("componentType": 5126, "count": 3, "type": "VEC3"}})",
                                                      base));
        w.add("meshes",
              std::format(R"({{"primitives": [{{"attributes": {{"POSITION": {}, "NORMAL": {}}}}}]}})", pa, na));
        const load_result result = load(r, "gltf.sparse-interleaved", one_node(w, folder / "sparse-interleaved.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.sparse-interleaved", result);
            m && one_mesh(r, "gltf.sparse-interleaved", *m))
        {
            compare(r, "gltf.sparse-interleaved-positions", *m, make_triangles(expected), 0);
            r.exact("gltf.sparse-interleaved-normals",
                    attribute_deviation(m->meshes[0], m->meshes[0].normals, 3, expected, normals), 0, 1e-6);
        }
    }

    // A strip and a fan (glTF 2.0 "Meshes", modes 5 and 6): a strip's triangle i is (vᵢ, vᵢ₊₁, vᵢ₊₂) for even i and
    // (vᵢ, vᵢ₊₂, vᵢ₊₁) for odd i, a fan's (vᵢ₊₁, vᵢ₊₂, v₀). Here every one of them turns counter-clockwise about +z.
    // The strip's positions are stored backwards and indexed (u16) in strip order; the fan, moved by 3 in x, isn't
    // indexed. The comparison matches corners up to a rotation, never a reflection: a wrong odd triangle fails it.
    {
        const std::array<vec, 5> strip{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 2, 0}}};
        const std::array<vec, 5> fan{{{3, 0, 0}, {4, 0, 0}, {4, 1, 0}, {3, 1, 0}, {2, 1, 0}}};
        std::array<float, 15> stored{};
        std::array<float, 15> fan_floats{};
        for (std::size_t k = 0; k < 5; ++k)
        {
            for (std::size_t c = 0; c < 3; ++c)
            {
                stored[3 * k + c] = static_cast<float>(strip[4 - k][c]);
                fan_floats[3 * k + c] = static_cast<float>(fan[k][c]);
            }
        }
        constexpr std::array<std::uint32_t, 5> order{4, 3, 2, 1, 0}; // stored[4 - i] is strip[i]
        std::vector<triangle> expected;
        for (std::size_t i = 0; i < 3; ++i)
        {
            expected.push_back(i % 2 == 0 ? triangle{strip[i], strip[i + 1], strip[i + 2]}
                                          : triangle{strip[i], strip[i + 2], strip[i + 1]});
        }
        for (std::size_t i = 0; i < 3; ++i)
        {
            expected.push_back({fan[i + 1], fan[i + 2], fan[0]});
        }
        std::size_t not_up = 0;
        for (const triangle& t : expected)
        {
            not_up += face_normal(t)[2] > 0 ? 0 : 1;
        }
        r.check("gltf.strip-fan-expected-up", not_up == 0,
                std::format("{} of 6 expected triangles not along +z", not_up),
                "all along +z (the test's own geometry)");
        gltf_writer w;
        add_mesh(w, {{.positions = stored, .indices = order, .index_type = 5123, .mode = 5},
                     {.positions = fan_floats, .mode = 6}});
        const load_result result = load(r, "gltf.strip-fan", one_node(w, folder / "strip-fan.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.strip-fan", result))
        {
            compare(r, "gltf.strip-fan-triangles", *m, expected, 0);
        }
    }

    // Two primitives over one POSITION accessor of 9 vertices (a 3 × 3 grid), each with its own indices (an exporter's
    // two material groups): one range of 9 vertices in the mesh, not one per primitive, and 9 vertices in the plan, so
    // a limit of 3 triangles (at most 9 vertices) lets it load; 18 would be refused
    {
        std::array<float, 27> grid{};
        for (std::size_t i = 0; i < 9; ++i)
        {
            grid[3 * i] = static_cast<float>(i % 3);
            grid[3 * i + 1] = static_cast<float>(i / 3);
        }
        constexpr std::array<std::uint32_t, 3> first{0, 1, 4};
        constexpr std::array<std::uint32_t, 3> second{4, 5, 8};
        gltf_writer w;
        const int pa = w.floats(grid, "VEC3", true);
        const int ia = w.indices(first, 5125);
        const int ib = w.indices(second, 5125);
        w.add("meshes", std::format(R"({{"primitives": [{{"attributes": {{"POSITION": {}}}, "indices": {}}}, )"
                                    R"({{"attributes": {{"POSITION": {}}}, "indices": {}}}]}})",
                                    pa, ia, pa, ib));
        const std::filesystem::path file = one_node(w, folder / "shared-vertices.gltf");
        const load_result result = load(r, "gltf.shared-vertices", file, {.triangles = 3});
        if (const meshes::model* m = loaded(r, "gltf.shared-vertices", result);
            m && one_mesh(r, "gltf.shared-vertices", *m))
        {
            r.equal("gltf.shared-vertices-count", m->meshes[0].positions.size() / 3, 9);
            std::vector<triangle> expected = make_triangles(grid, first);
            expected.push_back(make_triangles(grid, second)[0]);
            compare(r, "gltf.shared-vertices-triangles", *m, expected, 0);
        }
    }
}

// --- glTF: the node walk --------------------------------------------------------------------------------------------

void gltf_transforms(report& r, const std::filesystem::path& folder)
{
    const auto tetra_file = [&folder](std::string_view name, std::string_view members) {
        gltf_writer w;
        add_mesh(w, {{.positions = tetra_positions, .indices = tetra_indices}});
        return one_node(w, folder / name, members);
    };

    // T·R·S (glTF §3.5.3): translation (1, 2, 3), a 90° turn about y ((x, y, z) → (z, y, -x)), scale (2, 3, 0.5), so
    // world = (0.5·z + 1, 3·y + 2, -2·x + 3); the loader composes in double: the corners to 1e-12
    {
        constexpr matrix expected{0, 0, 0.5, 1, 0, 3, 0, 2, -2, 0, 0, 3};
        const load_result result =
            load(r, "gltf.trs",
                 tetra_file("trs.gltf", std::format(R"(, "translation": [1, 2, 3], "rotation": {}, )"
                                                    R"("scale": [2, 3, 0.5])",
                                                    quarter_turn_y)));
        if (const meshes::model* m = loaded(r, "gltf.trs", result))
        {
            const std::vector<triangle> world = make_triangles(tetra_positions, tetra_indices, expected);
            compare(r, "gltf.trs-world", *m, world, 1e-12);
            compare_bounds(r, "gltf.trs-bounds", *m, world, 1e-12);
        }
    }

    // A parent (translation (10, 0, 0) after the quarter turn) and its child (translation (1, 0, 0)) with the mesh:
    // world = parent·child, (x, y, z) → (z + 10, y, -x - 1). Node 2 is in scene 1, not the default scene 0: not loaded.
    {
        gltf_writer w;
        add_mesh(w, {{.positions = triangle_positions}});
        w.add("nodes",
              std::format(R"({{"translation": [10, 0, 0], "rotation": {}, "children": [1]}})", quarter_turn_y));
        w.add("nodes", R"({"mesh": 0, "translation": [1, 0, 0]})");
        w.add("nodes", R"({"mesh": 0, "translation": [0, 100, 0]})");
        w.add("scenes", R"({"nodes": [0]})");
        w.add("scenes", R"({"nodes": [2]})");
        const load_result result = load(r, "gltf.hierarchy", w.write(folder / "hierarchy.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.hierarchy", result))
        {
            constexpr matrix expected{0, 0, 1, 10, 0, 1, 0, 0, -1, 0, 0, -1};
            r.equal("gltf.hierarchy-placements", m->placements.size(), 1);
            compare(r, "gltf.hierarchy-world", *m, make_triangles(triangle_positions, {}, expected), 1e-12);
        }
    }

    // A matrix node (column-major in glTF): a turn of 90° about z times scale (1, 2, 4), translation (5, 6, 7);
    // world = (-2·y + 5, x + 6, 4·z + 7) (a transposed reading would differ)
    {
        constexpr matrix expected{0, -2, 0, 5, 1, 0, 0, 6, 0, 0, 4, 7};
        const load_result result = load(
            r, "gltf.matrix",
            tetra_file("matrix.gltf", R"(, "matrix": [0, 1, 0, 0, -2, 0, 0, 0, 0, 0, 4, 0, 5, 6, 7, 1])"));
        if (const meshes::model* m = loaded(r, "gltf.matrix", result))
        {
            compare(r, "gltf.matrix-world", *m, make_triangles(tetra_positions, tetra_indices, expected), 1e-12);
        }
    }

    // A mirroring node, scale (-1, 0.5, 2) and translation (3, 1, -2): det M < 0. The loader keeps the file's winding
    // (no index swap; glTF §3.7.4: the determinant decides the winding), so the renderer's face normal
    // M⁻ᵀ(a × b) still points out of the tetrahedron; bake, which swaps indices 1 and 2 under det < 0, gives a world
    // mesh whose plain a × b points out too.
    {
        constexpr matrix expected{-1, 0, 0, 3, 0, 0.5, 0, 1, 0, 0, 2, -2};
        const load_result result =
            load(r, "gltf.mirror", tetra_file("mirror.gltf", R"(, "translation": [3, 1, -2], "scale": [-1, 0.5, 2])"));
        if (const meshes::model* m = loaded(r, "gltf.mirror", result); m && one_mesh(r, "gltf.mirror", *m))
        {
            compare(r, "gltf.mirror-world", *m, make_triangles(tetra_positions, tetra_indices, expected), 1e-12);
            const meshes::mesh& mesh = m->meshes[0];
            const matrix& to_world = m->placements[0].object_to_world;
            const vec centre = point(to_world, {0, 0, 0});
            const double det = determinant(to_world);
            std::size_t inward = 0;
            for (std::size_t t = 0; t < triangle_count(mesh); ++t)
            {
                const triangle c = corners(mesh, t);
                const vec n = normal_to_world(to_world, face_normal(c));
                inward += dot(n, sub(point(to_world, centroid(c)), centre)) > 0 ? 0 : 1;
            }
            r.check("gltf.mirror-outside", det < 0 && inward == 0,
                    std::format("det M = {}, {} of {} faces with M^-T(a x b) inward", det, inward,
                                triangle_count(mesh)),
                    "det M < 0, no face inward");
            const meshes::mesh baked = meshes::bake(mesh, to_world);
            const std::string problem = mesh_problem(baked, 0);
            const std::size_t baked_inward =
                problem.empty() ? inward_faces(baked, [&centre](const vec&) { return centre; }) : triangle_count(mesh);
            r.check("gltf.mirror-baked-outside", problem.empty() && baked_inward == 0 && triangle_count(baked) == 4,
                    problem.empty() ? std::format("{} of {} baked faces with a x b inward", baked_inward,
                                                  triangle_count(baked))
                                    : problem,
                    "4 faces, none inward");
        }
    }

    // Two nodes sharing a mesh: one mesh, two placements of it; the bounds hold both
    {
        const load_result result = load(r, "gltf.shared", shared_file(folder / "shared.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.shared", result))
        {
            const bool shared = m->meshes.size() == 1 && m->placements.size() == 2 && m->placements[0].mesh == 0 &&
                                m->placements[1].mesh == 0;
            r.check("gltf.shared-mesh", shared,
                    std::format("{} meshes, {} placements", m->meshes.size(), m->placements.size()),
                    "1 mesh, 2 placements of it");
            std::vector<triangle> expected = make_triangles(triangle_positions, {}, moved(-2, 0, 0));
            expected.push_back(make_triangles(triangle_positions, {}, moved(2, 0, 0))[0]);
            compare(r, "gltf.shared-world", *m, expected, 0);
            compare_bounds(r, "gltf.shared-bounds", *m, expected, 0);
        }
    }

    // No "scene" and no "scenes": the root nodes (those without a parent) are shown. Node 0 at (5, 0, 0) with its
    // child node 1 at (0, 5, 0) in it, and node 2 at (0, 0, 5): three placements, the child at (5, 5, 0)
    {
        gltf_writer w;
        add_mesh(w, {{.positions = triangle_positions}});
        w.add("nodes", R"({"mesh": 0, "translation": [5, 0, 0], "children": [1]})");
        w.add("nodes", R"({"mesh": 0, "translation": [0, 5, 0]})");
        w.add("nodes", R"({"mesh": 0, "translation": [0, 0, 5]})");
        const load_result result = load(r, "gltf.no-scene", w.write(folder / "no-scene.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.no-scene", result))
        {
            std::vector<triangle> expected;
            for (const matrix& at : {moved(5, 0, 0), moved(5, 5, 0), moved(0, 0, 5)})
            {
                expected.push_back(make_triangles(triangle_positions, {}, at)[0]);
            }
            r.equal("gltf.no-scene-placements", m->placements.size(), 3);
            compare(r, "gltf.no-scene-world", *m, expected, 0);
        }
    }

    // No nodes at all: each mesh once, untransformed
    {
        std::array<float, 9> moved_positions = triangle_positions;
        moved_positions[0] = -1;
        gltf_writer w;
        add_mesh(w, {{.positions = triangle_positions}});
        add_mesh(w, {{.positions = moved_positions}});
        const load_result result = load(r, "gltf.no-nodes", w.write(folder / "no-nodes.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.no-nodes", result))
        {
            std::vector<triangle> expected = make_triangles(triangle_positions);
            expected.push_back(make_triangles(moved_positions)[0]);
            r.equal("gltf.no-nodes-placements", m->placements.size(), 2);
            compare(r, "gltf.no-nodes-world", *m, expected, 0);
        }
    }
}

// --- glTF: triangles dropped with a warning -------------------------------------------------------------------------

void gltf_dropped(report& r, const std::filesystem::path& folder)
{
    // Three triangles, the second with a NaN corner and the third with an infinite one: one kept, a warning, and the
    // bounds those of the kept triangle (finite)
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();
        const std::array<float, 15> positions{0, 0, 0, 1, 0, 0, 0, 1, 0, nan, 0, 0, 0, inf, 0};
        constexpr std::array<std::uint32_t, 9> order{0, 1, 2, 0, 2, 3, 0, 1, 4};
        gltf_writer w;
        add_mesh(w, {{.positions = positions, .indices = order}});
        const load_result result = load(r, "gltf.non-finite", one_node(w, folder / "non-finite.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.non-finite", result))
        {
            const std::vector<triangle> kept = make_triangles(triangle_positions);
            compare(r, "gltf.non-finite-kept", *m, kept, 0);
            r.check("gltf.non-finite-warned", !m->warnings.empty(), describe(m->warnings), "at least one warning");
            compare_bounds(r, "gltf.non-finite-bounds", *m, kept, 0);
        }
    }

    // Four triangles: a good one; three corners on a line (|a × b| = 0); a sliver of |a × b| = 1e-13 (≤ 1e-12·extent²,
    // the extent being 2: dropped); a sliver of 1e-9 (above it: kept)
    {
        constexpr std::array<float, 18> positions{0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 2, 1e-13f, 0, 2, 1e-9f, 0};
        constexpr std::array<std::uint32_t, 12> order{0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5};
        constexpr std::array<std::uint32_t, 6> kept_order{0, 1, 2, 0, 1, 5};
        gltf_writer w;
        add_mesh(w, {{.positions = positions, .indices = order}});
        const load_result result = load(r, "gltf.zero-area", one_node(w, folder / "zero-area.gltf"));
        if (const meshes::model* m = loaded(r, "gltf.zero-area", result))
        {
            compare(r, "gltf.zero-area-kept", *m, make_triangles(positions, kept_order), 0);
            r.check("gltf.zero-area-warned", !m->warnings.empty(), describe(m->warnings), "at least one warning");
        }
    }
}

// --- glTF: what the loader refuses ----------------------------------------------------------------------------------

void gltf_refusals(report& r, const std::filesystem::path& folder)
{
    // No such file
    {
        const std::filesystem::path file = folder / "nowhere.gltf";
        refused(r, "refusals.no-such-file", load(r, "refusals.no-such-file", file), file);
    }

    // An index past the vertices (3 with 3 vertices): cgltf_validate's "data too short", or the loader's own check
    {
        constexpr std::array<std::uint32_t, 3> order{0, 1, 3};
        gltf_writer w;
        add_mesh(w, {{.positions = triangle_positions, .indices = order}});
        const std::filesystem::path file = one_node(w, folder / "index-past-end.gltf");
        refused(r, "refusals.index-past-vertices", load(r, "refusals.index-past-vertices", file), file);
    }

    // A node cycle through the scene's root, 0 → 1 → 0, each node with the mesh: cgltf_parse refuses it (the root is
    // node 1's child, and a scene's roots have no parent). The tiny caps on placements stop even a walk that went
    // round without that check.
    {
        gltf_writer w;
        add_mesh(w, {{.positions = triangle_positions}});
        w.add("nodes", R"({"mesh": 0, "children": [1]})");
        w.add("nodes", R"({"mesh": 0, "children": [0]})");
        w.add("scenes", R"({"nodes": [0]})");
        const std::filesystem::path file = w.write(folder / "cycle.gltf");
        refused(r, "refusals.node-cycle",
                load(r, "refusals.node-cycle", file, {.placed_triangles = 4, .placements = 4}), file);
    }

    // A cycle off the scene (scene [0]; nodes 1 → 2 → 1): cgltf_parse lets it through, cgltf_validate refuses it, and
    // the message names the node
    {
        gltf_writer w;
        add_mesh(w, {{.positions = triangle_positions}});
        w.add("nodes", R"({"mesh": 0})");
        w.add("nodes", R"({"children": [2]})");
        w.add("nodes", R"({"children": [1]})");
        w.add("scenes", R"({"nodes": [0]})");
        const std::filesystem::path file = w.write(folder / "cycle-off-scene.gltf");
        refused(r, "refusals.node-cycle-off-scene", load(r, "refusals.node-cycle-off-scene", file), file, {"nodes["});
    }

    // Sizes that wrap modulo 2^64 in cgltf_validate's own sums (cgltf reads them with atoll, up to 2^63 − 1), refused
    // by name before anything is read. A 12-byte view of byteStride 2^60 under 33 POSITION elements (2^60·32 = 2^65
    // ≡ 0, so validate's sum is 12): unpacking would step 2^60 bytes.
    {
        constexpr std::array<float, 3> one_vertex{0, 0, 0};
        gltf_writer w;
        w.view(std::as_bytes(std::span(one_vertex)), 0, 34962);
        const int strided = w.add("bufferViews", R"({"buffer": 0, "byteOffset": 0, "byteLength": 12, )"
                                                 R"("byteStride": 1152921504606846976})");
        const int pa = w.add("accessors", std::format(R"({{"bufferView": {}, "componentType": 5126, "count": 33, )"
                                                      R"("type": "VEC3", "min": [0, 0, 0], "max": [0, 0, 0]}})",
                                                      strided));
        w.add("meshes", std::format(R"({{"primitives": [{{"attributes": {{"POSITION": {}}}}}]}})", pa));
        const std::filesystem::path file = one_node(w, folder / "stride-wraps.gltf");
        refused(r, "refusals.stride-wraps", load(r, "refusals.stride-wraps", file), file,
                {std::format("bufferViews[{}]", strided), "byteStride"});
    }

    // An index accessor of 2^62 + 2 u32 indices over an 8-byte view: 4·(2^62 + 1) + 4 = 2^64 + 8 ≡ 8, so validate's
    // sum fits the view; the refusal names the accessor (a triangle limit's refusal would not)
    {
        constexpr std::array<std::uint32_t, 2> two{0, 1};
        gltf_writer w;
        const int pa = w.floats(triangle_positions, "VEC3", true);
        const int iv = w.view(std::as_bytes(std::span(two)), 0, 34963);
        const int ia = w.add("accessors", std::format(R"({{"bufferView": {}, "componentType": 5125, )"
                                                      R"("count": 4611686018427387906, "type": "SCALAR"}})",
                                                      iv));
        w.add("meshes", std::format(R"({{"primitives": [{{"attributes": {{"POSITION": {}}}, "indices": {}}}]}})", pa,
                                    ia));
        const std::filesystem::path file = one_node(w, folder / "count-wraps.gltf");
        refused(r, "refusals.count-wraps", load(r, "refusals.count-wraps", file), file,
                {std::format("accessors[{}]", ia)});
    }

    // URIs that aren't UTF-8 once their percent escapes are decoded ("%FF" is the byte 0xFF): refused by name, before
    // they become paths (std::filesystem would throw)
    {
        const std::filesystem::path image = textured({}, [](gltf_writer&) {
                                                return std::string(R"({"uri": "tex%FF.png"})");
                                            }).write(folder / "image-uri-not-utf8.gltf");
        refused(r, "refusals.image-uri-not-utf8", load(r, "refusals.image-uri-not-utf8", image), image,
                {"images[0]", "UTF-8"});
        const std::filesystem::path buffer =
            textured([](gltf_writer& g, textured_parts&) { g.buffer_uri = "geo%FF.bin"; })
                .write(folder / "buffer-uri-not-utf8.gltf");
        refused(r, "refusals.buffer-uri-not-utf8", load(r, "refusals.buffer-uri-not-utf8", buffer), buffer,
                {"buffers[0]", "UTF-8"});
    }

    // Required extensions: compression ("export the model without compression", Draco by name), texture
    // transforms (refused when required), and one nobody knows (refused by name)
    struct refusal
    {
        std::string_view extension;
        std::string_view check;
        std::string_view says; // empty: only refused
    };
    for (const refusal& x : {refusal{"KHR_draco_mesh_compression", "draco", "without compression"},
                             refusal{"EXT_meshopt_compression", "meshopt", ""},
                             refusal{"KHR_texture_basisu", "basisu", ""}, refusal{"EXT_texture_webp", "webp", ""},
                             refusal{"KHR_texture_transform", "texture-transform-required", "KHR_texture_transform"},
                             refusal{"EXT_mesh_tests_unknown", "unknown-required", "EXT_mesh_tests_unknown"}})
    {
        gltf_writer w;
        w.extension(x.extension, true);
        add_mesh(w, {{.positions = triangle_positions}});
        const std::filesystem::path file = one_node(w, folder / std::format("required-{}.gltf", x.check));
        const std::string name = std::format("refusals.{}", x.check);
        const load_result result = load(r, name, file);
        if (x.says.empty())
        {
            refused(r, name, result, file);
        }
        else
        {
            refused(r, name, result, file, {x.says});
        }
    }

    // A base-color image whose file isn't there: an error naming images[0]
    {
        const std::filesystem::path file =
            textured({}, [](gltf_writer&) { return std::string(R"({"uri": "nowhere.png"})"); })
                .write(folder / "missing-image.gltf");
        refused(r, "refusals.missing-image", load(r, "refusals.missing-image", file), file, {"images[0]"});
    }
}

// --- Limits: tiny caps on both sides of each boundary ("refused with the numbers, before allocating") -----------

void limit_checks(report& r, const std::filesystem::path& folder)
{
    // file_bytes: the file's own size passes, one byte less doesn't
    {
        const std::filesystem::path file = triangle_file(folder / "limit-file.gltf");
        std::error_code failed;
        const std::uintmax_t size = std::filesystem::file_size(file, failed);
        if (failed)
        {
            throw setup_error{"can't read the size of limit-file.gltf"};
        }
        const load_result over = load(r, "limits.file-bytes", file, {.file_bytes = size - 1});
        refused(r, "limits.file-bytes", over, file);
        const std::string size_text = std::to_string(size), cap_text = std::to_string(size - 1);
        r.check("limits.file-bytes-numbers",
                !over && (over.error().contains(size_text) || over.error().contains(cap_text)),
                over ? std::string("a model") : std::format("'{}'", over.error()),
                std::format("the size {} or the cap {} in the message", size_text, cap_text));
        loaded(r, "limits.file-bytes-at-cap", load(r, "limits.file-bytes-at-cap", file, {.file_bytes = size}));
    }

    // triangles: five in the model
    {
        constexpr std::array<float, 24> strip{0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 1, 0, 3, 0, 0, 3, 1, 0};
        constexpr std::array<std::uint32_t, 15> order{0, 2, 3, 0, 3, 1, 2, 4, 5, 2, 5, 3, 4, 6, 7};
        gltf_writer w;
        add_mesh(w, {{.positions = strip, .indices = order}});
        const std::filesystem::path file = one_node(w, folder / "limit-triangles.gltf");
        refused(r, "limits.triangles", load(r, "limits.triangles", file, {.triangles = 4}), file);
        loaded(r, "limits.triangles-at-cap", load(r, "limits.triangles-at-cap", file, {.triangles = 5}));
    }

    // One triangle placed twice: 1 triangle in the model, 2 placed, 2 placements
    {
        const std::filesystem::path file = shared_file(folder / "limit-shared.gltf");
        loaded(r, "limits.triangles-of-the-model",
               load(r, "limits.triangles-of-the-model", file, {.triangles = 1}));
        refused(r, "limits.placed-triangles", load(r, "limits.placed-triangles", file, {.placed_triangles = 1}),
                file);
        loaded(r, "limits.placed-triangles-at-cap",
               load(r, "limits.placed-triangles-at-cap", file, {.placed_triangles = 2}));
        refused(r, "limits.placements", load(r, "limits.placements", file, {.placements = 1}), file);
        loaded(r, "limits.placements-at-cap", load(r, "limits.placements-at-cap", file, {.placements = 2}));
    }

    // A 2×2 base-color image: side 2, 16 bytes decoded (RGBA)
    {
        const std::filesystem::path file = textured().write(folder / "limit-image.gltf");
        refused(r, "limits.image-side", load(r, "limits.image-side", file, {.image_side = 1}), file);
        loaded(r, "limits.image-side-at-cap", load(r, "limits.image-side-at-cap", file, {.image_side = 2}));
        refused(r, "limits.image-bytes", load(r, "limits.image-bytes", file, {.image_bytes = 15}), file);
        loaded(r, "limits.image-bytes-at-cap", load(r, "limits.image-bytes-at-cap", file, {.image_bytes = 16}));
    }

    // Counts near 2^63 can't wrap a sum under a limit. One mesh of four strips over one 5-vertex POSITION: 5 indices
    // (3 triangles), then two index accessors without a buffer view (zeros, allowed by glTF) of 2^63 − 1 indices,
    // then 7: 3 + 2·(2^63 − 3) + 5 = 2^64 + 2, which a sum modulo 2^64 would take for 2 triangles. Refused by the
    // triangle limit at the second strip, before anything is allocated (a vector of 2^63 floats can't be asked for).
    {
        constexpr std::array<float, 15> strip{0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 2, 0};
        constexpr std::array<std::uint32_t, 5> order{0, 1, 2, 3, 4};
        gltf_writer w;
        const int pa = w.floats(strip, "VEC3", true);
        const int ia = w.indices(order, 5125);
        const auto zeros = [&w](std::string_view count) {
            return w.add("accessors",
                         std::format(R"({{"componentType": 5125, "count": {}, "type": "SCALAR"}})", count));
        };
        std::string primitives;
        for (const int indices : {ia, zeros("9223372036854775807"), zeros("9223372036854775807"), zeros("7")})
        {
            primitives += std::format(R"({}{{"attributes": {{"POSITION": {}}}, "indices": {}, "mode": 5}})",
                                      primitives.empty() ? "" : ", ", pa, indices);
        }
        w.add("meshes", R"({"primitives": [)" + primitives + "]}");
        const std::filesystem::path file = one_node(w, folder / "limit-huge-count.gltf");
        refused(r, "limits.huge-count", load(r, "limits.huge-count", file), file,
                {"meshes[0].primitives[1]", "limit"});
    }

    // The buffer files together: two buffers naming one 1024-byte file beside the model (the .gltf is smaller), each
    // within a file limit of 1024, 2048 bytes in all; a limit of 2048 lets it load
    {
        std::vector<std::uint8_t> bin(1024, 0);
        const auto positions = std::as_bytes(std::span(triangle_positions));
        std::ranges::transform(positions, bin.begin(), [](std::byte b) { return static_cast<std::uint8_t>(b); });
        write_bytes(folder / "twice.bin", std::as_bytes(std::span(bin)));
        const std::string json =
            R"({"asset": {"version": "2.0"}, "buffers": [{"uri": "twice.bin", "byteLength": 1024}, )"
            R"({"uri": "twice.bin", "byteLength": 1024}], "bufferViews": [{"buffer": 0, "byteLength": 36}], )"
            R"("accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", )"
            R"("min": [0, 0, 0], "max": [1, 1, 0]}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}], )"
            R"("nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0})";
        const std::filesystem::path file = folder / "limit-buffer-files.gltf";
        write_bytes(file, std::as_bytes(std::span(json)));
        refused(r, "limits.buffer-files", load(r, "limits.buffer-files", file, {.file_bytes = 1024}), file,
                {"2 buffer files", "2048"});
        loaded(r, "limits.buffer-files-at-cap", load(r, "limits.buffer-files-at-cap", file, {.file_bytes = 2048}));
    }

    // A .bin and a .png beside the model at file_bytes = their size (loads) and one byte less (refused, by name). The
    // .bin holds 2 kB of padding and the PNG is 32 × 32, so each is larger than its .gltf.
    {
        std::vector<std::byte> padding(2048);
        const gltf_writer w = textured([&padding](gltf_writer& g, textured_parts&) {
            g.buffer_uri = "limit.bin";
            g.view(padding);
        });
        write_bytes(folder / "limit.bin", std::as_bytes(std::span(w.bin())));
        const std::filesystem::path file = w.write(folder / "limit-bin.gltf");
        const std::uintmax_t size = w.bin().size();
        refused(r, "limits.bin-file-bytes", load(r, "limits.bin-file-bytes", file, {.file_bytes = size - 1}), file,
                {"buffers[0]"});
        loaded(r, "limits.bin-file-bytes-at-cap", load(r, "limits.bin-file-bytes-at-cap", file, {.file_bytes = size}));
    }
    {
        const std::vector<std::uint8_t> texels(32 * 32 * 4, 128);
        const std::vector<std::uint8_t> png = make_png(32, 32, 4, texels);
        write_bytes(folder / "limit.png", std::as_bytes(std::span(png)));
        const std::filesystem::path file = textured({}, [](gltf_writer&) {
                                               return std::string(R"({"uri": "limit.png"})");
                                           }).write(folder / "limit-png.gltf");
        const std::uintmax_t size = png.size();
        refused(r, "limits.png-file-bytes", load(r, "limits.png-file-bytes", file, {.file_bytes = size - 1}), file,
                {"images[0]"});
        loaded(r, "limits.png-file-bytes-at-cap", load(r, "limits.png-file-bytes-at-cap", file, {.file_bytes = size}));
    }

    // The JSON's own memory and the node count, on 8000 empty nodes (24 kB of "{},"). cgltf's memory is limited to
    // twice the file limit plus 1 MiB: at a file limit of the file's size that is about 1.1 MB, less than the nodes'
    // structures alone (8000 of a few hundred bytes); the default limit (513 MiB) lets it load. Nodes are limited to
    // 4 per allowed placement plus 4096: 975 placements allow 7996, 976 allow 8000.
    {
        std::string json = R"({"asset": {"version": "2.0"}, "nodes": [{})";
        for (int i = 1; i < 8000; ++i)
        {
            json += ",{}";
        }
        json += "]}";
        const std::filesystem::path file = folder / "limit-nodes.gltf";
        write_bytes(file, std::as_bytes(std::span(json)));
        refused(r, "limits.json-memory", load(r, "limits.json-memory", file, {.file_bytes = json.size()}), file,
                {"memory"});
        loaded(r, "limits.json-memory-default", load(r, "limits.json-memory-default", file));
        refused(r, "limits.nodes", load(r, "limits.nodes", file, {.placements = 975}), file, {"8000 nodes", "7996"});
        loaded(r, "limits.nodes-at-cap", load(r, "limits.nodes-at-cap", file, {.placements = 976}));
    }

    // A normal map is never decoded (the renderer ignores it): image_side 1 doesn't refuse a file whose only
    // texture is one
    {
        const std::filesystem::path file = textured([](gltf_writer&, textured_parts& p) {
                                               p.base_texture = false;
                                               p.material_members = R"(, "normalTexture": {"index": 0})";
                                           }).write(folder / "limit-normal-map.gltf");
        loaded(r, "limits.normal-map-not-decoded",
               load(r, "limits.normal-map-not-decoded", file, {.image_side = 1}));
    }
}

// --- Materials ------------------------------------------------------------------------------------------------------

void material_checks(report& r, const std::filesystem::path& folder)
{
    // One material with factors only, on a primitive; a second primitive (the triangle moved by +2 in x) names none
    std::array<float, 9> moved_positions = triangle_positions;
    for (std::size_t i = 0; i < moved_positions.size(); i += 3)
    {
        moved_positions[i] += 2;
    }
    gltf_writer w;
    w.add("materials", R"({"pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.25, 0.125, 1], )"
                       R"("metallicFactor": 0.75, "roughnessFactor": 0.5}})");
    add_mesh(w, {{.positions = triangle_positions, .material = 0}, {.positions = moved_positions}});
    const load_result result = load(r, "materials", one_node(w, folder / "materials.gltf"));
    const meshes::model* m = loaded(r, "materials", result);
    if (!m)
    {
        return;
    }
    r.equal("materials.count", m->materials.size(), 2);
    const std::optional<std::uint32_t> first = material_at(*m, make_triangles(triangle_positions)[0]);
    const std::optional<std::uint32_t> second = material_at(*m, make_triangles(moved_positions)[0]);
    const meshes::pbr_material* a = first ? &m->materials[*first] : nullptr;
    r.check("materials.factors",
            a && a->base_color == std::array<double, 3>{0.5, 0.25, 0.125} && a->metallic == 0.75 &&
                a->roughness == 0.5 && a->base_color_image == -1 && a->metal_rough_image == -1,
            a ? describe(*a) : std::string("no such triangle"),
            "base (0.5, 0.25, 0.125), metallic 0.75, roughness 0.5, images -1 -1");
    const bool last = second && *second + 1 == m->materials.size();
    r.check("materials.default", last && is_default(m->materials[*second]),
            second ? std::format("materials[{}] of {}: {}", *second, m->materials.size(),
                                 describe(m->materials[*second]))
                   : std::string("no such triangle"),
            "the last material: base (1, 1, 1), metallic 1, roughness 1, images -1 -1");
}

// --- Images ---------------------------------------------------------------------------------------------------------

void image_checks(report& r, const std::filesystem::path& folder)
{
    const meshes::image a = expected_image(texels_a, true, meshes::wrap::repeat, meshes::wrap::repeat, false);

    // The clean textured triangle: no warning, its UVs as written ((0, 0) the image's top-left: no flip), the PNG
    // (a data: URI) decoded to its exact bytes, sRGB as a base color
    {
        const load_result result = load(r, "images.png", textured().write(folder / "png.gltf"));
        if (const meshes::model* m = loaded(r, "images.png", result); m && one_mesh(r, "images.png", *m))
        {
            r.check("images.png-warnings", m->warnings.empty(), describe(m->warnings), "no warning");
            r.exact("images.png-uvs",
                    attribute_deviation(m->meshes[0], m->meshes[0].uvs, 2, triangle_positions, triangle_uvs), 0, 0);
            const meshes::pbr_material* material = first_material(*m);
            check_image(r, "images.png-bytes", material ? image_at(*m, material->base_color_image) : nullptr, a);
        }
    }

    // A .glb: the geometry and the PNG in its BIN chunk (the image by bufferView and mimeType)
    {
        const gltf_writer w = textured({}, [](gltf_writer& g) {
            return std::format(R"({{"bufferView": {}, "mimeType": "image/png"}})",
                               g.view(std::as_bytes(std::span(png_a()))));
        });
        const load_result result = load(r, "images.glb", w.write(folder / "embedded.glb"));
        if (const meshes::model* m = loaded(r, "images.glb", result))
        {
            compare(r, "images.glb-triangle", *m, make_triangles(triangle_positions), 0);
            const meshes::pbr_material* material = first_material(*m);
            check_image(r, "images.glb-bytes", material ? image_at(*m, material->base_color_image) : nullptr, a);
        }
    }

    // A .gltf whose buffer and image are files beside it, their URIs percent-escaped ("geometry%20data.bin"), in a
    // folder with a non-ASCII name (UTF-8 paths)
    {
        const std::filesystem::path sub = folder / std::filesystem::path(u8"модели é");
        std::error_code failed;
        std::filesystem::create_directories(sub, failed);
        if (failed)
        {
            throw setup_error{"can't create the test folder with a non-ASCII name"};
        }
        const gltf_writer w = textured([](gltf_writer& g, textured_parts&) { g.buffer_uri = "geometry%20data.bin"; },
                                       [](gltf_writer&) { return std::string(R"({"uri": "base%20color.png"})"); });
        write_bytes(sub / "geometry data.bin", std::as_bytes(std::span(w.bin())));
        write_bytes(sub / "base color.png", std::as_bytes(std::span(png_a())));
        const load_result result = load(r, "images.external-files", w.write(sub / "model.gltf"));
        if (const meshes::model* m = loaded(r, "images.external-files", result))
        {
            compare(r, "images.external-files-triangle", *m, make_triangles(triangle_positions), 0);
            const meshes::pbr_material* material = first_material(*m);
            check_image(r, "images.external-files-bytes",
                        material ? image_at(*m, material->base_color_image) : nullptr, a);
        }
    }

    // The samplers: material 0 (metallic 0.25) has base color A through sampler 0 (magFilter NEAREST 9728, wrapS
    // MIRRORED_REPEAT 33648, wrapT CLAMP_TO_EDGE 33071) and metallic-roughness B (RGB) without a sampler (glTF's
    // defaults: REPEAT, bilinear); material 1 (metallic 0.75) has base color C through sampler 1 (magFilter LINEAR
    // 9729, wrapS CLAMP_TO_EDGE, wrapT REPEAT 10497). Base colors are sRGB, metallic-roughness linear.
    {
        gltf_writer w;
        const int position = w.floats(triangle_positions, "VEC3", true);
        const int texcoord = w.floats(triangle_uvs, "VEC2");
        w.add("images", data_image(png_a()));
        w.add("images", data_image(png_b()));
        w.add("images", data_image(png_c()));
        w.add("samplers", R"({"magFilter": 9728, "minFilter": 9728, "wrapS": 33648, "wrapT": 33071})");
        w.add("samplers", R"({"magFilter": 9729, "wrapS": 33071, "wrapT": 10497})");
        w.add("textures", R"({"source": 0, "sampler": 0})");
        w.add("textures", R"({"source": 1})");
        w.add("textures", R"({"source": 2, "sampler": 1})");
        w.add("materials", R"({"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, )"
                           R"("metallicRoughnessTexture": {"index": 1}, "metallicFactor": 0.25}})");
        w.add("materials", R"({"pbrMetallicRoughness": {"baseColorTexture": {"index": 2}, "metallicFactor": 0.75}})");
        const std::string attributes = std::format(R"({{"POSITION": {}, "TEXCOORD_0": {}}})", position, texcoord);
        w.add("meshes", R"({"primitives": [{"attributes": )" + attributes + R"(, "material": 0}, {"attributes": )" +
                            attributes + R"(, "material": 1}]})");
        const load_result result = load(r, "images.samplers", one_node(w, folder / "samplers.gltf"));
        if (const meshes::model* m = loaded(r, "images.samplers", result))
        {
            const meshes::pbr_material* m0 = material_by_metallic(*m, 0.25);
            const meshes::pbr_material* m1 = material_by_metallic(*m, 0.75);
            check_image(r, "images.samplers-nearest-mirrored-clamp", m0 ? image_at(*m, m0->base_color_image) : nullptr,
                        expected_image(texels_a, true, meshes::wrap::mirrored_repeat, meshes::wrap::clamp, true));
            check_image(r, "images.samplers-default-rgb-linear", m0 ? image_at(*m, m0->metal_rough_image) : nullptr,
                        expected_image(texels_b_rgba, false, meshes::wrap::repeat, meshes::wrap::repeat, false));
            check_image(r, "images.samplers-bilinear-clamp-repeat", m1 ? image_at(*m, m1->base_color_image) : nullptr,
                        expected_image(texels_c, true, meshes::wrap::clamp, meshes::wrap::repeat, false));
        }
    }

    // One image behind two textures, the base colors of two materials: decoded once, both naming it
    {
        gltf_writer w;
        const int position = w.floats(triangle_positions, "VEC3", true);
        const int texcoord = w.floats(triangle_uvs, "VEC2");
        w.add("images", data_image(png_a()));
        w.add("textures", R"({"source": 0})");
        w.add("textures", R"({"source": 0})");
        w.add("materials", R"({"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.25}})");
        w.add("materials", R"({"pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "metallicFactor": 0.75}})");
        const std::string attributes = std::format(R"({{"POSITION": {}, "TEXCOORD_0": {}}})", position, texcoord);
        w.add("meshes", R"({"primitives": [{"attributes": )" + attributes + R"(, "material": 0}, {"attributes": )" +
                            attributes + R"(, "material": 1}]})");
        const load_result result = load(r, "images.used-twice", one_node(w, folder / "used-twice.gltf"));
        if (const meshes::model* m = loaded(r, "images.used-twice", result))
        {
            const meshes::pbr_material* m0 = material_by_metallic(*m, 0.25);
            const meshes::pbr_material* m1 = material_by_metallic(*m, 0.75);
            const auto decoded =
                std::ranges::count_if(m->images, [](const meshes::image& i) { return !i.rgba.empty(); });
            const bool same = m0 && m1 && m0->base_color_image >= 0 && m0->base_color_image == m1->base_color_image;
            r.check("images.used-twice-decoded-once", same && decoded == 1,
                    std::format("base-color images {} and {}, {} decoded", m0 ? m0->base_color_image : -2,
                                m1 ? m1->base_color_image : -2, decoded),
                    "one image named by both, 1 decoded");
            check_image(r, "images.used-twice-bytes", m0 ? image_at(*m, m0->base_color_image) : nullptr, a);
        }
    }

    // A material nobody wears (materials[1], metallic 0.75), its base color an image file that isn't there: the file
    // loads, images are read only for materials a placed primitive wears; materials[1] keeps its place and factors,
    // without an image
    {
        gltf_writer w;
        w.add("images", R"({"uri": "nowhere-unworn.png"})");
        w.add("textures", R"({"source": 0})");
        w.add("materials", R"({"pbrMetallicRoughness": {"metallicFactor": 0.25}})");
        w.add("materials", R"({"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.75}})");
        add_mesh(w, {{.positions = triangle_positions, .material = 0}});
        const load_result result = load(r, "images.unworn-material", one_node(w, folder / "unworn-material.gltf"));
        if (const meshes::model* m = loaded(r, "images.unworn-material", result))
        {
            const bool ok = m->materials.size() == 3 && m->materials[1].metallic == 0.75 &&
                            m->materials[1].base_color_image == -1 && m->images.empty();
            r.check("images.unworn-material-kept", ok,
                    std::format("{} materials, materials[1]: {}, {} images", m->materials.size(),
                                m->materials.size() > 1 ? describe(m->materials[1]) : std::string("none"),
                                m->images.size()),
                    "3 materials, materials[1] metallic 0.75 with images -1, no image");
        }
    }
}

// --- Warnings: what the renderer ignores loads, with a warning ---------------------------------------------

void warning_checks(report& r, const std::filesystem::path& folder)
{
    using extra_check = std::function<void(report&, const std::string&, const meshes::model&)>;
    struct feature_case
    {
        std::string_view name;
        feature_function feature;
        extra_check extra; // the case's own check, if any
    };
    const auto material = [](std::string members) {
        return [members](gltf_writer&, textured_parts& p) { p.material_members = members; };
    };
    const auto material_extension = [](std::string extension, std::string body) {
        return [extension, body](gltf_writer& w, textured_parts& p) {
            w.extension(extension, false);
            p.material_members = std::format(R"(, "extensions": {{"{}": {}}})", extension, body);
        };
    };
    const std::vector<triangle> base = make_triangles(triangle_positions);
    // A texture that can't be read where it is worn leaves the factor alone: the triangle's material has the material's
    // own factors (metallic 0.25, roughness 0.5, set by the case) and no image
    constexpr std::string_view factors = R"(, "metallicFactor": 0.25, "roughnessFactor": 0.5)";
    const auto factor_alone = [](report& rr, const std::string& name, const meshes::pbr_material* found) {
        const bool ok = found && found->base_color == std::array<double, 3>{1, 1, 1} && found->metallic == 0.25 &&
                        found->roughness == 0.5 && found->base_color_image == -1 && found->metal_rough_image == -1;
        rr.check(name, ok, found ? describe(*found) : std::string("no such triangle"),
                 "base (1, 1, 1), metallic 0.25, roughness 0.5, images -1 -1");
    };
    std::array<float, 9> moved_positions = triangle_positions;
    for (std::size_t i = 0; i < moved_positions.size(); i += 3)
    {
        moved_positions[i] += 2;
    }
    const std::vector<feature_case> cases{
        {"normal-texture", material(R"(, "normalTexture": {"index": 0})"), {}},
        {"occlusion-texture", material(R"(, "occlusionTexture": {"index": 0})"), {}},
        {"emissive-texture", material(R"(, "emissiveTexture": {"index": 0})"), {}},
        {"emissive-factor", material(R"(, "emissiveFactor": [1, 0.5, 0.25])"), {}},
        // Required, and allowed: it loads; the emission it scales is ignored
        {"emissive-strength-required",
         [](gltf_writer& w, textured_parts& p) {
             w.extension("KHR_materials_emissive_strength", true);
             p.material_members = R"(, "emissiveFactor": [1, 1, 1], )"
                                  R"("extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 4}})";
         },
         {}},
        {"alpha-mask", material(R"(, "alphaMode": "MASK", "alphaCutoff": 0.5)"), {}},
        {"alpha-blend", material(R"(, "alphaMode": "BLEND")"), {}},
        {"color-0",
         [](gltf_writer& w, textured_parts& p) {
             constexpr std::array<float, 12> colors{1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1};
             p.attributes = std::format(R"(, "COLOR_0": {})", w.floats(colors, "VEC4"));
         },
         {}},
        // TEXCOORD_1, present and used by the base color: the factor alone
        {"texcoord-1",
         [factors](gltf_writer& w, textured_parts& p) {
             constexpr std::array<float, 6> uv1{0.5f, 0.5f, 1, 0.5f, 0.5f, 1};
             p.attributes = std::format(R"(, "TEXCOORD_1": {})", w.floats(uv1, "VEC2"));
             p.base_texture_members = R"(, "texCoord": 1)";
             p.pbr_members = factors;
         },
         [factor_alone](report& rr, const std::string& name, const meshes::model& m) {
             factor_alone(rr, name + "-factor-alone", first_material(m));
         }},
        // A texture without TEXCOORD_0: the factor alone
        {"texture-without-texcoord",
         [factors](gltf_writer&, textured_parts& p) {
             p.texcoord_0 = false;
             p.pbr_members = factors;
         },
         [factor_alone](report& rr, const std::string& name, const meshes::model& m) {
             factor_alone(rr, name + "-factor-alone", first_material(m));
         }},
        // The same beside a second primitive (the triangle moved by 2 in x) that has TEXCOORD_0 and wears the same
        // material: that one keeps the texture, PNG A, the first the factor alone
        {"texture-beside-texcoord",
         [factors, &moved_positions](gltf_writer& w, textured_parts& p) {
             p.texcoord_0 = false;
             p.pbr_members = factors;
             p.more_primitives = std::format(R"(, {{"attributes": {{"POSITION": {}, "TEXCOORD_0": {}}}, )"
                                             R"("material": 0}})",
                                             w.floats(moved_positions, "VEC3", true), p.texcoord);
         },
         [factor_alone, &base, &moved_positions](report& rr, const std::string& name, const meshes::model& m) {
             const std::optional<std::uint32_t> plain = material_at(m, base[0]);
             const std::optional<std::uint32_t> textured_one = material_at(m, make_triangles(moved_positions)[0]);
             factor_alone(rr, name + "-factor-alone", plain ? &m.materials[*plain] : nullptr);
             const meshes::pbr_material* t = textured_one ? &m.materials[*textured_one] : nullptr;
             check_image(rr, name + "-textured", t ? image_at(m, t->base_color_image) : nullptr,
                         expected_image(texels_a, true, meshes::wrap::repeat, meshes::wrap::repeat, false));
         }},
        // Used but not required: ignored with a warning (required: refused, refusals.texture-transform-required)
        {"texture-transform-used",
         [](gltf_writer& w, textured_parts& p) {
             w.extension("KHR_texture_transform", false);
             p.base_texture_members = R"(, "extensions": {"KHR_texture_transform": {"scale": [2, 2]}})";
         },
         {}},
        {"clearcoat", material_extension("KHR_materials_clearcoat", R"({"clearcoatFactor": 1})"), {}},
        {"transmission", material_extension("KHR_materials_transmission", R"({"transmissionFactor": 1})"), {}},
        {"unlit", material_extension("KHR_materials_unlit", "{}"), {}},
        {"specular-glossiness",
         material_extension("KHR_materials_pbrSpecularGlossiness",
                            R"({"diffuseFactor": [1, 1, 1, 1], "glossinessFactor": 0.5})"),
         {}},
        {"camera",
         [](gltf_writer& w, textured_parts& p) {
             w.top(R"("cameras": [{"type": "perspective", "perspective": {"yfov": 0.8, "znear": 0.1}}])");
             p.nodes.emplace_back(R"({"camera": 0})");
         },
         {}},
        {"light",
         [](gltf_writer& w, textured_parts& p) {
             w.extension("KHR_lights_punctual", false);
             w.top(R"("extensions": {"KHR_lights_punctual": {"lights": [{"type": "point", "intensity": 5}]}})");
             p.nodes.emplace_back(R"({"extensions": {"KHR_lights_punctual": {"light": 0}}})");
         },
         {}},
        {"animation",
         [](gltf_writer& w, textured_parts&) {
             constexpr std::array<float, 2> times{0, 1};
             constexpr std::array<float, 6> steps{0, 0, 0, 0, 1, 0};
             const int input = w.floats(times, "SCALAR", true);
             const int output = w.floats(steps, "VEC3");
             w.top(std::format(R"("animations": [{{"channels": [{{"sampler": 0, "target": {{"node": 0, )"
                               R"("path": "translation"}}}}], "samplers": [{{"input": {}, "output": {}}}]}}])",
                               input, output));
         },
         {}},
        // A skin (joint: node 1): the bind pose, warned
        {"skin",
         [](gltf_writer& w, textured_parts& p) {
             constexpr std::array<std::uint8_t, 12> joints{};
             constexpr std::array<float, 12> weights{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
             const int view = w.view(std::as_bytes(std::span(joints)), 0, 34962);
             const int j = w.add("accessors", std::format(R"({{"bufferView": {}, "componentType": 5121, )"
                                                          R"("count": 3, "type": "VEC4"}})",
                                                          view));
             p.attributes = std::format(R"(, "JOINTS_0": {}, "WEIGHTS_0": {})", j, w.floats(weights, "VEC4"));
             p.node_members = R"(, "skin": 0)";
             p.nodes.emplace_back(R"({"translation": [0, 0, 0]})");
             w.top(R"("skins": [{"joints": [1]}])");
         },
         {}},
        // A morph target of weight 1 that would lift the triangle by 5 in z: the bind pose, warned
        {"morph-target",
         [](gltf_writer& w, textured_parts& p) {
             constexpr std::array<float, 9> lift{0, 0, 5, 0, 0, 5, 0, 0, 5};
             p.primitive_members = std::format(R"(, "targets": [{{"POSITION": {}}}])", w.floats(lift, "VEC3", true));
             p.mesh_members = R"(, "weights": [1])";
         },
         [&base](report& rr, const std::string& name, const meshes::model& m) {
             compare(rr, name + "-bind-pose", m, base, 0);
         }},
        // A POINTS primitive (mode 0) beside the triangle: skipped, warned
        {"points-primitive",
         [](gltf_writer&, textured_parts& p) {
             p.more_primitives = std::format(R"(, {{"attributes": {{"POSITION": {}}}, "mode": 0}})", p.position);
         },
         [&base](report& rr, const std::string& name, const meshes::model& m) {
             compare(rr, name + "-triangles", m, base, 0);
         }},
        // A triangle primitive without POSITION (only NORMAL) beside the triangle: skipped, warned (glTF 2.0 leaves
        // its rendering undefined; the file's other primitives stay)
        {"no-position",
         [](gltf_writer& w, textured_parts& p) {
             constexpr std::array<float, 9> up{0, 0, 1, 0, 0, 1, 0, 0, 1};
             p.more_primitives = std::format(R"(, {{"attributes": {{"NORMAL": {}}}}})", w.floats(up, "VEC3"));
         },
         [&base](report& rr, const std::string& name, const meshes::model& m) {
             compare(rr, name + "-triangles", m, base, 0);
         }},
        // Vertex colors on five more primitives (the same triangle): one warning line counting 5 and naming the first
        // four places, then "…"
        {"places-named",
         [](gltf_writer& w, textured_parts& p) {
             constexpr std::array<float, 12> colors{1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1};
             const int color = w.floats(colors, "VEC4");
             for (int k = 0; k < 5; ++k)
             {
                 p.more_primitives +=
                     std::format(R"(, {{"attributes": {{"POSITION": {}, "COLOR_0": {}}}}})", p.position, color);
             }
         },
         [](report& rr, const std::string& name, const meshes::model& m) {
             const auto line =
                 std::ranges::find_if(m.warnings, [](const std::string& w) { return w.contains("COLOR"); });
             const bool ok = line != m.warnings.end() && line->contains("(5)") && line->contains("primitives[4]") &&
                             !line->contains("primitives[5]") && line->ends_with(", \xE2\x80\xA6");
             rr.check(name + "-line", ok, line != m.warnings.end() ? std::format("'{}'", *line) : "no such line",
                      "the count (5), meshes[0].primitives[1] to [4], then an ellipsis");
         }},
    };
    for (const feature_case& c : cases)
    {
        const std::string name = std::format("warnings.{}", c.name);
        const load_result result = load(r, name, textured(c.feature).write(folder / std::format("{}.gltf", c.name)));
        if (const meshes::model* m = loaded(r, name, result))
        {
            r.check(name + "-warned", !m->warnings.empty(), describe(m->warnings), "at least one warning");
            if (c.extra)
            {
                c.extra(r, name, *m);
            }
        }
    }
}

// --- Meshes made in code --------------------------------------------------------------------------------------------

void icosphere_checks(report& r)
{
    // Radius 1, V = 10·4^k + 2, T = 20·4^k, closed, every face outward, normals = positions when smooth, none when
    // flat; generated meshes wear material 0
    for (int level = 0; level <= 5; ++level)
    {
        for (const bool smooth : {false, true})
        {
            const meshes::mesh m = meshes::icosphere(level, smooth);
            const std::string name = std::format("icosphere.level-{}-{}", level, smooth ? "smooth" : "flat");
            if (const std::string problem = mesh_problem(m, 0); !problem.empty())
            {
                r.check(name, false, problem, "a well-formed mesh");
                continue;
            }
            const std::size_t power = std::size_t{1} << (2 * level);
            const std::size_t expected_v = 10 * power + 2, expected_t = 20 * power;
            const std::size_t vertices = m.positions.size() / 3, triangles = triangle_count(m);
            double radius = 0, normal_error = 0;
            for (std::size_t i = 0; i < vertices; ++i)
            {
                const vec p = vertex(m.positions, i);
                radius = worse(radius, std::abs(length(p) - 1));
                if (!m.normals.empty())
                {
                    normal_error = worse(normal_error, gap(vertex(m.normals, i), p));
                }
            }
            const bool normals_ok = smooth ? !m.normals.empty() && normal_error <= 1e-6 : m.normals.empty();
            const std::size_t open = open_edges(m);
            const std::size_t inward = inward_faces(m, [](const vec&) { return vec{0, 0, 0}; });
            const bool zeros = std::ranges::all_of(m.material_of, [](std::uint32_t k) { return k == 0; });
            r.check(name,
                    vertices == expected_v && triangles == expected_t && radius <= 1e-6 && normals_ok && open == 0 &&
                        inward == 0 && zeros,
                    std::format("V {}, T {}, max ||p| - 1| {:.2g}, {} open edges, {} inward faces, {}, material_of {}",
                                vertices, triangles, radius, open, inward,
                                m.normals.empty()
                                    ? std::string("no normals")
                                    : std::format("normals at most {:.2g} from the positions", normal_error),
                                zeros ? "all 0" : "not all 0"),
                    std::format("V {}, T {}, radius 1 within 1e-6, 0 open edges, 0 inward faces, {}, material_of all 0",
                                expected_v, expected_t, smooth ? "normals = positions" : "no normals"));
            if (level == 5 && smooth)
            {
                r.exact("icosphere.level-5-area", area(m), 4 * std::numbers::pi, 0.003 * 4 * std::numbers::pi);
            }
        }
    }
}

// A torus: (around + 1)·(across + 1) vertices (both seams duplicated), 2·around·across triangles, closed once the seams
// are welded, every face pointing away from the core circle; normals, if any, unit and outward
void torus_check(report& r, double major, double minor, int around, int across, bool with_area)
{
    const meshes::mesh m = meshes::torus(major, minor, around, across);
    const std::string name = std::format("torus.{}x{}", around, across);
    if (const std::string problem = mesh_problem(m, 0); !problem.empty())
    {
        r.check(name, false, problem, "a well-formed mesh");
        return;
    }
    const std::size_t expected_v = static_cast<std::size_t>((around + 1) * (across + 1));
    const std::size_t expected_t = static_cast<std::size_t>(2 * around * across);
    const std::size_t vertices = m.positions.size() / 3;
    // The axis: the coordinate of the smallest extent (2·minor against 2·(major + minor)); the centre: the box's
    vec lo{no_match, no_match, no_match}, hi{-no_match, -no_match, -no_match};
    for (std::size_t i = 0; i < vertices; ++i)
    {
        const vec p = vertex(m.positions, i);
        for (std::size_t k = 0; k < 3; ++k)
        {
            lo[k] = std::min(lo[k], p[k]);
            hi[k] = std::max(hi[k], p[k]);
        }
    }
    const vec extent = sub(hi, lo);
    const std::size_t axis = extent[0] <= extent[1] && extent[0] <= extent[2] ? 0 : extent[1] <= extent[2] ? 1 : 2;
    const vec centre = mul(0.5, add(lo, hi));
    const auto core = [&centre, axis, major](const vec& q) {
        vec d = sub(q, centre);
        d[axis] = 0;
        return add(centre, mul(major / length(d), d));
    };
    const std::size_t inward = inward_faces(m, core);
    std::size_t bad_normals = 0;
    for (std::size_t i = 0; i < vertices && !m.normals.empty(); ++i)
    {
        const vec n = vertex(m.normals, i), p = vertex(m.positions, i);
        bad_normals += std::abs(length(n) - 1) <= 1e-5 && dot(n, sub(p, core(p))) > 0 ? 0 : 1;
    }
    const std::size_t open = open_edges(m);
    const bool zeros = std::ranges::all_of(m.material_of, [](std::uint32_t k) { return k == 0; });
    r.check(name,
            vertices == expected_v && triangle_count(m) == expected_t && open == 0 && inward == 0 &&
                bad_normals == 0 && zeros,
            std::format("V {}, T {}, {} open edges, {} inward faces, {}, material_of {}", vertices, triangle_count(m),
                        open, inward,
                        m.normals.empty() ? std::string("no normals")
                                          : std::format("{} normals not unit or not outward", bad_normals),
                        zeros ? "all 0" : "not all 0"),
            std::format("V {}, T {}, 0 open edges, 0 inward faces, normals (if any) unit and outward, "
                        "material_of all 0", expected_v, expected_t));
    if (with_area)
    {
        const double exact_area = 4 * std::numbers::pi * std::numbers::pi * major * minor;
        r.exact(name + "-area", area(m), exact_area, 0.005 * exact_area);
    }
}

void torus_checks(report& r)
{
    torus_check(r, 1, 0.25, 64, 32, true);
    torus_check(r, 2, 0.5, 7, 5, false); // odd counts: the seams' bookkeeping
}

// triangulate_quads: corners q, q+u, q+u+v, q+v; triangles (0, 1, 2), (0, 2, 3); front = u × v; UVs the quad's
// (α, β) of p = q + α·u + β·v; two triangles of equal area |u × v|/2; material_of the quad's
void quad_checks(report& r)
{
    // A parallelogram (u and v not perpendicular) and an axis-aligned wall, wearing materials 3 and 7
    const std::array<std::array<double, 9>, 2> quads{{{1, 2, 3, 2, 0, 1, 0.5, 3, 0}, {-1, 0, 0, 0, 0, 4, 0, 2, 0}}};
    constexpr std::array<std::uint32_t, 2> materials{3, 7};
    const meshes::mesh m = meshes::triangulate_quads(quads, materials);
    if (const std::string problem = mesh_problem(m, 0); !problem.empty())
    {
        r.check("quads.mesh", false, problem, "a well-formed mesh");
        return;
    }
    r.equal("quads.triangles", triangle_count(m), 4);
    if (triangle_count(m) != 4)
    {
        return;
    }
    r.check("quads.material-of", m.material_of == std::vector<std::uint32_t>{3, 3, 7, 7},
            std::format("{} {} {} {}", m.material_of[0], m.material_of[1], m.material_of[2], m.material_of[3]),
            "3 3 7 7");
    std::vector<triangle> expected;
    for (const std::array<double, 9>& quad : quads)
    {
        const vec q{quad[0], quad[1], quad[2]}, u{quad[3], quad[4], quad[5]}, v{quad[6], quad[7], quad[8]};
        expected.push_back({q, add(q, u), add(add(q, u), v)});
        expected.push_back({q, add(add(q, u), v), add(q, v)});
    }
    const double d = deviation(mesh_geometry(m), expected);
    r.check("quads.corners", d <= 1e-6, std::format("corners at most {:.3g} from the expected", d),
            "q, q+u, q+u+v and q, q+u+v, q+v within 1e-6");
    std::size_t backward = 0, unequal = 0, wrong_uv = 0, wrong_normal = 0;
    const bool has_uvs = m.uvs.size() == 2 * (m.positions.size() / 3);
    for (std::size_t t = 0; t < 4; ++t)
    {
        const std::array<double, 9>& quad = quads[t / 2];
        const vec q{quad[0], quad[1], quad[2]}, u{quad[3], quad[4], quad[5]}, v{quad[6], quad[7], quad[8]};
        const vec n = cross(u, v);
        const vec face = face_normal(corners(m, t));
        backward += dot(face, n) > 0 && length(cross(unit(face), unit(n))) < 1e-6 ? 0 : 1;
        unequal += std::abs(length(face) - length(n)) <= 1e-6 * length(n) ? 0 : 1;
        for (std::size_t k = 0; k < 3; ++k)
        {
            const std::uint32_t i = m.indices[3 * t + k];
            const vec p = sub(vertex(m.positions, i), q);
            const double alpha = dot(p, cross(v, n)) / dot(u, cross(v, n));
            const double beta = dot(p, cross(n, u)) / dot(v, cross(n, u));
            const bool uv_ok = has_uvs && std::abs(m.uvs[2 * i] - alpha) <= 1e-6 &&
                               std::abs(m.uvs[2 * i + 1] - beta) <= 1e-6;
            wrong_uv += uv_ok ? 0 : 1;
            wrong_normal += m.normals.empty() || gap(vertex(m.normals, i), unit(n)) <= 1e-6 ? 0 : 1;
        }
    }
    r.check("quads.front", backward == 0, std::format("{} of 4 faces not along u x v", backward),
            "every a x b along u x v");
    r.check("quads.areas", unequal == 0, std::format("{} of 4 areas off |u x v|/2", unequal),
            "each |u x v|/2 within 1e-6 (relative)");
    r.check("quads.uvs", wrong_uv == 0,
            has_uvs ? std::format("{} of 12 corners off their (alpha, beta)", wrong_uv) : std::string("no UVs"),
            "each corner's (alpha, beta) of p = q + alpha u + beta v");
    if (m.normals.empty())
    {
        r.info("quads.normals", "none (flat)");
    }
    else
    {
        r.check("quads.normals", wrong_normal == 0, std::format("{} of 12 corners' normals off", wrong_normal),
                "unit(u x v) at every corner");
    }
}

// unit_placement: a uniform scale that gives the box's bounding sphere (half its diagonal) radius 1, the box's centre
// over the origin (x = z = 0) and its lowest point on y = 0; to 1e-12
void placement_checks(report& r)
{
    constexpr std::array<double, 3> lower{-3, 2, 1}, upper{5, 6, 2}; // half diagonal |(4, 2, 0.5)| = 4.5
    const matrix m = meshes::unit_placement(lower, upper);
    const double s = m[0];
    double other = 0;
    for (const double e : {m[1], m[2], m[4], m[6], m[8], m[9], m[5] - s, m[10] - s})
    {
        other = worse(other, std::abs(e));
    }
    r.check("placement.unit-uniform", s > 0 && other <= 1e-12,
            std::format("diagonal {} {} {}, the largest other difference {:.3g}", m[0], m[5], m[10], other),
            "a uniform positive scale, no rotation");
    const vec lo = point(m, {lower[0], lower[1], lower[2]}), hi = point(m, {upper[0], upper[1], upper[2]});
    r.exact("placement.unit-radius", gap(hi, lo) / 2, 1, 1e-12);
    r.exact("placement.unit-lowest-y", std::min(lo[1], hi[1]), 0, 1e-12);
    const vec middle = mul(0.5, add(lo, hi));
    r.exact("placement.unit-centre-x", middle[0], 0, 1e-12);
    r.exact("placement.unit-centre-z", middle[2], 0, 1e-12);
}

// bake: world vertices M·p, normals unit(M⁻ᵀ·n); under a rotation the index order unchanged; under a mirror (det M < 0)
// indices 1 and 2 of every triangle swapped, so each face's a × b points outward and agrees with its vertex normals
void bake_checks(report& r)
{
    // The input: a smooth icosphere given texture coordinates (u = x, v = y of each vertex) and materials 0 to 4 in
    // turn, so that "carried unchanged" compares something
    meshes::mesh sphere = meshes::icosphere(2, true);
    for (std::size_t i = 0; i < sphere.positions.size() / 3; ++i)
    {
        sphere.uvs.insert(sphere.uvs.end(), {sphere.positions[3 * i], sphere.positions[3 * i + 1]});
    }
    for (std::size_t t = 0; t < sphere.material_of.size(); ++t)
    {
        sphere.material_of[t] = static_cast<std::uint32_t>(t % 5);
    }
    if (const std::string problem = mesh_problem(sphere, 0); !problem.empty() || sphere.normals.empty())
    {
        r.check("bake.input", false, problem.empty() ? std::string("no normals") : problem,
                "icosphere(2, true): a well-formed mesh with normals");
        return;
    }
    const std::size_t vertices = sphere.positions.size() / 3;
    // The world mesh's positions and normals against M·p and unit(M⁻ᵀ·n), vertex for vertex (bake keeps their order)
    const auto vertex_errors = [&sphere, vertices](const meshes::mesh& baked, const matrix& m) {
        if (baked.positions.size() != sphere.positions.size() || baked.normals.size() != sphere.normals.size())
        {
            return std::pair{no_match, no_match};
        }
        double p_error = 0, n_error = 0;
        for (std::size_t i = 0; i < vertices; ++i)
        {
            p_error = worse(p_error, gap(vertex(baked.positions, i), point(m, vertex(sphere.positions, i))));
            n_error =
                worse(n_error, gap(vertex(baked.normals, i), unit(normal_to_world(m, vertex(sphere.normals, i)))));
        }
        return std::pair{p_error, n_error};
    };

    // A turn of 40° about (1, 2, 3)/√14, then a translation
    {
        const matrix turn = rotation({1, 2, 3}, 40, {0.5, -1, 2});
        const meshes::mesh baked = meshes::bake(sphere, turn);
        if (const std::string problem = mesh_problem(baked, 0); !problem.empty())
        {
            r.check("bake.rotation", false, problem, "a well-formed mesh");
        }
        else
        {
            const bool same = baked.indices == sphere.indices;
            r.check("bake.rotation-indices", same, same ? "the input's" : "changed", "the input's, in its order");
            const auto [p_error, n_error] = vertex_errors(baked, turn);
            r.exact("bake.rotation-positions", p_error, 0, 1e-5);
            r.exact("bake.rotation-normals", n_error, 0, 1e-5);
            const bool carried =
                !baked.uvs.empty() && baked.uvs == sphere.uvs && baked.material_of == sphere.material_of;
            r.check("bake.rotation-carried", carried, carried ? "unchanged" : "changed",
                    "UVs and material_of unchanged");
        }
    }

    // A mirror: diag(-1, 1, 1)·diag(1.5, 0.5, 2), moved to c = (1, 2, 3): an ellipsoid around c
    {
        constexpr matrix mirror{-1.5, 0, 0, 1, 0, 0.5, 0, 2, 0, 0, 2, 3};
        const vec centre{1, 2, 3};
        const meshes::mesh baked = meshes::bake(sphere, mirror);
        if (const std::string problem = mesh_problem(baked, 0); !problem.empty())
        {
            r.check("bake.mirror", false, problem, "a well-formed mesh");
        }
        else
        {
            std::size_t unswapped = 0;
            const bool same_size = baked.indices.size() == sphere.indices.size();
            for (std::size_t t = 0; same_size && t < triangle_count(sphere); ++t)
            {
                const std::uint32_t* s = &sphere.indices[3 * t];
                const std::uint32_t* b = &baked.indices[3 * t];
                unswapped += b[0] == s[0] && b[1] == s[2] && b[2] == s[1] ? 0 : 1;
            }
            r.check("bake.mirror-indices", same_size && unswapped == 0,
                    same_size ? std::format("{} of {} triangles not swapped", unswapped, triangle_count(sphere))
                              : std::string("a different index count"),
                    "every triangle's indices 1 and 2 swapped");
            const std::size_t inward = inward_faces(baked, [&centre](const vec&) { return centre; });
            r.check("bake.mirror-outside", inward == 0,
                    std::format("{} of {} faces with a x b inward", inward, triangle_count(baked)), "none inward");
            std::size_t disagree = 0;
            for (std::size_t t = 0; t < triangle_count(baked) && !baked.normals.empty(); ++t)
            {
                const vec face = face_normal(corners(baked, t));
                for (std::size_t k = 0; k < 3; ++k)
                {
                    disagree += dot(vertex(baked.normals, baked.indices[3 * t + k]), face) > 0 ? 0 : 1;
                }
            }
            r.check("bake.mirror-normals-agree", disagree == 0 && !baked.normals.empty(),
                    baked.normals.empty() ? std::string("no normals")
                                          : std::format("{} corners whose normal opposes its face", disagree),
                    "every vertex normal on its face's side");
            const auto [p_error, n_error] = vertex_errors(baked, mirror);
            r.exact("bake.mirror-positions", p_error, 0, 1e-5);
            r.exact("bake.mirror-normals", n_error, 0, 1e-5);
            // Only the corners' labelling changes: each vertex keeps its UVs, each triangle its material
            const bool carried =
                !baked.uvs.empty() && baked.uvs == sphere.uvs && baked.material_of == sphere.material_of;
            r.check("bake.mirror-carried", carried, carried ? "unchanged" : "changed", "UVs and material_of unchanged");
        }
        const bool flat = meshes::bake(meshes::icosphere(1, false), mirror).normals.empty();
        r.check("bake.flat-stays-flat", flat, flat ? "no normals" : "normals", "no normals (a flat mesh stays flat)");
    }
}

// --- The committed models (Step 93) ---------------------------------------------------------------------------------
// TODO(Step 93): assets/models/marble_bust_01/ and assets/models/brass_vase_01/ (Poly Haven, CC0, committed as
// published): each .gltf loaded with the default limits; its triangle count (17,456 and 21,440), image count, world
// bounds and warnings (the unused normal-map JPEGs) as the files say; and every file's SHA-256 (a FIPS 180-4
// implementation here, with its "abc" vector as a self-check) against the table in docs/architecture.md's credits, so
// that a changed download is noticed. The files are found as GpuTests finds its assets: from the current folder or the
// executable's, upward.
void committed_models(report& r)
{
    r.info("models", "the committed models are checked from Step 93 on (TODO)");
}

// --- main -----------------------------------------------------------------------------------------------------------

int main()
{
    const std::filesystem::path folder =
        std::filesystem::temp_directory_path() /
        std::format("mesh-tests-{}", std::chrono::steady_clock::now().time_since_epoch().count());

    // A watchdog: a loader that loops (a node cycle walked without its validator) must not hold the computer. The
    // whole run takes well under a second. It removes the temporary folder too, as the normal end does.
    std::thread([folder] {
        std::this_thread::sleep_for(std::chrono::seconds(20));
        std::println("Error: MeshTests still runs after 20 s (a loader that loops?)");
        std::fflush(nullptr);
        std::error_code ignored;
        std::filesystem::remove_all(folder, ignored);
        std::_Exit(2);
    }).detach();
    report r;
    int code = 0;
    try
    {
        std::error_code failed;
        std::filesystem::create_directories(folder, failed);
        if (failed)
        {
            throw setup_error{std::format("can't create the temporary folder: {}", failed.message())};
        }
        self_checks(r);
        group(r, "gltf", [&] { gltf_geometry(r, folder); });
        group(r, "gltf", [&] { gltf_transforms(r, folder); });
        group(r, "gltf", [&] { gltf_dropped(r, folder); });
        group(r, "refusals", [&] { gltf_refusals(r, folder); });
        group(r, "limits", [&] { limit_checks(r, folder); });
        group(r, "materials", [&] { material_checks(r, folder); });
        group(r, "images", [&] { image_checks(r, folder); });
        group(r, "warnings", [&] { warning_checks(r, folder); });
        group(r, "icosphere", [&] { icosphere_checks(r); });
        group(r, "torus", [&] { torus_checks(r); });
        group(r, "quads", [&] { quad_checks(r); });
        group(r, "placement", [&] { placement_checks(r); });
        group(r, "bake", [&] { bake_checks(r); });
        committed_models(r);
        r.error_summary();
    }
    catch (const setup_error& error)
    {
        std::println("Error: {}", error.message);
        code = 2;
    }
    catch (const std::exception& error)
    {
        std::println("Error: {}", error.what());
        code = 2;
    }
    std::error_code ignored;
    std::filesystem::remove_all(folder, ignored);
    if (code != 0)
    {
        return code;
    }
    std::println("{}", r.failures() == 0
                           ? std::format("MeshTests: PASS ({} checks)", r.checks())
                           : std::format("MeshTests: FAIL ({} of {} checks)", r.failures(), r.checks()));
    return r.failures() == 0 ? 0 : 1;
}
