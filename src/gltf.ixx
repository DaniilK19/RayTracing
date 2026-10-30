// Module gltf: a glTF 2.0 file (.glb, or .gltf with its .bin and images) read with cgltf (v1.15, MIT, compiled here)
// and stb_image (a private STB_IMAGE_STATIC copy). What the renderer can't show is a warning; what it can't read is an
// error naming the file and the glTF object.
module;

// The global module fragment holds the two C libraries, compiled into this file alone (as image.ixx does with stb): a
// file that imports gltf sees neither. cgltf (external/cgltf, see cmake/meshes.cmake) parses the JSON or a .glb's
// chunks, resolves the references between glTF objects and checks them (cgltf_validate); it decodes no image, and
// reads the .bin files only through the two functions given to it below.
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

// stb_image a second time, for the materials' images. The image module compiles its own copy with external linkage, so
// this one is STB_IMAGE_STATIC: its functions have internal linkage, and the linker never meets two definitions of one
// name. glTF 2.0's images are PNG or JPEG ("Images" in the specification), so stb's other decoders are left out; files
// are read with std::ifstream, which takes UTF-8 paths through std::filesystem.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include <stb_image.h>

export module gltf;

import std;
import meshes;

// The model in the file's default scene: its meshes, each read once however many nodes place it; every placement's
// world transform; the materials (glTF's, then the default); the images the base-color and metallic-roughness
// textures use; the world bounds; and the warnings. Refused (an error, nothing loaded): a file over a limit, compressed
// meshes or textures, another required extension the renderer can't honor, what cgltf_validate refuses, and data that
// can't be read. A path is UTF-8 safe: std::filesystem throughout, never a narrow fopen.
export std::expected<meshes::model, std::string> load_gltf(const std::filesystem::path& file,
                                                           const meshes::limits& limits = {});

namespace
{
using matrix = std::array<double, 12>; // row-major 3×4, as meshes::placement

constexpr matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

// a·b for affine maps (A, a)(B, b) = (AB, Ab + a): b first, then a
matrix compose(const matrix& a, const matrix& b)
{
    matrix c{};
    for (int r = 0; r < 3; ++r)
    {
        for (int k = 0; k < 4; ++k)
        {
            double sum = k == 3 ? a[r * 4 + 3] : 0.0;
            for (int j = 0; j < 3; ++j)
            {
                sum += a[r * 4 + j] * b[j * 4 + k];
            }
            c[r * 4 + k] = sum;
        }
    }
    return c;
}

double determinant(const matrix& m)
{
    return m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
           m[2] * (m[4] * m[9] - m[5] * m[8]);
}

// On Windows std::filesystem::path holds UTF-16 and path.string() converts it to the process's code page; u8string()
// is always UTF-8 (cli.ixx's to_utf8, which this library can't import: it doesn't link the CPU renderer)
std::string to_utf8(const std::filesystem::path& path)
{
    const std::u8string text = path.u8string();
    return std::string(text.begin(), text.end());
}

std::filesystem::path from_utf8(std::string_view text)
{
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
}

// UTF-8 as RFC 3629 defines it: each character in its shortest form, no surrogates (U+D800 to U+DFFF), nothing past
// U+10FFFF. std::filesystem converts a path's UTF-8 to Windows' UTF-16 and throws on anything else, and a URI's percent
// escapes can make any byte ("tex%FF.png"), so a URI is checked before it becomes a path.
bool is_utf8(std::string_view text)
{
    constexpr std::array<char32_t, 5> shortest{0, 0, 0x80, 0x800, 0x10000}; // the least code point of each length
    for (std::size_t i = 0; i < text.size();)
    {
        const auto byte = [&text](std::size_t k) { return static_cast<unsigned char>(text[k]); };
        const unsigned char lead = byte(i);
        const std::size_t length = lead < 0x80             ? 1
                                   : (lead & 0xE0) == 0xC0 ? 2
                                   : (lead & 0xF0) == 0xE0 ? 3
                                   : (lead & 0xF8) == 0xF0 ? 4
                                                           : 0;
        if (length == 0 || length > text.size() - i)
        {
            return false;
        }
        auto c = static_cast<char32_t>(length == 1 ? lead : lead & (0x7Fu >> length));
        for (std::size_t k = 1; k < length; ++k)
        {
            if ((byte(i + k) & 0xC0) != 0x80)
            {
                return false;
            }
            c = static_cast<char32_t>(c << 6 | (byte(i + k) & 0x3Fu));
        }
        if (c < shortest[length] || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF))
        {
            return false;
        }
        i += length;
    }
    return true;
}

// A copy of a URI with its percent escapes decoded ("%20" → " "), as cgltf decodes a buffer's before opening its file
std::string decoded_uri(std::string_view uri)
{
    std::string name(uri);
    name.resize(cgltf_decode_uri(name.data()));
    return name;
}

std::string_view describe(cgltf_result result)
{
    switch (result)
    {
    case cgltf_result_data_too_short:
        return "data too short";
    case cgltf_result_unknown_format:
        return "unknown format";
    case cgltf_result_invalid_json:
        return "invalid JSON";
    case cgltf_result_invalid_gltf:
        return "invalid glTF";
    case cgltf_result_file_not_found:
        return "file not found";
    case cgltf_result_io_error:
        return "read error";
    case cgltf_result_out_of_memory:
        return "out of memory";
    case cgltf_result_legacy_gltf:
        return "glTF 1.0";
    default:
        return "error";
    }
}

std::string_view type_name(cgltf_type type)
{
    constexpr std::array<std::string_view, 8> names{"no type", "SCALAR", "VEC2", "VEC3",
                                                    "VEC4",    "MAT2",   "MAT3", "MAT4"};
    const auto i = static_cast<std::size_t>(type);
    return i < names.size() ? names[i] : names[0];
}

// The bytes a base64 text decodes to: 6 bits per character, the '=' padding excluded
std::size_t base64_bytes(std::string_view text)
{
    while (text.ends_with('='))
    {
        text.remove_suffix(1);
    }
    return text.size() * 6 / 8;
}

double unit_interval(float value) // NaN → 0
{
    return value > 1 ? 1.0 : value > 0 ? static_cast<double>(value) : 0.0;
}

bool is_png_or_jpeg(std::string_view mime_type)
{
    return mime_type == "image/png" || mime_type == "image/jpeg";
}

struct free_deleter
{
    void operator()(void* memory) const noexcept { std::free(memory); }
};

struct data_deleter
{
    void operator()(cgltf_data* data) const noexcept { cgltf_free(data); }
};

// --- cgltf's own memory -------------------------------------------------------------------------------------------
// cgltf allocates what it parses through these two functions: the JSON's tokens (32 bytes each, freed once parsed),
// a structure for every glTF object (a node a few hundred bytes, a material over a kilobyte), all zeroed and so
// touched, and the data: buffers it decodes. A crafted 256 MiB JSON of "{}," alone would ask for some 23 GB of nodes
// (264 bytes each) before any limit of the loader runs. The budget: twice the file limit plus 1 MiB (513 MiB by
// default). A data: URI decodes to 3/4 of its text, and a real model's JSON, kilobytes to a few megabytes, parses into
// a few times its size; past the budget cgltf sees an allocation fail, and the file is refused as needing too much.
struct memory_budget
{
    std::size_t limit = 0;
    std::size_t used = 0;
    bool exceeded = false;
};

// Each block starts with its size, for budget_free to give back, in a header of 16 bytes: the address after it keeps
// the 16-byte alignment malloc gives on x64
constexpr std::size_t header_bytes = 16;

void* budget_alloc(void* user, cgltf_size size)
{
    auto& budget = *static_cast<memory_budget*>(user);
    if (size > budget.limit - budget.used || size > std::numeric_limits<std::size_t>::max() - header_bytes)
    {
        budget.exceeded = true;
        return nullptr;
    }
    auto* const block = static_cast<std::byte*>(std::malloc(size + header_bytes));
    if (!block)
    {
        return nullptr;
    }
    std::memcpy(block, &size, sizeof size);
    budget.used += size;
    return block + header_bytes;
}

void budget_free(void* user, void* memory)
{
    if (!memory) // cgltf_free frees every optional member, most of them null
    {
        return;
    }
    std::byte* const block = static_cast<std::byte*>(memory) - header_bytes;
    cgltf_size size = 0;
    std::memcpy(&size, block, sizeof size);
    static_cast<memory_budget*>(user)->used -= size;
    std::free(block);
}

struct budget_deleter
{
    memory_budget* budget = nullptr;
    void operator()(void* memory) const noexcept { budget_free(budget, memory); }
};

// --- Files for cgltf ----------------------------------------------------------------------------------------------
// cgltf_load_buffers reads a .bin file named by a buffer's URI through these two functions instead of fopen, which
// would take the name in the process's code page: here it is UTF-8, and the file's size is checked against the limit
// before anything is allocated, as is the total of every buffer file read: a small .gltf can name one large .bin a
// thousand times. cgltf only learns success or failure; the reason waits in `problem` for the message.
struct file_reader
{
    std::uintmax_t limit = 0;
    std::uintmax_t total = 0; // the bytes read so far, every buffer file together
    std::string problem;
};

cgltf_result read_buffer_file(const cgltf_memory_options*, const cgltf_file_options* options, const char* path,
                              cgltf_size* size, void** data)
{
    auto& reader = *static_cast<file_reader*>(options->user_data);
    try
    {
        const std::filesystem::path file = from_utf8(path);
        std::error_code error;
        const std::uintmax_t bytes = std::filesystem::file_size(file, error);
        if (error)
        {
            reader.problem = std::format("there is no file '{}'", path);
            return cgltf_result_file_not_found;
        }
        if (bytes > reader.limit)
        {
            reader.problem = std::format("'{}' is {} bytes, more than the limit of {}", path, bytes, reader.limit);
            return cgltf_result_io_error;
        }
        // cgltf asks for the buffer's byteLength (0: the whole file); the file may be longer, never shorter
        const std::uintmax_t wanted = size && *size ? *size : bytes;
        if (wanted > bytes)
        {
            reader.problem = std::format("'{}' holds {} bytes, fewer than the buffer's byteLength of {}", path, bytes,
                                         wanted);
            return cgltf_result_data_too_short;
        }
        if (wanted > reader.limit - reader.total)
        {
            reader.problem = std::format("reading its {} bytes from '{}' would bring the buffer files read to more "
                                         "than the limit of {} bytes in all",
                                         wanted, path, reader.limit);
            return cgltf_result_io_error;
        }
        reader.total += wanted;
        const auto allocated = static_cast<std::size_t>(std::max<std::uintmax_t>(wanted, 1));
        std::unique_ptr<void, free_deleter> memory(std::malloc(allocated));
        if (!memory)
        {
            return cgltf_result_out_of_memory;
        }
        std::ifstream in(file, std::ios::binary);
        if (!in.read(static_cast<char*>(memory.get()), static_cast<std::streamsize>(wanted)))
        {
            reader.problem = std::format("could not read '{}'", path);
            return cgltf_result_io_error;
        }
        if (size)
        {
            *size = static_cast<cgltf_size>(wanted);
        }
        *data = memory.release();
        return cgltf_result_success;
    }
    // Nothing may cross back into C. The URI was checked for UTF-8 before (load_buffers), so a path that still can't
    // be converted, or a message that can't be formatted, ends here, said as well as memory allows.
    catch (const std::bad_alloc&)
    {
        return cgltf_result_out_of_memory;
    }
    catch (const std::exception& error)
    {
        try
        {
            reader.problem = std::format("could not read '{}' ({})", path, error.what());
        }
        catch (const std::exception&)
        {
            reader.problem.clear();
        }
        return cgltf_result_io_error;
    }
}

void release_buffer_file(const cgltf_memory_options*, const cgltf_file_options*, void* data)
{
    std::free(data);
}

// --- Warnings -----------------------------------------------------------------------------------------------------
// One line per kind of thing ignored or dropped, with how many and where (the first few glTF objects):
// "bust.gltf: normal textures ignored, no normal maps (1): materials[0]"
class warning_list
{
public:
    // A few dozen kinds at most, and of each only the places printed are kept: an add costs the same however many came
    // before (a model of 200,000 primitives with vertex colors warns 200,000 times)
    void add(std::string_view what, std::string_view where = {}, std::size_t count = 1)
    {
        auto kind = std::ranges::find(kinds_, what, &warning_kind::what);
        if (kind == kinds_.end())
        {
            kinds_.push_back({std::string(what)});
            kind = std::prev(kinds_.end());
        }
        kind->count += count;
        if (!where.empty() && !std::ranges::contains(kind->where, where))
        {
            if (kind->where.size() < named)
            {
                kind->where.emplace_back(where);
            }
            else
            {
                kind->more = true;
            }
        }
    }

    std::vector<std::string> lines(std::string_view file) const
    {
        std::vector<std::string> out;
        for (const warning_kind& kind : kinds_)
        {
            std::string line = std::format("{}: {}", file, kind.what);
            if (kind.count > 0)
            {
                line += std::format(" ({})", kind.count);
            }
            for (std::size_t i = 0; i < kind.where.size(); ++i)
            {
                line += (i == 0 ? ": " : ", ") + kind.where[i];
            }
            if (kind.more)
            {
                line += ", …";
            }
            out.push_back(std::move(line));
        }
        return out;
    }

private:
    static constexpr std::size_t named = 4; // the places a line names

    struct warning_kind
    {
        std::string what;
        std::size_t count = 0;
        std::vector<std::string> where; // the first `named` distinct places
        bool more = false;              // and others after them
    };
    std::vector<warning_kind> kinds_;
};

// --- The loader -----------------------------------------------------------------------------------------------------
class loader
{
public:
    loader(const std::filesystem::path& file, const meshes::limits& limits)
        : file_(file), name_(to_utf8(file)), limits_(limits)
    {
        reader_.limit = limits.file_bytes;
        // cgltf's budget: twice the file limit plus 1 MiB (no budget at all for a file limit that large)
        constexpr std::size_t most = std::numeric_limits<std::size_t>::max();
        constexpr std::size_t mebibyte = 1 << 20;
        const bool huge = limits.file_bytes > (most - mebibyte) / 2;
        budget_.limit = huge ? most : static_cast<std::size_t>(2 * limits.file_bytes + mebibyte);
        options_.memory.alloc_func = &budget_alloc;
        options_.memory.free_func = &budget_free;
        options_.memory.user_data = &budget_;
        options_.file.read = &read_buffer_file;
        options_.file.release = &release_buffer_file;
        options_.file.user_data = &reader_;
    }

    // The order: the size, the bytes, the JSON (within cgltf's memory budget, and the node count), the required
    // extensions, the byte ranges of the buffer views and accessors, the buffers, cgltf_validate, the scene's nodes,
    // the meshes (planned against the limits, then read), the materials worn and their images, the bounds
    std::expected<meshes::model, std::string> load()
    {
        for (const auto step : {&loader::read_file, &loader::parse, &loader::check_extensions, &loader::check_ranges,
                                &loader::load_buffers, &loader::validate, &loader::walk_scene, &loader::plan_meshes,
                                &loader::read_meshes, &loader::read_materials})
        {
            if (auto done = (this->*step)(); !done)
            {
                return std::unexpected(std::move(done.error()));
            }
        }
        finish();
        return std::move(model_);
    }

private:
    using step_result = std::expected<void, std::string>;

    // An error: the file's name, then the glTF object and what is wrong with it
    std::unexpected<std::string> fail(std::string_view what) const
    {
        return std::unexpected(std::format("{}: {}", name_, what));
    }

    step_result read_file()
    {
        std::error_code error;
        const std::uintmax_t bytes = std::filesystem::file_size(file_, error);
        if (error)
        {
            return fail(std::filesystem::exists(file_, error) ? "could not read the file" : "there is no such file");
        }
        if (bytes > limits_.file_bytes)
        {
            return fail(std::format("the file is {} bytes, more than the limit of {}", bytes, limits_.file_bytes));
        }
        bytes_.resize(static_cast<std::size_t>(bytes));
        std::ifstream in(file_, std::ios::binary);
        if (!in.read(bytes_.data(), static_cast<std::streamsize>(bytes_.size())))
        {
            return fail("could not read the file");
        }
        return {};
    }

    step_result parse()
    {
        cgltf_data* data = nullptr;
        const cgltf_result result = cgltf_parse(&options_, bytes_.data(), bytes_.size(), &data);
        data_.reset(data); // only set on success; the JSON's text and a .glb's binary chunk stay in bytes_
        switch (result)
        {
        case cgltf_result_success:
            break;
        case cgltf_result_legacy_gltf:
            return fail("a glTF 1.0 file: only glTF 2.0 is read");
        case cgltf_result_invalid_gltf:
            return fail("cgltf_parse refuses it: it breaks glTF 2.0's rules (a reference to an object that doesn't "
                        "exist, a node with two parents or in a cycle, a scene root that is another node's child, a "
                        "value of the wrong type, …)");
        case cgltf_result_out_of_memory:
            return fail(budget_.exceeded ? std::format("reading its JSON would take more than {} bytes of memory, the "
                                                       "limit (twice the file limit, plus 1 MiB)",
                                                       budget_.limit)
                                         : std::string("out of memory while reading its JSON"));
        default:
            return fail(std::format("not a glTF 2.0 file (cgltf_parse: {})", describe(result)));
        }
        if (const char* version = data_->asset.version; version && !std::string_view(version).starts_with("2."))
        {
            return fail(std::format("glTF {}: only glTF 2.x is read", version));
        }
        // The walk keeps a flag for every node and a stack entry for each one it has yet to visit: nodes are limited
        // to 4 per allowed placement, plus 4096 for those that place nothing (cameras, lights, skeleton joints, groups)
        constexpr std::size_t spare = 4096;
        const std::size_t most = std::numeric_limits<std::size_t>::max();
        const std::size_t nodes = limits_.placements > (most - spare) / 4 ? most : 4 * limits_.placements + spare;
        if (data_->nodes_count > nodes)
        {
            return fail(std::format("the file has {} nodes, more than the limit of {} (4 per allowed placement, plus "
                                    "{})",
                                    data_->nodes_count, nodes, spare));
        }
        return {};
    }

    // glTF's extensionsRequired: a file that needs what the renderer can't do is refused. KHR_mesh_quantization
    // (integer positions, normals and texture coordinates) is read by cgltf_accessor_unpack_floats, and
    // KHR_materials_emissive_strength only scales emission, which is ignored anyway.
    step_result check_extensions()
    {
        for (cgltf_size i = 0; i < data_->extensions_required_count; ++i)
        {
            const std::string_view name = data_->extensions_required[i];
            if (name == "KHR_mesh_quantization" || name == "KHR_materials_emissive_strength")
            {
                continue;
            }
            if (name == "KHR_draco_mesh_compression" || name == "EXT_meshopt_compression" ||
                name == "KHR_meshopt_compression")
            {
                return fail(std::format("it requires {}: compressed meshes, which this renderer doesn't decode; export "
                                        "the model without compression",
                                        name));
            }
            if (name == "KHR_texture_basisu" || name == "EXT_texture_webp")
            {
                return fail(std::format("it requires {}: images in a format this renderer doesn't decode (KTX2 or "
                                        "WebP); export the model without compression, with PNG or JPEG textures",
                                        name));
            }
            if (name == "KHR_texture_transform")
            {
                return fail("it requires KHR_texture_transform (textures moved, turned or scaled on the surface), "
                            "which this renderer doesn't apply");
            }
            return fail(std::format("it requires the extension {}, which this renderer doesn't support", name));
        }
        // The extensions it only uses: what they add is left out (KHR_texture_transform, warned per texture, and the
        // two above aside)
        for (cgltf_size i = 0; i < data_->extensions_used_count; ++i)
        {
            const std::string_view name = data_->extensions_used[i];
            if (name != "KHR_mesh_quantization" && name != "KHR_materials_emissive_strength" &&
                name != "KHR_texture_transform")
            {
                warnings_.add("extensions ignored", name, 0);
            }
        }
        if (data_->cameras_count > 0)
        {
            warnings_.add("cameras ignored, the scene sets its own", {}, data_->cameras_count);
        }
        if (data_->animations_count > 0)
        {
            warnings_.add("animations ignored, the meshes stand still", {}, data_->animations_count);
        }
        return {};
    }

    // --- Byte ranges ----------------------------------------------------------------------------------------------
    // `count` elements of `element` bytes, `stride` apart (stride ≥ 1), from byte `offset` of `size` bytes: whether
    // offset + stride·(count − 1) + element ≤ size, with no sum or product that can wrap (a difference only when it
    // can't go below 0, a quotient instead of a product)
    static bool fits(cgltf_size size, cgltf_size offset, cgltf_size element, cgltf_size stride, cgltf_size count)
    {
        return offset <= size && element <= size - offset &&
               (count == 0 || count - 1 <= (size - offset - element) / stride);
    }

    // What cgltf_validate checks of the buffer views and accessors, before it runs and before any buffer is loaded.
    // cgltf reads byteOffset, byteLength, byteStride and count with atoll, so each can be as large as 2^63 − 1, and
    // validate's own sums wrap modulo 2^64: a 12-byte view of byteStride 2^60 under 33 elements passes it (2^60·32 =
    // 2^65 ≡ 0), and unpacking then steps 2^60 bytes. glTF 2.0 ("Accessors", "Buffers and Buffer Views"): byteStride
    // 4 to 252, never less than an element; count at least 1; every element, and a sparse accessor's indices and
    // values (tightly packed), inside their view, every view inside its buffer. An index accessor without a view is
    // allowed (zeros, but for its sparse substitutions): plan_meshes bounds its count like any other.
    step_result check_ranges()
    {
        const cgltf_data& d = *data_;
        for (cgltf_size i = 0; i < d.buffer_views_count; ++i)
        {
            const cgltf_buffer_view& view = d.buffer_views[i];
            if (!fits(view.buffer->size, view.offset, view.size, 1, 1))
            {
                return fail(std::format("bufferViews[{}]: its {} bytes from byte {} reach past the end of buffers[{}] "
                                        "({} bytes)",
                                        i, view.size, view.offset, cgltf_buffer_index(&d, view.buffer),
                                        view.buffer->size));
            }
            if (view.stride != 0 && (view.stride < 4 || view.stride > 252))
            {
                return fail(std::format("bufferViews[{}]: its byteStride of {} isn't between 4 and 252", i,
                                        view.stride));
            }
        }
        for (cgltf_size i = 0; i < d.accessors_count; ++i)
        {
            const cgltf_accessor& a = d.accessors[i];
            if (a.component_type == cgltf_component_type_invalid || a.type == cgltf_type_invalid)
            {
                return fail(std::format("accessors[{}]: its type or componentType is missing", i));
            }
            if (a.count == 0)
            {
                return fail(std::format("accessors[{}]: it has no elements (count 0)", i));
            }
            const cgltf_size element = cgltf_calc_size(a.type, a.component_type);
            if (a.buffer_view)
            {
                const cgltf_size view = cgltf_buffer_view_index(&d, a.buffer_view);
                if (a.buffer_view->stride != 0 && a.buffer_view->stride < element)
                {
                    return fail(std::format("accessors[{}]: its elements of {} bytes are longer than the byteStride of "
                                            "bufferViews[{}], {}",
                                            i, element, view, a.buffer_view->stride));
                }
                if (!fits(a.buffer_view->size, a.offset, element, a.stride, a.count))
                {
                    return fail(std::format("accessors[{}]: its {} elements of {} bytes, {} apart from byte {}, reach "
                                            "past the end of bufferViews[{}] ({} bytes)",
                                            i, a.count, element, a.stride, a.offset, view, a.buffer_view->size));
                }
            }
            if (a.is_sparse)
            {
                const cgltf_accessor_sparse& s = a.sparse;
                if (s.count == 0 || s.count > a.count)
                {
                    return fail(std::format("accessors[{}]: {} sparse substitutions for its {} elements", i, s.count,
                                            a.count));
                }
                if (s.indices_component_type != cgltf_component_type_r_8u &&
                    s.indices_component_type != cgltf_component_type_r_16u &&
                    s.indices_component_type != cgltf_component_type_r_32u)
                {
                    return fail(std::format("accessors[{}]: its sparse indices aren't unsigned integers", i));
                }
                const cgltf_size index_size = cgltf_component_size(s.indices_component_type);
                if (!fits(s.indices_buffer_view->size, s.indices_byte_offset, index_size, index_size, s.count))
                {
                    return fail(std::format("accessors[{}]: its {} sparse indices reach past the end of "
                                            "bufferViews[{}]",
                                            i, s.count, cgltf_buffer_view_index(&d, s.indices_buffer_view)));
                }
                if (!fits(s.values_buffer_view->size, s.values_byte_offset, element, element, s.count))
                {
                    return fail(std::format("accessors[{}]: its {} sparse values reach past the end of bufferViews[{}]",
                                            i, s.count, cgltf_buffer_view_index(&d, s.values_buffer_view)));
                }
            }
        }
        return {};
    }

    // The buffers: a .glb's binary chunk, base64 data: URIs, or .bin files beside the model (read_buffer_file). Each
    // size is checked before cgltf allocates it: a data: URI's byteLength against what its text decodes to, a file's
    // against the file and the limit, and all files' byteLengths together against the limit too.
    step_result load_buffers()
    {
        std::size_t files = 0;
        std::uintmax_t declared = 0; // every file buffer's byteLength, saturating at the largest value
        cgltf_size passes = 0;       // the buffer whose file brings the sum past the limit
        for (cgltf_size i = 0; i < data_->buffers_count; ++i)
        {
            const cgltf_buffer& buffer = data_->buffers[i];
            if (!buffer.uri)
            {
                continue;
            }
            const std::string_view uri = buffer.uri;
            if (uri.starts_with("data:"))
            {
                const std::size_t comma = uri.find(',');
                if (comma == std::string_view::npos || !uri.substr(0, comma).ends_with(";base64"))
                {
                    return fail(std::format("buffers[{}]: a data: URI that isn't base64", i));
                }
                if (const std::size_t holds = base64_bytes(uri.substr(comma + 1)); buffer.size > holds)
                {
                    return fail(std::format("buffers[{}]: its byteLength of {} is more than its data: URI holds ({})",
                                            i, buffer.size, holds));
                }
            }
            else if (uri.contains("://"))
            {
                return fail(std::format("buffers[{}]: its data is on the web ({}); only files beside the model are "
                                        "read",
                                        i, uri));
            }
            else
            {
                if (!is_utf8(decoded_uri(uri)))
                {
                    return fail(std::format("buffers[{}]: its URI isn't valid UTF-8", i));
                }
                ++files;
                passes = declared > limits_.file_bytes ? passes : i;
                declared = buffer.size > std::numeric_limits<std::uintmax_t>::max() - declared
                               ? std::numeric_limits<std::uintmax_t>::max()
                               : declared + buffer.size;
            }
        }
        if (declared > limits_.file_bytes)
        {
            const std::string who = files == 1 ? std::string("its buffer file declares")
                                               : std::format("its {} buffer files declare", files);
            return fail(std::format("buffers[{}]: with it, {} {} bytes in all (byteLength), more than the limit of {}",
                                    passes, who, declared, limits_.file_bytes));
        }
        const cgltf_result result = cgltf_load_buffers(&options_, data_.get(), name_.c_str());
        if (result == cgltf_result_success)
        {
            return {};
        }
        // cgltf stops at the first buffer it can't load: the first one still without data that has some to load (a
        // URI, or a .glb's binary chunk for buffer 0)
        cgltf_size failed = 0;
        while (failed + 1 < data_->buffers_count &&
               (data_->buffers[failed].data || !(data_->buffers[failed].uri || (failed == 0 && data_->bin))))
        {
            ++failed;
        }
        if (!reader_.problem.empty())
        {
            return fail(std::format("buffers[{}]: {}", failed, reader_.problem));
        }
        if (budget_.exceeded)
        {
            return fail(std::format("buffers[{}]: decoding its data: URI would bring the memory of the parsed file "
                                    "past {} bytes, the limit (twice the file limit, plus 1 MiB)",
                                    failed, budget_.limit));
        }
        if (!data_->buffers[failed].uri)
        {
            return fail(std::format("buffers[{}]: the .glb's binary chunk is shorter than its byteLength", failed));
        }
        return fail(std::format("buffers[{}]: could not be loaded ({})", failed, describe(result)));
    }

    step_result validate()
    {
        if (const cgltf_result result = cgltf_validate(data_.get()); result != cgltf_result_success)
        {
            return fail(std::format("{} (cgltf_validate: {})", diagnose(), describe(result)));
        }
        return {};
    }

    // Why cgltf_validate refused the file, named: its checks that check_ranges hasn't made already, in its order, to
    // find the first that fails. Every range was checked, so nothing here reads outside its view, and nothing is
    // allocated: a refused file's counts may still be large.
    std::string diagnose() const
    {
        const cgltf_data& d = *data_;
        for (cgltf_size i = 0; i < d.accessors_count; ++i)
        {
            const cgltf_accessor& a = d.accessors[i];
            if (!a.is_sparse)
            {
                continue;
            }
            const cgltf_accessor_sparse& s = a.sparse;
            const cgltf_size index_size = cgltf_component_size(s.indices_component_type);
            if (const std::uint8_t* data = cgltf_buffer_view_data(s.indices_buffer_view))
            {
                data += s.indices_byte_offset;
                for (cgltf_size k = 0; k < s.count; ++k)
                {
                    std::uint32_t index = 0; // little-endian, as glTF's binary data (and x64)
                    std::memcpy(&index, data + k * index_size, index_size);
                    if (index >= a.count)
                    {
                        return std::format("accessors[{}]: sparse index {} is past its {} elements", i, index,
                                           a.count);
                    }
                }
            }
        }
        for (cgltf_size m = 0; m < d.meshes_count; ++m)
        {
            const cgltf_mesh& mesh = d.meshes[m];
            const cgltf_size targets = mesh.primitives_count > 0 ? mesh.primitives[0].targets_count : 0;
            if (mesh.primitives_count > 0 && ((mesh.weights && mesh.weights_count != targets) ||
                                              (mesh.target_names && mesh.target_names_count != targets)))
            {
                return std::format("meshes[{}]: its weights or target names don't match its {} morph targets", m,
                                   targets);
            }
            for (cgltf_size p = 0; p < mesh.primitives_count; ++p)
            {
                const cgltf_primitive& primitive = mesh.primitives[p];
                if (primitive.type == cgltf_primitive_type_invalid)
                {
                    return std::format("meshes[{}].primitives[{}]: its mode isn't one of glTF's 0 to 6", m, p);
                }
                if (primitive.targets_count != targets)
                {
                    return std::format("meshes[{}].primitives[{}]: {} morph targets, where primitives[0] has {}", m,
                                       p, primitive.targets_count, targets);
                }
                if (primitive.attributes_count == 0)
                {
                    return std::format("meshes[{}].primitives[{}]: it has no attributes", m, p);
                }
                const cgltf_size vertices = primitive.attributes[0].data->count;
                bool same = true;
                for (cgltf_size k = 0; k < primitive.attributes_count; ++k)
                {
                    same = same && primitive.attributes[k].data->count == vertices;
                }
                for (cgltf_size t = 0; t < primitive.targets_count; ++t)
                {
                    for (cgltf_size k = 0; k < primitive.targets[t].attributes_count; ++k)
                    {
                        same = same && primitive.targets[t].attributes[k].data->count == vertices;
                    }
                }
                if (!same)
                {
                    return std::format("meshes[{}].primitives[{}]: its attributes have different counts", m, p);
                }
                const cgltf_accessor* indices = primitive.indices;
                if (indices && (indices->type != cgltf_type_scalar ||
                                (indices->component_type != cgltf_component_type_r_8u &&
                                 indices->component_type != cgltf_component_type_r_16u &&
                                 indices->component_type != cgltf_component_type_r_32u) ||
                                indices->stride != cgltf_component_size(indices->component_type)))
                {
                    return std::format("meshes[{}].primitives[{}]: its indices aren't tightly packed unsigned "
                                       "integers (UNSIGNED_BYTE, UNSIGNED_SHORT or UNSIGNED_INT scalars)",
                                       m, p);
                }
                if (indices && indices->buffer_view && indices->buffer_view->buffer->data && !indices->is_sparse)
                {
                    for (cgltf_size k = 0; k < indices->count; ++k)
                    {
                        if (const cgltf_size index = cgltf_accessor_read_index(indices, k); index >= vertices)
                        {
                            return std::format("meshes[{}].primitives[{}]: index {} is past its {} vertices", m, p,
                                               index, vertices);
                        }
                    }
                }
                for (cgltf_size k = 0; k < primitive.mappings_count; ++k)
                {
                    if (primitive.mappings[k].variant >= d.variants_count)
                    {
                        return std::format("meshes[{}].primitives[{}]: a KHR_materials_variants mapping names a "
                                           "variant that doesn't exist",
                                           m, p);
                    }
                }
            }
        }
        for (cgltf_size i = 0; i < d.nodes_count; ++i)
        {
            const cgltf_node& node = d.nodes[i];
            if (node.weights && node.mesh && node.mesh->primitives_count > 0 &&
                node.mesh->primitives[0].targets_count != node.weights_count)
            {
                return std::format("nodes[{}]: its {} weights don't match its mesh's {} morph targets", i,
                                   node.weights_count, node.mesh->primitives[0].targets_count);
            }
            if (node.has_mesh_gpu_instancing)
            {
                const cgltf_mesh_gpu_instancing& instancing = node.mesh_gpu_instancing;
                bool broken = !node.mesh || instancing.attributes_count == 0;
                for (cgltf_size k = 0; !broken && k < instancing.attributes_count; ++k)
                {
                    broken = instancing.attributes[k].data->count != instancing.attributes[0].data->count;
                }
                if (broken)
                {
                    return std::format("nodes[{}]: its EXT_mesh_gpu_instancing has no mesh, no attributes, or "
                                       "attributes of different counts",
                                       i);
                }
            }
        }
        for (cgltf_size i = 0; i < d.nodes_count; ++i)
        {
            std::size_t steps = 0;
            for (const cgltf_node* node = &d.nodes[i]; node; node = node->parent)
            {
                if (++steps > d.nodes_count)
                {
                    return std::format("nodes[{}]: the node is its own ancestor (a cycle in the node hierarchy)", i);
                }
            }
        }
        for (cgltf_size i = 0; i < d.animations_count; ++i)
        {
            const cgltf_animation& animation = d.animations[i];
            for (cgltf_size j = 0; j < animation.channels_count; ++j)
            {
                const cgltf_animation_channel& channel = animation.channels[j];
                if (!channel.target_node)
                {
                    continue;
                }
                cgltf_size components = 1;
                if (channel.target_path == cgltf_animation_path_type_weights)
                {
                    const cgltf_mesh* mesh = channel.target_node->mesh;
                    if (!mesh || mesh->primitives_count == 0)
                    {
                        return std::format("animations[{}].channels[{}]: it animates the weights of a node without "
                                           "a mesh",
                                           i, j);
                    }
                    components = mesh->primitives[0].targets_count;
                }
                // validate's own product, wrapping as it does: the same verdict
                const cgltf_size values = channel.sampler->interpolation == cgltf_interpolation_type_cubic_spline ? 3
                                                                                                                  : 1;
                if (channel.sampler->input->count * components * values != channel.sampler->output->count)
                {
                    return std::format("animations[{}].channels[{}]: its sampler's output count doesn't match its "
                                       "input's",
                                       i, j);
                }
            }
        }
        return "it breaks glTF 2.0's rules";
    }

    // --- The scene: placements ----------------------------------------------------------------------------------
    // The default scene's nodes, depth first with an explicit stack (a deep hierarchy can't overflow the call stack),
    // each world matrix composed in double from its parent's. Every node is reached once: cgltf_parse refuses a node
    // with two parents and a cycle through a scene's root (a root that is a child), cgltf_validate a cycle elsewhere,
    // and a scene that lists a node twice is refused here. A file without scenes shows its root nodes (those without a
    // parent), and one without nodes each of its meshes once, untransformed. The roots are taken one at a time, when
    // the stack runs empty: it holds one path's siblings, never every root at once.
    step_result walk_scene()
    {
        const cgltf_scene* scene = data_->scene ? data_->scene : data_->scenes_count > 0 ? &data_->scenes[0] : nullptr;
        if (!scene && data_->nodes_count == 0)
        {
            if (data_->meshes_count > limits_.placements)
            {
                return fail(std::format("the file places {} meshes, more than the limit of {}", data_->meshes_count,
                                        limits_.placements));
            }
            for (cgltf_size i = 0; i < data_->meshes_count; ++i)
            {
                placements_.push_back({mesh_slot(&data_->meshes[i]), identity});
            }
            return {};
        }
        cgltf_size next = 0; // the next of the scene's nodes, or of all nodes to look at for one without a parent
        const auto next_root = [&]() -> const cgltf_node* {
            if (scene)
            {
                return next < scene->nodes_count ? scene->nodes[next++] : nullptr;
            }
            while (next < data_->nodes_count)
            {
                if (const cgltf_node* node = &data_->nodes[next++]; !node->parent)
                {
                    return node;
                }
            }
            return nullptr;
        };

        struct visit
        {
            const cgltf_node* node;
            matrix parent;
        };
        std::vector<visit> stack;
        std::vector<bool> reached(data_->nodes_count);
        for (;;)
        {
            if (stack.empty())
            {
                const cgltf_node* root = next_root();
                if (!root)
                {
                    break;
                }
                stack.push_back({root, identity});
            }
            const visit v = stack.back();
            stack.pop_back();
            const cgltf_size index = cgltf_node_index(data_.get(), v.node);
            if (reached[index])
            {
                return fail(std::format("nodes[{}]: reached twice from the scene (listed twice)", index));
            }
            reached[index] = true;
            const auto local = local_matrix(*v.node, index);
            if (!local)
            {
                return std::unexpected(local.error());
            }
            const matrix world = compose(v.parent, *local);
            if (!std::ranges::all_of(world, [](double x) { return std::isfinite(x); }))
            {
                return fail(std::format("nodes[{}]: its world transform isn't finite", index));
            }
            if (v.node->skin)
            {
                warnings_.add("skins ignored, the meshes in their bind pose", std::format("nodes[{}]", index));
            }
            if (v.node->mesh)
            {
                if (determinant(world) == 0)
                {
                    // A scale of 0 flattens the mesh to nothing (an animation's way to hide it): no normals exist
                    warnings_.add("meshes flattened to nothing by their node (determinant 0), skipped",
                                  std::format("nodes[{}]", index));
                }
                else if (placements_.size() == limits_.placements)
                {
                    return fail(std::format("the scene places more than {} meshes, the limit", limits_.placements));
                }
                else
                {
                    placements_.push_back({mesh_slot(v.node->mesh), world});
                }
            }
            for (cgltf_size i = v.node->children_count; i-- > 0;)
            {
                stack.push_back({v.node->children[i], world});
            }
        }
        return {};
    }

    // A node's own transform: its matrix (glTF stores 4×4 column-major), or T·R·S ("Transformations" in glTF 2.0:
    // scale, then rotate, then translate), with the rotation quaternion made unit in double first: a float quaternion
    // for 90° holds sin 45° only to 1e-8, which would put a turned point that far off
    std::expected<matrix, std::string> local_matrix(const cgltf_node& node, cgltf_size index) const
    {
        if (node.has_matrix)
        {
            const float* m = node.matrix;
            if (m[3] != 0 || m[7] != 0 || m[11] != 0 || m[15] != 1)
            {
                return fail(std::format("nodes[{}]: its matrix isn't an affine transform (the last row must be 0, 0, "
                                        "0, 1)",
                                        index));
            }
            return matrix{m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14]};
        }
        double x = node.rotation[0];
        double y = node.rotation[1];
        double z = node.rotation[2];
        double w = node.rotation[3];
        const double length = std::sqrt(x * x + y * y + z * z + w * w);
        if (!(length > 0) || !std::isfinite(length))
        {
            return fail(std::format("nodes[{}]: its rotation isn't a quaternion of nonzero length", index));
        }
        x /= length;
        y /= length;
        z /= length;
        w /= length;
        const std::array<double, 9> r{1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
                                      2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                                      2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
        const float* s = node.scale;
        const float* t = node.translation;
        return matrix{r[0] * s[0], r[1] * s[1], r[2] * s[2], t[0], r[3] * s[0], r[4] * s[1], r[5] * s[2], t[1],
                      r[6] * s[0], r[7] * s[1], r[8] * s[2], t[2]};
    }

    // The model's mesh for a glTF mesh, numbered by first placement: each is read once, however many nodes use it
    int mesh_slot(const cgltf_mesh* mesh)
    {
        const cgltf_size index = cgltf_mesh_index(data_.get(), mesh);
        if (slot_of_.empty())
        {
            slot_of_.assign(data_->meshes_count, -1);
        }
        if (slot_of_[index] < 0)
        {
            slot_of_[index] = static_cast<int>(reached_.size());
            reached_.push_back(mesh);
        }
        return slot_of_[index];
    }

    // --- The meshes ---------------------------------------------------------------------------------------------
    static bool is_triangles(cgltf_primitive_type type)
    {
        return type == cgltf_primitive_type_triangles || type == cgltf_primitive_type_triangle_strip ||
               type == cgltf_primitive_type_triangle_fan;
    }

    // A triangle primitive's vertex data: its POSITION, NORMAL and TEXCOORD_0 accessors (their indices; none: the
    // largest value). Primitives of one mesh with the same three share their vertices: exporters write a mesh's
    // material groups as primitives over one vertex array, differing only in their indices.
    using attribute_key = std::array<cgltf_size, 3>;

    attribute_key key_of(const cgltf_primitive& primitive) const
    {
        attribute_key key{};
        constexpr std::array<cgltf_attribute_type, 3> types{cgltf_attribute_type_position, cgltf_attribute_type_normal,
                                                            cgltf_attribute_type_texcoord};
        for (std::size_t k = 0; k < types.size(); ++k)
        {
            const cgltf_accessor* accessor = cgltf_find_accessor(&primitive, types[k], 0);
            key[k] = accessor ? cgltf_accessor_index(data_.get(), accessor) : std::numeric_limits<cgltf_size>::max();
        }
        return key;
    }

    // The triangle counts from the accessors' counts, before any array is allocated: the model's triangles, its
    // vertices (each set of shared vertices once; at most 3 per allowed triangle, an unindexed list's) and the
    // triangles all placements put in the scene. cgltf reads a count with atoll, up to 2^63 − 1: each count is
    // compared with what its limit has left before it is added, so no sum can wrap back under a limit.
    step_result plan_meshes()
    {
        constexpr std::size_t most = std::numeric_limits<std::size_t>::max();
        const std::size_t vertex_limit = limits_.triangles > most / 3 ? most : 3 * limits_.triangles;
        std::vector<std::size_t> planned(reached_.size(), 0);
        std::size_t triangles = 0;
        std::size_t vertices = 0;
        for (std::size_t s = 0; s < reached_.size(); ++s)
        {
            const cgltf_mesh& mesh = *reached_[s];
            const cgltf_size m = cgltf_mesh_index(data_.get(), &mesh);
            std::set<attribute_key> counted;
            for (cgltf_size p = 0; p < mesh.primitives_count; ++p)
            {
                const cgltf_primitive& primitive = mesh.primitives[p];
                const cgltf_accessor* position = cgltf_find_accessor(&primitive, cgltf_attribute_type_position, 0);
                if (!is_triangles(primitive.type) || !position) // skipped with a warning when read
                {
                    continue;
                }
                const std::size_t corners = primitive.indices ? primitive.indices->count : position->count;
                if (primitive.type == cgltf_primitive_type_triangles && corners % 3 != 0)
                {
                    return fail(std::format("meshes[{}].primitives[{}]: {} corners, not a multiple of 3", m, p,
                                            corners));
                }
                const std::size_t count =
                    primitive.type == cgltf_primitive_type_triangles ? corners / 3 : corners >= 3 ? corners - 2 : 0;
                if (count > limits_.triangles - triangles)
                {
                    return fail(std::format("meshes[{}].primitives[{}]: its {} triangles would bring the model past "
                                            "the limit of {} triangles",
                                            m, p, count, limits_.triangles));
                }
                triangles += count;
                planned[s] += count; // at most `triangles`: no wrap
                if (counted.insert(key_of(primitive)).second)
                {
                    if (position->count > vertex_limit - vertices)
                    {
                        return fail(std::format("meshes[{}].primitives[{}]: its {} vertices would bring the model "
                                                "past the limit of {} vertices (3 × the limit of {} triangles)",
                                                m, p, position->count, vertex_limit, limits_.triangles));
                    }
                    vertices += position->count;
                }
            }
        }
        std::size_t placed = 0;
        for (const meshes::placement& placement : placements_)
        {
            const std::size_t count = planned[static_cast<std::size_t>(placement.mesh)];
            if (count > limits_.placed_triangles - placed)
            {
                return fail(std::format("the scene places more triangles than the limit of {}",
                                        limits_.placed_triangles));
            }
            placed += count;
        }
        return {};
    }

    step_result read_meshes()
    {
        worn_.assign(data_->materials_count, false);
        copy_worn_.assign(data_->materials_count, false);
        for (const cgltf_mesh* mesh : reached_)
        {
            const cgltf_size m = cgltf_mesh_index(data_.get(), mesh);
            // The mesh holds normals if any of its triangle primitives has them (the others then get their faces'
            // normals, on vertices of their own), and texture coordinates if any has them (the others get (0, 0))
            bool normals = false;
            bool uvs = false;
            for (cgltf_size p = 0; p < mesh->primitives_count; ++p)
            {
                const cgltf_primitive& primitive = mesh->primitives[p];
                if (is_triangles(primitive.type) && cgltf_find_accessor(&primitive, cgltf_attribute_type_position, 0))
                {
                    normals = normals || cgltf_find_accessor(&primitive, cgltf_attribute_type_normal, 0);
                    uvs = uvs || cgltf_find_accessor(&primitive, cgltf_attribute_type_texcoord, 0);
                }
            }
            meshes::mesh out;
            std::map<attribute_key, vertex_range> ranges;
            for (cgltf_size p = 0; p < mesh->primitives_count; ++p)
            {
                if (auto read = read_primitive(mesh->primitives[p], std::format("meshes[{}].primitives[{}]", m, p),
                                               normals, uvs, out, ranges);
                    !read)
                {
                    return read;
                }
            }
            model_.meshes.push_back(std::move(out));
        }
        return {};
    }

    // An attribute's floats, through cgltf_accessor_unpack_floats: it converts integer and normalized components
    // (KHR_mesh_quantization) and follows any byte stride. A sparse accessor's substitutions are applied here:
    // cgltf_accessor_read_float refuses sparse accessors, and unpack_floats steps through their values at the base
    // view's byte stride, while glTF 2.0 packs them tightly ("Sparse Accessors": the values' view has no byteStride),
    // so an interleaved base would take every value after the first from the wrong bytes, past the values' view.
    std::expected<std::vector<float>, std::string> read_floats(const cgltf_accessor& accessor, cgltf_type type,
                                                               std::string_view where,
                                                               std::string_view attribute) const
    {
        const cgltf_size index = cgltf_accessor_index(data_.get(), &accessor);
        if (accessor.type != type)
        {
            return fail(std::format("{}: its {} (accessors[{}]) holds {}, not {}", where, attribute, index,
                                    type_name(accessor.type), type_name(type)));
        }
        const auto no_data = [&] {
            return fail(std::format("{}: its {} (accessors[{}]) has no data to read", where, attribute, index));
        };
        const auto unpack = [](const cgltf_accessor& a, std::vector<float>& out) {
            out.resize(cgltf_accessor_unpack_floats(&a, nullptr, 0));
            return cgltf_accessor_unpack_floats(&a, out.data(), out.size()) == out.size();
        };
        cgltf_accessor base = accessor; // the elements without the substitutions (zeros without a buffer view)
        base.is_sparse = 0;
        std::vector<float> floats;
        if (!unpack(base, floats))
        {
            return no_data();
        }
        if (accessor.is_sparse)
        {
            // The values and the indices as accessors of their own, tightly packed (check_ranges checked both ranges)
            const cgltf_accessor_sparse& sparse = accessor.sparse;
            cgltf_accessor values = base;
            values.buffer_view = sparse.values_buffer_view;
            values.offset = sparse.values_byte_offset;
            values.count = sparse.count;
            values.stride = cgltf_calc_size(accessor.type, accessor.component_type);
            std::vector<float> substitutes;
            cgltf_accessor indices{};
            indices.component_type = sparse.indices_component_type;
            indices.type = cgltf_type_scalar;
            indices.buffer_view = sparse.indices_buffer_view;
            indices.offset = sparse.indices_byte_offset;
            indices.count = sparse.count;
            indices.stride = cgltf_component_size(sparse.indices_component_type);
            std::vector<std::uint32_t> at(sparse.count);
            if (!unpack(values, substitutes) ||
                cgltf_accessor_unpack_indices(&indices, at.data(), sizeof(std::uint32_t), at.size()) != at.size())
            {
                return no_data();
            }
            const std::size_t n = cgltf_num_components(type);
            for (std::size_t k = 0; k < at.size(); ++k)
            {
                if (at[k] >= accessor.count)
                {
                    return fail(std::format("{}: its {} (accessors[{}]): sparse index {} is past its {} elements",
                                            where, attribute, index, at[k], accessor.count));
                }
                std::copy_n(substitutes.begin() + static_cast<std::ptrdiff_t>(n * k), n,
                            floats.begin() + static_cast<std::ptrdiff_t>(n * at[k]));
            }
        }
        // glTF decodes normalized signed integers as max(c / 127, −1) and max(c / 32767, −1); cgltf leaves the most
        // negative value slightly below −1
        if (accessor.normalized && (accessor.component_type == cgltf_component_type_r_8 ||
                                    accessor.component_type == cgltf_component_type_r_16))
        {
            for (float& value : floats)
            {
                value = std::max(value, -1.0f);
            }
        }
        return floats;
    }

    // The corners of a primitive's triangles in its own vertex numbers: its indices (u8, u16 or u32), or 0, 1, 2, …
    // without them. Every index is checked against the vertex count, sparse ones included.
    std::expected<std::vector<std::uint32_t>, std::string> read_indices(const cgltf_primitive& primitive,
                                                                        std::size_t vertices,
                                                                        std::string_view where) const
    {
        std::vector<std::uint32_t> indices;
        if (!primitive.indices)
        {
            indices.resize(vertices);
            std::iota(indices.begin(), indices.end(), 0u);
            return indices;
        }
        const cgltf_accessor& accessor = *primitive.indices;
        if (!accessor.is_sparse && accessor.buffer_view)
        {
            indices.resize(accessor.count);
            if (cgltf_accessor_unpack_indices(&accessor, indices.data(), sizeof(std::uint32_t), indices.size()) !=
                indices.size())
            {
                return fail(std::format("{}: its indices (accessors[{}]) have no data to read", where,
                                        cgltf_accessor_index(data_.get(), &accessor)));
            }
        }
        else
        {
            // cgltf_accessor_unpack_indices doesn't read sparse accessors; cgltf_accessor_unpack_floats does, and a
            // float holds every integer up to 2^24 exactly
            if (vertices > (1u << 24))
            {
                return fail(std::format("{}: sparse indices over {} vertices (more than 2^24) aren't read", where,
                                        vertices));
            }
            const auto floats = read_floats(accessor, cgltf_type_scalar, where, "indices");
            if (!floats)
            {
                return std::unexpected(floats.error());
            }
            indices.reserve(floats->size());
            for (const float value : *floats)
            {
                constexpr std::uint32_t invalid = std::numeric_limits<std::uint32_t>::max(); // refused just below
                const bool valid = value >= 0 && value < static_cast<float>(vertices);
                indices.push_back(valid ? static_cast<std::uint32_t>(value) : invalid);
            }
        }
        for (const std::uint32_t index : indices)
        {
            if (index >= vertices)
            {
                return fail(std::format("{}: index {} is past its {} vertices", where, index, vertices));
            }
        }
        return indices;
    }

    // The vertices a set of attribute accessors gave a mesh, for the primitives after the first that share them
    struct vertex_range
    {
        std::size_t first = 0; // in the mesh's vertices
        double extent = 0;     // the largest side of the vertices' bounding box
    };

    // One primitive appended to its mesh: positions, normals and texture coordinates through read_floats (or the
    // vertices an earlier primitive of the mesh read from the same accessors), the triangles of its topology (glTF
    // 2.0, "Meshes": a strip's triangle i is (vᵢ, vᵢ₊₁₊ᵢ﹪₂, vᵢ₊₂₋ᵢ﹪₂), a fan's (vᵢ₊₁, vᵢ₊₂, v₀), so all turn the same
    // way), minus those with a non-finite corner or no area (|e1 × e2| ≤ 10⁻¹²·extent², extent the largest side of the
    // bounding box of the primitive's vertices), and its material. A primitive without POSITION is skipped: glTF 2.0
    // ("Meshes") leaves such a primitive's rendering undefined, and the file's other primitives are still good.
    step_result read_primitive(const cgltf_primitive& primitive, const std::string& where, bool mesh_normals,
                               bool mesh_uvs, meshes::mesh& out, std::map<attribute_key, vertex_range>& ranges)
    {
        if (!is_triangles(primitive.type))
        {
            warnings_.add("points and lines skipped, only triangles are rendered", where);
            return {};
        }
        const cgltf_accessor* position = cgltf_find_accessor(&primitive, cgltf_attribute_type_position, 0);
        if (!position)
        {
            warnings_.add("triangle primitives without POSITION skipped", where);
            return {};
        }
        if (primitive.targets_count > 0)
        {
            warnings_.add("morph targets ignored, the meshes in their base shape", where);
        }
        for (cgltf_size k = 0; k < primitive.attributes_count; ++k)
        {
            const cgltf_attribute& attribute = primitive.attributes[k];
            if (attribute.type == cgltf_attribute_type_color)
            {
                warnings_.add("vertex colors (COLOR_n) ignored", where);
            }
            else if (attribute.type == cgltf_attribute_type_texcoord && attribute.index > 0)
            {
                warnings_.add("texture coordinates beyond TEXCOORD_0 ignored", where);
            }
        }

        const std::size_t vertices = position->count;
        if (vertices > std::numeric_limits<std::uint32_t>::max())
        {
            return fail(std::format("{}: {} vertices, more than 2^32", where, vertices));
        }
        const cgltf_accessor* normal = cgltf_find_accessor(&primitive, cgltf_attribute_type_normal, 0);
        const cgltf_accessor* texcoord = cgltf_find_accessor(&primitive, cgltf_attribute_type_texcoord, 0);
        // Shared vertices, as stored, unless the mesh has normals and this primitive none: then each triangle gets
        // three vertices of its own carrying its face's normal, which is flat shading exactly (and nothing to share)
        const bool flat_copies = mesh_normals && !normal;
        const attribute_key key = key_of(primitive);
        const auto shared = flat_copies ? ranges.end() : ranges.find(key);

        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> uvs;
        std::span<const float> p; // vertex i's three floats start at p[3i]
        double extent = 0;
        if (shared != ranges.end())
        {
            // The mesh's own copy: nothing is appended to out.positions while p looks into it (only indices are)
            p = std::span<const float>(out.positions).subspan(3 * shared->second.first, 3 * vertices);
            extent = shared->second.extent;
        }
        else
        {
            auto read = read_floats(*position, cgltf_type_vec3, where, "POSITION");
            if (!read)
            {
                return std::unexpected(read.error());
            }
            positions = std::move(*read);
            if (normal)
            {
                read = read_floats(*normal, cgltf_type_vec3, where, "NORMAL");
                if (!read)
                {
                    return std::unexpected(read.error());
                }
                normals = std::move(*read);
            }
            if (texcoord)
            {
                read = read_floats(*texcoord, cgltf_type_vec2, where, "TEXCOORD_0");
                if (!read)
                {
                    return std::unexpected(read.error());
                }
                uvs = std::move(*read);
            }
            // A NaN normal or texture coordinate would reach the renderer's interpolation: 0 instead (a zero normal
            // makes the renderer shade the face flat)
            std::size_t repaired = 0;
            for (std::vector<float>* values : {&normals, &uvs})
            {
                for (float& value : *values)
                {
                    if (!std::isfinite(value))
                    {
                        value = 0;
                        ++repaired;
                    }
                }
            }
            if (repaired > 0)
            {
                warnings_.add("normals or texture coordinates that aren't finite, set to 0", where, repaired);
            }
            p = positions;
            const double infinity = std::numeric_limits<double>::infinity();
            std::array<double, 3> lower{infinity, infinity, infinity};
            std::array<double, 3> upper{-infinity, -infinity, -infinity};
            for (std::size_t i = 0; i < vertices; ++i)
            {
                if (std::isfinite(p[3 * i]) && std::isfinite(p[3 * i + 1]) && std::isfinite(p[3 * i + 2]))
                {
                    for (std::size_t k = 0; k < 3; ++k)
                    {
                        lower[k] = std::min(lower[k], static_cast<double>(p[3 * i + k]));
                        upper[k] = std::max(upper[k], static_cast<double>(p[3 * i + k]));
                    }
                }
            }
            extent = std::max({upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2], 0.0});
        }

        const auto corners = read_indices(primitive, vertices, where);
        if (!corners)
        {
            return std::unexpected(corners.error());
        }
        std::vector<std::array<std::uint32_t, 3>> triangles;
        const std::vector<std::uint32_t>& c = *corners;
        const std::size_t n = c.size();
        triangles.reserve(primitive.type == cgltf_primitive_type_triangles ? n / 3 : n >= 3 ? n - 2 : 0);
        if (primitive.type == cgltf_primitive_type_triangles)
        {
            for (std::size_t i = 0; i + 2 < n; i += 3)
            {
                triangles.push_back({c[i], c[i + 1], c[i + 2]});
            }
        }
        else
        {
            for (std::size_t i = 0; i + 2 < n; ++i)
            {
                if (primitive.type == cgltf_primitive_type_triangle_fan)
                {
                    triangles.push_back({c[i + 1], c[i + 2], c[0]});
                }
                else
                {
                    triangles.push_back(i % 2 == 0 ? std::array{c[i], c[i + 1], c[i + 2]}
                                                   : std::array{c[i], c[i + 2], c[i + 1]});
                }
            }
        }

        // Vertex i's three floats start at p[3i] (in size_t: 3i overflows 32 bits from 1.4 billion vertices on)
        const auto at = [](std::uint32_t i) { return 3 * static_cast<std::size_t>(i); };
        const auto corner = [&](std::uint32_t i) {
            return std::array<double, 3>{p[at(i)], p[at(i) + 1], p[at(i) + 2]};
        };
        const auto finite = [&](std::uint32_t i) {
            return std::isfinite(p[at(i)]) && std::isfinite(p[at(i) + 1]) && std::isfinite(p[at(i) + 2]);
        };
        // e1 × e2 in double: twice the area, and the face's normal
        const auto face = [&](const std::array<std::uint32_t, 3>& t) {
            const auto a = corner(t[0]);
            const auto b = corner(t[1]);
            const auto d = corner(t[2]);
            const std::array<double, 3> e1{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
            const std::array<double, 3> e2{d[0] - a[0], d[1] - a[1], d[2] - a[2]};
            return std::array<double, 3>{e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                                         e1[0] * e2[1] - e1[1] * e2[0]};
        };
        std::size_t not_finite = 0;
        std::size_t no_area = 0;
        std::erase_if(triangles, [&](const std::array<std::uint32_t, 3>& t) {
            if (!finite(t[0]) || !finite(t[1]) || !finite(t[2]))
            {
                ++not_finite;
                return true;
            }
            const auto n2 = face(t);
            if (std::sqrt(n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]) <= 1e-12 * extent * extent)
            {
                ++no_area;
                return true;
            }
            return false;
        });
        if (not_finite > 0)
        {
            warnings_.add("triangles with a NaN or infinite corner, dropped", where, not_finite);
        }
        if (no_area > 0)
        {
            warnings_.add("triangles without area, dropped", where, no_area);
        }

        const std::uint32_t material = material_code(primitive, texcoord != nullptr, where);
        if (shared != ranges.end())
        {
            const auto base = static_cast<std::uint32_t>(shared->second.first); // fits: its vertices are in the mesh
            for (const auto& t : triangles)
            {
                out.indices.insert(out.indices.end(), {base + t[0], base + t[1], base + t[2]});
            }
            out.material_of.insert(out.material_of.end(), triangles.size(), material);
            return {};
        }
        const std::size_t first = out.positions.size() / 3;
        const std::size_t added = flat_copies ? 3 * triangles.size() : vertices;
        if (first + added > std::numeric_limits<std::uint32_t>::max())
        {
            return fail(std::format("{}: the mesh would have more than 2^32 vertices", where));
        }
        const auto push_vertex = [&](std::uint32_t i, const std::array<double, 3>* face_normal) {
            out.positions.insert(out.positions.end(), {p[at(i)], p[at(i) + 1], p[at(i) + 2]});
            if (face_normal)
            {
                out.normals.insert(out.normals.end(), {static_cast<float>((*face_normal)[0]),
                                                       static_cast<float>((*face_normal)[1]),
                                                       static_cast<float>((*face_normal)[2])});
            }
            else if (mesh_normals)
            {
                out.normals.insert(out.normals.end(), {normals[at(i)], normals[at(i) + 1], normals[at(i) + 2]});
            }
            if (mesh_uvs)
            {
                const std::size_t k = 2 * static_cast<std::size_t>(i);
                out.uvs.insert(out.uvs.end(), {uvs.empty() ? 0.0f : uvs[k], uvs.empty() ? 0.0f : uvs[k + 1]});
            }
        };
        if (flat_copies)
        {
            for (const auto& t : triangles)
            {
                const auto n2 = face(t);
                const double length = std::sqrt(n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]);
                const std::array<double, 3> unit{n2[0] / length, n2[1] / length, n2[2] / length};
                const auto base = static_cast<std::uint32_t>(out.positions.size() / 3);
                for (const std::uint32_t i : t)
                {
                    push_vertex(i, &unit);
                }
                out.indices.insert(out.indices.end(), {base, base + 1, base + 2});
            }
        }
        else
        {
            for (std::uint32_t i = 0; i < vertices; ++i)
            {
                push_vertex(i, nullptr);
            }
            ranges.emplace(key, vertex_range{first, extent});
            const auto base = static_cast<std::uint32_t>(first);
            for (const auto& t : triangles)
            {
                out.indices.insert(out.indices.end(), {base + t[0], base + t[1], base + t[2]});
            }
        }
        out.material_of.insert(out.material_of.end(), triangles.size(), material);
        return {};
    }

    // --- Materials and images -----------------------------------------------------------------------------------
    // While the meshes are read, a triangle's material is a code resolved by read_materials: glTF material i is 2i,
    // its copy without textures 2i + 1 (worn by a primitive without TEXCOORD_0: glTF's textures then have nothing to
    // be read at, and the factors alone remain), the default material default_code
    static constexpr std::uint32_t default_code = std::numeric_limits<std::uint32_t>::max();

    static bool uses_textures(const cgltf_material& material)
    {
        const cgltf_pbr_metallic_roughness& pbr = material.pbr_metallic_roughness;
        return (pbr.base_color_texture.texture && pbr.base_color_texture.texcoord == 0) ||
               (pbr.metallic_roughness_texture.texture && pbr.metallic_roughness_texture.texcoord == 0);
    }

    std::uint32_t material_code(const cgltf_primitive& primitive, bool has_uvs, std::string_view where)
    {
        if (!primitive.material)
        {
            return default_code;
        }
        const auto index = static_cast<std::uint32_t>(cgltf_material_index(data_.get(), primitive.material));
        if (!has_uvs && uses_textures(*primitive.material))
        {
            warnings_.add("textures on primitives without TEXCOORD_0, ignored there: the factors alone", where);
            if (!copy_worn_[index])
            {
                copy_worn_[index] = true;
                untextured_.push_back(index);
            }
            return 2 * index + 1;
        }
        worn_[index] = true;
        return 2 * index;
    }

    // Every glTF material in its own place (materials[i] is glTF's materials[i]), then the copies without textures,
    // then glTF's default material (a rough white metal: base color 1, metallic 1, roughness 1). Images are decoded
    // only for the materials a placed primitive wears with its textures: one of another scene, of a mesh no node
    // places, or of nobody keeps its factors and no image (its texture files are never opened, and its pixels cost
    // neither the image limit nor GPU memory); one worn only without TEXCOORD_0 lends its factors to its copy.
    step_result read_materials()
    {
        for (cgltf_size i = 0; i < data_->materials_count; ++i)
        {
            auto material = read_material(data_->materials[i], std::format("materials[{}]", i), worn_[i],
                                          worn_[i] || copy_worn_[i]);
            if (!material)
            {
                return std::unexpected(material.error());
            }
            model_.materials.push_back(*material);
        }
        std::vector<std::uint32_t> copy_of(data_->materials_count, 0);
        for (const std::uint32_t i : untextured_)
        {
            copy_of[i] = static_cast<std::uint32_t>(model_.materials.size());
            meshes::pbr_material copy = model_.materials[i];
            copy.base_color_image = -1;
            copy.metal_rough_image = -1;
            model_.materials.push_back(copy);
        }
        const auto default_index = static_cast<std::uint32_t>(model_.materials.size());
        model_.materials.push_back({});
        for (meshes::mesh& mesh : model_.meshes)
        {
            for (std::uint32_t& code : mesh.material_of)
            {
                code = code == default_code ? default_index : code % 2 == 0 ? code / 2 : copy_of[code / 2];
            }
        }
        return {};
    }

    // glTF's metallic-roughness factors, clamped to their ranges, and (`images`) the two textures the renderer reads.
    // A worn material's other features are warned about (`warn`): the renderer has no normal or occlusion maps, emits
    // no light from meshes and renders every surface opaque.
    std::expected<meshes::pbr_material, std::string> read_material(const cgltf_material& material,
                                                                   const std::string& where, bool images, bool warn)
    {
        const cgltf_pbr_metallic_roughness& pbr = material.pbr_metallic_roughness; // glTF's defaults when absent
        meshes::pbr_material out;
        out.base_color = {unit_interval(pbr.base_color_factor[0]), unit_interval(pbr.base_color_factor[1]),
                          unit_interval(pbr.base_color_factor[2])};
        out.metallic = unit_interval(pbr.metallic_factor);
        out.roughness = unit_interval(pbr.roughness_factor);
        if (!warn) // worn by nobody
        {
            return out;
        }
        if (!images)
        {
            return warn_features(material, where, out);
        }
        const auto base_color = image_for(pbr.base_color_texture, true, where + ".baseColorTexture");
        if (!base_color)
        {
            return std::unexpected(base_color.error());
        }
        out.base_color_image = *base_color;
        const auto metal_rough = image_for(pbr.metallic_roughness_texture, false, where + ".metallicRoughnessTexture");
        if (!metal_rough)
        {
            return std::unexpected(metal_rough.error());
        }
        out.metal_rough_image = *metal_rough;
        return warn_features(material, where, out);
    }

    meshes::pbr_material warn_features(const cgltf_material& material, const std::string& where,
                                       const meshes::pbr_material& out)
    {
        if (material.normal_texture.texture)
        {
            warnings_.add("normal textures ignored, no normal maps", where);
        }
        if (material.occlusion_texture.texture)
        {
            warnings_.add("occlusion textures ignored", where);
        }
        if (material.emissive_texture.texture ||
            std::ranges::any_of(material.emissive_factor, [](float e) { return e > 0; }))
        {
            warnings_.add("emission ignored, meshes don't glow", where);
        }
        if (material.alpha_mode != cgltf_alpha_mode_opaque)
        {
            warnings_.add("alphaMode MASK or BLEND ignored, rendered opaque", where);
        }
        return out;
    }

    // The model's image for a texture as one use reads it (sRGB or linear, its sampler's wrapping and filter): one
    // entry per distinct use, the glTF image decoded once however many uses share it. -1: no texture, or one the
    // renderer can't read (warned).
    std::expected<int, std::string> image_for(const cgltf_texture_view& view, bool srgb, const std::string& where)
    {
        if (!view.texture)
        {
            return -1;
        }
        if (view.texcoord != 0)
        {
            warnings_.add("textures read through TEXCOORD_1 or beyond, ignored: the factors alone", where);
            return -1;
        }
        if (view.has_transform)
        {
            warnings_.add("KHR_texture_transform ignored, textures as stored", where);
        }
        const cgltf_texture& texture = *view.texture;
        if (!texture.image)
        {
            warnings_.add("textures without a PNG or JPEG image, ignored: the factors alone",
                          std::format("textures[{}]", cgltf_texture_index(data_.get(), &texture)));
            return -1;
        }
        const cgltf_size source = cgltf_image_index(data_.get(), texture.image);
        // The sampler: glTF's REPEAT, MIRRORED_REPEAT and CLAMP_TO_EDGE, and magFilter NEAREST; repeat and linear
        // without one ("Samplers": "a sampler with repeat wrapping and auto filtering SHOULD be used")
        const auto wrap_of = [](cgltf_wrap_mode mode) {
            return mode == cgltf_wrap_mode_mirrored_repeat ? meshes::wrap::mirrored_repeat
                   : mode == cgltf_wrap_mode_clamp_to_edge ? meshes::wrap::clamp
                                                           : meshes::wrap::repeat;
        };
        const cgltf_sampler* sampler = texture.sampler;
        const meshes::wrap wrap_s = sampler ? wrap_of(sampler->wrap_s) : meshes::wrap::repeat;
        const meshes::wrap wrap_t = sampler ? wrap_of(sampler->wrap_t) : meshes::wrap::repeat;
        const bool nearest = sampler && sampler->mag_filter == cgltf_filter_type_nearest;
        for (std::size_t i = 0; i < image_sources_.size(); ++i)
        {
            const meshes::image& image = model_.images[i];
            if (image_sources_[i] == source && image.srgb == srgb && image.wrap_s == wrap_s &&
                image.wrap_t == wrap_t && image.nearest == nearest)
            {
                return static_cast<int>(i);
            }
        }

        meshes::image image;
        image.srgb = srgb;
        image.wrap_s = wrap_s;
        image.wrap_t = wrap_t;
        image.nearest = nearest;
        if (const auto decoded = std::ranges::find(image_sources_, source); decoded != image_sources_.end())
        {
            const meshes::image& first = model_.images[static_cast<std::size_t>(decoded - image_sources_.begin())];
            if (decoded_bytes_ + first.rgba.size() > limits_.image_bytes)
            {
                return fail(std::format("images[{}]: its second use would bring the decoded images to {} bytes, more "
                                        "than the limit of {}",
                                        source, decoded_bytes_ + first.rgba.size(), limits_.image_bytes));
            }
            image.width = first.width;
            image.height = first.height;
            image.rgba = first.rgba;
        }
        else if (auto read = decode(source, image); !read)
        {
            return std::unexpected(read.error());
        }
        decoded_bytes_ += image.rgba.size();
        image_sources_.push_back(source);
        model_.images.push_back(std::move(image));
        return static_cast<int>(model_.images.size() - 1);
    }

    // A glTF image's bytes, from wherever they are (cgltf loads buffers, not images): a buffer view (a .glb's embedded
    // images, which need a mimeType), a base64 data: URI, or a file beside the model, its URI percent-decoded and read
    // as UTF-8. Then its size from the header (stbi_info_from_memory) against the limits, and only then the pixels:
    // stbi_load_from_memory with 4 channels, 8-bit RGBA, rows from the top as glTF's (0, 0) wants. Never stbi_loadf,
    // whose gamma 2.2 isn't sRGB: the renderer decodes sRGB itself, exactly.
    step_result decode(cgltf_size index, meshes::image& image)
    {
        const cgltf_image& source = data_->images[index];
        const std::string where = std::format("images[{}]", index);
        if (source.mime_type && !is_png_or_jpeg(source.mime_type))
        {
            return fail(std::format("{}: its mimeType is {}, not image/png or image/jpeg", where, source.mime_type));
        }
        std::span<const unsigned char> bytes;
        std::unique_ptr<void, budget_deleter> base64(nullptr, budget_deleter{&budget_}); // decoded within the budget
        std::vector<unsigned char> file_bytes;
        if (source.buffer_view)
        {
            if (!source.mime_type)
            {
                return fail(std::format("{}: an image in a buffer view needs a mimeType", where));
            }
            const std::uint8_t* data = cgltf_buffer_view_data(source.buffer_view);
            if (!data)
            {
                return fail(std::format("{}: its buffer view has no data", where));
            }
            bytes = {data, source.buffer_view->size};
        }
        else if (!source.uri)
        {
            return fail(std::format("{}: it has neither a uri nor a bufferView", where));
        }
        else if (const std::string_view uri = source.uri; uri.starts_with("data:"))
        {
            const std::size_t comma = uri.find(',');
            if (comma == std::string_view::npos || !uri.substr(0, comma).ends_with(";base64"))
            {
                return fail(std::format("{}: a data: URI that isn't base64", where));
            }
            if (const std::string_view media = uri.substr(5, comma - 5 - 7); !media.empty() && !is_png_or_jpeg(media))
            {
                return fail(std::format("{}: its data: URI holds {}, not image/png or image/jpeg", where, media));
            }
            const std::string_view text = uri.substr(comma + 1);
            const std::size_t size = base64_bytes(text);
            void* decoded = nullptr;
            if (const cgltf_result result = cgltf_load_buffer_base64(&options_, size, text.data(), &decoded);
                result != cgltf_result_success)
            {
                return fail(result == cgltf_result_out_of_memory && budget_.exceeded
                                ? std::format("{}: decoding its data: URI would bring the memory of the parsed file "
                                              "past {} bytes, the limit (twice the file limit, plus 1 MiB)",
                                              where, budget_.limit)
                                : std::format("{}: its data: URI isn't valid base64", where));
            }
            base64.reset(decoded);
            bytes = {static_cast<const unsigned char*>(decoded), size};
        }
        else if (uri.contains("://"))
        {
            return fail(std::format("{}: the image is on the web ({}); only files beside the model are read", where,
                                    uri));
        }
        else
        {
            const std::string name = decoded_uri(uri);
            if (!is_utf8(name))
            {
                return fail(std::format("{}: its URI isn't valid UTF-8", where));
            }
            const std::filesystem::path path = file_.parent_path() / from_utf8(name);
            std::error_code error;
            const std::uintmax_t size = std::filesystem::file_size(path, error);
            if (error)
            {
                return fail(std::format("{}: there is no file '{}'", where, to_utf8(path)));
            }
            if (size > limits_.file_bytes)
            {
                return fail(std::format("{}: '{}' is {} bytes, more than the limit of {}", where, to_utf8(path), size,
                                        limits_.file_bytes));
            }
            file_bytes.resize(static_cast<std::size_t>(size));
            std::ifstream in(path, std::ios::binary);
            if (!in.read(reinterpret_cast<char*>(file_bytes.data()), static_cast<std::streamsize>(size)))
            {
                return fail(std::format("{}: could not read '{}'", where, to_utf8(path)));
            }
            bytes = file_bytes;
        }

        if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            return fail(std::format("{}: {} bytes, more than stb_image reads", where, bytes.size()));
        }
        const int length = static_cast<int>(bytes.size());
        int width = 0;
        int height = 0;
        int channels = 0;
        if (!stbi_info_from_memory(bytes.data(), length, &width, &height, &channels))
        {
            return fail(std::format("{}: not a PNG or JPEG image ({})", where, stbi_failure_reason()));
        }
        if (width > limits_.image_side || height > limits_.image_side)
        {
            return fail(std::format("{}: {} × {} pixels, more than the limit of {} a side", where, width, height,
                                    limits_.image_side));
        }
        const std::size_t size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4;
        if (decoded_bytes_ + size > limits_.image_bytes)
        {
            return fail(std::format("{}: it would bring the decoded images to {} bytes, more than the limit of {}",
                                    where, decoded_bytes_ + size, limits_.image_bytes));
        }
        stbi_uc* const pixels = stbi_load_from_memory(bytes.data(), length, &width, &height, &channels, 4);
        if (!pixels)
        {
            return fail(std::format("{}: could not decode it ({})", where, stbi_failure_reason()));
        }
        const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> owner(pixels, &stbi_image_free);
        image.width = width;
        image.height = height;
        image.rgba.assign(pixels, pixels + size);
        return {};
    }

    // --- The model ----------------------------------------------------------------------------------------------
    // Meshes left without triangles go, with their placements; the bounds are those of the vertices the triangles
    // use (a vertex only a dropped triangle used, such as a NaN corner, never counts), carried by every placement
    void finish()
    {
        std::vector<int> renumbered(model_.meshes.size(), -1);
        std::vector<meshes::mesh> kept;
        for (std::size_t s = 0; s < model_.meshes.size(); ++s)
        {
            if (!model_.meshes[s].indices.empty())
            {
                renumbered[s] = static_cast<int>(kept.size());
                kept.push_back(std::move(model_.meshes[s]));
            }
        }
        model_.meshes = std::move(kept);
        for (const meshes::placement& placement : placements_)
        {
            if (const int mesh = renumbered[static_cast<std::size_t>(placement.mesh)]; mesh >= 0)
            {
                model_.placements.push_back({mesh, placement.object_to_world});
            }
        }

        std::vector<std::vector<std::uint32_t>> used(model_.meshes.size());
        for (std::size_t s = 0; s < model_.meshes.size(); ++s)
        {
            std::vector<bool> seen(model_.meshes[s].positions.size() / 3);
            for (const std::uint32_t i : model_.meshes[s].indices)
            {
                if (!seen[i])
                {
                    seen[i] = true;
                    used[s].push_back(i);
                }
            }
        }
        const double infinity = std::numeric_limits<double>::infinity();
        std::array<double, 3> lower{infinity, infinity, infinity};
        std::array<double, 3> upper{-infinity, -infinity, -infinity};
        for (const meshes::placement& placement : model_.placements)
        {
            const matrix& m = placement.object_to_world;
            const std::vector<float>& p = model_.meshes[static_cast<std::size_t>(placement.mesh)].positions;
            for (const std::uint32_t i : used[static_cast<std::size_t>(placement.mesh)])
            {
                const double x = p[3 * static_cast<std::size_t>(i)];
                const double y = p[3 * static_cast<std::size_t>(i) + 1];
                const double z = p[3 * static_cast<std::size_t>(i) + 2];
                for (int r = 0; r < 3; ++r)
                {
                    const double world = m[r * 4] * x + m[r * 4 + 1] * y + m[r * 4 + 2] * z + m[r * 4 + 3];
                    lower[r] = std::min(lower[r], world);
                    upper[r] = std::max(upper[r], world);
                }
            }
        }
        if (model_.placements.empty())
        {
            warnings_.add("no triangles to render", {}, 0);
        }
        else
        {
            model_.lower = lower;
            model_.upper = upper;
        }
        model_.warnings = warnings_.lines(name_);
    }

    const std::filesystem::path file_;
    const std::string name_; // the path as given, in UTF-8: the start of every message
    const meshes::limits limits_;
    file_reader reader_;
    memory_budget budget_;
    cgltf_options options_{};
    std::vector<char> bytes_; // the file: cgltf's data points into it (the JSON, a .glb's binary chunk)
    std::unique_ptr<cgltf_data, data_deleter> data_; // declared after bytes_, reader_ and budget_: destroyed first
    std::vector<const cgltf_mesh*> reached_;          // the glTF meshes placed, in the order of their first placement
    std::vector<int> slot_of_;                        // a glTF mesh's place in reached_, -1: not placed
    std::vector<meshes::placement> placements_;       // mesh: the place in reached_
    std::vector<bool> worn_;                          // per glTF material: a placed primitive wears it, textures too
    std::vector<bool> copy_worn_;                     // per glTF material: its copy without textures is worn
    std::vector<std::uint32_t> untextured_;           // the materials whose copies are worn, in their copies' order
    std::vector<cgltf_size> image_sources_;           // per model image, the glTF image it was decoded from
    std::size_t decoded_bytes_ = 0;
    warning_list warnings_;
    meshes::model model_;
};
}

// Every error is returned, none thrown: the loader's own refusals, memory running out, and any other exception of the
// standard library (a path that can't be converted, a vector longer than it may be). The message names the file
// unless naming it fails too (a path that isn't valid Unicode, or no memory left): then a fixed text, short enough
// for the string's own buffer, so that making it allocates nothing.
std::expected<meshes::model, std::string> load_gltf(const std::filesystem::path& file, const meshes::limits& limits)
{
    const auto failure = [&file](std::string_view what) -> std::unexpected<std::string> {
        try
        {
            return std::unexpected(std::format("{}: {}", to_utf8(file), what));
        }
        catch (const std::exception&)
        {
            return std::unexpected(std::string("glTF: failed"));
        }
    };
    try
    {
        return loader(file, limits).load();
    }
    catch (const std::bad_alloc&)
    {
        return failure("out of memory");
    }
    catch (const std::exception& error)
    {
        return failure(error.what());
    }
}
