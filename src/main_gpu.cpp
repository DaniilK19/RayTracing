// RayTracingGPU: RayTracing's scenes rendered on an NVIDIA graphics card with CUDA (so far those in supported_scenes).
// The scene comes from the CPU ray tracer's own modules: the same objects and the same camera as in RayTracing.
// This program translates them into the plain data of the GPU renderer's interface (gpu/renderer.h), and saves the
// image with the CPU's image class. So the two programs differ only in how they render, and the CPU stays the
// reference: a scene rendered by both must come out the same up to the noise.
// RayTracingGPU weekend -s 1024

// The header includes standard headers (<vector>, <string>...), so it must come before `import std;`. MSVC can merge
// the two copies of the standard library, the headers' and the module's, only in this order: after the import, the
// headers would define its class templates a second time (error C2953).
#include "gpu/renderer.h"

import std;
import camera;
import cli;
import constant_medium;
import hittable;
import image;
import instance;
import material;
import perlin;
import quad;
import scenes;
import sphere;
import texture;
import vec3;

// --- The command line ------------------------------------------------------------------------------------------

// The scenes of RayTracing that the GPU can render so far. The others need features of the second and third books
// that the GPU doesn't have yet (lamps, smoke...), and to_gpu below says which one is missing.
// In the end to_gpu decides: any scene it can translate renders, listed here or not. (earth renders without its
// texture's file too, as the plain cyan sphere that the CPU renders then.)
// std::to_array (C++20) counts the names itself, so the array's size can't disagree with the list.
constexpr auto supported_scenes =
    std::to_array<std::string_view>({"materials", "weekend", "bouncing", "checker", "earth", "perlin", "quads"});
constexpr std::string_view default_gpu_scene = "weekend";

// The options are RayTracing's without -p (the GPU samples as the material scatters, as the CPU does in a scene lit
// by the sky), plus two of the GPU's own: --fp64 and --all-books-kernel. std::optional tells "not given: take it from
// the scene" apart from any given value.
struct options
{
    std::string_view scene = default_gpu_scene;
    std::optional<int> image_width;
    std::optional<int> samples_per_pixel;
    std::uint32_t seed = 0;
    std::vector<std::filesystem::path> outputs; // one per -o; parse_options adds <scene name>-gpu.png if there is none
    bool fp64 = false;
    bool all_books_kernel = false;
    bool help = false;
};

void print_usage()
{
    std::println("Usage: RayTracingGPU [scene] [options]\n\n"
                 "Renders RayTracing's scenes on an NVIDIA graphics card with CUDA. So far it can render\n"
                 "these scenes:");
    for (const std::string_view name : supported_scenes)
    {
        std::println("  {:10} {}", name, find_scene(name)->description);
    }
    std::println(R"(  Default: {}.

Options:
  -w, --width N        image width in pixels; the height follows the scene's aspect ratio
                       (each scene has its own default)
  -s, --samples N      rays per pixel: fewer is faster but noisier
                       (each scene has its own default); rounded to the nearest square k²,
                       because the rays form a k×k grid in the pixel
      --fp64           compute in 64-bit doubles like the CPU instead of 32-bit floats:
                       a check of the floats, many times slower on a GeForce card
      --all-books-kernel
                       render with the kernel for all three books, also the first book's scenes
                       (a check: the same image, bit for bit expected; and the price of generality)
      --seed N         which random numbers to draw, from 0 (the default) to {}:
                       another seed renders the same image with independent noise
  -o, --output FILE    where to save the image (default <scene>-gpu.png): a .png, or a .pfm that
                       keeps the linear values as floats, before gamma correction and clamping;
                       give it several times to save several files: -o out.png -o out.pfm
  -h, --help           show this help)",
                 default_gpu_scene, std::numeric_limits<std::uint32_t>::max());
}

std::expected<options, std::string> parse_options(std::span<char* const> args)
{
    options opts;
    for (std::size_t i = 1; i < args.size(); ++i) // args[0] is the program name
    {
        const std::string_view arg = args[i];
        if (arg == "-h" || arg == "--help")
        {
            opts.help = true;
            continue;
        }
        if (arg == "--fp64")
        {
            opts.fp64 = true;
            continue;
        }
        if (arg == "--all-books-kernel")
        {
            opts.all_books_kernel = true;
            continue;
        }

        // An argument without a dash is the scene name: any of RayTracing's scenes, for a clear message if the GPU
        // can't render it yet
        if (!arg.starts_with('-'))
        {
            if (!find_scene(arg))
            {
                return std::unexpected(std::format("unknown scene '{}'", arg));
            }
            opts.scene = arg;
            continue;
        }

        const bool is_width = arg == "-w" || arg == "--width";
        const bool is_samples = arg == "-s" || arg == "--samples";
        const bool is_output = arg == "-o" || arg == "--output";
        const bool is_seed = arg == "--seed";
        if (!is_width && !is_samples && !is_output && !is_seed)
        {
            return std::unexpected(std::format("unknown option '{}'", arg));
        }
        if (i + 1 == args.size())
        {
            return std::unexpected(std::format("{} needs a value", arg));
        }
        const std::string_view value = args[++i];

        if (is_output)
        {
            std::filesystem::path output = value;
            if (const auto checked = check_output(output); !checked)
            {
                return std::unexpected(std::format("{}: {}", arg, checked.error()));
            }
            opts.outputs.push_back(std::move(output));
            continue;
        }
        if (is_seed)
        {
            const std::optional<std::uint32_t> seed = parse_number<std::uint32_t>(value);
            if (!seed)
            {
                return std::unexpected(std::format("{}: expected an integer from 0 to {}, not '{}'", arg,
                                                   std::numeric_limits<std::uint32_t>::max(), value));
            }
            opts.seed = *seed;
            continue;
        }
        const std::optional<int> number = parse_positive_int(value);
        if (!number)
        {
            return std::unexpected(std::format("{}: expected an integer greater than zero, not '{}'", arg, value));
        }
        if (is_width)
        {
            opts.image_width = number;
        }
        else
        {
            opts.samples_per_pixel = number;
        }
    }
    if (opts.outputs.empty())
    {
        // -gpu: a GPU render never replaces the CPU's <scene>.png, so the two can be compared
        opts.outputs.emplace_back(std::format("{}-gpu.png", opts.scene));
    }
    return opts;
}

// --- The scene as the GPU renderer's data ----------------------------------------------------------------------

// The renderer can't use the CPU's objects: they are C++ classes with virtual functions, built by code the GPU
// can't run, and the GPU has its own memory. So they are translated into plain numbers (renderer.h), which the
// renderer copies into the GPU's memory. point3 and color are vec3 too.
gpu::vec3d to_gpu(const vec3& v)
{
    return {v.x, v.y, v.z};
}

// The pixel grid and the lens, as the CPU's camera computed them from the scene's settings
gpu::camera_desc to_gpu(const camera_frame& frame)
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
            .sky_top = to_gpu(frame.background.top)};
}

// Calls on_shape(shape) for every object in the lists under `object` that isn't a list itself (or for `object`, if it
// isn't a list). Only lists are opened so far, the world among them: an object that holds others in another way goes
// to on_shape whole, and add_shape says the GPU can't render it yet. Stops at the first error that on_shape returns.
// (An abbreviated function template, C++20: on_shape may be any callable that takes a hittable, here a lambda.)
std::expected<void, std::string> for_each_shape(const hittable& object, const auto& on_shape)
{
    // dynamic_cast asks an object what it really is: dynamic_cast<const hittable_list*>(&object) gives a pointer to
    // the list if object is one, and nullptr if it is another kind of hittable
    if (const auto* list = dynamic_cast<const hittable_list*>(&object))
    {
        for (const std::unique_ptr<hittable>& child : list->objects())
        {
            if (auto done = for_each_shape(*child, on_shape); !done)
            {
                return done;
            }
        }
        return {};
    }
    return on_shape(object);
}

// Translates the CPU's scene into the GPU renderer's tables (renderer.h), or says why the GPU can't render it yet. It
// reads the CPU's objects where they are, through dynamic_cast and their accessors, and changes nothing.
class scene_translator
{
public:
    explicit scene_translator(const camera_frame& frame) { desc_.camera = to_gpu(frame); }

    // Every shape of the world (spheres and quads), wherever it is in the world's lists
    std::expected<void, std::string> add_world(const hittable_list& world)
    {
        return for_each_shape(world, [this](const hittable& shape) { return add_shape(shape); });
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

    std::expected<void, std::string> add_shape(const hittable& shape);
    std::expected<int, std::string> material_index(const material* mat);
    std::expected<int, std::string> texture_index(const texture& tex);
};

std::expected<void, std::string> scene_translator::add_shape(const hittable& shape)
{
    if (const auto* ball = dynamic_cast<const sphere*>(&shape))
    {
        const auto mat = material_index(ball->mat());
        if (!mat)
        {
            return std::unexpected(mat.error());
        }
        // How far the center moves while the shutter is open, from the sphere's own accessors: exactly 0 for a sphere
        // that stands still (c + 1·0 - c), which is how the renderer tells the two kinds apart
        desc_.spheres.push_back({.center = to_gpu(ball->center_at(0)),
                                 .radius = ball->radius(),
                                 .material = *mat,
                                 .displacement = to_gpu(ball->center_at(1) - ball->center_at(0))});
        return {};
    }
    if (const auto* face = dynamic_cast<const quad*>(&shape))
    {
        const auto mat = material_index(face->mat());
        if (!mat)
        {
            return std::unexpected(mat.error());
        }
        desc_.quads.push_back(
            {.q = to_gpu(face->q()), .u = to_gpu(face->u()), .v = to_gpu(face->v()), .material = *mat});
        return {};
    }
    if (dynamic_cast<const translate*>(&shape) || dynamic_cast<const rotate_y*>(&shape))
    {
        return std::unexpected("it has objects moved or turned by translate or rotate_y");
    }
    if (dynamic_cast<const constant_medium*>(&shape))
    {
        return std::unexpected("it has smoke or fog");
    }
    return std::unexpected("it has an object of a kind the GPU can't render yet");
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
    else
    {
        return std::unexpected("it has a material other than lambertian, metal and dielectric");
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

// The whole scene, or why the GPU can't render it yet. So far the GPU renders spheres, standing still or moving, and
// quads, with matte surfaces of solid, checker, image or marble textures, metal and glass, lit by the sky.
std::expected<gpu::scene_desc, std::string> to_gpu(const scene& s, const camera_frame& frame)
{
    if (!s.lights.empty())
    {
        return std::unexpected("it has lamps");
    }
    // The GPU samples a diffuse bounce only as the material scatters. Without lamps the CPU does too (see
    // effective_sampling), unless the scene asks for uniform sampling.
    if (effective_sampling(s.camera.sampling, s.lights) != sampling_strategy::material)
    {
        return std::unexpected("its camera doesn't sample as the material scatters");
    }
    scene_translator translator(frame);
    if (const auto added = translator.add_world(s.world); !added)
    {
        return std::unexpected(added.error());
    }
    return std::move(translator).finish();
}

// --- Rendering -------------------------------------------------------------------------------------------------

// CUDA writes a version as 1000·major + 10·minor: 13040 is 13.4
std::string cuda_version(int version)
{
    return std::format("{}.{}", version / 1000, version % 1000 / 10);
}

int main(int argc, char* argv[])
{
    const auto opts = parse_options(std::span(argv, static_cast<std::size_t>(argc)));
    if (!opts)
    {
        std::println("Error: {}\n", opts.error());
        print_usage();
        return 1;
    }
    if (opts->help)
    {
        print_usage();
        return 0;
    }

    // The scene name was already checked in parse_options, so find_scene will find it
    scene s = find_scene(opts->scene)->build();
    s.camera.image_width = opts->image_width.value_or(s.camera.image_width);
    s.camera.samples_per_pixel = opts->samples_per_pixel.value_or(s.camera.samples_per_pixel);

    // The CPU's camera computes the pixel grid and the lens from the settings; the GPU renders with its numbers
    const camera cam(s.camera);
    const camera_frame frame = cam.frame();
    // The strategy the CPU would render with, as in RayTracing: lights and mixture fall back to material in a scene
    // without lamps
    const sampling_strategy requested = s.camera.sampling;
    const sampling_strategy sampling = effective_sampling(requested, s.lights);

    const auto gpu_scene = to_gpu(s, frame);
    if (!gpu_scene)
    {
        // C++23 formats ranges: n drops the brackets, and the empty format after the colon prints the names
        // without quotes
        std::println("Error: the GPU can't render '{}' yet: {}. So far it renders: {:n:}.", opts->scene,
                     gpu_scene.error(), supported_scenes);
        return 1;
    }

    // Every sampling_strategy has an entry in sampling_list (see camera.ixx), so find will find it
    const auto entry = std::ranges::find(sampling_list, sampling, &sampling_entry::strategy);
    std::println("Scene: {}, sampling: {}{}", opts->scene, entry->name,
                 sampling != requested ? " (no lights: the sky is the light)" : "");

    const int width = frame.image_width;
    const int height = frame.image_height;
    const std::int64_t rays_per_pixel = std::int64_t{frame.sqrt_spp} * frame.sqrt_spp;
    gpu::render_result result;
    try
    {
        const gpu::device_info device = gpu::query_device();
        std::println("GPU: {} ({} multiprocessors, compute capability {}.{}, {:.1f} GiB), CUDA runtime {} (the driver "
                     "supports up to {}), arithmetic: {}",
                     device.name, device.multiprocessors, device.compute_capability_major,
                     device.compute_capability_minor, device.memory_gib, cuda_version(device.runtime_version),
                     cuda_version(device.driver_version), opts->fp64 ? "float64" : "float32");
        std::println("Rendering {}×{}, rays per pixel: {} ({}×{} strata)", width, height, rays_per_pixel,
                     frame.sqrt_spp, frame.sqrt_spp);

        // The GPU renders in passes, each adding one sample (a camera ray and its bounces) to every pixel, and one
        // kernel launch may complete several of them, so the count goes down in steps (484, 483, 477, 468, ...). Like
        // the CPU's row count it never shows 0: the line is erased when the render is done. "\r" only moves back to
        // the start of the line, so each count is printed as wide as the first, the widest: a shorter number would
        // leave the last digits of the one before it on the screen (9999 after 10000 would read 99990).
        const std::size_t digits = std::to_string(rays_per_pixel).size();
        const auto show_progress = [digits](std::int64_t done, std::int64_t total) {
            if (done < total)
            {
                std::print("\rPasses remaining: {:{}}", total - done, digits); // {:{}}: the width is an argument
            }
        };
        show_progress(0, rays_per_pixel);
        const gpu::render_options gpu_options{
            .seed = opts->seed,
            .arithmetic = opts->fp64 ? gpu::precision::float64 : gpu::precision::float32,
            .progress = show_progress,
            .all_books_kernel = opts->all_books_kernel};
        result = gpu::render(*gpu_scene, gpu_options);
    }
    catch (const std::exception& error)
    {
        // The renderer throws when CUDA reports an error, maybe in the middle of the passes: erase the progress first
        std::println("\r{:30}\rError: {}", "", error.what());
        return 1;
    }
    std::print("\r{:30}\r", ""); // erase the progress line

    // Both times come from the renderer: the whole render() call (the BVH, copying the scene to the GPU and the image
    // back, all the passes), and the passes alone, timed on the GPU
    std::println("Render time: {:.2f} s (GPU passes: {:.2f} s)", result.seconds, result.kernel_seconds);
    // Which of the two kernels rendered (see the kernels in renderer.cu), and what decides how many of its threads
    // the GPU keeps in flight: the registers a thread takes (see blocks_per_multiprocessor)
    const gpu::kernel_info& kernel = result.kernel;
    std::println("Kernel: {}, {} registers and {} bytes of local memory per thread, {} blocks of {} threads per "
                 "multiprocessor ({} of its {} threads)",
                 kernel.name, kernel.registers, kernel.local_bytes, kernel.blocks_per_multiprocessor,
                 kernel.threads_per_block, kernel.blocks_per_multiprocessor * kernel.threads_per_block,
                 kernel.max_threads_per_multiprocessor);
    // A ray traced is a search for the closest hit, counted as on the CPU, so per sample it is the average number of
    // rays in a path: the camera ray and its bounces. The speed is that of the whole call, as the CPU's.
    const auto rays = static_cast<double>(result.rays);
    const double samples = static_cast<double>(width) * height * static_cast<double>(rays_per_pixel);
    std::println("Rays traced: {:.1f} million ({:.2f} per sample), {:.1f} million per second", rays / 1e6,
                 rays / samples, rays / result.seconds / 1e6);

    // The pixels go into the CPU's image class, so the mean radiance below and the saved files are computed exactly
    // as in RayTracing
    if (result.pixels.size() != static_cast<std::size_t>(width) * height * 3)
    {
        std::println("Error: the renderer returned {} numbers for {}×{} pixels", result.pixels.size(), width, height);
        return 1;
    }
    image img(width, height);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 3; // rows from the top, then r, g, b
            img[x, y] = color{result.pixels[i], result.pixels[i + 1], result.pixels[i + 2]};
        }
    }
    std::println("Mean radiance (linear, before clamping): {:.6f}", img.mean_radiance());

    // An image that can't be saved doesn't stop the others: the render is done, and each file may still be useful
    bool all_saved = true;
    for (const std::filesystem::path& output : opts->outputs)
    {
        if (const auto saved = img.save(output); !saved)
        {
            std::println("Error: {} ({})", saved.error(), to_utf8(output));
            all_saved = false;
            continue;
        }
        std::println("Image saved: {}", to_utf8(std::filesystem::absolute(output)));
    }
    return all_saved ? 0 : 1;
}
