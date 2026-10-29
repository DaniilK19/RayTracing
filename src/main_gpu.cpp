// RayTracingGPU: RayTracing's scenes rendered on an NVIDIA graphics card with CUDA.
// The scene comes from the CPU ray tracer's own modules: the same objects and the same camera as in RayTracing.
// The gpu_scene module translates them into the plain data of the GPU renderer's interface (gpu/renderer.h), and this
// program saves the image with the CPU's image class. So the two programs differ only in how they render, and the CPU
// stays the reference: a scene rendered by both must come out the same up to the noise.
// RayTracingGPU weekend -s 1024

// The header includes standard headers (<vector>, <string>...), so it must come before `import std;`. MSVC can merge
// the two copies of the standard library, the headers' and the module's, only in this order: after the import, the
// headers would define its class templates a second time (error C2953).
#include "gpu/renderer.h"

import std;
import camera;
import cli;
import gpu_scene;
import gpu_scenes;
import image;
import strategy_options;
import view_options;

// --- The command line ------------------------------------------------------------------------------------------

constexpr std::string_view default_gpu_scene = "weekend";
constexpr int max_launch_ms = static_cast<int>(gpu::max_launch_seconds * 1000);

// The options are RayTracing's, plus the GPU programs' own: --fp64, --all-books-kernel, --launch-ms, --strata and the
// view options (view_options.ixx). std::optional tells "not given: take it from the scene" apart from any given value.
struct options
{
    std::string_view scene = default_gpu_scene;
    std::optional<int> image_width;
    std::optional<int> samples_per_pixel;
    std::optional<int> strata; // rounds of strata × strata: -s is then exact (see gpu::session)
    std::optional<strategy> sampling; // a strategy of strategy_options.ixx: the books' four, then the GPU's own
    view_options view;                // the camera, if not the scene's own
    std::uint32_t seed = 0;
    std::vector<std::filesystem::path> outputs; // one per -o; parse_options adds <scene name>-gpu.png if there is none
    bool fp64 = false;
    bool all_books_kernel = false;
    int launch_ms = 50; // how long one kernel launch should take (see render_options::launch_seconds)
    bool help = false;
};

void print_usage()
{
    std::println("Usage: RayTracingGPU [scene] [options]\n\n"
                 "Renders RayTracing's scenes on an NVIDIA graphics card with CUDA.\n\n"
                 "Scenes:");
    std::print("{}", scenes_usage(default_gpu_scene));
    std::println(R"(
Options:
  -w, --width N        image width in pixels (each scene has its own default); the height follows
                       the scene's aspect ratio, unless --height is given
{}
  -s, --samples N      rays per pixel: fewer is faster but noisier
                       (each scene has its own default); rounded to the nearest square k²,
                       because the rays form a k×k grid in the pixel, unless --strata is given
      --strata K       sample every pixel in rounds of a K×K grid, K from 1 to {}: the rays per
                       pixel are then exactly -s (or the scene's), and each round has noise of its
                       own; --strata k -s k² renders the image without --strata. A last round not
                       complete has sampled only the top of every pixel, so the image leans up by up
                       to half a pixel: give -s in whole rounds (multiples of K²). K below 4 makes
                       short launches, as a launch holds at most one round: a check, not a speed-up
  -p, --sampling NAME  how a diffuse bounce chooses the next direction ("p" as in pdf,
                       probability density function): one of the strategies below
      --fp64           compute in 64-bit doubles like the CPU instead of 32-bit floats:
                       a check of the floats, many times slower on a GeForce card
      --all-books-kernel
                       render with the kernel for all three books, also the first book's scenes
                       (a check: the same image, bit for bit expected; and the price of generality)
      --launch-ms N    how long one kernel launch should take, in milliseconds, from 1 to {}
                       (default 50): a check, as the image doesn't depend on it, only the speed
      --seed N         which random numbers to draw, from 0 (the default) to {}:
                       another seed renders the same image with independent noise
  -o, --output FILE    where to save the image (default <scene>-gpu.png): a .png, or a .pfm that
                       keeps the linear values as floats, before gamma correction and clamping;
                       give it several times to save several files: -o out.png -o out.pfm
  -h, --help           show this help

Sampling (-p):)",
                 view_usage, max_strata, max_launch_ms, std::numeric_limits<std::uint32_t>::max());
    std::print("{}", strategy_usage());
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

        // An argument without a dash is the scene name: any of the GPU programs' scenes (gpu_scenes.ixx)
        if (!arg.starts_with('-'))
        {
            if (!find_gpu_scene(arg))
            {
                return std::unexpected(std::format("unknown scene '{}'", arg));
            }
            opts.scene = arg;
            continue;
        }

        // The view options: the image's height, where the camera stands and looks, its field of view and lens
        if (is_view_option(arg))
        {
            if (i + 1 == args.size())
            {
                return std::unexpected(std::format("{} needs a value", arg));
            }
            if (const auto parsed = parse_view_option(arg, args[++i], opts.view); !parsed)
            {
                return std::unexpected(parsed.error());
            }
            continue;
        }

        const bool is_width = arg == "-w" || arg == "--width";
        const bool is_samples = arg == "-s" || arg == "--samples";
        const bool is_output = arg == "-o" || arg == "--output";
        const bool is_seed = arg == "--seed";
        const bool is_sampling = arg == "-p" || arg == "--sampling";
        const bool is_launch = arg == "--launch-ms";
        const bool is_strata = arg == "--strata";
        if (!is_width && !is_samples && !is_output && !is_seed && !is_sampling && !is_launch && !is_strata)
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
        if (is_sampling)
        {
            // Any of the GPU programs' strategy names (strategy_options.ixx)
            const strategy_entry* entry = find_strategy(value);
            if (!entry)
            {
                return std::unexpected(std::format("unknown sampling strategy '{}'", value));
            }
            opts.sampling = entry->value;
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
        if (is_launch)
        {
            // Windows stops a kernel that runs for about 2 seconds, and a launch can run several times longer than
            // planned (see launch_planner in renderer.cu)
            if (!number || *number > max_launch_ms)
            {
                return std::unexpected(
                    std::format("{}: expected an integer from 1 to {}, not '{}'", arg, max_launch_ms, value));
            }
            opts.launch_ms = *number;
            continue;
        }
        if (is_strata)
        {
            if (!number || *number > max_strata)
            {
                return std::unexpected(
                    std::format("{}: expected an integer from 1 to {}, not '{}'", arg, max_strata, value));
            }
            opts.strata = number;
            continue;
        }
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

    // The scene with the options' settings, translated for the GPU (gpu_scene.ixx)
    const auto built = build_gpu_scene(
        opts->scene,
        {.image_width = opts->image_width,
         .samples_per_pixel = opts->samples_per_pixel,
         .sampling = opts->sampling,
         .view = opts->view,
         .strata = opts->strata});
    if (!built)
    {
        std::println("Error: {}", built.error());
        return 1;
    }
    const camera_frame& frame = built->frame;
    // The strategy rendered with, and whether it isn't the one asked for (-p, or the scene's own)
    std::println("Scene: {}, sampling: {}{}", opts->scene, strategy_name(built->sampling),
                 built->sampling != built->requested ? " (no lights: the sky is the light)" : "");
    // The view as options: pasted into a command line, they give this camera again bit for bit (and with the same
    // other options the same image), and they are the place to start from to change it
    std::println("View: {}", format_view(built->cpu.camera, frame.image_height));

    const int width = frame.image_width;
    const int height = frame.image_height;
    const std::int64_t rays_per_pixel = built->passes;
    gpu::render_result result;
    try
    {
        const gpu::device_info device = gpu::query_device();
        std::println("GPU: {} ({} multiprocessors, compute capability {}.{}, {:.1f} GiB), CUDA runtime {} (the driver "
                     "supports up to {}), arithmetic: {}",
                     device.name, device.multiprocessors, device.compute_capability_major,
                     device.compute_capability_minor, device.memory_gib, cuda_version(device.runtime_version),
                     cuda_version(device.driver_version), opts->fp64 ? "float64" : "float32");
        std::println("Rendering {}×{}, rays per pixel: {} ({}{}×{} strata)", width, height, rays_per_pixel,
                     opts->strata ? "rounds of " : "", frame.sqrt_spp, frame.sqrt_spp);

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
            .all_books_kernel = opts->all_books_kernel,
            .launch_seconds = opts->launch_ms / 1000.0,
            .passes = rays_per_pixel};
        result = gpu::render(built->desc, gpu_options);
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
    const auto img = to_image(result.pixels, width, height);
    if (!img)
    {
        std::println("Error: {}", img.error());
        return 1;
    }
    std::println("Mean radiance (linear, before clamping): {:.6f}", img->mean_radiance());

    // An image that can't be saved doesn't stop the others: the render is done, and each file may still be useful
    bool all_saved = true;
    for (const std::filesystem::path& output : opts->outputs)
    {
        if (const auto saved = img->save(output); !saved)
        {
            std::println("Error: {} ({})", saved.error(), to_utf8(output));
            all_saved = false;
            continue;
        }
        std::println("Image saved: {}", to_utf8(std::filesystem::absolute(output)));
    }
    return all_saved ? 0 : 1;
}
