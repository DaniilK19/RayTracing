// RayTracingGPU: RayTracing's scenes rendered on an NVIDIA graphics card with CUDA.
// The scene comes from the CPU ray tracer's own modules: the same objects and the same camera as in RayTracing.
// The gpu_scene module translates them into the plain data of the GPU renderer's interface (gpu/renderer.h), and this
// program saves the image with the CPU's image class. So the two programs differ only in how they render, and the CPU
// stays the reference: a scene rendered by both must come out the same up to the noise.
// RayTracingGPU weekend -s 1024

// The headers include standard headers (<vector>, <string>...), so they must come before `import std;`. MSVC can merge
// the two copies of the standard library, the headers' and the module's, only in this order: after the import, the
// headers would define its class templates a second time (error C2953).
#include "gpu/renderer.h"
#include "gpu/start_optix.h"

import std;
import camera;
import cli;
import gpu_scene;
import gpu_scenes;
import image;
import strategy_options;
import tone_options;
import view_options;

// --- The command line ------------------------------------------------------------------------------------------

constexpr std::string_view default_gpu_scene = "weekend";
constexpr int max_launch_ms = static_cast<int>(gpu::max_launch_seconds * 1000);

// The options are RayTracing's, plus the GPU programs' own: --fp64, --all-books-kernel, --quality-kernel, --optix,
// --launch-ms, --strata, --denoise, the view options (view_options.ixx) and the display's (tone_options.ixx).
// std::optional tells "not given: take it from the scene" apart from any given value.
struct options
{
    std::string_view scene = default_gpu_scene; // a scene's name, or a glTF model's file (is_model_path)
    std::optional<int> image_width;
    std::optional<int> samples_per_pixel;
    std::optional<int> strata; // rounds of strata × strata: -s is then exact (see gpu::session)
    std::optional<strategy> sampling; // a strategy of strategy_options.ixx: the books' four, then the GPU's own
    view_options view;                // the camera, if not the scene's own
    std::uint32_t seed = 0;
    std::vector<std::filesystem::path> outputs; // one per -o; parse_options adds <scene name>-gpu.png if there is none
    display_choice display;                     // --tonemap, --exposure: how a PNG shows the light
    bool fp64 = false;
    bool all_books_kernel = false;
    bool quality_kernel = false;
    bool optix = false;   // OptiX traces the quality kernel's integrator (render_options::optix_tracer)
    bool denoise = false; // also each image denoised by OptiX's AI denoiser, beside it
    int launch_ms = 50;   // how long one kernel launch should take (see render_options::launch_seconds)
    bool help = false;
};

void print_usage()
{
    std::println("Usage: RayTracingGPU [scene] [options]\n\n"
                 "Renders RayTracing's scenes on an NVIDIA graphics card with CUDA.\n\n"
                 "Scenes:");
    std::print("{}", scenes_usage(default_gpu_scene));
    std::println("\nOr a glTF 2.0 model, FILE.glb or FILE.gltf, in a photo studio (OptiX renders it; the image is\n"
                 "saved in the current folder as NAME-gpu.png, NAME the file's name without its extension).");
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
      --quality-kernel render with the quality kernel, the third one, also the books' scenes under
                       -p material (a check: the same image within its noise, mostly bit for bit)
      --optix          trace with OptiX (on the ray tracing cores where the GPU has them): the
                       quality kernel's integrator in an OptiX program (float; -p material, nee or
                       mis); scenes with triangle meshes always use it
      --launch-ms N    how long one kernel launch should take, in milliseconds, from 1 to {}
                       (default 50): a check, as the image doesn't depend on it, only the speed
      --seed N         which random numbers to draw, from 0 (the default) to {}:
                       another seed renders the same image with independent noise
  -o, --output FILE    where to save the image (default <scene>-gpu.png): a .png, or a .pfm that
                       keeps the linear values as floats, before gamma correction and clamping;
                       give it several times to save several files: -o out.png -o out.pfm
      --denoise        also write each image denoised by OptiX's AI denoiser, as
                       NAME-denoised.png/.pfm (for the eye: it is biased; the plain images stay
                       as rendered)
{}
  -h, --help           show this help

Sampling (-p):)",
                 view_usage, max_strata, max_launch_ms, std::numeric_limits<std::uint32_t>::max(), tone_usage());
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
        if (arg == "--quality-kernel")
        {
            opts.quality_kernel = true;
            continue;
        }
        if (arg == "--optix")
        {
            opts.optix = true;
            continue;
        }
        if (arg == "--denoise")
        {
            opts.denoise = true;
            continue;
        }

        // An argument without a dash is the scene name: any of the GPU programs' scenes (gpu_scenes.ixx), or a glTF 2.0
        // model's file (it ends in .glb or .gltf, which no scene's name does), which must exist
        if (!arg.starts_with('-'))
        {
            if (is_model_path(arg))
            {
                std::error_code error;
                if (!std::filesystem::is_regular_file(std::filesystem::path(arg), error))
                {
                    return std::unexpected(std::format("there is no file '{}'", arg));
                }
            }
            else if (!find_gpu_scene(arg))
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
        const bool is_tonemap = arg == "--tonemap";
        const bool is_exposure = arg == "--exposure";
        if (!is_width && !is_samples && !is_output && !is_seed && !is_sampling && !is_launch && !is_strata &&
            !is_tonemap && !is_exposure)
        {
            return std::unexpected(std::format("unknown option '{}'", arg));
        }
        if (i + 1 == args.size())
        {
            return std::unexpected(std::format("{} needs a value", arg));
        }
        const std::string_view value = args[++i];

        // The display: whether a tone map takes an exposure depends on the scene, checked once it is known
        if (is_tonemap)
        {
            const auto op = tone_map_option(value);
            if (!op)
            {
                return std::unexpected(op.error());
            }
            opts.display.op = *op;
            continue;
        }
        if (is_exposure)
        {
            const auto ev = exposure_option(arg, value);
            if (!ev)
            {
                return std::unexpected(ev.error());
            }
            opts.display.exposure = *ev;
            continue;
        }

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
        // -gpu: a GPU render never replaces the CPU's <scene>.png, so the two can be compared. A model's image is named
        // after its file, in the current folder: bust.glb gives bust-gpu.png
        const std::string stem =
            is_model_path(opts.scene) ? to_utf8(std::filesystem::path(opts.scene).stem()) : std::string(opts.scene);
        opts.outputs.emplace_back(std::format("{}-gpu.png", stem));
    }
    return opts;
}

// --- Rendering -------------------------------------------------------------------------------------------------

// CUDA writes a version as 1000·major + 10·minor: 13040 is 13.4
std::string cuda_version(int version)
{
    return std::format("{}.{}", version / 1000, version % 1000 / 10);
}

// --denoise's file for an output: <stem>-denoised.png or .pfm beside it
std::filesystem::path denoised_name(const std::filesystem::path& output)
{
    std::filesystem::path name = output.stem();
    name += "-denoised";
    name += output.extension();
    return output.parent_path() / name;
}

// --denoise: each output's denoised twin, after the plain files, which stay as they would be without it. A PNG holds
// the display's bytes of the denoised image (render_result::denoised_rgba, through the plain PNG's display: books, or
// the tone map and exposure), written by the books' writer through bytes_to_image as a tone map's PNG is; a PFM holds
// the denoised linear floats (render_result::denoised), before any exposure. One line names the files, and the tiles
// when the denoiser took the image in tiles (its bytes then differ slightly from one invoke's). False if one couldn't
// be saved; the others are saved all the same.
bool save_denoised(const gpu::render_result& result, const std::vector<std::filesystem::path>& outputs, int width,
                   int height)
{
    std::optional<image> png;
    try
    {
        png = bytes_to_image(result.denoised_rgba, width, height);
    }
    catch (const std::invalid_argument& error)
    {
        std::println("Error: {}", error.what());
        return false;
    }
    const std::vector<double> linear(result.denoised.begin(), result.denoised.end());
    const auto pfm = to_image(linear, width, height);
    if (!pfm)
    {
        std::println("Error: {}", pfm.error());
        return false;
    }
    std::vector<std::string> saved_files;
    bool all_saved = true;
    for (const std::filesystem::path& output : outputs)
    {
        const std::filesystem::path file = denoised_name(output);
        const image& source = is_png(file) ? *png : *pfm;
        if (const auto saved = source.save(file); !saved)
        {
            std::println("Error: {} ({})", saved.error(), to_utf8(file));
            all_saved = false;
            continue;
        }
        saved_files.push_back(to_utf8(std::filesystem::absolute(file)));
    }
    if (!saved_files.empty())
    {
        std::println("Denoised: {}{}",
                     saved_files | std::views::join_with(std::string(", ")) | std::ranges::to<std::string>(),
                     result.denoise_tile > 0 ? std::format(" (in tiles of {})", result.denoise_tile) : std::string());
    }
    return all_saved;
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
    // The denoiser is OptiX's, and works in float: refused before anything starts
    if (opts->denoise && opts->fp64)
    {
        std::println("Error: --denoise works in float only: drop --fp64");
        return 1;
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
    // The kernel asked for, and what the quality kernel or OptiX can't render (see check_kernel in gpu_scene.ixx)
    if (opts->quality_kernel && opts->all_books_kernel)
    {
        std::println("Error: --all-books-kernel picks the books' kernel for all three books; --quality-kernel needs "
                     "the quality kernel");
        return 1;
    }
    if (opts->optix && opts->quality_kernel)
    {
        std::println("Error: --optix and --quality-kernel each pick a renderer: give one");
        return 1;
    }
    if (opts->optix && opts->all_books_kernel)
    {
        std::println("Error: --all-books-kernel picks the books' kernel for all three books; --optix renders with "
                     "OptiX");
        return 1;
    }
    const kernel_choice chosen = opts->optix              ? kernel_choice::optix
                                 : opts->quality_kernel   ? kernel_choice::quality
                                 : opts->all_books_kernel ? kernel_choice::all_books
                                                          : kernel_choice::automatic;
    // A scene with triangle meshes (a model's file among them) renders with OptiX, which alone traces them, without
    // --optix: what it can't be rendered with is said first, in its own words (see check_meshes in gpu_scene.ixx)
    const bool meshes = !built->desc.meshes.instances.empty();
    if (const auto checked = check_meshes(*built, opts->scene, chosen, opts->fp64); !checked)
    {
        std::println("Error: {}", checked.error());
        return 1;
    }
    if (const auto checked = check_kernel(*built, opts->scene, chosen, opts->fp64); !checked)
    {
        std::println("Error: {}", checked.error());
        return 1;
    }
    // The display: the options', else the scene's own (books for the books' scenes), and what it can't be
    const gpu::display_options scene_display = scene_entry(opts->scene)->display;
    if (const auto checked = check_display(opts->display, scene_display, opts->fp64); !checked)
    {
        std::println("Error: {}", checked.error());
        return 1;
    }
    // OptiX, only for what needs it: --optix and a scene with triangle meshes, which it traces, and --denoise, whose
    // denoiser is OptiX's; one runtime for all three. Nothing of it starts otherwise, so every other run is as before.
    // It runs on the CUDA device query_device() starts, so it starts after that, below; a build without OptiX's
    // headers refuses them here instead, before CUDA starts (start_optix() then says why, and starts nothing). The
    // runtime lives until main returns, after every session that used it (never a static: CUDA's own exit comes
    // first). An error names what asked for it: --optix if given, else the scene's meshes, else --denoise.
    const bool use_optix = opts->optix || meshes; // the tracer (render_options::optix_tracer)
    const bool wants_optix = use_optix || opts->denoise;
    std::shared_ptr<gpu::optix_runtime> optix;
    const auto open_optix = [&optix, &opts, meshes] {
        try
        {
            optix = gpu::start_optix();
            return true;
        }
        catch (const std::exception& error)
        {
            // Why: no headers, the driver, OptiX's version
            if (opts->optix)
            {
                std::println("Error: --optix: {}", error.what());
            }
            else if (meshes)
            {
                std::println("Error: '{}' has triangle meshes, which only OptiX renders: {}", opts->scene,
                             error.what());
            }
            else
            {
                std::println("Error: --denoise: {}", error.what());
            }
            return false;
        }
    };
    if (wants_optix && !gpu::optix_built() && !open_optix())
    {
        return 1;
    }
    const gpu::display_options display = effective_display(opts->display, scene_display);
    const camera_frame& frame = built->frame;
    // The strategy rendered with, and whether it isn't the one asked for (-p, or the scene's own)
    std::println("Scene: {}, sampling: {}{}", opts->scene, strategy_name(built->sampling),
                 built->sampling != built->requested ? " (no lights: the sky is the light)" : "");
    // The view as options: pasted into a command line, they give this camera again bit for bit (and with the same
    // other options the same image), and they are the place to start from to change it
    std::println("View: {}", format_view(built->cpu.camera, frame.image_height));
    // A tone map is named; the books' bytes, as before, aren't
    if (display.op != gpu::tone_map::books)
    {
        std::println("Display: {}, {}", tone_map_name(display.op), format_exposure(display.exposure));
    }
    // What the renderer can't show of the glTF models it read (gltf.ixx), one line each, before it renders
    for (const model_load& model : built->models)
    {
        for (const std::string& warning : model.warnings)
        {
            std::println("Warning: {}", warning);
        }
    }

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
        if (wants_optix && !open_optix())
        {
            return 1;
        }
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
            .quality_kernel = opts->quality_kernel,
            .launch_seconds = opts->launch_ms / 1000.0,
            .passes = rays_per_pixel,
            .display = display,
            .optix_tracer = use_optix,
            .optix = optix,
            .denoise = opts->denoise};
        result = gpu::render(built->desc, gpu_options);
    }
    catch (const std::exception& error)
    {
        // The renderer throws when CUDA reports an error, maybe in the middle of the passes: erase the progress first
        std::println("\r{:30}\rError: {}", "", error.what());
        return 1;
    }
    std::print("\r{:30}\r", ""); // erase the progress line

    // Both times come from the renderer: the render() call (the BVH, copying the scene to the GPU and the image back,
    // all the passes; --denoise's work after them left out), and the passes alone, timed on the GPU
    std::println("Render time: {:.2f} s (GPU passes: {:.2f} s)", result.seconds, result.kernel_seconds);
    // Which kernel rendered (see the kernels in renderer.cu), and what decides how many of its threads the GPU keeps in
    // flight: the registers a thread takes (see blocks_per_multiprocessor)
    const gpu::kernel_info& kernel = result.kernel;
    if (!kernel.programs.empty())
    {
        // OptiX's programs: the graphics driver compiles them, and reports no registers or occupancy. The first 8 hex
        // digits of their PTX's SHA-256 name the programs that rendered, and the continuation stack is what replaced
        // the BVH's stack in each thread's memory.
        std::println("Kernel: {}, programs {} compiled by the graphics driver, {} bytes of continuation stack per "
                     "thread",
                     kernel.name, kernel.programs, kernel.local_bytes);
    }
    else
    {
        std::println("Kernel: {}, {} registers and {} bytes of local memory per thread, {} blocks of {} threads per "
                     "multiprocessor ({} of its {} threads)",
                     kernel.name, kernel.registers, kernel.local_bytes, kernel.blocks_per_multiprocessor,
                     kernel.threads_per_block, kernel.blocks_per_multiprocessor * kernel.threads_per_block,
                     kernel.max_threads_per_multiprocessor);
    }
    // A ray traced is a search for the closest hit, counted as on the CPU, so per sample it is the average number of
    // rays in a path: the camera ray and its bounces. The speed is that of the whole call, as the CPU's.
    const auto rays = static_cast<double>(result.rays);
    const double samples = static_cast<double>(width) * height * static_cast<double>(rays_per_pixel);
    std::println("Rays traced: {:.1f} million ({:.2f} per sample), {:.1f} million per second", rays / 1e6,
                 rays / samples, rays / result.seconds / 1e6);
    // Under -p nee and -p mis, the light samples' rays toward the lights, apart: they look for any blocker, not the
    // closest hit, so the rays above keep the meaning they have in the books' kernels
    if (built->desc.quality.combination != gpu::light_combination::bsdf_only)
    {
        const auto shadow_rays = static_cast<double>(result.shadow_rays);
        std::println("Shadow rays: {:.1f} million ({:.2f} per sample)", shadow_rays / 1e6, shadow_rays / samples);
    }

    // The pixels go into the CPU's image class, so the mean radiance below and the saved files are computed exactly
    // as in RayTracing
    const auto img = to_image(result.pixels, width, height);
    if (!img)
    {
        std::println("Error: {}", img.error());
        return 1;
    }
    std::println("Mean radiance (linear, before clamping): {:.6f}", img->mean_radiance());

    // A PNG of a tone map other than books holds the bytes the GPU made of the image (render_result::rgba), the bytes
    // the window shows: bytes_to_image gives the values image::save_png turns back into exactly those bytes. A PFM
    // keeps the linear values whatever the display, and the books' PNG is made from them as before.
    std::optional<image> toned;
    if (display.op != gpu::tone_map::books)
    {
        try
        {
            toned = bytes_to_image(result.rgba, width, height);
        }
        catch (const std::invalid_argument& error)
        {
            std::println("Error: {}", error.what());
            return 1;
        }
    }
    // An image that can't be saved doesn't stop the others: the render is done, and each file may still be useful
    bool all_saved = true;
    for (const std::filesystem::path& output : opts->outputs)
    {
        const image& source = toned && is_png(output) ? *toned : *img;
        if (const auto saved = source.save(output); !saved)
        {
            std::println("Error: {} ({})", saved.error(), to_utf8(output));
            all_saved = false;
            continue;
        }
        std::println("Image saved: {}", to_utf8(std::filesystem::absolute(output)));
    }
    // A denoise that failed after the passes (the denoiser's memory plan, its creation) leaves the plain files saved
    // above, as without --denoise, and no -denoised file: the error is said last, and the exit code is 1
    if (opts->denoise && !result.denoise_error.empty())
    {
        std::println("Error: --denoise: {}", result.denoise_error);
        return 1;
    }
    if (opts->denoise && !save_denoised(result, opts->outputs, width, height))
    {
        all_saved = false;
    }
    return all_saved ? 0 : 1;
}
