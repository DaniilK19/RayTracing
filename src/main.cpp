import std;
import bvh;
import camera;
import cli;
import hittable;
import image;
import scenes;

// Command-line options. Each scene sets its own width and number of rays;
// std::optional tells "not given: take it from the scene" apart from any given value.
struct options
{
    std::string_view scene = default_scene;
    std::optional<int> image_width;
    std::optional<int> samples_per_pixel;
    std::optional<sampling_strategy> sampling;
    std::optional<std::uint32_t> seed;
    std::vector<std::filesystem::path> outputs; // one per -o; parse_options adds <scene name>.png if there is none
    bool help = false;
};

void print_usage()
{
    std::println("Usage: RayTracing [scene] [options]\n\nScenes:");
    for (const scene_entry& entry : scene_list)
    {
        std::println("  {:10} {}", entry.name, entry.description);
    }
    std::println(R"(  Default: {}.

Options:
  -w, --width N        image width in pixels; the height follows the scene's aspect ratio
                       (each scene has its own default)
  -s, --samples N      rays per pixel: fewer is faster but noisier
                       (each scene has its own default); rounded to the nearest square k²,
                       because the rays form a k×k grid in the pixel
  -p, --sampling NAME  how a diffuse bounce chooses the next direction ("p" as in pdf,
                       probability density function): one of the strategies below
      --seed N         which random numbers to draw, from 0 (the default) to {}:
                       another seed renders the same image with independent noise
  -o, --output FILE    where to save the image (default <scene>.png): a .png, or a .pfm that
                       keeps the linear values as floats, before gamma correction and clamping;
                       give it several times to save several files: -o out.png -o out.pfm
  -h, --help           show this help

Sampling (-p):)",
                 default_scene, std::numeric_limits<std::uint32_t>::max());
    for (const sampling_entry& entry : sampling_list)
    {
        // camera_settings{} holds the camera's defaults
        const bool is_default = entry.strategy == camera_settings{}.sampling;
        std::println("  {:10} {}{}", entry.name, entry.description, is_default ? " (default)" : "");
    }
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

        // An argument without a dash is the scene name
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
        const bool is_sampling = arg == "-p" || arg == "--sampling";
        const bool is_seed = arg == "--seed";
        const bool is_output = arg == "-o" || arg == "--output";
        if (!is_width && !is_samples && !is_sampling && !is_seed && !is_output)
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
            opts.seed = parse_number<std::uint32_t>(value);
            if (!opts.seed)
            {
                return std::unexpected(std::format("{}: expected an integer from 0 to {}, not '{}'", arg,
                                                   std::numeric_limits<std::uint32_t>::max(), value));
            }
            continue;
        }
        if (is_sampling)
        {
            const auto it = std::ranges::find(sampling_list, value, &sampling_entry::name);
            if (it == sampling_list.end())
            {
                return std::unexpected(std::format("unknown sampling strategy '{}'", value));
            }
            opts.sampling = it->strategy;
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
        opts.outputs.emplace_back(std::format("{}.png", opts.scene));
    }
    return opts;
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

#ifndef NDEBUG // NDEBUG is defined in Release and RelWithDebInfo
    std::println("This is a Debug build: rendering is about 10 times slower than in Release.\n"
                 "For a final image choose the x64 Release preset; for a quick draft add -s 16.");
#endif

    // The scene name was already checked in parse_options, so find_scene will find it
    scene s = find_scene(opts->scene)->build();
    s.camera.image_width = opts->image_width.value_or(s.camera.image_width);
    s.camera.samples_per_pixel = opts->samples_per_pixel.value_or(s.camera.samples_per_pixel);
    s.camera.sampling = opts->sampling.value_or(s.camera.sampling);
    s.camera.seed = opts->seed.value_or(s.camera.seed);

    // Put the scene's objects into a BVH: each ray is then tested against a few objects, not all of them
    const std::unique_ptr<hittable> world = build_bvh(std::move(s.world));
    const camera cam(s.camera);

    // The strategy the camera will really use: lights and mixture fall back to material in a scene without lights.
    // Every sampling_strategy has an entry in sampling_list (see camera.ixx), so find will find it.
    const sampling_strategy sampling = effective_sampling(s.camera.sampling, s.lights);
    const auto entry = std::ranges::find(sampling_list, sampling, &sampling_entry::strategy);
    std::println("Scene: {}, sampling: {}{}", opts->scene, entry->name,
                 sampling != s.camera.sampling ? " (no lights: the sky is the light)" : "");
    const auto start = std::chrono::steady_clock::now();
    const auto [img, rays] = cam.render(*world, s.lights); // a structured binding (C++17) names the result's members
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::println("Render time: {:.1f} s", elapsed.count());

    // How much work that was. Rays per sample is the average length of a path, the camera ray included; rays per second
    // is the renderer's speed.
    const camera_frame frame = cam.frame();
    const double samples =
        static_cast<double>(frame.image_width) * frame.image_height * frame.sqrt_spp * frame.sqrt_spp;
    const double million_rays = static_cast<double>(rays) / 1e6;
    std::println("Rays traced: {:.1f} million ({:.2f} per sample), {:.1f} million per second", million_rays,
                 static_cast<double>(rays) / samples, million_rays / elapsed.count());

    // Not in the book: a diagnostic for comparing ways of sampling. Every pixel is a Monte Carlo estimate;
    // an unbiased one may be noisier or less noisy, but its expected value is the same, so the image mean
    // of every unbiased method converges to the same number as the rays per pixel grow. The PNG can't show
    // this: it keeps 8 bits per channel and nothing brighter than white, so a lamp of brightness 15 is just white.
    std::println("Mean radiance (linear, before clamping): {:.6f}", img.mean_radiance());

    // Each file is tried even if an earlier one failed: one bad path shouldn't throw the whole render away
    int exit_code = 0;
    for (const std::filesystem::path& output : opts->outputs)
    {
        if (const auto saved = img.save(output); !saved)
        {
            std::println("Error: {} ({})", saved.error(), to_utf8(output));
            exit_code = 1;
            continue;
        }
        std::println("Image saved: {}", to_utf8(std::filesystem::absolute(output)));
    }
    return exit_code;
}
