import std;
import camera;
import hittable;
import image;
import material;
import sphere;
import vec3;

// Command-line options
struct options
{
    int image_width = 400;
    int samples_per_pixel = 100;
    std::filesystem::path output = "image.png";
    bool help = false;
};

constexpr std::string_view usage = R"(Usage: RayTracing [options]
  -w, --width N      image width in pixels (default 400)
  -s, --samples N    rays per pixel (default 100): fewer is faster but noisier
  -o, --output FILE  where to save the PNG (default image.png)
  -h, --help         show this help)";

// An integer greater than zero. std::from_chars (C++17) is strict and fast parsing: no exceptions
// and no locales, and the end pointer shows whether the whole string was parsed (not just "12" of "12abc").
std::optional<int> parse_positive_int(std::string_view text)
{
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value <= 0)
    {
        return std::nullopt;
    }
    return value;
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

        const bool is_width = arg == "-w" || arg == "--width";
        const bool is_samples = arg == "-s" || arg == "--samples";
        const bool is_output = arg == "-o" || arg == "--output";
        if (!is_width && !is_samples && !is_output)
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
            opts.output = value;
            continue;
        }
        const std::optional<int> number = parse_positive_int(value);
        if (!number)
        {
            return std::unexpected(std::format("{}: expected an integer greater than zero, not '{}'", arg, value));
        }
        if (is_width)
        {
            opts.image_width = *number;
        }
        else
        {
            opts.samples_per_pixel = *number;
        }
    }
    return opts;
}

// On Windows std::filesystem::path stores the path in UTF-16, and path.string() converts it to the process
// code page. The manifest (utf8.manifest) makes it UTF-8, but only on Windows 10 1903+; on older systems
// Cyrillic in the path would turn into mojibake. u8string() always returns UTF-8, which is exactly
// what std::print expects (we compile with /utf-8).
std::string to_utf8(const std::filesystem::path& path)
{
    const std::u8string utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

int main(int argc, char* argv[])
{
    const auto opts = parse_options(std::span(argv, static_cast<std::size_t>(argc)));
    if (!opts)
    {
        std::println("Error: {}\n\n{}", opts.error(), usage);
        return 1;
    }
    if (opts->help)
    {
        std::println("{}", usage);
        return 0;
    }

    // Scene: "ground", a matte sphere in the center, a hollow glass sphere on the left, a metal one on the right
    const auto material_ground = std::make_shared<lambertian>(color{0.8, 0.8, 0.0});
    const auto material_center = std::make_shared<lambertian>(color{0.1, 0.2, 0.5});
    const auto material_left = std::make_shared<dielectric>(1.50);
    const auto material_bubble = std::make_shared<dielectric>(1.00 / 1.50);
    const auto material_right = std::make_shared<metal>(color{0.8, 0.6, 0.2}, 1.0);

    hittable_list world;
    world.add(std::make_unique<sphere>(point3{0.0, -100.5, -1.0}, 100.0, material_ground));
    world.add(std::make_unique<sphere>(point3{0.0, 0.0, -1.2}, 0.5, material_center));
    world.add(std::make_unique<sphere>(point3{-1.0, 0.0, -1.0}, 0.5, material_left));
    world.add(std::make_unique<sphere>(point3{-1.0, 0.0, -1.0}, 0.4, material_bubble)); // air inside the glass
    world.add(std::make_unique<sphere>(point3{1.0, 0.0, -1.0}, 0.5, material_right));

    const camera cam({
        .aspect_ratio = 16.0 / 9.0,
        .image_width = opts->image_width,
        .samples_per_pixel = opts->samples_per_pixel,
        .max_depth = 50,
        .vfov = 20,
        .lookfrom = {-2, 2, 1},
        .lookat = {0, 0, -1},
        .vup = {0, 1, 0},
        .defocus_angle = 10.0,
        .focus_dist = 3.4,
    });

    const auto start = std::chrono::steady_clock::now();
    const image img = cam.render(world);
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::println("Render time: {:.1f} s", elapsed.count());

    if (const auto saved = img.save_png(opts->output); !saved)
    {
        std::println("Error: {} ({})", saved.error(), to_utf8(opts->output));
        return 1;
    }
    std::println("Image saved: {}", to_utf8(std::filesystem::absolute(opts->output)));
}
