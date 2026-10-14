// Module view_options: the options that choose what a GPU program's camera sees, shared by the GPU programs. Each scene
// has its own view (scenes.ixx); these options change it: the image's exact height, where the camera stands and looks,
// its field of view and its lens. They change only the camera's settings, so the CPU's own camera computes the pixel
// grid from them, as for the scene's view.
export module view_options;

import std;
import camera;
import cli;
import vec3;

// What the options ask for; what is empty stays the scene's own
export struct view_options
{
    std::optional<int> image_height;
    std::optional<point3> lookfrom;
    std::optional<point3> lookat;
    std::optional<double> vfov;
    std::optional<double> defocus_angle;
    std::optional<double> focus_dist;
};

// The renderer's limit (see validate in gpu/renderer.cu): a launch covers at most 65535 rows of blocks of 8 rows
export constexpr int max_image_height = 65535 * 8;

// The camera makes the height int(width / aspect_ratio) (see camera), so an aspect ratio of width / height would give
// height - 1 rows for some sizes: width / height rounds to a double that can be a hair larger than the exact ratio,
// and the division back then falls a hair below height. With half a row more, the two roundings, each within 2^-53 of
// the exact value, keep the quotient within (height + 0.5)·2^-52 of height + 0.5, far inside (height, height + 1)
// for every height up to max_image_height and any width, so int() gives exactly height (also checked by brute force
// for every width and height up to 8192).
export double aspect_for_height(int width, int height)
{
    return static_cast<double>(width) / (height + 0.5);
}

export bool is_view_option(std::string_view arg)
{
    return arg == "--height" || arg == "--lookfrom" || arg == "--lookat" || arg == "--vfov" ||
           arg == "--defocus-angle" || arg == "--focus-dist";
}

namespace
{
// A finite number, or std::nullopt (std::from_chars also reads "inf" and "nan")
std::optional<double> parse_finite(std::string_view text)
{
    const std::optional<double> value = parse_number<double>(text);
    return value && std::isfinite(*value) ? value : std::nullopt;
}

// "X,Y,Z": three finite numbers
std::optional<point3> parse_point(std::string_view text)
{
    std::array<double, 3> xyz{};
    std::size_t start = 0;
    for (std::size_t k = 0; k < 3; ++k)
    {
        const std::size_t comma = k < 2 ? text.find(',', start) : text.size();
        if (comma == std::string_view::npos)
        {
            return std::nullopt;
        }
        const std::optional<double> value = parse_finite(text.substr(start, comma - start));
        if (!value)
        {
            return std::nullopt;
        }
        xyz[k] = *value;
        start = comma + 1;
    }
    return point3{xyz[0], xyz[1], xyz[2]};
}

bool finite(const vec3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// The lengths a camera may have in the scene, and the farthest its center may stand from the origin. Far wider than
// any scene needs (the books' are between 0.1 and 1000), and narrow enough that the GPU's 32-bit floats (largest
// about 3.4·10^38, smallest normal about 1.2·10^-38) hold every vector the kernel builds from them.
constexpr double smallest_length = 1e-20;
constexpr double largest_length = 1e20;

bool in_range(const vec3& v)
{
    return finite(v) && v.length() <= largest_length;
}

bool in_range_and_not_tiny(const vec3& v)
{
    return in_range(v) && v.length() >= smallest_length;
}
}

// Reads the value of one of the view options into `view`, or says what is wrong with it
export std::expected<void, std::string> parse_view_option(std::string_view arg, std::string_view value,
                                                          view_options& view)
{
    if (arg == "--height")
    {
        const std::optional<int> height = parse_positive_int(value);
        if (!height || *height > max_image_height)
        {
            return std::unexpected(
                std::format("{}: expected an integer from 1 to {}, not '{}'", arg, max_image_height, value));
        }
        view.image_height = height;
        return {};
    }
    if (arg == "--lookfrom" || arg == "--lookat")
    {
        const std::optional<point3> point = parse_point(value);
        if (!point)
        {
            return std::unexpected(std::format("{}: expected a point as three numbers X,Y,Z, not '{}'", arg, value));
        }
        (arg == "--lookfrom" ? view.lookfrom : view.lookat) = point;
        return {};
    }
    const std::optional<double> number = parse_finite(value);
    if (arg == "--vfov")
    {
        if (!number || !(*number > 0 && *number < 180))
        {
            return std::unexpected(
                std::format("{}: expected an angle in degrees between 0 and 180, not '{}'", arg, value));
        }
        view.vfov = number;
        return {};
    }
    if (arg == "--defocus-angle")
    {
        if (!number || !(*number >= 0 && *number < 180))
        {
            return std::unexpected(
                std::format("{}: expected an angle in degrees from 0 to below 180, not '{}'", arg, value));
        }
        view.defocus_angle = number;
        return {};
    }
    if (arg == "--focus-dist")
    {
        if (!number || !(*number > 0))
        {
            return std::unexpected(std::format("{}: expected a distance greater than zero, not '{}'", arg, value));
        }
        view.focus_dist = number;
        return {};
    }
    return std::unexpected(std::format("unknown view option '{}'", arg));
}

// Whether the camera can render the view in these settings, or what is wrong with it. Every way a view reaches the
// camera goes through this check: the options (apply below) and, in the window, the camera's moves. It checks the
// settings' own ranges and then the frame the camera computes from them, which must hold usable numbers: a view can
// pass every range and still be too large or too small to compute (a focus distance of 10^300 or 10^-30).
export std::expected<void, std::string> check_view(const camera_settings& settings)
{
    if (!(settings.image_width >= 1))
    {
        return std::unexpected("the image width must be greater than zero");
    }
    if (!(std::isfinite(settings.aspect_ratio) && settings.aspect_ratio > 0))
    {
        return std::unexpected("the aspect ratio must be greater than zero");
    }
    if (!(settings.vfov > 0 && settings.vfov < 180))
    {
        return std::unexpected("the vertical field of view must be between 0 and 180 degrees");
    }
    if (!(settings.defocus_angle >= 0 && settings.defocus_angle < 180))
    {
        return std::unexpected("the defocus angle must be from 0 to below 180 degrees");
    }
    if (!(std::isfinite(settings.focus_dist) && settings.focus_dist > 0))
    {
        return std::unexpected("the focus distance must be greater than zero");
    }
    if (!finite(settings.lookfrom) || !finite(settings.lookat))
    {
        return std::unexpected("lookfrom and lookat must be finite points");
    }
    // The camera's frame needs a direction to look in and an "up" that isn't along it (see camera)
    const vec3 back = settings.lookfrom - settings.lookat;
    if (back.x == 0 && back.y == 0 && back.z == 0)
    {
        return std::unexpected("lookfrom and lookat are the same point: the camera has no direction to look in");
    }
    if (!in_range_and_not_tiny(back))
    {
        return std::unexpected(
            std::format("lookfrom and lookat must be between {} and {} apart", smallest_length, largest_length));
    }
    if (!(cross(settings.vup, unit_vector(back)).length() >= 1e-9))
    {
        return std::unexpected("the camera looks straight up or down: lookfrom and lookat must differ in more than y");
    }
    // The focus distance is also the length of the camera's rays: the books' camera aims them at the image plane, which
    // stands on the focus plane, and a hit counts only from 0.001 of a ray's length on (see ray_color). So nothing
    // nearer than a thousandth of the focus distance is seen: a focus distance far beyond the scene clips it away, and
    // one far below it makes the rays so short that quads seen at a slant miss them (see quad::hit). It is kept within
    // a thousand times the distance to lookat either way, as every scene's own view is.
    const double distance = back.length();
    if (!(settings.focus_dist >= distance / 1000 && settings.focus_dist <= distance * 1000))
    {
        return std::unexpected(std::format("the focus distance must be from {:.4g} to {:.4g}: from a thousandth to a "
                                           "thousand times the distance from lookfrom to lookat",
                                           distance / 1000, distance * 1000));
    }
    const camera_frame frame = camera(settings).frame();
    if (frame.image_height > max_image_height)
    {
        return std::unexpected(std::format("the image would be {} pixels high, more than the {} the renderer allows",
                                           frame.image_height, max_image_height));
    }
    const bool usable = in_range(frame.center) && in_range_and_not_tiny(frame.pixel00_loc - frame.center) &&
                        in_range_and_not_tiny(frame.pixel_delta_u) && in_range_and_not_tiny(frame.pixel_delta_v) &&
                        in_range(frame.defocus_disk_u) && in_range(frame.defocus_disk_v);
    if (!usable)
    {
        return std::unexpected(std::format("the view's numbers are out of range: the camera's position and the "
                                           "lengths of its pixels and its lens must lie between {} and {}",
                                           smallest_length, largest_length));
    }
    return {};
}

// Puts the view into the camera's settings, once their image width is final, or says why the camera can't take it
// (and leaves the settings as they were)
export std::expected<void, std::string> apply(const view_options& view, camera_settings& settings)
{
    if (view.image_height && !(*view.image_height >= 1 && *view.image_height <= max_image_height))
    {
        return std::unexpected(std::format("the image height must be from 1 to {}", max_image_height));
    }
    camera_settings changed = settings;
    if (view.image_height)
    {
        changed.aspect_ratio = aspect_for_height(changed.image_width, *view.image_height);
    }
    changed.lookfrom = view.lookfrom.value_or(changed.lookfrom);
    changed.lookat = view.lookat.value_or(changed.lookat);
    changed.vfov = view.vfov.value_or(changed.vfov);
    changed.defocus_angle = view.defocus_angle.value_or(changed.defocus_angle);
    changed.focus_dist = view.focus_dist.value_or(changed.focus_dist);
    if (const auto checked = check_view(changed); !checked)
    {
        return checked;
    }
    settings = changed;
    return {};
}

// The view as options that give it back exactly: every number in the shortest form that reads back as the same double
// (std::format's "{}"). So a printed view renders the same camera bit for bit, and with the same scene and the same
// other options (samples, strategy, seed, precision) the same image. The points are quoted: Windows PowerShell reads
// an unquoted 12.613500118255617,2,3 as a list of numbers and passes it on rounded to 15 digits, a slightly different
// camera; quoted, every shell passes the text as it is.
export std::string format_view(const camera_settings& settings, int image_height)
{
    const auto point = [](const point3& p) { return std::format("\"{},{},{}\"", p.x, p.y, p.z); };
    return std::format("-w {} --height {} --lookfrom {} --lookat {} --vfov {} --defocus-angle {} --focus-dist {}",
                       settings.image_width, image_height, point(settings.lookfrom), point(settings.lookat),
                       settings.vfov, settings.defocus_angle, settings.focus_dist);
}

// The options' lines in a program's help, right after -w
export constexpr std::string_view view_usage =
    R"(      --height N       image height in pixels, exactly (default: from the width and the scene's
                       aspect ratio)
      --lookfrom X,Y,Z
                       where the camera stands (each scene has its own view: the View: line of a
                       render prints it as these options, ready to be changed)
      --lookat X,Y,Z   the point it looks at
      --vfov D         vertical field of view in degrees, between 0 and 180
      --defocus-angle D
                       the lens's angle in degrees, from 0 to below 180: 0 renders everything
                       sharp, a larger one blurs what is nearer or farther than the focus distance
      --focus-dist D   the distance from the camera to the plane in focus, from a thousandth to a
                       thousand times the distance to --lookat; the camera's rays are that long,
                       and nothing nearer than a thousandth of it is seen)";
