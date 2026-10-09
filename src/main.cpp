import std;
import hittable;
import image;
import interval;
import ray;
import sphere;
import vec3;

// On Windows std::filesystem::path stores the path in UTF-16, and path.string() converts it
// to the legacy Windows code page: Cyrillic in the path would turn into mojibake.
// u8string() returns UTF-8, which is exactly what std::print expects (we compile with /utf-8).
std::string to_utf8(const std::filesystem::path& path)
{
    const std::u8string utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

color ray_color(const ray& r, const hittable& world)
{
    if (const auto rec = world.hit(r, interval{0, infinity}))
    {
        return 0.5 * (rec->normal + color{1, 1, 1}); // normal components -1..1 → color 0..1
    }

    // Sky: a smooth blend from white (bottom) to blue (top) by the height of the ray direction.
    const vec3 unit_direction = unit_vector(r.direction);
    const double a = 0.5 * (unit_direction.y + 1.0); // y from -1 to 1 → a from 0 to 1
    return (1.0 - a) * color{1.0, 1.0, 1.0} + a * color{0.5, 0.7, 1.0};
}

int main()
{
    // Image
    constexpr double aspect_ratio = 16.0 / 9.0;
    constexpr int image_width = 400;
    constexpr int image_height = std::max(1, static_cast<int>(image_width / aspect_ratio));

    // The camera sits at the origin looking along -z. At distance focal_length in front of it stands
    // the viewport: a window with a ray shot through each of its pixels.
    // Everything is computed at compile time: vec3 operations are constexpr.
    constexpr double focal_length = 1.0;
    constexpr double viewport_height = 2.0;
    constexpr double viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);
    constexpr point3 camera_center{0, 0, 0};

    // Vectors along the window's top edge (rightward) and left edge (downward: image rows go top to bottom)
    constexpr vec3 viewport_u{viewport_width, 0, 0};
    constexpr vec3 viewport_v{0, -viewport_height, 0};

    // Step between neighboring pixel centers
    constexpr vec3 pixel_delta_u = viewport_u / image_width;
    constexpr vec3 pixel_delta_v = viewport_v / image_height;

    // Upper-left corner of the window and the center of pixel (0, 0)
    constexpr point3 viewport_upper_left =
        camera_center - vec3{0, 0, focal_length} - viewport_u / 2 - viewport_v / 2;
    constexpr point3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Scene: a small sphere and a huge "ground" sphere under it
    hittable_list world;
    world.add(std::make_unique<sphere>(point3{0, 0, -1}, 0.5));
    world.add(std::make_unique<sphere>(point3{0, -100.5, -1}, 100));

    image img(image_width, image_height);

    for (int j = 0; j < image_height; ++j)
    {
        std::print("\rRows remaining: {:3}", image_height - j);
        for (int i = 0; i < image_width; ++i)
        {
            const point3 pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            const ray r{camera_center, pixel_center - camera_center};
            img[i, j] = ray_color(r, world);
        }
    }

    const std::filesystem::path output = "image.png";
    if (const auto saved = img.save_png(output); !saved)
    {
        std::println("\nError: {} ({})", saved.error(), to_utf8(output));
        return 1;
    }
    std::println("\rDone: {}", to_utf8(std::filesystem::absolute(output)));
}
