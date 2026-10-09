import std;
import camera;
import hittable;
import image;
import material;
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

int main()
{
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
        .image_width = 400,
        .samples_per_pixel = 100,
        .max_depth = 50,
        .vfov = 20,
        .lookfrom = {-2, 2, 1},
        .lookat = {0, 0, -1},
        .vup = {0, 1, 0},
        .defocus_angle = 10.0,
        .focus_dist = 3.4,
    });
    const image img = cam.render(world);

    const std::filesystem::path output = "image.png";
    if (const auto saved = img.save_png(output); !saved)
    {
        std::println("Error: {} ({})", saved.error(), to_utf8(output));
        return 1;
    }
    std::println("Done: {}", to_utf8(std::filesystem::absolute(output)));
}
