import std;
import camera;
import hittable;
import image;
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
    // Scene: a small sphere and a huge "ground" sphere under it
    hittable_list world;
    world.add(std::make_unique<sphere>(point3{0, 0, -1}, 0.5));
    world.add(std::make_unique<sphere>(point3{0, -100.5, -1}, 100));

    const camera cam({.aspect_ratio = 16.0 / 9.0, .image_width = 400});
    const image img = cam.render(world);

    const std::filesystem::path output = "image.png";
    if (const auto saved = img.save_png(output); !saved)
    {
        std::println("Error: {} ({})", saved.error(), to_utf8(output));
        return 1;
    }
    std::println("Done: {}", to_utf8(std::filesystem::absolute(output)));
}
