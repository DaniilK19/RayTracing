import std;
import image;

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
    constexpr int width = 256;
    constexpr int height = 256;

    // Channel brightness 0.0–1.0 → byte 0–255. The factor is 255.999, not 255: otherwise only exactly
    // 1.0 would get 255, and the brightest shade would almost never appear.
    const auto to_byte = [](double value) { return static_cast<std::uint8_t>(255.999 * value); };

    image img(width, height);

    for (int y = 0; y < height; ++y)
    {
        std::print("\rRows remaining: {:3}", height - y);
        for (int x = 0; x < width; ++x)
        {
            const double r = static_cast<double>(x) / (width - 1);  // left to right: 0 → 1
            const double g = static_cast<double>(y) / (height - 1); // top to bottom: 0 → 1
            const double b = 0.0;
            img[x, y] = {to_byte(r), to_byte(g), to_byte(b)};
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
