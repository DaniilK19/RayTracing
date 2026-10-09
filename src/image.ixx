// Module image: an in-memory image and saving it as PNG.
module;

// The global module fragment (between `module;` and `export module`) is the place for ordinary #includes.
// Everything included here stays internal to the module: whoever writes `import image;`
// won't see the stb library.
#define STB_IMAGE_WRITE_IMPLEMENTATION // the stb implementation is compiled right in this file
#define STBI_WRITE_NO_STDIO            // we open files ourselves, via std::ofstream
#include <stb_image_write.h>

export module image;

import std;
import interval;
import vec3;

// Channel brightness 0.0–1.0 → byte 0–255: the range is split into 256 equal parts.
// Values outside the range (this happens after averaging samples) are clamped.
std::uint8_t to_byte(double value)
{
    constexpr interval intensity{0.000, 0.999};
    return static_cast<std::uint8_t>(256 * intensity.clamp(value));
}

export class image
{
public:
    image(int width, int height)
        : width_(width), height_(height), pixels_(static_cast<std::size_t>(width) * height)
    {
    }

    int width() const { return width_; }
    int height() const { return height_; }

    // Pixel color: img[x, y]. Multiple arguments to operator[] are allowed since C++23.
    // `this auto& self` is C++23 too ("deducing this"): one function instead of a const/non-const pair.
    // For a const image, self is a const image&, and a const color& is returned.
    auto& operator[](this auto& self, int x, int y)
    {
        return self.pixels_[static_cast<std::size_t>(y) * self.width_ + x];
    }

    // Saves the image as PNG. The error is returned as a value: std::expected (C++23)
    // holds either "success" (void) or an error description (std::string).
    std::expected<void, std::string> save_png(const std::filesystem::path& path) const
    {
        // PNG stores one byte per channel: convert the colors into a contiguous byte array R,G,B,R,G,B,...
        std::vector<std::uint8_t> bytes;
        bytes.reserve(pixels_.size() * 3);
        for (const color& pixel : pixels_)
        {
            bytes.push_back(to_byte(pixel.x));
            bytes.push_back(to_byte(pixel.y));
            bytes.push_back(to_byte(pixel.z));
        }

        std::ofstream file(path, std::ios::binary);
        if (!file)
        {
            return std::unexpected("could not open the file");
        }

        // stb compresses the image to PNG and hands the finished bytes in chunks to this function,
        // which appends them to the file. context is our file, passed through void*.
        auto write_chunk = [](void* context, void* data, int size) {
            static_cast<std::ofstream*>(context)->write(static_cast<const char*>(data), size);
        };
        const bool encoded =
            stbi_write_png_to_func(write_chunk, &file, width_, height_, 3, bytes.data(), width_ * 3) != 0;

        file.close();
        if (!encoded || !file)
        {
            return std::unexpected("could not write the PNG");
        }
        return {};
    }

private:
    int width_;
    int height_;
    std::vector<color> pixels_; // row by row, top to bottom: pixel (x, y) is at [y * width + x]
};
