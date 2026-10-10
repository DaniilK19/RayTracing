// Module image: an in-memory image, loading it from a file and saving it as PNG.
module;

// The global module fragment (between `module;` and `export module`) is the place for ordinary #includes.
// Everything included here stays internal to the module: whoever writes `import image;`
// won't see the stb library.
#define STB_IMAGE_WRITE_IMPLEMENTATION // the stb implementation is compiled right in this file
#define STBI_WRITE_NO_STDIO            // we open files ourselves, via std::ofstream
#include <stb_image_write.h>

#define STB_IMAGE_IMPLEMENTATION // the same for reading images
#define STBI_NO_STDIO            // we read the file ourselves, via std::ifstream
#include <stb_image.h>

export module image;

import std;
import interval;
import vec3;

// Gamma correction. The tracer computes physical ("linear") brightness, while image formats
// and monitors expect values in gamma space: without correction the image is too dark.
// The "gamma 2" approximation: take the square root of the linear brightness.
double linear_to_gamma(double linear)
{
    return linear > 0 ? std::sqrt(linear) : 0.0;
}

// Channel brightness 0.0–1.0 → byte 0–255: the range is split into 256 equal parts.
// Values outside the range (this happens after averaging samples) are clamped.
std::uint8_t to_byte(double linear)
{
    constexpr interval intensity{0.000, 0.999};
    return static_cast<std::uint8_t>(256 * intensity.clamp(linear_to_gamma(linear)));
}

export class image
{
public:
    image(int width, int height)
        : width_(width), height_(height), pixels_(static_cast<std::size_t>(width) * height)
    {
    }

    // Loads an image from a file: JPEG, PNG and other formats stb_image understands.
    // Files store colors in gamma space (sRGB), while the tracer works in linear space.
    // stbi_loadf converts them to linear right away by raising them to the power 2.2 (stb's default), roughly
    // the inverse of linear_to_gamma, which uses gamma 2.
    static std::expected<image, std::string> load(const std::filesystem::path& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            return std::unexpected("could not open the file");
        }
        const std::vector<char> bytes{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};

        // Ask for exactly 3 channels (R, G, B), even if the file has a different number, e.g. a PNG with transparency
        int width = 0;
        int height = 0;
        int channels_in_file = 0;
        float* const data = stbi_loadf_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()),
                                                   static_cast<int>(bytes.size()), &width, &height,
                                                   &channels_in_file, 3);
        if (!data)
        {
            return std::unexpected(std::format("could not read the image ({})", stbi_failure_reason()));
        }
        // The memory was allocated by a C library and must be returned with stbi_image_free. We hand this
        // to a std::unique_ptr with a custom "deleter": it will call that function however we leave this one.
        const std::unique_ptr<float, decltype(&stbi_image_free)> owner(data, &stbi_image_free);

        image img(width, height);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const float* const pixel = data + 3 * (static_cast<std::size_t>(y) * width + x);
                img[x, y] = color{pixel[0], pixel[1], pixel[2]};
            }
        }
        return img;
    }

    // inline so that calls from other modules (a texture reads pixels millions of times) get inlined
    inline int width() const { return width_; }
    inline int height() const { return height_; }

    // The average color of all pixels, in linear light and before any clamping: a channel value above 1 (a lamp,
    // a bright highlight) counts in full here, while the PNG clamps every channel at 255 (see to_byte), so a white
    // lamp of brightness 15 is stored as plain white.
    // std::ranges::fold_left (C++23) sums the range starting from zero; std::plus{} adds with our vec3 operator+.
    color average() const
    {
        return std::ranges::fold_left(pixels_, color{0, 0, 0}, std::plus{}) / static_cast<double>(pixels_.size());
    }

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
