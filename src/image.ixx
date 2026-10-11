// Module image: an in-memory image, loading it from a file, and saving it as PNG or PFM.
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

    // The mean radiance: the average of all three channels of all pixels, (r + g + b) / 3 in average(). The renderers
    // print it after every render (see main.cpp), and ImageCompare prints it for both of its images.
    double mean_radiance() const
    {
        const color mean = average();
        return (mean.x + mean.y + mean.z) / 3;
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

    // Saves the image as PFM (Portable Float Map), the simplest format that keeps the tracer's own numbers: three
    // 32-bit floats per pixel, linear, before gamma correction and clamping. A PNG keeps 8 bits per channel after
    // both, so it can't tell a lamp of brightness 15 from plain white, or a difference of 0.1% from none; a PFM can,
    // which is why ImageCompare compares PFMs. HDR viewers such as tev open them.
    // The format: three lines of text, "PF", then the width and the height separated by a space, then a scale whose
    // sign gives the byte order of the floats (negative: little-endian); then the rows from the BOTTOM of the image to
    // its top, each from left to right.
    std::expected<void, std::string> save_pfm(const std::filesystem::path& path) const
    {
        // Binary mode matters for the header too: in text mode Windows would write each "\n" as "\r\n"
        std::ofstream file(path, std::ios::binary);
        if (!file)
        {
            return std::unexpected("could not open the file");
        }
        std::print(file, "PF\n{} {}\n-1.0\n", width_, height_); // C++23: std::print into a stream

        std::vector<std::uint32_t> row(static_cast<std::size_t>(width_) * 3);
        for (int y = height_ - 1; y >= 0; --y)
        {
            for (int x = 0; x < width_; ++x)
            {
                const color& pixel = (*this)[x, y];
                const std::size_t i = static_cast<std::size_t>(x) * 3;
                row[i] = to_little_endian(static_cast<float>(pixel.x));
                row[i + 1] = to_little_endian(static_cast<float>(pixel.y));
                row[i + 2] = to_little_endian(static_cast<float>(pixel.z));
            }
            file.write(reinterpret_cast<const char*>(row.data()),
                       static_cast<std::streamsize>(row.size() * sizeof(std::uint32_t)));
        }

        file.close();
        if (!file)
        {
            return std::unexpected("could not write the PFM");
        }
        return {};
    }

    // Saves the image in the format the file's extension names: .png or .pfm, in any letter case.
    std::expected<void, std::string> save(const std::filesystem::path& path) const
    {
        switch (format_of(path))
        {
        case file_format::png:
            return save_png(path);
        case file_format::pfm:
            return save_pfm(path);
        case file_format::unknown:
            break;
        }
        return std::unexpected("unknown image format: the file name must end in .png or .pfm");
    }

    // Whether save() knows the format of a file with this name. The programs check their output names with it before
    // rendering, so that a mistake is reported at once, not after a render that may take minutes.
    static bool can_save(const std::filesystem::path& path) { return format_of(path) != file_format::unknown; }

    // Loads a color PFM, such as those save_pfm writes.
    static std::expected<image, std::string> load_pfm(const std::filesystem::path& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            return std::unexpected("could not open the file");
        }

        std::string magic;
        int width = 0;
        int height = 0;
        double scale = 0;
        file >> magic >> width >> height >> scale;
        // A limit on the size keeps a damaged header from asking for terabytes of memory
        constexpr int max_side = 1 << 16;
        if (!file || magic != "PF" || width <= 0 || height <= 0 || width > max_side || height > max_side ||
            scale == 0)
        {
            return std::unexpected("not a color PFM file (it must start with PF, the width, the height and a scale)");
        }
        file.get(); // the single whitespace character that ends the header; the floats follow it
        const bool little_endian = scale < 0;

        std::vector<std::uint32_t> data(static_cast<std::size_t>(width) * height * 3);
        const auto bytes = static_cast<std::streamsize>(data.size() * sizeof(std::uint32_t));
        file.read(reinterpret_cast<char*>(data.data()), bytes);
        if (file.gcount() != bytes)
        {
            return std::unexpected("the file is shorter than its header says");
        }

        image img(width, height);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                // The file's rows go from the bottom up
                const std::size_t i = (static_cast<std::size_t>(height - 1 - y) * width + x) * 3;
                img[x, y] = color{from_pfm(data[i], little_endian), from_pfm(data[i + 1], little_endian),
                                  from_pfm(data[i + 2], little_endian)};
            }
        }
        return img;
    }

private:
    enum class file_format
    {
        png,
        pfm,
        unknown,
    };

    // The format a file name's extension names, in any letter case: one rule for save() and can_save()
    static file_format format_of(const std::filesystem::path& path)
    {
        const auto to_lower = [](char8_t c) {
            return u8'A' <= c && c <= u8'Z' ? static_cast<char8_t>(c - u8'A' + u8'a') : c;
        };
        // std::ranges::to (C++23) collects a range into a container: here the lowercase letters into a string
        const std::u8string extension =
            path.extension().u8string() | std::views::transform(to_lower) | std::ranges::to<std::u8string>();
        if (extension == u8".png")
        {
            return file_format::png;
        }
        if (extension == u8".pfm")
        {
            return file_format::pfm;
        }
        return file_format::unknown;
    }

    // A float's bits in little-endian byte order, the order of x86 and ARM processors: for them this is just
    // std::bit_cast (C++20), which reinterprets the float's 4 bytes as an integer. std::byteswap (C++23) reverses
    // them on the rare big-endian machine.
    static std::uint32_t to_little_endian(float value)
    {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        return std::endian::native == std::endian::little ? bits : std::byteswap(bits);
    }

    // The float stored in a PFM's 4 bytes, read as an integer, in the byte order the file's header declares
    static double from_pfm(std::uint32_t bits, bool little_endian)
    {
        const bool native_order = little_endian == (std::endian::native == std::endian::little);
        return std::bit_cast<float>(native_order ? bits : std::byteswap(bits));
    }

    int width_;
    int height_;
    std::vector<color> pixels_; // row by row, top to bottom: pixel (x, y) is at [y * width + x]
};
