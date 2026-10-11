// Module cli: small helpers the programs share for their command lines.
export module cli;

import std;
import image;

// The whole text as a number of type T, or std::nullopt. std::from_chars (C++17) is strict and fast parsing: no
// exceptions and no locales, and the end pointer shows whether the whole string was parsed (not just "12" of "12abc").
// For an unsigned T it rejects a minus sign, and for any T a number that doesn't fit.
export template <typename T>
std::optional<T> parse_number(std::string_view text)
{
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
    {
        return std::nullopt;
    }
    return value;
}

// An integer greater than zero
export std::optional<int> parse_positive_int(std::string_view text)
{
    const std::optional<int> value = parse_number<int>(text);
    return value && *value > 0 ? value : std::nullopt;
}

// On Windows std::filesystem::path stores the path in UTF-16, and path.string() converts it to the process
// code page. The manifest (utf8.manifest) makes it UTF-8, but only on Windows 10 1903+; on older systems
// Cyrillic in the path would turn into mojibake. u8string() always returns UTF-8, which is exactly
// what std::print expects (we compile with /utf-8).
export std::string to_utf8(const std::filesystem::path& path)
{
    const std::u8string utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

// Looks for mistakes in an output file name before anything is rendered, so that they are reported at once, not after
// a render that may take minutes: a format image::save doesn't know, or a folder that doesn't exist. Whether the file
// can really be written is known only when it is written.
export std::expected<void, std::string> check_output(const std::filesystem::path& path)
{
    if (!image::can_save(path))
    {
        return std::unexpected(std::format("expected a file name ending in .png or .pfm, not '{}'", to_utf8(path)));
    }
    // With a std::error_code, is_directory doesn't throw: on an error it returns false, which counts as no folder
    std::error_code error;
    if (const std::filesystem::path folder = path.parent_path();
        !folder.empty() && !std::filesystem::is_directory(folder, error))
    {
        return std::unexpected(std::format("there is no folder '{}'", to_utf8(folder)));
    }
    return {};
}
