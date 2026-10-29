// Module tone_options: how the GPU programs show an image's light, their options --tonemap and --exposure, and the
// names the window's keys, scripts and recordings use. The renderer applies a tone map on the GPU
// (gpu::session::display, tonemap.cu); this module names the tone maps, reads the options, and works out what a scene
// shows: what the options (or the window's keys) choose, and the scene's own display (its entry in gpu_scenes.ixx) for
// what they leave open.
module;

// The renderer's display types, before `import std;` (see gpu_scene.ixx)
#include "gpu/renderer.h"

export module tone_options;

import std;
import cli;

// The renderer's enum under this module's name: the window's scripts (viewer_script.ixx), which read and write text
// only and include no renderer.h, name tone maps by it
export using tone_map = gpu::tone_map;

export struct tone_map_entry
{
    std::string_view name;
    gpu::tone_map value;
};

// Every tone map, in the order T steps through them in the window
export constexpr std::array tone_map_list{
    tone_map_entry{"books", gpu::tone_map::books},
    tone_map_entry{"none", gpu::tone_map::none},
    tone_map_entry{"agx", gpu::tone_map::agx},
    tone_map_entry{"neutral", gpu::tone_map::neutral},
};

// The entry of a name, or nullptr
export constexpr const tone_map_entry* find_tone_map(std::string_view name)
{
    const auto it = std::ranges::find(tone_map_list, name, &tone_map_entry::name);
    return it == tone_map_list.end() ? nullptr : &*it;
}

// Every tone map has an entry, so this finds one
export constexpr std::string_view tone_map_name(gpu::tone_map op)
{
    return std::ranges::find(tone_map_list, op, &tone_map_entry::value)->name;
}

// The tone map `step` places on in the list, round and round: the window's T (+1) and Shift+T (-1)
export constexpr gpu::tone_map step_tone_map(gpu::tone_map op, int step)
{
    const auto count = static_cast<std::ptrdiff_t>(tone_map_list.size());
    const std::ptrdiff_t at = std::ranges::find(tone_map_list, op, &tone_map_entry::value) - tone_map_list.begin();
    return tone_map_list[static_cast<std::size_t>(((at + step) % count + count) % count)].value;
}

// The names, for messages: "books, none, agx or neutral"
export std::string tone_map_names()
{
    std::string text;
    for (std::size_t i = 0; i < tone_map_list.size(); ++i)
    {
        text += std::format("{}{}", i == 0 ? "" : i + 1 == tone_map_list.size() ? " or " : ", ", tone_map_list[i].name);
    }
    return text;
}

// The exposure's range in stops, either way: the renderer's (renderer.h), which refuses more
export constexpr double max_exposure = gpu::max_exposure;

// A number of stops: finite and from -max_exposure to max_exposure. A leading + is allowed, as the window writes
// exposures (+0.5 EV), and -0 is read as 0, so that it is printed as 0 again.
export std::optional<double> parse_exposure(std::string_view text)
{
    if (text.starts_with('+'))
    {
        text.remove_prefix(1);
        if (text.starts_with('-'))
        {
            return std::nullopt;
        }
    }
    const std::optional<double> ev = parse_number<double>(text);
    if (!ev || !std::isfinite(*ev) || std::fabs(*ev) > max_exposure)
    {
        return std::nullopt;
    }
    return *ev == 0 ? 0.0 : *ev;
}

// An exposure as the window's numbers and RayTracingGPU's Display line show it: +0.5 EV, -2 EV
export std::string format_exposure(double ev)
{
    return std::format("{:+g} EV", ev);
}

// What the options ask for; what is empty is the scene's own
export struct display_choice
{
    std::optional<gpu::tone_map> op;
    std::optional<double> exposure;
};

// --tonemap's value, or the error as the programs print it after "Error: "
export std::expected<gpu::tone_map, std::string> tone_map_option(std::string_view value)
{
    if (const tone_map_entry* entry = find_tone_map(value))
    {
        return entry->value;
    }
    return std::unexpected(std::format("unknown tone map '{}'", value));
}

// --exposure's value (`arg` is the option as given), or the error as the programs print it after "Error: "
export std::expected<double, std::string> exposure_option(std::string_view arg, std::string_view value)
{
    if (const std::optional<double> ev = parse_exposure(value))
    {
        return *ev;
    }
    return std::unexpected(std::format("{}: expected a finite number from {} to {}, not '{}'", arg, -max_exposure,
                                       max_exposure, value));
}

// The display a scene shows: each choice made, else the scene's own (the books' bytes for the books' scenes). The
// books' bytes have no exposure: with books it is 0, and an exposure chosen for the other tone maps waits for them
// (check_display refuses one given with books on a command line).
export gpu::display_options effective_display(const display_choice& chosen, const gpu::display_options& scene)
{
    const gpu::tone_map op = chosen.op.value_or(scene.op);
    if (op == gpu::tone_map::books)
    {
        return {};
    }
    return {op, chosen.exposure.value_or(scene.exposure)};
}

// What a scene can't show with the options, as the programs say it after "Error: ", before anything is rendered: an
// exposure for the books' bytes, and a tone map beyond the books in float64 (what is new computes in float only)
export std::expected<void, std::string> check_display(const display_choice& chosen, const gpu::display_options& scene,
                                                      bool fp64)
{
    const gpu::display_options shown = effective_display(chosen, scene);
    if (shown.op == gpu::tone_map::books && chosen.exposure.value_or(0) != 0)
    {
        return std::unexpected("--exposure needs a tone map: books keeps the books' bytes");
    }
    if (shown.op != gpu::tone_map::books && fp64)
    {
        return std::unexpected(std::format("--tonemap {} is a new display transform, which works in float only: "
                                           "drop --fp64",
                                           tone_map_name(shown.op)));
    }
    return {};
}

// Whether a printed command or a recording names the display: unless both the display shown and the scene's own are
// the books' bytes. So the books' scenes' commands and recordings stay as they were, and a command names its display
// wherever the scene's default could show another one: it renders its image whatever the defaults.
export bool names_display(const gpu::display_options& shown, const gpu::display_options& scene)
{
    return shown != gpu::display_options{} || scene != gpu::display_options{};
}

// A file that takes the display's bytes when the tone map isn't the books': a PNG, by its extension in any letter case
// (as image::save reads it). A PFM keeps the linear values whatever the display.
export bool is_png(const std::filesystem::path& file)
{
    const auto to_lower = [](char8_t c) {
        return u8'A' <= c && c <= u8'Z' ? static_cast<char8_t>(c - u8'A' + u8'a') : c;
    };
    return (file.extension().u8string() | std::views::transform(to_lower) | std::ranges::to<std::u8string>()) ==
           u8".png";
}

// The two options' lines in RayTracingGPU's help, among its options (the window's help says it in short)
export std::string tone_usage()
{
    return std::format(
        "      --tonemap NAME   the PNG's display transform: books (the books' square root, clamped: their\n"
        "                       scenes' default), none (the sRGB curve, clipped), agx (the default of the\n"
        "                       scenes beyond the books) or neutral; a .pfm stays linear\n"
        "      --exposure EV    the light times 2^EV before the tone map, from {} to {} (default 0, or the\n"
        "                       scene's); not with books",
        -max_exposure, max_exposure);
}
