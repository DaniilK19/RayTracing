// Module viewer_actions: what the window can be asked to do by name and value, as text. The keys step: Tab goes to the
// next scene, M to the next strategy, `.` raises the exposure half a stop. An action says where to go: this scene, this
// strategy, this exposure. An input file's `action` line (--input, see viewer_script) asks for one, and the window
// performs each in one place (viewer::apply in main_viewer.cpp), with the checks and the notes of the key that leads to
// the same state. Nothing here knows SDL or the GPU: the module only reads and writes the text.
export module viewer_actions;

import std;
import cli;
import strategy_options;
import tone_options;

// --- The actions ---------------------------------------------------------------------------------------------------
//
// One action a line, in the words of the camera scripts where a script has the setting too:
//   scene NAME | scene "FILE.glb"   that scene, or a glTF 2.0 model's file (quoted: a path may hold spaces), at its
//                                   own view; the strategy named before (sampling, M, -p) goes along where the scene
//                                   takes it, as with Tab
//   sampling NAME | sampling scene  that strategy, or the scene's own, at the same view (M)
//   precision float | double        (F2)
//   optix on | off                  trace with OptiX, or with the command line's kernel (O)
//   denoise on | off                the denoised image (N)
//   tonemap books|none|agx|neutral  the display's tone map (T)
//   exposure EV                     the exposure, from -24 to 24 stops; not with books (, . 0)
//   samples N                       refine to N samples per pixel, rounded up to whole rounds of strata (Enter)
//   pause on | off                  (Space)
//   vfov D | defocus-angle D | focus-dist D
//                                   the field of view in degrees (1 to 160), the lens's angle (0 to 30), the focus
//                                   distance (kept within the range a view allows): the camera stays where it is
//   reset-view                      back to the scene's starting view (R)
//   save | copy                     save the image and print its command (P); copy the command (C)
//   hud on | off | help on | off    the numbers on screen (I), the controls over the image (H)
// An action the window can't perform leaves it as it was and says why, as the key does.

export struct show_scene
{
    std::string name;
    bool operator==(const show_scene&) const = default;
};
export struct use_sampling
{
    std::optional<strategy> sampling; // empty: the scene's own
    bool operator==(const use_sampling&) const = default;
};
export struct use_precision
{
    bool fp64;
    bool operator==(const use_precision&) const = default;
};
export struct use_optix
{
    bool on;
    bool operator==(const use_optix&) const = default;
};
export struct use_denoise
{
    bool on;
    bool operator==(const use_denoise&) const = default;
};
export struct use_tone_map
{
    tone_map op;
    bool operator==(const use_tone_map&) const = default;
};
export struct use_exposure
{
    double exposure;
    bool operator==(const use_exposure&) const = default;
};
export struct use_samples
{
    std::int64_t samples;
    bool operator==(const use_samples&) const = default;
};
export struct use_pause
{
    bool on;
    bool operator==(const use_pause&) const = default;
};
export struct use_vfov
{
    double degrees;
    bool operator==(const use_vfov&) const = default;
};
export struct use_lens
{
    double degrees;
    bool operator==(const use_lens&) const = default;
};
export struct use_focus
{
    double distance;
    bool operator==(const use_focus&) const = default;
};
export struct reset_view
{
    bool operator==(const reset_view&) const = default;
};
export struct save_image
{
    bool operator==(const save_image&) const = default;
};
export struct copy_command
{
    bool operator==(const copy_command&) const = default;
};
export struct show_hud
{
    bool on;
    bool operator==(const show_hud&) const = default;
};
export struct show_help
{
    bool on;
    bool operator==(const show_help&) const = default;
};

export using action =
    std::variant<show_scene, use_sampling, use_precision, use_optix, use_denoise, use_tone_map, use_exposure,
                 use_samples, use_pause, use_vfov, use_lens, use_focus, reset_view, save_image, copy_command, show_hud,
                 show_help>;

// The most samples an action may ask for: the renderer numbers a view's passes with ints
export constexpr std::int64_t max_action_samples = std::numeric_limits<std::int32_t>::max();

// An action from its words (a line's words after `action`, a quoted word without its quotes), or what is wrong with
// them
export std::expected<action, std::string> parse_action(std::span<const std::string_view> w)
{
    if (w.empty())
    {
        return std::unexpected("an action needs its name");
    }
    const std::string_view verb = w[0];
    const auto on_or_off = [&]() -> std::optional<bool> {
        if (w.size() == 2 && (w[1] == "on" || w[1] == "off"))
        {
            return w[1] == "on";
        }
        return std::nullopt;
    };
    const auto number = [&]() -> std::optional<double> {
        const std::optional<double> value = w.size() == 2 ? parse_number<double>(w[1]) : std::nullopt;
        return value && std::isfinite(*value) ? value : std::nullopt;
    };
    const auto alone = [&](action a) -> std::expected<action, std::string> {
        if (w.size() != 1)
        {
            return std::unexpected(std::format("{} takes no values", verb));
        }
        return a;
    };
    if (verb == "scene")
    {
        if (w.size() != 2 || w[1].empty())
        {
            return std::unexpected("scene needs a scene's name or a model's file");
        }
        return show_scene{std::string(w[1])};
    }
    if (verb == "sampling")
    {
        if (w.size() == 2 && w[1] == "scene")
        {
            return use_sampling{};
        }
        const strategy_entry* entry = w.size() == 2 ? find_strategy(w[1]) : nullptr;
        if (!entry)
        {
            return std::unexpected("sampling needs a strategy's name, or scene");
        }
        return use_sampling{entry->value};
    }
    if (verb == "precision")
    {
        if (w.size() != 2 || (w[1] != "float" && w[1] != "double"))
        {
            return std::unexpected("precision is float or double");
        }
        return use_precision{w[1] == "double"};
    }
    if (verb == "optix" || verb == "denoise" || verb == "pause" || verb == "hud" || verb == "help")
    {
        const std::optional<bool> on = on_or_off();
        if (!on)
        {
            return std::unexpected(std::format("{} is on or off", verb));
        }
        return verb == "optix"     ? action(use_optix{*on})
               : verb == "denoise" ? action(use_denoise{*on})
               : verb == "pause"   ? action(use_pause{*on})
               : verb == "hud"     ? action(show_hud{*on})
                                   : action(show_help{*on});
    }
    if (verb == "tonemap")
    {
        const tone_map_entry* entry = w.size() == 2 ? find_tone_map(w[1]) : nullptr;
        if (!entry)
        {
            return std::unexpected(std::format("tonemap is {}", tone_map_names()));
        }
        return use_tone_map{entry->value};
    }
    if (verb == "exposure")
    {
        const std::optional<double> ev = w.size() == 2 ? parse_exposure(w[1]) : std::nullopt;
        if (!ev)
        {
            return std::unexpected(
                std::format("exposure needs a number of stops from {} to {}", -max_exposure, max_exposure));
        }
        return use_exposure{*ev};
    }
    if (verb == "samples")
    {
        const std::optional<std::int64_t> n = w.size() == 2 ? parse_number<std::int64_t>(w[1]) : std::nullopt;
        if (!n || *n < 1 || *n > max_action_samples)
        {
            return std::unexpected(std::format("samples needs a whole number from 1 to {}", max_action_samples));
        }
        return use_samples{*n};
    }
    if (verb == "vfov" || verb == "defocus-angle" || verb == "focus-dist")
    {
        const std::optional<double> value = number();
        if (!value || (verb == "focus-dist" ? !(*value > 0) : *value < 0))
        {
            return std::unexpected(verb == "focus-dist" ? std::string("focus-dist needs a distance greater than zero")
                                                        : std::format("{} needs an angle in degrees, from 0", verb));
        }
        return verb == "vfov"            ? action(use_vfov{*value})
               : verb == "defocus-angle" ? action(use_lens{*value})
                                         : action(use_focus{*value});
    }
    if (verb == "reset-view")
    {
        return alone(reset_view{});
    }
    if (verb == "save")
    {
        return alone(save_image{});
    }
    if (verb == "copy")
    {
        return alone(copy_command{});
    }
    return std::unexpected(std::format("unknown action '{}'", verb));
}

namespace
{
std::string_view on_word(bool value)
{
    return value ? "on" : "off";
}

// Each action's line (format_action)
struct action_writer
{
    std::string operator()(const show_scene& s) const
    {
        const bool plain = !s.name.empty() && std::ranges::none_of(s.name, [](char c) {
            return c == ' ' || c == '\t' || c == '#' || c == '"';
        });
        return plain ? std::format("scene {}", s.name) : std::format("scene \"{}\"", s.name);
    }
    std::string operator()(const use_sampling& s) const
    {
        return std::format("sampling {}", s.sampling ? strategy_name(*s.sampling) : std::string_view("scene"));
    }
    std::string operator()(const use_precision& p) const
    {
        return std::format("precision {}", p.fp64 ? "double" : "float");
    }
    std::string operator()(const use_optix& o) const { return std::format("optix {}", on_word(o.on)); }
    std::string operator()(const use_denoise& d) const { return std::format("denoise {}", on_word(d.on)); }
    std::string operator()(const use_tone_map& t) const { return std::format("tonemap {}", tone_map_name(t.op)); }
    std::string operator()(const use_exposure& e) const { return std::format("exposure {}", e.exposure); }
    std::string operator()(const use_samples& s) const { return std::format("samples {}", s.samples); }
    std::string operator()(const use_pause& p) const { return std::format("pause {}", on_word(p.on)); }
    std::string operator()(const use_vfov& v) const { return std::format("vfov {}", v.degrees); }
    std::string operator()(const use_lens& l) const { return std::format("defocus-angle {}", l.degrees); }
    std::string operator()(const use_focus& f) const { return std::format("focus-dist {}", f.distance); }
    std::string operator()(const reset_view&) const { return "reset-view"; }
    std::string operator()(const save_image&) const { return "save"; }
    std::string operator()(const copy_command&) const { return "copy"; }
    std::string operator()(const show_hud& h) const { return std::format("hud {}", on_word(h.on)); }
    std::string operator()(const show_help& h) const { return std::format("help {}", on_word(h.on)); }
};
}

// An action as its line: the words parse_action reads back to the same action. A scene's name is quoted when it holds
// what would split it (a file's path may hold spaces), and a number is written with the digits that read back to it
// exactly.
export std::string format_action(const action& a)
{
    return std::visit(action_writer{}, a);
}

// The samples to refine to for a number asked for: whole rounds of strata (a round is strata² samples), rounded up,
// from one round to the most whole rounds an int numbers. The window never stops inside a round, whose samples lean
// toward the top of the pixel (see gpu::session).
export std::int64_t whole_rounds(std::int64_t samples, int strata)
{
    const std::int64_t round = std::int64_t{strata} * strata;
    const std::int64_t most = std::max<std::int64_t>(std::numeric_limits<std::int32_t>::max() / round, 1);
    const std::int64_t rounds = (std::max<std::int64_t>(samples, 1) + round - 1) / round;
    return std::clamp(rounds, std::int64_t{1}, most) * round;
}
