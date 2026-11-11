// Module viewer_script: the window's camera scripts and input files, as text. A script says what to render, frame by
// frame: RayTracingViewer --script plays one in the window, and --headless renders one without a window. --record
// writes what the window showed as a script, which --headless renders again bit for bit. An input file (--input)
// drives the window with scripted keyboard, mouse and window events, for tests. Nothing here knows SDL or the GPU: the
// module only reads and writes the text.
export module viewer_script;

import std;
import camera;
import cli;
import strategy_options;
import tone_options;
import vec3;
import viewer_actions;
import viewer_camera;

// --- Scripts -------------------------------------------------------------------------------------------------------
//
// One command per line; # starts a comment (outside a quoted word). Settings stay until changed:
//   scene NAME | scene "FILE.glb" | size W H | seed S | strata K | precision float|double |
//   kernel auto|all-books|quality|optix | sampling NAME|scene
//                a scene of the list, or a glTF 2.0 model's file (FILE.glb or FILE.gltf, quoted: a path may hold
//                spaces) in the photo studio, as on the command line. kernel optix traces with OptiX (the window's O,
//                --optix); a scene with triangle meshes renders with OptiX under kernel auto too, since only OptiX
//                traces triangles.
//   tonemap books|none|agx|neutral | exposure EV   the display the frames are shown with (EV from -24 to 24; the
//                books' bytes take none): it doesn't change the image, so a frame after it may show the samples of
//                the frame before through it. Without them each scene shows its own display, until one of them is
//                given; then that one holds for every scene, as the window's T and exposure keys hold.
//   denoise on|off   the window's N: the frames show the image denoised by OptiX's AI denoiser (float32 only), for
//                every scene after it. A hold's frames are denoised as the window denoises a still view: the view's
//                first frame, then a frame whose complete passes have doubled since the last denoise; the frames in
//                between show the last denoised image again, through a display changed since (tonemap, exposure)
//                re-toned with no denoise. A script has no "done": the window denoises a view once more when it is
//                done or paused, but a hold has only its first frame and the doublings, so the last frame of a hold
//                shows the last doubling's denoise unless it lands on a doubling itself (hold 16 +1 on a new view
//                ends on 16 passes, denoised; hold 10 +1 ends on 10, showing the denoise of 8). A move's frames never
//                are denoised (the camera is moving), and a frame line is when it says so (below).
//   camera [lookfrom "X,Y,Z"] [lookat "X,Y,Z"] [vfov D] [defocus-angle D] [focus-dist D]   (the view, changed)
//   focus-at X Y   the focus distance of the surface seen at pixel (X, Y) of the image, as a double-click sets it
// The camera stays when another setting changes; a new scene starts at its own view.
// Frames:
//   hold F +N    F frames of the view as it is, each with N more complete passes than the frame before (N from 1)
//   orbit F +N yaw D [pitch D] | look F +N yaw D [pitch D] | pan F +N dx X [dy Y] |
//   dolly F +N factor X | dolly F +N notches Y | zoom F +N factor X | zoom F +N notches Y
//                F frames of a camera move, eased in and out (smoothstep), each frame a new view with N passes (from
//                1). Every frame's view is computed from the view before the move, as the window's moves compute it
//                (the same functions, viewer_camera): orbit and look turn by degrees, pan moves by pixels of the image
//                at hand (so a script pans less at a larger size), dolly and zoom multiply the distance and the field
//                of view, by a factor or by the wheel's notches (the very powers the wheel takes)
//   frame N R gen G size W H lookfrom "X,Y,Z" lookat "X,Y,Z" vfov D defocus-angle D focus-dist D [denoised]
//                one frame: that view at N complete passes (from 1) and R rows of the next one (below H); G numbers
//                the views, a new G starting the image over even for the same camera (what --record writes, one line
//                per frame shown with new samples, or with the same ones through a new display). denoised: the window
//                denoised this frame's image, and the replay does; a frame without it shows the view's last denoised
//                image again (re-toned when the display changed since, however far the image has refined since), or
//                before the view's first denoise the image itself. So a recording replays the window's denoises
//                exactly, whatever the clock decided live. The mark needs the denoiser on (--denoise, or denoise on
//                before it): one under denoise off is an error when the script is read.
// A script ends with a frame: lines after its last hold, move or frame would render nothing, and are an error.

// The most passes a frame may have: as many as RayTracingGPU's -s
export constexpr std::int64_t max_script_passes = std::numeric_limits<int>::max();

// The view's fields a script may set; what is empty stays as it was
export struct view_change
{
    std::optional<point3> lookfrom;
    std::optional<point3> lookat;
    std::optional<double> vfov;
    std::optional<double> defocus_angle;
    std::optional<double> focus_dist;
};

export struct set_scene { std::string name; };
export struct set_size { int width; int height; };
export struct set_seed { std::uint32_t seed; };
export struct set_strata { int strata; };
export struct set_precision { bool fp64; };
export struct set_kernel { kernel_choice kernel; };
// empty: the scene's own. The type is written ::strategy: a member may not change what an unqualified name means in its
// class, and this one is named strategy too.
export struct set_sampling { std::optional<::strategy> strategy; };
export struct set_tonemap { tone_map op; };
export struct set_exposure { double exposure; };
export struct set_denoise { bool on; };
export struct set_camera { view_change view; };
export struct focus_at { double x; double y; };
export struct hold { int frames; std::int64_t passes; };
export enum class move_kind
{
    orbit,
    look,
    pan,
    dolly,
    zoom,
};
export struct move
{
    move_kind kind;
    int frames;
    std::int64_t passes;
    double a = 0; // orbit, look: yaw; pan: dx; dolly, zoom: the factor
    double b = 0; // orbit, look: pitch; pan: dy
};
export struct frame
{
    std::int64_t passes;
    int rows;
    std::uint64_t generation;
    int width;
    int height;
    view_change view;      // all set
    bool denoised = false; // the window denoised it (under denoise on)
};

export using script_command =
    std::variant<set_scene, set_size, set_seed, set_strata, set_precision, set_kernel, set_sampling, set_tonemap,
                 set_exposure, set_denoise, set_camera, focus_at, hold, move, frame>;

export struct script_line
{
    int line; // in the file, from 1, for messages
    script_command command;
};

// The commands that render frames, and those that change the view (a script's settings before both are the start's)
export bool renders(const script_command& c)
{
    return std::holds_alternative<hold>(c) || std::holds_alternative<move>(c) || std::holds_alternative<frame>(c);
}

export bool changes_view(const script_command& c)
{
    return std::holds_alternative<set_camera>(c) || std::holds_alternative<focus_at>(c) ||
           std::holds_alternative<move>(c) || std::holds_alternative<frame>(c);
}

namespace
{
// The words of a line, without its comment; a quoted word ("1,2,3", as format_view writes points, or a model's file,
// "C:\My models\bust.glb") loses its quotes, and holds the spaces and the # between them. A quote that isn't closed
// starts a word as any other character does, so every line of the scripts before quoted files reads as it did.
std::vector<std::string_view> words(std::string_view line)
{
    const auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\r'; };
    std::vector<std::string_view> result;
    std::size_t i = 0;
    while (true)
    {
        while (i < line.size() && blank(line[i]))
        {
            ++i;
        }
        if (i == line.size() || line[i] == '#')
        {
            break;
        }
        std::size_t end = i;
        if (const std::size_t close = line[i] == '"' ? line.find('"', i + 1) : std::string_view::npos;
            close != std::string_view::npos)
        {
            end = close + 1;
        }
        while (end < line.size() && !blank(line[end]) && line[end] != '#')
        {
            ++end;
        }
        std::string_view word = line.substr(i, end - i);
        if (word.size() >= 2 && word.front() == '"' && word.back() == '"')
        {
            word = word.substr(1, word.size() - 2);
        }
        result.push_back(word);
        i = end;
    }
    return result;
}

std::optional<double> finite_number(std::string_view text)
{
    const std::optional<double> value = parse_number<double>(text);
    return value && std::isfinite(*value) ? value : std::nullopt;
}

std::optional<point3> point(std::string_view text)
{
    std::array<double, 3> xyz{};
    std::size_t start = 0;
    for (std::size_t k = 0; k < 3; ++k)
    {
        const std::size_t comma = k < 2 ? text.find(',', start) : text.size();
        if (comma == std::string_view::npos)
        {
            return std::nullopt;
        }
        const std::optional<double> value = finite_number(text.substr(start, comma - start));
        if (!value)
        {
            return std::nullopt;
        }
        xyz[k] = *value;
        start = comma + 1;
    }
    return point3{xyz[0], xyz[1], xyz[2]};
}

// Reads `name value` pairs of view fields from w[from...] into view; returns the index of the first word it doesn't
// know, or an error
std::expected<std::size_t, std::string> view_fields(std::span<const std::string_view> w, std::size_t from,
                                                    view_change& view)
{
    std::size_t i = from;
    while (i + 1 < w.size())
    {
        const std::string_view key = w[i];
        const std::string_view value = w[i + 1];
        if (key == "lookfrom" || key == "lookat")
        {
            const std::optional<point3> p = point(value);
            if (!p)
            {
                return std::unexpected(std::format("{} needs a point X,Y,Z, not '{}'", key, value));
            }
            (key == "lookfrom" ? view.lookfrom : view.lookat) = p;
        }
        else if (key == "vfov" || key == "defocus-angle" || key == "focus-dist")
        {
            const std::optional<double> number = finite_number(value);
            if (!number)
            {
                return std::unexpected(std::format("{} needs a number, not '{}'", key, value));
            }
            (key == "vfov" ? view.vfov : key == "defocus-angle" ? view.defocus_angle : view.focus_dist) = number;
        }
        else
        {
            break;
        }
        i += 2;
    }
    return i;
}

// A number of passes from 1 to max_script_passes
std::optional<std::int64_t> passes_number(std::string_view text)
{
    const std::optional<std::int64_t> n = parse_number<std::int64_t>(text);
    return n && *n >= 1 && *n <= max_script_passes ? n : std::nullopt;
}

std::expected<script_command, std::string> command(std::span<const std::string_view> w)
{
    const std::string_view verb = w[0];
    const auto count = [&](std::size_t n) -> std::expected<void, std::string> {
        if (w.size() != n)
        {
            return std::unexpected(std::format("'{}' takes {} value{}", verb, n - 1, n == 2 ? "" : "s"));
        }
        return {};
    };
    const auto positive = [](std::string_view text) { return parse_positive_int(text); };
    if (verb == "scene")
    {
        if (auto ok = count(2); !ok)
        {
            return std::unexpected(ok.error());
        }
        return set_scene{std::string(w[1])};
    }
    if (verb == "size")
    {
        if (auto ok = count(3); !ok)
        {
            return std::unexpected(ok.error());
        }
        const auto width = positive(w[1]);
        const auto height = positive(w[2]);
        if (!width || !height)
        {
            return std::unexpected("size needs a width and a height greater than zero");
        }
        return set_size{*width, *height};
    }
    if (verb == "seed")
    {
        const std::optional<std::uint32_t> seed = w.size() == 2 ? parse_number<std::uint32_t>(w[1]) : std::nullopt;
        if (!seed)
        {
            return std::unexpected("seed needs an integer from 0 to 4294967295");
        }
        return set_seed{*seed};
    }
    if (verb == "strata")
    {
        // 46340: the renderer's limit (gpu::max_sqrt_spp, which this module, knowing no GPU, doesn't import)
        const auto strata = w.size() == 2 ? positive(w[1]) : std::nullopt;
        if (!strata || *strata > 46340)
        {
            return std::unexpected("strata needs an integer from 1 to 46340");
        }
        return set_strata{*strata};
    }
    if (verb == "precision")
    {
        if (w.size() != 2 || (w[1] != "float" && w[1] != "double"))
        {
            return std::unexpected("precision is float or double");
        }
        return set_precision{w[1] == "double"};
    }
    if (verb == "kernel")
    {
        for (const kernel_choice k :
             {kernel_choice::automatic, kernel_choice::all_books, kernel_choice::quality, kernel_choice::optix})
        {
            if (w.size() == 2 && w[1] == kernel_choice_name(k))
            {
                return set_kernel{k};
            }
        }
        return std::unexpected("kernel is auto, all-books, quality or optix");
    }
    if (verb == "sampling")
    {
        if (w.size() == 2 && w[1] == "scene")
        {
            return set_sampling{};
        }
        const strategy_entry* entry = w.size() == 2 ? find_strategy(w[1]) : nullptr;
        if (!entry)
        {
            return std::unexpected("sampling needs a strategy's name, or scene");
        }
        return set_sampling{entry->value};
    }
    if (verb == "tonemap")
    {
        const tone_map_entry* entry = w.size() == 2 ? find_tone_map(w[1]) : nullptr;
        if (!entry)
        {
            return std::unexpected(std::format("tonemap is {}", tone_map_names()));
        }
        return set_tonemap{entry->value};
    }
    if (verb == "exposure")
    {
        const std::optional<double> ev = w.size() == 2 ? parse_exposure(w[1]) : std::nullopt;
        if (!ev)
        {
            return std::unexpected(
                std::format("exposure needs a number of stops from {} to {}", -max_exposure, max_exposure));
        }
        return set_exposure{*ev};
    }
    if (verb == "denoise")
    {
        if (w.size() != 2 || (w[1] != "on" && w[1] != "off"))
        {
            return std::unexpected("denoise is on or off");
        }
        return set_denoise{w[1] == "on"};
    }
    if (verb == "camera")
    {
        set_camera result;
        const auto end = view_fields(w, 1, result.view);
        if (!end)
        {
            return std::unexpected(end.error());
        }
        if (*end != w.size())
        {
            return std::unexpected(std::format("camera doesn't know '{}'", w[*end]));
        }
        return result;
    }
    if (verb == "orbit" || verb == "look" || verb == "pan" || verb == "dolly" || verb == "zoom")
    {
        const auto frames = w.size() >= 5 ? positive(w[1]) : std::nullopt;
        const auto passes = w.size() >= 5 && w[2].starts_with('+') ? passes_number(w[2].substr(1)) : std::nullopt;
        if (!frames || !passes)
        {
            return std::unexpected(std::format("{} needs F +N (N from 1 to {}) and its amounts", verb,
                                               max_script_passes));
        }
        move result{.kind = verb == "orbit" ? move_kind::orbit
                          : verb == "look"  ? move_kind::look
                          : verb == "pan"   ? move_kind::pan
                          : verb == "dolly" ? move_kind::dolly
                                            : move_kind::zoom,
                     .frames = *frames,
                     .passes = *passes};
        // The amounts: name value pairs
        const bool turns = verb == "orbit" || verb == "look";
        const std::string_view first = turns ? "yaw" : verb == "pan" ? "dx" : "factor";
        const std::string_view second = turns ? "pitch" : verb == "pan" ? "dy" : "";
        // dolly and zoom: notches of the wheel instead of a factor
        const bool by_notches = (verb == "dolly" || verb == "zoom") && w.size() > 3 && w[3] == "notches";
        bool has_first = false;
        for (std::size_t i = 3; i < w.size(); i += 2)
        {
            const std::optional<double> value = i + 1 < w.size() ? finite_number(w[i + 1]) : std::nullopt;
            if (!value)
            {
                return std::unexpected(std::format("{} needs a number after '{}'", verb, w[i]));
            }
            if (by_notches && w[i] == "notches" && i == 3)
            {
                result.a = std::pow(verb == "dolly" ? wheel_dolly : wheel_zoom, *value);
                has_first = true;
            }
            else if (w[i] == first)
            {
                result.a = *value;
                has_first = true;
            }
            else if (!second.empty() && w[i] == second)
            {
                result.b = *value;
            }
            else
            {
                return std::unexpected(std::format("{} doesn't know '{}'", verb, w[i]));
            }
        }
        if (!has_first || ((verb == "dolly" || verb == "zoom") && !(result.a > 0)))
        {
            return std::unexpected(std::format("{} needs {}{}", verb, first,
                                               verb == "dolly" || verb == "zoom" ? " greater than zero, or notches"
                                                                                 : ""));
        }
        return result;
    }
    if (verb == "focus-at")
    {
        const std::optional<double> x = w.size() == 3 ? finite_number(w[1]) : std::nullopt;
        const std::optional<double> y = w.size() == 3 ? finite_number(w[2]) : std::nullopt;
        if (!x || !y || *x < 0 || *y < 0)
        {
            return std::unexpected("focus-at needs a pixel's X Y, from 0");
        }
        return focus_at{*x, *y};
    }
    if (verb == "hold")
    {
        const auto frames = w.size() == 3 ? positive(w[1]) : std::nullopt;
        const auto passes = w.size() == 3 && w[2].starts_with('+') ? passes_number(w[2].substr(1)) : std::nullopt;
        if (!frames || !passes)
        {
            return std::unexpected(std::format("hold needs F +N: a number of frames, and the passes each adds (+1 to "
                                               "+{})",
                                               max_script_passes));
        }
        return hold{*frames, *passes};
    }
    if (verb == "frame")
    {
        // frame N R gen G size W H <view fields>
        const std::string why = std::format("frame needs N R gen G size W H and the view's five fields: N passes from "
                                            "1 to {}, R rows below H",
                                            max_script_passes);
        if (w.size() < 9 || w[3] != "gen" || w[5] != "size")
        {
            return std::unexpected(why);
        }
        const std::optional<std::int64_t> passes = passes_number(w[1]);
        const std::optional<int> rows = parse_number<int>(w[2]);
        const std::optional<std::uint64_t> generation = parse_number<std::uint64_t>(w[4]);
        const auto width = positive(w[6]);
        const auto height = positive(w[7]);
        if (!passes || !rows || !generation || !width || !height || *rows < 0 || *rows >= *height)
        {
            return std::unexpected(why);
        }
        frame result{.passes = *passes, .rows = *rows, .generation = *generation, .width = *width, .height = *height};
        const auto end = view_fields(w, 8, result.view);
        if (!end)
        {
            return std::unexpected(end.error());
        }
        // The mark of a denoised frame, the line's last word
        std::size_t last = w.size();
        if (*end + 1 == w.size() && w[*end] == "denoised")
        {
            result.denoised = true;
            last = *end;
        }
        if (*end != last)
        {
            return std::unexpected(std::format("frame doesn't know '{}'", w[*end]));
        }
        const view_change& v = result.view;
        if (!v.lookfrom || !v.lookat || !v.vfov || !v.defocus_angle || !v.focus_dist)
        {
            return std::unexpected("frame needs all five view fields: lookfrom, lookat, vfov, defocus-angle and "
                                   "focus-dist");
        }
        return result;
    }
    return std::unexpected(std::format("unknown command '{}'", verb));
}
}

// The commands of a script, or the first error with its line number. denoise: the denoiser on before the first line
// (--denoise); the denoise lines turn it on and off from there, and a frame marked denoised under denoise off is an
// error (the window marks only frames it denoised).
export std::expected<std::vector<script_line>, std::string> parse_script(std::string_view text, bool denoise = false)
{
    std::vector<script_line> result;
    int number = 0;
    for (const auto line : std::views::split(text, '\n'))
    {
        ++number;
        const std::vector<std::string_view> w = words(std::string_view(line.begin(), line.end()));
        if (w.empty())
        {
            continue;
        }
        auto parsed = command(w);
        if (!parsed)
        {
            return std::unexpected(std::format("line {}: {}", number, parsed.error()));
        }
        if (const auto* d = std::get_if<set_denoise>(&*parsed))
        {
            denoise = d->on;
        }
        else if (const auto* f = std::get_if<frame>(&*parsed); f && f->denoised && !denoise)
        {
            return std::unexpected(
                std::format("line {}: a frame marked denoised needs the denoiser: denoise on before it", number));
        }
        result.push_back({number, std::move(*parsed)});
    }
    const auto last =
        std::ranges::find_if(result.rbegin(), result.rend(), [](const script_line& l) { return renders(l.command); });
    if (last == result.rend())
    {
        return std::unexpected("the script renders nothing: it has no hold, move or frame line");
    }
    if (last != result.rbegin())
    {
        return std::unexpected(std::format("line {}: nothing is rendered after this line (a script ends with a hold, a "
                                           "move or a frame)",
                                           (last - 1)->line));
    }
    return result;
}

// --- What --record writes ------------------------------------------------------------------------------------------

namespace
{
// A point as format_view writes it: shortest round trip, quoted
std::string quoted(const point3& p)
{
    return std::format("\"{},{},{}\"", p.x, p.y, p.z);
}

// A scene as a script names it: a name of the list as it is (each is lowercase letters, digits and dashes), a model's
// file quoted, as it may hold spaces (a Windows path never holds a quote)
std::string scene_word(std::string_view scene)
{
    const bool name = !scene.empty() && std::ranges::all_of(scene, [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-';
    });
    return name ? std::string(scene) : std::format("\"{}\"", scene);
}
}

// The display a recording's frames are shown with: a tone map and its exposure in EV
export struct display_setting
{
    tone_map op;
    double exposure;
};

// Its lines, the exposure with the shortest digits that read back as the same number, so a replay shows the frames
// with exactly the window's exposure
export std::string format_display(const display_setting& display)
{
    return std::format("tonemap {}\nexposure {}\n", tone_map_name(display.op), display.exposure);
}

// The settings lines that start a recording, and that follow a change of settings; the display's lines after them,
// when it is given (the window gives it unless both the display and the scene's own are the books' bytes, so the
// books' scenes' recordings stay as they were). A model's file is written quoted, and kernel optix only in a session
// that traces with OptiX, so the recordings of the window without OptiX stay as they were too.
export std::string format_settings(std::string_view scene, bool fp64, kernel_choice kernel,
                                   std::optional<strategy> s, std::uint32_t seed, int strata,
                                   const std::optional<display_setting>& display = std::nullopt)
{
    const std::string name = s ? std::string(strategy_name(*s)) : std::string("scene");
    return std::format("scene {}\nprecision {}\nkernel {}\nsampling {}\nseed {}\nstrata {}\n{}", scene_word(scene),
                       fp64 ? "double" : "float", kernel_choice_name(kernel), name, seed, strata,
                       display ? format_display(*display) : std::string());
}

// The denoiser's line, which a recording writes only once the window's N was used (or --denoise, or a script's denoise
// line), so the recordings without it stay as they were
export std::string format_denoise(bool on)
{
    return std::format("denoise {}\n", on ? "on" : "off");
}

// One frame shown: its view (from the camera's settings), how far it was rendered, and whether the window denoised it
// (the mark comes only under denoise on)
export std::string format_frame(std::int64_t passes, int rows, std::uint64_t generation, int width, int height,
                                const camera_settings& view, bool denoised = false)
{
    return std::format("frame {} {} gen {} size {} {} lookfrom {} lookat {} vfov {} defocus-angle {} focus-dist {}{}\n",
                       passes, rows, generation, width, height, quoted(view.lookfrom), quoted(view.lookat), view.vfov,
                       view.defocus_angle, view.focus_dist, denoised ? " denoised" : "");
}

// --- Input files ---------------------------------------------------------------------------------------------------
//
// One event per line, when to send it:
//   at F EVENT     before frame F (the window's F-th iterate, from 0); F never smaller than the line before's
//   after MS EVENT MS milliseconds after the window opened (when --stats' time_ms is 0), from a timer, whatever the
//                  frames (for latency measurements: the window may be waiting for an event then)
// Events:
//   key NAME | key-down NAME | key-up NAME          (NAME as SDL names keys: W, Escape, Left Shift...)
//   mouse-down left|middle|right X Y [CLICKS] | mouse-up left|middle|right X Y [CLICKS] | mouse-move X Y DX DY
//   wheel DY [shift|ctrl]                           (a modifier is pressed around the wheel step)
//   resize W H | device-reset | quit
//   action WORDS                                    an action by name and value (viewer_actions: scene cornell,
//                                                   exposure 0.5, samples 256...), performed before the next frame
//   click NAME                                      a left click at the middle of the panel's control of that name,
//                                                   where the panel drew it last (viewer_panel's panel_item: scene
//                                                   cornell, sampling, sampling mis, optix, save...): the mouse moves
//                                                   there, presses and lets go, as real events do

export enum class input_kind
{
    key_down,
    key_up,
    mouse_down,
    mouse_up,
    mouse_move,
    wheel,
    resize,
    device_reset,
    quit,
    action, // performed by the window itself, not sent through SDL
    click,  // a click on the panel's control named `key`
};

export struct input_event
{
    int line;
    std::int64_t at_frame = -1; // at F, or -1 for after MS
    double after_ms = -1;       // after MS, or -1 for at F
    input_kind kind;
    std::string key;    // keys: SDL's name of the key
    int button = 0;     // mouse: 1 left, 2 middle, 3 right (SDL's numbers)
    double x = 0;       // mouse: the position; resize: the width
    double y = 0;       // mouse: the position; resize: the height
    double dx = 0;      // mouse-move: the motion
    double dy = 0;      // mouse-move: the motion; wheel: the steps
    int clicks = 1;     // mouse-down, mouse-up
    std::optional<::action> act; // action: what to perform
};

export std::expected<std::vector<input_event>, std::string> parse_input(std::string_view text)
{
    std::vector<input_event> result;
    int number = 0;
    std::int64_t last_frame = 0;
    const auto button = [](std::string_view name) {
        return name == "left" ? 1 : name == "middle" ? 2 : name == "right" ? 3 : 0;
    };
    for (const auto line : std::views::split(text, '\n'))
    {
        ++number;
        const std::vector<std::string_view> w = words(std::string_view(line.begin(), line.end()));
        if (w.empty())
        {
            continue;
        }
        const auto fail = [&](std::string_view why) {
            return std::unexpected(std::format("line {}: {}", number, why));
        };
        if (w.size() < 3 || (w[0] != "at" && w[0] != "after"))
        {
            return fail("an event starts with 'at FRAME' or 'after MS'");
        }
        input_event e{.line = number};
        if (w[0] == "at")
        {
            const auto f = parse_number<std::int64_t>(w[1]);
            if (!f || *f < 0)
            {
                return fail("at needs a frame number from 0");
            }
            if (*f < last_frame)
            {
                return fail(std::format("at {} comes after an event at frame {}: the frames go in order", *f,
                                        last_frame));
            }
            e.at_frame = last_frame = *f;
        }
        else
        {
            const auto ms = finite_number(w[1]);
            if (!ms || *ms < 0 || *ms > 1e9)
            {
                return fail("after needs milliseconds from 0 to 1e9");
            }
            e.after_ms = *ms;
        }
        const std::string_view verb = w[2];
        const std::span<const std::string_view> args(w.begin() + 3, w.end());
        const auto numbers = [&](std::size_t n) -> std::optional<std::vector<double>> {
            if (args.size() < n)
            {
                return std::nullopt;
            }
            std::vector<double> values;
            for (std::size_t i = 0; i < n; ++i)
            {
                const auto v = finite_number(args[i]);
                if (!v)
                {
                    return std::nullopt;
                }
                values.push_back(*v);
            }
            return values;
        };
        if (verb == "key" || verb == "key-down" || verb == "key-up")
        {
            if (args.empty())
            {
                return fail("a key event needs the key's name");
            }
            // A key's name may have spaces ("Left Shift"): the rest of the line
            std::string name;
            for (const std::string_view part : args)
            {
                name += (name.empty() ? "" : " ") + std::string(part);
            }
            e.key = name;
            if (verb == "key")
            {
                e.kind = input_kind::key_down;
                result.push_back(e);
                e.kind = input_kind::key_up;
            }
            else
            {
                e.kind = verb == "key-down" ? input_kind::key_down : input_kind::key_up;
            }
        }
        else if (verb == "mouse-down" || verb == "mouse-up")
        {
            if (args.size() < 3 || args.size() > 4 || !button(args[0]))
            {
                return fail("a mouse button event needs left, middle or right, then X Y");
            }
            const auto x = finite_number(args[1]);
            const auto y = finite_number(args[2]);
            if (!x || !y)
            {
                return fail("a mouse button event needs X Y");
            }
            e.kind = verb == "mouse-down" ? input_kind::mouse_down : input_kind::mouse_up;
            e.button = button(args[0]);
            e.x = *x;
            e.y = *y;
            if (args.size() > 3)
            {
                const auto clicks = parse_positive_int(args[3]);
                if (!clicks || *clicks > 255)
                {
                    return fail("CLICKS is a number from 1 to 255");
                }
                e.clicks = *clicks;
            }
        }
        else if (verb == "mouse-move")
        {
            const auto v = numbers(4);
            if (!v || args.size() != 4)
            {
                return fail("mouse-move needs X Y DX DY");
            }
            e.kind = input_kind::mouse_move;
            e.x = (*v)[0];
            e.y = (*v)[1];
            e.dx = (*v)[2];
            e.dy = (*v)[3];
        }
        else if (verb == "wheel")
        {
            const auto v = numbers(1);
            if (!v || args.size() > 2 || (args.size() == 2 && args[1] != "shift" && args[1] != "ctrl"))
            {
                return fail("wheel needs DY, and maybe shift or ctrl");
            }
            const std::string modifier = args.size() == 2 ? (args[1] == "shift" ? "Left Shift" : "Left Ctrl") : "";
            if (!modifier.empty())
            {
                result.push_back({.line = number, .at_frame = e.at_frame, .after_ms = e.after_ms,
                                  .kind = input_kind::key_down, .key = modifier});
            }
            e.kind = input_kind::wheel;
            e.dy = (*v)[0];
            result.push_back(e);
            if (!modifier.empty())
            {
                result.push_back({.line = number, .at_frame = e.at_frame, .after_ms = e.after_ms,
                                  .kind = input_kind::key_up, .key = modifier});
            }
            continue;
        }
        else if (verb == "resize")
        {
            const auto width = args.size() == 2 ? parse_positive_int(args[0]) : std::nullopt;
            const auto height = args.size() == 2 ? parse_positive_int(args[1]) : std::nullopt;
            if (!width || !height)
            {
                return fail("resize needs W H, whole numbers from 1");
            }
            e.kind = input_kind::resize;
            e.x = *width;
            e.y = *height;
        }
        else if (verb == "action")
        {
            auto read = parse_action(args);
            if (!read)
            {
                return fail(read.error());
            }
            e.kind = input_kind::action;
            e.act = std::move(*read);
        }
        else if (verb == "click")
        {
            if (args.empty())
            {
                return fail("click needs the name of a control of the panel");
            }
            // A control's name may have spaces ("scene cornell"): the rest of the line
            for (const std::string_view part : args)
            {
                e.key += (e.key.empty() ? "" : " ") + std::string(part);
            }
            e.kind = input_kind::click;
        }
        else if (verb == "device-reset" || verb == "quit")
        {
            if (!args.empty())
            {
                return fail(std::format("{} takes no values", verb));
            }
            e.kind = verb == "quit" ? input_kind::quit : input_kind::device_reset;
        }
        else
        {
            return fail(std::format("unknown event '{}'", verb));
        }
        result.push_back(e);
    }
    return result;
}
