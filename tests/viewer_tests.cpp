// ViewerTests: the window's logic that needs neither a window nor a GPU, tested from its specification: the modules
// RayTracingViewer is built from besides main_viewer.cpp. It checks
//   - viewer_script's input files (--input): every event's line read into the events the window sends, a key's press
//     and release, a wheel step's modifier around it, the order of the frames, and what is refused, with the line;
//   - viewer_actions' actions: every action written as its line and read back to the same action, what is refused
//     and why, and the samples to refine to as whole rounds of strata;
//   - the panel (tests/panel_tests.cpp, where the build has Dear ImGui): drawn without a window, clicked, and what it
//     asks for;
//   - viewer_script's camera scripts: settings and frames read, and a recorded frame line (format_frame) read back to
//     the very numbers it was written from, bit for bit: a recording replays the view the window showed;
//   - viewer_camera's moves: a move by nothing returns the settings unchanged, bit for bit; each move changes what it
//     says and nothing else; every limit holds (the field of view, the lens, the nearest distance, the focus).
// Nothing here is random: no seeds. The program compiles its own copies of the viewer's modules (cmake/viewer.cmake),
// so it tests the sources the viewer is built from.
// Each check prints one line, as MeshTests and GpuTests do, then a summary:
//   <group>.<check>: <what was found>, expected <what the specification says>, PASS|FAIL
// Run: `ctest -R viewer_tests` in the build folder, or ViewerTests itself. Exit code: 0 every check passes, 1 a check
// fails.
import std;
import camera;
import strategy_options;
import tone_options;
import vec3;
import viewer_actions;
import viewer_camera;
import viewer_script;

namespace
{
// --- The report -----------------------------------------------------------------------------------------------------

class report
{
public:
    // A property: what was found, and what was expected
    void check(std::string_view name, bool pass, std::string_view found, std::string_view expected)
    {
        std::println("{}: {}, expected {}, {}", name, found, expected, pass ? "PASS" : "FAIL");
        ++checks_;
        failures_ += pass ? 0 : 1;
    }
    // A number against its value, within a tolerance (NaN fails)
    void near(std::string_view name, double value, double expected, double tolerance)
    {
        const double difference = std::abs(value - expected);
        check(name, difference <= tolerance, std::format("{:.17g}", value),
              std::format("{:.17g} within {:.3g}", expected, tolerance));
    }
    // A value against the very same value
    template <class T>
    void same(std::string_view name, const T& value, const T& expected)
    {
        check(name, value == expected, std::format("{}", value), std::format("{}", expected));
    }
    int checks() const { return checks_; }
    int failures() const { return failures_; }

private:
    int checks_ = 0;
    int failures_ = 0;
};

// --- Input files ----------------------------------------------------------------------------------------------------

std::string_view kind_name(input_kind kind)
{
    switch (kind)
    {
    case input_kind::key_down: return "key-down";
    case input_kind::key_up: return "key-up";
    case input_kind::mouse_down: return "mouse-down";
    case input_kind::mouse_up: return "mouse-up";
    case input_kind::mouse_move: return "mouse-move";
    case input_kind::wheel: return "wheel";
    case input_kind::resize: return "resize";
    case input_kind::device_reset: return "device-reset";
    case input_kind::quit: return "quit";
    case input_kind::action: return "action";
    case input_kind::click: return "click";
    case input_kind::drop: return "drop";
    }
    return "?";
}

// An event as one line of text, every field: what the checks compare
std::string describe(const input_event& e)
{
    return std::format("line {} at {} after {} {} key '{}' button {} x {} y {} dx {} dy {} clicks {}{}", e.line,
                       e.at_frame, e.after_ms, kind_name(e.kind), e.key, e.button, e.x, e.y, e.dx, e.dy, e.clicks,
                       e.act ? std::format(" does '{}'", format_action(*e.act)) : std::string());
}

std::string describe(const std::expected<std::vector<input_event>, std::string>& parsed)
{
    if (!parsed)
    {
        return std::format("error: {}", parsed.error());
    }
    std::string text;
    for (const input_event& e : *parsed)
    {
        text += std::format("{}[{}]", text.empty() ? "" : " ", describe(e));
    }
    return text.empty() ? "no events" : text;
}

void test_input(report& r)
{
    const auto events = [&](std::string_view name, std::string_view text, std::string_view expected) {
        r.same(std::format("input.{}", name), describe(parse_input(text)), std::string(expected));
    };
    events("key", "at 0 key P",
           "[line 1 at 0 after -1 key-down key 'P' button 0 x 0 y 0 dx 0 dy 0 clicks 1] "
           "[line 1 at 0 after -1 key-up key 'P' button 0 x 0 y 0 dx 0 dy 0 clicks 1]");
    events("key-with-a-space", "at 3 key-down Left Shift",
           "[line 1 at 3 after -1 key-down key 'Left Shift' button 0 x 0 y 0 dx 0 dy 0 clicks 1]");
    events("key-up", "at 3 key-up Left Shift",
           "[line 1 at 3 after -1 key-up key 'Left Shift' button 0 x 0 y 0 dx 0 dy 0 clicks 1]");
    events("double-click", "at 1 mouse-down left 10 20.5 2",
           "[line 1 at 1 after -1 mouse-down key '' button 1 x 10 y 20.5 dx 0 dy 0 clicks 2]");
    events("mouse-up", "at 1 mouse-up right 3 4",
           "[line 1 at 1 after -1 mouse-up key '' button 3 x 3 y 4 dx 0 dy 0 clicks 1]");
    events("mouse-move", "at 2 mouse-move 100 50 -4 8",
           "[line 1 at 2 after -1 mouse-move key '' button 0 x 100 y 50 dx -4 dy 8 clicks 1]");
    events("wheel", "at 2 wheel -1", "[line 1 at 2 after -1 wheel key '' button 0 x 0 y 0 dx 0 dy -1 clicks 1]");
    events("wheel-with-ctrl", "at 2 wheel 3 ctrl",
           "[line 1 at 2 after -1 key-down key 'Left Ctrl' button 0 x 0 y 0 dx 0 dy 0 clicks 1] "
           "[line 1 at 2 after -1 wheel key '' button 0 x 0 y 0 dx 0 dy 3 clicks 1] "
           "[line 1 at 2 after -1 key-up key 'Left Ctrl' button 0 x 0 y 0 dx 0 dy 0 clicks 1]");
    events("resize", "at 4 resize 320 180",
           "[line 1 at 4 after -1 resize key '' button 0 x 320 y 180 dx 0 dy 0 clicks 1]");
    events("device-reset-and-quit", "at 4 device-reset\nat 9 quit",
           "[line 1 at 4 after -1 device-reset key '' button 0 x 0 y 0 dx 0 dy 0 clicks 1] "
           "[line 2 at 9 after -1 quit key '' button 0 x 0 y 0 dx 0 dy 0 clicks 1]");
    events("after", "after 250.5 key-down W",
           "[line 1 at -1 after 250.5 key-down key 'W' button 0 x 0 y 0 dx 0 dy 0 clicks 1]");
    events("comments-and-empty-lines", "# a test\n\nat 0 quit  # the end\n",
           "[line 3 at 0 after -1 quit key '' button 0 x 0 y 0 dx 0 dy 0 clicks 1]");
    events("no-events", "# nothing\n", "no events");
    events("the-same-frame-twice", "at 5 key A\nat 5 quit",
           "[line 1 at 5 after -1 key-down key 'A' button 0 x 0 y 0 dx 0 dy 0 clicks 1] "
           "[line 1 at 5 after -1 key-up key 'A' button 0 x 0 y 0 dx 0 dy 0 clicks 1] "
           "[line 2 at 5 after -1 quit key '' button 0 x 0 y 0 dx 0 dy 0 clicks 1]");

    // An action: its words read by viewer_actions, a quoted file as one word
    events("action", "at 6 action exposure -1.5",
           "[line 1 at 6 after -1 action key '' button 0 x 0 y 0 dx 0 dy 0 clicks 1 does 'exposure -1.5']");
    events("action-with-a-file", "after 10 action scene \"C:\\My models\\bust (1).glb\"  # a model",
           "[line 1 at -1 after 10 action key '' button 0 x 0 y 0 dx 0 dy 0 clicks 1 does "
           "'scene \"C:\\My models\\bust (1).glb\"']");
    events("an-action-without-a-name", "at 0 action", "error: line 1: an action needs its name");
    events("an-unknown-action", "at 0 action fly", "error: line 1: unknown action 'fly'");
    events("click", "at 9 click scene cornell",
           "[line 1 at 9 after -1 click key 'scene cornell' button 0 x 0 y 0 dx 0 dy 0 clicks 1]");
    events("click-without-a-control", "at 9 click", "error: line 1: click needs the name of a control of the panel");
    events("drop", "at 3 drop \"..\\assets\\models\\a b.glb\"",
           "[line 1 at 3 after -1 drop key '..\\assets\\models\\a b.glb' button 0 x 0 y 0 dx 0 dy 0 clicks 1]");
    events("drop-two-files", "at 3 drop a.glb b.glb",
           "error: line 1: drop needs one file (quoted, if its path holds spaces)");
    events("an-action's-bad-value", "at 0 quit\nat 2 action exposure high",
           "error: line 2: exposure needs a number of stops from -24 to 24");

    // Refusals: each names its line
    events("no-time", "key P", "error: line 1: an event starts with 'at FRAME' or 'after MS'");
    events("a-negative-frame", "at -1 quit", "error: line 1: at needs a frame number from 0");
    events("frames-out-of-order", "at 5 key A\nat 4 quit",
           "error: line 2: at 4 comes after an event at frame 5: the frames go in order");
    events("after-out-of-range", "after 2e9 quit", "error: line 1: after needs milliseconds from 0 to 1e9");
    events("a-time-without-an-event", "at 0", "error: line 1: an event starts with 'at FRAME' or 'after MS'");
    events("a-key-without-a-name", "at 0 key", "error: line 1: a key event needs the key's name");
    events("an-unknown-button", "at 0 mouse-down fourth 1 2",
           "error: line 1: a mouse button event needs left, middle or right, then X Y");
    events("a-button-without-a-place", "at 0 mouse-down left 1 two", "error: line 1: a mouse button event needs X Y");
    events("too-many-clicks", "at 0 mouse-down left 1 2 256", "error: line 1: CLICKS is a number from 1 to 255");
    events("a-short-mouse-move", "at 0 mouse-move 1 2 3", "error: line 1: mouse-move needs X Y DX DY");
    events("a-wheel-with-alt", "at 0 wheel 1 alt", "error: line 1: wheel needs DY, and maybe shift or ctrl");
    events("a-resize-to-nothing", "at 0 resize 0 10", "error: line 1: resize needs W H, whole numbers from 1");
    events("quit-with-a-value", "at 0 quit now", "error: line 1: quit takes no values");
    events("an-unknown-event", "at 7 jump", "error: line 1: unknown event 'jump'");
    events("the-second-line's-error", "at 0 quit\nat 1 jump", "error: line 2: unknown event 'jump'");
}

// --- Actions --------------------------------------------------------------------------------------------------------

// A line's words as the input files split them: at blanks, a quoted word whole and without its quotes
std::vector<std::string> split(std::string_view line)
{
    std::vector<std::string> result;
    std::size_t i = 0;
    while (i < line.size())
    {
        if (line[i] == ' ')
        {
            ++i;
            continue;
        }
        std::size_t end = line[i] == '"' ? line.find('"', i + 1) : std::string_view::npos;
        if (end != std::string_view::npos)
        {
            result.emplace_back(line.substr(i + 1, end - i - 1));
            i = end + 1;
            continue;
        }
        end = std::min(line.find(' ', i), line.size());
        result.emplace_back(line.substr(i, end - i));
        i = end;
    }
    return result;
}

std::expected<action, std::string> read_action(std::string_view line)
{
    const std::vector<std::string> words = split(line);
    const std::vector<std::string_view> views(words.begin(), words.end());
    return parse_action(views);
}

void test_actions(report& r)
{
    // Every action, written and read back: the same action, and the same line
    const std::pair<action, std::string_view> lines[] = {
        {show_scene{"cornell"}, "scene cornell"},
        {show_scene{"C:\\My models\\bust (1).glb"}, "scene \"C:\\My models\\bust (1).glb\""},
        {use_sampling{strategy::mis}, "sampling mis"},
        {use_sampling{}, "sampling scene"},
        {use_precision{true}, "precision double"},
        {use_precision{false}, "precision float"},
        {use_optix{true}, "optix on"},
        {use_denoise{false}, "denoise off"},
        {use_tone_map{tone_map::agx}, "tonemap agx"},
        {use_exposure{-1.5}, "exposure -1.5"},
        {use_exposure{0.1}, "exposure 0.1"},
        {use_samples{4096}, "samples 4096"},
        {use_pause{true}, "pause on"},
        {use_vfov{37.5}, "vfov 37.5"},
        {use_lens{0.6}, "defocus-angle 0.6"},
        {use_focus{2.0 / 3.0}, "focus-dist 0.6666666666666666"}, // every digit a double needs
        {reset_view{}, "reset-view"},
        {save_image{}, "save"},
        {copy_command{}, "copy"},
        {show_hud{false}, "hud off"},
        {show_help{true}, "help on"},
        {show_panel{false}, "panel off"},
        {open_file{}, "open"},
        {render_image{1920, 1080, 256, false}, "render 1920 1080 256"},
        {render_image{96, 54, 16, true}, "render 96 54 16 denoised"},
        {cancel_render{}, "render-cancel"},
    };
    for (const auto& [a, line] : lines)
    {
        const std::string written = format_action(a);
        r.same(std::format("actions.write.{}", line), written, std::string(line));
        const auto read = read_action(written);
        r.check(std::format("actions.read-back.{}", line), read && *read == a,
                read ? format_action(*read) : read.error(), line);
    }
    // What is refused, and why
    const auto refused = [&](std::string_view line, std::string_view why) {
        const auto read = read_action(line);
        r.same(std::format("actions.refused.{}", line.empty() ? "nothing" : line),
               read ? std::format("read as '{}'", format_action(*read)) : read.error(), std::string(why));
    };
    refused("", "an action needs its name");
    refused("fly", "unknown action 'fly'");
    refused("scene", "scene needs a scene's name or a model's file");
    refused("scene a b", "scene needs a scene's name or a model's file");
    refused("sampling fast", "sampling needs a strategy's name, or scene");
    refused("precision half", "precision is float or double");
    refused("optix", "optix is on or off");
    refused("denoise maybe", "denoise is on or off");
    refused("pause on off", "pause is on or off");
    refused("tonemap film", std::format("tonemap is {}", tone_map_names()));
    refused("exposure 25", "exposure needs a number of stops from -24 to 24");
    refused("exposure nan", "exposure needs a number of stops from -24 to 24");
    refused("samples 0", "samples needs a whole number from 1 to 2147483647");
    refused("samples 2147483648", "samples needs a whole number from 1 to 2147483647");
    refused("samples 1.5", "samples needs a whole number from 1 to 2147483647");
    refused("vfov -1", "vfov needs an angle in degrees, from 0");
    refused("defocus-angle wide", "defocus-angle needs an angle in degrees, from 0");
    refused("focus-dist 0", "focus-dist needs a distance greater than zero");
    refused("focus-dist inf", "focus-dist needs a distance greater than zero");
    refused("reset-view now", "reset-view takes no values");
    refused("save all", "save takes no values");
    refused("panel", "panel is on or off");
    refused("render 1920 1080", "render needs W H N, and maybe denoised");
    refused("render 1920 1080 16 twice", "render needs W H N, and maybe denoised");
    refused("render 15 1080 16", "render's sides are from 16 to 16384 pixels, 33177600 pixels at most in all");
    refused("render 16384 16384 16", "render's sides are from 16 to 16384 pixels, 33177600 pixels at most in all");
    refused("render 1920 1080 0", "render needs samples from 1 to 2147483647");
    refused("render-cancel now", "render-cancel takes no values");

    // The samples to refine to: whole rounds of strata, rounded up, from one round up
    r.same("actions.rounds.exact", whole_rounds(128, 8), std::int64_t{128});
    r.same("actions.rounds.up", whole_rounds(100, 8), std::int64_t{128});
    r.same("actions.rounds.one-sample", whole_rounds(1, 8), std::int64_t{64});
    r.same("actions.rounds.strata-1", whole_rounds(7, 1), std::int64_t{7});
    r.same("actions.rounds.the-most", whole_rounds(2147483647, 8), std::int64_t{2147483647 / 64 * 64});
    r.same("actions.rounds.a-round-larger-than-an-int", whole_rounds(5, 46340), std::int64_t{46340} * 46340);
}

// --- Camera scripts -------------------------------------------------------------------------------------------------

camera_settings sample_view()
{
    camera_settings s;
    s.vfov = 37.5;
    s.lookfrom = {13.25, 2.0 / 3.0, -3.1};
    s.lookat = {0.1, 0.5, 1e-3};
    s.vup = {0, 1, 0};
    s.defocus_angle = 0.6;
    s.focus_dist = 10.123456789012345;
    return s;
}

bool same_point(const vec3& a, const vec3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

std::string point_text(const vec3& p)
{
    return std::format("({:.17g}, {:.17g}, {:.17g})", p.x, p.y, p.z);
}

void test_script(report& r)
{
    // Settings, then frames
    {
        const auto parsed = parse_script("size 64 36\nseed 7\nstrata 4\nhold 2 +1\n");
        const bool read = parsed && parsed->size() == 4;
        r.check("script.lines", read, parsed ? std::format("{} lines", parsed->size()) : parsed.error(), "4 lines");
        if (read)
        {
            const auto* size = std::get_if<set_size>(&(*parsed)[0].command);
            const auto* seed = std::get_if<set_seed>(&(*parsed)[1].command);
            const auto* strata = std::get_if<set_strata>(&(*parsed)[2].command);
            const auto* held = std::get_if<hold>(&(*parsed)[3].command);
            r.check("script.settings", size && seed && strata && size->width == 64 && size->height == 36 &&
                                           seed->seed == 7 && strata->strata == 4,
                    size && seed && strata ? std::format("size {} {} seed {} strata {}", size->width, size->height,
                                                         seed->seed, strata->strata)
                                           : "other commands",
                    "size 64 36 seed 7 strata 4");
            r.check("script.hold", held && held->frames == 2 && held->passes == 1,
                    held ? std::format("{} frames, +{}", held->frames, held->passes) : "another command",
                    "2 frames, +1");
            r.same("script.line-numbers", std::format("{} {} {} {}", (*parsed)[0].line, (*parsed)[1].line,
                                                      (*parsed)[2].line, (*parsed)[3].line),
                   std::string("1 2 3 4"));
        }
    }
    // A recorded frame, read back: the same view, bit for bit
    {
        const camera_settings view = sample_view();
        const std::string line = format_frame(48, 17, 3, 320, 180, view);
        const auto parsed = parse_script(line);
        const frame* f = parsed && parsed->size() == 1 ? std::get_if<frame>(&(*parsed)[0].command) : nullptr;
        r.check("script.frame-read-back", f != nullptr, parsed ? "a frame line" : parsed.error(), "a frame line");
        if (f)
        {
            r.same("script.frame-numbers",
                   std::format("{} {} gen {} size {} {}", f->passes, f->rows, f->generation, f->width, f->height),
                   std::string("48 17 gen 3 size 320 180"));
            const bool all = f->view.lookfrom && f->view.lookat && f->view.vfov && f->view.defocus_angle &&
                             f->view.focus_dist;
            r.check("script.frame-view-complete", all, all ? "every field" : "a field missing", "every field");
            if (all)
            {
                r.check("script.frame-lookfrom", same_point(*f->view.lookfrom, view.lookfrom),
                        point_text(*f->view.lookfrom), point_text(view.lookfrom));
                r.check("script.frame-lookat", same_point(*f->view.lookat, view.lookat), point_text(*f->view.lookat),
                        point_text(view.lookat));
                r.check("script.frame-lens",
                        *f->view.vfov == view.vfov && *f->view.defocus_angle == view.defocus_angle &&
                            *f->view.focus_dist == view.focus_dist,
                        std::format("{:.17g} {:.17g} {:.17g}", *f->view.vfov, *f->view.defocus_angle,
                                    *f->view.focus_dist),
                        std::format("{:.17g} {:.17g} {:.17g}", view.vfov, view.defocus_angle, view.focus_dist));
            }
            r.same("script.frame-not-denoised", f->denoised, false);
        }
        // The mark of a denoised frame needs the denoiser on
        const std::string marked = format_frame(48, 0, 3, 320, 180, view, true);
        const auto refused = parse_script(marked);
        r.check("script.denoised-mark-without-the-denoiser", !refused, refused ? "read" : "refused", "refused");
        const auto taken = parse_script(marked, true);
        const frame* g = taken && taken->size() == 1 ? std::get_if<frame>(&(*taken)[0].command) : nullptr;
        r.check("script.denoised-mark-with-it", g && g->denoised, g ? (g->denoised ? "denoised" : "plain") : "refused",
                "denoised");
    }
    // A script ends with a frame, and an unknown command names its line
    {
        const auto no_frame = parse_script("size 64 36\n");
        r.check("script.no-frames", !no_frame, no_frame ? "read" : "refused", "refused");
        const auto unknown = parse_script("size 64 36\njump 3\nhold 1 +1\n");
        r.check("script.unknown-command", !unknown && unknown.error().contains("line 2"),
                unknown ? "read" : unknown.error(), "an error that names line 2");
    }
}

// --- Camera moves ---------------------------------------------------------------------------------------------------

bool same_view(const camera_settings& a, const camera_settings& b)
{
    return same_point(a.lookfrom, b.lookfrom) && same_point(a.lookat, b.lookat) && same_point(a.vup, b.vup) &&
           a.vfov == b.vfov && a.defocus_angle == b.defocus_angle && a.focus_dist == b.focus_dist;
}

void test_camera(report& r)
{
    const camera_settings s = sample_view();
    const double distance = (s.lookfrom - s.lookat).length();
    const auto unchanged = [&](std::string_view name, const camera_settings& moved) {
        r.check(std::format("camera.{}", name), same_view(moved, s), same_view(moved, s) ? "the same view" : "another",
                "the same view, bit for bit");
    };
    // A move by nothing
    unchanged("orbit-by-nothing", orbit(s, 0, 0));
    unchanged("look-by-nothing", look(s, 0, 0));
    unchanged("pan-by-nothing", pan(s, 0, 0, 180));
    unchanged("dolly-by-nothing", dolly(s, 1, distance / 1000, s.focus_dist / 1000));
    unchanged("zoom-by-nothing", zoom(s, 1));
    unchanged("refocus-by-nothing", refocus(s, 1));
    unchanged("fly-by-nothing", fly(s, 0, 0, 0));

    // Orbit turns lookfrom about lookat: the distance and lookat stay, and a full turn comes back
    {
        const camera_settings turned = orbit(s, 90, 10);
        r.check("camera.orbit-keeps-lookat", same_point(turned.lookat, s.lookat), point_text(turned.lookat),
                point_text(s.lookat));
        r.near("camera.orbit-keeps-the-distance", (turned.lookfrom - turned.lookat).length(), distance, 1e-12);
        const camera_settings around = orbit(s, 360, 0);
        r.near("camera.orbit-a-full-turn", (around.lookfrom - s.lookfrom).length(), 0, 1e-12);
        // Straight up is out of reach: a degree short of it
        const camera_settings up = orbit(s, 0, 1000);
        const vec3 back = unit_vector(up.lookfrom - up.lookat);
        r.near("camera.orbit-stops-a-degree-from-vup", std::acos(dot(back, unit_vector(s.vup))) * 180 / std::numbers::pi,
               1, 1e-9);
    }
    // Look turns lookat about lookfrom
    {
        const camera_settings turned = look(s, -30, 5);
        r.check("camera.look-keeps-lookfrom", same_point(turned.lookfrom, s.lookfrom), point_text(turned.lookfrom),
                point_text(s.lookfrom));
        r.near("camera.look-keeps-the-distance", (turned.lookfrom - turned.lookat).length(), distance, 1e-12);
    }
    // Pan and fly move both points by the same vector
    {
        const camera_settings panned = pan(s, 40, -25, 180);
        r.near("camera.pan-moves-both-alike",
               ((panned.lookfrom - s.lookfrom) - (panned.lookat - s.lookat)).length(), 0, 1e-12);
        r.check("camera.pan-moves", (panned.lookfrom - s.lookfrom).length() > 0, "moved", "moved");
        // dx pixels of an image `height` high at lookat's distance: 2·d·tan(vfov/2)/height each
        const double per_pixel = 2 * distance * std::tan(s.vfov * std::numbers::pi / 360) / 180;
        r.near("camera.pan-distance", (panned.lookfrom - s.lookfrom).length(), per_pixel * std::hypot(40.0, 25.0),
               1e-9);
        const camera_settings flown = fly(s, 2, 0, 0);
        r.near("camera.fly-forward", (flown.lookfrom - s.lookfrom).length(), 2, 1e-12);
        r.near("camera.fly-moves-both-alike", ((flown.lookfrom - s.lookfrom) - (flown.lookat - s.lookat)).length(), 0,
               1e-12);
    }
    // Dolly: the distance times the factor, the focus by the same amount, neither below its floor
    {
        const camera_settings nearer = dolly(s, 0.5, distance / 1000, s.focus_dist / 1000);
        r.near("camera.dolly-distance", (nearer.lookfrom - nearer.lookat).length(), distance / 2, 1e-12);
        r.near("camera.dolly-focus", nearer.focus_dist, s.focus_dist - distance / 2, 1e-12);
        const camera_settings nearest = dolly(s, 1e-9, distance / 1000, s.focus_dist / 1000);
        r.near("camera.dolly-floor", (nearest.lookfrom - nearest.lookat).length(), distance / 1000, 1e-12);
        r.check("camera.dolly-focus-floor", nearest.focus_dist >= s.focus_dist / 1000,
                std::format("{:.6g}", nearest.focus_dist), std::format("at least {:.6g}", s.focus_dist / 1000));
    }
    // Zoom: the field of view within 1 to 160 degrees
    {
        r.near("camera.zoom", zoom(s, 0.9).vfov, s.vfov * 0.9, 1e-12);
        r.same("camera.zoom-narrowest", zoom(s, 1e-6).vfov, 1.0);
        r.same("camera.zoom-widest", zoom(s, 1e6).vfov, 160.0);
    }
    // The lens: a closed one opens at a quarter of a degree, a step back closes it, and it opens to 30 degrees
    {
        camera_settings closed = s;
        closed.defocus_angle = 0;
        const camera_settings opened = aperture_step(closed, true);
        r.same("camera.lens-opens", opened.defocus_angle, 0.25);
        r.same("camera.lens-closes", aperture_step(opened, false).defocus_angle, 0.0);
        r.same("camera.lens-stays-closed", aperture_step(closed, false).defocus_angle, 0.0);
        camera_settings wide = s;
        for (int i = 0; i < 100; ++i)
        {
            wide = aperture_step(wide, true);
        }
        r.same("camera.lens-widest", wide.defocus_angle, 30.0);
        r.near("camera.lens-step", aperture_step(s, true).defocus_angle, s.defocus_angle * 1.25, 1e-15);
    }
    // The focus: farther stops at ten times the distance to lookat; with_focus keeps check_view's range
    {
        r.near("camera.refocus", refocus(s, 1.05).focus_dist, s.focus_dist * 1.05, 1e-12);
        r.near("camera.refocus-farthest", refocus(s, 1e6).focus_dist, 10 * distance, 1e-9);
        r.near("camera.focus-set", with_focus(s, 3.5).focus_dist, 3.5, 0);
        const double far = with_focus(s, 1e12).focus_dist;
        r.check("camera.focus-in-range", far <= distance * 1000, std::format("{:.6g}", far),
                std::format("at most {:.6g}", distance * 1000));
    }
}
}

#ifdef VIEWER_TESTS_PANEL
// The panel's checks (tests/panel_tests.cpp), where the build has Dear ImGui
void test_panel(const std::function<void(std::string_view, bool, std::string_view, std::string_view)>& check);
#endif

int main()
{
    report r;
    test_input(r);
    test_actions(r);
    test_script(r);
    test_camera(r);
#ifdef VIEWER_TESTS_PANEL
    test_panel([&](std::string_view name, bool pass, std::string_view found, std::string_view expected) {
        r.check(name, pass, found, expected);
    });
#else
    std::println("panel: not in this build (no Dear ImGui), INFO");
#endif
    std::println("\nViewerTests: {} checks, {} failed", r.checks(), r.failures());
    return r.failures() == 0 ? 0 : 1;
}
