// RayTracingViewer: RayTracing's scenes in a window, rendered on the GPU and refined while the view stands still.
// The scene and the camera come from the same modules as RayTracingGPU's (gpu_scene, view_options), and the GPU renders
// with a session (gpu/renderer.h) that keeps the scene in its memory and adds samples pass after pass. The image
// converges to the image RayTracingGPU renders of the same view with the same samples, bit for bit, and every frame
// the window shows can be rendered again: --record writes the frames as a script, which --headless renders again.
// RayTracingViewer final
//
// The window comes from SDL 3 (libsdl.org), through its main callbacks: SDL calls SDL_AppInit once, SDL_AppEvent for
// every event, SDL_AppIterate again and again, and SDL_AppQuit at the end. SDL also calls SDL_AppIterate while the user
// drags the window's edge, when a program's own loop would stand still in Windows' loop for the drag, so the image
// goes on refining as the window changes size.

// SDL's headers and the renderer's include standard headers, so they come before `import std;` (see main_gpu.cpp).
// SDL_MAIN_USE_CALLBACKS asks SDL_main.h for the callbacks above instead of a main function of ours.
#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "gpu/renderer.h"
#include "gpu/start_optix.h"

import std;
import camera;
import cli;
import gpu_scene;
import gpu_scenes;
import hittable;
import image;
import interval;
import material;
import random;
import ray;
import scenes;
import strategy_options;
import tone_options;
import vec3;
import view_options;
import viewer_camera;
import viewer_script;

namespace
{
// --- The command line ------------------------------------------------------------------------------------------

constexpr std::string_view default_viewer_scene = "weekend";
constexpr int default_strata = 8; // a round of 8×8 samples per pixel (see --strata)

struct options
{
    std::string_view scene = default_viewer_scene; // a scene's name, or a glTF model's file (is_model_path)
    std::optional<int> image_width;
    std::optional<int> samples_per_pixel;
    int strata = default_strata;
    std::optional<strategy> sampling; // strategy_options.ixx
    view_options view;
    std::uint32_t seed = 0;
    bool fp64 = false;
    bool all_books_kernel = false;
    bool quality_kernel = false;
    bool optix = false; // trace with OptiX (O switches it), as RayTracingGPU's --optix
    std::vector<std::filesystem::path> outputs; // saved when the window closes
    display_choice display;                     // --tonemap, --exposure (tone_options.ixx)
    bool exit_when_done = false;
    std::optional<std::filesystem::path> frames; // a folder for every frame shown, as PNG
    bool frames_hud = false;                     // the frames with the on-screen display and the help on them
    std::optional<std::filesystem::path> stats;  // a CSV file
    std::optional<double> refresh_hz;            // a moving frame's GPU time as at this display rate
    std::optional<std::filesystem::path> script; // play a camera script
    bool headless = false;                       // without a window
    std::optional<std::filesystem::path> record; // write the frames shown as a script
    std::optional<std::filesystem::path> input;  // scripted mouse, keyboard and window events
    bool help = false;
};

void print_usage()
{
    std::println("Usage: RayTracingViewer [scene] [options]\n\n"
                 "Shows RayTracing's scenes in a window, rendered on an NVIDIA graphics card with CUDA: the\n"
                 "image refines while the view stands still. Options as in RayTracingGPU.\n\n"
                 "Scenes:");
    std::print("{}", scenes_usage(default_viewer_scene));
    std::println("\nOr a glTF 2.0 model, FILE.glb or FILE.gltf, in a photo studio (OptiX renders it; P names its\n"
                 "images after the file: FILE-view-001.png).");
    std::println(R"(
Options:
  -w, --width N        image width in pixels: the window's (default: the scene's aspect ratio at
                       the rows of --height, or at 720 rows made smaller if the screen is); the
                       height follows the scene's aspect ratio, unless --height is given
{}
  -s, --samples N      refine until every pixel has N samples (default: the scene's, rounded up to
                       whole rounds of strata)
      --strata K       a round of samples is a K×K grid in each pixel, K from 1 to {}
                       (default {}): --strata 22 -s 484 renders RayTracingGPU's image of the
                       scenes of 500 samples, --strata 32 -s 1024 of those of 1000
  -p, --sampling NAME  how a diffuse bounce chooses the next direction (see RayTracingGPU)
      --fp64           compute in 64-bit doubles: a check of the floats, many times slower
      --all-books-kernel
                       render with the kernel for all three books, also the first book's scenes
      --quality-kernel render with the quality kernel, also the books' scenes under -p material
      --optix          trace with OptiX (on the ray tracing cores where the GPU has them): the
                       quality kernel's integrator in an OptiX program (float; -p material, nee or
                       mis); scenes with triangle meshes always use it
      --seed N         which random numbers to draw, from 0 (the default) to {}
  -o, --output FILE    save the image when the window closes: a .png or a .pfm; several -o save
                       several files
      --tonemap NAME   {} (see RayTracingGPU): the display transform of the
                       window and of a .png; a .pfm stays linear
      --exposure EV    the light times 2^EV before the tone map, from {} to {}; not with books
      --exit-when-done close the window when the image has its samples (after saving -o)
  -h, --help           show this help

In the window:
  left drag            orbit about the point looked at
  right drag           look around
  middle drag, Shift+left drag
                       pan
  wheel                move closer or farther
  Ctrl+wheel, - and =  zoom
  W A S D Q E          fly (Shift faster, Ctrl slower)
  [ and ]              a narrower or wider lens (depth of field)
  Shift+wheel          focus nearer or farther
  double-click         focus on what is under the cursor
  Tab, Shift+Tab       the next or the previous scene; with OptiX (--optix, O, or a scene with
                       triangle meshes), also still-life, busts and cornell-bust
  M, Shift+M           the next or the previous sampling strategy, skipping those the scene can't take
  F2                   float32 or float64
  O                    OptiX on or off for the scene (see --optix); the scenes with triangle meshes
                       always use it
  T, Shift+T           the next or the previous tone map: books, none, agx, neutral
  , and .              exposure down or up by 0.5 EV (not with books)
  0                    exposure back to 0 EV
  Space                pause or go on refining
  Enter, Shift+Enter   twice or half the samples to refine to, in whole rounds of strata
  I                    the numbers on screen
  H, F1                this list in the window
  R                    back to the start view
  P                    save the image as <scene>-view-NNN.png and .pfm in the current folder,
                       and print the RayTracingGPU command that renders it
  C                    copy that command
  Esc                  close the help, let the mouse go, or close the window

Scripts and measurements (see docs/viewer.md):
      --script FILE    play a camera script instead of taking input, then close
      --headless       no window: play the script, or render the samples; then save and exit
      --record FILE    write every frame shown with new samples or a new display as a script line, to
                       render again with --headless --script FILE (a script's moves frame by frame)
      --input FILE     drive the window with scripted keyboard, mouse and window events
      --frames DIR     save every frame shown with new samples or a new display as
                       DIR\f00000.png... (the image only)
      --frames-hud     with --frames in a window: save the window as it is, the numbers and the help
                       on it
      --stats FILE     write a line of numbers per frame to a CSV file
      --refresh-hz N   give a moving view's frames the GPU time of frames at N Hz (for measurements;
                       default: the display's rate; the frames still come at the display's pace))",
                 view_usage, max_strata, default_strata, std::numeric_limits<std::uint32_t>::max(), tone_map_names(),
                 -max_exposure, max_exposure);
}

// The options that take a value, besides the view options (is_view_option); the others are flags
constexpr auto value_options = std::to_array<std::string_view>(
    {"-w", "--width", "-s", "--samples", "--strata", "-p", "--sampling", "--seed", "-o", "--output", "--frames",
     "--stats", "--script", "--record", "--input", "--refresh-hz", "--tonemap", "--exposure"});

// A scene as the command line or a script names it, checked as RayTracingGPU checks its own: one of the GPU programs'
// scenes (gpu_scenes.ixx), or a glTF 2.0 model's file (it ends in .glb or .gltf, which no scene's name does), which
// must exist. The model is read when the scene is built (gpu_scene's cache keeps the last two models read).
std::expected<void, std::string> check_scene(std::string_view scene)
{
    if (is_model_path(scene))
    {
        std::error_code error;
        if (!std::filesystem::is_regular_file(std::filesystem::path(scene), error))
        {
            return std::unexpected(std::format("there is no file '{}'", scene));
        }
        return {};
    }
    if (!find_gpu_scene(scene))
    {
        return std::unexpected(std::format("unknown scene '{}'", scene));
    }
    return {};
}

// A model's file as the window shows it (the HUD, the title): its file name; a scene's name as it is
std::string scene_label(std::string_view scene)
{
    return is_model_path(scene) ? to_utf8(std::filesystem::path(scene).filename()) : std::string(scene);
}

// The start of the names P saves under: a model's file's stem (bust.glb: bust-view-001.png), as RayTracingGPU names
// its image; a scene's name
std::string scene_stem(std::string_view scene)
{
    return is_model_path(scene) ? to_utf8(std::filesystem::path(scene).stem()) : std::string(scene);
}

// A word of a command the window prints, as a command line reads it back: as it is when it holds only letters, digits
// and - _ . : \ / (every scene's name, and the files P names after one), else in double quotes, which cmd.exe and
// PowerShell both read as one word, spaces, parentheses and commas included (a Windows path never holds a quote);
// `always` quotes it whatever it holds
std::string command_word(std::string_view word, bool always = false)
{
    const bool plain = !always && !word.empty() && std::ranges::all_of(word, [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' ||
               c == '.' || c == ':' || c == '\\' || c == '/';
    });
    return plain ? std::string(word) : std::format("\"{}\"", word);
}

// The scene as a RayTracingGPU command names it: a model's file always quoted, as a path may hold spaces
std::string scene_argument(std::string_view scene)
{
    return command_word(scene, is_model_path(scene));
}

std::expected<options, std::string> parse_options(std::span<char* const> args)
{
    options opts;
    for (std::size_t i = 1; i < args.size(); ++i) // args[0] is the program name
    {
        const std::string_view arg = args[i];
        if (arg == "-h" || arg == "--help")
        {
            opts.help = true;
            continue;
        }
        if (arg == "--fp64" || arg == "--all-books-kernel" || arg == "--quality-kernel" || arg == "--exit-when-done" ||
            arg == "--headless" || arg == "--frames-hud" || arg == "--optix")
        {
            (arg == "--fp64"               ? opts.fp64
             : arg == "--all-books-kernel" ? opts.all_books_kernel
             : arg == "--quality-kernel"   ? opts.quality_kernel
             : arg == "--headless"         ? opts.headless
             : arg == "--frames-hud"       ? opts.frames_hud
             : arg == "--optix"            ? opts.optix
                                           : opts.exit_when_done) = true;
            continue;
        }
        if (!arg.starts_with('-'))
        {
            if (const auto known = check_scene(arg); !known)
            {
                return std::unexpected(known.error());
            }
            opts.scene = arg;
            continue;
        }
        if (!is_view_option(arg) && std::ranges::find(value_options, arg) == value_options.end())
        {
            return std::unexpected(std::format("unknown option '{}'", arg));
        }
        if (i + 1 == args.size())
        {
            return std::unexpected(std::format("{} needs a value", arg));
        }
        const std::string_view value = args[++i];
        if (is_view_option(arg))
        {
            if (const auto parsed = parse_view_option(arg, value, opts.view); !parsed)
            {
                return std::unexpected(parsed.error());
            }
        }
        else if (arg == "-o" || arg == "--output")
        {
            std::filesystem::path output = value;
            if (const auto checked = check_output(output); !checked)
            {
                return std::unexpected(std::format("{}: {}", arg, checked.error()));
            }
            opts.outputs.push_back(std::move(output));
        }
        else if (arg == "--frames" || arg == "--stats" || arg == "--script" || arg == "--record" || arg == "--input")
        {
            (arg == "--frames"   ? opts.frames
             : arg == "--stats"  ? opts.stats
             : arg == "--script" ? opts.script
             : arg == "--record" ? opts.record
                                 : opts.input) = std::filesystem::path(value);
        }
        else if (arg == "-p" || arg == "--sampling")
        {
            const strategy_entry* entry = find_strategy(value);
            if (!entry)
            {
                return std::unexpected(std::format("unknown sampling strategy '{}'", value));
            }
            opts.sampling = entry->value;
        }
        else if (arg == "--seed")
        {
            const std::optional<std::uint32_t> seed = parse_number<std::uint32_t>(value);
            if (!seed)
            {
                return std::unexpected(std::format("{}: expected an integer from 0 to {}, not '{}'", arg,
                                                   std::numeric_limits<std::uint32_t>::max(), value));
            }
            opts.seed = *seed;
        }
        else if (arg == "--refresh-hz")
        {
            const std::optional<double> hz = parse_number<double>(value);
            if (!hz || !(*hz >= 1 && *hz <= 1000))
            {
                return std::unexpected(std::format("{}: expected a rate from 1 to 1000, not '{}'", arg, value));
            }
            opts.refresh_hz = hz;
        }
        else if (arg == "--tonemap")
        {
            const auto op = tone_map_option(value);
            if (!op)
            {
                return std::unexpected(op.error());
            }
            opts.display.op = *op;
        }
        else if (arg == "--exposure")
        {
            const auto ev = exposure_option(arg, value);
            if (!ev)
            {
                return std::unexpected(ev.error());
            }
            opts.display.exposure = *ev;
        }
        else // -w, --width, -s, --samples or --strata
        {
            const std::optional<int> number = parse_positive_int(value);
            if (!number || (arg == "--strata" && *number > max_strata))
            {
                return std::unexpected(arg == "--strata"
                                           ? std::format("{}: expected an integer from 1 to {}, not '{}'", arg,
                                                         max_strata, value)
                                           : std::format("{}: expected an integer greater than zero, not '{}'", arg,
                                                         value));
            }
            if (arg == "-w" || arg == "--width")
            {
                opts.image_width = number;
            }
            else if (arg == "--strata")
            {
                opts.strata = *number;
            }
            else
            {
                opts.samples_per_pixel = number;
            }
        }
    }
    if ((opts.headless || opts.script) && opts.input)
    {
        return std::unexpected("--input drives the live window: it can't go with --headless or --script");
    }
    return opts;
}

std::expected<std::string, std::string> read_text(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        return std::unexpected(std::format("can't open {}", to_utf8(path)));
    }
    std::string text(std::istreambuf_iterator<char>(file), {});
    // Text from Windows' editors and shells: UTF-8 with a byte order mark (dropped), or UTF-16 (Windows PowerShell's
    // > and Out-File write it), which isn't read
    if (text.starts_with("\xEF\xBB\xBF"))
    {
        text.erase(0, 3);
    }
    else if (text.starts_with("\xFF\xFE") || text.starts_with("\xFE\xFF"))
    {
        return std::unexpected(std::format("{} is UTF-16 text: save it as UTF-8", to_utf8(path)));
    }
    return text;
}

// --- Pacing ----------------------------------------------------------------------------------------------------

// How long a launch may take while the view stands still: RayTracingGPU's (see render_options). The window then shows
// about 20 frames a second, each with the samples of 50 ms.
constexpr double still_seconds = 0.05;
// After the view changes, the window shows the new view within a frame of the display when a pass fits in one (the
// view's first pass is completed whatever it takes, see gpu::session::render_for): a frame's GPU work is planned for
// this share of a display frame, which leaves the rest for the copy, the upload and the present.
constexpr double moving_share = 0.75;
// How long the view counts as moving after its last change: the frames go on at the display's rate for that long, so a
// camera moved step by step, or a window dragged, doesn't fall back to 20 frames a second between steps
constexpr std::uint64_t moving_ns = 150'000'000;

// The display's frame period and the vsync interval: SDL presents every n-th refresh, n = refresh / 60 rounded down
// (1 to 4, DXGI's limit), so a moving view shows 60 to 120 frames a second on a display that refreshes 60 to 300 times
// a second. A rate just under a multiple of 60, as displays often report one (119.88 for 120), counts as that multiple.
struct pacing
{
    int vsync = 1;
    double frame_seconds = 1.0 / 60;
};

pacing pacing_for(double refresh_hz)
{
    if (!(refresh_hz > 0))
    {
        refresh_hz = 60;
    }
    const int interval = std::clamp(static_cast<int>(std::floor((refresh_hz + 1) / 60)), 1, 4);
    return {.vsync = interval, .frame_seconds = interval / refresh_hz};
}

double display_refresh_hz(SDL_Window* window)
{
    const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window));
    return mode && mode->refresh_rate > 0 ? mode->refresh_rate : 60.0;
}

// The size of the window when the options don't give one: the scene's aspect ratio at 720 rows, shrunk to fit 90% of
// the screen's usable area
std::pair<int, int> default_size(double aspect, SDL_DisplayID display)
{
    double width = std::round(720 * aspect);
    double height = 720;
    SDL_Rect usable{};
    if (SDL_GetDisplayUsableBounds(display, &usable) && usable.w > 0 && usable.h > 0)
    {
        const double scale = std::min({1.0, 0.9 * usable.w / width, 0.9 * usable.h / height});
        width = std::floor(width * scale);
        height = std::floor(height * scale);
    }
    return {std::max(64, static_cast<int>(width)), std::max(36, static_cast<int>(height))};
}

// --- The viewer ------------------------------------------------------------------------------------------------

// The program's own events, registered with SDL when the window opens: one wakes SDL's loop (see iterate), the other
// carries an `after MS` event of --input: a timer pushes it, its code the event's index in the input, and the main
// thread performs it (SDL's timers run on a thread of their own)
Uint32 wake_type = 0;
Uint32 timed_input_type = 0;

Uint64 SDLCALL timed_input(void* index, SDL_TimerID, Uint64)
{
    SDL_Event event{};
    event.type = timed_input_type;
    event.user.code = static_cast<Sint32>(reinterpret_cast<std::intptr_t>(index));
    SDL_PushEvent(&event);
    return 0; // once
}

// A timer's wake event, for a loop that may be waiting for events when a note on screen is due to go
Uint64 SDLCALL wake_up(void*, SDL_TimerID, Uint64)
{
    SDL_Event event{};
    event.type = wake_type;
    SDL_PushEvent(&event);
    return 0; // once
}

// The keyboard and the mouse --input's events come from, so that under --input the window takes its keys and mouse
// from the input file alone (SDL numbers real keyboards and mice from 1 up, and touch and pen as the largest ones)
constexpr Uint32 scripted_device = 0x7E5701;

// The modifier a key sets while it is down, for the key events --input pushes
SDL_Keymod modifier_of(SDL_Scancode key)
{
    switch (key)
    {
    case SDL_SCANCODE_LSHIFT: return SDL_KMOD_LSHIFT;
    case SDL_SCANCODE_RSHIFT: return SDL_KMOD_RSHIFT;
    case SDL_SCANCODE_LCTRL: return SDL_KMOD_LCTRL;
    case SDL_SCANCODE_RCTRL: return SDL_KMOD_RCTRL;
    case SDL_SCANCODE_LALT: return SDL_KMOD_LALT;
    case SDL_SCANCODE_RALT: return SDL_KMOD_RALT;
    case SDL_SCANCODE_LGUI: return SDL_KMOD_LGUI;
    case SDL_SCANCODE_RGUI: return SDL_KMOD_RGUI;
    default: return SDL_KMOD_NONE;
    }
}

// What a session renders with, besides its view: a change of any of these builds the scene and the session again
struct render_settings
{
    std::string scene;
    bool fp64 = false;
    kernel_choice kernel = kernel_choice::automatic;
    std::optional<strategy> sampling; // empty: the scene's own
    std::uint32_t seed = 0;
    int strata = default_strata;
    bool operator==(const render_settings&) const = default;
};

// Whether a scene has triangle meshes (a showcase of glTF models, a model's file), which only OptiX traces
bool has_meshes(const gpu_build& built)
{
    return !built.desc.meshes.instances.empty();
}

// Whether a session of these settings traces with OptiX: asked for (--optix, O, a script's kernel optix), or needed by
// the scene's triangle meshes, whatever the kernel asked for (as in RayTracingGPU, which renders them without --optix)
bool traces_with_optix(const render_settings& settings, const gpu_build& built)
{
    return settings.kernel == kernel_choice::optix || has_meshes(built);
}

class viewer
{
public:
    explicit viewer(options opts) : opts_(std::move(opts)) {}
    ~viewer();
    viewer(const viewer&) = delete;
    viewer& operator=(const viewer&) = delete;

    SDL_AppResult init();
    SDL_AppResult event(const SDL_Event& e);
    SDL_AppResult iterate();

private:
    options opts_;
    SDL_Window* window_ = nullptr;
    SDL_Renderer* renderer_ = nullptr;
    SDL_Texture* texture_ = nullptr;
    int texture_width_ = 0;
    int texture_height_ = 0;

    // The scene, its session and the view
    render_settings settings_;
    std::optional<gpu_build> build_;
    // OptiX, for the sessions that trace with it: started at the first need (open_optix) and kept for the rest of the
    // run, so that Tab, M, F2 and O never compile its programs again. Declared before session_, it outlives every
    // session, and each session holds a copy too (session_options::optix).
    std::shared_ptr<gpu::optix_runtime> optix_;
    std::optional<std::string> optix_failure_; // why OptiX couldn't start: the answer from then on, never tried again
    bool optix_compiled_ = false;              // a session has traced with OptiX: its programs are compiled
    std::set<std::string> warned_models_;      // the glTF files whose warnings the console has shown
    std::optional<gpu::session> session_;
    camera_settings view_;          // the camera's settings: the scene's own view, changed by options or scripts
    camera_settings start_view_;    // the scene's view as it started (R goes back to it)
    double min_distance_ = 0;       // how near lookfrom may come to lookat: 1/1000 of the start's distance
    double min_focus_ = 0;          // and the focus distance: 1/1000 of the start's
    int width_ = 0;                 // the image: the window's client area in pixels
    int height_ = 0;
    std::uint64_t generation_ = 0;  // counts the views: a new one with every camera, size or settings change
    std::int64_t target_ = 0;       // passes to refine to
    int pending_width_ = 0;         // a new size from the window, applied at the next iterate
    int pending_height_ = 0;
    // The strategy the command line or M named (empty: each scene's own), which Tab takes to every scene that can take
    // it, as it takes a display chosen; settings_.sampling is the one the scene shown was asked for
    std::optional<strategy> named_sampling_{opts_.sampling};

    // The display (tone_options.ixx): what the command line, a script or the keys chose, for every scene; what they
    // leave open is each scene's own. A change shows the image again through the new display, without restarting it:
    // at the next iterate, as a frame of its own unless the image is refining anyway (redisplay_).
    display_choice display_{opts_.display};
    bool redisplay_ = false;
    std::optional<gpu::display_options> recorded_display_; // what --record last wrote, if it ever wrote a display

    // The last frame shown, as the session's display() gave it, and its size: what a device reset puts back
    std::vector<std::uint8_t> shown_;
    int shown_width_ = 0;
    int shown_height_ = 0;
    bool present_needed_ = true;
    bool upload_needed_ = false; // shown_ is newer than the texture

    pacing pacing_;
    std::uint64_t start_ns_ = 0;
    std::uint64_t last_iterate_ns_ = 0;
    std::uint64_t moving_until_ns_ = 0;
    bool minimized_ = false;
    bool done_reported_ = false;
    std::uint64_t frames_saved_ = 0;
    std::uint64_t frame_ = 0;
    std::ofstream stats_;
    std::ofstream record_;
    std::optional<render_settings> recorded_settings_;

    // A script being played, and where it is
    std::vector<script_line> script_;
    std::size_t script_next_ = 0;
    int hold_left_ = 0;            // frames left of a hold
    std::int64_t hold_passes_ = 0; // and the passes each adds
    std::optional<std::uint64_t> script_generation_; // the last frame line's generation
    gpu::render_report script_report_; // what rendering the script's last frame did (for --stats)

    // Scripted input: the events, the next `at F` one, and the modifiers and buttons the pushed events hold down
    std::vector<input_event> input_;
    std::size_t input_next_ = 0;
    SDL_Keymod input_mods_ = SDL_KMOD_NONE;
    SDL_MouseButtonFlags input_buttons_ = 0;
    float input_x_ = 0;
    float input_y_ = 0;

    // The mouse and the keys. Buttons, held keys and modifiers are tracked from the events themselves (not from SDL's
    // state functions, which events pushed by --input don't update), and cleared when the window loses the keyboard.
    enum class drag
    {
        none,
        orbit,
        look,
        pan,
    };
    drag drag_ = drag::none;
    Uint8 drag_button_ = 0; // the button that started the drag: one drag at a time
    std::array<bool, SDL_SCANCODE_COUNT> held_{};
    SDL_Keymod mods_ = SDL_KMOD_NONE;
    std::uint64_t fly_since_ns_ = 0; // the time the flight's last step covered up to
    int views_saved_ = 0;
    bool paused_ = false; // Space: no refining
    // A camera moved by this batch of events, shown at the next iterate: one new view for all of a frame's mouse moves
    std::optional<camera_settings> pending_view_;

    // The on-screen display: the numbers at the top, a note at the bottom, the help over the image. The rate and the
    // frame time are measured from one iterate to the next, over frames that rendered, and not over an interval that
    // holds other work: a wait for events, a switch of scene or precision, the pass P and C finish, a scripted resize.
    bool hud_ = true;                    // the numbers (I)
    bool help_ = false;                  // the controls (H, F1)
    std::string note_;                   // such as "nothing there to focus on" after a double-click on the sky
    std::uint64_t note_until_ns_ = 0;    // when the note goes (a timer wakes the loop then)
    double frame_ms_ = 0;                // the time from a frame that rendered to the next one
    double samples_per_second_ = 0;      // samples (pixels times passes) per second of wall time, averaged; 0: none yet
    double last_samples_ = 0;            // the samples the last frame rendered, for the rate at the next iterate
    bool skip_interval_ = false;         // the interval up to the next iterate holds other work: not measured
    double view_ms_ = 0;                 // the time the frames of the image on screen took (for the console's Done line)
    std::uint64_t timed_generation_ = 0; // the image view_ms_ counts for

    // A script's camera move in progress (see viewer_script)
    std::optional<move> move_;
    int move_done_ = 0;
    int move_line_ = 0;
    camera_settings move_start_;

    bool interactive() const { return !opts_.exit_when_done && !opts_.script && !opts_.headless && !opts_.input; }
    camera_settings current_view() const { return pending_view_.value_or(view_); }
    bool flying() const;
    std::expected<void, std::string> apply_pending_view();
    std::expected<gpu::accumulation, std::string> complete_pass();
    bool headless() const { return opts_.headless; }
    SDL_AppResult fail(const std::string& message);
    std::expected<void, std::string> use_settings(const render_settings& settings);
    std::expected<void, std::string> check_display_for(const render_settings& settings) const;
    std::expected<gpu_build, std::string> build_for(const render_settings& settings);
    std::expected<void, std::string> use_build(const render_settings& settings, gpu_build built);
    std::expected<void, std::string> open_optix();
    void note_optix_compile();
    void report_models(const gpu_build& built);
    kernel_choice command_line_kernel() const;
    camera_settings settings_for(const camera_settings& view, int width, int height) const;
    std::expected<void, std::string> show_view(const camera_settings& view, int width, int height, bool restart);
    std::expected<void, std::string> size_window(int width, int height);
    std::string gpu_command(std::int64_t passes) const;
    std::expected<void, std::string> keep_frame(std::span<const std::uint8_t> bytes);
    std::expected<bool, std::string> play_next();
    bool make_texture(int width, int height);
    bool upload_shown();
    bool draw_and_present(bool new_samples);
    bool save_frame_from_window();
    std::expected<void, std::string> save_frame_bytes(std::span<const std::uint8_t> bytes, int width, int height);
    void update_title();
    void print_settings() const;
    bool save_outputs();
    void record_frame();
    std::expected<void, std::string> send_input(std::uint64_t frame);
    std::expected<void, std::string> perform(const input_event& e);
    void write_stats(const gpu::render_report& report, bool new_samples, bool presented, bool moving, double wall_ms,
                     double dt_ms);
    SDL_AppResult run_headless();
    bool move_camera(const camera_settings& to);
    std::optional<camera_settings> pick_focus(double x, double y); // not const: a miss says so on screen
    std::optional<camera_settings> pick_focus_optix(const camera_settings& view, const ray& r, int i, int j);
    camera_settings moved(const move& m, double t) const;
    void save_view();
    SDL_AppResult switch_scene(int step);
    SDL_AppResult switch_precision();
    SDL_AppResult switch_strategy(int step);
    SDL_AppResult switch_optix();
    SDL_AppResult switched(const std::expected<void, std::string>& used, const std::string& what);
    void change_target(bool more);
    gpu::display_options scene_display() const;
    gpu::display_options shown_display() const;
    void step_display(int step);
    void change_exposure(std::optional<double> step);
    void display_changed(const gpu::display_options& before);
    void draw_overlay();
    void say(std::string message);
};

viewer::~viewer()
{
    session_.reset(); // the GPU's memory first, while CUDA is still up
    optix_.reset();   // then OptiX's programs and context, after every session that traced with them
    if (texture_)
    {
        SDL_DestroyTexture(texture_);
    }
    if (renderer_)
    {
        SDL_DestroyRenderer(renderer_);
    }
    if (window_)
    {
        SDL_DestroyWindow(window_);
    }
}

// Prints an error on the console, and in a message box when someone is watching the window, and ends the program
SDL_AppResult viewer::fail(const std::string& message)
{
    std::println("Error: {}", message);
    if (interactive() && window_)
    {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "RayTracingViewer", message.c_str(), window_);
    }
    return SDL_APP_FAILURE;
}

// Builds the scene with these settings and a new session for it, unless they are the ones in use. A new scene starts at
// its own view: the options' view on the scene of the command line, the scene's alone on another one, and the samples
// to refine to are its own (or -s); another precision, kernel, sampling or seed keeps the camera and the samples to
// refine to, and other strata keep the camera. Building a scene starts the random numbers over (see build_gpu_scene),
// so a scene built again is the same scene. The scene is built on the CPU first; the old session goes only then, before
// the new one, so the GPU never holds two scenes. If the new session fails, the scene that was shown comes back in a
// session of its own, its image started over; if even that fails, there is no session, and the caller ends.
std::expected<void, std::string> viewer::use_settings(const render_settings& settings)
{
    // The display is checked also when the settings stay, for a script's tonemap line before its next frame
    if (session_ && settings == settings_)
    {
        return check_display_for(settings);
    }
    auto built = build_for(settings);
    if (!built)
    {
        return std::unexpected(built.error());
    }
    return use_build(settings, std::move(*built));
}

// The display these settings would show: a tone map beyond the books computes in float32 only (as T and F2 keep it)
std::expected<void, std::string> viewer::check_display_for(const render_settings& settings) const
{
    const gpu_scene_entry* entry = scene_entry(settings.scene); // a model's file: the model entry's (AgX)
    if (const gpu::display_options shown = effective_display(display_, entry ? entry->display : gpu::display_options{});
        settings.fp64 && shown.op != gpu::tone_map::books)
    {
        return std::unexpected(std::format("{} works in float32 only", tone_map_name(shown.op)));
    }
    return {};
}

// The scene these settings build, as a new session would render it, or why the window can't show it, while nothing
// changes yet (M, Tab and O ask it of the settings they could go to): the display (check_display_for), the scene's own
// errors, what the quality kernel can't render (the float-only rule, the books' strategies but the material's), and
// what OptiX can't: a scene with triangle meshes refuses float64 and the other kernels (check_meshes), and a session
// that traces with OptiX takes its rules and needs it started (open_optix). The glTF files the build read are reported
// (report_models), also when the window then can't show the scene.
std::expected<gpu_build, std::string> viewer::build_for(const render_settings& settings)
{
    if (const auto shown = check_display_for(settings); !shown)
    {
        return std::unexpected(shown.error());
    }
    const bool own = settings.scene == opts_.scene;
    std::expected<gpu_build, std::string> built = std::unexpected(std::string());
    try
    {
        built = build_gpu_scene(settings.scene, {.sampling = settings.sampling,
                                                 .view = own ? opts_.view : view_options{},
                                                 .strata = settings.strata});
    }
    catch (const std::exception& error) // out of memory
    {
        return std::unexpected(error.what());
    }
    if (!built)
    {
        return built;
    }
    report_models(*built);
    if (const auto checked = check_meshes(*built, settings.scene, settings.kernel, settings.fp64); !checked)
    {
        return std::unexpected(checked.error());
    }
    const bool optix = traces_with_optix(settings, *built);
    if (const auto checked =
            check_kernel(*built, settings.scene, optix ? kernel_choice::optix : settings.kernel, settings.fp64);
        !checked)
    {
        return std::unexpected(checked.error());
    }
    if (const auto opened = optix ? open_optix() : std::expected<void, std::string>{}; !opened)
    {
        return std::unexpected(std::format("OptiX can't run here: {}", opened.error()));
    }
    return built;
}

// OptiX for the sessions that trace with it, started at the first need, after CUDA (start_optix: the graphics driver's
// OptiX and a context on CUDA's device; the programs compile at the first session's upload). A start that failed isn't
// tried again: its reason (no OptiX in this build, a driver too old, none in the driver) is the answer from then on.
std::expected<void, std::string> viewer::open_optix()
{
    if (optix_)
    {
        return {};
    }
    if (!optix_failure_)
    {
        try
        {
            optix_ = gpu::start_optix();
            return {};
        }
        catch (const std::exception& error)
        {
            optix_failure_ = error.what();
        }
    }
    return std::unexpected(*optix_failure_);
}

// The first session that traces with OptiX has the graphics driver compile OptiX's programs, seconds of the processor
// in which the window stands still (once per driver while OptiX's disk cache keeps them; at every start when the cache
// is off): a note says so first, on the console and, over the scene shown, on screen (in ASCII, the only characters
// SDL's debug font draws)
void viewer::note_optix_compile()
{
    const std::string note = !optix_ || gpu::optix_cache_enabled(*optix_)
                                 ? "compiling OptiX's programs (once per driver)..."
                                 : "compiling OptiX's programs (OptiX's program cache is off: at every start)...";
    std::println("Note: {}", note);
    say(note);
    if (window_ && texture_ && session_ && !minimized_ && !upload_needed_)
    {
        draw_and_present(false); // shows the note now: the window draws nothing more until the programs are compiled
    }
}

// What a build read from glTF files (gpu_scene's cache keeps the last two models read, so Tab and M read a file again
// only once it has left the cache or changed): each model's warnings, what the renderer can't show, once per run, as
// RayTracingGPU prints them before it renders; and with --stats a line per file read, a comment between the frames'
// rows ("# model read: FILE, ..."), its triangles as placed, its images and the milliseconds reading it took
void viewer::report_models(const gpu_build& built)
{
    for (const model_load& model : built.models)
    {
        if (warned_models_.insert(model.file).second)
        {
            for (const std::string& warning : model.warnings)
            {
                std::println("Warning: {}", warning);
            }
        }
        if (stats_.is_open())
        {
            stats_ << std::format("# model read: {}, {} triangles, {} images, {:.3f} ms\n", model.file,
                                  model.triangles, model.images, model.seconds * 1e3);
            stats_.flush();
        }
    }
}

// The kernel O turns OptiX off to: the command line's (the kernel each scene picks, unless --quality-kernel or
// --all-books-kernel picked one for all)
kernel_choice viewer::command_line_kernel() const
{
    return opts_.quality_kernel     ? kernel_choice::quality
           : opts_.all_books_kernel ? kernel_choice::all_books
                                    : kernel_choice::automatic;
}

// use_settings with the scene its settings built (build_for): the new session, or the old scene back
std::expected<void, std::string> viewer::use_build(const render_settings& settings, gpu_build built)
{
    const bool same_scene = session_ && settings.scene == settings_.scene;
    camera_settings view = built.cpu.camera; // the scene's sampling and samples, with its view
    if (same_scene)
    {
        view.lookfrom = view_.lookfrom;
        view.lookat = view_.lookat;
        view.vfov = view_.vfov;
        view.defocus_angle = view_.defocus_angle;
        view.focus_dist = view_.focus_dist;
    }
    // A new session: a new image, which the console reports when it is done, at a rate of its own, and the time the
    // session took to build is no frame's. One that traces with OptiX gets a copy of the runtime (build_for started
    // it), which it keeps alive.
    const auto session_for = [&](const gpu_build& b, const camera_settings& v, const render_settings& s) {
        gpu::scene_desc desc = b.desc;
        desc.camera = to_gpu(camera(settings_for(v, width_, height_)).frame(), b.sampling);
        session_.emplace(desc, gpu::session_options{.seed = s.seed,
                                                    .arithmetic = s.fp64 ? gpu::precision::float64
                                                                         : gpu::precision::float32,
                                                    .all_books_kernel = s.kernel == kernel_choice::all_books,
                                                    .quality_kernel = s.kernel == kernel_choice::quality,
                                                    .launch_seconds = still_seconds,
                                                    .wait = gpu::wait_mode::sleep,
                                                    .optix_tracer = traces_with_optix(s, b),
                                                    .optix = optix_});
        ++generation_;
        done_reported_ = false;
        samples_per_second_ = 0;
        skip_interval_ = true;
    };
    const bool optix = traces_with_optix(settings, built);
    if (optix && !optix_compiled_)
    {
        note_optix_compile();
    }
    session_.reset();
    try
    {
        session_for(built, view, settings);
    }
    catch (const std::exception& error)
    {
        if (build_)
        {
            try
            {
                session_for(*build_, view_, settings_);
            }
            catch (const std::exception&)
            {
                // no session: the caller ends the program
            }
        }
        return std::unexpected(error.what());
    }
    optix_compiled_ = optix_compiled_ || optix;
    if (!same_scene)
    {
        pending_view_.reset(); // a move of the old scene's camera still waiting
    }
    const bool keep_target = same_scene && settings.strata == settings_.strata;
    build_ = std::move(built);
    settings_ = settings;
    start_view_ = build_->cpu.camera;
    view_ = view;
    min_distance_ = (start_view_.lookfrom - start_view_.lookat).length() / 1000;
    min_focus_ = start_view_.focus_dist / 1000;
    if (window_)
    {
        update_title();
    }
    // The target: the samples asked for, or the scene's rounded up to whole rounds, so that a finished image never
    // stops inside a round (whose samples lean toward the top of the pixel, see gpu::session)
    if (!keep_target)
    {
        const std::int64_t round = std::int64_t{settings.strata} * settings.strata;
        target_ = opts_.samples_per_pixel ? *opts_.samples_per_pixel : (build_->passes + round - 1) / round * round;
    }
    return {};
}

// The view's settings for an image of width × height pixels, with the exact height (see aspect_for_height): the
// settings RayTracingGPU's options -w width --height height give the camera
camera_settings viewer::settings_for(const camera_settings& view, int width, int height) const
{
    camera_settings settings = view;
    settings.image_width = width;
    settings.aspect_ratio = aspect_for_height(width, height);
    return settings;
}

// Points the session's camera at `view` for an image of width × height pixels. The image starts over when the camera
// changes, and when `restart` asks for it with the same camera (a recorded frame's new generation); a session left
// without a view by a failed set_camera gets the camera again.
std::expected<void, std::string> viewer::show_view(const camera_settings& view, int width, int height, bool restart)
{
    const camera_settings settings = settings_for(view, width, height);
    if (const auto checked = check_view(settings); !checked)
    {
        return checked;
    }
    const gpu::camera_desc desc = to_gpu(camera(settings).frame(), build_->sampling);
    view_ = view;
    width_ = width;
    height_ = height;
    if (restart || !session_->has_view() || !(desc == session_->camera()))
    {
        try
        {
            session_->set_camera(desc);
        }
        catch (const std::exception& error)
        {
            return std::unexpected(error.what());
        }
        ++generation_;
    }
    return {};
}

// A script's size in the window: the window takes it, or the script stops (its frames would have another size than
// the script's). Without a window, only the image takes it.
std::expected<void, std::string> viewer::size_window(int width, int height)
{
    if (headless() || (width == width_ && height == height_))
    {
        return {};
    }
    SDL_SetWindowSize(window_, width, height);
    SDL_SyncWindow(window_);
    int w = 0;
    int h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    if (w != width || h != height)
    {
        return std::unexpected(
            std::format("the window can't be {}×{} (64×36 to the screen's size): use --headless", width, height));
    }
    pending_width_ = 0; // the size changes here, not from the window's event
    return {};
}

// The RayTracingGPU command that renders the image of `passes` samples of the view on screen again, bit for bit (a
// model's file quoted; --optix where OptiX traces a scene without triangle meshes, which imply it)
std::string viewer::gpu_command(std::int64_t passes) const
{
    std::string command =
        std::format("RayTracingGPU {} {} -s {} --strata {} --seed {}", scene_argument(settings_.scene),
                    format_view(settings_for(view_, width_, height_), height_), passes, settings_.strata,
                    settings_.seed);
    if (settings_.sampling)
    {
        command += std::format(" -p {}", strategy_name(*settings_.sampling));
    }
    if (settings_.fp64)
    {
        command += " --fp64";
    }
    if (settings_.kernel == kernel_choice::all_books)
    {
        command += " --all-books-kernel";
    }
    else if (settings_.kernel == kernel_choice::quality)
    {
        command += " --quality-kernel";
    }
    else if (settings_.kernel == kernel_choice::optix && !has_meshes(*build_))
    {
        command += " --optix";
    }
    // The display, wherever it or the scene's own isn't the books' bytes (see names_display), with the exposure's
    // shortest digits that read back as the same number
    if (const gpu::display_options shown = shown_display(); names_display(shown, scene_display()))
    {
        command += std::format(" --tonemap {} --exposure {}", tone_map_name(shown.op), shown.exposure);
    }
    return command;
}

// A camera move: taken when the camera can render the new view (see check_view), and ignored otherwise, as a move
// past a limit is. The view waits for the next iterate (apply_pending_view), so all the moves of a batch of events,
// a fast mouse's many motions, start the image over once.
bool viewer::move_camera(const camera_settings& to)
{
    if (!check_view(settings_for(to, width_, height_)))
    {
        return false;
    }
    pending_view_ = to;
    moving_until_ns_ = SDL_GetTicksNS() + moving_ns;
    return true;
}

// The view focused on the surface seen at pixel (x, y): the CPU's scene, the very objects the GPU renders, is asked
// where the ray through the pixel's center (at time 0) first hits something, and the distance is measured along the
// view's direction, as the focus distance is. The ray is the camera's at a focus distance of 1: the same direction
// whatever the focus, so the same click gives the same hit, bit for bit, and the image doesn't start over. Smoke and
// mist (an isotropic medium) don't stop the pick: it goes on to the surface behind them, and takes the medium's own
// point only when nothing is behind (the second book's final scene stands in a mist that fills the whole world). They
// are hit at random depths, so the random numbers start over first. Empty if the ray hits nothing.
//
// A scene with triangle meshes holds what the CPU's world doesn't (the CPU builds no triangle): there the session's
// own tracer finds the surface, one thread of OptiX's programs on the image's acceleration structures
// (session::pick_distance), triangles, spheres and quads alike, as the image sees them.
std::optional<camera_settings> viewer::pick_focus(double x, double y)
{
    const camera_settings view = current_view();
    camera_settings unit = settings_for(view, width_, height_);
    unit.focus_dist = 1;
    const camera_frame f = camera(unit).frame();
    // Clamped as doubles first: a script's X of 1e10 means the last column, not an int's overflow
    const int i = static_cast<int>(std::clamp(x, 0.0, width_ - 1.0));
    const int j = static_cast<int>(std::clamp(y, 0.0, height_ - 1.0));
    const point3 target = f.pixel00_loc + i * f.pixel_delta_u + j * f.pixel_delta_v;
    const ray r{.origin = f.center, .direction = target - f.center, .time = 0};
    if (has_meshes(*build_))
    {
        return pick_focus_optix(view, r, i, j);
    }
    seed_random(std::mt19937_64::default_seed);
    const auto medium = [](const std::optional<hit_record>& h) { return dynamic_cast<const isotropic*>(h->mat); };
    auto hit = build_->cpu.world.hit(r, interval{0.001, infinity});
    std::optional<hit_record> first_medium;
    for (int k = 0; hit && medium(hit) && k < 64; ++k)
    {
        if (!first_medium)
        {
            first_medium = hit;
        }
        const ray on{.origin = hit->p, .direction = r.direction, .time = 0};
        hit = build_->cpu.world.hit(on, interval{0.001, infinity});
    }
    if (!hit || medium(hit))
    {
        hit = first_medium;
    }
    if (!hit)
    {
        std::println("Focus: nothing there at pixel ({}, {})", i, j);
        say("nothing there to focus on");
        return std::nullopt;
    }
    const double along = dot(hit->p - view.lookfrom, unit_vector(view.lookat - view.lookfrom));
    const camera_settings to = with_focus(view, along);
    if (to.focus_dist == along)
    {
        std::println("Focus: {} at pixel ({}, {}), the hit at {},{},{}", to.focus_dist, i, j, hit->p.x, hit->p.y,
                     hit->p.z);
    }
    else
    {
        std::println("Focus: {} at pixel ({}, {}): the hit at {},{},{} is {} along the view, outside what a view "
                     "allows",
                     to.focus_dist, i, j, hit->p.x, hit->p.y, hit->p.z, along);
    }
    return to;
}

// pick_focus in a scene with triangle meshes: OptiX's pick of the nearest surface along the ray's unit direction (a
// distance t, ∞ for none), which the line on the console gives too, so that a check can compare it with the focus.
// Media aren't surfaces to it; the scenes with meshes have none.
std::optional<camera_settings> viewer::pick_focus_optix(const camera_settings& view, const ray& r, int i, int j)
{
    const vec3 d = unit_vector(r.direction);
    double t = 0;
    try
    {
        t = session_->pick_distance({r.origin.x, r.origin.y, r.origin.z}, {d.x, d.y, d.z});
    }
    catch (const std::exception& error)
    {
        std::println("Error: {}", error.what());
        say(std::format("can't focus: {}", error.what()));
        return std::nullopt;
    }
    if (!std::isfinite(t))
    {
        std::println("Focus: nothing there at pixel ({}, {})", i, j);
        say("nothing there to focus on");
        return std::nullopt;
    }
    const point3 p = r.origin + t * d;
    const double along = dot(p - view.lookfrom, unit_vector(view.lookat - view.lookfrom));
    const camera_settings to = with_focus(view, along);
    if (to.focus_dist == along)
    {
        std::println("Focus: {} at pixel ({}, {}), the hit at {},{},{}, {} along the ray (OptiX's pick)",
                     to.focus_dist, i, j, p.x, p.y, p.z, t);
    }
    else
    {
        std::println("Focus: {} at pixel ({}, {}): the hit at {},{},{}, {} along the ray (OptiX's pick), is {} along "
                     "the view, outside what a view allows",
                     to.focus_dist, i, j, p.x, p.y, p.z, t, along);
    }
    return to;
}

// The camera the last events moved, and the window's new size, given to the session
std::expected<void, std::string> viewer::apply_pending_view()
{
    const int width = pending_width_ > 0 ? pending_width_ : width_;
    const int height = pending_width_ > 0 ? pending_height_ : height_;
    pending_width_ = 0;
    if (pending_view_ || width != width_ || height != height_)
    {
        const camera_settings view = current_view();
        pending_view_.reset();
        return show_view(view, width, height, false);
    }
    return {};
}

// A fly key held: W A S D Q E
bool viewer::flying() const
{
    return held_[SDL_SCANCODE_W] || held_[SDL_SCANCODE_A] || held_[SDL_SCANCODE_S] || held_[SDL_SCANCODE_D] ||
           held_[SDL_SCANCODE_Q] || held_[SDL_SCANCODE_E];
}

// A script's move at t from 0 (the view before the move) to 1 (the whole move)
camera_settings viewer::moved(const move& m, double t) const
{
    switch (m.kind)
    {
    case move_kind::orbit:
        return orbit(move_start_, t * m.a, t * m.b);
    case move_kind::look:
        return look(move_start_, t * m.a, t * m.b);
    case move_kind::pan:
        return pan(move_start_, t * m.a, t * m.b, height_);
    case move_kind::dolly:
        return dolly(move_start_, std::pow(m.a, t), min_distance_, min_focus_);
    case move_kind::zoom:
        return zoom(move_start_, std::pow(m.a, t));
    }
    std::unreachable();
}

// The view the events moved to, with its pass in progress finished (at least one pass): the image P saves and C's
// command renders, shown in the window too
std::expected<gpu::accumulation, std::string> viewer::complete_pass()
{
    if (const auto applied = apply_pending_view(); !applied)
    {
        return std::unexpected(applied.error());
    }
    try
    {
        const gpu::accumulation state = session_->state();
        if (state.passes == 0 || state.rows > 0)
        {
            skip_interval_ = true;
            session_->render_to({state.passes + 1, 0});
            if (const auto kept = keep_frame(session_->display(shown_display())); !kept)
            {
                return std::unexpected(kept.error());
            }
            present_needed_ = true;
        }
        return session_->state();
    }
    catch (const std::exception& error)
    {
        return std::unexpected(error.what());
    }
}

// P: the image of the view on screen saved in the current folder as <scene>-view-NNN.png and .pfm, NNN the first
// number no file has yet (nothing earlier is overwritten), and the RayTracingGPU command that renders it again, printed
void viewer::save_view()
{
    const auto state = complete_pass();
    if (!state)
    {
        std::println("Error: {}", state.error());
        return;
    }
    const auto img = to_image(session_->pixels(), width_, height_);
    if (!img)
    {
        std::println("Error: {}", img.error());
        return;
    }
    // A tone map's PNG holds the bytes on screen, as RayTracingGPU's does (see bytes_to_image); the PFM the linear
    // values
    const gpu::display_options shown = shown_display();
    std::optional<image> toned;
    try
    {
        if (shown.op != gpu::tone_map::books)
        {
            toned = bytes_to_image(session_->display(shown), width_, height_);
        }
    }
    catch (const std::exception& error)
    {
        std::println("Error: {}", error.what());
        return;
    }
    std::string name;
    do
    {
        name = std::format("{}-view-{:03}", scene_stem(settings_.scene), ++views_saved_);
    } while (std::filesystem::exists(name + ".png") || std::filesystem::exists(name + ".pfm"));
    for (const std::string_view extension : {".png", ".pfm"})
    {
        const std::filesystem::path file = name + std::string(extension);
        const image& source = toned && extension == ".png" ? *toned : *img;
        if (const auto saved = source.save(file); !saved)
        {
            std::println("Error: {} ({})", saved.error(), to_utf8(file));
            return;
        }
    }
    std::println("Saved {} and .pfm ({} samples per pixel). The same image:",
                 to_utf8(std::filesystem::absolute(name + ".png")), state->passes);
    // With a tone map the command saves the PNG too, whose bytes it renders again. A model's file name may hold a space
    // or parentheses ("model (1).glb"): the files' names are then quoted, as the scene's is.
    std::println("{}{} -o {}", gpu_command(state->passes),
                 toned ? std::format(" -o {}", command_word(name + "-gpu.png")) : "", command_word(name + "-gpu.pfm"));
}

// A message on screen for two seconds. A loop waiting for events wouldn't take it off, so a timer wakes it then.
void viewer::say(std::string message)
{
    constexpr std::uint64_t shown_ns = 2'000'000'000;
    note_ = std::move(message);
    note_until_ns_ = SDL_GetTicksNS() + shown_ns;
    present_needed_ = true;
    if (window_)
    {
        SDL_AddTimerNS(shown_ns + 1'000'000, wake_up, nullptr); // a millisecond after the note's time (a timer that
                                                                // fails to start leaves the note until the next event)
    }
}

// The scene's line on the console, at the start and after a switch
void viewer::print_settings() const
{
    const std::string_view sampling = strategy_name(build_->sampling);
    // The quality kernel's integrator is named, in the quality kernel or in OptiX's programs with their version and
    // SHA-256 (the books' two kernels, as before, aren't): the image is its own
    std::string renderer;
    if (session_ && traces_with_optix(settings_, *build_))
    {
        renderer = std::format(", {}", optix_ ? gpu::describe(*optix_) : std::string("OptiX"));
    }
    else if (session_ && gpu::uses_quality_integrator(session_->kernel()))
    {
        renderer = ", the quality kernel";
    }
    const std::string precision = std::string(settings_.fp64 ? "float64" : "float32") + renderer;
    if (opts_.script)
    {
        // The script decides the frames and the views
        std::println("Scene: {}, sampling: {}, rounds of {}×{} strata, {}", settings_.scene, sampling,
                     settings_.strata, settings_.strata, precision);
    }
    else
    {
        std::println("Scene: {}, sampling: {}, {} samples per pixel in rounds of {}×{} strata, {}", settings_.scene,
                     sampling, target_, settings_.strata, settings_.strata, precision);
        std::println("View: {}", format_view(settings_for(view_, width_, height_), height_));
    }
    // A tone map is named, as RayTracingGPU names it; the books' bytes, as before, aren't
    if (const gpu::display_options shown = shown_display(); shown.op != gpu::tone_map::books)
    {
        std::println("Display: {}, {}", tone_map_name(shown.op), format_exposure(shown.exposure));
    }
}

// Tab, Shift+Tab: the next or the previous scene (see use_settings), at the window's size. It goes through the books'
// scenes, then the scenes beyond the books; from a test scene, Tab goes to the first of them and Shift+Tab to the
// last. The strategy named (-p, M) goes along, as a display chosen does: a scene that refuses it (a books' strategy
// but the material's, beyond the books) renders with its own instead and says so, and the next scene is asked for the
// named one again. A scene the window can't show here is skipped, the console saying why: the scenes beyond the books
// in float64 and under --all-books-kernel (only the quality kernel renders them, in float32), and, under
// --quality-kernel with no strategy named, a scene whose own strategy is a books' one but the material's.
//
// The scenes with triangle meshes, which only OptiX renders, are in the cycle of a session that traces with OptiX
// (--optix, O, a mesh scene or a model's file shown), after sunlit, where the list has them: a session without OptiX
// goes through the very scenes it went through before them, in the same order. Tab from a session with OptiX keeps it
// for the next scene, as --optix would, so the cycle stays the longer one until O turns OptiX off; a scene OptiX can't
// render is skipped as above (a books' strategy of its own, or OptiX that can't start, with its reason). A model's
// file is outside the cycle: Tab goes from it to the first scene and Shift+Tab to the last, as from a test scene.
SDL_AppResult viewer::switch_scene(int step)
{
    const bool optix = traces_with_optix(settings_, *build_);
    const auto in_cycle = [optix](const gpu_scene_entry& e) {
        return (e.role == scene_role::book || e.role == scene_role::showcase) && (optix || !has_triangle_meshes(e));
    };
    const std::vector<const gpu_scene_entry*> cycle = gpu_scene_list() | std::views::filter(in_cycle) |
                                                      std::views::transform([](const auto& e) { return &e; }) |
                                                      std::ranges::to<std::vector>();
    const auto count = static_cast<std::ptrdiff_t>(cycle.size());
    std::ptrdiff_t at = std::ranges::find(cycle, settings_.scene, &gpu_scene_entry::name) - cycle.begin();
    if (at == count)
    {
        at = step > 0 ? -1 : count;
    }
    std::string skipped;
    std::string why; // the first skipped scene's reason, on screen
    const auto skip = [&](const std::string& scene, const std::string& reason, std::string_view shown) {
        std::println("Note: {} skipped: {}", scene, reason);
        skipped += std::format("{}{}", skipped.empty() ? "" : ", ", scene);
        if (why.empty())
        {
            why = shown;
        }
    };
    for (std::ptrdiff_t k = 1; k <= count; ++k)
    {
        const gpu_scene_entry& entry = *cycle[static_cast<std::size_t>(((at + step * k) % count + count) % count)];
        render_settings next = settings_;
        next.scene = std::string(entry.name);
        next.sampling = named_sampling_;
        if (optix)
        {
            next.kernel = kernel_choice::optix;
        }
        if (next.scene == settings_.scene)
        {
            break; // around the cycle
        }
        // The scenes beyond the books, which the quality kernel renders, in float32 only: refused without a build
        if (entry.role == scene_role::showcase && next.kernel == kernel_choice::all_books)
        {
            skip(next.scene, "the quality kernel renders it, and --all-books-kernel picks the books' kernel",
                 "the quality kernel's, not --all-books-kernel's");
            continue;
        }
        if (entry.role == scene_role::showcase && next.fp64)
        {
            skip(next.scene, "the quality kernel renders it, in float32 only", "float32 only: F2 first");
            continue;
        }
        auto built = build_for(next);
        std::string own_note;
        if (!built && next.sampling)
        {
            render_settings own = next;
            own.sampling.reset();
            auto own_built = build_for(own);
            if (!own_built)
            {
                skip(next.scene, own_built.error(), own_built.error());
                continue;
            }
            std::println("Note: {}: {}; its own strategy instead", next.scene, built.error());
            own_note = std::format("{} can't take {}", next.scene, strategy_name(*next.sampling));
            next = std::move(own);
            built = std::move(own_built);
        }
        if (!built)
        {
            skip(next.scene, built.error(), built.error());
            continue;
        }
        const SDL_AppResult result =
            switched(use_build(next, std::move(*built)), std::format("can't show {}", next.scene));
        if (result == SDL_APP_CONTINUE && session_ && settings_ == next && (!own_note.empty() || !skipped.empty()))
        {
            const std::string skip_note = skipped.empty() ? "" : std::format("skipped {}: {}", skipped, why);
            say(own_note.empty() ? skip_note
                : skipped.empty()
                    ? std::format("sampling: {}, the scene's own ({})", strategy_name(build_->sampling), own_note)
                    : std::format("{}; sampling: {}, the scene's own ({})", skip_note,
                                  strategy_name(build_->sampling), own_note));
        }
        return result;
    }
    const std::string note = std::format("no other scene renders here (skipped {}: {})", skipped, why);
    std::println("Note: {}", note);
    say(note);
    return SDL_APP_CONTINUE;
}

// F2: the other precision, in a new session at the same view (the moves still waiting are shown first). The quality
// kernel's integrator (in the quality kernel or in OptiX), and a tone map other than books, compute in float32 only:
// there F2 stays and says so, with the way out.
SDL_AppResult viewer::switch_precision()
{
    if (!settings_.fp64 && session_ && gpu::uses_quality_integrator(session_->kernel()))
    {
        // A book scene renders with the quality kernel under nee and mis (or --quality-kernel); the others always do.
        // OptiX was asked for (O turns it off), or the scene's triangle meshes need it.
        const gpu_scene_entry* entry = scene_entry(settings_.scene);
        const std::string note =
            has_meshes(*build_) ? std::format("{} has triangle meshes, which only OptiX renders, in float32",
                                              scene_label(settings_.scene))
            : settings_.kernel == kernel_choice::optix ? "OptiX computes in float32 only: O to turn it off first"
            : settings_.kernel == kernel_choice::quality ? "--quality-kernel works in float32 only"
            : entry && entry->role == scene_role::book
                ? "the quality kernel works in float32 only: M to a books' strategy first"
                : std::format("{} renders with the quality kernel, in float32 only", settings_.scene);
        std::println("Note: {}", note);
        say(note);
        return SDL_APP_CONTINUE;
    }
    if (const gpu::display_options shown = shown_display(); !settings_.fp64 && shown.op != gpu::tone_map::books)
    {
        const std::string note = std::format("{} works in float32 only: T to books first", tone_map_name(shown.op));
        std::println("Note: {}", note);
        say(note);
        return SDL_APP_CONTINUE;
    }
    if (const auto applied = apply_pending_view(); !applied)
    {
        return fail(applied.error());
    }
    render_settings next = settings_;
    next.fp64 = !next.fp64;
    return switched(use_settings(next), std::format("can't render in {}", next.fp64 ? "float64" : "float32"));
}

// M, Shift+M: the next or the previous sampling strategy, in a new session at the same view: the scene's own, then the
// GPU programs' list (uniform, material, lights, mixture, nee, mis). The scene's own and the same strategy named
// differ across Tab, which takes a strategy named along. One the scene can't take as asked is skipped, named in the
// note and, with the reason, on the console: refused (a books' strategy but the material's beyond the books or under
// --quality-kernel; nee and mis in float64 or under --all-books-kernel), or one the scene would fall back from (lights
// and mixture without a lights list, nee and mis without a lamp or a map).
SDL_AppResult viewer::switch_strategy(int step)
{
    std::vector<std::optional<strategy>> cycle{std::nullopt};
    for (const strategy_entry& entry : strategy_list)
    {
        cycle.emplace_back(entry.value);
    }
    const auto count = static_cast<std::ptrdiff_t>(cycle.size());
    const std::ptrdiff_t at = std::ranges::find(cycle, settings_.sampling) - cycle.begin();
    std::string skipped;
    bool float_only = false; // nee or mis refused in float64, which F2 would allow
    for (std::ptrdiff_t k = 1; k < count; ++k)
    {
        render_settings next = settings_;
        next.sampling = cycle[static_cast<std::size_t>(((at + step * k) % count + count) % count)];
        auto built = build_for(next);
        if (!built || (next.sampling && built->sampling != *next.sampling))
        {
            const std::string_view name =
                next.sampling ? strategy_name(*next.sampling) : std::string_view("the scene's own");
            std::println("Note: {} skipped: {}", name,
                         built ? std::format("{} renders it as {}", next.scene, strategy_name(built->sampling))
                               : built.error());
            skipped += std::format("{}{}", skipped.empty() ? "" : ", ", name);
            float_only = float_only || (!built && next.fp64 && next.kernel != kernel_choice::all_books &&
                                        next.sampling && !entry_of(*next.sampling).books);
            continue;
        }
        if (const auto applied = apply_pending_view(); !applied)
        {
            return fail(applied.error());
        }
        const SDL_AppResult result = switched(use_build(next, std::move(*built)), "can't change the strategy");
        if (result == SDL_APP_CONTINUE && session_ && settings_ == next)
        {
            named_sampling_ = next.sampling;
            const std::string_view hint = float_only ? "; nee and mis in float32 only: F2 first" : "";
            say(std::format("sampling: {}{}{}", strategy_name(build_->sampling),
                            next.sampling ? "" : ", the scene's own",
                            skipped.empty() ? "" : std::format(" (skipped: {}{})", skipped, hint)));
        }
        return result;
    }
    const std::string note = std::format("no other strategy renders this scene here (skipped: {})", skipped);
    std::println("Note: {}", note);
    say(note);
    return SDL_APP_CONTINUE;
}

// O: OptiX on or off for the scene shown, in a new session at the same view (the moves still waiting are shown first),
// as F2 changes the precision: on, OptiX's programs trace the quality kernel's integrator (--optix); off, the scene
// renders with the kernel of the command line. Where it can't change, O stays and says why: a scene with triangle
// meshes always traces with OptiX; OptiX computes in float32 only; a strategy it can't take (a books' one but the
// material's), and OptiX that can't start here, with its reason (build_for).
SDL_AppResult viewer::switch_optix()
{
    const auto refused = [this](const std::string& note) {
        std::println("Note: {}", note);
        say(note);
        return SDL_APP_CONTINUE;
    };
    if (has_meshes(*build_))
    {
        return refused(std::format("{} has triangle meshes: always OptiX", scene_label(settings_.scene)));
    }
    if (settings_.fp64)
    {
        return refused("OptiX computes in float32 only: F2 first");
    }
    if (const auto applied = apply_pending_view(); !applied)
    {
        return fail(applied.error());
    }
    render_settings next = settings_;
    next.kernel = settings_.kernel == kernel_choice::optix ? command_line_kernel() : kernel_choice::optix;
    auto built = build_for(next);
    if (!built)
    {
        return refused(built.error());
    }
    const bool on = next.kernel == kernel_choice::optix;
    const SDL_AppResult result =
        switched(use_build(next, std::move(*built)), std::format("can't turn OptiX {}", on ? "on" : "off"));
    if (result == SDL_APP_CONTINUE && session_ && settings_ == next)
    {
        say(on ? "OptiX on" : "OptiX off");
    }
    return result;
}

// After a switch: the new scene's line on the console; or the error, on the console and on screen, over the scene that
// was shown, which use_settings brought back. If not even that scene could have a session again, the program ends.
SDL_AppResult viewer::switched(const std::expected<void, std::string>& used, const std::string& what)
{
    if (!used)
    {
        if (!session_)
        {
            return fail(std::format("{}: {}; the scene shown before couldn't come back either", what, used.error()));
        }
        std::println("Error: {}", used.error());
        say(std::format("{}: {}", what, used.error()));
        return SDL_APP_CONTINUE;
    }
    print_settings();
    moving_until_ns_ = SDL_GetTicksNS() + moving_ns; // the new image's first frames come quickly
    return SDL_APP_CONTINUE;
}

// Enter, Shift+Enter: twice or half as many samples to refine to, in whole rounds of strata, from one round up to
// 2³¹ - 1 samples. A target that isn't whole rounds (-s) is rounded up first, but never past it the wrong way: twice
// never fewer, half never more.
void viewer::change_target(bool more)
{
    const std::int64_t round = std::int64_t{settings_.strata} * settings_.strata;
    const std::int64_t rounds = (target_ + round - 1) / round;
    const std::int64_t most = std::max<std::int64_t>(std::numeric_limits<std::int32_t>::max() / round, 1);
    const std::int64_t next = std::clamp(more ? rounds * 2 : rounds / 2, std::int64_t{1}, most) * round;
    target_ = more ? std::max(next, target_) : std::min(next, target_);
    const std::int64_t passes = session_->state().passes;
    if (passes < target_)
    {
        done_reported_ = false; // the console reports it when it gets there
    }
    say(passes >= target_ ? std::format("{} samples: the image has them", target_)
        : paused_         ? std::format("{} samples to refine to (Space goes on)", target_)
                          : std::format("refining to {} samples", target_));
}

// The scene's own display (its entry in gpu_scenes.ixx: the books' bytes for the books' scenes; a model's file: AgX)
gpu::display_options viewer::scene_display() const
{
    const gpu_scene_entry* entry = scene_entry(settings_.scene);
    return entry ? entry->display : gpu::display_options{};
}

// The display the window shows: the one chosen, else the scene's own
gpu::display_options viewer::shown_display() const
{
    return effective_display(display_, scene_display());
}

// T, Shift+T: the next or the previous tone map. Those beyond the books compute in float only, so in float64 the
// window keeps the books' bytes and says so.
void viewer::step_display(int step)
{
    const gpu::display_options before = shown_display();
    const gpu::tone_map next = step_tone_map(before.op, step);
    if (settings_.fp64 && next != gpu::tone_map::books)
    {
        const std::string note = std::format("{} works in float32 only: F2 first", tone_map_name(next));
        std::println("Note: {}", note);
        say(note);
        return;
    }
    display_.op = next;
    display_changed(before);
}

// , and .: the exposure half a stop down or up, within ±24 EV; 0 (an empty step): back to 0 EV. The books' bytes have
// no exposure: under books the keys change nothing and say so.
void viewer::change_exposure(std::optional<double> step)
{
    const gpu::display_options before = shown_display();
    if (before.op == gpu::tone_map::books)
    {
        const std::string note = "exposure applies to a tone map: T picks one";
        std::println("Note: {}", note);
        say(note);
        return;
    }
    display_.exposure = step ? std::clamp(before.exposure + *step, -max_exposure, max_exposure) : 0.0;
    display_changed(before);
}

// After a key changed the display: the console's line, and the image shown again through it at the next iterate,
// with the numbers on screen. Nothing when it stayed as it was (an exposure at its limit, or 0 EV again).
void viewer::display_changed(const gpu::display_options& before)
{
    const gpu::display_options now = shown_display();
    if (now == before)
    {
        return;
    }
    std::println("Display: {}, {}", tone_map_name(now.op), format_exposure(now.exposure));
    redisplay_ = true;
    present_needed_ = true;
}

// The numbers at the top of the window, a note at the bottom and the help, in SDL's built-in 8×8 font, scaled with the
// display: three short lines on a dark bar, and the controls over the dimmed image. No device facts: the window shows
// the image's numbers.
void viewer::draw_overlay()
{
    static constexpr std::string_view help[] = {
        "H, F1, Esc          close this help",
        "left drag           orbit about the point looked at",
        "right drag          look around",
        "middle drag         pan",
        "Shift+left drag     pan",
        "wheel               move closer or farther",
        "Ctrl+wheel, - =     zoom",
        "Shift+wheel         focus nearer or farther",
        "double-click        focus on what is under the cursor",
        "[ ]                 a narrower or wider lens",
        "W A S D Q E         fly (Shift faster, Ctrl slower)",
        "R                   back to the start view",
        "Tab, Shift+Tab      the next or previous scene",
        "M, Shift+M          the next or previous strategy",
        "F2                  float32 or float64",
        "O                   OptiX on or off",
        "T, Shift+T          the next or previous tone map",
        ", .                 exposure -/+ 0.5 EV",
        "0                   exposure back to 0 EV",
        "Space               pause or go on refining",
        "Enter, Shift+Enter  twice or half the samples",
        "P                   save the image, print its command",
        "C                   copy the RayTracingGPU command",
        "I                   the numbers on screen",
    };
    constexpr float line = 10; // 8 pixels of text and 2 of space
    constexpr float bar_height = 3 * line + 6;
    constexpr float note_height = line + 6;
    const auto w = static_cast<float>(width_);
    const auto h = static_cast<float>(height_);
    // The quality kernel's integrator's strategy is named on the first line ("  mis"), as a tone map is on the third
    // ("  agx +0.5 EV"), and OptiX after it ("  optix"): the books' scenes, under their own kernels and bytes, show
    // the numbers they showed
    const std::string strategy = session_ && gpu::uses_quality_integrator(session_->kernel())
                                     ? std::format("  {}", strategy_name(build_->sampling))
                                     : std::string();
    const std::string_view optix = session_ && traces_with_optix(settings_, *build_) ? "  optix" : "";
    const gpu::display_options shown = shown_display();
    const std::string tone = shown.op == gpu::tone_map::books
                                 ? std::string()
                                 : std::format("  {} {}", tone_map_name(shown.op), format_exposure(shown.exposure));
    // The numbers and the note at twice the display's scale (SDL's font is 8 pixels), but no larger than lets the
    // widest line, about 56 characters (66 with a tone map's name on the third), and both bars fit the window; at
    // least 1
    const float widest = static_cast<float>((tone.empty() ? 56 : 66) * 8 + 16);
    const float scale = std::max(1.0f, std::min({std::round(2 * SDL_GetWindowDisplayScale(window_)),
                                                 std::floor(w / widest), std::floor(h / (bar_height + note_height + 8))}));
    // The help between the numbers and the note, at their scale or smaller where the window is short (the numbers keep
    // their size when it opens); at 1 it shows as many lines as fit, the way out first
    const float top = (hud_ ? bar_height + 6 : 8) * scale; // in the window's pixels
    const float help_height = static_cast<float>(std::size(help)) * line + 4;
    const float help_scale = std::max(1.0f, std::min(scale, std::floor((h - top - note_height * scale) / help_height)));
    SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);
    const auto text = [&](float x, float y, const std::string& s) {
        SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255);
        SDL_RenderDebugText(renderer_, x, y, s.c_str());
    };
    // The help first, under the numbers and the note
    if (help_)
    {
        SDL_SetRenderScale(renderer_, help_scale, help_scale);
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 200);
        const SDL_FRect box{0, 0, w / help_scale, h / help_scale};
        SDL_RenderFillRect(renderer_, &box);
        float y = top / help_scale;
        for (const std::string_view entry : help)
        {
            text(8, y, std::string(entry));
            y += line;
        }
    }
    SDL_SetRenderScale(renderer_, scale, scale);
    const float width = w / scale;
    const float height = h / scale;
    if (hud_)
    {
        // A script's frames have the samples it asks for, and it takes no keys: no target, no "Enter: more"
        const gpu::accumulation state = session_->state();
        const bool live = !opts_.script;
        std::string progress = live ? std::format("{} / {} samples", state.passes, target_)
                                    : std::format("{} samples", state.passes);
        if (live && paused_)
        {
            progress += "   paused (Space)";
        }
        else if (live && state.passes >= target_)
        {
            progress += "   done (Enter: more)";
        }
        else if (samples_per_second_ > 0)
        {
            progress += std::format("   {:.1f} M samples/s   frame {:.1f} ms", samples_per_second_ / 1e6, frame_ms_);
        }
        const std::string lines[] = {
            std::format("{}  {}x{}  {}  strata {}x{}{}{}", scene_label(settings_.scene), width_, height_,
                        settings_.fp64 ? "float64" : "float32", settings_.strata, settings_.strata, strategy, optix),
            progress,
            std::format("vfov {:.4g}  aperture {:.4g}  focus {:.4g}{}{}", view_.vfov, view_.defocus_angle,
                        view_.focus_dist, tone, live ? "   H: help" : ""),
        };
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 160);
        const SDL_FRect bar{0, 0, width, bar_height};
        SDL_RenderFillRect(renderer_, &bar);
        for (int i = 0; i < 3; ++i)
        {
            text(4, 4 + i * line, lines[i]);
        }
    }
    if (!note_.empty() && SDL_GetTicksNS() < note_until_ns_)
    {
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 160);
        const SDL_FRect bar{0, height - line - 6, width, line + 6};
        SDL_RenderFillRect(renderer_, &bar);
        text(4, height - line - 2, note_);
    }
    SDL_SetRenderScale(renderer_, 1, 1);
}

SDL_AppResult viewer::init()
{
    // The settings of the command line; a script's, up to its first frame, come first (so the session that starts is
    // the script's, and the window has the script's size from the start)
    render_settings settings{.scene = std::string(opts_.scene),
                             .fp64 = opts_.fp64,
                             .kernel = opts_.optix ? kernel_choice::optix : command_line_kernel(),
                             .sampling = opts_.sampling,
                             .seed = opts_.seed,
                             .strata = opts_.strata};
    std::optional<std::pair<int, int>> script_size;
    if (opts_.script)
    {
        const auto text = read_text(*opts_.script);
        if (!text)
        {
            return fail(text.error());
        }
        auto parsed = parse_script(*text);
        if (!parsed)
        {
            return fail(std::format("{}: {}", to_utf8(*opts_.script), parsed.error()));
        }
        script_ = std::move(*parsed);
        for (const script_line& line : script_)
        {
            const script_command& c = line.command;
            if (const auto* scene = std::get_if<set_scene>(&c))
            {
                if (const auto known = check_scene(scene->name); !known)
                {
                    return fail(std::format("{}: line {}: {}", to_utf8(*opts_.script), line.line, known.error()));
                }
                settings.scene = scene->name;
            }
            else if (const auto* size = std::get_if<set_size>(&c))
            {
                script_size = {size->width, size->height};
            }
            else if (const auto* f = std::get_if<frame>(&c))
            {
                script_size = {f->width, f->height};
                break;
            }
            else if (renders(c) || changes_view(c))
            {
                break;
            }
            else if (const auto* seed = std::get_if<set_seed>(&c))
            {
                settings.seed = seed->seed;
            }
            else if (const auto* strata = std::get_if<set_strata>(&c))
            {
                settings.strata = strata->strata;
            }
            else if (const auto* precision = std::get_if<set_precision>(&c))
            {
                settings.fp64 = precision->fp64;
            }
            else if (const auto* kernel = std::get_if<set_kernel>(&c))
            {
                settings.kernel = kernel->kernel;
            }
            else if (const auto* sampling = std::get_if<set_sampling>(&c))
            {
                settings.sampling = sampling->strategy;
            }
            else if (const auto* tonemap = std::get_if<set_tonemap>(&c))
            {
                display_.op = tonemap->op;
            }
            else if (const auto* exposure = std::get_if<set_exposure>(&c))
            {
                display_.exposure = exposure->exposure;
            }
        }
    }

    // The size: the script's, or the options', or (with a window) the default one. --height alone: the scene's own
    // aspect ratio at that many rows (the scene's size is built without it).
    const bool height_alone = opts_.view.image_height && !opts_.image_width;
    view_options sized = opts_.view;
    if (height_alone)
    {
        sized.image_height.reset();
    }
    const bool own_scene = settings.scene == opts_.scene; // the options' size and view are the command line scene's
    auto built = build_gpu_scene(settings.scene, {.image_width = own_scene ? opts_.image_width : std::optional<int>{},
                                                  .samples_per_pixel = opts_.samples_per_pixel,
                                                  .sampling = settings.sampling,
                                                  .view = own_scene ? sized : view_options{},
                                                  .strata = settings.strata});
    if (!built)
    {
        return fail(built.error());
    }
    // The kernel asked for, refused here as RayTracingGPU refuses it: before the files, the window and CUDA
    if (opts_.all_books_kernel && opts_.quality_kernel)
    {
        return fail(
            "--all-books-kernel picks the books' kernel for all three books; --quality-kernel needs the quality kernel");
    }
    if (opts_.optix && opts_.quality_kernel)
    {
        return fail("--optix and --quality-kernel each pick a renderer: give one");
    }
    if (opts_.optix && opts_.all_books_kernel)
    {
        return fail("--all-books-kernel picks the books' kernel for all three books; --optix renders with OptiX");
    }
    // A scene with triangle meshes (a model's file among them) renders with OptiX without --optix: what it can't be
    // rendered with is said first, in its own words (see check_meshes in gpu_scene.ixx)
    if (const auto checked = check_meshes(*built, settings.scene, settings.kernel, settings.fp64); !checked)
    {
        return fail(checked.error());
    }
    const bool optix = traces_with_optix(settings, *built);
    if (const auto checked =
            check_kernel(*built, settings.scene, optix ? kernel_choice::optix : settings.kernel, settings.fp64);
        !checked)
    {
        return fail(checked.error());
    }
    // The display, refused as RayTracingGPU refuses it: an exposure for the books' bytes, a tone map in float64
    if (const auto checked = check_display(opts_.display, scene_entry(settings.scene)->display, settings.fp64);
        !checked)
    {
        return fail(checked.error());
    }
    // and a script's display before its first frame, in float64 (later ones: use_settings, at their frame's line)
    if (const gpu::display_options shown = effective_display(display_, scene_entry(settings.scene)->display);
        opts_.script && settings.fp64 && shown.op != gpu::tone_map::books)
    {
        return fail(std::format("{}: {} works in float32 only", to_utf8(*opts_.script), tone_map_name(shown.op)));
    }
    // OptiX, for --optix (or a script's kernel optix) and for a scene with triangle meshes, and only for them, so that
    // every other run is as before. It starts on the device CUDA starts, after query_device below; a build without
    // OptiX's headers refuses here instead, before the window and CUDA, in RayTracingGPU's words (start_optix() says
    // why, and starts nothing).
    const auto optix_refused = [&](const std::string& reason) {
        return fail(opts_.optix ? std::format("--optix: {}", reason)
                    : has_meshes(*built)
                        ? std::format("'{}' has triangle meshes, which only OptiX renders: {}", settings.scene, reason)
                        : std::format("OptiX can't run here: {}", reason));
    };
    if (const auto opened = optix && !gpu::optix_built() ? open_optix() : std::expected<void, std::string>{};
        !opened)
    {
        return optix_refused(opened.error());
    }
    width_ = built->frame.image_width;
    height_ = built->frame.image_height;
    if (height_alone)
    {
        height_ = *opts_.view.image_height;
        width_ = std::max(1, static_cast<int>(std::lround(height_ * built->cpu.camera.aspect_ratio)));
    }
    if (script_size)
    {
        std::tie(width_, height_) = *script_size;
    }
    const bool explicit_size = opts_.image_width || opts_.view.image_height || script_size;
    if (opts_.input)
    {
        const auto text = read_text(*opts_.input);
        if (!text)
        {
            return fail(text.error());
        }
        auto parsed = parse_input(*text);
        if (!parsed)
        {
            return fail(std::format("{}: {}", to_utf8(*opts_.input), parsed.error()));
        }
        input_ = std::move(*parsed);
    }
    if (opts_.frames)
    {
        std::error_code error;
        std::filesystem::create_directories(*opts_.frames, error);
        if (error)
        {
            return fail(std::format("can't create the folder {}: {}", to_utf8(*opts_.frames), error.message()));
        }
    }
    if (opts_.stats)
    {
        stats_.open(*opts_.stats);
        if (!stats_)
        {
            return fail(std::format("can't write {}", to_utf8(*opts_.stats)));
        }
        stats_ << "frame,wall_ms,dt_ms,gpu_ms,launches,longest_launch_ms,passes,rows,new_samples,presented,moving,"
                  "generation,width,height,strata,time_ms\n";
    }
    // The glTF files the first build read (the session's own build below finds them in gpu_scene's cache)
    report_models(*built);
    if (opts_.record)
    {
        record_.open(*opts_.record, std::ios::binary);
        if (!record_)
        {
            return fail(std::format("can't write {}", to_utf8(*opts_.record)));
        }
    }

    if (!headless())
    {
        // A double-click focuses: its two clicks within 4 pixels, as a desktop's are (SDL's default of 32 would take
        // two quick short orbit drags for one)
        SDL_SetHint(SDL_HINT_MOUSE_DOUBLE_CLICK_RADIUS, "4");
        if (!SDL_Init(SDL_INIT_VIDEO))
        {
            return fail(std::format("SDL couldn't start: {}", SDL_GetError()));
        }
        if (!explicit_size)
        {
            std::tie(width_, height_) = default_size(built->cpu.camera.aspect_ratio, SDL_GetPrimaryDisplay());
        }
        const int asked_width = width_;
        const int asked_height = height_;
        window_ = SDL_CreateWindow(std::format("RayTracingViewer: {}", scene_label(settings.scene)).c_str(), width_,
                                   height_, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
        if (!window_)
        {
            return fail(std::format("SDL couldn't open the window: {}", SDL_GetError()));
        }
        // Dragged smaller, the window keeps 64×36 pixels at least, or the size asked for if that is smaller
        SDL_SetWindowMinimumSize(window_, std::min(64, width_), std::min(36, height_));
        renderer_ = SDL_CreateRenderer(window_, nullptr);
        if (!renderer_)
        {
            return fail(std::format("SDL couldn't create a renderer for the window: {}", SDL_GetError()));
        }
        SDL_SyncWindow(window_);
        if (!SDL_GetWindowSizeInPixels(window_, &width_, &height_))
        {
            return fail(std::format("SDL couldn't tell the window's size: {}", SDL_GetError()));
        }
        // A size asked for that the screen can't hold: Windows makes the window smaller. Someone watching sees it and
        // goes on at that size; a run that ends by itself (--exit-when-done, a script, scripted input) would save or
        // record another size than the one asked for, which is an error.
        if (explicit_size && (width_ != asked_width || height_ != asked_height))
        {
            if (!interactive())
            {
                return fail(std::format("the window can't be {}×{} on this screen (it became {}×{}){}", asked_width,
                                        asked_height, width_, height_, opts_.input ? "" : ": use --headless"));
            }
            std::println("Note: the window can't be {}×{} on this screen; the image is {}×{}", asked_width,
                         asked_height, width_, height_);
        }
        pacing_ = pacing_for(opts_.refresh_hz.value_or(display_refresh_hz(window_)));
        if (!SDL_SetRenderVSync(renderer_, pacing_for(display_refresh_hz(window_)).vsync) &&
            !SDL_SetRenderVSync(renderer_, 1))
        {
            return fail(std::format("SDL couldn't turn on vsync: {}", SDL_GetError()));
        }
        wake_type = SDL_RegisterEvents(2);
        if (wake_type == 0)
        {
            return fail(std::format("SDL has no events left for the program: {}", SDL_GetError()));
        }
        timed_input_type = wake_type + 1;
        // --input: key names SDL knows (the `after MS` events' timers start with the clock, below)
        for (const input_event& e : input_)
        {
            if (!e.key.empty() && SDL_GetKeyFromName(e.key.c_str()) == SDLK_UNKNOWN)
            {
                return fail(std::format("{}: line {}: SDL knows no key '{}'", to_utf8(*opts_.input), e.line, e.key));
            }
        }
    }

    try
    {
        gpu::query_device(gpu::wait_mode::sleep);
    }
    catch (const std::exception& error)
    {
        return fail(error.what());
    }
    if (const auto opened = optix ? open_optix() : std::expected<void, std::string>{}; !opened)
    {
        return optix_refused(opened.error());
    }
    if (const auto used = use_settings(settings); !used)
    {
        return fail(used.error());
    }
    print_settings();
    start_ns_ = SDL_GetTicksNS();
    last_iterate_ns_ = start_ns_;
    if (headless())
    {
        return run_headless();
    }
    moving_until_ns_ = start_ns_ + moving_ns; // the first frames come quickly
    // --input's `after MS` events, from now: the moment --stats' time_ms counts from too
    for (std::size_t i = 0; i < input_.size(); ++i)
    {
        if (input_[i].after_ms >= 0 &&
            !SDL_AddTimerNS(static_cast<Uint64>(input_[i].after_ms * 1e6), timed_input,
                            reinterpret_cast<void*>(static_cast<std::intptr_t>(i))))
        {
            return fail(std::format("SDL couldn't start a timer: {}", SDL_GetError()));
        }
    }
    if (!make_texture(width_, height_))
    {
        return SDL_APP_FAILURE;
    }
    if (const auto sent = send_input(0); !sent)
    {
        return fail(sent.error());
    }
    return SDL_APP_CONTINUE;
}

bool viewer::make_texture(int width, int height)
{
    if (texture_)
    {
        SDL_DestroyTexture(texture_);
        texture_ = nullptr;
    }
    // RGBA32 is the byte order R, G, B, A in memory, the session's display(), and the renderer's own format, so SDL
    // copies the bytes as they are. Drawn 1:1 without blending or filtering, the window shows exactly those bytes.
    texture_ = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, width, height);
    if (!texture_ || !SDL_SetTextureScaleMode(texture_, SDL_SCALEMODE_NEAREST) ||
        !SDL_SetTextureBlendMode(texture_, SDL_BLENDMODE_NONE))
    {
        fail(std::format("SDL couldn't create the image's texture: {}", SDL_GetError()));
        return false;
    }
    texture_width_ = width;
    texture_height_ = height;
    upload_needed_ = true;
    return true;
}

bool viewer::upload_shown()
{
    if (shown_.empty() || shown_width_ != texture_width_ || shown_height_ != texture_height_)
    {
        return true; // nothing of this size to show yet
    }
    if (!SDL_UpdateTexture(texture_, nullptr, shown_.data(), shown_width_ * 4))
    {
        fail(std::format("SDL couldn't update the image: {}", SDL_GetError()));
        return false;
    }
    upload_needed_ = false;
    return true;
}

bool viewer::draw_and_present(bool new_samples)
{
    SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
    SDL_RenderClear(renderer_);
    if (!shown_.empty() && shown_width_ == texture_width_ && shown_height_ == texture_height_)
    {
        const SDL_FRect where{0, 0, static_cast<float>(texture_width_), static_cast<float>(texture_height_)};
        SDL_RenderTexture(renderer_, texture_, nullptr, &where);
    }
    if (new_samples && opts_.frames && !minimized_ && !opts_.frames_hud && !save_frame_from_window())
    {
        return false;
    }
    draw_overlay();
    if (new_samples && opts_.frames && !minimized_ && opts_.frames_hud && !save_frame_from_window())
    {
        return false;
    }
    // A present can fail without the window being lost (SDL then rebuilds its buffers); only a lost device ends the
    // program (see event)
    SDL_RenderPresent(renderer_);
    present_needed_ = false;
    return true;
}

// The window's pixels of the image, read back from the renderer before the present, as a PNG
bool viewer::save_frame_from_window()
{
    // The image's pixels, or with --frames-hud the window's
    const SDL_Rect rect =
        opts_.frames_hud ? SDL_Rect{0, 0, width_, height_} : SDL_Rect{0, 0, shown_width_, shown_height_};
    SDL_Surface* pixels = SDL_RenderReadPixels(renderer_, &rect);
    if (!pixels)
    {
        fail(std::format("SDL couldn't read the window's pixels: {}", SDL_GetError()));
        return false;
    }
    const std::filesystem::path file = *opts_.frames / std::format("f{:05}.png", frames_saved_++);
    const bool saved = SDL_SavePNG(pixels, to_utf8(file).c_str());
    SDL_DestroySurface(pixels);
    if (!saved)
    {
        fail(std::format("SDL couldn't save {}: {}", to_utf8(file), SDL_GetError()));
        return false;
    }
    return true;
}

// The display's bytes as a PNG, without a window (--headless): SDL's PNG writer, as for the window's frames
std::expected<void, std::string> viewer::save_frame_bytes(std::span<const std::uint8_t> bytes, int width, int height)
{
    SDL_Surface* surface = SDL_CreateSurfaceFrom(width, height, SDL_PIXELFORMAT_RGBA32,
                                                 const_cast<std::uint8_t*>(bytes.data()), width * 4);
    if (!surface)
    {
        return std::unexpected(std::format("SDL couldn't wrap the frame: {}", SDL_GetError()));
    }
    const std::filesystem::path file = *opts_.frames / std::format("f{:05}.png", frames_saved_++);
    const bool saved = SDL_SavePNG(surface, to_utf8(file).c_str());
    SDL_DestroySurface(surface);
    if (!saved)
    {
        return std::unexpected(std::format("SDL couldn't save {}: {}", to_utf8(file), SDL_GetError()));
    }
    return {};
}

void viewer::update_title()
{
    SDL_SetWindowTitle(window_, std::format("RayTracingViewer: {}", scene_label(settings_.scene)).c_str());
}

// The image of complete passes, as RayTracingGPU saves it: a pass in progress is finished first, so the file holds
// exactly the samples of `passes` and matches RayTracingGPU's render of that many
bool viewer::save_outputs()
{
    if (opts_.outputs.empty())
    {
        return true;
    }
    try
    {
        gpu::accumulation state = session_->state();
        if (state.passes == 0 || state.rows > 0)
        {
            state = session_->render_to({state.passes + 1, 0}).state;
        }
        const auto img = to_image(session_->pixels(), width_, height_);
        if (!img)
        {
            fail(img.error());
            return false;
        }
        // A tone map's PNG from the display's bytes, as RayTracingGPU's (see bytes_to_image); a PFM from the pixels
        const gpu::display_options shown = shown_display();
        std::optional<image> toned;
        if (shown.op != gpu::tone_map::books)
        {
            toned = bytes_to_image(session_->display(shown), width_, height_);
        }
        std::string errors; // every file that failed, in one message (and one message box)
        for (const std::filesystem::path& output : opts_.outputs)
        {
            const image& source = toned && is_png(output) ? *toned : *img;
            if (const auto saved = source.save(output); !saved)
            {
                errors += std::format("{}{} ({})", errors.empty() ? "" : "\n", saved.error(), to_utf8(output));
                continue;
            }
            std::println("Image saved: {} ({} samples per pixel)", to_utf8(std::filesystem::absolute(output)),
                         state.passes);
        }
        std::println("The same image: {}", gpu_command(state.passes));
        if (!errors.empty())
        {
            fail(errors);
            return false;
        }
        return true;
    }
    catch (const std::exception& error)
    {
        fail(error.what());
        return false;
    }
}

// A frame with new samples, or with the same ones through another display, after display(): kept for the window, and
// written by --record and by --frames (without a window, or with a minimized one, which presents nothing to read back:
// from the bytes, which the window shows as they are)
std::expected<void, std::string> viewer::keep_frame(std::span<const std::uint8_t> bytes)
{
    shown_.assign(bytes.begin(), bytes.end());
    shown_width_ = width_;
    shown_height_ = height_;
    upload_needed_ = true;
    record_frame();
    if (opts_.frames && (headless() || minimized_))
    {
        return save_frame_bytes(bytes, width_, height_);
    }
    return {};
}

// --record: the settings when they change, then the frame. The display's lines follow the settings when the display,
// or the scene's own, isn't the books' bytes (names_display), and once written they come with every change of the
// settings or of the display: a script's display holds for every scene after it (see viewer_script), so a replay shows
// each frame as the window did. A recording of the books' scenes that never left their bytes stays as it was.
void viewer::record_frame()
{
    if (!record_.is_open())
    {
        return;
    }
    const gpu::display_options shown = shown_display();
    const bool display_lines = recorded_display_ || names_display(shown, scene_display());
    const display_setting display{shown.op, shown.exposure};
    if (recorded_settings_ != settings_)
    {
        record_ << format_settings(settings_.scene, settings_.fp64, settings_.kernel, settings_.sampling,
                                   settings_.seed, settings_.strata,
                                   display_lines ? std::optional{display} : std::nullopt);
        recorded_settings_ = settings_;
        if (display_lines)
        {
            recorded_display_ = shown;
        }
    }
    else if (display_lines && recorded_display_ != shown)
    {
        // A display changed on the same settings: its lines alone, then a frame of the same passes and generation
        record_ << format_display(display);
        recorded_display_ = shown;
    }
    const gpu::accumulation state = session_->state();
    record_ << format_frame(state.passes, state.rows, generation_, width_, height_, view_);
    record_.flush();
}

// Plays the script up to its next frame and renders it. Returns false when the script is over.
std::expected<bool, std::string> viewer::play_next()
{
    render_settings settings = settings_;
    if (move_)
    {
        // A frame of a move: smoothstep(i / F), so the move starts and ends at rest
        const double s = static_cast<double>(++move_done_) / move_->frames;
        const camera_settings to = moved(*move_, s * s * (3 - 2 * s));
        const auto at = [&](const std::string& message) {
            return std::unexpected(std::format("{}: line {}: {}", to_utf8(*opts_.script), move_line_, message));
        };
        const std::uint64_t before = generation_;
        if (const auto shown = show_view(to, width_, height_, false); !shown)
        {
            return at(shown.error());
        }
        const std::int64_t passes = move_->passes;
        if (move_done_ == move_->frames)
        {
            move_.reset();
        }
        // N passes of the new view; a frame that doesn't change the camera keeps the image, N more passes on
        try
        {
            const gpu::accumulation now = session_->state();
            const std::int64_t target = generation_ != before ? passes : now.passes + passes;
            script_report_ = session_->render_to({target, 0});
            if (const auto kept = keep_frame(session_->display(shown_display())); !kept)
            {
                return at(kept.error());
            }
        }
        catch (const std::exception& error)
        {
            return at(error.what());
        }
        return true;
    }
    while (hold_left_ == 0)
    {
        if (script_next_ == script_.size())
        {
            return false;
        }
        const script_line& line = script_[script_next_++];
        const auto at = [&](const std::string& message) {
            return std::unexpected(std::format("{}: line {}: {}", to_utf8(*opts_.script), line.line, message));
        };
        if (std::holds_alternative<set_scene>(line.command))
        {
            const auto* c = &std::get<set_scene>(line.command);
            if (const auto known = check_scene(c->name); !known)
            {
                return at(known.error());
            }
            settings.scene = c->name;
        }
        else if (std::holds_alternative<set_size>(line.command))
        {
            const auto* c = &std::get<set_size>(line.command);
            if (const auto sized = size_window(c->width, c->height); !sized)
            {
                return at(sized.error());
            }
            if (session_)
            {
                if (const auto shown = show_view(view_, c->width, c->height, false); !shown)
                {
                    return at(shown.error());
                }
            }
            width_ = c->width;
            height_ = c->height;
        }
        else if (std::holds_alternative<set_seed>(line.command))
        {
            const auto* c = &std::get<set_seed>(line.command);
            settings.seed = c->seed;
        }
        else if (std::holds_alternative<set_strata>(line.command))
        {
            const auto* c = &std::get<set_strata>(line.command);
            settings.strata = c->strata;
        }
        else if (std::holds_alternative<set_precision>(line.command))
        {
            const auto* c = &std::get<set_precision>(line.command);
            settings.fp64 = c->fp64;
        }
        else if (std::holds_alternative<set_kernel>(line.command))
        {
            const auto* c = &std::get<set_kernel>(line.command);
            settings.kernel = c->kernel;
        }
        else if (std::holds_alternative<set_sampling>(line.command))
        {
            const auto* c = &std::get<set_sampling>(line.command);
            settings.sampling = c->strategy;
        }
        else if (std::holds_alternative<set_tonemap>(line.command))
        {
            // The display needs no new session: the next frame shows its image through it (a recorded frame of the
            // same passes and generation renders nothing, and shows the image again)
            display_.op = std::get<set_tonemap>(line.command).op;
        }
        else if (std::holds_alternative<set_exposure>(line.command))
        {
            display_.exposure = std::get<set_exposure>(line.command).exposure;
        }
        else if (std::holds_alternative<set_camera>(line.command))
        {
            const auto* c = &std::get<set_camera>(line.command);
            if (const auto used = use_settings(settings); !used)
            {
                return at(used.error());
            }
            camera_settings view = view_;
            view.lookfrom = c->view.lookfrom.value_or(view.lookfrom);
            view.lookat = c->view.lookat.value_or(view.lookat);
            view.vfov = c->view.vfov.value_or(view.vfov);
            view.defocus_angle = c->view.defocus_angle.value_or(view.defocus_angle);
            view.focus_dist = c->view.focus_dist.value_or(view.focus_dist);
            if (const auto shown = show_view(view, width_, height_, false); !shown)
            {
                return at(shown.error());
            }
        }
        else if (std::holds_alternative<focus_at>(line.command))
        {
            const auto* c = &std::get<focus_at>(line.command);
            if (const auto used = use_settings(settings); !used)
            {
                return at(used.error());
            }
            if (const auto to = pick_focus(c->x, c->y))
            {
                if (const auto shown = show_view(*to, width_, height_, false); !shown)
                {
                    return at(shown.error());
                }
            }
        }
        else if (std::holds_alternative<move>(line.command))
        {
            const auto* c = &std::get<move>(line.command);
            if (const auto used = use_settings(settings); !used)
            {
                return at(used.error());
            }
            move_ = *c;
            move_done_ = 0;
            move_line_ = line.line;
            move_start_ = view_;
            return play_next();
        }
        else if (std::holds_alternative<hold>(line.command))
        {
            const auto* c = &std::get<hold>(line.command);
            if (const auto used = use_settings(settings); !used)
            {
                return at(used.error());
            }
            hold_left_ = c->frames;
            hold_passes_ = c->passes;
        }
        else if (std::holds_alternative<frame>(line.command))
        {
            const auto* c = &std::get<frame>(line.command);
            // A recorded frame: that view, at its size, rendered to exactly where the window had it. A new generation
            // starts the image over even when the camera is the same (the window's R key, or a size changed and back),
            // and so does a frame behind the image: an image of N passes and R rows is the same whatever came before.
            if (const auto used = use_settings(settings); !used)
            {
                return at(used.error());
            }
            if (const auto sized = size_window(c->width, c->height); !sized)
            {
                return at(sized.error());
            }
            camera_settings view = view_;
            view.lookfrom = *c->view.lookfrom;
            view.lookat = *c->view.lookat;
            view.vfov = *c->view.vfov;
            view.defocus_angle = *c->view.defocus_angle;
            view.focus_dist = *c->view.focus_dist;
            const gpu::accumulation want{c->passes, c->rows};
            const bool restart =
                !script_generation_ || c->generation != *script_generation_ || want < session_->state();
            script_generation_ = c->generation;
            if (const auto shown = show_view(view, c->width, c->height, restart); !shown)
            {
                return at(shown.error());
            }
            try
            {
                script_report_ = session_->render_to(want);
                if (const auto kept = keep_frame(session_->display(shown_display())); !kept)
                {
                    return at(kept.error());
                }
            }
            catch (const std::exception& error)
            {
                return at(error.what());
            }
            return true;
        }
    }
    // A frame of a hold
    --hold_left_;
    try
    {
        const gpu::accumulation now = session_->state();
        script_report_ = session_->render_to({now.passes + hold_passes_, 0});
        if (const auto kept = keep_frame(session_->display(shown_display())); !kept)
        {
            return std::unexpected(kept.error());
        }
    }
    catch (const std::exception& error)
    {
        return std::unexpected(std::format("{}: {}", to_utf8(*opts_.script), error.what()));
    }
    return true;
}

// --headless: everything here, without a window; SDL_AppInit's result ends the program
SDL_AppResult viewer::run_headless()
{
    try
    {
        // --stats: a row per frame, its time from the end of the frame before (dt) and its own (wall)
        const auto row = [&](const gpu::render_report& report, std::uint64_t start) {
            const std::uint64_t now = SDL_GetTicksNS();
            write_stats(report, true, false, false, static_cast<double>(now - start) / 1e6,
                        static_cast<double>(now - last_iterate_ns_) / 1e6);
            last_iterate_ns_ = now;
            ++frame_;
        };
        if (opts_.script)
        {
            while (true)
            {
                const std::uint64_t start = SDL_GetTicksNS();
                const auto played = play_next();
                if (!played)
                {
                    return fail(played.error());
                }
                if (!*played)
                {
                    break;
                }
                row(script_report_, start);
            }
        }
        else
        {
            const std::uint64_t start = SDL_GetTicksNS();
            const gpu::render_report report = session_->render_to({target_, 0});
            if (const auto kept = keep_frame(session_->display(shown_display())); !kept)
            {
                return fail(kept.error());
            }
            row(report, start);
            // As RayTracingGPU counts them: a ray is a search for the closest hit, so per sample it is the average
            // length of a path; the speed is that of the whole render
            const double rays = static_cast<double>(session_->rays());
            const double samples = static_cast<double>(width_) * height_ * static_cast<double>(target_);
            const double seconds = static_cast<double>(SDL_GetTicksNS() - start) / 1e9;
            std::println("Rays traced: {:.1f} million ({:.2f} per sample), {:.1f} million per second", rays / 1e6,
                         rays / samples, rays / seconds / 1e6);
        }
        const gpu::accumulation state = session_->state();
        std::println("Done: {} samples per pixel, {}×{}, {:.2f} s", state.passes, width_, height_,
                     static_cast<double>(SDL_GetTicksNS() - start_ns_) / 1e9);
        return save_outputs() ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
    }
    catch (const std::exception& error)
    {
        return fail(error.what());
    }
}

// --input: the `at F` events of the frame about to come, sent before it (the `after MS` ones come from their timers)
std::expected<void, std::string> viewer::send_input(std::uint64_t frame)
{
    while (input_next_ < input_.size() && (input_[input_next_].at_frame < 0 ||
                                           static_cast<std::uint64_t>(input_[input_next_].at_frame) <= frame))
    {
        const input_event& e = input_[input_next_++];
        if (e.at_frame >= 0)
        {
            if (const auto performed = perform(e); !performed)
            {
                return performed;
            }
        }
    }
    return {};
}

// An input event as SDL would send it: keys by name (the same key on any keyboard layout) with the modifiers the pushed
// events hold down, the mouse with its position and buttons; a resize resizes the window itself
std::expected<void, std::string> viewer::perform(const input_event& e)
{
    SDL_Event event{};
    const SDL_WindowID window = SDL_GetWindowID(window_);
    switch (e.kind)
    {
    case input_kind::resize:
        // A real resize of the window, whose events then come as from a user's drag (the time it takes is no frame's)
        skip_interval_ = true;
        if (!SDL_SetWindowSize(window_, static_cast<int>(e.x), static_cast<int>(e.y)))
        {
            return std::unexpected(std::format("{}: line {}: SDL couldn't resize the window: {}",
                                               to_utf8(*opts_.input), e.line, SDL_GetError()));
        }
        SDL_SyncWindow(window_);
        return {};
    case input_kind::device_reset:
        event.type = SDL_EVENT_RENDER_DEVICE_RESET;
        break;
    case input_kind::quit:
        event.type = SDL_EVENT_QUIT;
        break;
    case input_kind::key_down:
    case input_kind::key_up:
    {
        const bool down = e.kind == input_kind::key_down;
        event.type = down ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
        event.key.windowID = window;
        event.key.key = SDL_GetKeyFromName(e.key.c_str());
        event.key.scancode = SDL_GetScancodeFromName(e.key.c_str());
        if (event.key.scancode == SDL_SCANCODE_UNKNOWN)
        {
            event.key.scancode = SDL_GetScancodeFromKey(event.key.key, nullptr);
        }
        const SDL_Keymod modifier = modifier_of(event.key.scancode);
        input_mods_ = down ? static_cast<SDL_Keymod>(input_mods_ | modifier)
                           : static_cast<SDL_Keymod>(input_mods_ & ~modifier);
        event.key.mod = input_mods_;
        event.key.down = down;
        event.key.which = scripted_device;
        break;
    }
    case input_kind::mouse_down:
    case input_kind::mouse_up:
    {
        const bool down = e.kind == input_kind::mouse_down;
        event.type = down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
        event.button.windowID = window;
        event.button.which = scripted_device;
        event.button.button = static_cast<Uint8>(e.button);
        event.button.down = down;
        event.button.clicks = static_cast<Uint8>(e.clicks);
        event.button.x = input_x_ = static_cast<float>(e.x);
        event.button.y = input_y_ = static_cast<float>(e.y);
        const SDL_MouseButtonFlags mask = SDL_BUTTON_MASK(e.button);
        input_buttons_ = down ? input_buttons_ | mask : input_buttons_ & ~mask;
        break;
    }
    case input_kind::mouse_move:
        event.type = SDL_EVENT_MOUSE_MOTION;
        event.motion.windowID = window;
        event.motion.which = scripted_device;
        event.motion.state = input_buttons_;
        event.motion.x = input_x_ = static_cast<float>(e.x);
        event.motion.y = input_y_ = static_cast<float>(e.y);
        event.motion.xrel = static_cast<float>(e.dx);
        event.motion.yrel = static_cast<float>(e.dy);
        break;
    case input_kind::wheel:
        event.type = SDL_EVENT_MOUSE_WHEEL;
        event.wheel.windowID = window;
        event.wheel.which = scripted_device;
        event.wheel.y = static_cast<float>(e.dy);
        event.wheel.integer_y = static_cast<Sint32>(e.dy);
        event.wheel.direction = SDL_MOUSEWHEEL_NORMAL;
        event.wheel.mouse_x = input_x_;
        event.wheel.mouse_y = input_y_;
        break;
    }
    if (!SDL_PushEvent(&event))
    {
        return std::unexpected(std::format("{}: line {}: SDL couldn't send the event: {}", to_utf8(*opts_.input),
                                           e.line, SDL_GetError()));
    }
    return {};
}

SDL_AppResult viewer::event(const SDL_Event& e)
{
    if (e.type == timed_input_type && timed_input_type != 0)
    {
        if (const auto performed = perform(input_[static_cast<std::size_t>(e.user.code)]); !performed)
        {
            return fail(performed.error());
        }
    }
    // Under --input, the keys and the mouse are the input file's alone: a real mouse or key would change what a test
    // does
    if (opts_.input)
    {
        const bool real = ((e.type == SDL_EVENT_KEY_DOWN || e.type == SDL_EVENT_KEY_UP) &&
                           e.key.which != scripted_device) ||
                          ((e.type == SDL_EVENT_MOUSE_BUTTON_DOWN || e.type == SDL_EVENT_MOUSE_BUTTON_UP) &&
                           e.button.which != scripted_device) ||
                          (e.type == SDL_EVENT_MOUSE_MOTION && e.motion.which != scripted_device) ||
                          (e.type == SDL_EVENT_MOUSE_WHEEL && e.wheel.which != scripted_device);
        if (real)
        {
            return SDL_APP_CONTINUE;
        }
    }
    switch (e.type)
    {
    case SDL_EVENT_QUIT:
        return save_outputs() ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
    {
        const bool down = e.type == SDL_EVENT_KEY_DOWN;
        // Movement keys by scancode, the key's place on the keyboard (W A S D sit together whatever the layout);
        // commands by keycode, the letter printed on the key (P is P). The modifiers as the event has them: both
        // Shift keys, both Ctrl keys.
        if (e.key.scancode < SDL_SCANCODE_COUNT)
        {
            if (down && !flying())
            {
                fly_since_ns_ = SDL_GetTicksNS(); // a flight starts now, not at the last frame
            }
            held_[e.key.scancode] = down;
        }
        mods_ = e.key.mod;
        if (!down)
        {
            break;
        }
        if (e.key.key == SDLK_ESCAPE && !e.key.repeat)
        {
            // Esc closes the help first, then lets the mouse go, and only then the window
            if (help_)
            {
                help_ = false;
                present_needed_ = true;
                break;
            }
            if (SDL_GetWindowRelativeMouseMode(window_))
            {
                SDL_SetWindowRelativeMouseMode(window_, false);
                drag_ = drag::none;
                break;
            }
            return save_outputs() ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
        }
        if (opts_.script)
        {
            break; // a script moves the camera alone
        }
        // - = [ ] by the letter on the key, or by their place on a US keyboard, right of 0 and of P (on many layouts
        // the brackets need AltGr, which the key event doesn't carry)
        const auto is = [&](SDL_Keycode key, SDL_Scancode place) {
            return e.key.key == key || e.key.scancode == place;
        };
        if (is(SDLK_MINUS, SDL_SCANCODE_MINUS) || is(SDLK_EQUALS, SDL_SCANCODE_EQUALS)) // held, they zoom on
        {
            move_camera(zoom(current_view(), std::pow(wheel_zoom, is(SDLK_MINUS, SDL_SCANCODE_MINUS) ? -1 : 1)));
        }
        else if (is(SDLK_LEFTBRACKET, SDL_SCANCODE_LEFTBRACKET) || is(SDLK_RIGHTBRACKET, SDL_SCANCODE_RIGHTBRACKET))
        {
            // held, they open or close the lens step by step
            move_camera(aperture_step(current_view(), is(SDLK_RIGHTBRACKET, SDL_SCANCODE_RIGHTBRACKET)));
        }
        if (e.key.repeat)
        {
            break;
        }
        if (e.key.key == SDLK_TAB || e.key.key == SDLK_F2)
        {
            // A switch that fails shows the scene it came from; one that leaves no session ends the program
            const SDL_AppResult result =
                e.key.key == SDLK_TAB ? switch_scene((mods_ & SDL_KMOD_SHIFT) ? -1 : 1) : switch_precision();
            if (result != SDL_APP_CONTINUE)
            {
                return result;
            }
        }
        else if (e.key.key == SDLK_SPACE)
        {
            paused_ = !paused_;
            present_needed_ = true;
        }
        else if (e.key.key == SDLK_RETURN || e.key.key == SDLK_KP_ENTER)
        {
            change_target(!(mods_ & SDL_KMOD_SHIFT));
        }
        else if (e.key.key == SDLK_I)
        {
            hud_ = !hud_;
            present_needed_ = true;
        }
        else if (e.key.key == SDLK_H || e.key.key == SDLK_F1)
        {
            help_ = !help_;
            present_needed_ = true;
        }
        else if (e.key.key == SDLK_R)
        {
            move_camera(start_view_);
        }
        else if (e.key.key == SDLK_P)
        {
            save_view();
        }
        else if (e.key.key == SDLK_C)
        {
            // P's command: the view on screen, its pass in progress finished
            if (const auto state = complete_pass(); state)
            {
                const std::string command = gpu_command(state->passes);
                if (SDL_SetClipboardText(command.c_str()))
                {
                    std::println("Copied: {}", command);
                }
                else
                {
                    std::println("Error: SDL couldn't copy the command ({}): {}", SDL_GetError(), command);
                }
            }
            else
            {
                std::println("Error: {}", state.error());
            }
        }
        // The display: the image shown again through it, without new samples (see redisplay_). , . and 0 by their
        // character or by their place on a US keyboard, as - = [ ] above
        else if (e.key.key == SDLK_T)
        {
            step_display((mods_ & SDL_KMOD_SHIFT) ? -1 : 1);
        }
        else if (e.key.key == SDLK_M)
        {
            // A switch that fails shows the strategy it came from; one that leaves no session ends the program
            if (const SDL_AppResult result = switch_strategy((mods_ & SDL_KMOD_SHIFT) ? -1 : 1);
                result != SDL_APP_CONTINUE)
            {
                return result;
            }
        }
        else if (e.key.key == SDLK_O)
        {
            // OptiX on or off: a switch that fails shows the session it came from; one that leaves none ends the
            // program
            if (const SDL_AppResult result = switch_optix(); result != SDL_APP_CONTINUE)
            {
                return result;
            }
        }
        else if (is(SDLK_COMMA, SDL_SCANCODE_COMMA) || is(SDLK_PERIOD, SDL_SCANCODE_PERIOD))
        {
            change_exposure(is(SDLK_PERIOD, SDL_SCANCODE_PERIOD) ? 0.5 : -0.5);
        }
        else if (is(SDLK_0, SDL_SCANCODE_0))
        {
            change_exposure(std::nullopt);
        }
        break;
    }
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        // A key or a button let go in another window sends nothing here: nothing counts as held any more, and the
        // cursor is free (SDL would capture it again when the window comes back). Under --input the state is the
        // input file's, which a desktop's focus doesn't change.
        if (!opts_.input)
        {
            held_.fill(false);
            mods_ = SDL_KMOD_NONE;
            drag_ = drag::none;
            drag_button_ = 0;
            SDL_SetWindowRelativeMouseMode(window_, false);
        }
        break;
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
        if (!opts_.input)
        {
            mods_ = SDL_GetModState(); // a Shift held while clicking into the window counts
        }
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        if (opts_.script || drag_ != drag::none)
        {
            break; // a second button during a drag does nothing
        }
        drag_button_ = e.button.button;
        if (e.button.button == SDL_BUTTON_LEFT && e.button.clicks == 2)
        {
            // A double-click focuses; it starts no drag
            drag_ = drag::none;
            drag_button_ = 0;
            const float density = SDL_GetWindowPixelDensity(window_);
            if (const auto to = pick_focus(e.button.x * density, e.button.y * density))
            {
                move_camera(*to);
            }
        }
        else if (e.button.button == SDL_BUTTON_LEFT)
        {
            drag_ = (mods_ & SDL_KMOD_SHIFT) ? drag::pan : drag::orbit;
        }
        else if (e.button.button == SDL_BUTTON_MIDDLE)
        {
            drag_ = drag::pan;
        }
        else if (e.button.button == SDL_BUTTON_RIGHT)
        {
            drag_ = drag::look;
            // The cursor hidden and held in the window while looking around, so the view can turn without end (not
            // under scripted input, whose motion doesn't come from a real mouse)
            if (!opts_.input)
            {
                SDL_SetWindowRelativeMouseMode(window_, true);
            }
        }
        else
        {
            drag_button_ = 0;
        }
        break;
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (drag_ != drag::none && e.button.button == drag_button_)
        {
            if (drag_ == drag::look)
            {
                SDL_SetWindowRelativeMouseMode(window_, false);
            }
            drag_ = drag::none;
            drag_button_ = 0;
        }
        break;
    case SDL_EVENT_MOUSE_MOTION:
        if (drag_ != drag::none)
        {
            // Window coordinates to pixels (they differ under display scaling on some systems)
            const float density = SDL_GetWindowPixelDensity(window_);
            const double dx = e.motion.xrel * density;
            const double dy = e.motion.yrel * density;
            // A quarter of a degree per pixel. Orbit and pan grab the scene: dragged to the right, it turns or moves to
            // the right. Look turns the head the way the mouse goes.
            move_camera(drag_ == drag::orbit  ? orbit(current_view(), -0.25 * dx, 0.25 * dy)
                        : drag_ == drag::look ? look(current_view(), -0.25 * dx, -0.25 * dy)
                                              : pan(current_view(), dx, dy, height_));
        }
        break;
    case SDL_EVENT_MOUSE_WHEEL:
        if (!opts_.script)
        {
            const double steps = e.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -e.wheel.y : e.wheel.y;
            const camera_settings from = current_view();
            if (mods_ & SDL_KMOD_CTRL)
            {
                move_camera(zoom(from, std::pow(wheel_zoom, steps)));
            }
            else if (mods_ & SDL_KMOD_SHIFT)
            {
                move_camera(refocus(from, std::pow(wheel_focus, steps)));
            }
            else
            {
                move_camera(dolly(from, std::pow(wheel_dolly, steps), min_distance_, min_focus_));
            }
        }
        break;
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        if (!opts_.script) // a script sets the size itself
        {
            pending_width_ = e.window.data1;
            pending_height_ = e.window.data2;
        }
        moving_until_ns_ = SDL_GetTicksNS() + moving_ns;
        break;
    case SDL_EVENT_WINDOW_MOVED:
    case SDL_EVENT_WINDOW_RESIZED:
        // While Windows moves or resizes the window, SDL calls iterate from inside its loop for the drag, which can't
        // move the window until iterate returns: the short launches of a moving view keep the window following the
        // mouse
        moving_until_ns_ = SDL_GetTicksNS() + moving_ns;
        break;
    case SDL_EVENT_WINDOW_EXPOSED:
        present_needed_ = true;
        break;
    case SDL_EVENT_WINDOW_MINIMIZED:
        minimized_ = true;
        break;
    case SDL_EVENT_WINDOW_RESTORED:
    case SDL_EVENT_WINDOW_MAXIMIZED:
        minimized_ = false;
        present_needed_ = true;
        break;
    case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
    case SDL_EVENT_DISPLAY_CURRENT_MODE_CHANGED:
        // Another display, or another rate: the pacing and the vsync interval follow (as in init)
        if (!opts_.refresh_hz)
        {
            pacing_ = pacing_for(display_refresh_hz(window_));
        }
        if (!SDL_SetRenderVSync(renderer_, pacing_for(display_refresh_hz(window_)).vsync))
        {
            SDL_SetRenderVSync(renderer_, 1);
        }
        break;
    case SDL_EVENT_RENDER_DEVICE_RESET:
        // The renderer's textures are gone: a new one, with the last frame shown
        if (!make_texture(texture_width_, texture_height_) || !upload_shown())
        {
            return SDL_APP_FAILURE;
        }
        present_needed_ = true;
        break;
    case SDL_EVENT_RENDER_DEVICE_LOST:
        return fail("the graphics device was lost");
    default:
        break;
    }
    // Any event may change what to do next: iterate again at once instead of waiting
    SDL_SetHint(SDL_HINT_MAIN_CALLBACK_RATE, "0");
    return SDL_APP_CONTINUE;
}

SDL_AppResult viewer::iterate()
{
    const std::uint64_t now = SDL_GetTicksNS();
    const double dt_ms = static_cast<double>(now - last_iterate_ns_) / 1e6;
    last_iterate_ns_ = now;
    // The last frame's interval, if that frame rendered and nothing else happened in it: the frame time on screen, and
    // the rate of the samples the frame rendered
    if (!skip_interval_ && last_samples_ > 0)
    {
        frame_ms_ = dt_ms;
        const double rate = last_samples_ / std::max(1e-6, dt_ms / 1000);
        samples_per_second_ = samples_per_second_ > 0 ? 0.8 * samples_per_second_ + 0.2 * rate : rate;
    }
    last_samples_ = 0;
    skip_interval_ = false;
    // The samples (pixels times passes) of an image `state` has come to
    const auto samples_in = [&](const gpu::accumulation& state) {
        return static_cast<double>(width_) * (static_cast<double>(state.passes) * height_ + state.rows);
    };
    try
    {
        gpu::render_report report;
        bool new_samples = false;
        bool redisplayed = false; // the same samples through another display: a frame too (--frames, --record)
        bool presented = false;
        bool moving = false;
        bool done = false;
        // Minimized, a window someone watches waits to be restored; a run that ends by itself goes on
        const bool minimized_pause = minimized_ && interactive();
        if (opts_.script)
        {
            // A script's frames, one per iterate, rendered to where the script says whatever the time
            const gpu::accumulation before = session_->state();
            const std::uint64_t before_generation = generation_;
            const auto played = play_next();
            if (!played)
            {
                return fail(played.error());
            }
            new_samples = *played;
            done = !*played;
            if (new_samples)
            {
                report = script_report_;
                // a new view's samples are all the frame's
                last_samples_ = samples_in(session_->state()) - (generation_ == before_generation ? samples_in(before)
                                                                                                    : 0);
            }
        }
        else
        {
            // Fly: half the distance to lookat per second, four times faster with Shift, four times slower with Ctrl;
            // the step covers the time since the last one, or since the key went down (at most a tenth of a second)
            const auto key = [&](SDL_Scancode s) { return held_[s] ? 1.0 : 0.0; };
            const double forward = key(SDL_SCANCODE_W) - key(SDL_SCANCODE_S);
            const double right = key(SDL_SCANCODE_D) - key(SDL_SCANCODE_A);
            const double up = key(SDL_SCANCODE_E) - key(SDL_SCANCODE_Q);
            if (flying())
            {
                const camera_settings from = current_view();
                const double speed = 0.5 * (from.lookfrom - from.lookat).length() *
                                     ((mods_ & SDL_KMOD_SHIFT) ? 4.0 : (mods_ & SDL_KMOD_CTRL) ? 0.25 : 1.0);
                const double step = speed * std::min(static_cast<double>(now - fly_since_ns_) / 1e9, 0.1);
                fly_since_ns_ = now;
                move_camera(fly(from, forward * step, right * step, up * step));
            }
            if (const auto applied = apply_pending_view(); !applied)
            {
                return fail(applied.error());
            }
            moving = now < moving_until_ns_;
            // Paused, the image doesn't refine, but a new view (a camera move, a new size) still gets its first pass
            const std::int64_t limit = paused_ ? std::min(target_, std::max<std::int64_t>(session_->state().passes, 1))
                                               : target_;
            if (!minimized_pause && session_->state().passes < limit)
            {
                const double budget = moving ? moving_share * pacing_.frame_seconds : still_seconds;
                const gpu::accumulation before = session_->state();
                report = session_->render_for(budget, limit);
                last_samples_ = samples_in(session_->state()) - samples_in(before);
                if (const auto kept = keep_frame(session_->display(shown_display())); !kept)
                {
                    return fail(kept.error());
                }
                new_samples = true;
                redisplay_ = false; // this frame has the display as it is now
            }
            else if (redisplay_ && !minimized_pause)
            {
                // A display changed on an image that doesn't refine (done, or paused): its samples once more, through
                // the new display, as a frame that adds none. keep_frame records it with the same passes, rows and
                // generation, which a replay renders by rendering nothing; the rendering stays asleep.
                if (const auto kept = keep_frame(session_->display(shown_display())); !kept)
                {
                    return fail(kept.error());
                }
                redisplay_ = false;
                redisplayed = true;
            }
            done = session_->state().passes >= target_;
        }
        // A frame to show and to save: new samples, or the same ones through another display
        const bool new_frame = new_samples || redisplayed;
        if (new_frame)
        {
            present_needed_ = true;
        }
        if (present_needed_ && !minimized_)
        {
            if (!shown_.empty() && (shown_width_ != texture_width_ || shown_height_ != texture_height_) &&
                !make_texture(shown_width_, shown_height_))
            {
                return SDL_APP_FAILURE;
            }
            if ((upload_needed_ && !upload_shown()) || !draw_and_present(new_frame))
            {
                return SDL_APP_FAILURE;
            }
            presented = true;
        }
        const double wall_ms = static_cast<double>(SDL_GetTicksNS() - now) / 1e6;
        if (new_samples)
        {
            if (generation_ != timed_generation_)
            {
                view_ms_ = 0;
                timed_generation_ = generation_;
            }
            view_ms_ += wall_ms;
        }
        write_stats(report, new_frame, presented, moving, wall_ms, dt_ms);
        ++frame_;
        if (const auto sent = send_input(frame_); !sent)
        {
            return fail(sent.error());
        }
        if (done && !done_reported_)
        {
            // The time the image's frames took (a script's: the whole script's), without waits for events
            done_reported_ = true;
            std::println("Done: {} samples per pixel in {:.2f} s", session_->state().passes,
                         opts_.script ? static_cast<double>(SDL_GetTicksNS() - start_ns_) / 1e9 : view_ms_ / 1000);
            if (opts_.exit_when_done || opts_.script)
            {
                return save_outputs() ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
            }
        }
        if (!note_.empty() && SDL_GetTicksNS() >= note_until_ns_)
        {
            note_.clear();
            present_needed_ = true;
        }
        // Nothing to render and nothing new to show (or nowhere to show it): SDL waits for the next event before it
        // calls iterate again, and the program sleeps (not while scripted input waits for its frames or a fly key is
        // held). The timers of the `after MS` events, and the one that takes a note off the screen, end the wait.
        if ((minimized_pause || ((done || paused_) && !present_needed_ && !flying())) && pending_width_ == 0 &&
            !pending_view_ && input_next_ == input_.size())
        {
            SDL_SetHint(SDL_HINT_MAIN_CALLBACK_RATE, "waitevent");
            skip_interval_ = true;
        }
        else
        {
            // Work is left. While the user drags the window's edge, SDL calls iterate from inside Windows' loop for the
            // drag, and that loop may have started inside SDL's wait for an event: the drag's own calls take every
            // event out of the queue, so when the drag ends the wait would go on, the image half refined, until the
            // next event. An event of our own ends it.
            SDL_Event wake{};
            wake.type = wake_type;
            SDL_PushEvent(&wake);
        }
    }
    catch (const std::exception& error)
    {
        return fail(error.what());
    }
    return SDL_APP_CONTINUE;
}

// --stats: a row per frame. new_samples: the frame has samples the one before didn't, or shows the same ones through
// another display (the frames --frames saves); presented: the window was drawn and presented; moving: the frame had a
// moving view's budget; time_ms: when the row was written, from the start (the times of --input's `after MS` events
// count from there too).
void viewer::write_stats(const gpu::render_report& report, bool new_samples, bool presented, bool moving,
                         double wall_ms, double dt_ms)
{
    if (!stats_.is_open())
    {
        return;
    }
    const gpu::accumulation state = session_->state();
    stats_ << std::format("{},{:.3f},{:.3f},{:.3f},{},{:.3f},{},{},{},{},{},{},{},{},{},{:.3f}\n", frame_, wall_ms,
                          dt_ms, report.gpu_seconds * 1e3, report.launches, report.longest_launch_seconds * 1e3,
                          state.passes, state.rows, new_samples ? 1 : 0, presented ? 1 : 0, moving ? 1 : 0, generation_,
                          width_, height_, settings_.strata, static_cast<double>(SDL_GetTicksNS() - start_ns_) / 1e6);
    stats_.flush(); // a run that is stopped still leaves its lines
}
}

// --- SDL's callbacks -------------------------------------------------------------------------------------------

SDL_AppResult SDL_AppInit(void** appstate, int argc, char* argv[])
{
    const auto opts = parse_options(std::span(argv, static_cast<std::size_t>(argc)));
    if (!opts)
    {
        std::println("Error: {}\n", opts.error());
        print_usage();
        return SDL_APP_FAILURE;
    }
    if (opts->help)
    {
        print_usage();
        return SDL_APP_SUCCESS;
    }
    auto* app = new viewer(*opts);
    *appstate = app;
    return app->init();
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event)
{
    return static_cast<viewer*>(appstate)->event(*event);
}

SDL_AppResult SDL_AppIterate(void* appstate)
{
    return static_cast<viewer*>(appstate)->iterate();
}

void SDL_AppQuit(void* appstate, SDL_AppResult)
{
    delete static_cast<viewer*>(appstate); // nullptr after a command line error or -h
}
