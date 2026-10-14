// RayTracingViewer: RayTracing's scenes in a window, rendered on the GPU and refined while the view stands still.
// The scene and the camera come from the same modules as RayTracingGPU's (gpu_scene, view_options), and the GPU renders
// with a session (gpu/renderer.h) that keeps the scene in its memory and adds samples pass after pass. The image
// converges to the image RayTracingGPU renders of the same view with the same samples, bit for bit.
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

import std;
import camera;
import cli;
import gpu_scene;
import image;
import scenes;
import view_options;

namespace
{
// --- The command line ------------------------------------------------------------------------------------------

constexpr std::string_view default_viewer_scene = "weekend";
constexpr int default_strata = 8; // a round of 8×8 samples per pixel (see --strata)

struct options
{
    std::string_view scene = default_viewer_scene;
    std::optional<int> image_width;
    std::optional<int> samples_per_pixel;
    int strata = default_strata;
    std::optional<sampling_strategy> sampling;
    view_options view;
    std::uint32_t seed = 0;
    bool fp64 = false;
    bool all_books_kernel = false;
    std::vector<std::filesystem::path> outputs; // saved when the window closes
    bool exit_when_done = false;
    std::optional<std::filesystem::path> frames; // a folder for every frame shown, as PNG
    std::optional<std::filesystem::path> stats;  // a CSV file
    std::optional<double> refresh_hz;            // pace as if the display ran at this rate
    bool help = false;
};

void print_usage()
{
    std::println("Usage: RayTracingViewer [scene] [options]\n\n"
                 "Shows RayTracing's scenes in a window, rendered on an NVIDIA graphics card with CUDA: the\n"
                 "image refines while the view stands still. Options as in RayTracingGPU.\n\n"
                 "Scenes:");
    for (const scene_entry& entry : scene_list)
    {
        std::println("  {:10} {}", entry.name, entry.description);
    }
    std::println(R"(  Default: {}.

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
      --seed N         which random numbers to draw, from 0 (the default) to {}
  -o, --output FILE    save the image when the window closes: a .png or a .pfm; several -o save
                       several files
      --exit-when-done close the window when the image has its samples (after saving -o)
      --frames DIR     save every frame the window shows with new samples, as DIR\f00000.png...
      --stats FILE     write a line of numbers per frame to a CSV file
      --refresh-hz N   pace the frames that follow a resize as if the display ran at N Hz (for
                       measurements; default: the display's own rate)
  -h, --help           show this help)",
                 default_viewer_scene, view_usage, max_strata, default_strata,
                 std::numeric_limits<std::uint32_t>::max());
}

// The options that take a value, besides the view options (is_view_option); the others are flags
constexpr auto value_options = std::to_array<std::string_view>(
    {"-w", "--width", "-s", "--samples", "--strata", "-p", "--sampling", "--seed", "-o", "--output", "--frames",
     "--stats", "--refresh-hz"});

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
        if (arg == "--fp64" || arg == "--all-books-kernel" || arg == "--exit-when-done")
        {
            (arg == "--fp64" ? opts.fp64 : arg == "--all-books-kernel" ? opts.all_books_kernel : opts.exit_when_done) =
                true;
            continue;
        }
        if (!arg.starts_with('-'))
        {
            if (!find_scene(arg))
            {
                return std::unexpected(std::format("unknown scene '{}'", arg));
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
        else if (arg == "--frames" || arg == "--stats")
        {
            (arg == "--frames" ? opts.frames : opts.stats) = std::filesystem::path(value);
        }
        else if (arg == "-p" || arg == "--sampling")
        {
            const auto it = std::ranges::find(sampling_list, value, &sampling_entry::name);
            if (it == sampling_list.end())
            {
                return std::unexpected(std::format("unknown sampling strategy '{}'", value));
            }
            opts.sampling = it->strategy;
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
    return opts;
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

// --- The window ------------------------------------------------------------------------------------------------

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

    std::optional<gpu_build> build_;
    std::optional<gpu::session> session_;
    int width_ = 0;                // the image: the window's client area in pixels
    int height_ = 0;
    std::uint64_t generation_ = 0; // counts the views: a new one with every camera or size change
    std::int64_t target_ = 0;      // passes to refine to
    int pending_width_ = 0;        // a new size from the window, applied at the next iterate
    int pending_height_ = 0;

    // The last frame shown, as the session's display() gave it, and its size: what a device reset puts back
    std::vector<std::uint8_t> shown_;
    int shown_width_ = 0;
    int shown_height_ = 0;
    bool present_needed_ = true;

    pacing pacing_;
    std::uint64_t start_ns_ = 0;
    std::uint64_t last_iterate_ns_ = 0;
    std::uint64_t moving_until_ns_ = 0;
    bool minimized_ = false;
    bool done_reported_ = false;
    std::uint64_t frames_saved_ = 0;
    std::uint64_t frame_ = 0;
    std::ofstream stats_;

    bool interactive() const { return !opts_.exit_when_done; }
    SDL_AppResult fail(const std::string& message);
    camera_settings settings_for(int width, int height) const;
    gpu::camera_desc camera_for(int width, int height) const;
    std::string gpu_command(std::int64_t passes) const;
    void set_size(int width, int height);
    bool make_texture(int width, int height);
    bool upload_shown();
    bool draw_and_present(bool new_samples);
    bool save_frame();
    void update_title();
    bool save_outputs();
    void write_stats(const gpu::render_report& report, bool new_samples, bool presented, bool moving, double wall_ms,
                     double dt_ms);
};

viewer::~viewer()
{
    session_.reset(); // the GPU's memory first, while CUDA is still up
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

// The scene's view (with the options' changes) for an image of width × height pixels, with the exact height (see
// aspect_for_height): the settings RayTracingGPU's options -w width --height height give the scene's camera
camera_settings viewer::settings_for(int width, int height) const
{
    camera_settings settings = build_->cpu.camera;
    settings.image_width = width;
    settings.aspect_ratio = aspect_for_height(width, height);
    return settings;
}

// The camera the GPU renders with: the CPU's camera computes it from the settings, as for RayTracingGPU
gpu::camera_desc viewer::camera_for(int width, int height) const
{
    return to_gpu(camera(settings_for(width, height)).frame(), build_->sampling);
}

// The RayTracingGPU command that renders the image of `passes` samples of the view on screen again, bit for bit
std::string viewer::gpu_command(std::int64_t passes) const
{
    std::string command = std::format("RayTracingGPU {} {} -s {} --strata {} --seed {}", opts_.scene,
                                      format_view(settings_for(width_, height_), height_), passes, opts_.strata,
                                      opts_.seed);
    if (opts_.sampling)
    {
        command += std::format(" -p {}",
                               std::ranges::find(sampling_list, *opts_.sampling, &sampling_entry::strategy)->name);
    }
    if (opts_.fp64)
    {
        command += " --fp64";
    }
    if (opts_.all_books_kernel)
    {
        command += " --all-books-kernel";
    }
    return command;
}

SDL_AppResult viewer::init()
{
    // The scene, built once at its own size, for its aspect ratio. --height alone: the scene's own aspect ratio at that
    // many rows (the scene is built without it; settings_for gives the camera any size)
    const bool height_alone = opts_.view.image_height && !opts_.image_width;
    view_options sized = opts_.view;
    if (height_alone)
    {
        sized.image_height.reset();
    }
    auto built = build_gpu_scene(opts_.scene, {.image_width = opts_.image_width,
                                               .samples_per_pixel = opts_.samples_per_pixel,
                                               .sampling = opts_.sampling,
                                               .view = sized,
                                               .strata = opts_.strata});
    if (!built)
    {
        return fail(built.error());
    }
    build_ = std::move(*built);
    width_ = build_->frame.image_width;
    height_ = build_->frame.image_height;
    if (height_alone)
    {
        height_ = *opts_.view.image_height;
        width_ = std::max(1, static_cast<int>(std::lround(height_ * build_->cpu.camera.aspect_ratio)));
    }
    const bool explicit_size = opts_.image_width || opts_.view.image_height;

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
                  "generation,width,height,strata\n";
    }

    // The window: the size of the options, or the default one
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        return fail(std::format("SDL couldn't start: {}", SDL_GetError()));
    }
    if (!explicit_size)
    {
        std::tie(width_, height_) = default_size(build_->cpu.camera.aspect_ratio, SDL_GetPrimaryDisplay());
    }
    const int asked_width = width_;
    const int asked_height = height_;
    window_ = SDL_CreateWindow(std::format("RayTracingViewer: {}", opts_.scene).c_str(), width_, height_,
                               SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
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
    // A size asked for that the screen can't hold: Windows makes the window smaller. Someone watching sees it and goes
    // on at that size; a run that ends by itself (--exit-when-done) would save another size than the one asked for,
    // which is an error.
    if (explicit_size && (width_ != asked_width || height_ != asked_height))
    {
        if (!interactive())
        {
            return fail(std::format("the window can't be {}×{} on this screen (it became {}×{})", asked_width,
                                    asked_height, width_, height_));
        }
        std::println("Note: the window can't be {}×{} on this screen; the image is {}×{}", asked_width, asked_height,
                     width_, height_);
    }
    pacing_ = pacing_for(opts_.refresh_hz.value_or(display_refresh_hz(window_)));
    if (!SDL_SetRenderVSync(renderer_, pacing_for(display_refresh_hz(window_)).vsync) &&
        !SDL_SetRenderVSync(renderer_, 1))
    {
        return fail(std::format("SDL couldn't turn on vsync: {}", SDL_GetError()));
    }

    // The session. Rounds of strata: the target is the samples asked for, or the scene's rounded up to whole rounds,
    // so that a finished image never stops inside a round (whose samples lean toward the top of the pixel, see
    // gpu::session).
    const std::int64_t round = std::int64_t{opts_.strata} * opts_.strata;
    target_ = opts_.samples_per_pixel ? *opts_.samples_per_pixel : (build_->passes + round - 1) / round * round;
    try
    {
        gpu::query_device(gpu::wait_mode::sleep);
        gpu::scene_desc desc = build_->desc;
        desc.camera = camera_for(width_, height_);
        session_.emplace(desc, gpu::session_options{.seed = opts_.seed,
                                                    .arithmetic = opts_.fp64 ? gpu::precision::float64
                                                                             : gpu::precision::float32,
                                                    .all_books_kernel = opts_.all_books_kernel,
                                                    .launch_seconds = still_seconds,
                                                    .wait = gpu::wait_mode::sleep});
    }
    catch (const std::exception& error)
    {
        return fail(error.what());
    }
    const auto entry = std::ranges::find(sampling_list, build_->sampling, &sampling_entry::strategy);
    std::println("Scene: {}, sampling: {}, {} samples per pixel in rounds of {}×{} strata, {}", opts_.scene,
                 entry->name, target_, opts_.strata, opts_.strata, opts_.fp64 ? "float64" : "float32");
    std::println("View: {}", format_view(settings_for(width_, height_), height_));
    start_ns_ = SDL_GetTicksNS();
    last_iterate_ns_ = start_ns_;
    moving_until_ns_ = start_ns_ + moving_ns; // the first frames come quickly
    if (!make_texture(width_, height_))
    {
        return SDL_APP_FAILURE;
    }
    update_title();
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
    return true;
}

// The window has a new size: the image follows it (a new view of the same camera)
void viewer::set_size(int width, int height)
{
    width_ = std::max(1, width);
    height_ = std::max(1, height);
    session_->set_camera(camera_for(width_, height_));
    ++generation_;
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
    if (new_samples && opts_.frames && !save_frame())
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
bool viewer::save_frame()
{
    const SDL_Rect rect{0, 0, shown_width_, shown_height_};
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

void viewer::update_title()
{
    const gpu::accumulation state = session_->state();
    const std::string title = std::format("RayTracingViewer: {}, {} / {} samples", opts_.scene, state.passes, target_);
    SDL_SetWindowTitle(window_, title.c_str());
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
        std::string errors; // every file that failed, in one message (and one message box)
        for (const std::filesystem::path& output : opts_.outputs)
        {
            if (const auto saved = img->save(output); !saved)
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

SDL_AppResult viewer::event(const SDL_Event& e)
{
    switch (e.type)
    {
    case SDL_EVENT_QUIT:
        return save_outputs() ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
    case SDL_EVENT_KEY_DOWN:
        if (e.key.key == SDLK_ESCAPE)
        {
            return save_outputs() ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
        }
        break;
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        pending_width_ = e.window.data1;
        pending_height_ = e.window.data2;
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
    try
    {
        if (pending_width_ > 0 && (pending_width_ != width_ || pending_height_ != height_))
        {
            set_size(pending_width_, pending_height_);
        }
        pending_width_ = 0;
        gpu::render_report report;
        bool new_samples = false;
        bool presented = false;
        const bool moving = now < moving_until_ns_;
        // Minimized, a window someone watches waits to be restored; a run that ends by itself goes on
        const bool minimized_pause = minimized_ && interactive();
        if (!minimized_pause && session_->state().passes < target_)
        {
            const double budget = moving ? moving_share * pacing_.frame_seconds : still_seconds;
            report = session_->render_for(budget, target_);
            const std::span<const std::uint8_t> bytes = session_->display();
            shown_.assign(bytes.begin(), bytes.end());
            shown_width_ = width_;
            shown_height_ = height_;
            new_samples = true;
            present_needed_ = true;
        }
        if (present_needed_ && !minimized_)
        {
            if (shown_width_ != texture_width_ || shown_height_ != texture_height_)
            {
                if (!shown_.empty() && !make_texture(shown_width_, shown_height_))
                {
                    return SDL_APP_FAILURE;
                }
            }
            if ((new_samples && !upload_shown()) || !draw_and_present(new_samples))
            {
                return SDL_APP_FAILURE;
            }
            presented = true;
            update_title();
        }
        const double wall_ms = static_cast<double>(SDL_GetTicksNS() - now) / 1e6;
        write_stats(report, new_samples, presented, moving, wall_ms, dt_ms);
        ++frame_;
        const bool done = session_->state().passes >= target_;
        if (done && !done_reported_)
        {
            done_reported_ = true;
            std::println("Done: {} samples per pixel in {:.2f} s", target_,
                         static_cast<double>(SDL_GetTicksNS() - start_ns_) / 1e9);
            if (opts_.exit_when_done)
            {
                return save_outputs() ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
            }
        }
        // Nothing to render and nothing new to show (or nowhere to show it): SDL waits for the next event before it
        // calls iterate again, and the program sleeps
        if ((minimized_pause || (done && !present_needed_)) && pending_width_ == 0)
        {
            SDL_SetHint(SDL_HINT_MAIN_CALLBACK_RATE, "waitevent");
        }
        else
        {
            // Work is left. While the user drags the window's edge, SDL calls iterate from inside Windows' loop for the
            // drag, and that loop may have started inside SDL's wait for an event: the drag's own calls take every
            // event out of the queue, so when the drag ends the wait would go on, the image half refined, until the
            // next event. An event of our own ends it.
            SDL_Event wake{};
            wake.type = SDL_EVENT_USER;
            SDL_PushEvent(&wake);
        }
    }
    catch (const std::exception& error)
    {
        return fail(error.what());
    }
    return SDL_APP_CONTINUE;
}

// --stats: a row per iterate. new_samples: the frame has samples the one before didn't (the frames --frames saves);
// presented: the window was drawn and presented; moving: the frame had a moving view's budget.
void viewer::write_stats(const gpu::render_report& report, bool new_samples, bool presented, bool moving,
                         double wall_ms, double dt_ms)
{
    if (!stats_.is_open())
    {
        return;
    }
    const gpu::accumulation state = session_->state();
    stats_ << std::format("{},{:.3f},{:.3f},{:.3f},{},{:.3f},{},{},{},{},{},{},{},{},{}\n", frame_, wall_ms, dt_ms,
                          report.gpu_seconds * 1e3, report.launches, report.longest_launch_seconds * 1e3, state.passes,
                          state.rows, new_samples ? 1 : 0, presented ? 1 : 0, moving ? 1 : 0, generation_, width_,
                          height_, opts_.strata);
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
