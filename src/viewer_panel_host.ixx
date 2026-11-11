// Module viewer_panel_host: the panel (viewer_panel) in the window. It gives Dear ImGui the window's events, its size
// and its display's scale, and draws the panel through SDL's renderer, after the image and the numbers. The drawing
// is Dear ImGui's own backend for SDL's renderer (imgui_impl_sdlrenderer3); the events are given here, by hand, in a
// few lines, rather than by Dear ImGui's backend for SDL's events, which also reads the desktop's mouse by itself and
// changes SDL's hints for capturing the mouse: here the panel sees exactly the events the window sees, so a run driven
// by an input file (--input) takes no real mouse, and the window's drags behave as they always did.
// A build without Dear ImGui compiles viewer_panel_absent.ixx under this module's name instead.
module;
#include <SDL3/SDL.h>
#include "imgui.h"
#include "imgui_impl_sdlrenderer3.h"
export module viewer_panel_host;

import std;
import viewer_actions;
import viewer_panel;

namespace
{
// SDL's keys as Dear ImGui names them: the keys a text field and the panel's own navigation use. The panel's keys go
// to Dear ImGui only while it is typing into a field (see panel_host::event); the others are the window's.
ImGuiKey imgui_key(SDL_Keycode key)
{
    switch (key)
    {
    case SDLK_TAB: return ImGuiKey_Tab;
    case SDLK_LEFT: return ImGuiKey_LeftArrow;
    case SDLK_RIGHT: return ImGuiKey_RightArrow;
    case SDLK_UP: return ImGuiKey_UpArrow;
    case SDLK_DOWN: return ImGuiKey_DownArrow;
    case SDLK_HOME: return ImGuiKey_Home;
    case SDLK_END: return ImGuiKey_End;
    case SDLK_DELETE: return ImGuiKey_Delete;
    case SDLK_BACKSPACE: return ImGuiKey_Backspace;
    case SDLK_RETURN: return ImGuiKey_Enter;
    case SDLK_KP_ENTER: return ImGuiKey_KeypadEnter;
    case SDLK_ESCAPE: return ImGuiKey_Escape;
    case SDLK_A: return ImGuiKey_A;
    case SDLK_C: return ImGuiKey_C;
    case SDLK_V: return ImGuiKey_V;
    case SDLK_X: return ImGuiKey_X;
    case SDLK_Y: return ImGuiKey_Y;
    case SDLK_Z: return ImGuiKey_Z;
    default: return ImGuiKey_None;
    }
}

const char* get_clipboard(ImGuiContext*)
{
    static std::string text; // Dear ImGui keeps the pointer until the next call
    char* got = SDL_GetClipboardText();
    text = got ? got : "";
    SDL_free(got);
    return text.c_str();
}

void set_clipboard(ImGuiContext*, const char* text)
{
    SDL_SetClipboardText(text);
}
}

// The panel in a window: started at its first frame, stopped with the window (before its renderer is destroyed)
export class panel_host
{
public:
    panel_host() = default;
    ~panel_host() { stop(); }
    panel_host(const panel_host&) = delete;
    panel_host& operator=(const panel_host&) = delete;

    // Whether this build has the panel (Dear ImGui)
    static constexpr bool available() { return true; }

    std::expected<void, std::string> start(SDL_Window* window, SDL_Renderer* renderer)
    {
        if (context_)
        {
            return {};
        }
        context_ = ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr; // no settings file: every run starts the same
        io.LogFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange; // the cursor is the window's (a right drag hides it)
        io.BackendPlatformName = "RayTracingViewer";
        ImGuiPlatformIO& platform = ImGui::GetPlatformIO();
        platform.Platform_GetClipboardTextFn = get_clipboard;
        platform.Platform_SetClipboardTextFn = set_clipboard;
        window_ = window;
        renderer_ = renderer;
        scale_ = display_scale();
        style_panel(scale_);
        load_panel_font();
        if (!ImGui_ImplSDLRenderer3_Init(renderer))
        {
            stop();
            return std::unexpected("Dear ImGui couldn't draw with SDL's renderer");
        }
        last_ns_ = SDL_GetTicksNS();
        return {};
    }

    void stop()
    {
        if (!context_)
        {
            return;
        }
        ImGui_ImplSDLRenderer3_Shutdown();
        ImGui::DestroyContext(context_);
        context_ = nullptr;
    }

    bool started() const { return context_ != nullptr; }

    // An event of the window, given to Dear ImGui: the mouse, the keys while it is typing, text, focus. The window
    // performs the event too, unless the panel takes it (over_panel, typing).
    void event(const SDL_Event& e)
    {
        if (!context_)
        {
            return;
        }
        ImGuiIO& io = ImGui::GetIO();
        switch (e.type)
        {
        case SDL_EVENT_MOUSE_MOTION:
            io.AddMousePosEvent(e.motion.x, e.motion.y);
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP:
        {
            // SDL numbers left, middle, right as 1, 2, 3; Dear ImGui as 0, 2, 1
            const int button = e.button.button == SDL_BUTTON_LEFT    ? 0
                               : e.button.button == SDL_BUTTON_RIGHT ? 1
                               : e.button.button == SDL_BUTTON_MIDDLE ? 2
                                                                      : -1;
            if (button >= 0)
            {
                io.AddMousePosEvent(e.button.x, e.button.y);
                io.AddMouseButtonEvent(button, e.button.down);
            }
            break;
        }
        case SDL_EVENT_MOUSE_WHEEL:
            io.AddMouseWheelEvent(-e.wheel.x, e.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -e.wheel.y : e.wheel.y);
            break;
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP:
        {
            const bool down = e.type == SDL_EVENT_KEY_DOWN;
            io.AddKeyEvent(ImGuiMod_Ctrl, (e.key.mod & SDL_KMOD_CTRL) != 0);
            io.AddKeyEvent(ImGuiMod_Shift, (e.key.mod & SDL_KMOD_SHIFT) != 0);
            io.AddKeyEvent(ImGuiMod_Alt, (e.key.mod & SDL_KMOD_ALT) != 0);
            // Only while typing, or a key let go: a key the window performs never reaches a field
            if (const ImGuiKey key = imgui_key(e.key.key); key != ImGuiKey_None && (typing() || !down))
            {
                io.AddKeyEvent(key, down);
            }
            break;
        }
        case SDL_EVENT_TEXT_INPUT:
            io.AddInputCharactersUTF8(e.text.text);
            break;
        case SDL_EVENT_WINDOW_MOUSE_LEAVE:
            io.AddMousePosEvent(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
            break;
        case SDL_EVENT_WINDOW_FOCUS_GAINED:
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            io.AddFocusEvent(e.type == SDL_EVENT_WINDOW_FOCUS_GAINED);
            break;
        case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
            scale_ = display_scale();
            style_panel(scale_);
            break;
        default:
            break;
        }
    }

    // Whether a point of the window (its coordinates, as SDL's mouse events give them) is the panel's: a click or a
    // wheel step there is the panel's, not the camera's
    bool over_panel(float x, float y) const
    {
        if (!context_)
        {
            return false;
        }
        int w = 0;
        int h = 0;
        SDL_GetWindowSize(window_, &w, &h);
        return panel_place(static_cast<float>(w), static_cast<float>(h), scale_).holds(x, y) ||
               ImGui::GetIO().WantCaptureMouse;
    }

    // Where the panel drew its control of that name in its last frame (an input file's click line), or nothing
    std::optional<std::array<float, 4>> item(std::string_view name) const
    {
        return context_ ? panel_item(name) : std::nullopt;
    }

    // Whether a field of the panel has the keyboard: the window's keys then type into it
    bool typing() const { return context_ && ImGui::GetIO().WantTextInput; }

    // A frame of the panel over what the renderer holds: drawn from `state`, and what was asked for in it
    std::vector<action> frame(const panel_state& state)
    {
        if (!context_)
        {
            return {};
        }
        ImGuiIO& io = ImGui::GetIO();
        int w = 0;
        int h = 0;
        SDL_GetWindowSize(window_, &w, &h);
        int pw = 0;
        int ph = 0;
        SDL_GetWindowSizeInPixels(window_, &pw, &ph);
        io.DisplaySize = ImVec2(static_cast<float>(w), static_cast<float>(h));
        io.DisplayFramebufferScale = ImVec2(w > 0 ? static_cast<float>(pw) / static_cast<float>(w) : 1.0f,
                                            h > 0 ? static_cast<float>(ph) / static_cast<float>(h) : 1.0f);
        const std::uint64_t now = SDL_GetTicksNS();
        io.DeltaTime = std::max(static_cast<float>(now - last_ns_) / 1e9f, 1e-4f);
        last_ns_ = now;
        // A field asks for text: SDL's text events on (they also bring up an input method's window), and off again
        if (io.WantTextInput != text_input_)
        {
            text_input_ = io.WantTextInput;
            text_input_ ? SDL_StartTextInput(window_) : SDL_StopTextInput(window_);
        }

        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui::NewFrame();
        std::vector<action> asked =
            draw_panel(state, static_cast<float>(w), static_cast<float>(h), scale_);
        ImGui::Render();
        SDL_SetRenderScale(renderer_, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer_);
        SDL_SetRenderScale(renderer_, 1, 1);
        return asked;
    }

    // The renderer's textures are gone (a device reset): Dear ImGui's made again at the next frame
    void device_reset()
    {
        if (context_)
        {
            ImGui_ImplSDLRenderer3_DestroyDeviceObjects();
            ImGui_ImplSDLRenderer3_CreateDeviceObjects();
        }
    }

private:
    float display_scale() const
    {
        const float scale = SDL_GetWindowDisplayScale(window_);
        return scale > 0 ? scale : 1.0f;
    }

    ImGuiContext* context_ = nullptr;
    SDL_Window* window_ = nullptr;
    SDL_Renderer* renderer_ = nullptr;
    float scale_ = 1;
    std::uint64_t last_ns_ = 0;
    bool text_input_ = false;
};
