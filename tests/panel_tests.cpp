// The panel's checks for ViewerTests (tests/viewer_tests.cpp), in a file of its own: compiled only where the build has
// Dear ImGui (cmake/imgui.cmake), and run without a window, a renderer or a GPU. The panel (viewer_panel) is drawn
// from a made-up snapshot of the window by Dear ImGui's null backends (imgui_impl_null: no screen, its textures taken
// as made), and clicked by mouse events given to Dear ImGui as the window gives them: what it asks for must be the
// action of the control clicked, and nothing for a greyed control, the current choice, or frames nobody touched.
#include "imgui.h"
#include "imgui_impl_null.h"

import std;
import strategy_options;
import tone_options;
import viewer_actions;
import viewer_panel;

namespace
{
constexpr float window_width = 1280;
constexpr float window_height = 720;

// A window as the panel would see it: a book scene under the books' kernels, a few scenes, every strategy and tone map
panel_state made_up_state()
{
    panel_state s;
    s.scene = "checker";
    s.scene_name = "checker";
    s.width = 1280;
    s.height = 720;
    s.strata = 8;
    s.sampling = "material";
    s.passes = 64;
    s.target = 64;
    s.vfov = 20;
    s.lens = 0;
    s.focus = 10;
    s.tone_map = "books";
    s.exposure_refused = "exposure applies to a tone map: T picks one";
    s.focus_min = 0.01;
    s.focus_max = 130;
    for (const std::string_view name : {"materials", "weekend", "bouncing", "checker", "earth"})
    {
        s.scenes.push_back({.name = std::string(name), .group = "The books", .tip = "a book scene"});
    }
    s.scenes.push_back({.name = "studio", .group = "Beyond the books", .tip = "spheres in a photo studio"});
    s.strategies.push_back({.name = "scene", .group = "the scene's own (material)"});
    for (const strategy_entry& e : strategy_list)
    {
        s.strategies.push_back({.name = std::string(e.name), .group = std::string(e.name)});
    }
    for (const tone_map_entry& t : tone_map_list)
    {
        s.tone_maps.push_back({.name = std::string(t.name), .group = std::string(t.name)});
    }
    return s;
}

// Dear ImGui headless: a context of its own per check, no settings file, the null backends
class headless_panel
{
public:
    headless_panel()
    {
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        style_panel(1);
        load_panel_font();
        ImGui_ImplNullPlatform_Init();
        ImGui_ImplNullRender_Init();
    }
    ~headless_panel()
    {
        ImGui_ImplNullRender_Shutdown();
        ImGui_ImplNullPlatform_Shutdown();
        ImGui::DestroyContext();
    }
    headless_panel(const headless_panel&) = delete;
    headless_panel& operator=(const headless_panel&) = delete;

    // One frame of the panel: what it asked for
    std::vector<action> frame(const panel_state& state)
    {
        ImGui_ImplNullPlatform_NewFrame();
        ImGui_ImplNullRender_NewFrame();
        ImGui::GetIO().DisplaySize = ImVec2(window_width, window_height);
        ImGui::NewFrame();
        std::vector<action> asked = draw_panel(state, window_width, window_height, 1);
        ImGui::Render();
        ImGui_ImplNullRender_RenderDrawData(ImGui::GetDrawData());
        return asked;
    }

    // A left click at the middle of the control named so, where the last frame drew it, over the frames Dear ImGui
    // takes it in (the press in one, the release in the next): what the frames asked for, or nothing if there is no
    // such control
    std::optional<std::vector<action>> click(const panel_state& state, std::string_view name)
    {
        const std::optional<std::array<float, 4>> place = panel_item(name);
        if (!place)
        {
            return std::nullopt;
        }
        ImGuiIO& io = ImGui::GetIO();
        io.AddMousePosEvent(((*place)[0] + (*place)[2]) / 2, ((*place)[1] + (*place)[3]) / 2);
        io.AddMouseButtonEvent(0, true);
        std::vector<action> asked = frame(state);
        io.AddMouseButtonEvent(0, false);
        for (action& a : frame(state))
        {
            asked.push_back(std::move(a));
        }
        for (action& a : frame(state))
        {
            asked.push_back(std::move(a));
        }
        return asked;
    }
};

std::string describe(const std::optional<std::vector<action>>& asked)
{
    if (!asked)
    {
        return "no such control";
    }
    std::string text;
    for (const action& a : *asked)
    {
        text += std::format("{}[{}]", text.empty() ? "" : " ", format_action(a));
    }
    return text.empty() ? "nothing" : text;
}
}

// The checks, each reported by `check(name, pass, found, expected)`
void test_panel(const std::function<void(std::string_view, bool, std::string_view, std::string_view)>& check)
{
    // Clicks: one fresh panel each, drawn twice first (the second frame has every control where the first put it)
    const auto clicked = [&](std::string_view name, const panel_state& state, std::string_view control,
                             std::string_view expected, std::string_view first = {}) {
        headless_panel panel;
        panel.frame(state);
        panel.frame(state);
        if (!first.empty())
        {
            panel.click(state, first); // a combo box opened
        }
        const std::string found = describe(panel.click(state, control));
        check(std::format("panel.click.{}", name), found == expected, found, expected);
    };
    const panel_state state = made_up_state();
    clicked("scene", state, "scene earth", "[scene earth]");
    clicked("the-current-scene", state, "scene checker", "nothing");
    panel_state greyed = state;
    greyed.scenes[1].refused = "float32 only: F2 first";
    clicked("a-greyed-scene", greyed, "scene weekend", "nothing");
    clicked("strategy", state, "sampling mis", "[sampling mis]", "sampling");
    clicked("the-scene's-own-strategy", state, "sampling scene", "nothing", "sampling");
    clicked("tone-map", state, "tonemap neutral", "[tonemap neutral]", "tonemap");
    clicked("optix", state, "optix", "[optix on]");
    panel_state no_optix = state;
    no_optix.optix_refused = "OptiX computes in float32 only: F2 first";
    clicked("optix-greyed", no_optix, "optix", "nothing");
    clicked("float64", state, "precision double", "[precision double]");
    panel_state no_double = state;
    no_double.precision_refused = "agx works in float32 only: T to books first";
    clicked("float64-greyed", no_double, "precision double", "nothing");
    clicked("float32-already", state, "precision float", "nothing");
    clicked("pause", state, "pause", "[pause on]");
    clicked("samples-twice", state, "samples twice", "[samples 128]");
    clicked("samples-half", state, "samples half", "[samples 32]");
    clicked("reset-view", state, "reset-view", "[reset-view]");
    clicked("exposure-greyed-under-books", state, "exposure up", "nothing");
    panel_state agx = state;
    agx.tone_map = "agx";
    agx.has_exposure = true;
    agx.exposure = 1;
    agx.exposure_refused.clear();
    clicked("exposure-up", agx, "exposure up", "[exposure 1.5]");
    clicked("exposure-0", agx, "exposure 0", "[exposure 0]");
    clicked("denoiser", state, "denoise", "[denoise on]");
    clicked("save", state, "save", "[save]");
    clicked("copy", state, "copy", "[copy]");
    clicked("no-such-control", state, "fly", "no such control");

    // Frames nobody touched ask for nothing, a slider left alone included
    {
        headless_panel panel;
        std::string found;
        for (int i = 0; i < 5; ++i)
        {
            for (const action& a : panel.frame(agx))
            {
                found += format_action(a) + ";";
            }
        }
        check("panel.untouched-frames", found.empty(), found.empty() ? "nothing" : found, "nothing");
    }
    // The panel's place: the right edge, 320 units wide at 100%, 480 at 150%, never more than 60% of the window
    {
        const panel_rectangle a = panel_place(1280, 720, 1);
        const panel_rectangle b = panel_place(1280, 720, 1.5f);
        const panel_rectangle c = panel_place(320, 180, 1);
        const std::string found = std::format("{} {} | {} {} | {} {}", a.x, a.width, b.x, b.width, c.x, c.width);
        check("panel.place", found == "960 320 | 800 480 | 128 192", found, "960 320 | 800 480 | 128 192");
    }
}
