// Module viewer_panel: the window's panel, drawn with Dear ImGui (github.com/ocornut/imgui): a column at the window's
// right edge that shows what the window is rendering. The panel is a function of a snapshot of the window
// (panel_state, in viewer_actions) that returns what was asked for as actions: it reads nothing else and changes
// nothing, so it can be drawn without a window, a scene or a GPU, and the window performs what it returns where it
// performs every action (viewer::apply). This module knows Dear ImGui and nothing of SDL; viewer_panel_host gives it
// the window's events and draws it.
//
// Dear ImGui is an immediate-mode library: there are no widget objects. Each frame the program calls a function per
// control, which draws it and says whether it was just used (`if (ImGui::Button("Save")) ...`), so the panel below is
// straight-line code that runs once per frame shown.
module;
#include "imgui.h"
export module viewer_panel;

import std;
import viewer_actions;

// The panel's look, at a display scale (1 at 100%, 1.5 at 150%): Dear ImGui's dark colors, square corners against the
// window's edge, every size and the font scaled. Called when the panel starts and when the display's scale changes.
export void style_panel(float scale)
{
    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle();
    ImGui::StyleColorsDark();
    style.WindowRounding = 0;
    style.WindowBorderSize = 0;
    style.FrameRounding = 3;
    style.GrabRounding = 3;
    style.Colors[ImGuiCol_WindowBg].w = 0.92f; // the image shows through a little
    style.ScaleAllSizes(scale);
    style.FontScaleDpi = scale;
}

// The panel's font: Dear ImGui's own scalable one, embedded in the library, so the program carries no font file
export void load_panel_font()
{
    ImGui::GetIO().Fonts->AddFontDefaultVector();
}

// The panel's rectangle in a window of that size, in the window's coordinates: the right edge, the full height, 320
// units wide at the display's scale but never more than 60% of the window
export struct panel_rectangle
{
    float x = 0;
    float y = 0;
    float width = 0;
    float height = 0;
    bool holds(float px, float py) const { return px >= x && px < x + width && py >= y && py < y + height; }
};

export panel_rectangle panel_place(float window_width, float window_height, float scale)
{
    const float width = std::min(std::round(320 * scale), std::floor(window_width * 0.6f));
    return {window_width - width, 0, width, window_height};
}

namespace
{
// A line of text from std::format (Dear ImGui's own Text takes printf's formats)
void text(const std::string& line)
{
    ImGui::TextUnformatted(line.c_str(), line.c_str() + line.size());
}

// The image's numbers: what the bar at the top of the window shows when the panel is hidden
void draw_status(const panel_state& state)
{
    text(state.scene);
    text(std::format("{}x{}  {}  strata {}x{}", state.width, state.height, state.fp64 ? "float64" : "float32",
                     state.strata, state.strata));
    text(state.renderer.empty() ? std::format("sampling: {}", state.sampling)
                                : std::format("sampling: {}, {}", state.sampling, state.renderer));
    // The samples so far against those to refine to, as a bar with the numbers on it
    const float done = state.target > 0 ? static_cast<float>(std::min(state.passes, state.target)) /
                                              static_cast<float>(state.target)
                                        : 0.0f;
    ImGui::ProgressBar(done, ImVec2(-1, 0), std::format("{} / {} samples", state.passes, state.target).c_str());
    if (state.paused)
    {
        text("paused");
    }
    else if (state.passes >= state.target)
    {
        text("done");
    }
    else if (state.samples_per_second > 0)
    {
        text(std::format("{:.1f} M samples/s   frame {:.1f} ms", state.samples_per_second / 1e6, state.frame_ms));
    }
    else
    {
        text("refining");
    }
    if (!state.denoised.empty())
    {
        text(state.denoised);
    }
    text(std::format("vfov {:.4g}  aperture {:.4g}  focus {:.4g}", state.vfov, state.lens, state.focus));
    text(state.has_exposure ? std::format("display: {} {:+.4g} EV", state.tone_map, state.exposure)
                            : std::format("display: {}", state.tone_map));
    if (!state.note.empty())
    {
        ImGui::Separator();
        ImGui::PushTextWrapPos(0);
        text(state.note);
        ImGui::PopTextWrapPos();
    }
}
}

// One frame of the panel, between ImGui::NewFrame and ImGui::Render: drawn from `state` in a window of that size, and
// what was asked for in it, in order. The panel can't be moved, resized or collapsed, and keeps no settings between
// runs: it is where it was drawn.
export std::vector<action> draw_panel(const panel_state& state, float window_width, float window_height, float scale)
{
    std::vector<action> asked;
    const panel_rectangle place = panel_place(window_width, window_height, scale);
    ImGui::SetNextWindowPos(ImVec2(place.x, place.y));
    ImGui::SetNextWindowSize(ImVec2(place.width, place.height));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                                   ImGuiWindowFlags_NoNav;
    if (ImGui::Begin("RayTracingViewer", nullptr, flags))
    {
        draw_status(state);
    }
    ImGui::End();
    return asked;
}
