// Module viewer_panel: the window's panel, drawn with Dear ImGui (github.com/ocornut/imgui): a column at the window's
// right edge that shows what the window is rendering and holds its controls. The panel is a function of a snapshot of
// the window (panel_state, in viewer_actions) that returns what was asked for as actions: it reads nothing else and
// changes nothing, so it can be drawn without a window, a scene or a GPU (ViewerTests does, clicking through it), and
// the window performs what it returns where it performs every action (viewer::apply), with the checks and the notes of
// the keys. This module knows Dear ImGui and nothing of SDL; viewer_panel_host gives it the window's events and draws
// it.
//
// Dear ImGui is an immediate-mode library: there are no widget objects. Each frame the program calls a function per
// control, which draws it and says whether it was just used (`if (ImGui::Button("Save")) ...`), so the panel below is
// straight-line code that runs once per frame shown.
module;
#include "imgui.h"
export module viewer_panel;

import std;
import strategy_options;
import tone_options;
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
// Where each control was drawn in the last frame, by the name an input file's `click` line gives it (panel_item)
std::map<std::string, std::array<float, 4>, std::less<>> item_places;

// The control just drawn, under that name
void remember(std::string name)
{
    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    item_places[std::move(name)] = {min.x, min.y, max.x, max.y};
}

// A line of text from std::format (Dear ImGui's own Text takes printf's formats)
void text(std::string_view line)
{
    ImGui::TextUnformatted(line.data(), line.data() + line.size());
}

// A control's label on the left of its line; the control after it takes the rest of the line
void label(std::string_view name)
{
    text(name);
    ImGui::SameLine(ImGui::GetFontSize() * 6.5f);
    ImGui::SetNextItemWidth(-1);
}

// The tooltip of the control just drawn, at once (the window may be asleep, so no delay that needs frames to pass),
// also over a greyed control, whose tooltip says why it is greyed
void tip(std::string_view what)
{
    if (!what.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled | ImGuiHoveredFlags_DelayNone))
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24);
        text(what);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

// A choice of a list as a line that can be clicked, greyed when the window would refuse it: whether it was clicked
bool choice_line(const panel_choice& c, const std::string& shown, bool current, std::string_view group)
{
    ImGui::BeginDisabled(!c.refused.empty());
    ImGui::PushID(c.name.c_str());
    const bool clicked = ImGui::Selectable(shown.c_str(), current);
    ImGui::PopID();
    ImGui::EndDisabled();
    remember(std::format("{} {}", group, c.name));
    tip(c.refused.empty() ? c.tip : c.refused);
    return clicked && c.refused.empty() && !current;
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
    const std::string how = state.paused                    ? "paused"
                            : state.passes >= state.target  ? "done"
                            : state.samples_per_second > 0
                                ? std::format("{:.1f} M/s", state.samples_per_second / 1e6)
                                : "refining";
    ImGui::ProgressBar(done, ImVec2(-1, 0),
                       std::format("{} / {} samples, {}", state.passes, state.target, how).c_str());
    if (!state.denoised.empty())
    {
        text(state.denoised);
    }
}

// Scene: the scenes in their groups, the test scenes folded away, in a box of eight lines that scrolls, so the sections
// below stay in sight
void draw_scenes(const panel_state& state, std::vector<action>& asked)
{
    ImGui::BeginChild("scenes", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 8), ImGuiChildFlags_Borders);
    std::string group;
    bool open = false;
    for (const panel_choice& c : state.scenes)
    {
        if (c.group != group)
        {
            if (open)
            {
                ImGui::TreePop();
            }
            group = c.group;
            ImGui::SetNextItemOpen(group != "Test scenes", ImGuiCond_Once);
            open = ImGui::TreeNode(group.c_str());
        }
        // A model by its file's name (its path in the tooltip and in the name a click line gives)
        const std::string shown =
            group == "Models" ? std::filesystem::path(c.name).filename().string() : c.name;
        if (open && choice_line(c, shown, c.name == state.scene_name, "scene"))
        {
            asked.push_back(show_scene{c.name});
        }
    }
    if (open)
    {
        ImGui::TreePop();
    }
    ImGui::EndChild();
}

// A list as a combo box: its current choice shown, the others in the box when it opens. `choose` gets the name of the
// choice clicked.
template <class Choose>
void draw_combo(std::string_view label, const std::string& preview, const std::vector<panel_choice>& choices,
                const std::string& current, std::string_view group, const Choose& choose)
{
    const bool opened = ImGui::BeginCombo(std::format("##{}", label).c_str(), preview.c_str());
    remember(std::string(group));
    tip(std::format("{}: {}", label, preview));
    if (opened)
    {
        for (const panel_choice& c : choices)
        {
            if (choice_line(c, c.group, c.name == current, group))
            {
                choose(c.name);
            }
        }
        ImGui::EndCombo();
    }
}

// A checkbox that asks for its other state, greyed with a reason
template <class Ask>
void draw_switch(const char* label, bool on, const std::string& refused, std::string_view key, std::string_view what,
                 const Ask& ask)
{
    ImGui::BeginDisabled(!refused.empty());
    bool value = on;
    const bool changed = ImGui::Checkbox(label, &value);
    ImGui::EndDisabled();
    remember(std::string(key));
    tip(refused.empty() ? what : refused);
    if (changed && refused.empty())
    {
        ask(value);
    }
}

// A button: whether it was pressed. Its tooltip names the key that does the same.
bool button(const char* label, std::string_view key, std::string_view what)
{
    const bool pressed = ImGui::Button(label);
    remember(std::string(key));
    tip(what);
    return pressed;
}

// A slider of a value in double, which asks for the value only when the hand moved it: a value never touched is never
// written back, rounded or not
bool slider(std::string_view key, double& value, double low, double high, const char* format, std::string_view what,
            bool logarithmic = false)
{
    const bool changed =
        ImGui::SliderScalar(std::format("##{}", key).c_str(), ImGuiDataType_Double, &value, &low, &high, format,
                            logarithmic ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
    remember(std::string(key));
    tip(what);
    return changed;
}

void draw_render(const panel_state& state, std::vector<action>& asked)
{
    const std::string own = state.strategies.empty() ? std::string() : state.strategies.front().group;
    label("sampling");
    draw_combo("sampling", state.sampling_named.empty() ? own : state.sampling_named, state.strategies,
               state.sampling_named.empty() ? std::string("scene") : state.sampling_named, "sampling",
               [&](const std::string& name) {
                   if (name == "scene")
                   {
                       asked.push_back(use_sampling{});
                   }
                   else if (const strategy_entry* entry = find_strategy(name))
                   {
                       asked.push_back(use_sampling{entry->value});
                   }
               });
    // float32 or float64: two radio buttons, float64 greyed with F2's reason
    if (ImGui::RadioButton("float32", !state.fp64) && state.fp64)
    {
        asked.push_back(use_precision{false});
    }
    remember("precision float");
    tip("32-bit floats, the GPU's own (F2)");
    ImGui::SameLine();
    ImGui::BeginDisabled(!state.precision_refused.empty());
    if (ImGui::RadioButton("float64", state.fp64) && !state.fp64)
    {
        asked.push_back(use_precision{true});
    }
    ImGui::EndDisabled();
    remember("precision double");
    tip(state.precision_refused.empty() ? "64-bit doubles: a check of the floats, many times slower (F2)"
                                        : state.precision_refused);
    draw_switch("OptiX", state.optix, state.optix_refused, "optix",
                "trace with OptiX, on the ray tracing cores where the GPU has them (O)",
                [&](bool on) { asked.push_back(use_optix{on}); });
    ImGui::SameLine();
    draw_switch("Pause", state.paused, "", "pause", "stop refining; a move still shows its new view (Space)",
                [&](bool on) { asked.push_back(use_pause{on}); });
    // The samples to refine to: half, twice, or a number typed and Enter
    text("samples");
    ImGui::SameLine(ImGui::GetFontSize() * 6.5f);
    if (button("/2", "samples half", "half the samples, in whole rounds of strata (Shift+Enter)"))
    {
        asked.push_back(use_samples{std::max<std::int64_t>(state.target / 2, 1)});
    }
    ImGui::SameLine();
    if (button("x2", "samples twice", "twice the samples (Enter)"))
    {
        asked.push_back(use_samples{std::min(state.target * 2, max_action_samples)});
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    std::int64_t samples = state.target;
    const bool typed = ImGui::InputScalar("##samples", ImGuiDataType_S64, &samples, nullptr, nullptr, nullptr,
                                          ImGuiInputTextFlags_EnterReturnsTrue);
    remember("samples");
    tip("a number of samples per pixel, then Enter: rounded up to whole rounds of strata");
    if (typed && samples >= 1 && samples <= max_action_samples && samples != state.target)
    {
        asked.push_back(use_samples{samples});
    }
}

void draw_camera(const panel_state& state, std::vector<action>& asked)
{
    double vfov = state.vfov;
    label("vfov");
    if (slider("vfov", vfov, 1, 160, "%.4g degrees", "Ctrl+wheel, - and ="))
    {
        asked.push_back(use_vfov{vfov});
    }
    double lens = state.lens;
    label("lens");
    if (slider("defocus-angle", lens, 0, 30, "%.4g degrees", "the defocus angle: 0 is all sharp ([ and ])"))
    {
        asked.push_back(use_lens{lens});
    }
    double focus = state.focus;
    label("focus");
    if (state.focus_max > state.focus_min &&
        slider("focus-dist", focus, state.focus_min, state.focus_max, "%.4g", "Shift+wheel, or a double-click on the "
                                                                              "image",
               true))
    {
        asked.push_back(use_focus{focus});
    }
    if (button("Reset view", "reset-view", "back to the scene's starting view (R)"))
    {
        asked.push_back(reset_view{});
    }
}

void draw_display(const panel_state& state, std::vector<action>& asked)
{
    label("tone map");
    draw_combo("tone map", state.tone_map, state.tone_maps, state.tone_map, "tonemap", [&](const std::string& name) {
        if (const tone_map_entry* entry = find_tone_map(name))
        {
            asked.push_back(use_tone_map{entry->value});
        }
    });
    ImGui::BeginDisabled(!state.exposure_refused.empty());
    label("exposure");
    double exposure = state.exposure;
    if (slider("exposure", exposure, -max_exposure, max_exposure, "%+.2f EV",
               state.exposure_refused.empty() ? "the light times 2^EV before the tone map (, and .)"
                                              : state.exposure_refused))
    {
        asked.push_back(use_exposure{exposure});
    }
    ImGui::Dummy(ImVec2(0, 0));
    ImGui::SameLine(ImGui::GetFontSize() * 6.5f);
    if (button("-0.5", "exposure down", "half a stop darker (,)"))
    {
        asked.push_back(use_exposure{state.exposure - 0.5});
    }
    ImGui::SameLine();
    if (button("0 EV", "exposure 0", "back to 0 EV (0)"))
    {
        asked.push_back(use_exposure{0});
    }
    ImGui::SameLine();
    if (button("+0.5", "exposure up", "half a stop brighter (.)"))
    {
        asked.push_back(use_exposure{state.exposure + 0.5});
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    draw_switch("Denoiser", state.denoise, state.denoise_refused, "denoise",
                "OptiX's AI denoiser on a still view, for the eye: it is biased, and the saved PFM never uses it (N)",
                [&](bool on) { asked.push_back(use_denoise{on}); });
}

void draw_output(std::vector<action>& asked)
{
    if (button("Save image", "save", "the image as <scene>-view-NNN.png and .pfm, and its RayTracingGPU command (P)"))
    {
        asked.push_back(save_image{});
    }
    ImGui::SameLine();
    if (button("Copy command", "copy", "the RayTracingGPU command that renders the image again (C)"))
    {
        asked.push_back(copy_command{});
    }
}
}

// Where the control named so (as an input file's `click` line names it: "scene cornell", "sampling", "sampling mis",
// "optix", "save"...) was drawn in the last frame: left, top, right and bottom, in the window's coordinates
export std::optional<std::array<float, 4>> panel_item(std::string_view name)
{
    if (const auto found = item_places.find(name); found != item_places.end())
    {
        return found->second;
    }
    return std::nullopt;
}

// One frame of the panel, between ImGui::NewFrame and ImGui::Render: drawn from `state` in a window of that size, and
// what was asked for in it, in order. The panel can't be moved, resized or collapsed, and keeps no settings between
// runs: it is where it was drawn.
export std::vector<action> draw_panel(const panel_state& state, float window_width, float window_height, float scale)
{
    std::vector<action> asked;
    item_places.clear();
    const panel_rectangle place = panel_place(window_width, window_height, scale);
    ImGui::SetNextWindowPos(ImVec2(place.x, place.y));
    ImGui::SetNextWindowSize(ImVec2(place.width, place.height));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                                   ImGuiWindowFlags_NoNav;
    if (ImGui::Begin("RayTracingViewer", nullptr, flags))
    {
        draw_status(state);
        const auto section = [](const char* name, bool open) {
            ImGui::SetNextItemOpen(open, ImGuiCond_Once);
            const bool shown = ImGui::CollapsingHeader(name);
            remember(std::format("section {}", name));
            return shown;
        };
        if (section("Scene", true))
        {
            draw_scenes(state, asked);
            if (button("Open...", "open", "a glTF 2.0 model's file (.gltf, .glb), shown in the photo studio; a file "
                                          "dropped on the window opens too"))
            {
                asked.push_back(open_file{});
            }
        }
        if (section("Render", true))
        {
            draw_render(state, asked);
        }
        if (section("Camera", true))
        {
            draw_camera(state, asked);
        }
        if (section("Display", true))
        {
            draw_display(state, asked);
        }
        if (section("Output", true))
        {
            draw_output(asked);
        }
        if (section("Keys", false))
        {
            for (const std::string_view line : state.keys)
            {
                text(line);
            }
        }
        text("F3 hides this panel");
    }
    ImGui::End();
    return asked;
}
