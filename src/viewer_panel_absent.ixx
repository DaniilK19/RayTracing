// Module viewer_panel_host for a build without Dear ImGui (cmake/imgui.cmake: its download failed, or RAYTRACING_PANEL
// is off): the same class, which has no panel to show. The window asks available() and says so when F3 asks for it.
module;
#include <SDL3/SDL.h>
export module viewer_panel_host;

import std;
import viewer_actions;

export class panel_host
{
public:
    static constexpr bool available() { return false; }
    std::expected<void, std::string> start(SDL_Window*, SDL_Renderer*)
    {
        return std::unexpected("this build has no panel (Dear ImGui wasn't downloaded: see docs/building.md)");
    }
    void stop() {}
    bool started() const { return false; }
    void event(const SDL_Event&) {}
    bool over_panel(float, float) const { return false; }
    bool typing() const { return false; }
    std::optional<std::array<float, 4>> item(std::string_view) const { return std::nullopt; }
    std::vector<action> frame(const panel_state&) { return {}; }
    void device_reset() {}
};
