// The environment map on the GPU (quality_view::environment): its texels with their densities, its rows' cosines and
// its alias table, a part of the quality kernel's arrays (quality_scene in quality.cu) in a file of its own, so that
// the environment fills it without touching the other parts. Made from scene_desc::quality.environment by
// build_environment (environment_table.cu). Host code.
//
// Without a map, view() is the camera's sky (texels nullptr). A map with light is a light of the set, with its alias
// table (sampled()); a black map has none (bins nullptr): it is seen, never sampled. The table is uploaded whatever the
// strategy: under -p material no light sample reads it, and its 4 MiB (a 1k map) cost no time per ray.
#pragma once

#include <cmath>
#include <numbers>

#include "device_memory.h"
#include "environment_data.h"
#include "quality_view.cuh"
#include "renderer.h"

namespace gpu
{
class environment_on_device
{
public:
    explicit environment_on_device(const scene_desc& scene)
    {
        if (!scene.quality.environment)
        {
            return;
        }
        const environment_desc& env = *scene.quality.environment;
        // The tables live on the host only until they are copied: 12 MiB for a 1k map
        const environment_tables tables = build_environment(env);
        texels_ = device_array<env_texel>(tables.texels);
        row_cos_ = device_array<float>(tables.row_cos);
        bins_ = device_array<alias_bin>(tables.bins);
        // The yaw's cosine and sine in double, rounded once: a yaw of 0 gives exactly 1 and 0, so to_map changes no
        // direction
        const double yaw = env.yaw_degrees * std::numbers::pi / 180;
        view_ = {.texels = texels_.data(),
                 .bins = bins_.size() > 0 ? bins_.data() : nullptr,
                 .row_cos = row_cos_.data(),
                 .width = tables.width,
                 .height = tables.height,
                 .log2_bins = tables.log2_bins,
                 .cos_yaw = static_cast<float>(std::cos(yaw)),
                 .sin_yaw = static_cast<float>(std::sin(yaw)),
                 .scale = static_cast<float>(env.scale)};
    }

    environment_view view() const { return view_; }
    // A map with light in it, and so its alias table: a light of the set (environment_sampled on the device), which
    // quality.cu counts in light_pmf
    bool sampled() const { return view_.bins != nullptr; }

private:
    device_array<env_texel> texels_{0};
    device_array<float> row_cos_{0};
    device_array<alias_bin> bins_{0};
    environment_view view_{};
};
}
