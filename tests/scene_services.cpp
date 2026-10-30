// GpuTests' host services that build the GPU programs' scenes (declared in host_services.h with the others), for the
// OptiX suites (tests/optix_tests.cu): a registered scene as gpu_scene builds it for RayTracingGPU, and its geometry
// alone, which the tracer check uploads. A file of its own beside gpu_tests.cpp (the harness and the older services),
// added to GpuTests by cmake/optix.cmake in the builds with and without OptiX. cl.exe, C++23: the plain headers come
// before `import std;` (see renderer.h).
#include "host_services.h"
#include "renderer.h"

import std;
import gpu_scene;
import strategy_options;
import view_options;

namespace gpu_test::host
{
gpu::scene_desc built_scene(std::string_view name, const scene_request& request)
{
    scene_overrides overrides;
    if (request.width > 0)
    {
        overrides.image_width = request.width;
    }
    if (request.height > 0)
    {
        overrides.view.image_height = request.height;
    }
    if (request.samples > 0)
    {
        overrides.samples_per_pixel = request.samples;
    }
    switch (request.sampling)
    {
    case scene_strategy::own:
        break;
    case scene_strategy::material:
        overrides.sampling = strategy::material;
        break;
    case scene_strategy::mis:
        overrides.sampling = strategy::mis;
        break;
    }
    auto build = build_gpu_scene(name, overrides);
    if (!build)
    {
        throw std::runtime_error(build.error());
    }
    return std::move(build->desc);
}

namespace
{
gpu::vec3d mean_of(const gpu::vec3d& a, const gpu::vec3d& b)
{
    return {.x = (a.x + b.x) / 2, .y = (a.y + b.y) / 2, .z = (a.z + b.z) / 2};
}

// An image's mean color: its texels' linear values, averaged in double
gpu::vec3d mean_color(const gpu::image_desc& image)
{
    std::array<double, 3> sum{};
    const std::size_t texels = image.rgb.size() / 3;
    for (std::size_t k = 0; k < texels; ++k)
    {
        for (std::size_t c = 0; c < 3; ++c)
        {
            sum[c] += image.rgb[3 * k + c];
        }
    }
    const double n = texels > 0 ? static_cast<double>(texels) : 1.0;
    return {.x = sum[0] / n, .y = sum[1] / n, .z = sum[2] / n};
}
}

gpu::scene_desc geometry_only(const gpu::scene_desc& scene)
{
    // Each texture's one color, in the list's order: a checker's two textures come before it (renderer.h), so theirs
    // are known when it is reached
    std::vector<gpu::vec3d> colors;
    colors.reserve(scene.textures.size());
    const auto earlier = [&colors](int index) {
        if (index < 0 || index >= std::ssize(colors))
        {
            throw std::invalid_argument(std::format("a checker names texture {}, which doesn't come before it", index));
        }
        return colors[static_cast<std::size_t>(index)];
    };
    for (const gpu::texture_desc& texture : scene.textures)
    {
        switch (texture.kind)
        {
        case gpu::texture_kind::solid:
            colors.push_back(texture.rgb);
            break;
        case gpu::texture_kind::checker:
            colors.push_back(mean_of(earlier(texture.even), earlier(texture.odd)));
            break;
        case gpu::texture_kind::image:
            colors.push_back(mean_color(scene.images.at(static_cast<std::size_t>(texture.image))));
            break;
        case gpu::texture_kind::noise:
            colors.push_back({.x = 0.5, .y = 0.5, .z = 0.5});
            break;
        }
    }

    gpu::scene_desc geometry = scene;
    for (std::size_t k = 0; k < colors.size(); ++k)
    {
        geometry.textures[k] = {.kind = gpu::texture_kind::solid, .rgb = colors[k]};
    }
    geometry.images.clear();
    geometry.perlins.clear();
    geometry.lights.clear();
    if (geometry.camera.sampling == gpu::sampling_strategy::lights ||
        geometry.camera.sampling == gpu::sampling_strategy::mixture)
    {
        geometry.camera.sampling = gpu::sampling_strategy::material; // what those strategies aim at is gone
    }
    return geometry;
}
}
