// Suite services: the test programs' own tools, which the later suites build on. A scene_desc on the GPU for tests
// (test_scene.h) holds the BVH's ids and the media's base as the renderer's; the host services (host_services.h) find
// the project's files and give the PNG's bytes back exactly.
#include <cstdint>
#include <vector>

#include "gpu_test.h"
#include "host_services.h"
#include "test_scene.h"

namespace
{
GPU_TEST(services, test_scene)
{
    // Two spheres, a quad and a volume of smoke bounded by a sphere
    gpu::scene_desc scene;
    scene.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = {0.5, 0.25, 0.125}});
    scene.materials.push_back({.kind = gpu::material_kind::lambertian, .texture = 0});
    scene.materials.push_back({.kind = gpu::material_kind::metal, .albedo = {0.9, 0.8, 0.7}, .fuzz = 0.1});
    scene.materials.push_back({.kind = gpu::material_kind::isotropic, .texture = 0});
    scene.spheres.push_back({.center = {0, 0, -3}, .radius = 1, .material = 0});
    scene.spheres.push_back({.center = {2, 0, -3}, .radius = 0.5, .material = 1});
    scene.quads.push_back({.q = {-5, -1, -8}, .u = {10, 0, 0}, .v = {0, 0, 10}, .material = 0});
    gpu::medium_desc smoke{.boundary = gpu::boundary_shape::sphere, .neg_inv_density = -2, .material = 2};
    smoke.sphere = {.center = {-2, 0, -3}, .radius = 0.75};
    scene.media.push_back(smoke);

    const gpu_test::device_scene device(scene);
    ctx.exact("surface-count", device.surface_count(), 3, 0);
    ctx.exact("sphere-ids", device.sphere_ids().size() == 2 && device.sphere_ids()[0] != device.sphere_ids()[1], 1, 0);
    ctx.exact("quad-id", device.quad_ids().size() == 1 ? device.quad_ids()[0] : -1, device.view().tree.quad_base, 0);
    ctx.exact("media-base", device.view().media.base, 3, 0);
    ctx.exact("media-count", device.view().media.count, 1, 0);
    ctx.exact("sampling-material",
              device.view().sampling == gpu::sampling_strategy::material && !device.view().motion ? 1 : 0, 1, 0);
}

GPU_TEST(services, host)
{
    // Every byte through bytes_to_linear, image::save_png and image::load comes back as itself
    std::vector<std::uint8_t> rgba;
    for (int b = 0; b < 256; ++b)
    {
        rgba.insert(rgba.end(), {static_cast<std::uint8_t>(b), static_cast<std::uint8_t>(255 - b),
                                 static_cast<std::uint8_t>(b / 2), 255});
    }
    const std::vector<std::uint8_t> back = gpu_test::host::png_round_trip(rgba, 16, 16);
    int differ = 0;
    for (std::size_t i = 0; i < 256; ++i)
    {
        for (std::size_t k = 0; k < 3; ++k)
        {
            differ += back[3 * i + k] == rgba[4 * i + k] ? 0 : 1;
        }
    }
    ctx.count("png-round-trip", differ, 0, 768);
    ctx.exact("find-repo-file", gpu_test::host::find_repo_file("src/gpu/renderer.h").filename() == "renderer.h", 1, 0);
}
}
