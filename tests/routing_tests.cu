// Suite routing: which kernel a session takes, and what it refuses. The programs refuse the quality kernel's
// impossible combinations first, in their own words (check_kernel in gpu_scene.ixx); the session refuses them again,
// for any caller of the renderer: float64, the kernel for all three books asked for at once, and a books strategy other
// than the material's. These refusals come before the session touches the GPU. A session the quality kernel can render
// is made, and says it renders with it.
#include <stdexcept>
#include <string>

#include "gpu_test.h"
#include "renderer.h"

namespace
{
// A tiny scene: one matte sphere under a sky, a 4×4 image of one sample, sampled as the material scatters
gpu::scene_desc tiny_scene()
{
    gpu::scene_desc scene;
    scene.textures.push_back({.kind = gpu::texture_kind::solid, .rgb = {0.5, 0.5, 0.5}});
    scene.materials.push_back({.kind = gpu::material_kind::lambertian, .texture = 0});
    scene.spheres.push_back({.center = {0, 0, -2}, .radius = 1, .material = 0});
    scene.camera = {.image_width = 4,
                    .image_height = 4,
                    .sqrt_spp = 1,
                    .max_depth = 4,
                    .center = {0, 0, 0},
                    .pixel00_loc = {-0.375, 0.375, -1},
                    .pixel_delta_u = {0.25, 0, 0},
                    .pixel_delta_v = {0, -0.25, 0},
                    .sky_bottom = {1, 1, 1},
                    .sky_top = {0.5, 0.7, 1},
                    .sampling = gpu::sampling_strategy::material};
    return scene;
}

// 1 when making the session throws std::invalid_argument, 0 otherwise
double refused(const gpu::scene_desc& scene, const gpu::session_options& options)
{
    try
    {
        gpu::session s(scene, options);
    }
    catch (const std::invalid_argument&)
    {
        return 1;
    }
    catch (const std::exception&)
    {
        return 0;
    }
    return 0;
}

GPU_TEST(routing, refusals)
{
    const gpu::scene_desc scene = tiny_scene();
    ctx.exact("quality-refuses-float64",
              refused(scene, {.arithmetic = gpu::precision::float64, .quality_kernel = true}), 1, 0);
    ctx.exact("quality-refuses-all-books-kernel", refused(scene, {.all_books_kernel = true, .quality_kernel = true}),
              1, 0);
    gpu::scene_desc mixture = scene;
    mixture.camera.sampling = gpu::sampling_strategy::mixture;
    mixture.lights.push_back({.kind = gpu::light_shape::sphere, .center = {0, 0, -2}, .radius = 1, .weight = 1});
    ctx.exact("quality-refuses-mixture", refused(mixture, {.quality_kernel = true}), 1, 0);
    // The books' kernels take what the quality kernel refuses
    ctx.exact("books-take-float64", refused(scene, {.arithmetic = gpu::precision::float64}), 0, 0);
    ctx.exact("books-take-mixture", refused(mixture, {}), 0, 0);
}

GPU_TEST(routing, kernels)
{
    const gpu::scene_desc scene = tiny_scene();
    const auto kernel_of = [&scene](const gpu::session_options& options) {
        return gpu::session(scene, options).kernel().name;
    };
    ctx.exact("first-book-scene-takes-first-book", kernel_of({}) == "first book" ? 1 : 0, 1, 0);
    ctx.exact("all-books-kernel-asked", kernel_of({.all_books_kernel = true}) == "all books" ? 1 : 0, 1, 0);
    ctx.exact("quality-kernel-asked", kernel_of({.quality_kernel = true}) == "quality" ? 1 : 0, 1, 0);
    // And it renders: one sample of 16 pixels, every one of them counted as a ray at least
    gpu::session s(scene, {.quality_kernel = true});
    s.render_to({1, 0});
    ctx.exact("quality-renders", s.rays() >= 16 ? 1 : 0, 1, 0);
    ctx.exact("quality-material-only-no-shadow-rays", static_cast<double>(s.shadow_rays()), 0, 0);
}
}
