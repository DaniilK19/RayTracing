# RayTracing

A learning ray tracer following the first two books by Peter Shirley, ["Ray Tracing in One Weekend"](https://raytracing.github.io/books/RayTracingInOneWeekend.html) and ["Ray Tracing: The Next Week"](https://raytracing.github.io/books/RayTracingTheNextWeek.html), rewritten in modern C++23: modules, `import std;`, multithreading.

![Final scene of the second book: 1080×1080, 5000 rays per pixel](docs/final.png)

*Final scene of the second book: 1080×1080, 5000 rays per pixel, 24 minutes on all cores (Release).*

## Scenes

All scenes from both books; the scene name is the first argument: `RayTracing cornell`.

| | |
|---|---|
| ![weekend](docs/scenes/weekend.png) **weekend**: the first book's cover, spheres of three materials ([in Full HD](docs/weekend.png)) | ![bouncing](docs/scenes/bouncing.png) **bouncing**: the same, but the matte spheres bounce, giving motion blur |
| ![checker](docs/scenes/checker.png) **checker**: a solid checker texture | ![earth](docs/scenes/earth.png) **earth**: an image wrapped around a sphere |
| ![perlin](docs/scenes/perlin.png) **perlin**: marble made of Perlin noise | ![quads](docs/scenes/quads.png) **quads**: quadrilaterals |
| ![light](docs/scenes/light.png) **light**: light sources | ![cornell](docs/scenes/cornell.png) **cornell**: the Cornell box, translated and rotated objects |
| ![smoke](docs/scenes/smoke.png) **smoke**: blocks of smoke | ![final](docs/scenes/final.png) **final**: the second book's final scene, everything together |

## Features

**First book:**

- Spheres of three materials: matte (Lambertian), metal with blurred reflections, and glass with refraction, total internal reflection and glare by Schlick's approximation.
- Antialiasing (several random rays per pixel) and gamma correction.
- A camera with any position, field of view and depth of field.

**Second book:**

- Motion blur: each ray is emitted at a random moment while the shutter is open, and spheres can move.
- BVH, a bounding volume hierarchy: speeds up rendering of the first book's cover about 8 times.
- Textures: checker, an image from a file (JPEG, PNG) and marble made of Perlin noise with turbulence.
- Quadrilaterals and boxes made of them.
- Light sources: emissive materials and a configurable background.
- Instances: any object can be translated and rotated without recomputing it.
- Smoke and fog: volumes uniformly filled with particles.

**Beyond the books:**

- Multithreaded rendering on all cores. The result is deterministic: identical on every run, in Debug and in Release.
- Choosing the scene and quality from the command line, saving to PNG.

## Building and running

Requires Visual Studio 2026 with the "Desktop development with C++" workload: the MSVC compiler, CMake and Ninja come with it.

**In Visual Studio:** open the project folder, select the `x64 Release` preset and the `RayTracing.exe` startup item, press Ctrl+F5. The program renders the final scene and prints the path to the finished image.

**From the command line** (Developer PowerShell for VS):

```
cmake --preset x64-release
cmake --build --preset x64-release
out\build\x64-release\RayTracing.exe cornell -s 100
```

| Preset | Purpose |
|---|---|
| `x64 Debug` | step-by-step debugging; rendering is about 8 times slower |
| `x64 RelWithDebInfo` | optimizations plus debug information: profiling and debugging fast code |
| `x64 Release` | final rendering |

The program looks for the Earth texture in the `assets` folder, in the current folder and up the tree, so run it from the project root or from Visual Studio. If the texture isn't found, the Earth will be cyan, and the program will warn about it.

## Command-line options

```
RayTracing [scene] [options]
  scene              weekend, bouncing, checker, earth, perlin, quads, light,
                     cornell, smoke or final (default final)
  -w, --width N      image width in pixels; the height follows the scene's aspect ratio
  -s, --samples N    rays per pixel: fewer is faster but noisier
  -o, --output FILE  where to save the PNG (default <scene>.png)
  -h, --help         help and the list of scenes
```

Each scene has its own default size and number of rays. Time on all cores in Release:

| Scene | Size | Rays per pixel | Time |
|---|---|---|---|
| `weekend` | 1920×1080 | 500 | ~1.5 min |
| `bouncing` | 1920×1080 | 500 | ~1.7 min |
| `checker` | 1920×1080 | 500 | ~35 s |
| `earth` | 1920×1080 | 500 | ~7 s |
| `perlin` | 1920×1080 | 500 | ~40 s |
| `quads` | 1080×1080 | 500 | ~10 s |
| `light` | 1920×1080 | 1000 | ~45 s |
| `cornell` | 1080×1080 | 1000 | ~2.7 min |
| `smoke` | 1080×1080 | 1000 | ~3 min |
| `final` | 1080×1080 | 1000 | ~5 min |

Time grows in proportion to the number of pixels and the number of rays per pixel: for a quick draft add `-s 10`. Scenes in a closed room with a small lamp (`cornell`, `smoke`, `final`) are noisier than the rest: only a small fraction of the random rays reaches the lamp. The third book explains how to deal with this.

To pass options when launching from Visual Studio, open **Debug → Debug and Launch Settings for RayTracing** and add `"args": ["cornell", "-s", "100"]` to the configuration.

## Structure

Each file in `src/` is a C++20 module (`export module ...`):

| Module | Contents |
|---|---|
| `vec3` | vector for points, directions and colors (`point3`, `color`), random vectors, reflection and refraction |
| `ray` | the ray `P(t) = origin + t·direction` and the moment it is emitted |
| `interval` | the range `[min, max]` |
| `random` | random numbers: each thread has its own generator |
| `aabb` | axis-aligned box and a fast test of whether a ray hits it |
| `hittable` | hit record, the `hittable` and `material` interfaces, object list |
| `sphere` | sphere, possibly moving |
| `quad` | quadrilateral and a box of six such faces |
| `instance` | `translate` and `rotate_y`: a translated or rotated object |
| `constant_medium` | smoke and fog |
| `bvh` | tree of boxes for fast intersection search |
| `texture` | textures: solid color, checker, image, marble |
| `perlin` | Perlin noise and turbulence |
| `material` | `lambertian`, `metal`, `dielectric`, `diffuse_light`, `isotropic` |
| `camera` | camera, background and multithreaded rendering |
| `image` | in-memory image, loading from a file, gamma correction, PNG writing |
| `scenes` | all scenes from both books and the list for choosing one by name |
| `main.cpp` | command-line options |

Third-party code: only [stb_image and stb_image_write](https://github.com/nothings/stb) in `external/stb`. The Earth map `assets/earthmap.jpg` comes from the [books' repository](https://github.com/RayTracing/raytracing.github.io) (CC0).

## Modern C++ in the project

- **Modules and `import std;`** (C++20/23) instead of headers. The `std` module is built from `std.ixx`, which ships with MSVC (see `CMakeLists.txt`). The stb libraries are included in the global module fragment of the `image` module and are not visible outside it.
- **`constexpr` and `inline` in modules.** A function defined inside a class in a named module doesn't become `inline` by itself. Without this, `vec3` calls wouldn't be inlined into other modules (`vec3.ixx`).
- **Multidimensional `operator[]` and deducing this** (C++23): `img[x, y]` with one function for const and non-const (`image.ixx`).
- **`std::expected`** (C++23) for errors and **`std::optional`** instead of a "`bool` + out parameter" pair; `std::optional` and `value_or` for options the scene sets itself unless they are given.
- **Designated initializers** (C++20) for camera settings: `{.aspect_ratio = 1.0, .vfov = 40}`.
- **`std::jthread` and `std::atomic::wait`/`notify`** (C++20) in the multithreaded renderer (`camera.ixx`).
- **Algorithms with projections** (C++20): `std::ranges::sort` when building the BVH, `std::ranges::find(scene_list, name, &scene_entry::name)` when choosing a scene; **`std::ranges::iota`** (C++23) in Perlin noise.
- **Scene table**: a `constexpr std::array` of function pointers, so only the chosen scene gets built.
- **`std::unique_ptr` with a custom deleter** frees memory allocated by a C library (`image::load`).
- **Delegating constructors**: a stationary sphere is a special case of a moving one, a color is a special case of a texture.
- **Ref-qualifier `&&`** on a member function: `std::move(list).release_objects()`.
- **Ownership:** objects via `std::unique_ptr` (wrappers nest: `translate(rotate_y(box))`), shared materials and textures via `std::shared_ptr<const ...>`, and a non-owning `const material*` in the hot loop.
- **UTF-8 everywhere:** `/utf-8` for sources and strings, and the `src/utf8.manifest` manifest switches the process to UTF-8 (Windows 10 1903+), so paths with Cyrillic characters work in arguments too.

## Commit history as a lesson plan

Each commit is one step, roughly one book chapter, with a description of what was done and why:

```
git log --reverse
```

To see the project at any step: `git switch --detach <commit>`; to come back: `git switch main`.

## What's next

- The third book of the series, ["Ray Tracing: The Rest of Your Life"](https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html): Monte Carlo for real and importance sampling. Rays are sent toward the lamp on purpose, so `cornell` and `final` get much less noise for the same number of rays.
- CPU performance: `float` instead of `double`, SIMD, rendering in tiles instead of rows.
- GPU rendering: CUDA or DirectX 12 compute shaders. Every pixel is computed independently, so ray tracing maps very well onto thousands of GPU threads.
- A window with progressive rendering (e.g. SDL3 via vcpkg).
