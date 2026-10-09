# RayTracing

A learning ray tracer following Peter Shirley's book ["Ray Tracing in One Weekend"](https://raytracing.github.io/books/RayTracingInOneWeekend.html), rewritten in modern C++23: modules, `import std;`, multithreading and the BVH from the second book of the series.

![Final scene: 1200×675, 500 rays per pixel](docs/final.png)

*The book's final scene: 1200×675, 500 rays per pixel, 35 seconds on all cores (Release).*

## Features

- Spheres of three materials: matte (Lambertian), metal with blurred reflections, and glass with refraction, total internal reflection and glare by Schlick's approximation.
- Antialiasing (several random rays per pixel) and gamma correction.
- A camera with any position, field of view and depth of field.
- Multithreaded rendering on all cores. The result is deterministic: identical on every run, in Debug and in Release.
- BVH, a bounding volume hierarchy: speeds up rendering of the final scene about 8 times.
- Saving to PNG.

## Building and running

Requires Visual Studio 2026 with the "Desktop development with C++" workload: the MSVC compiler, CMake and Ninja come with it.

**In Visual Studio:** open the project folder, select the `x64 Release` preset and the `RayTracing.exe` startup item, press Ctrl+F5. The program prints the path to the finished image.

**From the command line** (Developer PowerShell for VS):

```
cmake --preset x64-release
cmake --build --preset x64-release
out\build\x64-release\RayTracing.exe -s 10
```

| Preset | Purpose |
|---|---|
| `x64 Debug` | step-by-step debugging; rendering is about 8 times slower |
| `x64 RelWithDebInfo` | optimizations plus debug information: profiling and debugging fast code |
| `x64 Release` | final rendering |

## Command-line options

```
RayTracing [options]
  -w, --width N      image width in pixels (default 1200)
  -s, --samples N    rays per pixel (default 500): fewer is faster but noisier
  -o, --output FILE  where to save the PNG (default image.png)
  -h, --help         show help
```

On all cores at width 1200: `-s 10` takes under a second (draft), `-s 500` about 35 seconds.

To pass options when launching from Visual Studio, open **Debug → Debug and Launch Settings for RayTracing** and add `"args": ["-s", "10"]` to the configuration.

## Structure

Each file in `src/` is a C++20 module (`export module ...`):

| Module | Contents |
|---|---|
| `vec3` | vector for points, directions and colors (`point3`, `color`), random vectors, reflection and refraction |
| `ray` | the ray `P(t) = origin + t·direction` |
| `interval` | the range `[min, max]` |
| `random` | random numbers: each thread has its own generator |
| `aabb` | axis-aligned box and a fast test of whether a ray hits it |
| `hittable` | hit record, the `hittable` and `material` interfaces, object list |
| `sphere` | sphere |
| `material` | `lambertian`, `metal`, `dielectric` |
| `bvh` | tree of boxes for fast intersection search |
| `camera` | camera and multithreaded rendering |
| `image` | in-memory image, gamma correction, PNG writing |
| `main.cpp` | command-line options and the final scene |

Third-party code: only [stb_image_write](https://github.com/nothings/stb) in `external/stb`.

## Modern C++ in the project

- **Modules and `import std;`** (C++20/23) instead of headers. The `std` module is built from `std.ixx`, which ships with MSVC (see `CMakeLists.txt`).
- **`constexpr` and `inline` in modules.** A function defined inside a class in a named module doesn't become `inline` by itself. Without this, `vec3` calls wouldn't be inlined into other modules (`vec3.ixx`).
- **Multidimensional `operator[]` and deducing this** (C++23): `img[x, y]` with one function for const and non-const (`image.ixx`).
- **`std::expected`** (C++23) for errors and **`std::optional`** instead of a "`bool` + out parameter" pair.
- **Designated initializers** (C++20) for camera settings: `camera cam({.image_width = 400, .vfov = 20})`.
- **`std::jthread` and `std::atomic::wait`/`notify`** (C++20) in the multithreaded renderer (`camera.ixx`).
- **`std::ranges::sort` with a projection** (C++20) when building the BVH; **`std::span`**, **`std::from_chars`**, **`std::print`**.
- **Ref-qualifier `&&`** on a member function: `std::move(list).release_objects()`.
- **Ownership:** objects via `std::unique_ptr`, shared materials via `std::shared_ptr<const material>`, and a non-owning `const material*` in the hot loop.
- **UTF-8 everywhere:** `/utf-8` for sources and strings, and the `src/utf8.manifest` manifest switches the process to UTF-8 (Windows 10 1903+), so paths with Cyrillic characters work in arguments too.

## Commit history as a lesson plan

Each commit is one step, roughly one book chapter, with a description of what was done and why:

```
git log --reverse
```

To see the project at any step: `git switch --detach <commit>`; to come back: `git switch main`.

## What's next

- The second book of the series, ["Ray Tracing: The Next Week"](https://raytracing.github.io/books/RayTracingTheNextWeek.html): motion blur, textures, Perlin noise, rectangles, light sources, smoke and fog. Its BVH is already here.
- The third, ["Ray Tracing: The Rest of Your Life"](https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html): Monte Carlo for real and importance sampling, i.e. less noise for the same number of rays.
- Performance: `float` instead of `double`, SIMD, rendering in tiles instead of rows.
- A window with progressive rendering (e.g. SDL3 via vcpkg).
