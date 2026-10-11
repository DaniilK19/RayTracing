# RayTracing

A learning ray tracer following all three books of Peter Shirley's series, ["Ray Tracing in One Weekend"](https://raytracing.github.io/books/RayTracingInOneWeekend.html), ["Ray Tracing: The Next Week"](https://raytracing.github.io/books/RayTracingTheNextWeek.html) and ["Ray Tracing: The Rest of Your Life"](https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html), rewritten in modern C++23: modules, `import std;`, multithreading, concepts and `std::variant`.

| ![Final scene of the second book](docs/final.png) | ![Final scene of the third book](docs/glass.png) |
|---|---|
| *The second book's final scene, rendered with the third book's light sampling: 1080×1080, 10,000 rays per pixel, 39 minutes.* | *The third book's final scene: a glass sphere in the Cornell box, sampled like a light, which makes the caustic under it (the light the glass focuses onto the floor) less noisy: 1080×1080, 4096 rays per pixel, 8 minutes.* |

*Both rendered on the CPU, on all cores, in Release.*

## Scenes

All scenes from the three books; the scene name is the first argument: `RayTracing cornell`.

| | |
|---|---|
| ![weekend](docs/scenes/weekend.png) **weekend**: the first book's cover, spheres of three materials ([in Full HD](docs/weekend.png)) | ![bouncing](docs/scenes/bouncing.png) **bouncing**: the same, but the matte spheres bounce, giving motion blur |
| ![checker](docs/scenes/checker.png) **checker**: a solid checker texture | ![earth](docs/scenes/earth.png) **earth**: an image wrapped around a sphere |
| ![perlin](docs/scenes/perlin.png) **perlin**: marble made of Perlin noise | ![quads](docs/scenes/quads.png) **quads**: quadrilaterals |
| ![light](docs/scenes/light.png) **light**: light sources | ![cornell](docs/scenes/cornell.png) **cornell**: the Cornell box, translated and rotated objects |
| ![smoke](docs/scenes/smoke.png) **smoke**: blocks of smoke | ![final](docs/scenes/final.png) **final**: the second book's final scene, everything together |
| ![aluminum](docs/scenes/aluminum.png) **aluminum**: a mirror block; the light it throws onto the ceiling stays noisy | ![glass](docs/scenes/glass.png) **glass**: the third book's final scene, a glass sphere sampled like a light |

## Sampling: the third book in four pictures

The third book is about where to send a ray after a diffuse bounce, off a matte surface or a particle of smoke. The new direction may be drawn with any probability density p (a pdf: how likely each direction is to be drawn), if the light it brings back is then weighted by pScatter / p, where pScatter is the material's own density: how much of the light it scatters goes that way. A direction that p draws more often than the material would counts for less, and one that p draws less often counts for more. So the image converges to the same result whatever p is, as long as p > 0 wherever the material scatters light: the estimate is unbiased. What p changes is the noise, and choosing a p that lowers it is called importance sampling. The Cornell box with each strategy (`-p`), 480×480 and 64 rays per pixel:

| `-p uniform` | `-p material` | `-p lights` | `-p mixture` (default) |
|---|---|---|---|
| ![uniform](docs/sampling/uniform.png) | ![material](docs/sampling/material.png) | ![lights](docs/sampling/lights.png) | ![mixture](docs/sampling/mixture.png) |
| the same density in every direction | as the surface scatters, as in the first two books | only toward the lamp | half as the material, half toward the lamp |
| error 0.175, 0.49 s | error 0.092, 2.0 s | error 0.045, 0.44 s, but the image is 21% too dark | error 0.022, 0.69 s |

*Error: the root-mean-square error (RMSE) of the pixels against a reference rendered with 16384 rays per pixel, in linear light (before gamma correction). Time: for these 64 rays per pixel on all cores.*

`uniform` is the noisiest: half of its directions point into the surface and bring nothing. `lights` is far less noisy than `material`, but wrong: p is 0 in every direction that misses the lamp, which breaks the rule above. Its error is mostly missing light, not noise: all the light bounced off the walls is lost, and the ceiling, which sees only the dark back of the lamp (lamps glow on their front side only), is black.

The mixture keeps most of the low noise of `lights` and stays unbiased: wherever the material scatters, p is at least half of pScatter, so p > 0 there and the weight pScatter / p is never more than 2. It is also faster for the same number of rays per pixel: half of the bounces aim at the lamp, and since a lamp reflects nothing, a path that reaches it ends there. Paths get shorter.

The mixture can't help everywhere. In `aluminum` it aims at the lamp, not at its reflection in the mirror, so the light the mirror throws onto the ceiling stays noisy (the book shows the same). And smoke and fog don't mix the lamps in at all, which goes beyond the book: deep in a dense volume, light comes from other particles rather than straight from a lamp, and the mixture's weights of up to 2 per bounce would multiply along a path that scatters again and again, giving rare paths huge weights. In the final scene's blue fog that showed as bright specks and more than doubled the error, so volumes sample with their own density, where every weight is exactly 1.

## The MonteCarlo program

Before the ray tracer learns to choose its rays, the third book studies Monte Carlo estimation (an integral as the average of random samples) on small problems with known answers. The `MonteCarlo` program runs those experiments. Every estimate is printed next to the exact value, its standard error SE (how far a correct estimate typically lands from the exact value) and the error in SEs: up to about 3 SE is noise, much more points to a bug. For example, `MonteCarlo importance` integrates x² over [0, 2] with four sampling densities. The end of its output repeats each estimate 1000 times and compares the RMS error of these runs with σ/√N, the error predicted from the spread σ of a single sample f/p:

```
  The RMS error of 1000 runs with N = 100 samples each, next to σ/√N:
  PDF         p(x)                 σ of f/p      σ/√N       RMS  largest x
  uniform     1/2                  2.385139  0.238514  0.242261   1.999991
  two halves  1/(2√2) | 1/(4-2√2)  1.503819  0.150382  0.151427   1.999983
  linear      x/2                  0.942809  0.094281  0.091921   1.999998
  quadratic   3x²/8                0.000000  0.000000  0.000000   1.999990
```

The closer the density p is in shape to the function being integrated, f(x) = x², the smaller the error; a p proportional to f gives the exact answer with every sample. The last column, the largest x drawn, catches a mistake in the book: its inverse CDF for the quadratic density (the formula that turns a uniform random number into an x with that density) draws numbers up to 8, outside [0, 2]. The estimate itself can't show this, because f(x) / p(x) = 8/3 for every x.

| Experiment | Book section | What it shows |
|---|---|---|
| `pi` | 2.1 | estimate π from random points in a square |
| `convergence` | 2.2 | the law of diminishing returns: the error falls as 1/√N |
| `stratified` | 2.3 | one random point in each cell of a grid (jittering) vs points anywhere |
| `integrate` | 3.2 | integrals as averages of random samples: x², sin⁵ x and ln(sin x) over [0, 2] |
| `halfway` | 3.5 | the point that halves the area under a curve, found by sorting the samples |
| `importance` | 3.7 | ∫ x² dx with four densities: the closer p is to f, the smaller the error |
| `sphere` | 4 | ∫ cos²θ dω over the sphere of directions, with densities per steradian |
| `sphere-plot` | 7.1 | 200 random directions as "x y z" lines, the input of a 3D scatter plot |
| `cos-cubed` | 7.2 | ∫ cos³θ dω over the hemisphere: uniform vs cosine-distributed directions |
| `light-pdfs` | 9.1 | beyond the book: do `pdf_value()` and `random()` agree for quads, spheres, lists and instances? |

Run without arguments, `MonteCarlo` takes about 6 s in Release and runs every experiment except `sphere-plot`; `MonteCarlo importance sphere` runs only the named ones, `MonteCarlo sphere-plot -o points.txt` writes the plot's points to a file, and `MonteCarlo -h` lists them all.

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
- Instances: any object can be translated and rotated around the vertical axis without recomputing it.
- Smoke and fog: volumes uniformly filled with particles.

**Third book:**

- Monte Carlo integration from the ground up, as a separate program (see above).
- Stratified pixel samples: the rays of a pixel form a k×k grid of cells with one random point each, which makes edges and textures less noisy (the noise of the lighting hardly changes).
- Importance sampling made explicit: materials describe how they scatter with a probability density, and every diffuse bounce is weighted by pScatter / p.
- Cosine-weighted directions drawn around the z axis by the inversion method (two random numbers, no rejection loop) and turned to point around the normal with an orthonormal basis.
- Light sampling: quads, spheres, lists of them and (beyond the book) translated or rotated instances know the density of the directions toward them and can draw such directions. Lamps glow on their front side only.
- A mixture of the material's density and the lamps' density, the default: the scenes lit by small lamps get far less noisy for the same number of rays, and most of them render faster too.
- Mirrors and glass stay a special case that skips the densities: the camera follows the one ray they choose, as in the first two books (a perfect mirror's density would be infinite in one direction and 0 in all others).
- Two new scenes: an aluminum block and a glass sphere in the Cornell box.

**Beyond the books:**

- Multithreaded rendering on all cores. The result is deterministic: identical on every run, in Debug and in Release.
- Choosing the scene, the quality and the sampling strategy from the command line, saving to PNG.
- No heap allocation per bounce: the book creates its densities with `make_shared` at every bounce; here they are small values on the stack.
- Smoke and fog keep sampling with their own density under the default mixture: mixing the lamps in multiplies the weights along paths that scatter many times (see the sampling section).
- The program prints the image's mean radiance (its average brightness in linear light, before clamping): an unbiased sampling strategy changes the noise but leaves this number the same up to that noise, while a biased one, like `-p lights`, moves it.
- Mistakes in the third book are fixed rather than repeated, each explained in a comment next to the fix: the inverse CDF of section 3.7 (it draws x up to 8 instead of 2), the uniform density of section 6.2 (it makes the image too bright, not just noisier), reading the first element of an empty list of lights, light directions hundreds of units long (the search for a hit starts at t = 0.001 of their length, so they skip the first half unit of their way and anything there), and NaN from the square root of a negative number for a point inside a spherical lamp.

## Building and running

Requires Visual Studio 2026 with the "Desktop development with C++" workload: the MSVC compiler, CMake and Ninja come with it.

**In Visual Studio:** open the project folder, select the `x64 Release` preset and the `RayTracing.exe` startup item, press Ctrl+F5. The program renders the final scene and prints the path to the finished image. The `MonteCarlo.exe` startup item runs the Monte Carlo experiments.

**From the command line** (Developer PowerShell for VS):

```
cmake --preset x64-release
cmake --build --preset x64-release
out\build\x64-release\RayTracing.exe cornell -s 100
out\build\x64-release\MonteCarlo.exe
```

| Preset | Purpose |
|---|---|
| `x64 Debug` | step-by-step debugging; rendering is about 10 times slower |
| `x64 RelWithDebInfo` | optimizations plus debug information: profiling and debugging fast code |
| `x64 Release` | final rendering |

The program looks for the Earth texture in the `assets` folder, in the current folder and up the tree, so run it from the project root or from Visual Studio. If the texture isn't found, the Earth will be cyan, and the program will warn about it.

## Command-line options

```
RayTracing [scene] [options]
  scene                weekend, bouncing, checker, earth, perlin, quads, light, cornell,
                       smoke, final, aluminum or glass (default final)
  -w, --width N        image width in pixels; the height follows the scene's aspect ratio
  -s, --samples N      rays per pixel, rounded to the nearest square k²: the rays form
                       a k×k grid in the pixel
  -p, --sampling NAME  how a diffuse bounce chooses the next direction: uniform, material,
                       lights or mixture (default)
  -o, --output FILE    where to save the PNG (default <scene>.png)
  -h, --help           help, the lists of scenes and sampling strategies
```

Each scene has its own default size and number of rays. Time on all cores in Release:

| Scene | Size | Rays per pixel | Time |
|---|---|---|---|
| `weekend` | 1920×1080 | 484 | ~1.4 min |
| `bouncing` | 1920×1080 | 484 | ~1.6 min |
| `checker` | 1920×1080 | 484 | ~30 s |
| `earth` | 1920×1080 | 484 | ~6 s |
| `perlin` | 1920×1080 | 484 | ~33 s |
| `quads` | 1080×1080 | 484 | ~9 s |
| `light` | 1920×1080 | 1024 | ~47 s |
| `cornell` | 1080×1080 | 1024 | ~54 s |
| `smoke` | 1080×1080 | 1024 | ~1.4 min |
| `final` | 1080×1080 | 1024 | ~4 min |
| `aluminum` | 1080×1080 | 1024 | ~1 min |
| `glass` | 1080×1080 | 1024 | ~2 min |

Time grows in proportion to the number of pixels and the number of rays per pixel: for a quick draft add `-s 16`. The scenes lit by small lamps (`light`, `cornell`, `smoke`, `final`, `aluminum`, `glass`) aim half of the diffuse bounces off their surfaces at the lamps (`glass` also at its glass sphere); with `-p material` they are sampled as in the first two books: noisier, and all but `light` slower. The other scenes are lit by the sky and have no lamps to aim at, so `-p lights` and `-p mixture` render them as `-p material` does, and the program says so.

To pass options when launching from Visual Studio, open **Debug → Debug and Launch Settings for RayTracing** and add `"args": ["cornell", "-s", "100"]` to the configuration.

## Structure

Each `.ixx` file in `src/` is a C++20 module (`export module ...`). The modules are compiled once into a static library, `raytracer`, and both programs link it:

| Module | Contents |
|---|---|
| `vec3` | vector for points, directions and colors (`point3`, `color`), random vectors and directions, reflection and refraction |
| `ray` | the ray `P(t) = origin + t·direction` and the moment it is emitted |
| `interval` | the range `[min, max]` |
| `random` | random numbers: each thread has its own generator |
| `onb` | orthonormal basis: turns directions drawn around the z axis to point around a normal |
| `pdf` | densities of directions: the `direction_pdf` concept, uniform, cosine and mixture densities |
| `aabb` | axis-aligned box and a fast test of whether a ray hits it |
| `hittable` | hit record, the `hittable` and `material` interfaces, how a material scatters, object list, the density toward objects |
| `sphere` | sphere, possibly moving; can be sampled as a light |
| `quad` | quadrilateral and a box of six such faces; can be sampled as a light |
| `instance` | `translate` and `rotate_y`: a translated or rotated object; can be sampled as a light |
| `constant_medium` | smoke and fog |
| `bvh` | tree of boxes for fast intersection search |
| `texture` | textures: solid color, checker, image, marble |
| `perlin` | Perlin noise and turbulence |
| `material` | `lambertian`, `metal`, `dielectric`, `diffuse_light`, `isotropic` |
| `camera` | camera, background, stratified pixel samples, sampling strategies, multithreaded rendering |
| `image` | in-memory image, loading from a file, gamma correction, PNG writing |
| `scenes` | all scenes from the three books and the list for choosing one by name |

The two programs: `main.cpp` (`RayTracing`, the command-line options) and `monte_carlo.cpp` (`MonteCarlo`, the third book's numerical experiments).

Third-party code: only [stb_image and stb_image_write](https://github.com/nothings/stb) in `external/stb`. The Earth map `assets/earthmap.jpg` comes from the [books' repository](https://github.com/RayTracing/raytracing.github.io) (CC0).

## Modern C++ in the project

- **Modules and `import std;`** (C++20/23) instead of headers. The `std` module is built from `std.ixx`, which ships with MSVC (see `CMakeLists.txt`). The stb libraries are included in the global module fragment of the `image` module and are not visible outside it. An `import` isn't passed on (only `export import` is): `image` imports `vec3`, but a file that imports `image` still has to import `vec3` to use `color`, so every file imports what it uses.
- **`constexpr` and `inline` in modules.** A function defined inside a class in a named module doesn't become `inline` by itself. Without this, `vec3` calls wouldn't be inlined into other modules (`vec3.ixx`).
- **Static and dynamic polymorphism side by side.** Objects and materials are open families of long-lived objects, so they use virtual functions. Densities of directions are a closed set of short-lived values on the hottest path, so they do without them: a **concept** (`direction_pdf`, C++20) instead of a base class, a class template `mixture_pdf` whose template arguments are deduced from the constructor's arguments (**CTAD**, C++17), a generic lambda with a constrained `auto` parameter (C++20), and **`std::variant` with `std::visit`** (C++17) for the density a material returns (`pdf.ixx`, `camera.ixx`).
- **`std::variant` and `std::get_if`** for what a material does with a ray: a diffuse scatter (with a density) or a specular one (with a ready ray), wrapped in `std::optional` for absorption.
- **Ranges.** C++20 algorithms with projections: `std::ranges::sort` when building the BVH and in MonteCarlo, `std::ranges::find(scene_list, name, &scene_entry::name)` when choosing a scene. C++23: `std::views::cartesian_product` for the cells of a pixel, `std::views::zip` in MonteCarlo's convergence table, `std::ranges::fold_left` for the image's mean, `std::ranges::contains` in MonteCarlo's command line, `std::ranges::iota` in Perlin noise.
- **`std::unreachable`** (C++23) after a `switch` that returns in every case.
- **Multidimensional `operator[]` and deducing this** (C++23): `img[x, y]` with one function for const and non-const (`image.ixx`).
- **`std::expected`** (C++23) for errors and **`std::optional`** instead of a "`bool` + out parameter" pair; `std::optional` and `value_or` for options the scene sets itself unless they are given.
- **Designated initializers** (C++20) for camera settings: `{.aspect_ratio = 1.0, .vfov = 40}`.
- **`std::jthread` and `std::atomic::wait`/`notify`** (C++20) in the multithreaded renderer (`camera.ixx`).
- **Tables of scenes, sampling strategies and experiments**: `constexpr std::array`s. A scene's entry holds a pointer to the function that builds it, so only the chosen scene gets built; an experiment's entry holds the function that runs it.
- **`std::unique_ptr` with a custom deleter** frees memory allocated by a C library (`image::load`).
- **Delegating constructors**: a stationary sphere is a special case of a moving one, a color is a special case of a texture.
- **Ref-qualifier `&&`** on a member function: `std::move(list).release_objects()`.
- **Ownership:** objects via `std::unique_ptr` (wrappers nest: `translate(rotate_y(box))`), shared materials and textures via `std::shared_ptr<const ...>`, and a non-owning `const material*` in the hot loop.
- **UTF-8 everywhere:** `/utf-8` for sources and strings, and the `src/utf8.manifest` manifest switches the process to UTF-8 (Windows 10 1903+), so paths with Cyrillic characters work in arguments too.

## Commit history as a lesson plan

Each step is one commit, roughly one book chapter, with a description of what was done and why: steps 1–17 follow the first book (step 16 brings in the second book's BVH early), 18–28 the second and 29–42 the third.

```
git log --reverse
```

To see the project at any step: `git switch --detach <commit>`; to come back: `git switch main`.

## What's next

- GPU rendering: CUDA or DirectX 12 compute shaders. Every pixel is computed independently, so ray tracing maps very well onto thousands of GPU threads.
- CPU performance: `float` instead of `double`, SIMD, a BVH built with the surface area heuristic, rendering in tiles instead of rows.
- Multiple importance sampling: weighting the material's and the lamps' samples by the power heuristic instead of a fixed half-and-half mixture.
- A glossy material between matte and mirror (the third book's parting advice), and environment lighting from an HDR image.
- A window with progressive rendering (e.g. SDL3 via vcpkg).
