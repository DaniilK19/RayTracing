# Building and running

[RayTracing](../README.md) › Building and running

- [Requirements](#requirements)
- [SDL3 for the window](#sdl3-for-the-window)
- [Building](#building)
- [RayTracing options](#raytracing-options)
- [Render times](#render-times)
- [Scenes](#scenes)
- [The other programs](#the-other-programs)

## Requirements

Visual Studio 2026 with the "Desktop development with C++" workload, which brings the MSVC compiler, CMake and Ninja. `RayTracingGPU` and `RayTracingViewer` also need an NVIDIA graphics card and the CUDA Toolkit 13; see [the CUDA renderer](gpu.md#building-and-running). `RayTracingViewer` also needs SDL3, which CMake downloads.

## SDL3 for the window

`RayTracingViewer` draws its window with [SDL 3.4.18](https://github.com/libsdl-org/SDL/releases/tag/release-3.4.18) (zlib license). The first configure downloads SDL's release archive (about 15 MiB) into `out/downloads`, where every preset's build finds it, checks its SHA-256, and builds SDL as a static library, so the program needs no DLL. SDL is downloaded and built only when the CUDA Toolkit is found, as the window has nothing to show without it.

The download never stops the configure: if it fails, or gets the wrong bytes, CMake warns that it skipped the window and builds everything else. Without a network, get `SDL3-3.4.18.tar.gz` from SDL's release page by hand and point CMake at it, or at an unpacked SDL source folder, or turn the window off:

```
cmake --preset x64-release -DRAYTRACING_SDL3_URL=file:///C:/Downloads/SDL3-3.4.18.tar.gz
cmake --preset x64-release -DFETCHCONTENT_SOURCE_DIR_SDL3=C:/src/SDL3-3.4.18
cmake --preset x64-release -DRAYTRACING_VIEWER=OFF
```

## Building

**In Visual Studio:** open the project folder, select the `x64 Release` preset and the `RayTracing.exe` startup item, and press Ctrl+F5. The program renders the `final` scene, its default, and prints the path to the finished image. The `MonteCarlo.exe` startup item runs the Monte Carlo experiments, and `ImageCompare.exe` compares two renders. `RayTracingGPU.exe` and `RayTracingViewer.exe` are there if the CUDA Toolkit is installed.

To pass options when launching from Visual Studio, open **Debug → Debug and Launch Settings for RayTracing** and add `"args": ["cornell", "-s", "100"]` to the configuration.

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

## RayTracing options

```
RayTracing [scene] [options]
  scene                materials, weekend, bouncing, checker, earth, perlin, quads, light,
                       cornell, smoke, final, aluminum or glass (default final)
  -w, --width N        image width in pixels; the height follows the scene's aspect ratio
  -s, --samples N      rays per pixel, rounded to the nearest square k²: the rays form
                       a k×k grid in the pixel
  -p, --sampling NAME  how a diffuse bounce chooses the next direction: uniform, material,
                       lights or mixture (default)
      --seed N         which random numbers to draw, from 0 (default) to 4294967295
  -o, --output FILE    where to save the image, a .png or a .pfm (default <scene>.png);
                       give it several times to save several files
  -h, --help           help, the lists of scenes and sampling strategies
```

The same seed always gives the same image, bit for bit; another seed renders the same scene with independent noise. A `.pfm` (Portable Float Map) keeps the linear values as 32-bit floats, before gamma correction and clamping. So it can tell a lamp of brightness 15 from plain white, and a difference of 0.1% from none, which a PNG can't. HDR viewers such as [tev](https://github.com/Tom94/tev) open it.

After the render the program prints its time, the rays it traced and the image's mean radiance. For example, for `RayTracing materials`:

```
Render time: 51.4 s
Rays traced: 3631.4 million (3.62 per sample), 70.6 million per second
Mean radiance (linear, before clamping): 0.266692
```

A ray traced is one search of the scene for the nearest hit: the camera ray or a bounce. Per sample, that is the average length of a path; per second, the renderer's speed. The mean radiance is the image's average brightness in linear light, before clamping. An unbiased sampling strategy changes the noise but leaves this number the same up to that noise, while a biased one, like `-p lights`, moves it.

## Render times

Each scene has its own default size and number of rays. These are the CPU renderer's times in Release, on all cores of the same desktop machine as [the performance table](../README.md#performance):

| Scene | Size | Rays per pixel | Time |
|---|---|---|---|
| `materials` | 1920×1080 | 484 | ~50 s |
| `weekend` | 1920×1080 | 484 | ~1.5 min |
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

Time grows in proportion to the number of pixels and the number of rays per pixel: for a quick draft, add `-s 16`. The scenes lit by small lamps render with the mixture by default; [Sampling](sampling.md#four-strategies) compares it with the other strategies and says how each scene responds to them. The CUDA renderer's times for the same scenes are in [the performance table](../README.md#performance).

## Scenes

The 13 scenes as the CPU renderer draws them, 480 pixels wide. The scene name is the first argument: `RayTracing cornell`. The CUDA renderer's images, larger, are in [the gallery](../README.md#gallery).

| | |
|---|---|
| ![A blue matte sphere in focus between a blurred hollow glass sphere and a blurred fuzzy metal sphere, on a yellow-green ground](scenes/materials.png) **materials**: the first book's three materials up close, matte, fuzzy metal and hollow glass, through a wide-open lens | ![Hundreds of small matte, metal and glass spheres around three large ones of glass, brown matte and metal, on a gray ground](scenes/weekend.png) **weekend**: the first book's cover, spheres of three materials ([in Full HD](weekend.png)) |
| ![Hundreds of small matte, metal and glass spheres around three large ones of glass, brown matte and metal, on a green and white checkered ground, the small matte ones blurred by their vertical motion](scenes/bouncing.png) **bouncing**: the same, but the matte spheres bounce, giving motion blur | ![Two large spheres with a green and white checker texture, one above the other](scenes/checker.png) **checker**: a solid checker texture |
| ![The Earth wrapped around a sphere, the Americas facing the camera, against a pale sky](scenes/earth.png) **earth**: an image wrapped around a sphere | ![A marble sphere on a marble ground, both made of Perlin noise](scenes/perlin.png) **perlin**: marble made of Perlin noise |
| ![Five colored quadrilaterals against a pale sky: a green square in the center, framed by red, orange, blue and teal ones](scenes/quads.png) **quads**: quadrilaterals | ![The Cornell box: a white room with a green and a red wall, a ceiling lamp, and a tall and a short white box, each turned](scenes/cornell.png) **cornell**: the Cornell box, translated and rotated objects |
| ![The Cornell box with its tall box made of black smoke and its short box made of white smoke](scenes/smoke.png) **smoke**: blocks of smoke | ![The second book's final scene: the Earth, spheres of glass, fuzzy metal and marble, a glass sphere filled with blue fog, a cube of small white spheres and a motion-blurred orange sphere above green boxes](scenes/final.png) **final**: the second book's final scene, everything together |
| ![The Cornell box with a tall mirror block beside a short white box: the mirror is black where it reflects the room's open front, with the short box and the floor reflected in its lower half](scenes/aluminum.png) **aluminum**: a mirror block; the light it throws onto the ceiling stays noisy | ![The Cornell box with a tall white box and a glass sphere that focuses the lamp's light into a bright spot on the floor](scenes/glass.png) **glass**: the third book's final scene, a glass sphere sampled like a light |
| ![A marble sphere on a marble ground in the dark, lit by a spherical lamp above it and a rectangular lamp beside it](scenes/light.png) **light**: light sources | |

## The other programs

- **`RayTracingGPU`** takes the same options and adds the image's height and the camera (`--height`, `--lookfrom`, `--lookat`, `--vfov`, `--defocus-angle`, `--focus-dist`), `--strata`, `--fp64`, `--all-books-kernel` and `--launch-ms`; see [the CUDA renderer](gpu.md#building-and-running).
- **`RayTracingViewer`** shows a scene in a window and refines it while the camera stands still, with `RayTracingGPU`'s options for the scene, the size, the camera, the sampling and the precision; see [the viewer](viewer.md).
- **`ImageCompare A.pfm B.pfm`** tests whether two renders are the same picture up to noise; see [ImageCompare](image-compare.md).
- **`MonteCarlo [experiment...]`** runs the third book's numerical experiments; see [the MonteCarlo program](sampling.md#the-montecarlo-program).
