# The viewer

[RayTracing](../README.md) › The viewer

`RayTracingViewer` shows any of the GPU programs' scenes in a window, the books' 13 and those [beyond the books](quality.md), rendered by [the CUDA renderer](gpu.md) or [OptiX](optix.md), and glTF models. It refines the image for as long as the camera stands still. The mouse and the keys move the camera, open the lens and focus it, and switch scenes, sampling strategies, precision, tone maps, OptiX and the denoised image. The window renders exactly what `RayTracingGPU` renders: the image it converges to is `RayTracingGPU`'s image of the same view with the same samples, bit for bit, and every frame it shows can be rendered again offline, from a recording or from the command that P prints. With OptiX or the denoiser, both hold on the same graphics driver, which compiles OptiX's programs and carries the denoiser's weights.

![A marble bust and a brass vase in RayTracingViewer: the image starts noisy and turns clean when the denoiser comes on, turns noisy while the camera orbits and clean again when it stops; then the Cornell box with the bust refines from a single sample, denoised](viewer.gif)

*Replayed by `RayTracingViewer` from a camera script ([viewer-tour.txt](viewer-tour.txt)), not captured from the screen. `still-life` refines from one sample per pixel, and the denoiser comes on (N). A moving view shows the image itself, and a still view is denoised as its samples double; then `cornell-bust`. The samples per frame are the script's, chosen for the picture, and say nothing of the live window's speed. OptiX's frames and the denoiser's depend on the graphics driver.*

- [Running it](#running-it)
- [Controls](#controls)
- [How it renders](#how-it-renders)
- [Performance](#performance)
- [Rendering a view again](#rendering-a-view-again)
- [Scripts, recordings and scripted input](#scripts-recordings-and-scripted-input)
- [How it is verified](#how-it-is-verified)
- [Remaking the GIF](#remaking-the-gif)
- [Limits](#limits)

## Running it

```
out\build\x64-release\RayTracingViewer.exe final
```

The viewer is built with `RayTracingGPU`, when CMake finds the CUDA Toolkit, and draws its window with SDL3, which CMake downloads the first time it configures ([building](building.md#sdl3-for-the-window)). In Visual Studio it is the `RayTracingViewer.exe` startup item. Like the other programs, it looks for the Earth texture from the current folder up, so start it from the project folder.

The window opens at the scene's aspect ratio, 720 pixels high (smaller if the screen is), at the scene's own view. It takes most of the options of `RayTracingGPU` (all but `--launch-ms`): the scene, `-w` and `--height` for the window's size in pixels, the view options (`--lookfrom`, `--lookat`, `--vfov`, `--defocus-angle`, `--focus-dist`), `-p`, `--fp64`, `--all-books-kernel`, `--quality-kernel`, `--seed`, and the display's `--tonemap` and `--exposure`. `-s` and `--strata` mean what they mean for `RayTracingGPU` with `--strata`: the window refines until every pixel has `-s` samples, in rounds of a K×K grid ([rounds](#rounds-of-strata)). Its own options:

| Option | What it does |
|---|---|
| `-s N` | refine until every pixel has N samples; by default the scene's, rounded up to whole rounds: 512 for the scenes of 500 samples, 1024 for those of 1000 and of 1024 |
| `--strata K` | a round is a K×K grid in each pixel, K from 1 to 46340 (default 8) |
| `-o FILE` | save the image when the window closes, as a `.png` or a `.pfm`; give it several times to save several files. A PNG holds the bytes on screen, through the display; a PFM the linear values |
| `--exit-when-done` | close the window when the image has its samples, after saving `-o` |
| `--script FILE`, `--headless`, `--record FILE`, `--input FILE` | play a camera script, render without a window, record what the window shows, drive the window with scripted input ([scripts](#scripts-recordings-and-scripted-input)) |
| `--frames DIR`, `--frames-hud` | save every frame shown with new samples or a new display as a PNG: the image only, or with `--frames-hud` (in a window) the window as it is, the numbers and the help on it |
| `--stats FILE` | a row of numbers per frame in a CSV file: times, launches, passes, whether the view moved |
| `--refresh-hz N` | give a moving view's frames the GPU time of a frame of a display refreshing N times a second (the frames still come at the display's pace), for measurements that mustn't depend on the display |

`RayTracingViewer --help` prints them all, and the controls.

`--optix` traces the scene with OptiX ([OptiX, meshes and the denoiser](optix.md)), in `float`, under `-p material`, `nee` or `mis`, and `--denoise` starts with N on. A glTF model's file takes a scene's name's place, `RayTracingViewer assets\models\marble_bust_01\marble_bust_01_1k.gltf`, on a gray ground in the photo studio's light; a scene with triangle meshes, a model's file or `still-life`, `busts` or `cornell-bust`, starts OptiX by itself. The refusals are `RayTracingGPU`'s, in its words and order, before the window and CUDA.

`RayTracingViewer studio` opens one of the scenes beyond the books with its own strategy and display, `-p mis` and AgX, as `RayTracingGPU studio` renders it. The quality kernel that renders them, and the tone maps beyond the books' bytes, compute in `float` only ([why a third kernel](quality.md#why-a-third-kernel)): the window refuses them with `--fp64` before it opens, in one line, as `RayTracingGPU` does.

## Controls

| Input | What it does |
|---|---|
| left drag | **orbit**: the camera turns about the point it looks at, a quarter of a degree per pixel, and stops a degree short of looking straight up or down |
| right drag | **look around**: the camera stays where it is and turns, as a head does; the cursor hides and stays in the window until the button is released (or Esc) |
| middle drag, Shift+left drag | **pan**: the camera and the point it looks at move together, so the point under the cursor follows the cursor |
| wheel | **dolly**: the camera moves toward the point it looks at, 15% of the distance a notch, or back; the focus distance moves with it, so what was in focus stays in focus |
| Ctrl+wheel, `-` and `=` | **zoom**: the field of view times 0.9 a notch, or divided by 0.9, from 1 to 160 degrees |
| W A S D, Q E | **fly** forward, left, back and right, down and up, at half the distance to the point looked at per second; Shift 4 times faster, Ctrl 4 times slower. The keys go by their place, where W A S D are on a US keyboard, whatever the layout |
| `[` and `]` | **the lens** narrower or wider: the defocus angle divided or multiplied by 1.25. A closed lens opens at a quarter of a degree, a step back closes it, and it opens no wider than 30 degrees |
| Shift+wheel | **focus** farther or nearer, 5% a notch, out to ten times the distance to the point looked at |
| double-click | **focus on what is under the cursor** ([how](#click-to-focus)) |
| Tab, Shift+Tab | the next or the previous scene: the books' 13, then the four beyond them, `cornell-ggx`, `veach`, `studio` and `sunlit`, and, in a session with OptiX (`--optix`, O, or a scene with triangle meshes), `still-life`, `busts` and `cornell-bust` after `sunlit`. Each opens at its own view (the command line's view for the command line's scene) and with its own samples to refine to (or `-s`), with the same precision, kernel, seed, strata and display, and with the strategy that `-p` or M named, where the scene takes it ([which scenes, which strategy](#strategies-and-scenes)) |
| M, Shift+M | the next or the previous **sampling strategy**, in a new image at the same view, refining to the same samples: the scene's own, then `uniform`, `material`, `lights`, `mixture`, `nee` and `mis`, skipping those the scene can't take ([which](#strategies-and-scenes)) |
| F2 | `float` or `double` (`--fp64`; float32 and float64 on screen), at the same view, refining to the same samples. The quality kernel and the tone maps beyond `books` compute in `float` only: under them F2 stays and says why ([float only](#strategies-and-scenes)) |
| O | **OptiX** on or off for the scene shown, in a new session at the same view, as F2 switches the precision. A scene with triangle meshes always uses it, and says so; F2 under OptiX says to turn O off first |
| N | **the denoised image** on or off ([denoising in the window](#denoising-in-the-window)) |
| T, Shift+T | the next or the previous **tone map**: `books`, `none`, `agx`, `neutral` ([tone maps](#tone-maps-and-exposure)) |
| `,` and `.` | the **exposure** half a stop down or up, within ±24 EV; not with `books` |
| `0` | the exposure back to 0 EV |
| Space | pause: no refining, though a move still shows its new view |
| Enter, Shift+Enter | twice or half the samples to refine to, in whole rounds of strata |
| R | back to the scene's starting view |
| P | save the image as `<scene>-view-NNN.png` and `.pfm` in the current folder, and print the `RayTracingGPU` command that renders it ([rendering a view again](#rendering-a-view-again)); with N on, also `<scene>-view-NNN-denoised.png` |
| C | copy that command to the clipboard |
| I | the numbers on screen, on or off |
| H, F1 | the controls over the image |
| Esc | close the help; during a right drag, let the cursor go; otherwise close the window (saving `-o`) |

The numbers on screen are three lines at the top: the scene, the image's size, the precision and the strata, and the strategy when the quality kernel renders (`mis`); the samples so far and the samples to refine to (under a script, the frame's samples alone), with the rate in millions of samples per second and the time from one frame to the next, or "paused", or "done"; and the field of view, the lens's angle and the focus distance, and the tone map and its exposure when the display isn't the books' bytes (`agx +0.5 EV`). The books' scenes, under their own kernels and bytes, show the numbers they always showed. Under OptiX the first line ends in `optix`, and with N on the second says what the image shows ("denoised at 64 of 100 passes"). A line at the bottom says for two seconds what a key did, or why it couldn't. The console gets a line for every scene shown, with its strategy, its precision (and the quality kernel, when it renders) and its view as `RayTracingGPU` options; a `Display:` line for a display other than the books' bytes, and at every change of display; a `Note:` line for every scene or strategy a key skipped, with the reason; and a `Done:` line, with the time the view's frames took, the first time an image has its samples after the window opens, after Tab, Shift+Tab, M, Shift+M or F2, and after Enter or Shift+Enter sets a target the image hasn't reached.

If the GPU can't hold the scene, the strategy or the precision a key switches to, the window shows the scene it came from again, its image started over, and says why; if not even that scene can come back, the program ends with the error.

### Strategies and scenes

M steps through the scene's own strategy, then the GPU programs' list: the books' `uniform`, `material`, `lights` and `mixture` ([sampling](sampling.md#four-strategies)), and the quality kernel's `nee` and `mis`, light samples at every bounce and their weighing against the material's own samples ([next-event estimation and multiple importance sampling](quality.md#next-event-estimation-and-multiple-importance-sampling)). Each is a new image at the same view. A strategy the scene can't take as asked is skipped: named in the note at the bottom, and on the console in a `Note:` line with the reason. It is skipped when the scene refuses it: a books' strategy other than `material` on a scene beyond the books or under `--quality-kernel`, and `nee` and `mis` in `double` or under `--all-books-kernel`. It is skipped too when the scene would only fall back from it to the material's own sampling, as `RayTracingGPU` falls back: `lights` and `mixture` without a lights list, `nee` and `mis` without a lamp or an environment map (the console says the scene "renders it as material"). In `double`, when `nee` or `mis` was refused (on a scene with a lamp, where they would need the quality kernel), the note adds "nee and mis in float32 only: F2 first".

The scene's own strategy and the same strategy named are two entries of M's cycle, because a strategy named, by M or by `-p`, goes along with Tab, as a display chosen does, and the scene's own doesn't. Tab goes through the books' 13 scenes and then the four beyond them, `cornell-ggx`, `veach`, `studio` and `sunlit`, in the order of the help; the test scenes aren't in the cycle, and from one of them Tab goes to the first scene and Shift+Tab to the last. A scene that refuses the strategy named renders with its own instead, for that scene only, and says so ("sampling: mis, the scene's own (cornell-ggx can't take uniform)"); the next scene is asked for the named strategy again. `nee` and `mis` named go along to the books' scenes lit by their sky alone, which render them as `material`, as `RayTracingGPU` does: their gradient sky isn't a light ([known limits](quality.md#known-limits)). Tab skips a scene it can't show at all, with a `Note:` line on the console for each and the first reason on screen: the scenes beyond the books in `double` ("float32 only: F2 first") and under `--all-books-kernel`, as only the quality kernel renders them; and, under `--quality-kernel` with no strategy named, a scene whose own strategy is a books' one other than `material`.

M and Tab build the scene they could go to and check it before the session changes, so a key the window refuses leaves the image as it was. F2 stays in `float` wherever the quality kernel renders, and says why, with the way out: "the quality kernel works in float32 only: M to a books' strategy first" on a book scene under `nee` or `mis`, "studio renders with the quality kernel, in float32 only" on a scene beyond the books, and "--quality-kernel works in float32 only" under that option. Under a tone map other than `books` it says "agx works in float32 only: T to books first".

### Tone maps and exposure

The window shows its image through the same display transforms as `RayTracingGPU`'s PNGs ([tone mapping](quality.md#tone-mapping)): `books`, the books' square root, clamped, which their scenes show by default; `none`, the sRGB curve, clipped; `agx`, which the scenes beyond the books show by default, each at an exposure of its own; and `neutral`, Khronos PBR Neutral. T and Shift+T step through them in that order, from the display shown: on `studio`, T goes from its own `agx` to `neutral`. `,` and `.` move the exposure half a stop (0.5 EV) down or up, within ±24 EV, and `0` brings it back to 0 EV; like `-`, `=`, `[` and `]`, they also work by their place on a US keyboard. The books' bytes have no exposure: under `books` the three keys change nothing, and say so ("exposure applies to a tone map: T picks one").

`--tonemap` and `--exposure` choose a display from the start, refused as `RayTracingGPU` refuses them. A display chosen, by an option, a key or a script, holds for every scene Tab goes to, as the window's other settings do; what it leaves open is each scene's own: an exposure never set is the scene's, −1 EV in `veach` and −0.5 EV in `sunlit`, and under `books` an exposure set waits for a tone map. The tone maps beyond `books` compute in `float` only: in `double`, T says so ("none works in float32 only: F2 first", or `neutral` for Shift+T).

A display changes how the samples look, not the samples, and keeps the session: on an image that is done or paused, the window [shows the same samples again](#the-image-on-the-screen) through the new display. What the window saves follows it: P and `-o` write a PNG of the bytes on screen, while the PFM stays linear, and the command P prints ([rendering a view again](#rendering-a-view-again)) carries the display, so it renders the window's bytes too.

### OptiX and models in the window

O traces the scene shown with OptiX, in a new session at the same view; O off goes back to the command line's kernel (`--quality-kernel`, `--all-books-kernel`, or the scene's own). The window keeps one OptiX runtime, started at the first need and shared by every session, so Tab, M, F2 and O never compile OptiX's programs twice; before the first OptiX session the console and the screen say "Note: compiling OptiX's programs (once per driver)...", as the window stands still during the compile. Without OptiX in the session Tab keeps its old cycle, the books' scenes and the four beyond them; with OptiX it adds `still-life`, `busts` and `cornell-bust` after `sunlit`, and keeps OptiX for the next scene. Sessions in `double` never trace with OptiX, so they never see those three. A model's file is outside the cycle. If OptiX can't run, it is tried once, and each key that needs it gives that reason.

### Denoising in the window

With N on, a still view's frames show its image denoised by OptiX's AI denoiser ([what it is, and what it removes](optix.md#the-denoiser)): at its first still frame, whenever its complete passes have doubled since its last denoise (1, 2, 4, 8...), and once more when it is done or paused. A view still counted as moving shows the image itself, and a denoise due once it stands still comes as a frame of its own. The frames between two denoises show the last denoised bytes, kept on the host, and the numbers say so ("denoised at 64 of 100 passes"). T, the exposure keys and a script's `tonemap` and `exposure` re-tone the kept denoised floats and never denoise, however far the image has refined since. N in a CUDA session starts OptiX if it isn't running and gives the session the window's runtime: there is no new session, and the image goes on. N off frees what the denoise kept on the GPU. N in `double` says to press F2 first, and F2 under N to turn N off first. A denoise that fails turns N off with an `Error:` line.

The denoiser is biased, and is for the eye: the window's PFMs and every statistic use the image without it. P with N on also saves `<scene>-view-NNN-denoised.png`, the bytes on screen, and prints the `RayTracingGPU` command with `--denoise`. `-o` under N writes each output's `-denoised` twin, as `RayTracingGPU --denoise` does.

### Click to focus

A double-click asks the CPU's copy of the scene, the very objects the GPU renders, where the ray through the pixel's center first hits something, and sets the focus distance to that point's distance along the view. In a scene with triangle meshes, the CPU has no triangles to ask, so the pick goes through the image's own tracer, OptiX, and the console gives the distance along the ray too ("(OptiX's pick)"). The pick uses the camera's ray at a focus distance of 1, whose direction doesn't depend on the focus, so the same click gives the same focus, bit for bit (through OptiX's pick, on the same graphics driver), and a second click doesn't start the image over. Smoke and fog are seen through: they are hit at a random depth, so the pick goes on to the surface behind them, and takes the volume's own point only when there is nothing behind. That matters in the second book's final scene, which stands inside a mist that fills its whole world. The console prints the focus, or that there was nothing there.

## How it renders

### A session that keeps the scene on the GPU

The window builds a scene with the same code as `RayTracingGPU` (`gpu_scene.ixx`), and a rendering session (`gpu::session` in `renderer.h`) copies it to the GPU once and keeps it there for as long as the window shows that scene. A camera move gives the session only a new camera: the image starts over with no samples, and nothing else is copied. Each frame, the window asks the session for a slice of GPU time, and the session launches the kernel again and again, each launch adding whole passes (one more sample in every pixel) or, when one pass takes longer than a launch should, a band of rows of the next one. `RayTracingGPU` renders through the same session, with its one view, so both programs run the same host code, launch planner and accumulation.

Tab, Shift+Tab, M, Shift+M and F2 build a new session; the display's keys keep the session and its samples. The scene is built on the CPU first and checked against what the kernels can render, and only then is the old session destroyed, before the new one is created, so the GPU never holds two scenes, and a switch the window refuses leaves the image as it was. Building a scene draws random numbers (the first book's cover places its small spheres at random, the marble shuffles its noise tables), and every build starts them over, so every scene the window shows is the one `RayTracingGPU` renders.

### Rounds of strata

The books' camera spreads a pixel's samples over a k×k grid, one random point in each cell, so `-s 500` renders 484 samples, 22×22. That needs the number of samples before the render starts, and a window doesn't know how many it will render before the camera moves. So the session renders rounds: grid after grid, every round a complete k×k grid of cells with random numbers of its own (round r draws those of seed + r·2³², so no two rounds and no two seeds share them). Round 0 is the books' grid, and N samples are the same N samples however many frames and launches they took. `RayTracingGPU --strata k -s N` renders the same rounds, which is how a window's image is rendered again offline: `--strata 22 -s 484` is `RayTracingGPU`'s own image of the scenes of 500 samples, `--strata 32 -s 1024` of those of 1000.

The window's default is k = 8, rounds of 64 samples: the smallest grid whose rounds cost at most 5% of GPU time against one round of 32×32, measured on five scenes. A launch never spans two rounds, so a small grid makes short launches, and each launch loses a little time at its end, when its last threads finish while the rest of the GPU idles: rounds of 4×4 cost up to 9.7%, of 8×8 3.8%, of 16×16 1.6%. Rendered in 16 rounds of 8×8, the second and third books' 11 scenes, in `float` and in `double`, were consistent with the CPU's renders of 1024 samples in 43 of 44 statistical comparisons, and the 44th passed with three other seeds.

A round that isn't complete has sampled only the top rows of cells of every pixel, and the left cells of its last row, so its samples lean up and to the left. After r complete rows of cells, the samples sit r/(2k) − 1/2 of a pixel from the pixel's center vertically, and m cells into the next row, −m(k−m)/(2k·p) of a pixel horizontally, for p samples in the round; R complete rounds before it dilute the offset by p/(k²R + p). At k = 8 that is −7/16 of a pixel after 8 samples, −1/4 after 32 and 0 after 64. A window's finished image always ends on a whole round: the default targets are rounded up, and Enter and Shift+Enter double and halve whole rounds. Only `-s N` given explicitly stops where it is told. A moving frame is a partial round in most scenes, from a few samples of the 64 to most of them, and the offset doesn't show in the noise of a moving frame.

### The image on the screen

Every frame shows the session's image converted to bytes on the GPU (`display.cu`): each channel's sum divided by the samples, its square root for gamma 2, clamped to [0, 0.999], times 256 and rounded down. That is what `image::save_png` computes on the CPU, so the window shows the very bytes the PNG holds; `display.cu` is compiled without fused multiply-adds, as the CPU computes. The rows that a pass in progress has reached hold one sample more than the others and are divided by their own number of samples, so the image never shows a brighter or darker band. The bytes, 4 per pixel, are copied to the host and uploaded into an SDL texture, which is drawn one pixel per pixel from the top left.

Under a tone map other than `books`, a kernel of its own (`tonemap.cu`) makes the bytes instead: the light times 2^EV, the tone curve (AgX's gives display values itself; `none` and `neutral` then take the sRGB encoding) and a dither, from the same sums and with the same rows of a pass in progress. They are the bytes of `RayTracingGPU`'s PNG with the same display ([tone mapping](quality.md#tone-mapping)). A display key changes only this conversion. On an image that refines, the next frame simply takes the new display. On an image that is done or paused, the window shows the same samples once more through the new display, as a frame that adds none, and goes back to waiting; keys pressed together make one frame. `--frames` saves that frame, and `--record` writes it as a frame of the same passes, which a replay renders by rendering nothing.

### Frames: refining and moving

- **Standing still,** each frame gets 50 ms of GPU time, the length `RayTracingGPU` plans its launches for, and then shows the image: about 20 frames a second while the image refines, with the GPU busy nearly all the time.
- **Moving,** each frame gets three quarters of the time between two presented frames: SDL presents with vsync, every n-th refresh, n the refresh rate over 60 rounded down (a rate just under a multiple of 60 counting as that multiple), from 1 to 4. A view counts as moving for 150 ms after its last change, so a camera moved step by step doesn't drop to 20 frames a second between the steps. A new view always gets one complete pass before it is shown, however long that takes, so the window never shows rows of the old view.
- **All the mouse's motions of one frame make one new view,** so a fast mouse doesn't start the image over for each of its events.
- **While a window's edge is dragged,** Windows runs a loop of its own for the drag, and a program's main loop stands still until it ends. The viewer uses SDL's main callbacks: SDL calls the viewer's frame from inside that loop too, so the image keeps refining at the size the window has at each moment.
- **When the image is done,** or paused, or the window minimized, the program waits for the next event and uses neither the CPU nor the GPU. On an image that is done or paused, a display key wakes it for one frame, the same samples through the new display.

### Waits that sleep

The host waits for the GPU at the end of every launch, and by default the CUDA runtime waits by spinning: a CPU core asks the GPU again and again whether it is done, at 100% of the core. `RayTracingGPU` keeps that, as it renders and exits and a spinning wait reacts at once. The window renders for as long as it is open, so its waits sleep until the GPU signals: the device's flag (`cudaDeviceScheduleBlockingSync`) makes copies and device-wide waits sleep, and the session's events, created with `cudaEventBlockingSync`, make the waits for launches sleep. Both are needed: an event is waited for by spinning unless it was created to sleep, whatever the device's flag. So the window takes about 3% of one CPU core while it renders, and nothing when the image is done.

## Performance

On the same desktop machine as [the performance table](../README.md#performance), in `float`, at 1280×720, with `--refresh-hz 60`:

| Scene | Samples per pixel in a moving frame | The scene's samples | Time to them in the window | `RayTracingGPU`'s render time |
|---|---|---|---|---|
| `materials` | 53 | 512 | 0.17 s | 0.14 s |
| `weekend` | 12 | 512 | 0.46 s | 0.40 s |
| `bouncing` | 7 | 512 | 0.63 s | 0.58 s |
| `checker` | 44 | 512 | 0.17 s | 0.12 s |
| `earth` | 256 | 512 | 0.03 s | 0.03 s |
| `perlin` | 34 | 512 | 0.22 s | 0.18 s |
| `quads` | 192 | 512 | 0.06 s | 0.05 s |
| `light` | 48 | 1024 | 0.26 s | 0.23 s |
| `cornell` | 63 | 1024 | 0.41 s | 0.37 s |
| `smoke` | 52 | 1024 | 0.43 s | 0.38 s |
| `final` | 3 | 1024 | 2.29 s | 2.13 s |
| `aluminum` | 62 | 1024 | 0.46 s | 0.41 s |
| `glass` | 29 | 1024 | 0.95 s | 0.87 s |

*A moving frame: the median of an orbit dragged for 300 frames, every frame a new view with the GPU time of a frame of 1/60 s; the samples vary with what the view sees (in `cornell`, from 30 to 114). The scene's samples are its 500 or 1000 rounded up to rounds of 8×8, rendered with the camera standing still, in the window from its first frame (`--exit-when-done`), by `RayTracingGPU` with `--strata 8 -s` the same, its render time as it prints it.*

- **A still view refines at 90 to 99% of `RayTracingGPU`'s speed:** the window's samples per second of wall time, over its still frames, against `RayTracingGPU`'s per second of GPU time. The rest is the frames: a still frame spends about 1.1 ms at this size on the conversion to bytes, the copy, the upload and the present, about 20 times a second. (`earth` and `quads` are done within the 150 ms that count as moving, too soon to measure.) The times above add the first frames, whose budgets are those of a moving view.
- **The CPU:** about 3% of one core while the image refines, nothing once it is done; `RayTracingGPU`'s spinning waits keep a core busy for the whole render.
- **Launches stay short:** the longest launch in any of the window's checks took 85 ms, in `double`, in the second book's final scene at 3840×2160, far below the 2 seconds after which Windows resets a card.

What could make the window faster, left for later:

- **Interop.** CUDA can write into a Direct3D texture directly, without the copy through the host and the upload, most of the 1.1 ms a still frame spends. It needs the CUDA device to be the adapter that drives the display, which a computer with two GPUs may not have, and it would be a second display path to verify.
- **Two frames in flight.** The window waits for each frame's GPU work before it presents it; a queue of two frames would overlap the copy and the upload with the next frame's rendering.
- **Two streams.** A moving frame's launches are short, and each loses a little time at its end; rows rendered in two stripes on two CUDA streams would fill those gaps.

## Rendering a view again

P saves the image on screen, its pass in progress finished first, as `<scene>-view-NNN.png` and `.pfm` in the current folder (NNN the first number no file has yet, so nothing is overwritten), and prints the command that renders it again. In `RayTracingViewer final -w 960 --height 540`, after a double-click on the glass sphere in front and `]` four times:

```
Saved C:\...\final-view-001.png and .pfm (694 samples per pixel). The same image:
RayTracingGPU final -w 960 --height 540 --lookfrom "478,278,-600" --lookat "278,278,0" --vfov 40 --defocus-angle 0.48828125 --focus-dist 631.6939630053682 -s 694 --strata 8 --seed 0 -o final-view-001-gpu.pfm
```

`RayTracingGPU` saves the very same PFM, bit for bit (with OptiX or the denoiser, on the same graphics driver). Every number is printed with the digits it needs to come back as the same `double`, and the points are quoted because Windows PowerShell 5.1 rounds an unquoted list of numbers like `1,2,3` to 15 digits. The height is given exactly: `--height` sets the aspect ratio so that the camera makes exactly that many rows. C copies the same command to the clipboard.

The command names `-p` when a strategy was named (by `-p` or by M), `--fp64`, `--all-books-kernel`, `--quality-kernel` or `--optix` when they were asked for, `--denoise` under N, a model's file quoted when it needs quotes, and `--tonemap` and `--exposure` wherever the display shown or the scene's own isn't the books' bytes, so it renders the window's image whatever the defaults. Under a tone map, P's PNG holds the bytes on screen, and the command saves a PNG of its own too (`-o <scene>-view-NNN-gpu.png`): `RayTracingGPU` writes the same bytes.

## Scripts, recordings and scripted input

**A camera script** says what to render, frame by frame. `--script FILE` plays it in the window, which then takes no keys but Esc, and closes; `--headless --script FILE` renders it without a window, as fast as the GPU goes. One command a line, `#` starts a comment, and settings stay until changed. The GIF above is this script, [viewer-tour.txt](viewer-tour.txt):

```
scene still-life
size 960 540
strata 4
seed 0
hold 10 +1                     # a sample per pixel more each frame: the image itself, noisy
denoise on                     # the window's N
hold 1 +6                      # 16 samples, denoised at once
hold 12 +4                     # refining to 64: denoised again at the doublings, 32 and 64
orbit 16 +8 yaw 20             # a moving view shows the image itself, every frame a new view
hold 1 +8                      # still again: denoised at its first frame, 16 samples
hold 12 +4                     # and at 32 and 64
scene cornell-bust             # Tab, in a window
hold 1 +1                      # denoised at every doubling, from a single sample
hold 1 +1
hold 1 +2
hold 1 +4
hold 1 +8
hold 1 +16
hold 1 +32
hold 1 +64
hold 1 +128                    # 256 samples: the frame the GIF ends on
```

`hold F +N` shows F frames of the view as it is, each N samples more than the one before. The moves are `orbit F +N yaw D [pitch D]`, `look F +N yaw D [pitch D]`, `pan F +N dx X [dy Y]`, `dolly F +N factor X` and `zoom F +N factor X` (or `notches Y` for either, the wheel's own steps): F frames, eased in and out, every frame a new view with N samples. They call the same functions as the mouse and the keys (`viewer_camera.ixx`), so a script moves the camera exactly as the hand does. `camera` sets any of `lookfrom "X,Y,Z"`, `lookat "X,Y,Z"`, `vfov`, `defocus-angle` and `focus-dist`; `focus-at X Y` focuses as a double-click on that pixel does. The other settings are `precision float|double`, `sampling NAME|scene` (a strategy's name, or the scene's own), `kernel auto|all-books|quality|optix`, `denoise on|off`, and the display: `tonemap NAME` (`books`, `none`, `agx` or `neutral`) and `exposure EV` (from −24 to 24). The display doesn't change the image, so a frame after a `tonemap` line may show the samples of the frame before through it. Without these two lines each scene shows its own display; once one is given, it holds for every scene, as the window's keys hold, and under `books` an exposure waits for a tone map. A script that asks for a tone map other than `books` in `double` is refused, before that frame renders. `scene` takes a model's file, quoted (`scene "assets\models\marble_bust_01\marble_bust_01_1k.gltf"`). Under `denoise on`, a hold's frames are denoised as a still view is, at its first frame and at the doublings of its passes (a script has no "done", so a hold's last frame shows the last doubling's denoise unless it lands on one); a move's frames are the image itself. A script ends with a frame: lines after its last `hold`, move or `frame` would render nothing, and are an error.

**A recording.** `--record FILE` writes every frame the window shows with new samples or a new display as a `frame` line: the view, its size, how many passes and rows of the next pass it had, and a number for the view. The settings come first, and again whenever they change; the display's `tonemap` and `exposure` lines come before the frame they apply to (a recording of the books' scenes that never leaves their bytes has none, and stays as it was). A frame shown again through a new display is a `frame` line with the same passes and the same number. A recording writes `denoise on|off` only once N was used, or with `--denoise` or a script's `denoise` line, and marks each frame it denoised with a last word, `denoised`; a replay denoises exactly the marked frames, whatever the clock decided live. `--headless --script FILE` renders a recording again, frame by frame, bit for bit (with OptiX or the denoiser, on the same graphics driver): a session of the live window, cut into frames by the moments a user moved the mouse, gives the same images as a session rendering straight to each frame's samples.

**Scripted input.** `--input FILE` drives a live window with keyboard, mouse and window events, each before a frame of the window's loop (`at F`; the loop runs fast once the image is done) or a number of milliseconds after the window opened (`after MS`):

```
at 10 mouse-down left 200 200
at 11 mouse-move 202 200 2 0
at 12 mouse-up left 202 200
after 2000 key Tab
after 3000 wheel 1 shift
after 4000 quit
```

Events: `key`, `key-down` and `key-up` with SDL's names of the keys; `mouse-down BUTTON X Y [CLICKS]` and `mouse-up BUTTON X Y [CLICKS]` (`left`, `middle` or `right`; 2 clicks for a double-click); `mouse-move X Y DX DY`; `wheel DY [shift|ctrl]`; `resize W H`; `device-reset`; `quit`. While a file drives the window, the real keyboard and mouse are ignored. That is how the window's input is tested: the same events, then the recording replayed without a window. Four such files are in the repository: [`tests/window-tone.input`](../tests/window-tone.input) presses the display's keys on the Cornell box, paused, refining and done; [`tests/window-strategies.input`](../tests/window-strategies.input) takes M, Tab, F2 and Shift+Tab through the strategies and the scenes beyond the books, in `float` and in `double`; [`tests/window-optix.input`](../tests/window-optix.input) takes O, Tab, a double-click and P through the scenes with meshes; and [`tests/window-denoise.input`](../tests/window-denoise.input) takes N through a pause, an orbit, the re-tones and P. The first lines of each give the two commands that record it and replay it.

**Numbers per frame.** `--stats FILE` writes one row a frame: `frame, wall_ms, dt_ms, gpu_ms, launches, longest_launch_ms, passes, rows, new_samples, presented, moving, generation, width, height, strata, time_ms`. Between the rows, comment lines say when a model was read (`# model read: FILE, N triangles, M images, T ms`) and when a frame was denoised (`# denoised: frame F, P passes and R rows, generation G, T ms`). With `--refresh-hz 60`, a moving view gets the budget of a frame of 1/60 s whatever the display, so measurements compare.

## How it is verified

The window was verified as the GPU renderer was, bit for bit wherever the bits are defined: as a whole at the third GPU milestone (`gpu-3`), and then in what the fourth (`gpu-4`) and the fifth (`gpu-5`) added to it, the last two items below.

- **`RayTracingGPU` unchanged.** It renders through the session now, and 42 configurations (every scene in `float` and `double`, each sampling strategy, both kernels) still give the PFMs they gave before the session existed, byte for byte. The kernels' machine code is the same, instruction for instruction; the launches of a render, planned with synthetic times, are the same list; and its render times are within 1.3% of what they were.
- **Rounds.** `--strata K -s K²` gives the image without `--strata` in 108 pairs (13 scenes, four grids, `float` and `double`, and the general kernel). Renders in rounds give the same bytes with launches of 1, 16, 50 and 200 ms. The round seeds were checked exactly, and rounds were compared with the CPU's renders statistically, as above.
- **The window against `RayTracingGPU`.** 60 headless renders (13 scenes, strata 1 to 8, float and double, odd sizes from 1×1, the four strategies, both kernels) give the PFM `RayTracingGPU` gives with the same options, the same ray count, and a printed command that reproduces it; `weekend` at 1920×1080 with `--strata 22 -s 484` gives the image of the GPU milestone before the window. 16 renders through the real window give the same PFMs and PNGs, and the last frame read back from the window equals `RayTracingGPU`'s PNG. Frames that stop inside a pass show the right rows of the two passes.
- **The live window.** Camera tours, orbits, the lens, double-clicks, Tab through the books' 13 scenes and F2 there and back, driven by scripted input: each recording, replayed without a window, gives the window's frames, byte for byte, and the first frame of every view equals a fresh `RayTracingGPU` render. P's printed commands render the saved PFMs. 3,200 random input events (drags, wheels, keys, resizes, scene and precision switches) give valid recordings that replay to the same frames (all of them for 1,200 of the events, 50 sampled frames for the other 2,000); 200 mutated scripts end in a render or a clean error, never a crash or a hang; and a simulated reset of the window's graphics device in the middle of a render changes no frame.
- **Robustness.** 104 scene switches without growing the GPU's memory by a byte; a session that fails to build, injected into a test build, brings the old scene back, or ends the program with the error; CUDA's memory checker on the session and the display; Debug and Release giving the same frames; a storm of resizes of the window's edge; an impossible image size ending in a clean error.
- **Speed.** A still view at 90% or more of `RayTracingGPU`'s speed; moving frames at the display's pace in `weekend`, `materials` and `cornell`; the first frame of a new view within 61 ms (the 95th percentile of 50) of the wheel step that asked for it; the CPU nearly idle.
- **Strategies, the scenes beyond the books and tone maps.** The two input files of [scripted input](#scripts-recordings-and-scripted-input), recorded and replayed without a window, give the same frames, decoded. `tests/window-tone.input` (the Cornell box at 320×180, 64 samples) gives 7, among them a paused frame of 1 pass shown again through `none` and one frame for two keys pressed together. `tests/window-strategies.input` (320×180, 32 samples) gives 104, its console showing M's cycle, a named `uniform` taken by Tab to `glass` and refused by `cornell-ggx`, F2 refused under the quality kernel, and Shift+Tab in `double` past the four scenes beyond the books. The views P saved under `material`, `nee` and `mis`, and under AgX, are byte for byte what `RayTracingGPU` renders from the printed commands. With a tone map the window equals `RayTracingGPU`: the Cornell box at 160×90, 16 samples, AgX at +0.5 EV, gives the same PNG and PFM, and the last frame equals the PNG; in the quality kernel, and under `-p mis`, it gives the same PFMs. A done window shows no frame until a display key, then exactly one. [`tests/regress-tone.txt`](../tests/regress-tone.txt), a headless script through every display, is compared with the build before at every change: its frames, its recording and what the window prints. The strategy on the numbers' first line was checked by reading the code, not in a saved frame.
- **OptiX, models and the denoiser.** The window equals `RayTracingGPU`, every float of the PFM, on `cornell --optix -p mis` and `still-life`, and under N on the plain and denoised PFMs, on the same graphics driver. [`tests/regress-optix-viewer.txt`](../tests/regress-optix-viewer.txt) and [`tests/regress-denoise-viewer.txt`](../tests/regress-denoise-viewer.txt), headless scripts of 40 and 47 frames, are compared with the build before under a stamp of the driver and the programs; the second denoises exactly its 17 marked and scheduled frames, and none at a change of display. `tests/window-optix.input` and `tests/window-denoise.input`, live in a window, replay without a window to the same frames, and P's printed commands render the same PFMs. Five sessions of each kind, made and destroyed in turn, lose none of the GPU's memory.

## Remaking the GIF

```
powershell -ExecutionPolicy Bypass -File scripts\make-gif.ps1
```

After an x64-release build, the script renders the tour with `RayTracingViewer --headless --script docs\viewer-tour.txt --frames <folder>`, which saves every frame as a PNG; the script then holds the last frame for 2 seconds and writes `docs\viewer.gif` with [gifski](https://gif.ski) (`scoop install gifski`; an external program, run but not part of the project) at 12.5 frames a second, 960×540. The same frames and gifski 1.34.0 make the same file, byte for byte; the frames of OptiX and the denoiser depend on the graphics driver. `-Frames <folder>` encodes frames rendered before instead. gifski runs at quality 100, since lower qualities leave blotches of the frame before, and the GIF stays under 8 MiB.

## Limits

- **`double` is a check.** With `--fp64` or F2 the window renders the same views 12 to 40 times slower, as `RayTracingGPU` does. Each new view gets at least one whole pass, and in the heavier scenes that one pass takes longer than a frame of the display, so the frames come slower. What is beyond the books computes in `float` only ([known limits](quality.md#known-limits)): in `double` the window keeps the books' kernels and bytes, Tab skips the scenes beyond the books, M skips `nee` and `mis`, and T stays at `books`.
- **Moving frames are partial rounds,** noisy and leaning by a fraction of a pixel ([rounds](#rounds-of-strata)); the window has no lower resolution for motion, and its denoiser works on a still view only (no temporal denoising).
- **The copy through the host:** the conversion, the copy and the upload take most of a still frame's 1.1 ms at 1280×720 ([interop](#performance) would remove most of it).
- **One GPU:** the first CUDA device renders, and SDL presents on whatever adapter drives the display.
