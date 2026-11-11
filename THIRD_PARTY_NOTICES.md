# Third-party notices

The project's own code is under the [MIT license](LICENSE). This file lists the code and data it holds or adapts from others, under their own licenses.

## Code in this repository

| What | Where | License |
|---|---|---|
| [stb_image and stb_image_write](https://github.com/nothings/stb), by Sean Barrett | `external/stb` | MIT or public domain, at your choice; the notice is at the end of each file |
| [cgltf](https://github.com/jkuhlmann/cgltf), by Johannes Kuhlmann | `external/cgltf` | MIT; the notice is at the end of the file |
| PCG32, from the [minimal C implementation](https://www.pcg-random.org/download.html) of PCG by Melissa O'Neill (`pcg_basic.c`), ported to CUDA | `src/gpu/random.cuh` | Apache License 2.0 (below) |
| Minimal AgX by iolite (© 2024 Missing Deadlines) and [linlin's closed-form AgX curve](https://github.com/bWFuanVzYWth/AgX) (© 2024 linlin) | `src/gpu/tone_curves.cuh` | MIT; the notice is in the file |

### PCG32

```
PCG Random Number Generation for C.

Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

Changes: `pcg32_srandom_r` and `pcg32_random_r` rewritten as a C++ class for CUDA device code, with the rotation written as a right shift and a masked left shift; the generator's arithmetic is unchanged.

## Code downloaded when the project is built

CMake downloads these when it configures; they aren't in the repository.

| What | License |
|---|---|
| [SDL3](https://libsdl.org) | zlib |
| [Dear ImGui](https://github.com/ocornut/imgui) | MIT |
| [NVIDIA's OptiX headers](https://github.com/NVIDIA/optix-dev) | NVIDIA's licenses, as their files say; the programs use the OptiX runtime in NVIDIA's graphics driver |

A binary built from this project links SDL3 and Dear ImGui statically; whoever distributes one should include their notices with it.

## Data

| What | Where | License |
|---|---|---|
| The Earth map, from the [books' repository](https://github.com/RayTracing/raytracing.github.io) | `assets/earthmap.jpg` | CC0 |
| `studio_small_09` and `kloofendal_48d_partly_cloudy_puresky`, HDR environment maps from [Poly Haven](https://polyhaven.com) | `assets` | CC0 |
| `marble_bust_01` and `brass_vase_01`, glTF models from [Poly Haven](https://polyhaven.com) | `assets/models` | CC0 |

The scenes and algorithms of the CPU renderer are those of the books by Peter Shirley, Trevor David Black and Steve Hollasch; the books' code is CC0. Published methods are credited in [docs/architecture.md](docs/architecture.md#third-party-code-and-credits) and in the code's comments.
