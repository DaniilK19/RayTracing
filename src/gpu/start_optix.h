// OptiX for the programs (main_gpu.cpp, main_viewer.cpp, gpu_scene, GpuTests). gpu_optix implements it in
// src/gpu/optix/scene.cu, beside the concrete optix_runtime, on runtime.cpp's optix_core, or absent.cpp in a build
// without OptiX's headers.
//
// Plain C++, no OptiX or CUDA type: a file that also has `import std;` includes it BEFORE that line, as renderer.h.
#pragma once
#include <memory>
#include <string>

namespace gpu
{
class optix_runtime;

// Whether CMake found OptiX's headers: when false, --optix, --denoise and the mesh scenes refuse before CUDA starts
bool optix_built();

// Loads OptiX from the graphics driver and creates its context on the CUDA device query_device() started; the
// programs are compiled at the first upload, the denoiser made at the first denoise. Throws std::runtime_error saying
// why OptiX can't run here: no headers in this build, a driver older than CUDA 13.4, no OptiX in the driver, an
// OptiX older than the headers this build compiled with (9.0).
std::shared_ptr<optix_runtime> start_optix();

// "OptiX 9.0, programs 3f2a9c1e": for the HUD and the programs' messages
std::string describe(const optix_runtime& runtime);

// Whether OptiX keeps the programs it compiled in its disk cache (optixDeviceContextGetCacheEnabled, asked when the
// context was made): when it doesn't (OPTIX_CACHE_MAXSIZE=0, or a cache that couldn't start), every start compiles
// them again, which the window's Note says
bool optix_cache_enabled(const optix_runtime& runtime);
}
