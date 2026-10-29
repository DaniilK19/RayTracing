# 3.28 is the first CMake version with stable support for C++20 modules (needed for `import std;`)
cmake_minimum_required(VERSION 3.28)

project(RayTracing LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR "The project is only set up for MSVC (Visual Studio) so far.")
endif()

# The whole project is C++23. For MSVC, CMake turns this into /std:c++latest:
# C++23 plus the little of C++26 the compiler already has.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug information: /ZI in Debug (Hot Reload: edit code while debugging),
# a regular PDB (/Zi) in RelWithDebInfo, none in Release.
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
    "$<$<CONFIG:Debug>:EditAndContinue>$<$<CONFIG:RelWithDebInfo>:ProgramDatabase>")

# Flags for all C++ sources (the std module and our code must use the same ones):
#   /utf-8           - sources and strings are UTF-8: non-ASCII text in code and output works
#   /Zc:__cplusplus  - __cplusplus reports the real standard version instead of 199711L
#   /Zc:preprocessor - a standard-conforming preprocessor
# They are flags of cl.exe, MSVC's compiler. The GPU renderer's .cu file is compiled by nvcc, NVIDIA's CUDA compiler,
# which doesn't accept them: it compiles the GPU code itself and hands the rest of the file to cl.exe, with the flags
# given after -Xcompiler. $<COMPILE_LANGUAGE:...> is a generator expression, which CMake works out for each source file
# when it writes the build files, so each kind of source gets the flags in its compiler's form.
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:/utf-8;/Zc:__cplusplus;/Zc:preprocessor>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/Zc:__cplusplus,/Zc:preprocessor>")

# --- import std; -------------------------------------------------------------
# The standard library as a module (C++23). MSVC ships its source, std.ixx,
# and we build it as an ordinary module in a separate library. CMake's built-in support
# (CMAKE_CXX_MODULE_STD) is still experimental and breaks between updates.
# cl.exe lives in <MSVC>/bin/Hostx64/x64/, module sources in <MSVC>/modules/.
cmake_path(SET MSVC_MODULES_DIR NORMALIZE "${CMAKE_CXX_COMPILER}/../../../../modules")
if(NOT EXISTS "${MSVC_MODULES_DIR}/std.ixx")
    # message() joins its strings without a separator: one message on two lines of code
    message(FATAL_ERROR "${MSVC_MODULES_DIR}/std.ixx not found: "
                        "check the MSVC installation in the Visual Studio Installer.")
endif()

add_library(std_module STATIC)
target_sources(std_module PUBLIC
    FILE_SET CXX_MODULES
    BASE_DIRS "${MSVC_MODULES_DIR}"
    FILES "${MSVC_MODULES_DIR}/std.ixx")

# --- Ray tracer library ------------------------------------------------------
# Five programs use the ray tracer's modules: RayTracing renders the scenes; MonteCarlo runs the third book's numerical
# experiments with the ray tracer's own random numbers, directions and densities, and tests its light sampling
# (light-pdfs); ImageCompare reads and compares renders with the image module; and the GPU programs (below),
# RayTracingGPU and RayTracingViewer, build their scenes with the very same code. So the modules are compiled once into
# a static library, and the programs link it.
add_library(raytracer STATIC)
target_sources(raytracer PUBLIC  # PUBLIC: programs that link the library can `import` its modules
    FILE_SET CXX_MODULES FILES   # our modules: files with `export module ...`
        src/aabb.ixx
        src/bvh.ixx
        src/camera.ixx
        src/cli.ixx
        src/constant_medium.ixx
        src/hittable.ixx
        src/image.ixx
        src/instance.ixx
        src/interval.ixx
        src/material.ixx
        src/onb.ixx
        src/pdf.ixx
        src/perlin.ixx
        src/quad.ixx
        src/random.ixx
        src/ray.ixx
        src/scenes.ixx
        src/sphere.ixx
        src/texture.ixx
        src/vec3.ixx)

# Third-party code lives in external/. SYSTEM keeps the compiler from showing its warnings.
# PRIVATE: stb is used only inside the library (by the image module), not by the programs.
#   stb/stb_image_write.h - v1.16, github.com/nothings/stb, commit 2c980bb (PNG writing)
#   stb/stb_image.h       - v2.30, same source, same commit (JPEG/PNG reading for textures)
target_include_directories(raytracer SYSTEM PRIVATE external/stb)

# PUBLIC: the programs `import std;` too and get the std module along with the library
target_link_libraries(raytracer PUBLIC std_module)
target_compile_options(raytracer PRIVATE /W4)

# --- Programs ----------------------------------------------------------------
# utf8.manifest: UTF-8 as the process code page (see the comment in the file)
add_executable(RayTracing src/main.cpp src/utf8.manifest)
target_link_libraries(RayTracing PRIVATE raytracer)
target_compile_options(RayTracing PRIVATE /W4)

add_executable(MonteCarlo src/monte_carlo.cpp src/utf8.manifest)
target_link_libraries(MonteCarlo PRIVATE raytracer)
target_compile_options(MonteCarlo PRIVATE /W4)

# ImageCompare: are two renders (PFM files) the same picture up to Monte Carlo noise? It uses the image module
# to read them and write its map, so it links the library too.
add_executable(ImageCompare src/image_compare.cpp src/utf8.manifest)
target_link_libraries(ImageCompare PRIVATE raytracer)
target_compile_options(ImageCompare PRIVATE /W4)

# PfmStats: numbers about renders that ImageCompare doesn't give (noise, exact values, identical bytes) for the checks
# of the GPU's kernels (tests/pfm_stats.cpp). It reads the files with the image module.
add_executable(PfmStats tests/pfm_stats.cpp src/utf8.manifest)
target_link_libraries(PfmStats PRIVATE raytracer)
target_compile_options(PfmStats PRIVATE /W4)

# --- GPU programs (optional) -------------------------------------------------
# RayTracingGPU renders RayTracing's scenes on an NVIDIA graphics card (see main_gpu.cpp), and RayTracingViewer shows
# them in a window as they render (main_viewer.cpp). They need the CUDA Toolkit (nvcc and the CUDA libraries); without
# it, everything else builds as before. CMake looks for nvcc in the folder named by the
# CUDA_PATH environment variable and on the PATH, both set by the toolkit's installer: a program started before the
# installation, such as Visual Studio or a terminal, keeps its old environment until it is restarted.
# check_language tries the CUDA compiler on a small test project and caches the answer, "not found" too, so after
# installing the toolkit the CMake cache must go as well (in Visual Studio: Project > Delete Cache and Reconfigure;
# on the command line: cmake --fresh).
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # The GPUs to compile the kernels for: by default the one in this computer ("native"; with no
    # GPU, nvcc warns and takes its default). Choose others with -DCMAKE_CUDA_ARCHITECTURES=89 or the CUDAARCHS
    # environment variable. It must be set before enable_language(CUDA), which would otherwise take nvcc's default.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)

    # C++20, not 23: with cl.exe as its host compiler, nvcc ignores -std=c++23 ("not supported with the configured host
    # compiler... Flag will be ignored") and falls back to C++14, and CMake refuses CUDA_STANDARD 23 for this pair. nvcc
    # can't import modules either, so the GPU code is written in ordinary headers. C++20 still gives it concepts.
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # The CUDA runtime library (GPU memory, kernel launches...) is the static one, cudart_static.lib: the program
    # needs only the graphics driver, no CUDA DLL. CUDA 13's nvcc links a "hybrid" cudart.lib by default, which on
    # Windows holds the same code, but CMake counts that default as the shared runtime; naming Static keeps the
    # choice explicit and the same with every CUDA version.
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

    # The renderer: kernels (functions that run on the GPU) and the host code that launches them. nvcc splits a .cu
    # file in two: it compiles the device code itself and hands the host code to cl.exe with the flags given after
    # -Xcompiler: those of our C++ code (at the top of this file), and /W4 as for our programs.
    # Device code is optimized in every configuration, Debug too: -G (debug information for device code) would turn its
    # optimizations off and make the kernels hundreds of times slower (400-600 times, measured on the test GPU), so a
    # render of a second would take ten minutes. -lineinfo, in Debug and RelWithDebInfo, only maps the GPU's machine
    # code to source lines, for profilers such as Nsight Compute.
    # quality.cu, the third kernel's (light and materials beyond the books), is a translation unit of its own too, with
    # renderer.cu's flags: nvcc compiles the books' kernels in renderer.cu as if it didn't exist. So is tonemap.cu, the
    # display's tone maps beyond the books, with display.cu's flags (below). lamp_table.cu builds the quality kernel's
    # light set on the host: no device code.
    add_library(gpu_renderer STATIC src/gpu/renderer.cu src/gpu/display.cu src/gpu/quality.cu src/gpu/tonemap.cu
        src/gpu/lamp_table.cu)
    target_compile_options(gpu_renderer PRIVATE -Xcompiler=/W4 $<$<CONFIG:Debug,RelWithDebInfo>:-lineinfo>)
    # The window's conversion of the image to bytes (display.cu) must give the bytes of the PNG files, which the CPU
    # computes: -fmad=false keeps nvcc from fusing a multiplication and an addition into one operation with one
    # rounding, which the CPU doesn't do. (The kernels in renderer.cu keep nvcc's default, which they were measured
    # with.)
    set_source_files_properties(src/gpu/display.cu PROPERTIES COMPILE_OPTIONS -fmad=false)
    # The tone maps' bytes (tonemap.cu) have no CPU twin, but they too should follow from the code as written, every
    # product and sum rounded on its own: the same flag.
    set_source_files_properties(src/gpu/tonemap.cu PROPERTIES COMPILE_OPTIONS -fmad=false)

    # The CPU's scenes as the renderer's data (gpu_scene.ixx): a module in a library of its own, which every GPU
    # program links. It includes the renderer's header and imports the ray tracer's modules, so it comes with both.
    # With it, the GPU programs' own lists of scenes (gpu_scenes.ixx and its partition gpu_scene_parts.ixx) and of
    # sampling strategies (strategy_options.ixx), and their view and display options (view_options.ixx,
    # tone_options.ixx).
    add_library(gpu_scene STATIC)
    target_sources(gpu_scene PUBLIC FILE_SET CXX_MODULES FILES
        src/gpu_scene.ixx src/gpu_scene_parts.ixx src/gpu_scenes.ixx src/strategy_options.ixx src/tone_options.ixx
        src/view_options.ixx)
    target_link_libraries(gpu_scene PUBLIC raytracer gpu_renderer)
    target_compile_options(gpu_scene PRIVATE /W4)

    add_executable(RayTracingGPU src/main_gpu.cpp src/utf8.manifest)
    target_link_libraries(RayTracingGPU PRIVATE gpu_scene)
    target_compile_options(RayTracingGPU PRIVATE /W4)

    # GpuTests: numeric tests of the renderer's device functions, run on the GPU (tests/gpu_tests.cpp and the suites,
    # tests/*_tests.cu). A suite is a file of its own: the glob finds new ones, and CONFIGURE_DEPENDS has the build
    # look for them again each time it runs.
    file(GLOB gpu_test_suites CONFIGURE_DEPENDS tests/*_tests.cu)
    add_executable(GpuTests tests/gpu_tests.cpp ${gpu_test_suites} src/utf8.manifest)
    # gpu_tests.cpp, compiled by cl.exe, asks the CUDA runtime whether there is a GPU at all: the toolkit's headers
    target_include_directories(GpuTests PRIVATE src/gpu tests ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(GpuTests PRIVATE gpu_scene)
    target_compile_options(GpuTests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
else()
    message(STATUS "No working CUDA compiler found: the GPU programs, RayTracingGPU, RayTracingViewer and GpuTests, "
                   "are skipped")
    message(STATUS "After installing the CUDA Toolkit, restart Visual Studio (or the terminal) and delete the CMake "
                   "cache: Project > Delete Cache and Reconfigure (or cmake --fresh)")
endif()

# RayTracingViewer, the window, with SDL3 for the window and for drawing the image in it (cmake/sdl3.cmake). Without
# CUDA the file only declares its options, so that giving them is no mistake.
include(cmake/sdl3.cmake)

# --- Tests ---------------------------------------------------------------------
# ctest (in the build folder) runs PfmStats' check of its own arithmetic and GpuTests' quick run: every test kernel on
# a few samples, the statistical verdicts skipped. GpuTests exits with 77 on a computer without an NVIDIA GPU, which
# ctest reports as skipped. The full statistical runs, the renders and the gates of scripts/verify are run by hand.
enable_testing()
add_test(NAME pfm_stats_self_test COMMAND PfmStats self-test)
if(TARGET GpuTests)
    add_test(NAME gpu_tests_quick COMMAND GpuTests all --quick)
    set_tests_properties(gpu_tests_quick PROPERTIES SKIP_RETURN_CODE 77)
endif()
