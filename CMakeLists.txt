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

# Flags for all targets (the std module and our code must use the same ones):
#   /utf-8           - sources and strings are UTF-8: non-ASCII text in code and output works
#   /Zc:__cplusplus  - __cplusplus reports the real standard version instead of 199711L
#   /Zc:preprocessor - a standard-conforming preprocessor
add_compile_options(/utf-8 /Zc:__cplusplus /Zc:preprocessor)

# --- import std; -------------------------------------------------------------
# The standard library as a module (C++23). MSVC ships its source, std.ixx,
# and we build it as an ordinary module in a separate library. CMake's built-in support
# (CMAKE_CXX_MODULE_STD) is still experimental and breaks between updates.
# cl.exe lives in <MSVC>/bin/Hostx64/x64/, module sources in <MSVC>/modules/.
cmake_path(SET MSVC_MODULES_DIR NORMALIZE "${CMAKE_CXX_COMPILER}/../../../../modules")
if(NOT EXISTS "${MSVC_MODULES_DIR}/std.ixx")
    message(FATAL_ERROR "${MSVC_MODULES_DIR}/std.ixx not found: check the MSVC installation in the Visual Studio Installer.")
endif()

add_library(std_module STATIC)
target_sources(std_module PUBLIC
    FILE_SET CXX_MODULES
    BASE_DIRS "${MSVC_MODULES_DIR}"
    FILES "${MSVC_MODULES_DIR}/std.ixx")

# --- Ray tracer library ------------------------------------------------------
# Two programs use the ray tracer's modules: RayTracing renders the scenes, and MonteCarlo runs
# the third book's numerical experiments with the ray tracer's own random numbers. So the modules
# are compiled once into a static library, and both programs link it.
add_library(raytracer STATIC)
target_sources(raytracer PUBLIC  # PUBLIC: programs that link the library can `import` its modules
    FILE_SET CXX_MODULES FILES   # our modules: files with `export module ...`
        src/aabb.ixx
        src/bvh.ixx
        src/camera.ixx
        src/constant_medium.ixx
        src/hittable.ixx
        src/image.ixx
        src/instance.ixx
        src/interval.ixx
        src/material.ixx
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
