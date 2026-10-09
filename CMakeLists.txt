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

# --- Ray tracer --------------------------------------------------------------
add_executable(RayTracing src/main.cpp)
target_sources(RayTracing PRIVATE
    FILE_SET CXX_MODULES FILES   # our modules: files with `export module ...`
        src/image.ixx
        src/vec3.ixx)

# Third-party code lives in external/. SYSTEM keeps the compiler from showing its warnings.
#   stb/stb_image_write.h - v1.16, github.com/nothings/stb, commit 2c980bb (PNG writing)
target_include_directories(RayTracing SYSTEM PRIVATE external/stb)

target_link_libraries(RayTracing PRIVATE std_module)
target_compile_options(RayTracing PRIVATE /W4)
