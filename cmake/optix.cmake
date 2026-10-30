# NVIDIA OptiX 9.0.0 for --optix, triangle meshes and the denoiser: its headers only, from github.com/NVIDIA/optix-dev
# (tag v9.0.0), downloaded once into out/downloads, never committed. They come under NVIDIA's SDK license agreement
# (the archive's LICENSE.txt) and, header by header, NVIDIA's proprietary license (eight of them) or BSD-3 (six), as
# the archive's license_info.txt and each header's first comment say. The OptiX runtime is the display driver's
# nvoptix.dll, which optix_stubs.h loads: nothing to link, nothing to ship. Trouble never stops the configure: --optix
# is left out with a warning, everything else builds.
option(RAYTRACING_OPTIX "Build --optix, meshes and the denoiser (CMake downloads NVIDIA's OptiX 9.0 headers once)" ON)
set(RAYTRACING_OPTIX_URL "" CACHE STRING
    "Where to get optix-dev 9.0.0's archive instead of GitHub: a URL, or file:///C:/path/optix-dev-9.0.0.tar.gz")
set(OptiX_INSTALL_DIR "$ENV{OptiX_INSTALL_DIR}" CACHE PATH
    "An installed OptiX SDK 9.x (the folder holding include/), instead of the download")
set(optix_url "https://github.com/NVIDIA/optix-dev/archive/refs/tags/v9.0.0.tar.gz")
set(optix_sha256 069a5860040ea611e7eb6317f8e3bb0f0d54a5acac744568f7290d7cb8711c05) # 104,546 bytes (2026-10-30)
set(optix_archive "${RAYTRACING_DOWNLOADS}/optix-dev-9.0.0.tar.gz")
# The three ways to OptiX that every warning names, after its own first words (one string: message() would join a
# list's items without their separators anyway)
string(CONCAT optix_routes
    "--optix, triangle meshes and the denoiser are left out; everything else builds. Get optix-dev-9.0.0.tar.gz from "
    "${optix_url} by hand (SHA-256 ${optix_sha256}) and configure with -DRAYTRACING_OPTIX_URL=file:///<path to it>, "
    "or point -DOptiX_INSTALL_DIR=<folder> at an installed OptiX SDK 9.x, or turn OptiX off with "
    "-DRAYTRACING_OPTIX=OFF.")
# A URL given by hand, or empty for GitHub's archive (not GitHub's URL as the default: the cache would keep it, as
# sdl3.cmake says)
if(RAYTRACING_OPTIX_URL)
    set(optix_url "${RAYTRACING_OPTIX_URL}")
endif()
if(NOT CMAKE_CUDA_COMPILER)
    return()
endif()

# The folder that holds OptiX's include/: an installed SDK's, read where it is, or the download's
set(optix_root "")
if(RAYTRACING_OPTIX AND OptiX_INSTALL_DIR)
    # An installed SDK, read where it is, never through FETCHCONTENT_SOURCE_DIR_OPTIX: FetchContent stops the configure
    # when that folder isn't there, and keeps it in its cache entry when OptiX_INSTALL_DIR changes. Here a folder
    # without OptiX 9's headers is the warning below, and a new OptiX_INSTALL_DIR (empty: the download) counts at the
    # next configure.
    file(TO_CMAKE_PATH "${OptiX_INSTALL_DIR}" optix_root)
elseif(RAYTRACING_OPTIX AND FETCHCONTENT_SOURCE_DIR_OPTIX)
    # FetchContent's own variable for a folder in place of the download (sdl3.cmake honors FETCHCONTENT_SOURCE_DIR_SDL3
    # too), read the same way
    file(TO_CMAKE_PATH "${FETCHCONTENT_SOURCE_DIR_OPTIX}" optix_root)
elseif(RAYTRACING_OPTIX)
    # The download, as sdl3.cmake's: the archive is kept in out/downloads, shared by all the presets' build folders. It
    # goes to a file next to it named after the build folder (optix-dev-9.0.0.tar.gz.<SHA-1 of the folder's path>.part)
    # and takes the archive's name only once its SHA-256 is right, so an interrupted or wrong download never passes for
    # the archive, and two configures at once never write the same file. Only an archive that is there and wrong is
    # deleted: one that is missing may be on its way from another configure.
    set(optix_hash "")
    if(EXISTS "${optix_archive}")
        file(SHA256 "${optix_archive}" optix_hash)
    endif()
    if(NOT optix_hash STREQUAL optix_sha256)
        if(optix_hash)
            file(REMOVE "${optix_archive}")
        endif()
        string(SHA1 optix_tag "${CMAKE_BINARY_DIR}")
        set(optix_part "${optix_archive}.${optix_tag}.part")
        message(STATUS "Downloading NVIDIA's OptiX 9.0 headers for --optix (once, into ${RAYTRACING_DOWNLOADS}); they "
                       "come under NVIDIA's licenses, which the archive's LICENSE.txt and license_info.txt state")
        # Without EXPECTED_HASH: with it, a failed download stops the whole configure. The hash is checked below.
        file(DOWNLOAD "${optix_url}" "${optix_part}" STATUS optix_status TIMEOUT 120 INACTIVITY_TIMEOUT 30)
        list(GET optix_status 0 optix_code)
        set(optix_got "")
        if(optix_code EQUAL 0)
            file(SHA256 "${optix_part}" optix_got)
        endif()
        if(optix_got STREQUAL optix_sha256)
            # NO_REPLACE: if another configure got there first, its archive (checked again below) stays
            file(RENAME "${optix_part}" "${optix_archive}" RESULT optix_renamed NO_REPLACE)
        endif()
        file(REMOVE "${optix_part}") # a failed or wrong download leaves a file too
        set(optix_hash "")
        if(EXISTS "${optix_archive}")
            file(SHA256 "${optix_archive}" optix_hash)
        endif()
        if(NOT optix_hash STREQUAL optix_sha256)
            if(optix_hash)
                file(REMOVE "${optix_archive}")
            endif()
            if(optix_code EQUAL 0)
                set(optix_problem "${optix_url} is not optix-dev 9.0.0's archive (its SHA-256 is ${optix_got})")
            else()
                set(optix_problem "OptiX 9.0's headers could not be downloaded from ${optix_url} (${optix_status})")
            endif()
            message(WARNING "${optix_problem}: ${optix_routes}")
        endif()
    endif()
    # The archive unpacked into the build folder (it holds no CMakeLists.txt, so FetchContent only unpacks it)
    if(EXISTS "${optix_archive}")
        include(FetchContent)
        FetchContent_Declare(optix URL "${optix_archive}" URL_HASH SHA256=${optix_sha256})
        FetchContent_MakeAvailable(optix)
        set(optix_root "${optix_SOURCE_DIR}")
    endif()
endif()

# The headers and their version, read from optix.h: 9.0 is the one this code is written against, a later 9.x compiles
# too (one #if in runtime.cpp). A folder that isn't there, or holds no OptiX 9.x, is a warning with the three ways.
set(optix_include "")
if(optix_root)
    set(optix_version "")
    if(EXISTS "${optix_root}/include/optix.h")
        file(STRINGS "${optix_root}/include/optix.h" optix_version_line REGEX "^#define OPTIX_VERSION [0-9]+")
        string(REGEX MATCH "[0-9]+$" optix_version "${optix_version_line}")
    endif()
    if(NOT optix_version)
        message(WARNING "${optix_root} has no include/optix.h that says its version: ${optix_routes}")
    elseif(optix_version LESS 90000 OR NOT optix_version LESS 100000)
        message(WARNING "${optix_root}/include holds OptiX ${optix_version}, not 9.x: ${optix_routes}")
    else()
        set(optix_include "${optix_root}/include")
    endif()
endif()

# bin2c, the CUDA Toolkit's tool that writes a file as a C array (embed.cmake), next to nvcc
if(optix_include)
    get_filename_component(optix_nvcc_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    find_program(RAYTRACING_BIN2C bin2c HINTS "${optix_nvcc_dir}" NO_DEFAULT_PATH)
    if(NOT RAYTRACING_BIN2C)
        message(WARNING "The CUDA Toolkit's bin2c isn't next to ${CMAKE_CUDA_COMPILER}: --optix, triangle meshes and "
                        "the denoiser are left out; everything else builds. bin2c comes with nvcc in the CUDA "
                        "Toolkit's bin folder: repair or reinstall the toolkit so that its bin folder has it, or turn "
                        "OptiX off with -DRAYTRACING_OPTIX=OFF.")
        set(optix_include "")
    endif()
endif()

# For scripts/verify/irgate.ps1: the include folder this build compiles against (empty without OptiX), whose file names
# no project header may take
set(RAYTRACING_OPTIX_INCLUDE_DIR "${optix_include}" CACHE INTERNAL "The OptiX headers this build compiles against")

# One device module as PTX inside a program (embed.cmake): its object, a .ptx file, written as the array `name`, its
# size and its SHA-256 (src/gpu/optix/embedded.h declares them) into <build>/optix/<name>.cpp, and copied to
# <build>/optix/<copy>.ptx with its SHA-256 in <copy>.sha256 for irgate.ps1. Each is rewritten only when the PTX
# changed, so an unchanged module recompiles nothing.
set(raytracing_embed_script "${CMAKE_CURRENT_LIST_DIR}/embed.cmake")
function(raytracing_embed_ptx module name copy)
    set(dir "${CMAKE_CURRENT_BINARY_DIR}/optix")
    add_custom_command(OUTPUT "${dir}/${name}.cpp"
        BYPRODUCTS "${dir}/${copy}.ptx" "${dir}/${copy}.sha256"
        COMMAND "${CMAKE_COMMAND}" "-DBIN2C=${RAYTRACING_BIN2C}" "-DNAME=${name}" "-DPTX=$<TARGET_OBJECTS:${module}>"
                "-DCPP=${dir}/${name}.cpp" "-DCOPY=${dir}/${copy}.ptx" "-DSHA256=${dir}/${copy}.sha256"
                -P "${raytracing_embed_script}"
        DEPENDS "$<TARGET_OBJECTS:${module}>" "${raytracing_embed_script}"
        COMMENT "Embedding ${copy}.ptx"
        VERBATIM)
endfunction()

if(optix_include)
    message(STATUS "OptiX headers ${optix_version} in ${optix_include}")
    add_library(optix_headers INTERFACE)
    target_include_directories(optix_headers SYSTEM INTERFACE "${optix_include}")
    # The OptiX programs as PTX for the lowest virtual architecture CUDA 13 compiles for (OptiX compiles it again for
    # the GPU it runs on). CMake's architectures would become -gencode and "native" emits no PTX: set by hand. The math
    # flags are quality.cu's: nvcc's default -fmad, no --use_fast_math. No -lineinfo in any configuration and no -G:
    # the PTX then holds no file name, and Debug's and Release's are the same bytes (irgate.ps1 checks both). No -rdc:
    # no callables, one translation unit per module.
    add_library(optix_programs OBJECT src/gpu/optix/programs.cu)
    set_target_properties(optix_programs PROPERTIES CUDA_PTX_COMPILATION ON CUDA_ARCHITECTURES OFF)
    target_compile_options(optix_programs PRIVATE --gpu-architecture=compute_75)
    target_include_directories(optix_programs PRIVATE src/gpu src/gpu/optix)
    target_link_libraries(optix_programs PRIVATE optix_headers)
    # GpuTests' own module (probes and the tracer check): the same properties, never shipped
    add_library(optix_check_programs OBJECT tests/optix_check_programs.cu)
    set_target_properties(optix_check_programs PROPERTIES CUDA_PTX_COMPILATION ON CUDA_ARCHITECTURES OFF)
    target_compile_options(optix_check_programs PRIVATE --gpu-architecture=compute_75)
    target_include_directories(optix_check_programs PRIVATE src/gpu src/gpu/optix tests)
    target_link_libraries(optix_check_programs PRIVATE optix_headers)
    # No project header may share a base name with a header of OptiX's include/ folder: the project's include
    # directories come before the SYSTEM one, so a src/gpu/optix.h would shadow NVIDIA's optix.h (in programs.cu,
    # optix_tracer.cuh, runtime.cpp, and inside optix_denoiser_tiling.h's own `#include <optix.h>`). The programs'
    # header is therefore src/gpu/start_optix.h; irgate.ps1 fails on any such clash, and programs.cu, the check module,
    # runtime.cpp and denoiser.cpp each `static_assert(OPTIX_VERSION >= 90000)` after their OptiX include.
    raytracing_embed_ptx(optix_programs optix_programs_ptx optix_programs)
    raytracing_embed_ptx(optix_check_programs optix_check_ptx optix_check_ptx)
    # The runtime: runtime.cpp, accel.cpp and denoiser.cpp, compiled by cl.exe, make every OptiX call (optix.h takes
    # its host branch there); scene.cu and meshes.cu, compiled by nvcc, fill the device structs; they meet in
    # runtime_core.h. The render module's PTX is linked in as an array.
    add_library(gpu_optix STATIC src/gpu/optix/runtime.cpp src/gpu/optix/accel.cpp src/gpu/optix/denoiser.cpp
        src/gpu/optix/scene.cu src/gpu/optix/meshes.cu "${CMAKE_CURRENT_BINARY_DIR}/optix/optix_programs_ptx.cpp")
    add_dependencies(gpu_optix optix_programs)
    target_link_libraries(gpu_optix PUBLIC gpu_renderer PRIVATE optix_headers)
    target_include_directories(gpu_optix PRIVATE src/gpu src/gpu/optix)
    target_include_directories(gpu_optix SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # cuda.h for .cpp
    target_compile_definitions(gpu_optix PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN) # never add_compile_definitions (gate F)
    target_compile_options(gpu_optix PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
    # GpuTests' check module, linked into the test program only
    target_sources(GpuTests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/optix/optix_check_ptx.cpp")
    add_dependencies(GpuTests optix_check_programs)
else()
    if(NOT RAYTRACING_OPTIX)
        message(STATUS "OptiX is off (RAYTRACING_OPTIX): --optix, triangle meshes and the denoiser are left out")
    endif()
    add_library(gpu_optix STATIC src/gpu/optix/absent.cpp) # start_optix() throws "this build has no OptiX ..."
    target_link_libraries(gpu_optix PUBLIC gpu_renderer)
    target_compile_options(gpu_optix PRIVATE /W4)
    target_sources(GpuTests PRIVATE tests/optix_check_absent.cpp) # an empty check module: the OptiX suites SKIP
endif()
target_link_libraries(gpu_scene PUBLIC gpu_optix)
target_sources(GpuTests PRIVATE tests/scene_services.cpp) # the scenes as gpu_scene builds them, for the OptiX suites
# The denoiser's CUDA kernels (both builds): new translation units of the renderer, its existing ones untouched
target_sources(gpu_renderer PRIVATE src/gpu/guides.cu src/gpu/denoise.cu)
