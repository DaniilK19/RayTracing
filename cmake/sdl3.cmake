# SDL 3.4.18 (zlib license, github.com/libsdl-org/SDL, tag release-3.4.18, commit 829a65d) for RayTracingViewer, the
# window: CMake downloads its release archive once, checks its SHA-256 and builds it as a static library, so the
# program needs no DLL. Trouble with the download never stops the configure: the window is skipped with a warning and
# everything else builds. Without the CUDA compiler there is no viewer: the file only declares its options.
option(RAYTRACING_VIEWER "Build RayTracingViewer, the window (CMake downloads SDL3 once)" ON)
# A URL given by hand, or empty for GitHub's release. (Not GitHub's URL as the default: the cache would keep it, and an
# update of SDL in this file would then download the old version.)
set(RAYTRACING_SDL3_URL "" CACHE STRING
    "Where to get SDL 3.4.18's archive instead of GitHub: a URL, or file:///C:/path/SDL3-3.4.18.tar.gz")
set(RAYTRACING_DOWNLOADS "${CMAKE_SOURCE_DIR}/out/downloads" CACHE PATH "Where downloaded archives are kept")
set(sdl3_url "https://github.com/libsdl-org/SDL/releases/download/release-3.4.18/SDL3-3.4.18.tar.gz")
if(RAYTRACING_SDL3_URL)
    set(sdl3_url "${RAYTRACING_SDL3_URL}")
endif()
set(sdl3_sha256 9c75cf16330322c217dedd2e0609f1124f1b54b8633e763467b4684d0f4334a3)
set(sdl3_archive "${RAYTRACING_DOWNLOADS}/SDL3-3.4.18.tar.gz")
if(NOT CMAKE_CUDA_COMPILER)
    return()
endif()

# The archive is kept in out/downloads, shared by all the presets' build folders: downloaded once. A download goes to
# a file next to it named after the build folder (SDL3-3.4.18.tar.gz.<SHA-1 of the folder's path>.part) and takes the
# archive's name only once its SHA-256 is right, so an interrupted or wrong download never passes for the archive, and
# two configures at once never write the same file. Only an archive that is there and wrong is deleted: one that is
# missing may be on its way from another configure.
if(RAYTRACING_VIEWER AND NOT FETCHCONTENT_SOURCE_DIR_SDL3)
    set(sdl3_hash "")
    if(EXISTS "${sdl3_archive}")
        file(SHA256 "${sdl3_archive}" sdl3_hash)
    endif()
    if(NOT sdl3_hash STREQUAL sdl3_sha256)
        if(sdl3_hash)
            file(REMOVE "${sdl3_archive}")
        endif()
        string(SHA1 sdl3_tag "${CMAKE_BINARY_DIR}")
        set(sdl3_part "${sdl3_archive}.${sdl3_tag}.part")
        message(STATUS "Downloading SDL 3.4.18 for RayTracingViewer (once, into ${RAYTRACING_DOWNLOADS})")
        # Without EXPECTED_HASH: with it, a failed download stops the whole configure. The hash is checked below.
        file(DOWNLOAD "${sdl3_url}" "${sdl3_part}" STATUS sdl3_status TIMEOUT 300 INACTIVITY_TIMEOUT 30)
        list(GET sdl3_status 0 sdl3_code)
        set(sdl3_got "")
        if(sdl3_code EQUAL 0)
            file(SHA256 "${sdl3_part}" sdl3_got)
        endif()
        if(sdl3_got STREQUAL sdl3_sha256)
            # NO_REPLACE: if another configure got there first, its archive (checked again below) stays
            file(RENAME "${sdl3_part}" "${sdl3_archive}" RESULT sdl3_renamed NO_REPLACE)
        endif()
        file(REMOVE "${sdl3_part}") # a failed or wrong download leaves a file too
        set(sdl3_hash "")
        if(EXISTS "${sdl3_archive}")
            file(SHA256 "${sdl3_archive}" sdl3_hash)
        endif()
        if(NOT sdl3_hash STREQUAL sdl3_sha256)
            if(sdl3_hash)
                file(REMOVE "${sdl3_archive}")
            endif()
            if(sdl3_code EQUAL 0)
                set(sdl3_problem "${sdl3_url} is not SDL 3.4.18's release archive (its SHA-256 is ${sdl3_got})")
            else()
                set(sdl3_problem "SDL 3.4.18 could not be downloaded from ${sdl3_url} (${sdl3_status})")
            endif()
            message(WARNING "${sdl3_problem}: RayTracingViewer, the window, is skipped; everything else builds. "
                            "Get SDL3-3.4.18.tar.gz from github.com/libsdl-org/SDL/releases/tag/release-3.4.18 by "
                            "hand (SHA-256 ${sdl3_sha256}) and configure with "
                            "-DRAYTRACING_SDL3_URL=file:///<path to it>, or turn the window off with "
                            "-DRAYTRACING_VIEWER=OFF.")
        endif()
    endif()
endif()

if(RAYTRACING_VIEWER AND (FETCHCONTENT_SOURCE_DIR_SDL3 OR EXISTS "${sdl3_archive}"))
    # A static library only, without SDL's test library (built by default even when SDL is part of another project)
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    include(FetchContent)
    # SYSTEM: SDL's headers raise no warnings in our code; EXCLUDE_FROM_ALL: only what the viewer links is built
    FetchContent_Declare(SDL3 URL "${sdl3_archive}" URL_HASH SHA256=${sdl3_sha256} SYSTEM EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(SDL3)

    add_executable(RayTracingViewer src/main_viewer.cpp src/utf8.manifest)
    target_link_libraries(RayTracingViewer PRIVATE gpu_scene SDL3::SDL3-static)
    target_compile_options(RayTracingViewer PRIVATE /W4)
elseif(NOT RAYTRACING_VIEWER)
    message(STATUS "RayTracingViewer, the window, is off (RAYTRACING_VIEWER)")
endif()
