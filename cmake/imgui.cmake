# Dear ImGui 1.93.0 (MIT license, github.com/ocornut/imgui, tag v1.93.0) for RayTracingViewer's panel: the scenes, the
# settings and the renders to a file, chosen with the mouse (src/viewer_panel.ixx). CMake downloads the tag's archive
# once, as it downloads SDL's (cmake/sdl3.cmake), checks its SHA-256 and compiles the library's own files, four of its
# core and the two that draw it through SDL 3 and SDL's renderer, into a static library: no DLL, nothing installed.
# Trouble with the download never stops the configure: the viewer is then built without its panel, which it says when
# asked for it (src/viewer_panel_absent.ixx), and everything else is as before. Included after cmake/viewer.cmake.
option(RAYTRACING_PANEL "Build RayTracingViewer's panel (CMake downloads Dear ImGui once)" ON)
# A URL given by hand, or empty for GitHub's archive of the tag (not GitHub's URL as the default: see cmake/sdl3.cmake)
set(RAYTRACING_IMGUI_URL "" CACHE STRING
    "Where to get Dear ImGui 1.93.0's archive instead of GitHub: a URL, or file:///C:/path/imgui-1.93.0.tar.gz")
set(imgui_url "https://github.com/ocornut/imgui/archive/refs/tags/v1.93.0.tar.gz")
if(RAYTRACING_IMGUI_URL)
    set(imgui_url "${RAYTRACING_IMGUI_URL}")
endif()
set(imgui_sha256 f7bf6bc76e5d79d40d751096663e6c5d18daf08129dd52e7f7b09e4251c8eeee)
set(imgui_archive "${RAYTRACING_DOWNLOADS}/imgui-1.93.0.tar.gz")
if(NOT TARGET RayTracingViewer)
    return() # no window (no CUDA, no SDL, or RAYTRACING_VIEWER off): no panel to build
endif()

# The archive, kept in out/downloads beside SDL's and found or downloaded the same way: to a file named after the
# build folder, which takes the archive's name only once its SHA-256 is right
if(RAYTRACING_PANEL)
    set(imgui_hash "")
    if(EXISTS "${imgui_archive}")
        file(SHA256 "${imgui_archive}" imgui_hash)
    endif()
    if(NOT imgui_hash STREQUAL imgui_sha256)
        if(imgui_hash)
            file(REMOVE "${imgui_archive}")
        endif()
        string(SHA1 imgui_tag "${CMAKE_BINARY_DIR}")
        set(imgui_part "${imgui_archive}.${imgui_tag}.part")
        message(STATUS "Downloading Dear ImGui 1.93.0 for RayTracingViewer's panel (once, into ${RAYTRACING_DOWNLOADS})")
        # Without EXPECTED_HASH: with it, a failed download stops the whole configure. The hash is checked below.
        file(DOWNLOAD "${imgui_url}" "${imgui_part}" STATUS imgui_status TIMEOUT 300 INACTIVITY_TIMEOUT 30)
        list(GET imgui_status 0 imgui_code)
        set(imgui_got "")
        if(imgui_code EQUAL 0)
            file(SHA256 "${imgui_part}" imgui_got)
        endif()
        if(imgui_got STREQUAL imgui_sha256)
            # NO_REPLACE: if another configure got there first, its archive (checked again below) stays
            file(RENAME "${imgui_part}" "${imgui_archive}" RESULT imgui_renamed NO_REPLACE)
        endif()
        file(REMOVE "${imgui_part}") # a failed or wrong download leaves a file too
        set(imgui_hash "")
        if(EXISTS "${imgui_archive}")
            file(SHA256 "${imgui_archive}" imgui_hash)
        endif()
        if(NOT imgui_hash STREQUAL imgui_sha256)
            if(imgui_hash)
                file(REMOVE "${imgui_archive}")
            endif()
            if(imgui_code EQUAL 0)
                set(imgui_problem
                    "${imgui_url} is not Dear ImGui 1.93.0's archive (its SHA-256 is ${imgui_got})")
            else()
                set(imgui_problem "Dear ImGui 1.93.0 could not be downloaded from ${imgui_url} (${imgui_status})")
            endif()
            message(WARNING "${imgui_problem}: RayTracingViewer is built without its panel; everything else builds. "
                            "Get v1.93.0.tar.gz from github.com/ocornut/imgui/releases/tag/v1.93.0 by hand "
                            "(SHA-256 ${imgui_sha256}) and configure with "
                            "-DRAYTRACING_IMGUI_URL=file:///<path to it>, or turn the panel off with "
                            "-DRAYTRACING_PANEL=OFF.")
        endif()
    endif()
endif()

if(RAYTRACING_PANEL AND EXISTS "${imgui_archive}")
    include(FetchContent)
    # The archive has no CMake project of its own: FetchContent only unpacks it, and the library is declared here
    FetchContent_Declare(imgui URL "${imgui_archive}" URL_HASH SHA256=${imgui_sha256})
    FetchContent_MakeAvailable(imgui)
    # The core without its demo and debug windows, and the two backends: SDL 3 for the input, SDL's renderer for the
    # drawing (the one the window already draws its image with)
    add_library(imgui STATIC
        "${imgui_SOURCE_DIR}/imgui.cpp" "${imgui_SOURCE_DIR}/imgui_draw.cpp" "${imgui_SOURCE_DIR}/imgui_tables.cpp"
        "${imgui_SOURCE_DIR}/imgui_widgets.cpp" "${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer3.cpp")
    # SYSTEM: its headers raise no warnings in our code. The definitions are part of its interface: every file that
    # includes imgui.h must see the same ones.
    target_include_directories(imgui SYSTEM PUBLIC "${imgui_SOURCE_DIR}" "${imgui_SOURCE_DIR}/backends")
    target_compile_definitions(imgui PUBLIC IMGUI_DISABLE_OBSOLETE_FUNCTIONS IMGUI_DISABLE_DEMO_WINDOWS
        IMGUI_DISABLE_DEBUG_TOOLS)
    target_link_libraries(imgui PUBLIC SDL3::SDL3-static)
    # Third-party code: its warnings off, and no scan for modules it doesn't have
    target_compile_options(imgui PRIVATE /W0)
    set_target_properties(imgui PROPERTIES CXX_SCAN_FOR_MODULES OFF)

    # The panel: what it shows and asks for (viewer_panel, Dear ImGui alone, which ViewerTests draws without a window),
    # and its place in the window (viewer_panel_host: SDL's events in, SDL's renderer out)
    target_sources(RayTracingViewer PRIVATE FILE_SET CXX_MODULES FILES
        src/viewer_panel.ixx src/viewer_panel_host.ixx)
    target_link_libraries(RayTracingViewer PRIVATE imgui)
else()
    if(NOT RAYTRACING_PANEL)
        message(STATUS "RayTracingViewer's panel is off (RAYTRACING_PANEL)")
    endif()
    # The window without the panel: the same module, viewer_panel_host, saying that this build has none
    target_sources(RayTracingViewer PRIVATE FILE_SET CXX_MODULES FILES src/viewer_panel_absent.ixx)
endif()
