# What RayTracingViewer gains after cmake/sdl3.cmake, which stays as Step 70 left it (scripts/verify/frozen.ps1): a file
# of its own, included last, which touches the viewer's target only where the viewer is built.

# ViewerTests: the window's logic that needs neither a window nor a GPU (tests/viewer_tests.cpp): the viewer's modules,
# its camera's moves and its scripts, checked on the CPU. The viewer's target owns those module files (a program's
# modules can't be imported from another target), so the test compiles copies of its own from the same sources: what
# it tests is what the viewer is built from. ctest runs it.
if(TARGET RayTracingViewer)
    add_executable(ViewerTests tests/viewer_tests.cpp src/utf8.manifest)
    target_sources(ViewerTests PRIVATE FILE_SET CXX_MODULES FILES src/viewer_camera.ixx src/viewer_script.ixx)
    target_link_libraries(ViewerTests PRIVATE gpu_scene)
    target_compile_options(ViewerTests PRIVATE /W4)
    add_test(NAME viewer_tests COMMAND ViewerTests)
endif()
