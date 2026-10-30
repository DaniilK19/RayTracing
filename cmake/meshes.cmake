# Triangle meshes on the host: module meshes (src/meshes.ixx: the model types, meshes made in code, placements) and
# module gltf (src/gltf.ixx: glTF 2.0 files read with cgltf), compiled by cl.exe alone: no CUDA, no OptiX, not the CPU
# renderer. So the library and its test program, MeshTests, build and run on any computer with Visual Studio, and the
# GPU programs get the modules through gpu_scene. This file is included after everything else (outside the CUDA
# block), and touches the targets above only by linking gpu_scene to the library when the GPU programs exist.
# Third-party code (SYSTEM: its warnings stay out of /W4):
#   cgltf/cgltf.h  - cgltf v1.15 (MIT), github.com/jkuhlmann/cgltf, tag v1.15: tag object bbeb5b0, commit 360db1a;
#                    gltf.ixx compiles its implementation
#   stb/stb_image.h - the CPU renderer's copy (see CMakeLists.txt), compiled a second time by gltf.ixx, privately
add_library(mesh_io STATIC)
target_sources(mesh_io PUBLIC FILE_SET CXX_MODULES FILES src/meshes.ixx src/gltf.ixx)
target_include_directories(mesh_io SYSTEM PRIVATE external/cgltf external/stb)
target_link_libraries(mesh_io PUBLIC std_module)
target_compile_options(mesh_io PRIVATE /W4)

# MeshTests: the loader and the generators checked on the CPU (tests/mesh_tests.cpp), no GPU needed. Its glTF files
# are written by the test itself, small; ctest runs it.
add_executable(MeshTests tests/mesh_tests.cpp src/utf8.manifest)
target_link_libraries(MeshTests PRIVATE mesh_io)
target_compile_options(MeshTests PRIVATE /W4)
add_test(NAME mesh_tests COMMAND MeshTests)

if(TARGET gpu_scene)
    target_link_libraries(gpu_scene PUBLIC mesh_io)
endif()
