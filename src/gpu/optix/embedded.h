// The OptiX modules inside the programs: their PTX as arrays, written at build time by cmake/embed.cmake from the
// modules nvcc compiled (optix.cmake). The render module (programs.cu) is gpu_optix's, for RayTracingGPU and
// RayTracingViewer; the check module (tests/optix_check_programs.cu) is GpuTests' alone (tests/optix_check_absent.cpp
// defines it empty in a build without OptiX). OptiX compiles a module's PTX again for the GPU it runs on.
#pragma once
#include <cstddef>

extern "C" const unsigned char optix_programs_ptx[]; // the PTX's bytes, without a terminating zero
extern "C" const std::size_t optix_programs_ptx_size;
extern "C" const char optix_programs_ptx_sha256[];   // their SHA-256, 64 lowercase hex digits

extern "C" const unsigned char optix_check_ptx[]; // GpuTests only
extern "C" const std::size_t optix_check_ptx_size;
extern "C" const char optix_check_ptx_sha256[];
