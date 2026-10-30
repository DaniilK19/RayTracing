// The launch parameters of every OptiX program: one __constant__ variable named `params` in each module, of this type
// (OptiX requires the same size in every module of a pipeline). scene.cu fills it and copies it to the GPU before each
// optixLaunch, which reads it from device memory. No OptiX header: the traversable handle is an integer.
#pragma once
#include <cstdint>
#include "camera.cuh"
#include "mesh_tables.h"
#include "quality_view.cuh"

namespace gpu
{
struct optix_test_fields // GpuTests' check module only; zero in every other launch
{
    unsigned long long* counts;
    float* values;
    int fault;
    int samples;
    std::uint64_t seed;
};

struct optix_params
{
    camera<float> cam;
    quality_view view;          // quality_view_of(...); materials, microfacets and lamp_of lengthened in mesh scenes
    std::uint64_t root;         // an OptixTraversableHandle: the books' GAS, the IAS over it and the meshes, or 0
    mesh_tables meshes;
    std::uint64_t seed;
    int first_pass;
    int pass_count;
    int first_row;
    int row_count;
    double* sums;               // __raygen__render: 3 per pixel of the whole image
    unsigned long long* counts; // [0] rays, [1] shadow rays
    float4* albedo;             // __raygen__guides: width × height
    float4* normal;
    std::uint64_t guide_seed;
    ray<float> pick_ray;        // __raygen__pick
    float* pick_t;
    optix_test_fields test;
};
}
