// OptiX's acceleration structures (runtime_core.h): the books' custom-primitive GAS, the meshes' triangle GASes and the
// IAS over them. Every build runs on stream 0 and waits; temp, AABB and instance buffers are freed before it returns.
// Compiled by cl.exe with OptiX's host API (core.h). Two regions, one per track: neither edits the other's lines.
#include "core.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace gpu::optix_core
{
// --- The books' region (track A, Step 88): one custom-primitive build input of every primitive in global-id order, an
// sbtIndexOffset per primitive (0 sphere, 1 moving sphere, 2 quad), PREFER_FAST_TRACE, no compaction -----------------

static_assert(sizeof(aabb) == sizeof(OptixAabb) && alignof(aabb) == alignof(OptixAabb),
              "runtime_core.h's aabb has OptixAabb's layout: scene.cu's boxes go to the build as they are");

namespace
{
namespace books // the books' region's own helpers: the meshes' region keeps its own, so neither edits the other's lines
{
// The SBT records a primitive's kind picks among (scene.cu's kinds: 0 sphere, 1 moving sphere, 2 quad): the books'
// three intersection programs, the first three hit groups (runtime.cpp's binding_table)
constexpr unsigned int kind_count = 3;


// The build options: a GAS built once per session and traced for its whole life, so the trace's speed comes first;
// no update, and no compaction: a few thousand boxes, whose memory a second build couldn't give back much of
OptixAccelBuildOptions build_options()
{
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    return options;
}

// The host arrays a custom-primitive build input points to, which OptiX reads during the call that takes the input
struct custom_arrays
{
    CUdeviceptr boxes = 0; // the AABB buffer of the one motion key (none: a moving sphere's box is its whole sweep)
    // Per SBT record: no any-hit program anywhere, so the traversal never stops to call one
    std::array<unsigned int, kind_count> flags{OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
                                               OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
};

// One build input of every books' primitive in global-id order. primitiveIndexOffset 0: optixGetPrimitiveIndex() is
// the global id, which every table of the books (materials, sphere frames, lamp_of) is indexed by. A uint8
// sbtIndexOffset per primitive, its kind, picks its intersection program: every value < numSbtRecords, an empty kind
// needs nothing, and the records never shift.
OptixBuildInput custom_input(const custom_arrays& arrays, CUdeviceptr kinds, unsigned int count)
{
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    OptixBuildInputCustomPrimitiveArray& custom = input.customPrimitiveArray;
    custom.aabbBuffers = &arrays.boxes;
    custom.numPrimitives = count;
    custom.strideInBytes = 0; // packed OptixAabbs
    custom.flags = arrays.flags.data();
    custom.numSbtRecords = kind_count;
    custom.sbtIndexOffsetBuffer = kinds;
    custom.sbtIndexOffsetSizeInBytes = 1;
    custom.sbtIndexOffsetStrideInBytes = 0; // packed bytes
    custom.primitiveIndexOffset = 0;
    return input;
}

OptixAccelBufferSizes sizes_of(core& c, const OptixBuildInput& input)
{
    const OptixAccelBuildOptions options = build_options();
    OptixAccelBufferSizes sizes = {};
    check_optix(optixAccelComputeMemoryUsage(c.context, &options, &input, 1, &sizes),
                "planning the memory of the books' acceleration structure");
    return sizes;
}

// Device memory for an acceleration structure, freed with its last copy: cudaMalloc's alignment (256 bytes) is
// OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT's 128 and more. (A shared_ptr that can't be made frees the memory itself.)
std::shared_ptr<void> accel_memory(std::size_t bytes)
{
    void* memory = nullptr;
    check(cudaMalloc(&memory, bytes), "allocating GPU memory for an acceleration structure");
    return std::shared_ptr<void>(memory, [](void* p) { cudaFree(p); });
}
}
}

std::size_t custom_bytes(core& c, unsigned int count)
{
    if (count == 0)
    {
        return 0; // no books' primitive, no GAS (scene.cu's root is then 0, or an IAS of the meshes alone)
    }
    // "optixAccelComputeMemoryUsage does not initiate any activity on the device; pointers to device memory or contents
    // of input buffers are not required to point to allocated memory" (OptiX Programming Guide §5.2): the sizes alone
    const books::custom_arrays arrays;
    const OptixAccelBufferSizes sizes = books::sizes_of(c, books::custom_input(arrays, 0, count));
    return sizes.outputSizeInBytes + sizes.tempSizeInBytes;
}

accel build_custom(core& c, const aabb* boxes, const std::uint8_t* kinds, unsigned int count)
{
    if (count == 0)
    {
        return {}; // handle 0: nothing to hit
    }
    // OptiX's alignment rule for the AABB buffer (8 bytes): cudaMalloc's memory (scene.cu's) meets it, and a
    // sub-allocation must too
    if (reinterpret_cast<std::uintptr_t>(boxes) % OPTIX_AABB_BUFFER_BYTE_ALIGNMENT != 0)
    {
        throw std::logic_error("the books' boxes must start at a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT");
    }
    books::custom_arrays arrays;
    arrays.boxes = reinterpret_cast<CUdeviceptr>(boxes);
    const OptixBuildInput input = books::custom_input(arrays, reinterpret_cast<CUdeviceptr>(kinds), count);
    const OptixAccelBufferSizes sizes = books::sizes_of(c, input);
    const OptixAccelBuildOptions options = books::build_options();

    accel built;
    built.bytes = sizes.outputSizeInBytes;
    built.memory = books::accel_memory(sizes.outputSizeInBytes);
    const device_array<unsigned char> temp(sizes.tempSizeInBytes); // cudaMalloc's alignment again
    OptixTraversableHandle handle = 0;
    check_optix(optixAccelBuild(c.context, nullptr, &options, &input, 1,
                                       reinterpret_cast<CUdeviceptr>(temp.data()), sizes.tempSizeInBytes,
                                       reinterpret_cast<CUdeviceptr>(built.memory.get()), sizes.outputSizeInBytes,
                                       &handle, nullptr, 0),
                       "building the books' acceleration structure");
    // On stream 0, as the launches are queued: waited for here, so the temp memory can go, and the caller's boxes and
    // kinds, which a custom-primitive GAS doesn't keep
    check(cudaStreamSynchronize(nullptr), "building the books' acceleration structure");
    built.handle = handle;
    return built;
}

// --- The meshes' region (track B, Step 91): a triangle GAS per mesh, compacted, and the IAS over the books' GAS and
// the meshes' instances (visibilityMask 0xFF and OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT on every instance) -------------

std::size_t triangles_bytes(core& /*c*/, std::span<const triangle_mesh> /*meshes*/)
{
    throw std::logic_error("optix_core::triangles_bytes comes in Step 91");
}

std::vector<accel> build_triangles(core& /*c*/, std::span<const triangle_mesh> /*meshes*/)
{
    throw std::logic_error("optix_core::build_triangles comes in Step 91");
}

accel build_instances(core& /*c*/, std::span<const instance> /*instances*/)
{
    throw std::logic_error("optix_core::build_instances comes in Step 91");
}
}
