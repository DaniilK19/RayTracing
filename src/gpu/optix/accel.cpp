// OptiX's acceleration structures (runtime_core.h): the books' custom-primitive GAS, the meshes' triangle GASes and the
// IAS over them. Every build runs on stream 0 and waits; temp, AABB and instance buffers are freed before it returns.
// Compiled by cl.exe with OptiX's host API (core.h). Two regions, one per track: neither edits the other's lines.
#include "core.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
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

namespace
{
// An OptiX call of the meshes' builds: its error as std::runtime_error, with what the call was for and OptiX's name
// and description of the error (runtime.cpp's own check is its file's)
void check_mesh_build(OptixResult result, const char* what)
{
    if (result != OPTIX_SUCCESS)
    {
        throw std::runtime_error(std::string(what) + ": " + optixGetErrorString(result) + " (" +
                                 optixGetErrorName(result) + ")");
    }
}

// The device memory an acceleration structure lives in, freed with the last copy of its accel. cudaMalloc's addresses
// are 256-byte aligned, more than the 128 OptiX asks of an output buffer (OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT). If the
// shared_ptr can't be made, it frees the memory itself.
std::shared_ptr<void> structure_memory(std::size_t bytes)
{
    void* memory = nullptr;
    check(cudaMalloc(&memory, bytes), "allocating GPU memory for an acceleration structure");
    return {memory, [](void* p) { cudaFree(p); }};
}

// No any-hit program anywhere: a triangle is opaque, and OptiX keeps its whole traversal on the hardware
constexpr unsigned int mesh_geometry_flags[1] = {static_cast<unsigned int>(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT)};

// A mesh's GAS: built for the fastest traversal, and compactable, since the size OptiX plans for is an upper bound
// that the finished structure often stays well below (OptiX Programming Guide §5.9)
OptixAccelBuildOptions mesh_build_options()
{
    OptixAccelBuildOptions options = {};
    options.buildFlags =
        static_cast<unsigned int>(OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    return options;
}

// One mesh as OptiX's build input: float3 vertices 12 bytes apart, uint3 triangles that index them, one SBT record
// (the triangles' empty hit group, at a mesh instance's sbtOffset 3). `vertices` holds the vertex buffer's address:
// OptiX takes an array of them, one per motion key, and reads it during the call it is passed to.
OptixBuildInput mesh_input(const triangle_mesh& mesh, const CUdeviceptr& vertices)
{
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    OptixBuildInputTriangleArray& triangles = input.triangleArray;
    triangles.vertexBuffers = &vertices;
    triangles.numVertices = mesh.vertex_count;
    triangles.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangles.vertexStrideInBytes = 12;
    triangles.indexBuffer = reinterpret_cast<CUdeviceptr>(mesh.triangles);
    triangles.numIndexTriplets = mesh.triangle_count;
    triangles.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangles.indexStrideInBytes = 12;
    triangles.flags = mesh_geometry_flags;
    triangles.numSbtRecords = 1;
    return input;
}

// What each mesh's build needs: optixAccelComputeMemoryUsage starts nothing on the GPU and reads neither the device
// pointers nor what they point to ("not required to point to allocated memory": the OptiX Programming Guide, chapter
// 5's introduction), so the memory plan asks it before anything is allocated
std::vector<OptixAccelBufferSizes> mesh_sizes(core& c, std::span<const triangle_mesh> meshes)
{
    const OptixAccelBuildOptions options = mesh_build_options();
    std::vector<OptixAccelBufferSizes> sizes(meshes.size());
    for (std::size_t i = 0; i < meshes.size(); ++i)
    {
        const auto vertices = reinterpret_cast<CUdeviceptr>(meshes[i].positions);
        const OptixBuildInput input = mesh_input(meshes[i], vertices);
        check_mesh_build(optixAccelComputeMemoryUsage(c.context, &options, &input, 1, &sizes[i]),
                         "planning a mesh's acceleration structure");
    }
    return sizes;
}

// The IAS's build: traced for the scene's whole life, so for the fastest trace; built once, never updated
OptixAccelBuildOptions instance_build_options()
{
    OptixAccelBuildOptions options = {};
    options.buildFlags = static_cast<unsigned int>(OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    return options;
}

// `count` OptixInstance records at `records` as the IAS's one build input
OptixBuildInput instance_input(CUdeviceptr records, unsigned int count)
{
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = records;
    input.instanceArray.numInstances = count;
    return input;
}
}

// The most the builds hold at once: every mesh's structure at its planned size, and the larger of the shared temp
// buffer (while they are built) and one compacted copy (while they are compacted, each full-size one freed after its
// copy), and the compacted sizes OptiX writes
std::size_t triangles_bytes(core& c, std::span<const triangle_mesh> meshes)
{
    std::size_t outputs = 0;
    std::size_t largest_output = 0;
    std::size_t temp = 0;
    for (const OptixAccelBufferSizes& sizes : mesh_sizes(c, meshes))
    {
        outputs += sizes.outputSizeInBytes;
        largest_output = std::max(largest_output, sizes.outputSizeInBytes);
        temp = std::max(temp, sizes.tempSizeInBytes);
    }
    return outputs + std::max(temp, largest_output) + meshes.size() * sizeof(std::uint64_t);
}

namespace
{
// The sizes of build_instances' IAS of `count` instances, from optixAccelComputeMemoryUsage on an input of records
// that don't exist yet (it reads no device memory): they depend on the count alone
OptixAccelBufferSizes instance_sizes(core& c, unsigned int count)
{
    const OptixAccelBuildOptions options = instance_build_options();
    const OptixBuildInput input = instance_input(0, count);
    OptixAccelBufferSizes sizes = {};
    check_mesh_build(optixAccelComputeMemoryUsage(c.context, &options, &input, 1, &sizes),
                     "planning the instances' acceleration structure");
    return sizes;
}
}

// What build_instances holds at once: the records' device copy, the structure and its temp
std::size_t instances_bytes(core& c, unsigned int count)
{
    if (count == 0)
    {
        return 0;
    }
    const OptixAccelBufferSizes sizes = instance_sizes(c, count);
    return sizes.outputSizeInBytes + sizes.tempSizeInBytes + std::size_t{count} * sizeof(OptixInstance);
}

// What stays once build_instances returns: the structure alone (its records and temp freed)
std::size_t instance_structure_bytes(core& c, unsigned int count)
{
    return count == 0 ? 0 : instance_sizes(c, count).outputSizeInBytes;
}

// The meshes' GASes, built back to back on stream 0 on one temp buffer as large as the largest needs (the builds run
// one after another on the stream, so none reads it while another writes it), each emitting its compacted size. One
// wait for them all, the temp freed, then each structure whose compacted size is smaller is copied into a buffer of
// that size (optixAccelCompact) and the full-size one freed once the copy is done.
std::vector<accel> build_triangles(core& c, std::span<const triangle_mesh> meshes)
{
    const OptixAccelBuildOptions options = mesh_build_options();
    const std::vector<OptixAccelBufferSizes> sizes = mesh_sizes(c, meshes);
    std::size_t temp_bytes = 0;
    for (const OptixAccelBufferSizes& s : sizes)
    {
        temp_bytes = std::max(temp_bytes, s.tempSizeInBytes);
    }
    // Declared before the try: if a call throws, the catch waits for stream 0 before these are freed
    std::vector<accel> built(meshes.size());
    device_array<unsigned char> temp(temp_bytes); // 256-byte aligned, as OptiX's 128 asks
    device_array<std::uint64_t> compacted(meshes.size());
    try
    {
        for (std::size_t i = 0; i < meshes.size(); ++i)
        {
            built[i].memory = structure_memory(sizes[i].outputSizeInBytes);
            built[i].bytes = sizes[i].outputSizeInBytes;
            const auto vertices = reinterpret_cast<CUdeviceptr>(meshes[i].positions);
            const OptixBuildInput input = mesh_input(meshes[i], vertices);
            OptixAccelEmitDesc emit = {};
            emit.result = reinterpret_cast<CUdeviceptr>(compacted.data() + i);
            emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            OptixTraversableHandle handle = 0;
            check_mesh_build(optixAccelBuild(c.context, nullptr, &options, &input, 1,
                                             reinterpret_cast<CUdeviceptr>(temp.data()), temp.size(),
                                             reinterpret_cast<CUdeviceptr>(built[i].memory.get()), built[i].bytes,
                                             &handle, &emit, 1),
                             "building a mesh's acceleration structure");
            built[i].handle = handle;
        }
        // A synchronous copy on stream 0: it waits for every build before it reads their sizes
        const std::vector<std::uint64_t> compacted_bytes = compacted.to_host();
        temp = device_array<unsigned char>(0);
        for (std::size_t i = 0; i < built.size(); ++i)
        {
            const auto bytes = static_cast<std::size_t>(compacted_bytes[i]);
            if (bytes == 0 || bytes >= built[i].bytes)
            {
                continue; // nothing to gain: the guide's `if (compactedSize < outputSize)`
            }
            accel smaller{.handle = 0, .memory = structure_memory(bytes), .bytes = bytes};
            OptixTraversableHandle handle = 0;
            check_mesh_build(optixAccelCompact(c.context, nullptr, built[i].handle,
                                               reinterpret_cast<CUdeviceptr>(smaller.memory.get()), bytes, &handle),
                             "compacting a mesh's acceleration structure");
            // The copy runs on stream 0: the full-size structure may go only once it has been read
            check(cudaStreamSynchronize(nullptr), "compacting a mesh's acceleration structure");
            smaller.handle = handle;
            built[i] = std::move(smaller);
        }
    }
    catch (...)
    {
        (void)cudaStreamSynchronize(nullptr); // a build or a copy may still read the buffers about to be freed
        throw;
    }
    return built;
}

// The IAS: instance 0 the books' GAS (if the scene has one), instance k + 1 mesh instance k, as the upload lists them.
// Each OptixInstance is zero but for what is set here, and two of its fields must not stay zero: a zero
// visibilityMask culls the instance for every ray ("if rayMask & instanceMask == 0 the instance is culled"), so it is
// 0xFF, the mask every ray carries; and DISABLE_ANYHIT, as on the GASes' inputs. The instances are copied into the
// structure, so their buffer goes with the temp once the build is done. No instance: no structure (root 0).
accel build_instances(core& c, std::span<const instance> instances)
{
    if (instances.empty())
    {
        return {};
    }
    std::vector<OptixInstance> records(instances.size());
    for (std::size_t i = 0; i < instances.size(); ++i)
    {
        OptixInstance& record = records[i];
        std::copy(std::begin(instances[i].transform), std::end(instances[i].transform), std::begin(record.transform));
        record.instanceId = instances[i].id;
        record.sbtOffset = instances[i].sbt_offset;
        record.visibilityMask = 0xFFu;
        record.flags = static_cast<unsigned int>(OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT);
        record.traversableHandle = static_cast<OptixTraversableHandle>(instances[i].handle);
    }
    // cudaMalloc's 256-byte alignment covers the 16 OptiX asks of instances (OPTIX_INSTANCE_BYTE_ALIGNMENT)
    const device_array<OptixInstance> on_device(records);
    const OptixBuildInput input =
        instance_input(reinterpret_cast<CUdeviceptr>(on_device.data()), static_cast<unsigned int>(records.size()));
    const OptixAccelBuildOptions options = instance_build_options();
    OptixAccelBufferSizes sizes = {};
    check_mesh_build(optixAccelComputeMemoryUsage(c.context, &options, &input, 1, &sizes),
                     "planning the instances' acceleration structure");
    accel built{.handle = 0, .memory = structure_memory(sizes.outputSizeInBytes), .bytes = sizes.outputSizeInBytes};
    const device_array<unsigned char> temp(sizes.tempSizeInBytes);
    try
    {
        OptixTraversableHandle handle = 0;
        check_mesh_build(optixAccelBuild(c.context, nullptr, &options, &input, 1,
                                         reinterpret_cast<CUdeviceptr>(temp.data()), temp.size(),
                                         reinterpret_cast<CUdeviceptr>(built.memory.get()), built.bytes, &handle,
                                         nullptr, 0),
                         "building the instances' acceleration structure");
        check(cudaStreamSynchronize(nullptr), "building the instances' acceleration structure");
        built.handle = handle;
    }
    catch (...)
    {
        (void)cudaStreamSynchronize(nullptr);
        throw;
    }
    return built;
}
}
