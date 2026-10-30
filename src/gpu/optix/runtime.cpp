// OptiX's host side for the whole program (runtime_core.h): optixInit and the context, the render module from the PTX
// embedded in the program, its program groups, one pipeline over them with its stack sizes, the SBT, the launches, and
// GpuTests' test pipelines. Compiled by cl.exe, so optix.h takes its host branch here (core.h); scene.cu, compiled by
// nvcc, fills the device structs these launches read.
#include "core.h"
static_assert(OPTIX_VERSION >= 90000); // NVIDIA's optix.h, not a project header of that name

#pragma warning(push, 0)
#include <optix_function_table_definition.h> // OptiX's table of functions: defined in this translation unit only
#include <optix_stack_size.h>
#pragma warning(pop)

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "embedded.h"

namespace gpu::optix_core
{
void check_optix(OptixResult result, const char* call)
{
    if (result != OPTIX_SUCCESS)
    {
        throw std::runtime_error(
            std::format("{}: {} ({})", call, optixGetErrorString(result), optixGetErrorName(result)));
    }
}

namespace
{
// The bytes of a log OptiX writes when it compiles (module, program groups, pipeline): its first lines are the message
constexpr std::size_t log_bytes = 4096;
// How many of its lines an error quotes
constexpr int log_lines = 10;

// Why optixInit failed, in the driver's terms. Never through optixGetErrorName: after a failed optixInit optix_stubs.h
// has zeroed the table it would ask. The version is the headers' (OPTIX_VERSION, 90000 for 9.0): optixInit asks the
// driver for their ABI.
std::string init_problem(OptixResult result)
{
    switch (result)
    {
    case OPTIX_ERROR_LIBRARY_NOT_FOUND:
        return "the graphics driver has no OptiX";
    case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION:
    case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH:
    case OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND:
        return std::format("the graphics driver's OptiX is older than OptiX {}.{}", OPTIX_VERSION / 10000,
                           OPTIX_VERSION % 10000 / 100);
    default:
        return std::format("OptiX could not start (error {})", static_cast<int>(result));
    }
}

// Why the programs can't run: a module, a program group or the pipeline that didn't compile, with the first lines of
// its log
std::string compile_problem(OptixResult result, const std::string& log)
{
    std::string text = std::format("OptiX could not compile its programs: {}", optixGetErrorName(result));
    // The log is a C string inside the buffer (its last byte stays 0: OptiX is given one less)
    std::string_view rest(log.c_str());
    for (int line = 0; line < log_lines && !rest.empty(); ++line)
    {
        const std::size_t end = std::min(rest.find('\n'), rest.size());
        text += std::format("\n  {}", rest.substr(0, end));
        rest.remove_prefix(std::min(end + 1, rest.size()));
    }
    return text;
}

// The context's messages to stderr: levels 1 and 2, fatal and error, and in validation mode level 3, warning, so that
// what validation finds is never filtered out (logCallbackLevel asks for no others; 4, print, is OptiX's chatter). A
// function OptiX calls: it throws nothing.
void log_message(unsigned int level, const char* tag, const char* message, void* /*data*/)
{
    const char* kind = level == 1 ? "fatal error" : level == 2 ? "error" : "warning";
    std::fprintf(stderr, "OptiX %s (%s): %s\n", kind, tag, message);
}

// Validation mode (OptiX's own checks of every call, slow) for a whole process: RAYTRACING_OPTIX_VALIDATION=1, read
// when the context is made, as the context's option it is
bool validation_asked()
{
    char* value = nullptr;
    std::size_t length = 0;
    const bool asked = _dupenv_s(&value, &length, "RAYTRACING_OPTIX_VALIDATION") == 0 && value != nullptr &&
                       std::string_view(value) == "1";
    std::free(value);
    return asked;
}

// The compile options, fixed in code (never from NDEBUG): Debug's programs are Release's. Every struct starts zeroed
// and is filled by name: 9.1 inserted members in the middle of some.
OptixModuleCompileOptions module_options()
{
    OptixModuleCompileOptions options = {};
    options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE; // FULL would test another module than the one that ships
    return options;                                      // no bound values, no payload types
}

OptixPipelineCompileOptions pipeline_options()
{
    OptixPipelineCompileOptions options = {};
    options.usesMotionBlur = 0; // a moving sphere is the intersection program's business, at the time in payload 3
    // The books' GAS alone as the root, or an IAS over it and the meshes' GASes
    options.traversableGraphFlags = static_cast<unsigned int>(
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING);
    options.numPayloadValues = 5;   // optix_tracer.cuh's payload: from, inward, skip, the ray's time, 1/|d|²
    options.numAttributeValues = 2; // the triangles' barycentrics; the custom hits report none
    options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    options.pipelineLaunchParamsVariableName = "params";
    options.usesPrimitiveTypeFlags =
        static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    options.allowOpacityMicromaps = 0;
    options.allowClusteredGeometry = 0;
#if OPTIX_VERSION >= 90100
    options.pipelineLaunchParamsSizeInBytes = params_bytes();
#endif
    return options;
}

OptixModule create_module(core& c, std::span<const unsigned char> ptx)
{
    const OptixModuleCompileOptions module_compile = module_options();
    const OptixPipelineCompileOptions pipeline_compile = pipeline_options();
    std::string log(log_bytes, '\0');
    std::size_t log_size = log.size() - 1;
    OptixModule module = nullptr;
    // Plain optixModuleCreate, no task pool: the compile runs on this one thread
    const OptixResult result =
        optixModuleCreate(c.context, &module_compile, &pipeline_compile, reinterpret_cast<const char*>(ptx.data()),
                          ptx.size(), log.data(), &log_size, &module);
    if (result != OPTIX_SUCCESS)
    {
        throw std::runtime_error(compile_problem(result, log));
    }
    return module;
}

OptixProgramGroupDesc raygen_group(OptixModule module, const char* entry)
{
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = module;
    desc.raygen.entryFunctionName = entry;
    return desc;
}

// A miss group without a program: optixTraverse invokes none, and the SBT needs its record all the same
OptixProgramGroupDesc miss_group()
{
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    return desc;
}

// A hit group of an intersection program alone (custom primitives), or of none (the triangles: OptiX intersects them
// itself, and optixTraverse invokes no closest-hit or any-hit program)
OptixProgramGroupDesc hit_group(OptixModule module, const char* intersection)
{
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc.hitgroup.moduleIS = module;
    desc.hitgroup.entryFunctionNameIS = intersection;
    return desc;
}

OptixProgramGroup create_group(core& c, const OptixProgramGroupDesc& desc)
{
    const OptixProgramGroupOptions options = {};
    std::string log(log_bytes, '\0');
    std::size_t log_size = log.size() - 1;
    OptixProgramGroup group = nullptr;
    const OptixResult result = optixProgramGroupCreate(c.context, &desc, 1, &options, log.data(), &log_size, &group);
    if (result != OPTIX_SUCCESS)
    {
        throw std::runtime_error(compile_problem(result, log));
    }
    return group;
}

// The stack sizes: every group's accumulated, then a trace depth of 1 (raygen → traversal, no closest-hit
// recursion), no callables, a traversable graph of depth 2 (an IAS over GASes). Returns the continuation stack's bytes.
int set_stack_sizes(OptixPipeline pipeline, std::span<const OptixProgramGroup> groups)
{
    OptixStackSizes sizes = {};
    for (OptixProgramGroup group : groups)
    {
        check_optix(optixUtilAccumulateStackSizes(group, &sizes, pipeline), "reading a program group's stack sizes");
    }
    unsigned int from_traversal = 0;
    unsigned int from_state = 0;
    unsigned int continuation = 0;
    check_optix(optixUtilComputeStackSizes(&sizes, 1, 0, 0, &from_traversal, &from_state, &continuation),
                "computing OptiX's stack sizes");
    check_optix(optixPipelineSetStackSize(pipeline, from_traversal, from_state, continuation, 2),
                "setting OptiX's stack sizes");
    return static_cast<int>(continuation);
}

// One pipeline over the groups, with its stack sizes
OptixPipeline create_pipeline(core& c, std::span<const OptixProgramGroup> groups, int& continuation_stack)
{
    const OptixPipelineCompileOptions compile = pipeline_options();
    OptixPipelineLinkOptions link = {};
    link.maxTraceDepth = 1;
    std::string log(log_bytes, '\0');
    std::size_t log_size = log.size() - 1;
    OptixPipeline pipeline = nullptr;
    const OptixResult result = optixPipelineCreate(c.context, &compile, &link, groups.data(),
                                                   static_cast<unsigned int>(groups.size()), log.data(), &log_size,
                                                   &pipeline);
    if (result != OPTIX_SUCCESS)
    {
        throw std::runtime_error(compile_problem(result, log));
    }
    try
    {
        continuation_stack = set_stack_sizes(pipeline, groups);
    }
    catch (...)
    {
        optixPipelineDestroy(pipeline);
        throw;
    }
    return pipeline;
}

constexpr auto record_bytes = static_cast<unsigned int>(OPTIX_SBT_RECORD_HEADER_SIZE);

// SBT records of headers alone, one per group, in order: the programs read everything from params, so no record holds
// data (and 32 bytes keep OPTIX_SBT_RECORD_ALIGNMENT's 16)
device_array<unsigned char> pack_records(std::span<const OptixProgramGroup> groups)
{
    std::vector<unsigned char> headers(groups.size() * record_bytes);
    for (std::size_t i = 0; i < groups.size(); ++i)
    {
        check_optix(optixSbtRecordPackHeader(groups[i], headers.data() + i * record_bytes),
                    "packing a record of OptiX's shader binding table");
    }
    return device_array<unsigned char>(headers);
}

// The table over records laid out as `raygens` raygen records, the miss record, then the four hit groups: the books'
// sphere, moving sphere and quad (the books' GAS's sbtIndexOffset 0, 1, 2) and the triangles' (a mesh instance's
// sbtOffset 3). raygenRecord is the first raygen's.
OptixShaderBindingTable binding_table(const device_array<unsigned char>& records, unsigned int raygens)
{
    const auto base = reinterpret_cast<CUdeviceptr>(records.data());
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = base;
    sbt.missRecordBase = base + CUdeviceptr{raygens} * record_bytes;
    sbt.missRecordStrideInBytes = record_bytes;
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = base + CUdeviceptr{raygens + 1} * record_bytes;
    sbt.hitgroupRecordStrideInBytes = record_bytes;
    sbt.hitgroupRecordCount = 4;
    return sbt;
}

// The render module's objects, newest first, as compile_programs left them (also after it failed halfway)
void release_programs(core& c) noexcept
{
    if (c.pipeline)
    {
        optixPipelineDestroy(c.pipeline);
        c.pipeline = nullptr;
    }
    for (OptixProgramGroup& group : c.groups)
    {
        if (group)
        {
            optixProgramGroupDestroy(group);
        }
    }
    c.groups.clear();
    if (c.module)
    {
        optixModuleDestroy(c.module);
        c.module = nullptr;
    }
    c.records = device_array<unsigned char>(0); // the old records go with the temporary
    c.sbt = {};
    c.continuation_stack = 0;
}

void release_test(test_pipeline& test) noexcept
{
    if (test.pipeline)
    {
        optixPipelineDestroy(test.pipeline);
    }
    if (test.probe)
    {
        optixProgramGroupDestroy(test.probe);
    }
    if (test.raygen)
    {
        optixProgramGroupDestroy(test.raygen);
    }
    test = test_pipeline{};
}

// GpuTests' pipeline of a raygen of another module: made at its first launch, kept until the core goes. It links the
// render module's own groups, so a check tests the intersection programs that ship (a pipeline may combine groups of
// modules compiled with the same pipeline options, and both modules declare `params` of the same type), or under
// probe_intersections the test module's __intersection__probe in their place (P7, Step 91).
test_pipeline& test_pipeline_of(core& c, std::span<const unsigned char> ptx, const char* entry,
                                bool probe_intersections)
{
    const auto key = std::make_tuple(ptx.data(), std::string(entry), probe_intersections);
    if (const auto found = c.tests.find(key); found != c.tests.end())
    {
        return found->second;
    }
    OptixModule& module = c.test_modules[ptx.data()];
    if (!module)
    {
        module = create_module(c, ptx);
    }
    test_pipeline test;
    try
    {
        test.raygen = create_group(c, raygen_group(module, entry));
        if (probe_intersections)
        {
            test.probe = create_group(c, hit_group(module, "__intersection__probe"));
        }
        const OptixProgramGroup sphere = probe_intersections ? test.probe : c.groups[hit_sphere];
        const OptixProgramGroup moving_sphere = probe_intersections ? test.probe : c.groups[hit_moving_sphere];
        const OptixProgramGroup quad = probe_intersections ? test.probe : c.groups[hit_quad];
        // The pipeline links each group once; the records may name one group several times
        std::vector<OptixProgramGroup> linked{test.raygen, c.groups[miss], sphere};
        if (!probe_intersections)
        {
            linked.push_back(moving_sphere);
            linked.push_back(quad);
        }
        linked.push_back(c.groups[hit_triangles]);
        int continuation_stack = 0;
        test.pipeline = create_pipeline(c, linked, continuation_stack);
        const std::array records{test.raygen, c.groups[miss], sphere, moving_sphere, quad, c.groups[hit_triangles]};
        test.records = pack_records(records);
        test.sbt = binding_table(test.records, 1);
    }
    catch (...)
    {
        release_test(test);
        throw;
    }
    return c.tests.emplace(key, std::move(test)).first->second;
}
}

core_ptr open_core()
{
    // The reasons start_optix.h lists, in their order; the first, a build without OptiX's headers, is absent.cpp's.
    // The driver must run the PTX of the CUDA that compiled the programs: OptiX hands it to the driver's compiler.
    int driver = 0;
    check(cudaDriverGetVersion(&driver), "reading the graphics driver's CUDA version");
    if (driver < CUDART_VERSION)
    {
        throw std::runtime_error(
            std::format("the graphics driver is older than CUDA {}.{}, whose PTX OptiX compiles: update the driver",
                        CUDART_VERSION / 1000, CUDART_VERSION % 1000 / 10));
    }
    // optixInit loads the driver's nvoptix.dll and asks it for the functions of the headers' ABI (OPTIX_ABI_VERSION)
    if (const OptixResult init = optixInit(); init != OPTIX_SUCCESS)
    {
        throw std::runtime_error(init_problem(init));
    }
    // The context on the CUDA device the program started (query_device): its primary context, current after this
    check(cudaSetDevice(0), "selecting the CUDA device");
    core_ptr c(new core);
    const bool validation = validation_asked();
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &log_message;
    options.logCallbackLevel = validation ? 3 : 2; // validation's warnings too (log_message)
    options.validationMode =
        validation ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    check_optix(optixDeviceContextCreate(nullptr, &options, &c->context), "creating OptiX's context");
    // OptiX's disk cache of compiled programs: on unless the user turned it off or it couldn't start (it then turns
    // itself off, with a log message only), and then every start compiles the programs again
    int cache = 0;
    check_optix(optixDeviceContextGetCacheEnabled(c->context, &cache), "asking whether OptiX's disk cache is on");
    c->cache_enabled = cache != 0;
    return c;
}

// The order of destruction: the GPU done with everything first, then the denoiser, the test pipelines and their
// modules, the render pipeline, its groups and module, the SBT, and the context last (destroyed explicitly, so that
// compute-sanitizer --check-optix-leaks sees a clean exit)
void core_deleter::operator()(core* c) const noexcept
{
    (void)cudaDeviceSynchronize();
    c->denoiser.reset();
    for (auto& entry : c->tests)
    {
        release_test(entry.second);
    }
    c->tests.clear();
    for (auto& entry : c->test_modules)
    {
        if (entry.second)
        {
            optixModuleDestroy(entry.second);
        }
    }
    c->test_modules.clear();
    release_programs(*c);
    if (c->context)
    {
        optixDeviceContextDestroy(c->context);
    }
    delete c;
}

std::string programs_sha256()
{
    return optix_programs_ptx_sha256;
}

int headers_version()
{
    return OPTIX_VERSION;
}

bool cache_enabled(const core& c)
{
    return c.cache_enabled;
}

void compile_programs(core& c)
{
    if (c.pipeline)
    {
        return; // once per core: the first upload (or GpuTests' compile_programs)
    }
    try
    {
        c.module = create_module(c, {optix_programs_ptx, optix_programs_ptx_size});
        c.groups.assign(group_count, nullptr);
        c.groups[raygen_render] = create_group(c, raygen_group(c.module, "__raygen__render"));
        c.groups[raygen_guides] = create_group(c, raygen_group(c.module, "__raygen__guides"));
        c.groups[raygen_pick] = create_group(c, raygen_group(c.module, "__raygen__pick"));
        c.groups[miss] = create_group(c, miss_group());
        c.groups[hit_sphere] = create_group(c, hit_group(c.module, "__intersection__sphere"));
        c.groups[hit_moving_sphere] = create_group(c, hit_group(c.module, "__intersection__moving_sphere"));
        c.groups[hit_quad] = create_group(c, hit_group(c.module, "__intersection__quad"));
        c.groups[hit_triangles] = create_group(c, hit_group(nullptr, nullptr));
        c.pipeline = create_pipeline(c, c.groups, c.continuation_stack);
        c.records = pack_records(c.groups);
        c.sbt = binding_table(c.records, 3);
    }
    catch (...)
    {
        release_programs(c); // a later call starts over
        throw;
    }
}

int continuation_stack_bytes(const core& c)
{
    return c.continuation_stack;
}

void launch(core& c, raygen entry, const void* params, std::size_t params_bytes, unsigned int width,
            unsigned int height)
{
    compile_programs(c);
    // The table is passed by value: this launch's raygen record, the others' untouched
    OptixShaderBindingTable sbt = c.sbt;
    sbt.raygenRecord += static_cast<CUdeviceptr>(entry) * record_bytes;
    check_optix(optixLaunch(c.pipeline, nullptr, reinterpret_cast<CUdeviceptr>(params), params_bytes, &sbt, width,
                            height, 1),
                "launching OptiX's programs");
    // A launch that couldn't start shows here; a program that fails while it runs, when the host waits for it
    check(cudaGetLastError(), "launching OptiX's programs");
}

void launch_test(core& c, std::span<const unsigned char> ptx, const char* entry, const void* params,
                 std::size_t params_bytes, unsigned int width, unsigned int height, bool probe_intersections)
{
    compile_programs(c);
    test_pipeline& test = test_pipeline_of(c, ptx, entry, probe_intersections);
    check_optix(optixLaunch(test.pipeline, nullptr, reinterpret_cast<CUdeviceptr>(params), params_bytes, &test.sbt,
                            width, height, 1),
                "launching a test program of OptiX");
    check(cudaGetLastError(), "launching a test program of OptiX");
}
}
