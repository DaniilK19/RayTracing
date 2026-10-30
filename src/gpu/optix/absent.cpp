// start_optix.h in a build without OptiX's headers (RAYTRACING_OPTIX off, or the download failed when CMake
// configured): the programs ask optix_built() before CUDA starts and refuse --optix, --denoise and the mesh scenes in
// their own words; start_optix() says why for those that ask it anyway (GpuTests' fixture: its suites SKIP).
#include "../start_optix.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace gpu
{
bool optix_built()
{
    return false;
}

std::shared_ptr<optix_runtime> start_optix()
{
    throw std::runtime_error("this build has no OptiX: CMake found no OptiX 9 headers when it configured (see "
                             "docs/building.md)");
}

// No runtime can exist in this build: start_optix never returns one
std::string describe(const optix_runtime& /*runtime*/)
{
    return "no OptiX";
}

bool optix_cache_enabled(const optix_runtime& /*runtime*/)
{
    return false;
}
}
