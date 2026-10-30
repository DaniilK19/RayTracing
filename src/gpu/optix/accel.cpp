// OptiX's acceleration structures (runtime_core.h): the books' custom-primitive GAS, the meshes' triangle GASes and the
// IAS over them. Every build runs on stream 0 and waits; temp, AABB and instance buffers are freed before it returns.
// Compiled by cl.exe with OptiX's host API (core.h). Two regions, one per track: neither edits the other's lines.
#include "core.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace gpu::optix_core
{
// --- The books' region (track A, Step 88): one custom-primitive build input of every primitive in global-id order, an
// sbtIndexOffset per primitive (0 sphere, 1 moving sphere, 2 quad), PREFER_FAST_TRACE, no compaction -----------------

std::size_t custom_bytes(core& /*c*/, unsigned int /*count*/)
{
    throw std::logic_error("optix_core::custom_bytes comes in Step 88");
}

accel build_custom(core& /*c*/, const aabb* /*boxes*/, const std::uint8_t* /*kinds*/, unsigned int /*count*/)
{
    throw std::logic_error("optix_core::build_custom comes in Step 88");
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
