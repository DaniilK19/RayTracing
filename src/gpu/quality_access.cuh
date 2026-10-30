// The quality kernel's view of its scene, for the code that traces its paths elsewhere (OptiX's programs, the
// denoiser's guide kernel): one accessor, so quality_scene stays private to quality.cu. Device types: nvcc
// translation units only, never renderer.cu (quality.cuh's rule).
#pragma once
#include "quality.cuh"
#include "quality_view.cuh"

namespace gpu::detail
{
quality_view quality_view_of(const quality_scene& q, const scene_view<float>& base); // quality.cu: q.view(base)
}
