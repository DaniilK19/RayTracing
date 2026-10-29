// Multiple importance sampling: the weights that combine a light sample and the material's own sample of the same
// light, so that each counts where it is the better estimator and their sum stays unbiased (Veach 1997, chapter 9).
//
// A stub until light samples come: every light a ray finds counts in full (the material's samples alone, -p material).
#pragma once

#include "renderer.h"

namespace gpu
{
// The weight of a light sample of density p_light whose direction the material would draw with density p_bsdf
__device__ inline float light_weight(light_combination /*c*/, float /*p_light*/, float /*p_bsdf*/)
{
    return 1.0f;
}

// The weight of light a ray of the material's found: the ray was drawn with density p_bsdf (0 after a vertex that took
// no light sample), and a light sample could have drawn its direction with density p_light (0 if none could)
__device__ inline float bsdf_weight(light_combination /*c*/, float /*p_bsdf*/, float /*p_light*/)
{
    return 1.0f;
}
}
