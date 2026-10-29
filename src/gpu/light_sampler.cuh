// The quality kernel's light sampling: the lamps and the environment map as one set, one of them picked uniformly per
// light sample (light_pmf), which draws a direction toward itself. The integrator (quality_path.cuh) traces a shadow
// ray along it and weighs the light against the material's own sample of the same direction (mis.cuh).
//
// A stub until light samples come: no light to sample, every density 0.
#pragma once

#include "quality_view.cuh"
#include "random.cuh"
#include "vec3.cuh"

namespace gpu
{
struct light_sample
{
    vec3<float> wi;  // unit, from x toward the light
    float distance;  // the shadow ray's t_max: to the sampled point; infinity for the environment
    color<float> le; // the radiance arriving along -wi, before the shadow ray and the media
    float pdf;       // the light technique's density of wi: light_pmf · the light's own density
    int skip;        // the lamp's global id (-1 for the environment)
};

// How many lights the set holds
__device__ inline int light_count(const quality_view& /*v*/)
{
    return 0;
}

// A light sample from x; false when the picked light can't light x (the back of a quad, the inside of a sphere)
__device__ inline bool sample_light(const quality_view& /*v*/, const point3<float>& /*x*/, pcg32& /*rng*/,
                                    light_sample& /*out*/)
{
    return false;
}

// The light technique's density (pmf included) of the direction from x along which a ray hit the lamp with this
// global id at this distance; 0: no lamp of the set
__device__ inline float lamp_hit_pdf(const quality_view& /*v*/, int /*global_id*/, const point3<float>& /*x*/,
                                     const vec3<float>& /*unit_dir*/, float /*distance*/)
{
    return 0.0f;
}

// The same for a ray that escaped toward the environment, along dir (of any length: the ray's own direction, which
// background() looks the radiance up with); 0: the map isn't sampled
__device__ inline float environment_hit_pdf(const quality_view& /*v*/, const vec3<float>& /*dir*/)
{
    return 0.0f;
}
}
