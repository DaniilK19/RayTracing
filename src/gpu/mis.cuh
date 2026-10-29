// Multiple importance sampling: the weights that combine a light sample and the material's own sample of the same
// light, so that each counts where it is the better estimator and their sum stays unbiased (Veach 1997, chapter 9).
//
// At every vertex with a density the path takes two samples of the light arriving there: a direction the light sampler
// draws toward a light (light_sampler.cuh), and one the material draws (bsdf.cuh), which is also the path's next ray
// and counts the light it happens to find. Veach's multi-sample estimator (thesis eq. 9.4, p. 260, one sample of each)
// adds both, each weighted:
//
//   F = w_L(ω_L)·f(ω_L)·Le(ω_L)·V / p_L(ω_L) + w_B(ω_B)·f(ω_B)·Le(ω_B) / p_B(ω_B)
//
// It is unbiased whatever the weights, as long as (W1) w_L + w_B = 1 wherever f·Le ≠ 0 and both techniques can draw
// the direction, and (W2) a technique that can't draw it (density 0) gets weight 0. The books' -p mixture is the
// one-sample model instead, one direction from half of one density and half of the other, which its balance weights
// already combine as well as any weights could (Veach's theorem 9.4): what two samples gain is that the light is
// sampled at every bounce, and the material's sample too.
#pragma once

#include "renderer.h"
#include "vec3.cuh"

namespace gpu
{
// Veach's power heuristic with β = 2, self²/(self² + other²): near 1 where this technique's density dominates, so a
// sample the other technique draws far more often counts little (a lamp a matte bounce finds from far away), and the
// variance of the sum stays close to the better technique's (Veach, section 9.2.4). Computed as 1/(1 + r²) with
// r = other/self, so no density is squared: a quad seen nearly edge on has the density d²/(cos θ·A), which can pass
// √FLT_MAX ≈ 1.8·10^19, and its square would be infinite. The limits: self not > 0 (0, negative, NaN: this technique
// can't draw the direction, W2) gives 0; other = ∞ gives 1/(1 + ∞) = 0; other = 0 gives 1; equal densities give
// exactly 1/2. A NaN other (or both infinite) gives 0: the weight is never NaN and always in [0, 1].
__device__ inline float power_weight(float self, float other)
{
    if (!(self > 0.0f))
    {
        return 0.0f;
    }
    const float r = other / self;
    const float w = 1.0f / fma(r, r, 1.0f);
    return w == w ? w : 0.0f; // only a NaN differs from itself
}

// The weight of a light sample of density p_light (the light sampler's, its pmf included) whose direction the material
// would draw with density p_bsdf. Under -p nee the light sample counts in full and the material's ray gives way to it
// (bsdf_weight below); -p material takes no light sample.
__device__ inline float light_weight(light_combination c, float p_light, float p_bsdf)
{
    return c == light_combination::power_heuristic ? power_weight(p_light, p_bsdf) : 1.0f;
}

// The weight of light a ray of the material's found. The ray was drawn with density p_bsdf: 0 after the camera, a delta
// vertex or a vertex that took no light sample, where no light sample competes and the light counts in full. A light
// sample could have drawn its direction with density p_light: 0 if none could (a lamp outside the light set, the
// camera's sky), and the light counts in full too. Under -p material no light sample is taken, so the material's ray
// counts in full whatever the densities. Under -p nee the light sample alone counts wherever it could have drawn the
// direction: 0 here, 1 there, W1 again.
__device__ inline float bsdf_weight(light_combination c, float p_bsdf, float p_light)
{
    if (c == light_combination::bsdf_only || !(p_bsdf > 0.0f) || !(p_light > 0.0f))
    {
        return 1.0f;
    }
    return c == light_combination::power_heuristic ? power_weight(p_bsdf, p_light) : 0.0f;
}
}
