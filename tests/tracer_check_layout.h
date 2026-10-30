// The tracer check's results, as __raygen__check (tests/optix_check_programs.cu, the check module) writes them and suite
// optix-tracer (tests/optix_tests.cu) reads them: one list for both, so the two can't drift apart. Plain C++, no device
// code: nvcc includes it in the module's PTX and in the suite's translation unit alike.
#pragma once

namespace tracer_check
{
// params.test.counts, the tallies of one launch
enum counter : int
{
    camera_rays,            // closest() queries at vertex 0: one per sample, width × height × samples in all
    closest_queries,        // every closest(): both tracers on the same ray, spawn and random numbers
    same,                   // index, entering, t's bits and the generator left behind all equal
    ulp,                    // index, entering and the generator equal, |Δt| ≤ 1e-5·t: the last bits of a contraction
    mismatches,             // anything else
    ties,                   // of the mismatches: another primitive at the very same t (coplanar faces)
    numbers_only,           // of the mismatches: the same hit, bit for bit, and another generator left (the media)
    shadow_queries,         // every visibility() pair
    visibility_differences, // the two floats differ in any bit
    escaped,                // INFO: closest() queries the BVH answered with a miss
    particles,              // INFO: ... with a particle of smoke
    own_sphere_inward,      // INFO: ... of a ray going into the sphere it leaves (glass: the far root)
    shadows_blocked,        // INFO: shadow queries the BVH found blocked (visibility 0)
    mismatch_claimed,       // 1 once the first mismatch is written into values (an example for the diagnosis)
    visibility_claimed,     // the same for the first visibility difference
    counter_count,
};

// params.test.values
enum value : int
{
    largest_ulp_ratio = 0,   // the largest |Δt|/t of the ulp category (atomicMax on the bits of floats ≥ 0)
    mismatch_example = 1,    // 13 floats: i, j, sample, vertex, from, inward; the BVH's index, t, entering; OptiX's
                             // index, t, entering; 1 if the generators left behind are equal
    visibility_example = 14, // 10 floats: i, j, sample, vertex, from, inward, skip, t_max; the BVH's and OptiX's floats
    value_count = 24,
};
}
