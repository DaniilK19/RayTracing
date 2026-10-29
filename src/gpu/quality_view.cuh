// The scene as the quality kernel sees it (quality.cu): the books' tables, exactly as the kernel for all three books
// reads them, and beside them the quality kernel's own: the materials' GGX side entries, the lamps it samples, the
// environment map. The books' structs keep their layout (scene_view, material<Real>...): the books' kernels are
// compiled from them, and a member added to one would change their code (see first_book_scene in scene.cuh).
// Device code; included by quality.cu and the tests only.
#pragma once

#include "renderer.h"
#include "scene.cuh"

namespace gpu
{
struct emitter;   // lamps.cuh: a lamp of the light set
struct env_texel; // environment_data.h: a texel of the environment map with its sampling density
struct alias_bin; // environment_data.h: a bin of the map's alias table

// A material's GGX side entry as the kernel reads it, by material index (see microfacet_desc in renderer.h)
struct microfacet
{
    microfacet_kind kind;
    float alpha;     // r², from the host
    float roughness; // r: the energy compensation table's axis
};

struct lamps_view
{
    const emitter* lamps = nullptr;
    const int* lamp_of = nullptr; // by global id of a surface primitive: its index in lamps, -1 if it is no lamp
    int count = 0;
};

struct environment_view
{
    const env_texel* texels = nullptr; // width·height {r, g, b, pdf}, row 0 = +y; nullptr: the camera's sky
    const alias_bin* bins = nullptr;   // 1 << log2_bins; nullptr: the map isn't sampled as a light
    const float* row_cos = nullptr;    // height + 1: cos(π·j/height)
    int width = 0;
    int height = 0;
    int log2_bins = 0;
    float cos_yaw = 1;
    float sin_yaw = 0;
    float scale = 1;
};

// The kernel's parameter (copied into constant memory per launch): the books' view unchanged inside, the rest beside it
struct quality_view
{
    scene_view<float> base;        // exactly the tables all_books_kernel reads (its lights list stays empty here)
    const microfacet* microfacets; // by material index; nullptr: no GGX in the scene
    lamps_view lamps;
    environment_view environment;
    float light_pmf; // 1 / (lamps.count + (environment sampled ? 1 : 0)); 0: no light to sample
    light_combination combination;
};
}
