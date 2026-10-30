// Partition gpu_scenes:parts, the types of the GPU programs' list of scenes (see gpu_scenes.ixx). A module partition
// (C++20) is a part of one module in a file of its own: gpu_scenes re-exports it, and the files that add scenes beyond
// the books import it without importing each other.
module;

// The renderer's types (a GGX side entry), before `import std;` (see gpu_scene.ixx)
#include "gpu/renderer.h"

export module gpu_scenes:parts;

import std;
import hittable; // class material, which a mesh_ref names
import material;
import scenes;
import strategy_options;

// What a scene of the list is for. The help shows each group under a heading of its own.
export enum class scene_role
{
    book,     // the books' 13, as RayTracing renders them
    showcase, // the GPU's own scenes of light and materials beyond the books, which only the quality kernel renders
    test,     // checks of the renderers, such as a white furnace
};

// Where a scene's surroundings come from, when it has an HDR environment instead of the camera's sky
export enum class env_source
{
    none,  // the camera's sky
    asset, // an equirectangular .hdr image from the assets folder
    white, // a constant map of radiance 1, made in code (a furnace)
};

export struct environment_choice
{
    env_source source = env_source::none;
    std::string_view asset; // env_source::asset: its file name in assets/
    double yaw_degrees = 0; // turns the map about +y
    double scale = 1;       // multiplies its radiance
};

// A mesh in a scene of the list: the CPU builds no triangle, so a scene's meshes ride beside its CPU objects (the
// ground, the Cornell box's walls and lamp stay the books' shapes)
export struct mesh_ref
{
    std::string_view model;                    // a file under assets/models (a glTF model), or empty
    gpu::mesh_desc (*generated)() = nullptr;   // or a mesh made in code (material_of all 0), wearing `material`;
                                               // gpu_scenes_meshes.ixx passes captureless lambdas over meshes::
    std::shared_ptr<::material> material;   // the CPU material a generated mesh wears (its GGX side entry, if any,
                                            // in microfacets, keyed by this address as today); ::material: the
                                            // member's own name hides the class inside mesh_ref
    std::array<double, 12> placement{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}; // after the model's own transforms
};
// Each distinct source is uploaded once (one GAS); every mesh_ref is an instance of it.

// A scene as the GPU programs build it: the CPU's objects and camera, from which gpu_scene.ixx translates the
// renderer's tables, and what the CPU's classes can't describe
export struct gpu_scene_parts
{
    scene cpu;
    // The GGX side entry of a CPU material, found by its address when the scene is translated: the CPU material is the
    // placeholder that carries the color (a metal's albedo is F0, a matte one's texture the coat's base)
    std::vector<std::pair<const material*, gpu::microfacet_desc>> microfacets;
    environment_choice environment;
    std::vector<mesh_ref> meshes; // triangle meshes, which only OptiX renders; empty in the scenes without them
};

// A scene in the list: its name on the command line, a description for the help, its group and the function that
// builds it (only the chosen scene is built, as in the CPU's scene_list), and its own strategy when that isn't one of
// the books' (which its camera holds: empty)
export struct gpu_scene_entry
{
    std::string_view name;
    std::string_view description;
    scene_role role;
    gpu_scene_parts (*build)();
    std::optional<strategy> default_strategy = {};
    // The display it opens with (tone_options.ixx): the books' bytes for the books' scenes and the test scenes, whose
    // images stay what they were; a tone map at the scene's exposure for the scenes beyond the books, whose light the
    // books' clamp would cut
    gpu::display_options display = {};
};
