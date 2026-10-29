// Partition gpu_scenes:parts, the types of the GPU programs' list of scenes (see gpu_scenes.ixx). A module partition
// (C++20) is a part of one module in a file of its own: gpu_scenes re-exports it, and the files that add scenes beyond
// the books import it without importing each other.
export module gpu_scenes:parts;

import std;
import scenes;
import strategy_options;

// What a scene of the list is for. The help shows each group under a heading of its own.
export enum class scene_role
{
    book,     // the books' 13, as RayTracing renders them
    showcase, // the GPU's own scenes of light and materials beyond the books
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

// A scene as the GPU programs build it: the CPU's objects and camera, from which gpu_scene.ixx translates the
// renderer's tables, and what the CPU's classes can't describe
export struct gpu_scene_parts
{
    scene cpu;
    environment_choice environment;
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
};
