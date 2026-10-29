// Module strategy_options: the sampling strategies of the GPU programs' -p option. They begin with the books' four,
// which the CPU's camera knows (sampling_list in camera.ixx), and the GPU's third kernel adds its own later: the CPU's
// list stays as it is, so RayTracing never parses a strategy it can't render. The window's scripts use these names
// too, so the module knows nothing of the GPU renderer (no renderer.h): only the names and what they stand for.
export module strategy_options;

import std;
import camera;

// A strategy of the GPU programs. The enumerators start as the books' four, in their order; the later ones are
// appended.
export enum class strategy
{
    uniform,
    material,
    lights,
    mixture,
};

export struct strategy_entry
{
    std::string_view name;
    strategy value;
    std::string_view description;
    std::optional<sampling_strategy> books; // the books' strategy it is, which the CPU's camera renders
};

// A books' strategy, under the name and with the description the CPU's list gives it, so the help of RayTracingGPU
// and RayTracingViewer stays RayTracing's. consteval: it runs only at compile time, where a strategy missing from
// sampling_list makes the throw a compile error.
consteval strategy_entry books_entry(strategy value, sampling_strategy books)
{
    for (const sampling_entry& entry : sampling_list)
    {
        if (entry.strategy == books)
        {
            return {entry.name, value, entry.description, books};
        }
    }
    throw "every books strategy has an entry in sampling_list";
}

export constexpr std::array strategy_list{
    books_entry(strategy::uniform, sampling_strategy::uniform),
    books_entry(strategy::material, sampling_strategy::material),
    books_entry(strategy::lights, sampling_strategy::lights),
    books_entry(strategy::mixture, sampling_strategy::mixture),
};

// The entry of a name, or nullptr
export constexpr const strategy_entry* find_strategy(std::string_view name)
{
    const auto it = std::ranges::find(strategy_list, name, &strategy_entry::name);
    return it == strategy_list.end() ? nullptr : &*it;
}

// Every strategy has an entry (strategy_list holds them all), so these find one
export constexpr const strategy_entry& entry_of(strategy value)
{
    return *std::ranges::find(strategy_list, value, &strategy_entry::value);
}

export constexpr std::string_view strategy_name(strategy value) { return entry_of(value).name; }

// The GPU's strategy for one of the books'
export constexpr strategy from_books(sampling_strategy books)
{
    return std::ranges::find(strategy_list, std::optional{books}, &strategy_entry::books)->value;
}

// The list for the programs' help, a line per strategy: as RayTracing prints it, the camera's default marked
export std::string strategy_usage()
{
    std::string text;
    for (const strategy_entry& entry : strategy_list)
    {
        // camera_settings{} holds the camera's defaults
        const bool is_default = entry.books == camera_settings{}.sampling;
        text += std::format("  {:10} {}{}\n", entry.name, entry.description, is_default ? " (default)" : "");
    }
    return text;
}
