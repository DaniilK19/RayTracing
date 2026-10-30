// Module strategy_options: the sampling strategies of the GPU programs' -p option, and the choice of kernel. The
// strategies begin with the books' four, which the CPU's camera knows (sampling_list in camera.ixx), and the GPU's third
// kernel adds its own later: the CPU's list stays as it is, so RayTracing never parses a strategy it can't render. The
// window's scripts use these names too, so the module knows nothing of the GPU renderer (no renderer.h): only the names
// and what they stand for.
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
    nee, // the quality kernel's light samples: next-event estimation
    mis, // light samples and the material's own, by multiple importance sampling
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
    strategy_entry{"nee", strategy::nee, "a shadow ray to a light at every bounce; the quality kernel, float",
                   std::nullopt},
    strategy_entry{"mis", strategy::mis,
                   "shadow rays and the material's samples, weighed: low noise; the quality kernel, float",
                   std::nullopt},
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

// Whether the quality kernel's integrator renders a strategy (the quality kernel, and OptiX's programs under --optix):
// its bounces sample as the material scatters, so of the books' strategies only material, and every one of its own
export constexpr bool integrator_renders(strategy value)
{
    return entry_of(value).books.value_or(sampling_strategy::material) == sampling_strategy::material;
}

// Those strategies as the programs' refusals list them: material, then every one that isn't the books' ("material, nee
// or mis")
export std::string integrator_strategies()
{
    std::string strategies = "material";
    const auto own = strategy_list | std::views::filter([](const strategy_entry& e) { return !e.books; }) |
                     std::ranges::to<std::vector>();
    for (std::size_t i = 0; i < own.size(); ++i)
    {
        strategies += std::format("{}{}", i + 1 == own.size() ? " or " : ", ", own[i].name);
    }
    return strategies;
}

// Which kernel renders: the one the scene needs, the books' kernel for all three books even where the first book's
// would do (--all-books-kernel), the quality kernel even for the books' scenes (--quality-kernel), or the quality
// kernel's integrator in OptiX's programs (--optix: RayTracingGPU's from Step 89, the window's from Step 95)
export enum class kernel_choice
{
    automatic,
    all_books,
    quality,
    optix,
};

// Its name in the window's scripts (kernel auto|all-books|quality; optix from Step 95)
export constexpr std::string_view kernel_choice_name(kernel_choice kernel)
{
    return kernel == kernel_choice::all_books ? "all-books"
           : kernel == kernel_choice::quality ? "quality"
           : kernel == kernel_choice::optix   ? "optix"
                                              : "auto";
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
