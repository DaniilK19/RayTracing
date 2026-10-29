// GpuTests: numeric tests of the GPU renderer's device functions, on the GPU: the samplers draw what their densities
// say, the densities integrate to 1, the estimators converge to known values. See gpu_test.h for the checks and the
// report; the suites are the files tests/*_tests.cu.
//   GpuTests [all | SUITE...] [--quick] [--list] [--write FILE] [--check FILE]
// Exit code: 0 every check passes, 1 a check fails, 2 an error (an unknown suite or option, a CUDA error), 77 no CUDA
// device or no driver (as ctest's SKIP_RETURN_CODE: the tests are skipped on a computer without an NVIDIA GPU). A
// driver or runtime that is there but fails is an error, 2: a broken installation must not pass for a skipped test.

// The headers before `import std;` (see renderer.h)
#include <cuda_runtime_api.h>

#include "gpu_test.h"
#include "host_services.h"
#include "renderer.h"
#include "statistics.h"

import std;
import camera;
import constant_medium;
import gpu_scene;
import gpu_scenes;
import hittable;
import image;
import material;
import quad;
import scenes;
import sphere;
import vec3;

namespace gpu_test
{
std::vector<test_entry>& registry()
{
    static std::vector<test_entry> tests;
    return tests;
}

// --- The host services of host_services.h -------------------------------------------------------------------------

namespace host
{
namespace
{
std::filesystem::path executable_folder; // set by main

// The run's temp folder, made on first use and removed when the program ends (a static object's destructor)
struct temp_holder
{
    std::filesystem::path path;
    ~temp_holder()
    {
        if (!path.empty())
        {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    }
};
}

std::filesystem::path temp_folder()
{
    static temp_holder temp;
    if (temp.path.empty())
    {
        temp.path = std::filesystem::temp_directory_path() /
                    std::format("gpu-tests-{}", std::chrono::steady_clock::now().time_since_epoch().count());
        std::filesystem::create_directories(temp.path);
    }
    return temp.path;
}

std::filesystem::path find_repo_file(std::string_view relative)
{
    for (const std::filesystem::path& start : {std::filesystem::current_path(), executable_folder})
    {
        for (std::filesystem::path folder = start; !folder.empty(); folder = folder.parent_path())
        {
            if (std::filesystem::exists(folder / relative))
            {
                return folder / relative;
            }
            if (folder == folder.parent_path())
            {
                break; // the root
            }
        }
    }
    throw std::runtime_error(std::format("no {} in the current folder, the executable's or above them", relative));
}

gpu::image_desc load_image(const std::filesystem::path& file)
{
    auto loaded = image::load(file);
    if (!loaded)
    {
        throw std::runtime_error(std::format("{}: {}", file.string(), loaded.error()));
    }
    gpu::image_desc desc{.width = loaded->width(), .height = loaded->height()};
    desc.rgb.reserve(3 * static_cast<std::size_t>(desc.width) * static_cast<std::size_t>(desc.height));
    for (int y = 0; y < desc.height; ++y)
    {
        for (int x = 0; x < desc.width; ++x)
        {
            const color& c = (*loaded)[x, y]; // doubles made from stb_image's floats: back to floats exactly
            desc.rgb.push_back(static_cast<float>(c.x));
            desc.rgb.push_back(static_cast<float>(c.y));
            desc.rgb.push_back(static_cast<float>(c.z));
        }
    }
    return desc;
}

// RayTracingGPU's own function (gpu_scene.ixx), the one its PNGs go through
std::vector<double> bytes_to_linear(std::span<const std::uint8_t> rgba, int width, int height)
{
    const image img = bytes_to_image(rgba, width, height);
    std::vector<double> values;
    values.reserve(3 * static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const color& c = img[x, y];
            values.insert(values.end(), {c.x, c.y, c.z});
        }
    }
    return values;
}

std::vector<std::uint8_t> png_round_trip(std::span<const std::uint8_t> rgba, int width, int height)
{
    const image img = bytes_to_image(rgba, width, height);
    const std::filesystem::path file = temp_folder() / "round-trip.png";
    if (const auto saved = img.save_png(file); !saved)
    {
        throw std::runtime_error(saved.error());
    }
    auto loaded = image::load(file);
    if (!loaded)
    {
        throw std::runtime_error(loaded.error());
    }
    // stb_image decodes byte k as (k/255)^2.2: the byte whose decoding is nearest
    std::array<double, 256> decoded{};
    for (int k = 0; k < 256; ++k)
    {
        decoded[static_cast<std::size_t>(k)] = std::pow(k / 255.0, 2.2);
    }
    const auto byte_of = [&decoded](double v) {
        const auto nearest = std::ranges::min_element(decoded, {}, [v](double d) { return std::abs(d - v); });
        return static_cast<std::uint8_t>(nearest - decoded.begin());
    };
    std::vector<std::uint8_t> bytes;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const color& c = (*loaded)[x, y];
            bytes.push_back(byte_of(c.x));
            bytes.push_back(byte_of(c.y));
            bytes.push_back(byte_of(c.z));
        }
    }
    return bytes;
}

translation_check check_ggx_translation()
{
    const auto matte = std::make_shared<const lambertian>(color{0.5, 0.5, 0.5});
    const auto lamp = std::make_shared<const diffuse_light>(color{4, 4, 4});
    const auto coat = std::make_shared<const lambertian>(color{0.7, 0.2, 0.2});
    const auto rough = std::make_shared<const metal>(color{0.9, 0.9, 0.9}, 0.0);
    const auto stray = std::make_shared<const metal>(color{0.5, 0.5, 0.5}, 0.0); // in no object of the world
    const gpu::microfacet_desc rough_entry{.kind = gpu::microfacet_kind::conductor, .roughness = 0.5};
    const gpu::microfacet_desc coat_entry{.kind = gpu::microfacet_kind::gloss, .roughness = 0.3};
    const auto parts_with = [&](std::vector<std::pair<const material*, gpu::microfacet_desc>> extra) {
        gpu_scene_parts parts;
        parts.cpu.camera = {.aspect_ratio = 1, .image_width = 8, .samples_per_pixel = 1};
        hittable_list& world = parts.cpu.world;
        world.add(std::make_unique<sphere>(point3{0, -100, 0}, 99, matte));
        world.add(std::make_unique<quad>(point3{-1, 3, -1}, vec3{2, 0, 0}, vec3{0, 0, 2}, lamp));
        world.add(std::make_unique<constant_medium>(std::make_unique<sphere>(point3{0, 0, -5}, 1, matte), 0.1,
                                                    color{0.9, 0.9, 0.9}));
        world.add(std::make_unique<sphere>(point3{-1, 0.5, 0}, 0.5, coat));
        world.add(std::make_unique<sphere>(point3{1, 0.5, 0}, 0.5, rough));
        parts.microfacets = {{rough.get(), rough_entry}, {coat.get(), coat_entry}};
        parts.microfacets.insert(parts.microfacets.end(), extra.begin(), extra.end());
        return parts;
    };
    const auto refused = [&](std::vector<std::pair<const material*, gpu::microfacet_desc>> extra) {
        try
        {
            (void)translate_parts(parts_with(std::move(extra)));
        }
        catch (const std::logic_error&)
        {
            return true;
        }
        return false;
    };

    translation_check out;
    const auto desc = translate_parts(parts_with({}));
    if (!desc)
    {
        out.aligned = "not translated: " + desc.error();
    }
    else
    {
        // The materials in the order of first use: the matte floor, the lamp, the smoke's phase function, the gloss's
        // placeholder, the conductor's
        const std::vector<gpu::microfacet_desc>& side = desc->quality.microfacets;
        const std::array<gpu::microfacet_kind, 5> kinds{gpu::microfacet_kind::none, gpu::microfacet_kind::none,
                                                        gpu::microfacet_kind::none, gpu::microfacet_kind::gloss,
                                                        gpu::microfacet_kind::conductor};
        if (desc->materials.size() != kinds.size() || side.size() != kinds.size())
        {
            out.aligned = std::format("{} materials and {} side entries, expected 5 and 5", desc->materials.size(),
                                      side.size());
        }
        for (std::size_t i = 0; out.aligned.empty() && i < kinds.size(); ++i)
        {
            const double expected = i == 3 ? 0.3 : i == 4 ? 0.5 : 0.0;
            if (side[i].kind != kinds[i] || side[i].roughness != expected)
            {
                out.aligned = std::format("side entry {} isn't its material's", i);
            }
        }
    }
    out.same_entry_twice = !refused({{rough.get(), rough_entry}});
    out.two_entries_refused =
        refused({{rough.get(), gpu::microfacet_desc{.kind = gpu::microfacet_kind::conductor, .roughness = 0.6}}});
    out.unused_refused = refused({{stray.get(), rough_entry}});
    return out;
}

std::vector<scene_display> scene_displays()
{
    std::vector<scene_display> scenes;
    for (const gpu_scene_entry& entry : gpu_scene_list())
    {
        scenes.push_back({std::string(entry.name), entry.display});
    }
    return scenes;
}
}

namespace
{
std::string suite_name(const char* identifier)
{
    std::string name = identifier;
    std::ranges::replace(name, '_', '-');
    return name;
}

std::int64_t device_bytes = 0;
std::int64_t device_peak = 0;

std::int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

constexpr double suite_seconds_limit = 10;
constexpr std::int64_t suite_bytes_limit = 64 << 20;
}

void note_device_bytes(std::int64_t change)
{
    device_bytes += change;
    device_peak = std::max(device_peak, device_bytes);
}

std::int64_t device_peak_bytes() { return device_peak; }

void reset_device_peak() { device_peak = device_bytes; }

void context::mean(std::string_view name, double estimate, double standard_error, double expected, double z_limit)
{
    const double z = standard_error > 0 ? (estimate - expected) / standard_error
                     : estimate == expected ? 0.0
                                            : std::numeric_limits<double>::infinity();
    results_.push_back({.name = std::string(name), .kind = check_kind::mean, .estimate = estimate,
                        .expected = expected, .statistic = z, .limit = z_limit});
}

void context::chi2(std::string_view name, double statistic, int dof, double p)
{
    results_.push_back({.name = std::string(name), .kind = check_kind::chi2, .estimate = statistic,
                        .expected = static_cast<double>(dof), .statistic = p, .dof = dof});
}

void context::must_reject(std::string_view name, double statistic, int dof, double p, double p_max)
{
    results_.push_back({.name = std::string(name), .kind = check_kind::must_reject, .estimate = statistic,
                        .expected = static_cast<double>(dof), .statistic = p, .limit = p_max, .dof = dof});
}

void context::exact(std::string_view name, double estimate, double expected, double tolerance)
{
    results_.push_back({.name = std::string(name), .kind = check_kind::exact, .estimate = estimate,
                        .expected = expected, .statistic = std::abs(estimate - expected), .limit = tolerance});
}

void context::count(std::string_view name, std::int64_t found, std::int64_t allowed, std::int64_t out_of)
{
    results_.push_back({.name = std::string(name), .kind = check_kind::count, .estimate = static_cast<double>(found),
                        .limit = static_cast<double>(allowed), .out_of = out_of});
}

const std::string* context::argument(std::string_view name) const
{
    const auto it = std::ranges::find(arguments_, name, &std::pair<std::string, std::string>::first);
    return it == arguments_.end() ? nullptr : &it->second;
}

void context::info(std::string_view name, std::string_view text)
{
    infos_.emplace_back(results_.size(), std::format("{}.{}: {}, INFO", suite_, name, text));
}

void context::begin_suite(std::string_view suite)
{
    suite_ = suite;
    results_.clear();
    infos_.clear();
    reset_device_peak();
    start_ns_ = now_ns();
}

int context::end_suite()
{
    // The suite's budget, as two checks of its own
    const double seconds = static_cast<double>(now_ns() - start_ns_) * 1e-9;
    count("budget-gpu-memory-bytes", device_peak_bytes(), suite_bytes_limit, suite_bytes_limit);
    exact("budget-seconds", seconds, 0, suite_seconds_limit);

    const auto chi2_tests = std::ranges::count(results_, check_kind::chi2, &check_result::kind);
    const double threshold = sidak_threshold(0.01, static_cast<int>(chi2_tests));
    int failures = 0;
    std::size_t next_info = 0;
    for (std::size_t i = 0; i < results_.size(); ++i)
    {
        for (; next_info < infos_.size() && infos_[next_info].first <= i; ++next_info)
        {
            std::println("{}", infos_[next_info].second);
        }
        const check_result& r = results_[i];
        const bool statistical =
            r.kind == check_kind::mean || r.kind == check_kind::chi2 || r.kind == check_kind::must_reject;
        bool pass = false;
        std::string detail;
        switch (r.kind)
        {
        case check_kind::mean:
            pass = std::abs(r.statistic) <= r.limit;
            detail = std::format("{:.9g}, expected {:.9g}, z = {:.3f} (|z| <= {})", r.estimate, r.expected,
                                 r.statistic, r.limit);
            break;
        case check_kind::chi2:
            pass = r.statistic >= threshold;
            detail = std::format("chi2 = {:.2f} ({} dof), expected {}, p = {:.4g} (>= {:.4g})", r.estimate, r.dof,
                                 r.dof, r.statistic, threshold);
            break;
        case check_kind::must_reject:
            pass = r.statistic < r.limit;
            detail = std::format("chi2 = {:.2f} ({} dof), expected a rejection, p = {:.4g} (< {:.0e})", r.estimate,
                                 r.dof, r.statistic, r.limit);
            break;
        case check_kind::exact:
            pass = r.statistic <= r.limit;
            detail = std::format("{:.9g}, expected {:.9g}, |difference| = {:.3g} (<= {:.3g})", r.estimate, r.expected,
                                 r.statistic, r.limit);
            break;
        case check_kind::count:
            pass = r.estimate <= r.limit;
            detail = std::format("{} of {}, expected at most {}", static_cast<std::int64_t>(r.estimate), r.out_of,
                                 static_cast<std::int64_t>(r.limit));
            break;
        }
        const bool skipped = statistical && quick_;
        std::println("{}.{}: {}, {}", suite_, r.name, detail, skipped ? "SKIP (quick)" : pass ? "PASS" : "FAIL");
        failures += !skipped && !pass ? 1 : 0;
    }
    for (; next_info < infos_.size(); ++next_info)
    {
        std::println("{}", infos_[next_info].second);
    }
    return failures;
}
}

int main(int argc, char* argv[])
{
    using namespace gpu_test;
    std::error_code no_folder;
    host::executable_folder = std::filesystem::absolute(argv[0], no_folder).parent_path(); // for find_repo_file
    bool quick = false;
    bool list = false;
    std::vector<std::string> wanted;
    std::vector<std::pair<std::string, std::string>> arguments; // --write FILE, --check FILE
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg == "--quick")
        {
            quick = true;
        }
        else if (arg == "--list")
        {
            list = true;
        }
        else if (arg == "-h" || arg == "--help")
        {
            std::println("Usage: GpuTests [all | SUITE...] [--quick] [--list] [--write FILE] [--check FILE]\n"
                         "Numeric tests of the GPU renderer's device functions (see tests/gpu_test.h).\n"
                         "--write and --check give a suite a file (a table it writes or checks).\n"
                         "Exit code: 0 pass, 1 a check fails, 2 an error, 77 no CUDA device.");
            return 0;
        }
        else if (arg == "--write" || arg == "--check")
        {
            if (i + 1 == argc)
            {
                std::println("Error: {} needs a file", arg);
                return 2;
            }
            arguments.emplace_back(std::string(arg), std::string(argv[++i]));
        }
        else if (arg.starts_with('-'))
        {
            std::println("Error: unknown option '{}' (GpuTests --help)", arg);
            return 2;
        }
        else
        {
            wanted.emplace_back(arg);
        }
    }

    // The suites by name, each with its tests in the order its file registered them
    std::map<std::string, std::vector<const test_entry*>> suites;
    for (const test_entry& t : registry())
    {
        suites[suite_name(t.suite)].push_back(&t);
    }
    if (list)
    {
        for (const auto& [name, tests] : suites)
        {
            std::println("{}: {}", name,
                         tests | std::views::transform([](const test_entry* t) { return std::string(t->name); }) |
                             std::views::join_with(std::string(", ")) | std::ranges::to<std::string>());
        }
        return 0;
    }
    for (const std::string& name : wanted)
    {
        if (name != "all" && !suites.contains(name))
        {
            std::println("Error: no suite '{}' (GpuTests --list names them)", name);
            return 2;
        }
    }
    if (wanted.empty() || std::ranges::contains(wanted, std::string("all")))
    {
        wanted.clear();
        for (const auto& [name, tests] : suites)
        {
            wanted.push_back(name);
        }
    }

    // Skipped only without a GPU or without a driver; any other failure to start CUDA is an error
    int devices = 0;
    const cudaError_t counted = cudaGetDeviceCount(&devices);
    if (counted == cudaErrorNoDevice || counted == cudaErrorInsufficientDriver ||
        (counted == cudaSuccess && devices == 0))
    {
        std::println("No CUDA device ({}): the tests are skipped", cudaGetErrorString(counted));
        return 77;
    }
    try
    {
        const gpu::device_info device = gpu::query_device();
        std::println("GpuTests on a GPU of compute capability {}.{}{}", device.compute_capability_major,
                     device.compute_capability_minor, quick ? ", quick: statistical verdicts skipped" : "");
    }
    catch (const std::exception& error)
    {
        std::println("Error: CUDA can't start: {}", error.what());
        return 2;
    }

    int failures = 0;
    context ctx(quick);
    for (auto& [name, value] : arguments)
    {
        ctx.set_argument(std::move(name), std::move(value));
    }
    for (const std::string& name : wanted)
    {
        ctx.begin_suite(name);
        try
        {
            for (const test_entry* t : suites[name])
            {
                t->run(ctx);
            }
        }
        catch (const std::exception& error)
        {
            ctx.end_suite();
            std::println("Error in suite {}: {}", name, error.what());
            return 2;
        }
        failures += ctx.end_suite();
    }
    std::println("{}", failures == 0 ? "GpuTests: PASS" : std::format("GpuTests: FAIL ({} checks)", failures));
    return failures == 0 ? 0 : 1;
}
