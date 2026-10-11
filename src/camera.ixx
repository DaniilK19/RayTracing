// Module camera: the camera shoots rays through pixels and builds an image from their colors.
export module camera;

import std;
import hittable;
import image;
import interval;
import pdf;
import random;
import ray;
import vec3;

// The background is the light brought by a ray that hit nothing. The color changes smoothly with the height
// of the ray direction: from bottom (the ray looks down) to top (up). Equal colors give a flat background.
// The background is black by default: the scene is lit only by its own light sources.
export struct sky
{
    color bottom{0, 0, 0};
    color top{0, 0, 0};

    inline color in_direction(const vec3& direction) const
    {
        const double a = 0.5 * (unit_vector(direction).y + 1.0); // y from -1 to 1 → a from 0 to 1
        return (1.0 - a) * bottom + a * top;
    }
};

// The daylight sky from the first book: white at the horizon, blue at the zenith. The second book uses
// a flat blue background (0.7, 0.8, 1.0) instead, but the gradient is prettier and keeps the first book's scenes
// unchanged.
export constexpr sky daylight_sky{.bottom = {1.0, 1.0, 1.0}, .top = {0.5, 0.7, 1.0}};

// How the camera chooses the next direction after a diffuse bounce (a matte surface or smoke): with what
// density p. Any p that is above 0 wherever the material scatters light gives an unbiased estimate of the same
// image (book 3, chapter 6), only with more or less noise: the more closely p follows pScatter · (incoming
// light), the less noise. lights breaks that rule and is biased; mixture keeps it (see ray_color).
export enum class sampling_strategy
{
    uniform,  // the same density, 1/(4π), in every direction
    material, // with the material's own density, p = pScatter
    lights,   // toward a random object of the scene's lights list, usually a lamp, with the list's density p
    mixture,  // half of the time as material, half as lights: p is the average of the two densities
              // (smoke and fog: as material, see isotropic)
};

// The strategies for the command line: a name, the value and a description for the help. Every strategy must have
// an entry here: main also looks up the name of the strategy it renders with.
export struct sampling_entry
{
    std::string_view name;
    sampling_strategy strategy;
    std::string_view description;
};

export constexpr std::array sampling_list{
    sampling_entry{"uniform", sampling_strategy::uniform, "the same density in all directions (for comparison: noisy)"},
    sampling_entry{"material", sampling_strategy::material, "as the surface scatters: the first two books' behavior"},
    sampling_entry{"lights", sampling_strategy::lights,
                   "aims only at the lamps: low-noise direct light, but biased: indirect light is lost"},
    sampling_entry{"mixture", sampling_strategy::mixture, "half material, half lamps: low noise and unbiased"},
};

// The strategy the camera really renders with. lights and mixture need lamps to aim at: a scene without them, like
// those lit by the sky, is sampled as the material scatters.
export inline sampling_strategy effective_sampling(sampling_strategy requested, const hittable_list& lights)
{
    const bool aims_at_lamps = requested == sampling_strategy::lights || requested == sampling_strategy::mixture;
    return aims_at_lamps && lights.empty() ? sampling_strategy::material : requested;
}

// Camera settings. They are convenient to set with designated initializers (C++20):
//   camera cam({.aspect_ratio = 16.0 / 9.0, .image_width = 400});
// Fields that aren't mentioned get their default values.
export struct camera_settings
{
    double aspect_ratio = 1.0;  // image width to height ratio
    int image_width = 100;      // image width in pixels
    int samples_per_pixel = 10; // how many random rays to average per pixel (rounded to a square, see camera)
    int max_depth = 10;         // at most this many rays per path, the camera ray included: max_depth - 1 bounces

    // How a diffuse bounce chooses the next direction
    sampling_strategy sampling = sampling_strategy::mixture;

    // Which random numbers the render draws (see render). Another seed renders the same scene with independent noise:
    // the same expected value in every pixel, so a difference between two such images that is larger than their noise
    // explains is a real one. That is how two renderers, or two versions of one, can be compared.
    std::uint32_t seed = 0;

    double vfov = 90;         // vertical field of view in degrees: a smaller angle means a stronger "zoom"
    point3 lookfrom{0, 0, 0}; // where the camera stands
    point3 lookat{0, 0, -1};  // where it looks
    vec3 vup{0, 1, 0};        // which direction counts as the frame's "up"

    // Depth of field. A real camera gathers light with a lens, and only objects at the focus distance
    // come out sharp. We imitate this: rays start not from a single point
    // but from random points of a "lens" disk and converge on the focus plane.
    double defocus_angle = 0; // angle of the ray cone through each pixel, in degrees (0 means everything is sharp)
    double focus_dist = 10;   // distance from the camera to the plane of perfect focus

    sky background; // what is visible where a ray hits nothing
};

// What the camera computes from its settings, as plain numbers: the pixel grid its rays aim at and the lens disk they
// start from. main reads the grid's size from it to count the samples of a render.
export struct camera_frame
{
    int image_width;
    int image_height;
    int sqrt_spp;          // rays per pixel: a sqrt_spp × sqrt_spp grid
    int max_depth;
    point3 center;         // camera position
    point3 pixel00_loc;    // center of pixel (0, 0)
    vec3 pixel_delta_u;    // step to the next pixel to the right
    vec3 pixel_delta_v;    // step to the next pixel below
    double defocus_angle;  // 0: every ray starts at the center; otherwise on the lens disk
    vec3 defocus_disk_u;   // horizontal radius of the lens disk
    vec3 defocus_disk_v;   // vertical radius of the lens disk
    sky background;
};

// What camera::render returns: the image and how much work it took.
export struct render_result
{
    image img;
    std::uint64_t rays = 0; // rays traced: the camera rays and all their bounces (see ray_color)
};

// How many rays the current thread has traced. Each worker thread counts its own, just as it has its own random number
// generator. A single atomic counter, changed by all threads tens of millions of times a second, would make them wait
// for each other: every change needs the counter in its own core's cache, taken away from the core that changed it
// last. render adds up the threads' counts, one addition per thread.
thread_local std::uint64_t rays_traced_by_this_thread = 0;

export class camera
{
public:
    explicit camera(const camera_settings& settings)
        : image_width_(settings.image_width),
          image_height_(std::max(1, static_cast<int>(settings.image_width / settings.aspect_ratio))),
          sqrt_spp_(std::max(1, static_cast<int>(std::lround(std::sqrt(settings.samples_per_pixel))))),
          // Members are initialized in the order they are declared in the class, not in the order of this list.
          // sqrt_spp_ is declared before the next two, so they can already use it.
          pixel_samples_scale_(1.0 / rays_per_pixel()),
          recip_sqrt_spp_(1.0 / sqrt_spp_),
          max_depth_(settings.max_depth),
          sampling_(settings.sampling),
          seed_(settings.seed),
          defocus_angle_(settings.defocus_angle),
          background_(settings.background)
    {
        center_ = settings.lookfrom;

        // The viewport stands on the focus plane in front of the camera: through each of its pixels
        // we shoot a ray. The field of view sets the window height: tan(vfov / 2) = (height / 2) / distance.
        const double h = std::tan(degrees_to_radians(settings.vfov) / 2);
        const double viewport_height = 2 * h * settings.focus_dist;
        const double viewport_width = viewport_height * (static_cast<double>(image_width_) / image_height_);

        // The camera's own axes (an orthonormal basis): w points backward (from the target to the camera),
        // u to the right, v up. vup defines where "up" is; cross gives the axes perpendicular to it.
        const vec3 w = unit_vector(settings.lookfrom - settings.lookat);
        const vec3 u = unit_vector(cross(settings.vup, w));
        const vec3 v = cross(w, u);

        // Vectors along the window's top edge (rightward) and left edge (downward: image rows go top to bottom)
        const vec3 viewport_u = viewport_width * u;
        const vec3 viewport_v = viewport_height * -v;

        // Step between neighboring pixel centers
        pixel_delta_u_ = viewport_u / image_width_;
        pixel_delta_v_ = viewport_v / image_height_;

        // Upper-left corner of the window and the center of pixel (0, 0)
        const point3 viewport_upper_left = center_ - settings.focus_dist * w - viewport_u / 2 - viewport_v / 2;
        pixel00_loc_ = viewport_upper_left + 0.5 * (pixel_delta_u_ + pixel_delta_v_);

        // The "lens" disk: its radius is such that from the viewport center it is seen at the angle defocus_angle
        const double defocus_radius = settings.focus_dist * std::tan(degrees_to_radians(settings.defocus_angle / 2));
        defocus_disk_u_ = defocus_radius * u;
        defocus_disk_v_ = defocus_radius * v;
    }

    // The pixel grid and the lens as numbers, for a renderer outside this class (see camera_frame)
    camera_frame frame() const
    {
        return {.image_width = image_width_,
                .image_height = image_height_,
                .sqrt_spp = sqrt_spp_,
                .max_depth = max_depth_,
                .center = center_,
                .pixel00_loc = pixel00_loc_,
                .pixel_delta_u = pixel_delta_u_,
                .pixel_delta_v = pixel_delta_v_,
                .defocus_angle = defocus_angle_,
                .defocus_disk_u = defocus_disk_u_,
                .defocus_disk_v = defocus_disk_v_,
                .background = background_};
    }

    // Render: for each pixel average the colors of several rays and write the result into the image.
    // Rows are computed in parallel on all CPU threads. lights are the objects the lights and mixture strategies
    // aim at. Returns the image and the number of rays traced.
    render_result render(const hittable& world, const hittable_list& lights) const
    {
        const sampling_strategy strategy = effective_sampling(sampling_, lights);
        const unsigned thread_count = std::max(1u, std::thread::hardware_concurrency());
        std::println("Rendering {}×{}, rays per pixel: {} ({}×{} strata), threads: {}", image_width_, image_height_,
                     rays_per_pixel(), sqrt_spp_, sqrt_spp_, thread_count);

        image img(image_width_, image_height_);

        // Each thread takes the next free row until the rows run out. The counters are atomic
        // because several threads change them at once. The pixels, however, differ between threads (each writes
        // only to its own rows), so the image needs no protection.
        std::atomic<int> next_row = 0;
        std::atomic<int> rows_done = 0;
        std::atomic<std::uint64_t> rays = 0;

        // All cells (s_j, s_i) of the pixel's grid, row by row. std::views::cartesian_product (C++23) gives every
        // pair of an element of the first range with one of the second: one loop instead of two nested ones.
        const auto strata =
            std::views::cartesian_product(std::views::iota(0, sqrt_spp_), std::views::iota(0, sqrt_spp_));

        auto render_rows = [&] {
            rays_traced_by_this_thread = 0; // the count belongs to the thread, not to this render: start it afresh
            for (int j = next_row++; j < image_height_; j = next_row++)
            {
                // Each row has its own random number sequence: the image doesn't depend
                // on which thread computed the row or when, and is the same on every run.
                // The seed setting fills the upper 32 bits of the generator's 64-bit seed and the row number the lower
                // ones, so each (seed, row) pair gets a sequence of its own. Seed 0 seeds row j with j, as the camera
                // did before it had a seed setting, and renders the same images as then, bit for bit.
                seed_random((std::uint64_t{seed_} << 32) | static_cast<std::uint64_t>(j));
                for (int i = 0; i < image_width_; ++i)
                {
                    color pixel_color{0, 0, 0};
                    for (const auto [s_j, s_i] : strata)
                    {
                        const color sample = ray_color(get_ray(i, j, s_i, s_j), max_depth_, world, lights, strategy);
                        pixel_color += without_nan(sample);
                    }
                    img[i, j] = pixel_samples_scale_ * pixel_color;
                }
                ++rows_done;
                rows_done.notify_one(); // wake the main thread: time to update the progress
            }
            rays += rays_traced_by_this_thread;
        };

        {
            // std::jthread (C++20), unlike std::thread, waits for its thread to finish
            // in the destructor, that is, when leaving this block.
            std::vector<std::jthread> workers;
            for (unsigned t = 0; t < thread_count; ++t)
            {
                workers.emplace_back(render_rows);
            }

            // The main thread shows progress. atomic::wait (C++20) puts the thread to sleep
            // until the counter changes, without polling in a loop.
            for (int done = 0; done < image_height_; done = rows_done.load())
            {
                std::print("\rRows remaining: {:4}", image_height_ - done);
                rows_done.wait(done);
            }
        }
        std::print("\r{:30}\r", ""); // erase the progress line

        // Every worker has added its count by now: the jthreads' destructors waited for them to finish
        return {std::move(img), rays.load()};
    }

private:
    int image_width_;
    int image_height_;
    // The rays of a pixel form a sqrt_spp × sqrt_spp grid (see sample_square_stratified), so their number is a
    // square. We take the one nearest to samples_per_pixel: its square root rounded, where the book truncates
    // it (1000 rays → 32×32 = 1024, not 31×31 = 961; 500 → 22×22 = 484 either way).
    int sqrt_spp_;
    double pixel_samples_scale_; // 1 / sqrt_spp²: the averaging factor
    double recip_sqrt_spp_;      // 1 / sqrt_spp: the side of one grid cell, in pixel widths
    int max_depth_;
    sampling_strategy sampling_;
    std::uint32_t seed_;
    double defocus_angle_;
    sky background_;
    point3 center_;              // camera position
    point3 pixel00_loc_;         // center of pixel (0, 0)
    vec3 pixel_delta_u_;         // step to the next pixel to the right
    vec3 pixel_delta_v_;         // step to the next pixel below
    vec3 defocus_disk_u_;        // horizontal radius of the "lens" disk
    vec3 defocus_disk_v_;        // vertical radius of the "lens" disk

    // sqrt_spp² in 64 bits: for -s near the largest int (2^31 - 1) the rounded-up square no longer fits in an int
    std::int64_t rays_per_pixel() const { return std::int64_t{sqrt_spp_} * sqrt_spp_; }

    // A ray through a random point of cell (s_i, s_j) of pixel (i, j). If all rays went exactly through
    // the center, object edges would look like "staircases"; random points inside the pixel smooth them out.
    // The ray starts at a random point of the lens (or at the camera center if defocus is off).
    // The emission moment is random too: the shutter is open from time 0 to 1, and the pixel averages
    // everything a moving object managed to "sweep" in that time, which gives motion blur.
    ray get_ray(int i, int j, int s_i, int s_j) const
    {
        const vec3 offset = sample_square_stratified(s_i, s_j);
        const point3 pixel_sample =
            pixel00_loc_ + ((i + offset.x) * pixel_delta_u_) + ((j + offset.y) * pixel_delta_v_);
        const point3 ray_origin = defocus_angle_ <= 0 ? center_ : defocus_disk_sample();
        const double ray_time = random_double();
        return ray{ray_origin, pixel_sample - ray_origin, ray_time};
    }

    // Random point in cell (s_i, s_j) of the pixel's sqrt_spp × sqrt_spp grid, as an offset from the pixel center
    // (the pixel is the square [-0.5, 0.5] × [-0.5, 0.5]). This is stratified sampling (book 3, section 2.3):
    // the cells ("strata") have equal areas and each gets exactly one random point, so the average still estimates
    // the average over the whole pixel without bias, but the points can't clump together and leave holes. Where
    // an object's edge or a sharp line of a texture crosses the pixel, only the cells it passes through can get
    // their point on either side of it, so this part of the error falls as N^(-3/4) instead of N^(-1/2)
    // (MonteCarlo stratified shows this for a circle). The lens point, the time and the bounce directions
    // (each bounce adds two more dimensions) stay purely random. The more dimensions d, the less a grid helps:
    // at an edge the error falls as N^(-1/2 - 1/(2d)), the book's "curse of dimensionality". So edges and
    // textures get less noisy, but the noise of the lighting hardly changes.
    vec3 sample_square_stratified(int s_i, int s_j) const
    {
        // One random number per statement fixes their order (x, then y); in a + b or f(a, b) C++ leaves it unspecified
        const double px = (s_i + random_double()) * recip_sqrt_spp_ - 0.5;
        const double py = (s_j + random_double()) * recip_sqrt_spp_ - 0.5;
        return {px, py, 0};
    }

    // Random point on the "lens" disk around the camera center.
    point3 defocus_disk_sample() const
    {
        const vec3 p = random_in_unit_disk();
        return center_ + p.x * defocus_disk_u_ + p.y * defocus_disk_v_;
    }

    // A safety net (book 3, section 12.6): a sample that came out as NaN ("not a number", e.g. from 0/0 or 0·∞)
    // counts as black in that channel. Left in, one NaN would make the pixel's whole sum NaN, which is saved as black.
    // The book replaces NaN in the finished pixel, which loses all its samples in that channel; here only the bad
    // sample is lost. It is just in case: ray_color doesn't trace a direction whose p isn't above 0 (NaN included),
    // and the sphere's light sampling has a branch for points inside it, so no NaN is expected.
    static color without_nan(const color& c)
    {
        // The book tests x != x instead of std::isnan: NaN is the only value that isn't equal to itself
        return {std::isnan(c.x) ? 0.0 : c.x, std::isnan(c.y) ? 0.0 : c.y, std::isnan(c.z) ? 0.0 : c.z};
    }

    // The color a ray "brings back". Having hit an object, the ray is reflected or refracted,
    // and we recursively ask what color comes from there. depth is how many more bounces are allowed.
    // Each bounce is one more call on the thread's stack (1 MB for a std::jthread on Windows), so a max_depth of about
    // 2000 would overflow it in a scene whose paths really get that long, such as light trapped inside a glass ball.
    // The book's scenes stop at 50.
    color ray_color(const ray& r, int depth, const hittable& world, const hittable_list& lights,
                    sampling_strategy strategy) const
    {
        if (depth <= 0)
        {
            return color{0, 0, 0}; // too many bounces: assume no light reached the camera
        }

        // Look for hits with t from 0.001, not from 0: due to rounding errors the reflection point
        // ends up slightly below the surface, and the ray would immediately "hit" the same surface. In the image
        // this would give dark spots ("shadow acne").
        const auto rec = world.hit(r, interval{0.001, infinity});

        // A search of the whole scene for the nearest hit: one ray traced. (The lights and mixture strategies also test
        // directions against single lamps, in hittable_pdf, but those tests don't search the scene and don't count.)
        ++rays_traced_by_this_thread;
        if (!rec)
        {
            // The ray went off to infinity and brought back the background color
            return background_.in_direction(r.direction);
        }

        // The light from the hit point is the light the surface emits by itself plus the reflected light.
        // The material decides whether the ray is absorbed, follows a mirror or glass, or scatters diffusely;
        // for a diffuse bounce the camera's sampling strategy picks the direction (below).
        const color emitted = rec->mat->emitted(*rec);
        const auto scatter = rec->mat->scatter(r, *rec);
        if (!scatter)
        {
            return emitted; // the ray was absorbed or hit a light source that reflects nothing
        }

        // A mirror or glass: follow the ray it chose. std::get_if (C++17) gives a pointer to the variant's value
        // if it holds that type, and nullptr otherwise.
        if (const auto* specular = std::get_if<specular_scatter>(&*scatter))
        {
            return emitted + specular->attenuation * ray_color(specular->scattered, depth - 1, world, lights, strategy);
        }
        const auto& diffuse = std::get<diffuse_scatter>(*scatter);

        // The light a diffuse bounce reflects is the integral over all directions ω of attenuation · pScatter(ω) ·
        // (light coming from ω). One Monte Carlo sample of it: a direction drawn from some pdf, integrand / p.
        // The parameter type `const direction_pdf auto&` (C++20) makes the lambda a template: it is compiled
        // separately for each pdf type it is called with below, and accepts only types that meet the concept.
        const auto sample = [&](const direction_pdf auto& pdf) -> color {
            const ray scattered{rec->p, pdf.generate(), r.time}; // keep the time: motion blur
            const double pdf_value = pdf.value(scattered.direction);
            const double scattering_pdf = rec->mat->scattering_pdf(r, *rec, scattered);

            // A direction the material doesn't scatter into (below the surface) brings nothing: it isn't traced.
            // Written as !(x > 0) rather than x <= 0, the test also catches NaN: <, <=, >, >= and == with a NaN
            // are all false (only != is true), so x <= 0 would let a NaN through.
            if (!(pdf_value > 0) || !(scattering_pdf > 0))
            {
                return color{0, 0, 0};
            }
            // The ratio first: when the direction was drawn with the material's own density, both are the same
            // number, the ratio is exactly 1.0, and the image is bit for bit the same as without the ratio.
            return diffuse.attenuation * (scattering_pdf / pdf_value) *
                   ray_color(scattered, depth - 1, world, lights, strategy);
        };

        switch (strategy)
        {
        case sampling_strategy::uniform:
            // Ignore the material's pdf and draw from the same density in every direction of the whole sphere
            // (smoke scatters into all of it; for a surface the half below it brings nothing). Unbiased,
            // because we divide by the density the direction was really drawn with. The book (section 6.2)
            // divides the material's directions, drawn with cos θ / π, by 1/(2π) instead: each bounce then gets
            // the weight 2 cos θ, on average 4/3, and a white wall (albedo 0.73) acts like one of albedo 0.97:
            // not a noisier image, but a wrong, too bright one.
            return emitted + sample(sphere_pdf{});
        case sampling_strategy::material:
            return emitted + sample(diffuse.pdf);
        case sampling_strategy::lights:
            // Aim at a random object of the lights list, usually a lamp (book 3, chapters 9 and 12). Direct light
            // gets much less noisy, but the estimate is biased: importance sampling is unbiased only if p > 0
            // wherever the integrand isn't 0, and p is 0 in every direction that misses the list's objects. So only
            // light arriving from their directions counts: all of the direct light, plus what an object standing
            // between the point and a lamp reflects. All other indirect light (from walls, ceiling, blocks) is lost,
            // so shadows and the Cornell box's ceiling go black, and so is the light of lamps missing from the list.
            // The mixture below fixes this.
            return emitted + sample(hittable_pdf{lights, rec->p});
        case sampling_strategy::mixture:
            // Half of the directions toward the lamps, half as the material scatters (book 3, section 10.3). The
            // lamps' half makes their direct light much less noisy, and nothing is lost: wherever the material
            // scatters, p is at least half of pScatter, so p > 0 there (unbiased), and the weight pScatter / p is
            // never more than 2. Light that only the material's half finds (from a lamp missing from the list, or
            // a lamp seen in a mirror) is drawn half as often, with twice the weight: it gets noisier.
            // Smoke and fog opt out and sample with their own density (beyond the book, see isotropic).
            if (!rec->mat->mixes_in_lights())
            {
                return emitted + sample(diffuse.pdf);
            }
            return emitted + sample(mixture_pdf{hittable_pdf{lights, rec->p}, diffuse.pdf});
        }
        std::unreachable(); // C++23: tells the compiler this point is never reached (every case returns)
    }
};
