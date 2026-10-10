// Module camera: the camera shoots rays through pixels and builds an image from their colors.
export module camera;

import std;
import hittable;
import image;
import interval;
import random;
import ray;
import vec3;

constexpr double degrees_to_radians(double degrees) { return degrees * std::numbers::pi / 180.0; }

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
// a flat blue background (0.7, 0.8, 1.0) instead, but the gradient is prettier and keeps the first book's scenes unchanged.
export constexpr sky daylight_sky{.bottom = {1.0, 1.0, 1.0}, .top = {0.5, 0.7, 1.0}};

// Camera settings. They are convenient to set with designated initializers (C++20):
//   camera cam({.aspect_ratio = 16.0 / 9.0, .image_width = 400});
// Fields that aren't mentioned get their default values.
export struct camera_settings
{
    double aspect_ratio = 1.0;  // image width to height ratio
    int image_width = 100;      // image width in pixels
    int samples_per_pixel = 10; // how many random rays to average per pixel
    int max_depth = 10;         // how many times a ray may bounce before we give up on it

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

export class camera
{
public:
    explicit camera(const camera_settings& settings)
        : image_width_(settings.image_width),
          image_height_(std::max(1, static_cast<int>(settings.image_width / settings.aspect_ratio))),
          samples_per_pixel_(settings.samples_per_pixel),
          pixel_samples_scale_(1.0 / settings.samples_per_pixel),
          max_depth_(settings.max_depth),
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

    // Render: for each pixel average the colors of several rays and write the result into the image.
    // Rows are computed in parallel on all CPU threads.
    image render(const hittable& world) const
    {
        const unsigned thread_count = std::max(1u, std::thread::hardware_concurrency());
        std::println("Rendering {}×{}, rays per pixel: {}, threads: {}", image_width_, image_height_,
                     samples_per_pixel_, thread_count);

        image img(image_width_, image_height_);

        // Each thread takes the next free row until the rows run out. The counters are atomic
        // because several threads change them at once. The pixels, however, differ between threads (each writes
        // only to its own rows), so the image needs no protection.
        std::atomic<int> next_row = 0;
        std::atomic<int> rows_done = 0;
        auto render_rows = [&] {
            for (int j = next_row++; j < image_height_; j = next_row++)
            {
                // Each row has its own random number sequence: the image doesn't depend
                // on which thread computed the row or when, and is the same on every run.
                seed_random(static_cast<std::uint64_t>(j));
                for (int i = 0; i < image_width_; ++i)
                {
                    color pixel_color{0, 0, 0};
                    for (int sample = 0; sample < samples_per_pixel_; ++sample)
                    {
                        pixel_color += ray_color(get_ray(i, j), max_depth_, world);
                    }
                    img[i, j] = pixel_samples_scale_ * pixel_color;
                }
                ++rows_done;
                rows_done.notify_one(); // wake the main thread: time to update the progress
            }
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
        return img;
    }

private:
    int image_width_;
    int image_height_;
    int samples_per_pixel_;
    double pixel_samples_scale_; // 1 / samples_per_pixel: the averaging factor
    int max_depth_;
    double defocus_angle_;
    sky background_;
    point3 center_;              // camera position
    point3 pixel00_loc_;         // center of pixel (0, 0)
    vec3 pixel_delta_u_;         // step to the next pixel to the right
    vec3 pixel_delta_v_;         // step to the next pixel below
    vec3 defocus_disk_u_;        // horizontal radius of the "lens" disk
    vec3 defocus_disk_v_;        // vertical radius of the "lens" disk

    // A ray through a random point of pixel (i, j). If all rays went exactly through the center,
    // object edges would look like "staircases"; random points inside the pixel smooth them out.
    // The ray starts at a random point of the lens (or at the camera center if defocus is off).
    // The emission moment is random too: the shutter is open from time 0 to 1, and the pixel averages
    // everything a moving object managed to "sweep" in that time, which gives motion blur.
    ray get_ray(int i, int j) const
    {
        const vec3 offset = sample_square();
        const point3 pixel_sample =
            pixel00_loc_ + ((i + offset.x) * pixel_delta_u_) + ((j + offset.y) * pixel_delta_v_);
        const point3 ray_origin = defocus_angle_ <= 0 ? center_ : defocus_disk_sample();
        const double ray_time = random_double();
        return ray{ray_origin, pixel_sample - ray_origin, ray_time};
    }

    // Random point in the square [-0.5, 0.5] × [-0.5, 0.5]: an offset from the pixel center.
    static vec3 sample_square() { return {random_double() - 0.5, random_double() - 0.5, 0}; }

    // Random point on the "lens" disk around the camera center.
    point3 defocus_disk_sample() const
    {
        const vec3 p = random_in_unit_disk();
        return center_ + p.x * defocus_disk_u_ + p.y * defocus_disk_v_;
    }

    // The color a ray "brings back". Having hit an object, the ray is reflected or refracted,
    // and we recursively ask what color comes from there. depth is how many more bounces are allowed.
    color ray_color(const ray& r, int depth, const hittable& world) const
    {
        if (depth <= 0)
        {
            return color{0, 0, 0}; // too many bounces: assume no light reached the camera
        }

        // Look for hits with t from 0.001, not from 0: due to rounding errors the reflection point
        // ends up slightly below the surface, and the ray would immediately "hit" the same surface. In the image
        // this would give dark spots ("shadow acne").
        const auto rec = world.hit(r, interval{0.001, infinity});
        if (!rec)
        {
            return background_.in_direction(r.direction); // the ray went off to infinity and brought back the background color
        }

        // The light from the hit point is the light the surface emits by itself plus the reflected light:
        // the light that came along the scattered ray, attenuated by the material. The material decides what to do with the ray.
        const color emitted = rec->mat->emitted(rec->u, rec->v, rec->p);
        if (const auto s = rec->mat->scatter(r, *rec))
        {
            return emitted + s->attenuation * ray_color(s->scattered, depth - 1, world);
        }
        return emitted; // the ray was absorbed or hit a light source that reflects nothing
    }
};
