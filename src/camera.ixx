// Module camera: the camera shoots rays through pixels and builds an image from their colors.
export module camera;

import std;
import hittable;
import image;
import interval;
import random;
import ray;
import vec3;

// Camera settings. They are convenient to set with designated initializers (C++20):
//   camera cam({.aspect_ratio = 16.0 / 9.0, .image_width = 400});
// Fields that aren't mentioned get their default values.
export struct camera_settings
{
    double aspect_ratio = 1.0;  // image width to height ratio
    int image_width = 100;      // image width in pixels
    int samples_per_pixel = 10; // how many random rays to average per pixel
    int max_depth = 10;         // how many times a ray may bounce before we give up on it
};

export class camera
{
public:
    explicit camera(const camera_settings& settings)
        : image_width_(settings.image_width),
          image_height_(std::max(1, static_cast<int>(settings.image_width / settings.aspect_ratio))),
          samples_per_pixel_(settings.samples_per_pixel),
          pixel_samples_scale_(1.0 / settings.samples_per_pixel),
          max_depth_(settings.max_depth)
    {
        // The camera sits at the origin looking along -z. At distance focal_length in front of it stands
        // the viewport: a window with a ray shot through each of its pixels.
        const double focal_length = 1.0;
        const double viewport_height = 2.0;
        const double viewport_width = viewport_height * (static_cast<double>(image_width_) / image_height_);
        center_ = point3{0, 0, 0};

        // Vectors along the window's top edge (rightward) and left edge (downward: image rows go top to bottom)
        const vec3 viewport_u{viewport_width, 0, 0};
        const vec3 viewport_v{0, -viewport_height, 0};

        // Step between neighboring pixel centers
        pixel_delta_u_ = viewport_u / image_width_;
        pixel_delta_v_ = viewport_v / image_height_;

        // Upper-left corner of the window and the center of pixel (0, 0)
        const point3 viewport_upper_left = center_ - vec3{0, 0, focal_length} - viewport_u / 2 - viewport_v / 2;
        pixel00_loc_ = viewport_upper_left + 0.5 * (pixel_delta_u_ + pixel_delta_v_);
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
    point3 center_;              // camera position
    point3 pixel00_loc_;         // center of pixel (0, 0)
    vec3 pixel_delta_u_;         // step to the next pixel to the right
    vec3 pixel_delta_v_;         // step to the next pixel below

    // A ray from the camera through a random point of pixel (i, j). If all rays went exactly through
    // the center, object edges would look like "staircases"; random points inside the pixel smooth them out.
    ray get_ray(int i, int j) const
    {
        const vec3 offset = sample_square();
        const point3 pixel_sample =
            pixel00_loc_ + ((i + offset.x) * pixel_delta_u_) + ((j + offset.y) * pixel_delta_v_);
        return ray{center_, pixel_sample - center_};
    }

    // Random point in the square [-0.5, 0.5] × [-0.5, 0.5]: an offset from the pixel center.
    static vec3 sample_square() { return {random_double() - 0.5, random_double() - 0.5, 0}; }

    // The color a ray "brings back". Having hit an object, the ray bounces in a random direction,
    // and we recursively ask what color comes from there. depth is how many more bounces are allowed.
    static color ray_color(const ray& r, int depth, const hittable& world)
    {
        if (depth <= 0)
        {
            return color{0, 0, 0}; // too many bounces: assume no light reached the camera
        }

        // Look for hits with t from 0.001, not from 0: due to rounding errors the reflection point
        // ends up slightly below the surface, and the ray would immediately "hit" the same surface. In the image
        // this would give dark spots ("shadow acne").
        if (const auto rec = world.hit(r, interval{0.001, infinity}))
        {
            // Lambertian (diffuse) reflection: normal + a random unit vector.
            // Directions close to the normal come up more often, like on real matte surfaces.
            const vec3 direction = rec->normal + random_unit_vector();
            return 0.5 * ray_color(ray{rec->p, direction}, depth - 1, world); // half of the light is absorbed
        }

        // Sky: a smooth blend from white (bottom) to blue (top) by the height of the ray direction.
        const vec3 unit_direction = unit_vector(r.direction);
        const double a = 0.5 * (unit_direction.y + 1.0); // y from -1 to 1 → a from 0 to 1
        return (1.0 - a) * color{1.0, 1.0, 1.0} + a * color{0.5, 0.7, 1.0};
    }
};
