// Module camera: the camera shoots rays through pixels and builds an image from their colors.
export module camera;

import std;
import hittable;
import image;
import interval;
import ray;
import vec3;

// Camera settings. They are convenient to set with designated initializers (C++20):
//   camera cam({.aspect_ratio = 16.0 / 9.0, .image_width = 400});
// Fields that aren't mentioned get their default values.
export struct camera_settings
{
    double aspect_ratio = 1.0; // image width to height ratio
    int image_width = 100;     // image width in pixels
};

export class camera
{
public:
    explicit camera(const camera_settings& settings)
        : image_width_(settings.image_width),
          image_height_(std::max(1, static_cast<int>(settings.image_width / settings.aspect_ratio)))
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

    // Render: shoot a ray for each pixel and write its color into the image.
    image render(const hittable& world) const
    {
        image img(image_width_, image_height_);
        for (int j = 0; j < image_height_; ++j)
        {
            std::print("\rRows remaining: {:4}", image_height_ - j);
            for (int i = 0; i < image_width_; ++i)
            {
                const point3 pixel_center = pixel00_loc_ + (i * pixel_delta_u_) + (j * pixel_delta_v_);
                const ray r{center_, pixel_center - center_};
                img[i, j] = ray_color(r, world);
            }
        }
        std::print("\r{:30}\r", ""); // erase the progress line
        return img;
    }

private:
    int image_width_;
    int image_height_;
    point3 center_;      // camera position
    point3 pixel00_loc_; // center of pixel (0, 0)
    vec3 pixel_delta_u_; // step to the next pixel to the right
    vec3 pixel_delta_v_; // step to the next pixel below

    static color ray_color(const ray& r, const hittable& world)
    {
        if (const auto rec = world.hit(r, interval{0, infinity}))
        {
            return 0.5 * (rec->normal + color{1, 1, 1}); // normal components -1..1 → color 0..1
        }

        // Sky: a smooth blend from white (bottom) to blue (top) by the height of the ray direction.
        const vec3 unit_direction = unit_vector(r.direction);
        const double a = 0.5 * (unit_direction.y + 1.0); // y from -1 to 1 → a from 0 to 1
        return (1.0 - a) * color{1.0, 1.0, 1.0} + a * color{0.5, 0.7, 1.0};
    }
};
