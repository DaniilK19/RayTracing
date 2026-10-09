// Module material: surface materials, i.e. how they reflect and refract light.
export module material;

import std;
import hittable;
import random;
import ray;
import vec3;

// Matte (diffuse) surface: scatters light in all directions, more often close to the normal.
export class lambertian : public material
{
public:
    explicit lambertian(const color& albedo) : albedo_(albedo) {}

    std::optional<scatter_result> scatter(const ray&, const hit_record& rec) const override
    {
        vec3 direction = rec.normal + random_unit_vector();

        // If the random vector is almost opposite to the normal, the sum is close to zero: that would give
        // a degenerate ray, then division by zero and NaN. In that case just reflect along the normal.
        if (direction.near_zero())
        {
            direction = rec.normal;
        }
        return scatter_result{albedo_, ray{rec.p, direction}};
    }

private:
    color albedo_; // fraction of light reflected per channel: the surface's "color"
};

// Metal: mirror reflection blurred by fuzz (0 is a perfect mirror, 1 is a very matte metal).
export class metal : public material
{
public:
    metal(const color& albedo, double fuzz) : albedo_(albedo), fuzz_(std::min(fuzz, 1.0)) {}

    std::optional<scatter_result> scatter(const ray& r_in, const hit_record& rec) const override
    {
        // Blur: move the tip of the reflected unit vector to a random point on a sphere of radius fuzz
        const vec3 reflected = unit_vector(reflect(r_in.direction, rec.normal)) + fuzz_ * random_unit_vector();

        // Blurring can send the ray below the surface; such a ray counts as absorbed
        if (dot(reflected, rec.normal) <= 0)
        {
            return std::nullopt;
        }
        return scatter_result{albedo_, ray{rec.p, reflected}};
    }

private:
    color albedo_;
    double fuzz_;
};
