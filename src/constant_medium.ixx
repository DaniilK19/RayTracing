// Module constant_medium: smoke, fog, clouds, i.e. a volume uniformly filled with particles.
export module constant_medium;

import std;
import aabb;
import hittable;
import interval;
import material;
import random;
import ray;
import texture;
import vec3;

// A ray going through smoke collides with a particle on each short path segment dL with probability
// density·dL. So the distance to a collision is random and exponentially distributed:
// −ln(ξ) / density, where ξ is a random number from 0 to 1. If it is longer than the ray's path inside
// the volume, the ray passes through; otherwise at that point it "hit" a particle and scatters.
// The denser the smoke, the shorter the path to a collision and the less light the smoke lets through.
// The volume's shape is given by boundary, a convex object (sphere, box): a ray enters and leaves it once.
export class constant_medium : public hittable
{
public:
    constant_medium(std::unique_ptr<hittable> boundary, double density, std::shared_ptr<const texture> tex)
        : boundary_(std::move(boundary)),
          neg_inv_density_(-1 / density),
          phase_function_(std::make_shared<isotropic>(std::move(tex)))
    {
    }

    constant_medium(std::unique_ptr<hittable> boundary, double density, const color& albedo)
        : constant_medium(std::move(boundary), density, std::make_shared<solid_color>(albedo))
    {
    }

    std::optional<hit_record> hit(const ray& r, interval ray_t) const override
    {
        // Where the ray's line enters and leaves the volume. Search the whole line, not just ray_t:
        // the ray may start inside the smoke, and then the entry point is behind it.
        const auto hit_in = boundary_->hit(r, interval{-infinity, infinity});
        if (!hit_in)
        {
            return std::nullopt;
        }
        const auto hit_out = boundary_->hit(r, interval{hit_in->t + 0.0001, infinity});
        if (!hit_out)
        {
            return std::nullopt;
        }

        // Keep the part of the path inside the volume that lies in ray_t and ahead of the ray's origin
        double t_in = std::max(hit_in->t, ray_t.min);
        const double t_out = std::min(hit_out->t, ray_t.max);
        if (t_in >= t_out)
        {
            return std::nullopt;
        }
        t_in = std::max(t_in, 0.0);

        // t is measured in lengths of the direction vector, which isn't necessarily unit: convert to distances
        const double ray_length = r.direction.length();
        const double distance_inside = (t_out - t_in) * ray_length;
        const double hit_distance = neg_inv_density_ * std::log(random_double());
        if (hit_distance > distance_inside)
        {
            return std::nullopt; // passed through without colliding with anything
        }

        hit_record rec;
        rec.t = t_in + hit_distance / ray_length;
        rec.p = r.at(rec.t);
        rec.normal = vec3{1, 0, 0}; // a particle has no surface: the normal and the side don't matter, any values will do
        rec.front_face = true;
        rec.mat = phase_function_.get();
        return rec;
    }

    aabb bounding_box() const override { return boundary_->bounding_box(); }

private:
    std::unique_ptr<hittable> boundary_;
    double neg_inv_density_;                          // −1 / density
    std::shared_ptr<const material> phase_function_; // how a particle scatters light: isotropically
};
