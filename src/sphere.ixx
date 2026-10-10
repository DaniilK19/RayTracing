// Module sphere: a sphere, stationary or moving.
export module sphere;

import std;
import aabb;
import hittable;
import interval;
import ray;
import vec3;

export class sphere : public hittable
{
public:
    // A stationary sphere is a special case of a moving one that doesn't go anywhere.
    // The constructor delegates the work to another constructor (C++11).
    sphere(const point3& center, double radius, std::shared_ptr<const material> mat)
        : sphere(center, center, radius, std::move(mat))
    {
    }

    // A moving sphere: while the shutter is open (time 0 to 1), the center moves uniformly from center1 to center2.
    sphere(const point3& center1, const point3& center2, double radius, std::shared_ptr<const material> mat)
        : center_{center1, center2 - center1}, radius_(std::fmax(0.0, radius)), mat_(std::move(mat))
    {
        // The box must contain the sphere at any moment: combine the boxes at the start and at the end of the path
        const vec3 radius_vector{radius_, radius_, radius_};
        const aabb box1 = aabb::from_corners(center_.at(0) - radius_vector, center_.at(0) + radius_vector);
        const aabb box2 = aabb::from_corners(center_.at(1) - radius_vector, center_.at(1) + radius_vector);
        bbox_ = enclose(box1, box2);
    }

    aabb bounding_box() const override { return bbox_; }

    std::optional<hit_record> hit(const ray& r, interval ray_t) const override
    {
        // Where the sphere's center was when the ray was emitted
        const point3 center = center_.at(r.time);

        // |P(t) - center|² = radius² is a quadratic equation a·t² + b·t + c = 0.
        // Writing b = -2h simplifies the root formula to (h ± √(h² - a·c)) / a.
        const vec3 oc = center - r.origin;
        const double a = r.direction.length_squared();
        const double h = dot(r.direction, oc);
        const double c = oc.length_squared() - radius_ * radius_;

        const double discriminant = h * h - a * c;
        if (discriminant < 0)
        {
            return std::nullopt;
        }

        // Take the nearest root that lies in the allowed interval
        const double sqrtd = std::sqrt(discriminant);
        double root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root))
        {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
            {
                return std::nullopt;
            }
        }

        hit_record rec;
        rec.t = root;
        rec.p = r.at(rec.t);
        const vec3 outward_normal = (rec.p - center) / radius_; // (p - center) / radius is the unit normal
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat_.get();

        // Texture coordinates work like longitude and latitude on a globe, both from 0 to 1.
        // The normal is a point on the unit sphere centered at the origin; we find the angles from it:
        // φ around the Y axis, measured from the -X direction; θ from the south pole (-Y) to the north one (+Y).
        const double phi = std::atan2(-outward_normal.z, outward_normal.x) + std::numbers::pi;
        const double theta = std::acos(-outward_normal.y);
        rec.u = phi / (2 * std::numbers::pi);
        rec.v = theta / std::numbers::pi;
        return rec;
    }

private:
    // The center's path is a ray too: center_.at(time) = start point + time × displacement over the exposure.
    // A stationary sphere has zero displacement.
    ray center_;
    double radius_;
    std::shared_ptr<const material> mat_; // several objects can share one material, so ownership is shared
    aabb bbox_;
};
