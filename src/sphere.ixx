// Module sphere: a sphere, the only kind of object in the first book's scenes.
export module sphere;

import std;
import hittable;
import interval;
import ray;
import vec3;

export class sphere : public hittable
{
public:
    sphere(const point3& center, double radius) : center_(center), radius_(std::fmax(0.0, radius)) {}

    std::optional<hit_record> hit(const ray& r, interval ray_t) const override
    {
        // |P(t) - center|² = radius² is a quadratic equation a·t² + b·t + c = 0.
        // Writing b = -2h simplifies the root formula to (h ± √(h² - a·c)) / a.
        const vec3 oc = center_ - r.origin;
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
        rec.set_face_normal(r, (rec.p - center_) / radius_); // (p - center) / radius is the unit normal
        return rec;
    }

private:
    point3 center_;
    double radius_;
};
