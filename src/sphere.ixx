// Module sphere: a sphere, stationary or moving.
export module sphere;

import std;
import aabb;
import hittable;
import interval;
import onb;
import pdf;
import random;
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

    // The sphere as plain numbers, for code that copies a scene into its own format: the GPU program.
    inline point3 center_at(double time) const { return center_.at(time); }
    inline bool is_moving() const { return center_.direction.length_squared() > 0; }
    inline double radius() const { return radius_; }
    inline const material* mat() const { return mat_.get(); } // nullptr for a shape in a scene's lights list

    std::optional<hit_record> hit(const ray& r, interval ray_t) const override
    {
        // Where the sphere's center was when the ray was emitted
        const point3 center = center_.at(r.time);

        // |P(t) - center|² = radius² is a quadratic equation a·t² + b·t + c = 0.
        // Writing b = -2h simplifies the root formula to (h ± √(h² - a·c)) / a.
        // In double this is accurate for every scene of the books. h² and a·c are both about a·|oc|², though, so for a
        // sphere ten million times smaller than its distance their difference loses most of its digits; the GPU, which
        // computes in float, gets the same number another way (see src/gpu/sphere.cuh).
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
        // Due to rounding |y| can slightly exceed 1, and acos would return NaN, hence std::clamp.
        const double phi = std::atan2(-outward_normal.z, outward_normal.x) + std::numbers::pi;
        const double theta = std::acos(std::clamp(-outward_normal.y, -1.0, 1.0));
        rec.u = phi / (2 * std::numbers::pi);
        rec.v = theta / std::numbers::pi;
        return rec;
    }

    // Book 3, sections 12.3–12.4: a sphere as a light. A uniform point of its surface, as for a quad, would often
    // lie on the back half, hidden behind the front one. But seen from a point outside, the sphere fills a cone of
    // directions around the direction to its center, with sin θmax = radius / distance, and random() draws
    // directions uniformly within that cone: the density is 1 / (its solid angle) = 1 / (2π·(1 - cos θmax)).
    // A moving sphere is aimed at where it is at time 0. The two functions still describe one distribution, so the
    // mixture strategy stays unbiased; its lamp half just aims less well.
    double pdf_value(const point3& origin, const vec3& direction) const override
    {
        const double distance_squared = (center_.at(0) - origin).length_squared();
        if (distance_squared < radius_ * radius_)
        {
            return uniform_sphere_density; // from inside: see random()
        }
        // Directions that miss the sphere have density 0: random() never draws them
        if (!hit(ray{origin, direction}, interval{0.001, infinity}))
        {
            return 0;
        }
        return 1 / (2 * std::numbers::pi * one_minus_cos_theta_max(distance_squared));
    }

    vec3 random(const point3& origin) const override
    {
        const vec3 to_center = center_.at(0) - origin;
        const double distance_squared = to_center.length_squared();
        if (distance_squared < radius_ * radius_)
        {
            // From inside, every direction hits the sphere: draw them all, uniformly. (The book's formula takes
            // the square root of 1 - radius² / distance², a negative number here, and gets NaN.)
            return random_unit_vector();
        }
        // A direction in the cone around the z axis, turned so that the z axis points at the center (chapter 8)
        const onb uvw{unit_vector(to_center)};
        return uvw.transform(random_to_sphere(one_minus_cos_theta_max(distance_squared)));
    }

private:
    // The center's path is a ray too: center_.at(time) = start point + time × displacement over the exposure.
    // A stationary sphere has zero displacement.
    ray center_;
    double radius_;
    std::shared_ptr<const material> mat_; // several objects can share one material, so ownership is shared
    aabb bbox_;

    // 1 - cos θmax of the cone seen from the distance √distance_squared, which is at least the radius. The book
    // computes 1 - √(1 - sin²θmax), which loses digits when the sphere looks small: the root is then close to 1.
    // Multiplied by (1 + cos θmax) / (1 + cos θmax), it becomes sin²θmax / (1 + cos θmax), which keeps them.
    double one_minus_cos_theta_max(double distance_squared) const
    {
        const double sin_squared = radius_ * radius_ / distance_squared;
        return sin_squared / (1 + std::sqrt(1 - sin_squared));
    }

    // A random direction, uniform within the cone of directions at most θmax away from the z axis, given
    // one_minus_cos_max = 1 - cos θmax. The inversion method of chapter 7: φ = 2π·r1, and θ has the cumulative
    // distribution (1 - cos θ) / (1 - cos θmax); setting it to r2 gives 1 - cos θ = r2·(1 - cos θmax).
    static vec3 random_to_sphere(double one_minus_cos_max)
    {
        const double r1 = random_double();
        const double r2 = random_double();
        const double phi = 2 * std::numbers::pi * r1;
        const double one_minus_z = r2 * one_minus_cos_max; // 1 - cos θ
        // sin θ = √(1 - z²), and 1 - z² = (1 - z)(1 + z): again without subtracting numbers close to 1
        const double sin_theta = std::sqrt(one_minus_z * (2 - one_minus_z));
        return {std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, 1 - one_minus_z};
    }
};
