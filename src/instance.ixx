// Module instance: instances, i.e. a ready object, translated or rotated.
//
// Moving the object itself is inconvenient: a box would need all six faces recomputed, and a sphere
// has nothing to recompute after a rotation at all. Instead we move the ray the opposite way: hitting
// the original object with a ray moved back is the same as hitting the moved object with the original ray.
// The found point and normal are then brought back into the world. Wrappers can be nested.
export module instance;

import std;
import aabb;
import hittable;
import interval;
import ray;
import vec3;

// An object translated by offset.
export class translate : public hittable
{
public:
    translate(std::unique_ptr<hittable> object, const vec3& offset)
        : object_(std::move(object)), offset_(offset), bbox_(object_->bounding_box() + offset)
    {
    }

    std::optional<hit_record> hit(const ray& r, interval ray_t) const override
    {
        // Move the ray's origin back; the direction and t don't change with translation
        const ray moved{r.origin - offset_, r.direction, r.time};
        auto rec = object_->hit(moved, ray_t);
        if (rec)
        {
            rec->p += offset_; // bring the hit point back into the world
        }
        return rec;
    }

    aabb bounding_box() const override { return bbox_; }

    // For code that copies a scene into its own format: the GPU program
    inline const hittable& object() const { return *object_; }
    inline const vec3& offset() const { return offset_; }

    // Light sampling (book 3, chapter 9) the same way: move the origin back. Directions don't change, and neither
    // do the solid angles they cover, so the density and the drawn direction are the original object's. (Beyond
    // the book, whose instances keep hittable's defaults and can't be lights.)
    double pdf_value(const point3& origin, const vec3& direction) const override
    {
        return object_->pdf_value(origin - offset_, direction);
    }

    vec3 random(const point3& origin) const override { return object_->random(origin - offset_); }

private:
    std::unique_ptr<hittable> object_;
    vec3 offset_;
    aabb bbox_;
};

// An object rotated around the Y axis by angle degrees (counterclockwise when viewed from above).
export class rotate_y : public hittable
{
public:
    rotate_y(std::unique_ptr<hittable> object, double angle)
        : object_(std::move(object)),
          sin_theta_(std::sin(degrees_to_radians(angle))),
          cos_theta_(std::cos(degrees_to_radians(angle)))
    {
        // The rotated object's box: rotate all 8 corners of the original box
        // and take the box around them. A for loop over the list {a, b} goes through two values.
        const aabb box = object_->bounding_box();
        for (const double x : {box.x.min, box.x.max})
        {
            for (const double y : {box.y.min, box.y.max})
            {
                for (const double z : {box.z.min, box.z.max})
                {
                    const point3 corner = to_world(point3{x, y, z});
                    bbox_ = enclose(bbox_, aabb::from_corners(corner, corner));
                }
            }
        }
    }

    std::optional<hit_record> hit(const ray& r, interval ray_t) const override
    {
        // Rotation doesn't change lengths, so t in the rotated ray stays the same
        const ray rotated{to_object(r.origin), to_object(r.direction), r.time};
        auto rec = object_->hit(rotated, ray_t);
        if (rec)
        {
            rec->p = to_world(rec->p);
            rec->normal = to_world(rec->normal);
        }
        return rec;
    }

    aabb bounding_box() const override { return bbox_; }

    // For code that copies a scene into its own format: the GPU program
    inline const hittable& object() const { return *object_; }
    inline double sin_theta() const { return sin_theta_; }
    inline double cos_theta() const { return cos_theta_; }

    // Light sampling: turn the origin and the direction into the object's coordinates, and the drawn direction
    // back into the world's. A rotation keeps all angles, so solid angles, and densities per steradian, stay the same.
    double pdf_value(const point3& origin, const vec3& direction) const override
    {
        return object_->pdf_value(to_object(origin), to_object(direction));
    }

    vec3 random(const point3& origin) const override { return to_world(object_->random(to_object(origin))); }

private:
    std::unique_ptr<hittable> object_;
    double sin_theta_;
    double cos_theta_;
    aabb bbox_;

    // From the world into the object's coordinates: rotation by −θ
    vec3 to_object(const vec3& v) const
    {
        return {cos_theta_ * v.x - sin_theta_ * v.z, v.y, sin_theta_ * v.x + cos_theta_ * v.z};
    }

    // From the object's coordinates into the world: rotation by +θ
    vec3 to_world(const vec3& v) const
    {
        return {cos_theta_ * v.x + sin_theta_ * v.z, v.y, -sin_theta_ * v.x + cos_theta_ * v.z};
    }
};
