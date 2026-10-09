// Module hittable: everything a ray can hit, and the hit record.
export module hittable;

import std;
import interval;
import ray;
import vec3;

// Where and how the ray hit an object.
export struct hit_record
{
    point3 p;                // hit point
    vec3 normal;             // unit-length normal, always facing against the ray
    double t = 0;            // ray parameter: p = r.at(t)
    bool front_face = false; // the ray came from outside the object (true) or from inside (false)

    // outward_normal is a unit-length normal pointing out of the object.
    // Turn it to face the ray and remember which side the ray came from.
    constexpr void set_face_normal(const ray& r, const vec3& outward_normal)
    {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// Interface of everything a ray can hit: spheres, object lists and so on.
export class hittable
{
public:
    virtual ~hittable() = default;

    // The nearest hit with parameter t strictly inside ray_t, or std::nullopt on a miss.
    virtual std::optional<hit_record> hit(const ray& r, interval ray_t) const = 0;
};

// An object list is a hittable too: it looks for the nearest hit among all objects.
export class hittable_list : public hittable
{
public:
    void add(std::unique_ptr<hittable> object) { objects_.push_back(std::move(object)); }

    std::optional<hit_record> hit(const ray& r, interval ray_t) const override
    {
        std::optional<hit_record> closest;
        for (const auto& object : objects_)
        {
            if (auto rec = object->hit(r, ray_t))
            {
                ray_t.max = rec->t; // no point looking farther than the hit already found
                closest = rec;
            }
        }
        return closest;
    }

private:
    std::vector<std::unique_ptr<hittable>> objects_; // the list owns the objects
};
