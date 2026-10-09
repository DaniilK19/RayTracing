// Module hittable: everything a ray can hit, the hit record and the material interface.
export module hittable;

import std;
import interval;
import ray;
import vec3;

// hit_record and material refer to each other, so they live in one module: a class can't
// be declared in one module and defined in another. Only the material interface is here;
// concrete materials are in the material module.
export class material;

// Where and how the ray hit an object.
export struct hit_record
{
    point3 p;                      // hit point
    vec3 normal;                   // unit-length normal, always facing against the ray
    const material* mat = nullptr; // the object's material (a non-owning pointer: the object owns the material)
    double t = 0;                  // ray parameter: p = r.at(t)
    bool front_face = false;       // the ray came from outside the object (true) or from inside (false)

    // outward_normal is a unit-length normal pointing out of the object.
    // Turn it to face the ray and remember which side the ray came from.
    constexpr void set_face_normal(const ray& r, const vec3& outward_normal)
    {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// What happened to the ray after the hit: where it went next and how much it was attenuated.
export struct scatter_result
{
    color attenuation; // fraction of light kept in each channel (R, G, B)
    ray scattered;     // the new ray
};

// A material decides what happens to a ray at the hit point: reflection, refraction or absorption.
export class material
{
public:
    virtual ~material() = default;

    // Ray r_in hit point rec. std::nullopt means the ray was absorbed.
    virtual std::optional<scatter_result> scatter(const ray& r_in, const hit_record& rec) const = 0;
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
