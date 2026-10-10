// Module hittable: everything a ray can hit, the hit record, the material interface and hittable_pdf.
export module hittable;

import std;
import aabb;
import interval;
import pdf;
import random;
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
    double u = 0;                  // texture coordinates of the point on the object's surface
    double v = 0;
    bool front_face = false;       // the ray came from outside the object (true) or from inside (false)

    // outward_normal is a unit-length normal pointing out of the object.
    // Turn it to face the ray and remember which side the ray came from.
    constexpr void set_face_normal(const ray& r, const vec3& outward_normal)
    {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// A matte surface or a smoke particle scatters light into a whole range of directions, with a density given by
// the material's scattering_pdf. The material doesn't choose one of them (book 3, section 12.1): it hands that
// density to the camera as a pdf, and the camera's sampling strategy draws the direction, from it or another pdf.
export struct diffuse_scatter
{
    color attenuation; // fraction of light kept in each channel (R, G, B)
    material_pdf pdf;  // scattering_pdf as a pdf: the same density, and a way to draw directions with it
};

// A mirror (blurred or not) or glass: the camera simply follows the ray the material chose, as in the first two
// books. The weight pScatter / p is implicitly 1, as the direction comes from the material's own distribution
// (most renderers make mirrors such a special case).
export struct specular_scatter
{
    color attenuation;
    ray scattered;
};

// What happened to the ray after the hit: one of the two kinds of scattering, which the camera treats differently
// (book 3, chapter 6). std::variant (C++17) holds a value of exactly one of the listed types.
export using scatter_record = std::variant<diffuse_scatter, specular_scatter>;

// A material decides what happens to a ray at the hit point: reflection, refraction or absorption.
// A surface can also glow by itself.
export class material
{
public:
    virtual ~material() = default;

    // Ray r_in hit point rec. std::nullopt means the ray was absorbed.
    virtual std::optional<scatter_record> scatter(const ray& r_in, const hit_record& rec) const = 0;

    // pScatter in the book (chapter 5): the density, per steradian, with which the material scatters the light
    // of r_in into the direction of `scattered`. It lets the camera weigh a direction drawn with any other
    // density. 0 by default: mirrors and glass don't use it, their scatter() returns a specular_scatter
    // (a perfect mirror has no density at all: it would be infinite in one direction and 0 everywhere else).
    virtual double scattering_pdf(const ray& /*r_in*/, const hit_record& /*rec*/, const ray& /*scattered*/) const
    {
        return 0;
    }

    // Light the surface emits by itself at the hit point. rec gives the point, its texture coordinates (u, v) and
    // the side the ray came from (book 3, chapter 9: a lamp may glow on one side only).
    // Most materials don't glow, so for them this is black; light sources override the function.
    virtual color emitted(const hit_record& /*rec*/) const { return color{0, 0, 0}; }
};

// Interface of everything a ray can hit: spheres, object lists and so on.
export class hittable
{
public:
    virtual ~hittable() = default;

    // The nearest hit with parameter t strictly inside ray_t, or std::nullopt on a miss.
    virtual std::optional<hit_record> hit(const ray& r, interval ray_t) const = 0;

    // A box that fully contains the object (for the BVH)
    virtual aabb bounding_box() const = 0;

    // Light sampling (book 3, chapter 9): the camera can aim rays at an object, usually a lamp. random(origin)
    // draws a direction from origin toward the object, of length 1 (up to rounding), as the camera looks for hits
    // from t = 0.001, measured in direction lengths. pdf_value(origin, direction) is the density, per steradian,
    // with which random() draws that direction. The two must describe the same distribution, or the image is
    // biased. These defaults don't: no density anywhere, but a direction anyway. So only objects that override
    // both may be lights.
    virtual double pdf_value(const point3& /*origin*/, const vec3& /*direction*/) const { return 0; }
    virtual vec3 random(const point3& /*origin*/) const { return vec3{1, 0, 0}; }
};

// An object list is a hittable too: it looks for the nearest hit among all objects.
export class hittable_list : public hittable
{
public:
    void add(std::unique_ptr<hittable> object)
    {
        bbox_ = enclose(bbox_, object->bounding_box());
        objects_.push_back(std::move(object));
    }

    // Takes the objects out of the list, e.g. to build a BVH from them. `&&` after the parentheses means
    // it can only be called on a temporary or explicitly handed-over list: std::move(list).release_objects().
    std::vector<std::unique_ptr<hittable>> release_objects() &&
    {
        bbox_ = {};
        return std::exchange(objects_, {});
    }

    aabb bounding_box() const override { return bbox_; }

    inline bool empty() const { return objects_.empty(); }

    // A list of lamps draws a direction toward a randomly chosen one of them, each with probability 1/n. So the
    // density of a direction is the average of all the objects' densities: it could have been drawn toward any
    // of them whose solid angle contains it.
    // An empty list has nothing to aim at: density 0 everywhere and an arbitrary direction (the book's version
    // reads objects[0] of an empty vector here, undefined behavior). The camera never asks an empty list.
    double pdf_value(const point3& origin, const vec3& direction) const override
    {
        if (objects_.empty())
        {
            return 0;
        }
        double sum = 0;
        for (const auto& object : objects_)
        {
            sum += object->pdf_value(origin, direction);
        }
        return sum / static_cast<double>(objects_.size());
    }

    vec3 random(const point3& origin) const override
    {
        if (objects_.empty())
        {
            return vec3{1, 0, 0};
        }
        const int index = random_int(0, static_cast<int>(objects_.size()) - 1);
        return objects_[static_cast<std::size_t>(index)]->random(origin);
    }

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
    aabb bbox_;                                      // box around all objects
};

// Directions toward objects, usually the lamps, as a pdf (book 3, section 10.2): seen from origin, its density is
// the objects' pdf_value() and its directions are their random(). It lives here, not in the pdf module, because it
// needs hittable, while hittable needs pdf (for diffuse_scatter), and two modules can't import each other.
export class hittable_pdf
{
public:
    inline hittable_pdf(const hittable& objects, const point3& origin) : objects_(objects), origin_(origin) {}

    inline double value(const vec3& direction) const { return objects_.pdf_value(origin_, direction); }
    inline vec3 generate() const { return objects_.random(origin_); }

private:
    const hittable& objects_; // a reference, not a copy: the camera makes a hittable_pdf for one bounce only
    point3 origin_;
};
