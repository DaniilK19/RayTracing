// Module aabb: an axis-aligned bounding box.
// Testing a ray against a box is much cheaper than against a complex object or a group of objects
// inside it: if the ray misses the box, all of its contents can be skipped. The BVH is built on this.
export module aabb;

import std;
import interval;
import ray;
import vec3;

export struct aabb
{
    interval x, y, z; // all intervals are empty by default: an empty box

    // Box with opposite corners a and b
    static constexpr aabb from_corners(const point3& a, const point3& b)
    {
        return {axis_span(a.x, b.x), axis_span(a.y, b.y), axis_span(a.z, b.z)};
    }

    constexpr const interval& axis_interval(int axis) const { return axis == 0 ? x : axis == 1 ? y : z; }

    // Index of the box's longest axis: 0 is x, 1 is y, 2 is z
    constexpr int longest_axis() const
    {
        if (x.size() > y.size())
        {
            return x.size() > z.size() ? 0 : 2;
        }
        return y.size() > z.size() ? 1 : 2;
    }

    // The "slab" method: a box is the intersection of three slabs between pairs of parallel faces.
    // For each slab find the segment of t where the ray is inside it; the ray hits the box
    // if all three segments (and ray_t) have a common part.
    constexpr bool hit(const ray& r, interval ray_t) const
    {
        return hit_slab(x, r.origin.x, r.direction.x, ray_t) && hit_slab(y, r.origin.y, r.direction.y, ray_t) &&
               hit_slab(z, r.origin.z, r.direction.z, ray_t);
    }

private:
    // The segment between a and b on one axis, no thinner than min_size. For a flat object (e.g.
    // a quad in the z = 0 plane) the box would otherwise have zero thickness: the t segment inside
    // such a "slab" degenerates to a point, hit_slab would reject every ray, and the object would vanish from the BVH.
    static constexpr interval axis_span(double a, double b)
    {
        constexpr double min_size = 0.0001;
        const interval span{std::min(a, b), std::max(a, b)};
        return span.size() < min_size ? span.expand(min_size) : span;
    }

    // Narrows ray_t to the segment where the ray is between the slab's faces. false means no common part.
    // If the ray is parallel to the faces (direction = 0), the division gives ±∞, and the comparisons below still work.
    static constexpr bool hit_slab(const interval& slab, double origin, double direction, interval& ray_t)
    {
        const double inverse = 1.0 / direction;
        double t0 = (slab.min - origin) * inverse;
        double t1 = (slab.max - origin) * inverse;
        if (t0 > t1)
        {
            std::swap(t0, t1); // the ray goes in the negative direction of the axis
        }
        ray_t.min = std::max(ray_t.min, t0);
        ray_t.max = std::min(ray_t.max, t1);
        return ray_t.min < ray_t.max;
    }
};

// The box shifted by offset
export constexpr aabb operator+(const aabb& box, const vec3& offset)
{
    return {box.x + offset.x, box.y + offset.y, box.z + offset.z};
}

// The smallest box containing both given ones
export constexpr aabb enclose(const aabb& a, const aabb& b)
{
    return {enclose(a.x, b.x), enclose(a.y, b.y), enclose(a.z, b.z)};
}
