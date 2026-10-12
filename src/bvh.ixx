// Module bvh: the Bounding Volume Hierarchy from the second book of the series.
//
// Without it every ray is tested against every object in the scene; in the first book's cover (the weekend
// scene) that is almost 500 spheres per ray. The BVH arranges objects into a binary tree of boxes: if the ray misses
// a node's box, the whole subtree is skipped. Instead of N tests it takes on the order of log₂N.
export module bvh;

import std;
import aabb;
import hittable;
import interval;
import ray;

// A tree node: two subtrees and a common box around them. The tree's leaves are the scene objects themselves.
// To render, the tree is just a hittable, which build_bvh() builds. The class is exported only so that code that
// copies a scene into its own format can look inside it (left and right below).
export class bvh_node : public hittable
{
public:
    bvh_node(std::unique_ptr<hittable> left, std::unique_ptr<hittable> right)
        : left_(std::move(left)), right_(std::move(right)),
          bbox_(enclose(left_->bounding_box(), right_->bounding_box()))
    {
    }

    std::optional<hit_record> hit(const ray& r, interval ray_t) const override
    {
        if (!bbox_.hit(r, ray_t))
        {
            return std::nullopt; // the ray misses the box, so it misses everything inside it too
        }

        const auto hit_left = left_->hit(r, ray_t);
        // In the right subtree look only for what is closer than the hit already found in the left one
        const auto hit_right = right_->hit(r, interval{ray_t.min, hit_left ? hit_left->t : ray_t.max});
        return hit_right ? hit_right : hit_left;
    }

    aabb bounding_box() const override { return bbox_; }

    // For code that copies a scene into its own format: the GPU program, which takes the objects at the tree's leaves
    // and builds a tree of its own over all of the world's shapes
    inline const hittable& left() const { return *left_; }
    inline const hittable& right() const { return *right_; }

private:
    std::unique_ptr<hittable> left_;
    std::unique_ptr<hittable> right_;
    aabb bbox_;
};

// Builds a tree from the objects, taking ownership of them out of the span.
std::unique_ptr<hittable> build(std::span<std::unique_ptr<hittable>> objects)
{
    if (objects.size() == 1)
    {
        return std::move(objects.front()); // a tree leaf is the object itself
    }

    // Split the objects in half along the longest axis of their common box: sort by the start
    // of the box on that axis (std::ranges::sort with a projection, C++20) and cut the list in the middle.
    aabb bbox;
    for (const auto& object : objects)
    {
        bbox = enclose(bbox, object->bounding_box());
    }
    const int axis = bbox.longest_axis();
    std::ranges::sort(objects, std::less{},
                      [axis](const auto& object) { return object->bounding_box().axis_interval(axis).min; });

    const std::size_t middle = objects.size() / 2;
    return std::make_unique<bvh_node>(build(objects.first(middle)), build(objects.subspan(middle)));
}

// Rebuilds an object list into a BVH. Returns the tree root, which can be rendered instead of the list.
export std::unique_ptr<hittable> build_bvh(hittable_list list)
{
    std::vector<std::unique_ptr<hittable>> objects = std::move(list).release_objects();
    if (objects.empty())
    {
        return std::make_unique<hittable_list>(); // empty scene: nothing to build a tree from
    }
    return build(objects);
}
