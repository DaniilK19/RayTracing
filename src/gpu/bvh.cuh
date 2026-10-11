// The GPU's BVH (see bvh.ixx): built on the CPU as flat arrays, walked on the GPU with a loop and a small stack.
//
// The CPU's tree is made of objects that point to their children, and its hit() calls itself for them. Neither carries
// over: pointers into the CPU's memory mean nothing to the GPU, and recursion would need a deep call stack in every
// thread. So the tree is built into arrays, where nodes refer to each other by index, and the spheres are sorted so
// that each leaf's spheres lie next to each other. The GPU walks it in a loop, keeping the nodes still to visit on a
// stack.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "renderer.h"
#include "sphere.cuh"

namespace gpu
{
// A node: its box, and either its two children (an inner node) or its spheres (a leaf). 32 bytes in float, read with
// two 16-byte loads. The nodes are stored depth first: an inner node's first child comes right after it, so one index
// is enough for either kind of node: of the second child, or of the leaf's first sphere.
template <real_number Real>
struct alignas(16) bvh_node
{
    point3<Real> box_min;
    int index; // an inner node: its second child's index; a leaf: its first sphere's index
    point3<Real> box_max;
    int count; // a leaf: how many spheres it has; an inner node: 0

    __host__ __device__ bool is_leaf() const { return count > 0; }
};

// The traversal keeps the nodes it has yet to visit on a stack in each thread's memory. A median split halves the
// spheres at every level, so a tree over n spheres is about log2(n) levels deep, and the stack never holds more nodes
// than there are levels. build_bvh checks that the stack is deep enough (it is for any scene that fits in memory).
inline constexpr int bvh_stack_size = 32;

// The tree for the GPU: the nodes, and the spheres and their material indices in the order of the leaves
template <real_number Real>
struct bvh
{
    std::vector<bvh_node<Real>> nodes;
    std::vector<sphere<Real>> spheres;
    std::vector<int> sphere_materials; // the index of each sphere's material in scene_desc::materials
};

// Host code (it runs on the CPU, once, before the render) from here to build_bvh.

// A box in double, for building the tree
struct build_box
{
    std::array<double, 3> min{infinity<double>, infinity<double>, infinity<double>};
    std::array<double, 3> max{-infinity<double>, -infinity<double>, -infinity<double>};

    void enclose(const build_box& other)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            min[axis] = std::min(min[axis], other.min[axis]);
            max[axis] = std::max(max[axis], other.max[axis]);
        }
    }

    // As aabb::longest_axis(): 0 is x, 1 is y, 2 is z, and a tie goes to the later axis
    int longest_axis() const
    {
        const double x = max[0] - min[0];
        const double y = max[1] - min[1];
        const double z = max[2] - min[2];
        if (x > y)
        {
            return x > z ? 0 : 2;
        }
        return y > z ? 1 : 2;
    }
};

// A box coordinate in Real, rounded outward. The nearest float to a box's corner may lie inside the sphere's extent,
// and then a ray grazing the sphere there would miss the box and with it the sphere.
template <real_number Real>
Real round_down(double x)
{
    const auto rounded = static_cast<Real>(x);
    return rounded <= x ? rounded : std::nextafter(rounded, -infinity<Real>);
}

template <real_number Real>
Real round_up(double x)
{
    const auto rounded = static_cast<Real>(x);
    return rounded >= x ? rounded : std::nextafter(rounded, infinity<Real>);
}

template <real_number Real>
class bvh_builder
{
public:
    bvh_builder(std::span<const sphere_desc> spheres, int max_leaf_size) : max_leaf_size_(max_leaf_size)
    {
        // The spheres in Real first, and their boxes from those: the boxes must contain the spheres the GPU will test.
        // center ± radius may round inward, so each bound moves out by one step of double (std::nextafter).
        for (const sphere_desc& s : spheres)
        {
            const sphere<Real> converted{
                {static_cast<Real>(s.center.x), static_cast<Real>(s.center.y), static_cast<Real>(s.center.z)},
                static_cast<Real>(s.radius)};
            const double radius = converted.radius;
            const std::array<double, 3> center{converted.center.x, converted.center.y, converted.center.z};
            build_box box;
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                box.min[axis] = std::nextafter(center[axis] - radius, -infinity<double>);
                box.max[axis] = std::nextafter(center[axis] + radius, infinity<double>);
            }
            spheres_.push_back(converted);
            sphere_materials_.push_back(s.material);
            boxes_.push_back(box);
        }
    }

    bvh<Real> build() &&
    {
        std::vector<int> order(spheres_.size());
        std::iota(order.begin(), order.end(), 0); // 0, 1, 2, ...: sphere indices, sorted below
        if (!order.empty())
        {
            add_subtree(order, 1);
        }
        return std::move(tree_);
    }

private:
    int max_leaf_size_;
    std::vector<sphere<Real>> spheres_;
    std::vector<int> sphere_materials_;
    std::vector<build_box> boxes_;
    bvh<Real> tree_;

    // Adds the subtree over the spheres in `indices` to the nodes, depth first, and returns its root's index. Recursion
    // is fine on the CPU: it goes as deep as the tree, about log2 of the number of spheres.
    int add_subtree(std::span<int> indices, int level)
    {
        if (level > bvh_stack_size)
        {
            throw std::runtime_error("the BVH is too deep for the GPU's traversal stack");
        }
        build_box box;
        for (const int i : indices)
        {
            box.enclose(boxes_[static_cast<std::size_t>(i)]);
        }
        const auto index = static_cast<int>(tree_.nodes.size());
        tree_.nodes.push_back({.box_min = {round_down<Real>(box.min[0]), round_down<Real>(box.min[1]),
                                           round_down<Real>(box.min[2])},
                               .index = 0,
                               .box_max = {round_up<Real>(box.max[0]), round_up<Real>(box.max[1]),
                                           round_up<Real>(box.max[2])},
                               .count = 0});

        if (std::ssize(indices) <= max_leaf_size_)
        {
            // A leaf: its spheres go to the end of the sphere array, so the leaves' ranges follow each other
            tree_.nodes[static_cast<std::size_t>(index)].index = static_cast<int>(tree_.spheres.size());
            tree_.nodes[static_cast<std::size_t>(index)].count = static_cast<int>(indices.size());
            for (const int i : indices)
            {
                tree_.spheres.push_back(spheres_[static_cast<std::size_t>(i)]);
                tree_.sphere_materials.push_back(sphere_materials_[static_cast<std::size_t>(i)]);
            }
            return index;
        }

        // The CPU's heuristic (see build in bvh.ixx): split in half along the longest axis of the common box, sorted
        // by where the spheres' boxes start on it
        const auto axis = static_cast<std::size_t>(box.longest_axis());
        std::ranges::sort(indices, std::less{}, [&](int i) { return boxes_[static_cast<std::size_t>(i)].min[axis]; });
        const std::size_t middle = indices.size() / 2;
        add_subtree(indices.first(middle), level + 1); // the first child: right after this node
        // push_back may have moved the nodes to new memory, so the node is looked up again by its index
        tree_.nodes[static_cast<std::size_t>(index)].index = add_subtree(indices.subspan(middle), level + 1);
        return index;
    }
};

// Builds the tree with leaves of at most max_leaf_size spheres. The CPU's leaves are single objects; renderer.cu
// explains the sizes the GPU uses.
template <real_number Real>
bvh<Real> build_bvh(std::span<const sphere_desc> spheres, int max_leaf_size)
{
    return bvh_builder<Real>(spheres, max_leaf_size).build();
}

// Device code (it runs on the GPU) from here on.

// The tree as the kernel sees it: where its arrays are in the GPU's memory
template <real_number Real>
struct bvh_view
{
    const bvh_node<Real>* nodes;
    const sphere<Real>* spheres;
    int node_count; // 0 for an empty scene
};

// 1 / x for the slab test below, never infinite: an x closer to 0 than 2^-80 is moved out to ±2^-80 first, as in Aila
// and Laine's GPU traversal kernels. A direction component can be exactly 0 (about one ray in 40 million in the weekend
// scene in float): a camera ray whose lens point and pixel sample round to the same float in one coordinate, for one.
// Its inverse would be infinite, and box_entry's box · ∞ - origin · ∞ gives ∞ - ∞ = NaN. For a box that spans 0 on
// that axis while the ray runs inside it, the slab then comes out as [-∞, -∞] or [∞, ∞] (one distance infinite, the
// other NaN, which fmin and fmax skip), and the box is rejected: the ray could miss the ground right in front of it.
// With ±2^80 every distance stays finite, a ray inside the slab gets two huge distances of opposite signs, as the exact
// test would, and only the slab test sees the change.
template <real_number Real>
__device__ Real safe_inverse(Real x)
{
    const Real tiny = Real(0x1p-80); // a hexadecimal float literal: 2^-80
    return Real(1) / (fabs(x) > tiny ? x : copysign(tiny, x));
}

// The slab test (see aabb.ixx): the distance at which the ray enters the box, or infinity if it misses the box or
// enters it only beyond t_max. The CPU computes a slab's distances as (box - origin) · (1 / direction). Here the
// caller divides once per ray: inverse_direction = 1 / direction and origin_scaled = origin / direction, so each slab
// costs one multiply-add, box · (1 / direction) - origin / direction. Unlike the CPU's form, this one needs a finite
// inverse_direction (see safe_inverse).
template <real_number Real>
__device__ Real box_entry(const bvh_node<Real>& node, const vec3<Real>& inverse_direction,
                          const vec3<Real>& origin_scaled, Real t_min, Real t_max)
{
    const Real x0 = node.box_min.x * inverse_direction.x - origin_scaled.x;
    const Real x1 = node.box_max.x * inverse_direction.x - origin_scaled.x;
    const Real y0 = node.box_min.y * inverse_direction.y - origin_scaled.y;
    const Real y1 = node.box_max.y * inverse_direction.y - origin_scaled.y;
    const Real z0 = node.box_min.z * inverse_direction.z - origin_scaled.z;
    const Real z1 = node.box_max.z * inverse_direction.z - origin_scaled.z;
    const Real enter = fmax(fmax(fmin(x0, x1), fmin(y0, y1)), fmax(fmin(z0, z1), t_min));
    const Real exit = fmin(fmin(fmax(x0, x1), fmax(y0, y1)), fmin(fmax(z0, z1), t_max));
    return enter <= exit ? enter : infinity<Real>;
}

// What closest_hit found: the sphere's index in the GPU's order (-1: none), the distance to it, and whether the ray
// enters the sphere there (see hit_sphere)
template <real_number Real>
struct bvh_hit
{
    int index;
    Real t;
    bool entering;
};

// The closest sphere the ray hits at a distance t in (0.001, ∞): the CPU's world.hit(r, interval{0.001, infinity}),
// except for the sphere the ray starts on, from_sphere (-1 for a camera ray), which follows the self-intersection rule:
//
// In float, a point on the ground sphere (radius 1000, centered at y = -1000) is known relative to that center only to
// about 3·10^-5, half the distance between neighboring floats near 1000, and a computed hit point is often further off
// the surface than that. So a ray that bounces off the ground may start slightly inside it, and t > 0.001 isn't
// enough: a ray leaving at a grazing angle, below about 2°, meets the ground again from inside beyond t = 0.001, gets
// trapped there, and its light is lost. So for the sphere a ray starts on, the test goes by geometry instead of by
// distance: a ray going outward (inward = false) can't hit it again, as a sphere is convex, and a ray going inward
// (into glass, or back into a sphere it is inside) hits it only at the far root. In exact arithmetic this changes
// nothing: for a point p on the sphere with outward normal n, |p + t·d - center|² = radius² + 2t·radius·(d·n) +
// t²·|d|², which equals radius² only at t = 0 and at t = -2·radius·(d·n) / |d|², a hit ahead only if d·n < 0. Other
// spheres keep the CPU's t > 0.001. (trace_ray goes by geometry too, to tell which side of a sphere a ray hit.)
// Measured with the same random numbers, with and without the rule: without it the weekend scene in float came out
// 0.011% darker and a view of the ground alone 0.06% darker (about as many of the ground's bounces were lost), far
// beyond the noise of a long render; in double the images were identical.
//
// The rule covers only the sphere the ray leaves. Where two spheres touch, as a ball resting on the ground, a ray that
// leaves one of them is tested against the other one with t > 0.001 alone, and that t is computed in float, with an
// error of about 3·10^-5 at radius 1000 as above. So in float a slightly larger part of the gap near the contact traps
// paths (a gap narrower than 0.001 hides the other sphere's near side). In a test made to show it, a perfect mirror
// ball on a perfect mirror ground of radius 1000 under a uniform white sky, float lost 3.7% more paths there than
// double: about 3·10^-6 of the image's light, invisible. Such errors grow with the coordinates, as float resolves
// about 10^-7 of the largest one, so what matters is how small a scene's details are compared to its coordinates:
// details of 10^-4 of them and less come out measurably different in float. A ground of radius 10^4 or more, or a
// scene 10^4 or more away from the origin, with balls of radius 1 as in the books, needs double (RayTracingGPU
// --fp64), and so do balls of radius 10^-3 at 10 from the origin.
template <real_number Real>
__device__ bvh_hit<Real> closest_hit(const bvh_view<Real>& tree, const ray<Real>& r, int from_sphere, bool inward)
{
    const Real t_min = Real(0.001);
    bvh_hit<Real> closest{.index = -1, .t = infinity<Real>, .entering = false};
    if (tree.node_count == 0)
    {
        return closest;
    }

    const Real inverse_a = Real(1) / r.direction.length_squared();
    const vec3<Real> inverse_direction{safe_inverse(r.direction.x), safe_inverse(r.direction.y),
                                       safe_inverse(r.direction.z)};
    const vec3<Real> origin_scaled = r.origin * inverse_direction;

    // The nodes still to visit, with the distance at which the ray enters each one's box: a node the ray enters only
    // beyond the closest hit found since it was put here is skipped. An array indexed by a variable can't live in
    // registers, so it goes to the thread's local memory (cached, but slower than registers).
    struct stack_entry
    {
        int node;
        Real t;
    };
    stack_entry stack[bvh_stack_size];
    int stack_size = 0;

    int node_index = 0; // the root
    while (node_index >= 0)
    {
        const bvh_node<Real> node = tree.nodes[node_index];
        if (node.is_leaf())
        {
            for (int i = node.index; i < node.index + node.count; ++i)
            {
                const bool own = i == from_sphere;
                if (own && !inward)
                {
                    continue; // the self-intersection rule: a ray going out of a sphere can't hit it
                }
                Real t;
                bool entering;
                if (hit_sphere(tree.spheres[i], r, inverse_a, t_min, closest.t, own, t, entering))
                {
                    closest = {.index = i, .t = t, .entering = entering};
                }
            }
            node_index = -1; // continue from the stack
        }
        else
        {
            // Look into both children's boxes and descend into the nearer one first: a hit found there spares testing
            // the other one's spheres that lie beyond it. The farther one waits on the stack, if the ray enters it.
            // (On the weekend scene this took 0.76 s instead of 0.86 s: 13% faster than always taking the first child
            // first.)
            const int first = node_index + 1; // depth first: right after its parent
            const int second = node.index;
            const Real t_first = box_entry(tree.nodes[first], inverse_direction, origin_scaled, t_min, closest.t);
            const Real t_second = box_entry(tree.nodes[second], inverse_direction, origin_scaled, t_min, closest.t);
            const bool second_is_nearer = t_second < t_first;
            const Real t_far = fmax(t_first, t_second);
            if (t_far < infinity<Real>)
            {
                stack[stack_size++] = {second_is_nearer ? first : second, t_far};
            }
            node_index = fmin(t_first, t_second) < infinity<Real> ? (second_is_nearer ? second : first) : -1;
        }

        // Nowhere to descend: take the next node from the stack that the ray enters before the closest hit so far
        while (node_index < 0 && stack_size > 0)
        {
            const stack_entry entry = stack[--stack_size];
            if (entry.t < closest.t)
            {
                node_index = entry.node;
            }
        }
    }
    return closest;
}
}
