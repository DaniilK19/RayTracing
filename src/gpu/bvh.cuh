// The GPU's BVH (see bvh.ixx): built on the CPU as flat arrays, walked on the GPU with a loop and a small stack.
//
// The CPU's tree is made of objects that point to their children, and its hit() calls itself for them. Neither carries
// over: pointers into the CPU's memory mean nothing to the GPU, and recursion would need a deep call stack in every
// thread. So the tree is built into arrays, where nodes refer to each other by index, and the primitives (the shapes
// rays hit) are sorted so that each leaf's primitives lie next to each other. The GPU walks it in a loop, keeping the
// nodes still to visit on a stack.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "quad.cuh"
#include "renderer.h"
#include "sphere.cuh"

namespace gpu
{
// A node: its box, and either its two children (an inner node) or its primitives (a leaf). 32 bytes in float, read
// with two 16-byte loads. The nodes are stored depth first: an inner node's first child comes right after it, so one
// index is enough for either kind of node: of the second child, or of the leaf's first primitive.
template <real_number Real>
struct alignas(16) bvh_node
{
    point3<Real> box_min;
    int index; // an inner node: its second child's index; a leaf: its first primitive's index in its kind's array
    point3<Real> box_max;
    int count; // a leaf: how many primitives it has, and of which kind (see leaf_size and leaf_kind); an inner node: 0

    __host__ __device__ bool is_leaf() const { return count > 0; }
};

// The kinds of primitives in the tree. Each kind has an array of its own (see bvh_view), so that a sphere that stands
// still is still read with one 16-byte load in float (two in double, against four for a moving one), and a leaf holds
// primitives of one kind.
enum class primitive_kind
{
    sphere,        // a sphere that stands still. 0, so that a leaf of n of them has count n: a tree of spheres that
                   // stand still is exactly the first book's
    moving_sphere, // a sphere that moves while the shutter is open
    quad,          // a parallelogram
};

// A leaf's count holds two numbers, n | kind << 8: how many primitives it has in the low 8 bits (the builder makes at
// most 8) and their kind above them
inline constexpr int leaf_kind_shift = 8;

template <real_number Real>
__host__ __device__ int leaf_size(const bvh_node<Real>& node)
{
    return node.count & ((1 << leaf_kind_shift) - 1);
}

template <real_number Real>
__host__ __device__ primitive_kind leaf_kind(const bvh_node<Real>& node)
{
    return static_cast<primitive_kind>(node.count >> leaf_kind_shift);
}

// The traversal keeps the nodes it has yet to visit on a stack in each thread's memory. A median split halves the
// primitives at every level, so a tree over n primitives is about log2(n) levels deep, and the stack never holds more
// nodes than there are levels. build_bvh checks that the stack is deep enough (it is for any scene that fits in
// memory).
inline constexpr int bvh_stack_size = 32;

// Which kernel renders (see the kernels in renderer.cu): the first book's, for scenes that need nothing beyond the
// first book, or the one for all three books
enum class kernel_kind
{
    first_book, // spheres that stand still; lambertian with one color, metal, dielectric; the sky
    all_books,  // everything else too
};

// The tree for the GPU: the nodes, each kind's primitives in the order of the leaves, and what only the closest hit
// reads about each primitive, its material and (a sphere's) frame. Every primitive has a global id, one number that
// tells the kernel which it is: the spheres that stand still are 0, 1, 2... in their array's order, the moving spheres
// follow them from moving_base on, and the quads from quad_base on. The arrays of what the hit reads are indexed by it.
template <real_number Real>
struct bvh
{
    std::vector<bvh_node<Real>> nodes;
    std::vector<sphere<Real>> spheres;
    std::vector<moving_sphere<Real>> moving_spheres;
    std::vector<quad<Real>> quads;
    std::vector<int> primitive_materials;          // by global id: its material's index in scene_desc::materials
    std::vector<sphere_frame<Real>> sphere_frames; // by global id, for the spheres of both kinds
    int moving_base = 0;                           // the global id of moving_spheres[0]: how many spheres stand still
    int quad_base = 0;                             // the global id of quads[0]: how many spheres there are
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
    bvh_builder(std::span<const sphere_desc> spheres, std::span<const quad_desc> quads, int max_leaf_size)
        : max_leaf_size_(max_leaf_size)
    {
        // The spheres that stand still first, in the scene's order, then the moving ones. A scene whose spheres all
        // stand still then gives the builder the input of the first book's renderer, and so the same tree.
        for (const sphere_desc& s : spheres)
        {
            if (is_stationary(s))
            {
                add_sphere(s);
            }
        }
        tree_.moving_base = static_cast<int>(inputs_.size());
        for (const sphere_desc& s : spheres)
        {
            if (!is_stationary(s))
            {
                add_moving_sphere(s);
            }
        }
        tree_.quad_base = static_cast<int>(inputs_.size());
        for (const quad_desc& q : quads)
        {
            add_quad(q);
        }
    }

    bvh<Real> build() &&
    {
        tree_.primitive_materials.resize(inputs_.size());
        tree_.sphere_frames.resize(static_cast<std::size_t>(tree_.quad_base));
        std::vector<int> order(inputs_.size());
        std::iota(order.begin(), order.end(), 0); // 0, 1, 2, ...: indices into inputs_, sorted below
        if (!order.empty())
        {
            add_subtree(order, 1);
        }
        return std::move(tree_);
    }

private:
    // A primitive of the builder's input: its kind, where it is in its kind's array below, and what the tree needs
    // to know about it
    struct input
    {
        primitive_kind kind;
        int index; // in spheres_, moving_spheres_ or quads_
        int material;
        sphere_frame<Real> frame; // a sphere's
        build_box box;
    };

    int max_leaf_size_;
    std::vector<input> inputs_;
    std::vector<sphere<Real>> spheres_;
    std::vector<moving_sphere<Real>> moving_spheres_;
    std::vector<quad<Real>> quads_;
    bvh<Real> tree_;

    static bool is_stationary(const sphere_desc& s)
    {
        return s.displacement.x == 0 && s.displacement.y == 0 && s.displacement.z == 0;
    }

    // The box, in double, of a sphere in Real that goes from one center to another: the box must contain the spheres
    // the GPU will test. center ± radius may round inward, so each bound moves out by one step of double
    // (std::nextafter).
    static build_box sphere_box(const point3<Real>& start, const point3<Real>& end, Real radius)
    {
        const std::array<double, 3> a{start.x, start.y, start.z};
        const std::array<double, 3> b{end.x, end.y, end.z};
        build_box box;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            box.min[axis] = std::nextafter(std::min(a[axis], b[axis]) - radius, -infinity<double>);
            box.max[axis] = std::nextafter(std::max(a[axis], b[axis]) + radius, infinity<double>);
        }
        return box;
    }

    void add_sphere(const sphere_desc& s)
    {
        const sphere<Real> converted{to_real<Real>(s.center), static_cast<Real>(s.radius)};
        // The frame's numbers (see onto_sphere), in double from the sphere in Real: the squares of floats are exact in
        // double, so for the ground (center (0, -1000, 0), radius 1000) k comes out exactly 0
        const double x = converted.center.x;
        const double y = converted.center.y;
        const double z = converted.center.z;
        const double radius = converted.radius;
        const double center_squared = x * x + y * y + z * z;
        const sphere_frame<Real> frame{.k = static_cast<Real>(center_squared - radius * radius),
                                       .limit = static_cast<Real>(2 * radius * std::sqrt(center_squared))};
        inputs_.push_back({.kind = primitive_kind::sphere,
                           .index = static_cast<int>(spheres_.size()),
                           .material = s.material,
                           .frame = frame,
                           .box = sphere_box(converted.center, converted.center, converted.radius)});
        spheres_.push_back(converted);
    }

    void add_moving_sphere(const sphere_desc& s)
    {
        const moving_sphere<Real> converted{.center = to_real<Real>(s.center),
                                            .radius = static_cast<Real>(s.radius),
                                            .displacement = to_real<Real>(s.displacement),
                                            .unused = 0};
        // Where the center is at the end of the shutter time, computed in Real as the GPU computes it. The GPU's center
        // at a time in [0, 1), center + time·displacement, is a number between center and center + displacement,
        // rounded, and rounding never puts a larger number before a smaller one: so in every coordinate it lies
        // between center and this end, rounded the same way. The box around both spheres holds the sphere at every
        // time, as the CPU's does (sphere.ixx).
        const point3<Real> end = converted.center + converted.displacement;
        inputs_.push_back({.kind = primitive_kind::moving_sphere,
                           .index = static_cast<int>(moving_spheres_.size()),
                           .material = s.material,
                           .frame = {.k = 0, .limit = 0},
                           .box = sphere_box(converted.center, end, converted.radius)});
        moving_spheres_.push_back(converted);
    }

    void add_quad(const quad_desc& desc)
    {
        // The box of the four corners, in double. The GPU tests the quad in Real, whose rounded axes accept points up
        // to ~10^-7 of the coordinates beyond the exact edges, so the box is widened by 10^-6 of them on every side:
        // this margin, which grows with the coordinates, is what keeps the slab test from missing the box by rounding.
        // And as the CPU pads a flat box to 0.0001 (aabb::axis_span), an axis still thinner than that after the margin
        // gets 0.0001 more: the flat side of a quad in a plane x = c, 2·10^-6·(1 + the largest coordinate) thick,
        // while the largest coordinate is below 49 (in the Cornell box the margin alone makes it 0.0011).
        const vec3d corners[4] = {desc.q,
                                  {desc.q.x + desc.u.x, desc.q.y + desc.u.y, desc.q.z + desc.u.z},
                                  {desc.q.x + desc.v.x, desc.q.y + desc.v.y, desc.q.z + desc.v.z},
                                  {desc.q.x + desc.u.x + desc.v.x, desc.q.y + desc.u.y + desc.v.y,
                                   desc.q.z + desc.u.z + desc.v.z}};
        build_box box;
        double largest = 0;
        for (const vec3d& c : corners)
        {
            const std::array<double, 3> xyz{c.x, c.y, c.z};
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                box.min[axis] = std::min(box.min[axis], xyz[axis]);
                box.max[axis] = std::max(box.max[axis], xyz[axis]);
                largest = std::max(largest, std::fabs(xyz[axis]));
            }
        }
        const double margin = 1e-6 * (1 + largest);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            box.min[axis] -= margin;
            box.max[axis] += margin;
            constexpr double min_size = 0.0001;
            if (box.max[axis] - box.min[axis] < min_size)
            {
                box.min[axis] -= min_size / 2;
                box.max[axis] += min_size / 2;
            }
        }
        inputs_.push_back({.kind = primitive_kind::quad,
                           .index = static_cast<int>(quads_.size()),
                           .material = desc.material,
                           .frame = {},
                           .box = box});
        quads_.push_back(make_quad<Real>(desc));
    }

    // Puts the input's primitive at the end of its kind's array in the tree, and its material and frame at its global
    // id
    void add_to_leaf(const input& in)
    {
        int id = 0;
        switch (in.kind)
        {
        case primitive_kind::sphere:
            id = static_cast<int>(tree_.spheres.size());
            tree_.spheres.push_back(spheres_[static_cast<std::size_t>(in.index)]);
            break;
        case primitive_kind::moving_sphere:
            id = tree_.moving_base + static_cast<int>(tree_.moving_spheres.size());
            tree_.moving_spheres.push_back(moving_spheres_[static_cast<std::size_t>(in.index)]);
            break;
        case primitive_kind::quad:
            id = tree_.quad_base + static_cast<int>(tree_.quads.size());
            tree_.quads.push_back(quads_[static_cast<std::size_t>(in.index)]);
            tree_.primitive_materials[static_cast<std::size_t>(id)] = in.material;
            return; // a quad has no frame
        }
        tree_.primitive_materials[static_cast<std::size_t>(id)] = in.material;
        tree_.sphere_frames[static_cast<std::size_t>(id)] = in.frame;
    }

    // How many primitives of a kind the tree's array has so far: where a new leaf of that kind starts
    int tree_size(primitive_kind kind) const
    {
        switch (kind)
        {
        case primitive_kind::sphere:
            return static_cast<int>(tree_.spheres.size());
        case primitive_kind::moving_sphere:
            return static_cast<int>(tree_.moving_spheres.size());
        default:
            return static_cast<int>(tree_.quads.size());
        }
    }

    // Adds the subtree over the inputs in `indices` to the nodes, depth first, and returns its root's index. Recursion
    // is fine on the CPU: it goes as deep as the tree, about log2 of the number of primitives.
    int add_subtree(std::span<int> indices, int level)
    {
        if (level > bvh_stack_size)
        {
            throw std::runtime_error("the BVH is too deep for the GPU's traversal stack");
        }
        build_box box;
        for (const int i : indices)
        {
            box.enclose(inputs_[static_cast<std::size_t>(i)].box);
        }
        const auto index = static_cast<int>(tree_.nodes.size());
        tree_.nodes.push_back({.box_min = {round_down<Real>(box.min[0]), round_down<Real>(box.min[1]),
                                           round_down<Real>(box.min[2])},
                               .index = 0,
                               .box_max = {round_up<Real>(box.max[0]), round_up<Real>(box.max[1]),
                                           round_up<Real>(box.max[2])},
                               .count = 0});

        std::size_t middle = indices.size() / 2;
        if (std::ssize(indices) <= max_leaf_size_)
        {
            // Few enough for a leaf, but a leaf holds one kind: those of the first kind present go first
            // (stable_partition keeps their order), and if there are others, an inner node splits the two groups
            const auto kind_of = [&](int i) { return inputs_[static_cast<std::size_t>(i)].kind; };
            const primitive_kind kind = kind_of(std::ranges::min(indices, std::less{}, kind_of));
            const auto others = std::ranges::stable_partition(indices, [&](int i) { return kind_of(i) == kind; });
            if (others.empty())
            {
                // A leaf: its primitives go to the end of their kind's array, so the leaves' ranges follow each other
                tree_.nodes[static_cast<std::size_t>(index)].index = tree_size(kind);
                tree_.nodes[static_cast<std::size_t>(index)].count =
                    static_cast<int>(indices.size()) | static_cast<int>(kind) << leaf_kind_shift;
                for (const int i : indices)
                {
                    add_to_leaf(inputs_[static_cast<std::size_t>(i)]);
                }
                return index;
            }
            middle = indices.size() - others.size();
        }
        else
        {
            // The CPU's heuristic (see build in bvh.ixx): split in half along the longest axis of the common box,
            // sorted by where the primitives' boxes start on it
            const auto axis = static_cast<std::size_t>(box.longest_axis());
            std::ranges::sort(indices, std::less{},
                              [&](int i) { return inputs_[static_cast<std::size_t>(i)].box.min[axis]; });
        }
        add_subtree(indices.first(middle), level + 1); // the first child: right after this node
        // push_back may have moved the nodes to new memory, so the node is looked up again by its index
        tree_.nodes[static_cast<std::size_t>(index)].index = add_subtree(indices.subspan(middle), level + 1);
        return index;
    }
};

// Builds the tree with leaves of at most max_leaf_size primitives. The CPU's leaves are single objects; renderer.cu
// explains the sizes the GPU uses.
template <real_number Real>
bvh<Real> build_bvh(std::span<const sphere_desc> spheres, std::span<const quad_desc> quads, int max_leaf_size)
{
    return bvh_builder<Real>(spheres, quads, max_leaf_size).build();
}

// Device code (it runs on the GPU) from here on.

// The tree as the kernel for all three books sees it: where its arrays are in the GPU's memory
template <real_number Real>
struct bvh_view
{
    const bvh_node<Real>* nodes;
    const sphere<Real>* spheres;               // global ids [0, moving_base)
    int node_count;                            // 0 for an empty scene
    const moving_sphere<Real>* moving_spheres; // global ids [moving_base, quad_base)
    int moving_base;
    const quad<Real>* quads;                   // global ids from quad_base on
    int quad_base;
};

// The tree as the first book's kernel sees it: Step 46's arrays, in a struct that never grows (see first_book_scene in
// scene.cuh for why)
template <real_number Real>
struct first_book_tree
{
    const bvh_node<Real>* nodes;
    const sphere<Real>* spheres;
    int node_count;
};

// The tree of each kind of kernel. std::conditional_t (C++14) picks one of two types by a compile-time condition.
template <real_number Real, kernel_kind Kind>
using kernel_tree = std::conditional_t<Kind == kernel_kind::first_book, first_book_tree<Real>, bvh_view<Real>>;

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

// What closest_hit found: the primitive's global id (-1: none), the distance to it, and whether the ray hits the front
// of the surface there: for a sphere, whether it enters it (see hit_sphere), for a quad, whether it hits the side its
// normal u × v points to (see hit_quad)
template <real_number Real>
struct bvh_hit
{
    int index;
    Real t;
    bool entering;
};

// The closest primitive the ray hits at a distance t in (0.001, ∞): the CPU's world.hit(r, interval{0.001, infinity}),
// except for the primitive the ray starts on, `from` (its global id; -1 for a camera ray), which follows the
// self-intersection rule. For a quad the rule is simple: a ray never hits the quad it leaves. A line meets a plane
// only once, and this ray met it at its start, at t = 0 (or lies in it, which the parallel test rejects anyway). In
// float that is needed: a hit point lies off its quad's plane by a float step of the coordinates (6·10^-5 in the
// Cornell box), so a bounce at a grazing angle would meet its own quad again beyond t = 0.001, from the wrong side,
// and its light would be lost. For a sphere:
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
// spheres keep the CPU's t > 0.001. (trace_first_book_ray goes by geometry too, to tell which side of a sphere a ray
// hit.) A moving sphere is where the ray left it, as the ray keeps the time of the camera ray it comes from.
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
//
// Kind is the kernel's: the first book's kernel tests its leaves exactly as Step 46 did, and the kernel for all three
// books tests every kind of primitive.
template <real_number Real, kernel_kind Kind>
__device__ bvh_hit<Real> closest_hit(const kernel_tree<Real, Kind>& tree, const ray<Real>& r, int from, bool inward)
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
            if constexpr (Kind == kernel_kind::first_book)
            {
                // Spheres that stand still: a leaf's count is its number of spheres, and their global ids are their
                // indices
                for (int i = node.index; i < node.index + node.count; ++i)
                {
                    const bool own = i == from;
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
            }
            else if (leaf_kind(node) == primitive_kind::quad)
            {
                const int end = node.index + leaf_size(node);
                for (int i = node.index; i < end; ++i)
                {
                    const int id = tree.quad_base + i;
                    if (id == from)
                    {
                        continue; // the self-intersection rule: a ray never hits the quad it leaves
                    }
                    // The whole record first: three 16-byte loads in float (six in double), rather than reads scattered
                    // over hit_quad's steps
                    const quad<Real> face = tree.quads[i];
                    Real t;
                    bool front_face;
                    if (hit_quad(face, r, t_min, closest.t, t, front_face))
                    {
                        closest = {.index = id, .t = t, .entering = front_face};
                    }
                }
            }
            else
            {
                // A warp's 32 threads each stand at their own node in an iteration of this loop, so a warp whose
                // threads are at leaves of different kinds runs one kind's code after the other's. The two kinds of
                // sphere share one test, then: only where the sphere comes from differs. (In bouncing most of the small
                // spheres move, mixed with those that don't.)
                const bool moving = leaf_kind(node) == primitive_kind::moving_sphere;
                const int end = node.index + leaf_size(node);
                for (int i = node.index; i < end; ++i)
                {
                    const int id = moving ? tree.moving_base + i : i;
                    const bool own = id == from;
                    if (own && !inward)
                    {
                        continue; // the self-intersection rule: a ray going out of a sphere can't hit it
                    }
                    // A moving sphere where it is at the ray's time
                    const sphere<Real> s = moving ? tree.moving_spheres[i].at(r.time) : tree.spheres[i];
                    Real t;
                    bool entering;
                    if (hit_sphere(s, r, inverse_a, t_min, closest.t, own, t, entering))
                    {
                        closest = {.index = id, .t = t, .entering = entering};
                    }
                }
            }
            node_index = -1; // continue from the stack
        }
        else
        {
            // Look into both children's boxes and descend into the nearer one first: a hit found there spares testing
            // the other one's primitives that lie beyond it. The farther one waits on the stack, if the ray enters it.
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
