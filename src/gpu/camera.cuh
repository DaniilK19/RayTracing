// The GPU's camera (see camera.ixx): the rays of a pixel's samples, the sky, and the path one sample follows.
#pragma once

#include "bvh.cuh"
#include "material.cuh"
#include "random.cuh"
#include "vec3.cuh"

namespace gpu
{
// The scene as the kernel sees it: where its arrays are in the GPU's memory
template <real_number Real>
struct scene_view
{
    bvh_view<Real> tree;
    const int* sphere_materials; // the material index of each sphere, in the BVH's order
    const material<Real>* materials;
};

// A sample on its way. The CPU's ray_color calls itself for each bounce and multiplies the attenuations on the way back
// out of the recursion. The GPU's version is a loop that goes forward instead: the path keeps the product of the
// attenuations so far, its throughput, and the light it finds at its end is multiplied by it.
template <real_number Real>
struct path
{
    ray<Real> r; // the ray to trace next
    color<Real> throughput;
    int rays_left;   // how many more rays the path may trace, r included (the CPU's depth)
    int from_sphere; // the sphere r starts on, -1 for a camera ray (see closest_hit)
    bool inward;     // r goes into that sphere
};

// camera_desc (see renderer.h) in Real, with the camera's functions that the kernel needs
template <real_number Real>
struct camera
{
    int image_width;
    int image_height;
    int sqrt_spp;
    int max_depth;
    Real recip_sqrt_spp; // 1 / sqrt_spp: the side of a cell of the pixel's grid, in pixel widths
    point3<Real> center;
    vec3<Real> center_to_pixel00; // pixel00_loc - center, computed in double (see get_ray)
    vec3<Real> pixel_delta_u;
    vec3<Real> pixel_delta_v;
    bool defocus;
    vec3<Real> defocus_disk_u;
    vec3<Real> defocus_disk_v;
    color<Real> sky_bottom;
    color<Real> sky_top;

    // The path of pass `pass` of pixel (i, j). Pass s takes cell (s % sqrt_spp, s / sqrt_spp) of the pixel's grid, so
    // the pixel's sqrt_spp² passes visit every cell once, as the CPU's loop over the strata does.
    __device__ path<Real> start_path(int i, int j, int pass, pcg32& rng) const
    {
        return {.r = get_ray(i, j, pass % sqrt_spp, pass / sqrt_spp, rng),
                .throughput = {Real(1), Real(1), Real(1)},
                .rays_left = max_depth,
                .from_sphere = -1,
                .inward = false};
    }

    // camera::get_ray: a ray through a random point of cell (s_i, s_j) of pixel (i, j)'s grid, from the lens.
    //
    // The CPU computes two positions, the point of the pixel the ray aims at (pixel_sample) and the lens point it
    // starts from, and the direction as their difference. In float that difference can lose most of its digits. A float
    // has 24 significant bits, so each coordinate of a position is rounded to about 10^-7 of its size: the error
    // depends on how far the point is from the scene's origin, wherever that happens to be, and not on the ray's
    // length. A difference of two positions keeps both rounding errors, however short it is. With the camera at
    // (-2, 2, 1) and a focus distance of 10^-5, the pixels of a 160×90 render are 4·10^-8 apart on the focus plane,
    // while floats just below 2 are 1.2·10^-7 apart: about four pixels shared each direction, up to four pixels off,
    // and the image came out blocky and shifted. (Doubles, the CPU's, round to 10^-16.) So get_ray builds a camera ray
    // without subtracting one position from another. The host subtracts the camera center from pixel00_loc once, in
    // double (see to_camera in renderer.cu), and the direction is a sum of vectors relative to the center, each rounded
    // to about 10^-7 of its own length. In exact arithmetic this is the CPU's ray: the origin plus the direction is
    // (center + lens) + (pixel_sample - center - lens) = pixel_sample. The origin is still a position, rounded like the
    // spheres' centers, which moves a ray by a hair without turning it (closest_hit in bvh.cuh says when such hairs
    // matter). The direction is a float too, good to about 10^-7 radians: far finer than a pixel unless the field of
    // view is a small fraction of a degree.
    __device__ ray<Real> get_ray(int i, int j, int s_i, int s_j, pcg32& rng) const
    {
        // One random number per statement fixes their order, as on the CPU (x, then y)
        const Real dx = (Real(s_i) + random_real<Real>(rng)) * recip_sqrt_spp - Real(0.5);
        const Real dy = (Real(s_j) + random_real<Real>(rng)) * recip_sqrt_spp - Real(0.5);
        // The CPU's pixel_sample - center
        const vec3<Real> center_to_sample =
            center_to_pixel00 + (Real(i) + dx) * pixel_delta_u + (Real(j) + dy) * pixel_delta_v;
        if (!defocus)
        {
            return {center, center_to_sample};
        }
        // The ray starts at a random point of the lens, lens away from the center: seen from there, the pixel's point
        // is at center_to_sample - lens
        const vec3<Real> p = random_in_unit_disk<Real>(rng);
        const vec3<Real> lens = p.x * defocus_disk_u + p.y * defocus_disk_v;
        return {center + lens, center_to_sample - lens};
    }

    // sky::in_direction: the light of a ray that hit nothing
    __device__ color<Real> sky(const vec3<Real>& direction) const
    {
        const Real a = Real(0.5) * (unit_vector(direction).y + Real(1));
        return (Real(1) - a) * sky_bottom + a * sky_top;
    }
};

// Traces the path's next ray: one level of the CPU's ray_color. Returns true when the sample is complete, with its
// color in `sample`; otherwise the path now holds the ray to trace next.
template <real_number Real>
__device__ bool trace_ray(path<Real>& p, const scene_view<Real>& scene, const camera<Real>& cam, pcg32& rng,
                          color<Real>& sample)
{
    const bvh_hit<Real> hit = closest_hit(scene.tree, p.r, p.from_sphere, p.inward);
    if (hit.index < 0)
    {
        sample = p.throughput * cam.sky(p.r.direction);
        return true;
    }

    const sphere<Real> s = scene.tree.spheres[hit.index];
    hit_record<Real> rec;
    rec.p = p.r.at(hit.t);
    const vec3<Real> outward = (Real(1) / s.radius) * (rec.p - s.center);
    // The side of the surface the ray came from. The CPU takes the sign of dot(direction, outward), which in exact
    // arithmetic says the same as the root: dot(direction, p - center) = |direction|²·(t - h/a) (see hit_sphere) is
    // negative at the near root, where the ray enters, and positive at the far one. The root is the safer test in
    // float: for a ray that grazes a sphere, both roots lie close to the ray's closest approach to the center, and the
    // rounding of the computed hit point can flip the dot product's sign. Before hit_sphere computed its discriminant
    // with fmas, that happened to about 50 camera rays at the ground's horizon in 130 million samples of the weekend
    // scene. Each one counted as hitting the ground from inside: its bounce went into the ground, the self-intersection
    // rule (see closest_hit) kept it there, and the sample came out black after max_depth rays.
    rec.front_face = hit.entering;
    rec.normal = rec.front_face ? outward : -outward;

    const material<Real>& mat = scene.materials[scene.sphere_materials[hit.index]];
    vec3<Real> direction;
    if (!scatter(mat, p.r.direction, rec, rng, direction, p.throughput))
    {
        sample = {}; // absorbed: no light
        return true;
    }
    if (--p.rays_left == 0)
    {
        sample = {}; // the path may trace no more rays (the CPU's depth reached 0): no light
        return true;
    }
    p.r = {rec.p, direction};
    p.from_sphere = hit.index;
    p.inward = dot(direction, outward) < Real(0);
    return false;
}
}
