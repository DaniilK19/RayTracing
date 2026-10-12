// The GPU's camera (see camera.ixx): the rays of a pixel's samples, the sky, and the path one sample follows.
#pragma once

#include "bvh.cuh"
#include "material.cuh"
#include "quad.cuh"
#include "random.cuh"
#include "scene.cuh"
#include "sphere.cuh"
#include "texture.cuh"
#include "vec3.cuh"

namespace gpu
{
// A sample on its way. The CPU's ray_color calls itself for each bounce and multiplies the attenuations on the way back
// out of the recursion. The GPU's version is a loop that goes forward instead: the path keeps the product of the
// attenuations so far, its throughput, and the light it finds at its end is multiplied by it.
template <real_number Real>
struct path
{
    ray<Real> r; // the ray to trace next
    color<Real> throughput;
    int rays_left; // how many more rays the path may trace, r included (the CPU's depth)
    int from;      // the global id of the primitive r starts on, -1 for a camera ray (see closest_hit)
    bool inward;   // r goes into that sphere
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
                .from = -1,
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

// Traces the path's next ray in the first book's kernel: one level of the CPU's ray_color for the first book's scenes.
// Returns true when the sample is complete, with its color in `sample`; otherwise the path now holds the ray to trace
// next.
template <real_number Real>
__device__ bool trace_first_book_ray(path<Real>& p, const first_book_scene<Real>& scene, const camera<Real>& cam,
                                     pcg32& rng, color<Real>& sample)
{
    const bvh_hit<Real> hit = closest_hit<Real, kernel_kind::first_book>(scene.tree, p.r, p.from, p.inward);
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

    const material<Real>& mat = scene.materials[scene.primitive_materials[hit.index]];
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
    p.from = hit.index;
    p.inward = dot(direction, outward) < Real(0);
    return false;
}

// What the kernel for all three books knows about a hit: Step 46's hit_record (the point, the normal facing the ray,
// the side), the material, and where its texture is read
template <real_number Real>
struct surface
{
    hit_record<Real> rec;
    vec3<Real> outward;         // the outward unit normal, before it is turned to face the ray (a quad's: u × v)
    point3<Real> texture_point; // where the texture is read: rec.p, or for a checker on a sphere rec.p moved exactly
                                // onto the sphere (onto_sphere). The next ray starts at rec.p either way.
    Real u;                     // the texture coordinates, computed only when the material's texture reads them
    Real v;                     // (needs_uv), otherwise 0
    int material;               // an index into the materials
    int texture_index;          // the material's texture_index (material_texture): -1 for one solid color
};

// The hit as the materials and textures need it (the CPU's hit_record, which sphere::hit and quad::hit fill)
template <real_number Real>
__device__ surface<Real> surface_at(const scene_view<Real>& scene, const ray<Real>& r, const bvh_hit<Real>& hit)
{
    surface<Real> s{};
    s.rec.p = r.at(hit.t);
    s.rec.front_face = hit.entering; // from the root for a sphere, from the sign of normal·direction for a quad
    s.texture_point = s.rec.p;
    s.material = scene.primitive_materials[hit.index];
    const material_texture shading = scene.material_textures[s.material];
    s.texture_index = shading.texture_index;
    const bvh_view<Real>& tree = scene.tree;
    if (hit.index >= tree.quad_base)
    {
        const quad<Real> face = tree.quads[hit.index - tree.quad_base];
        s.outward = face.normal;
        s.rec.normal = s.rec.front_face ? s.outward : -s.outward;
        if (shading.needs_uv)
        {
            quad_uv(face, r, hit.t, s.u, s.v);
        }
        if (shading.needs_exact_point)
        {
            // A checker's colors jump across the faces of its cubes, and a quad can lie in one (a floor at y = 0):
            // there the point's rounding error alone decides the side it falls on, and so its color. The CPU rounds
            // t·direction, then the sum: for rays from above, the floor's points come out at y = 0 or just above, the
            // color of the cubes above. at() lets nvcc fuse the two into one rounding, and then about half of the
            // points fell below: a floor of mixed colors. Computed as the CPU computes them, the points fall on the
            // CPU's side as often as the CPU's do in float, and in double exactly where they do for a quad placed in
            // the world directly (an axis-aligned quad's t is then the CPU's own number). For a quad baked out of
            // instances the CPU finds the point in the instance's coordinates and moves it back: there the two agree
            // only to rounding, also in double.
            s.texture_point = r.at_unfused(hit.t);
        }
        return s;
    }
    // A sphere that stands still, or a moving one where it is at the ray's time. Then Step 46's lines (see
    // trace_first_book_ray above).
    const sphere<Real> ball = hit.index < tree.moving_base
                                  ? tree.spheres[hit.index]
                                  : tree.moving_spheres[hit.index - tree.moving_base].at(r.time);
    s.outward = (Real(1) / ball.radius) * (s.rec.p - ball.center);
    s.rec.normal = s.rec.front_face ? s.outward : -s.outward;
    if (shading.needs_uv || shading.needs_exact_point)
    {
        const sphere_frame<Real> frame = scene.sphere_frames[hit.index];
        if (shading.needs_uv)
        {
            sphere_uv(s.outward, frame, s.u, s.v);
        }
        if (shading.needs_exact_point)
        {
            s.texture_point = onto_sphere(s.rec.p, s.outward, ball, frame);
        }
    }
    return s;
}

// The material's color at the hit: its texture there, or its one solid color without a lookup
template <real_number Real>
__device__ color<Real> surface_color(const scene_view<Real>& scene, const material<Real>& mat, const surface<Real>& s)
{
    return s.texture_index < 0 ? mat.albedo
                               : texture_value(scene.textures, s.texture_index, s.u, s.v, s.texture_point);
}

// Traces the path's next ray in the kernel for all three books: one level of the CPU's ray_color, with the features of
// the books so far. Returns true when the sample is complete, with its color in `sample`; otherwise the path now holds
// the ray to trace next.
template <real_number Real>
__device__ bool trace_ray(path<Real>& p, const scene_view<Real>& scene, const camera<Real>& cam, pcg32& rng,
                          color<Real>& sample)
{
    const bvh_hit<Real> hit = closest_hit<Real, kernel_kind::all_books>(scene.tree, p.r, p.from, p.inward);
    if (hit.index < 0)
    {
        sample = p.throughput * cam.sky(p.r.direction);
        return true;
    }
    const surface<Real> s = surface_at(scene, p.r, hit);
    const material<Real> mat = scene.materials[s.material];
    // The material's color here, for a kind that has one: a matte surface, or a lamp seen from its front. Computed at
    // one place, so that the texture code is in the kernel once; textures draw no random numbers, so computing it
    // before the bounce changes nothing else.
    const bool colored =
        mat.kind == material_kind::lambertian || (mat.kind == material_kind::diffuse_light && s.rec.front_face);
    const color<Real> c = colored ? surface_color(scene, mat, s) : color<Real>{};

    vec3<Real> direction;
    switch (mat.kind)
    {
    case material_kind::diffuse_light:
        // A lamp glows on its front side only (c is 0 on its back) and reflects nothing: the path ends here, whichever
        // side it hit. Its light counts on the path's last ray too, as on the CPU: there the depth test at the top of
        // ray_color decides only whether a ray is traced, so the last ray, once traced, still returns what its hit
        // emits. The test that ends a path is the next ray's: here the --p.rays_left test below, which a lamp never
        // reaches.
        //
        // The CPU's ray_color adds up emitted + attenuation·(the light of the next ray) down its recursion. Here a
        // path keeps no running sum of light, only its throughput: the only material that emits, diffuse_light,
        // ends the path, and every other material emits nothing. A material that both emitted and scattered would
        // need such a sum (renderer.cu's validate accepts only the materials that this holds for).
        sample = p.throughput * c;
        return true;
    case material_kind::lambertian:
        // Step 46's matte bounce (see scatter in material.cuh), attenuated by the texture's color
        direction = around(s.rec.normal, random_cosine_direction<Real>(rng));
        if (!(dot(s.rec.normal, direction) > Real(0)))
        {
            sample = {}; // a direction the CPU doesn't trace: no light
            return true;
        }
        p.throughput *= c;
        break;
    case material_kind::metal:
    case material_kind::dielectric:
        if (!scatter(mat, p.r.direction, s.rec, rng, direction, p.throughput))
        {
            sample = {}; // absorbed: no light
            return true;
        }
        break;
    }
    if (--p.rays_left == 0)
    {
        sample = {}; // the path may trace no more rays (the CPU's depth reached 0): no light
        return true;
    }
    // A bounce keeps the ray's time (material.ixx for metal and glass, ray_color in camera.ixx for a diffuse bounce)
    p.r = {s.rec.p, direction, p.r.time};
    p.from = hit.index;
    p.inward = dot(direction, s.outward) < Real(0); // read only when the ray leaves a sphere (see closest_hit)
    return false;
}

// The path of pass `pass` of pixel (i, j), as the kernel's kind starts it: the camera ray of camera::start_path and, in
// a scene where something moves, the moment it is emitted, drawn after the camera ray's numbers as on the CPU
// (camera::get_ray in camera.ixx). The CPU draws the time for every ray; the GPU only when a sphere moves, as a scene
// that stands still never reads it: drawing it anyway would shift every random number after it, and the first book's
// scenes would get other noise than in Step 46 (the same picture, no longer the same bytes).
template <real_number Real, kernel_kind Kind>
__device__ path<Real> start_sample(const camera<Real>& cam, const kernel_scene<Real, Kind>& scene, int i, int j,
                                   int pass, pcg32& rng)
{
    path<Real> p = cam.start_path(i, j, pass, rng);
    if constexpr (Kind == kernel_kind::all_books)
    {
        if (scene.motion)
        {
            p.r.time = random_real<Real>(rng);
        }
    }
    return p;
}

// One ray of the path, as the kernel's kind traces it
template <real_number Real, kernel_kind Kind>
__device__ bool trace(path<Real>& p, const kernel_scene<Real, Kind>& scene, const camera<Real>& cam, pcg32& rng,
                      color<Real>& sample)
{
    if constexpr (Kind == kernel_kind::first_book)
    {
        return trace_first_book_ray(p, scene, cam, rng, sample);
    }
    else
    {
        return trace_ray(p, scene, cam, rng, sample);
    }
}
}
