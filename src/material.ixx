// Module material: surface materials, i.e. how they reflect and refract light.
export module material;

import std;
import hittable;
import pdf;
import random;
import ray;
import texture;
import vec3;

// Matte (diffuse) surface: scatters light into all directions above the surface, more often close to the normal.
export class lambertian : public material
{
public:
    explicit lambertian(const color& albedo) : lambertian(std::make_shared<solid_color>(albedo)) {}
    explicit lambertian(std::shared_ptr<const texture> tex) : tex_(std::move(tex)) {}

    // The Lambertian density cos θ / π around the normal, as a pdf. The material no longer draws the direction
    // (book 3, section 12.1): the camera draws it, from this pdf or, depending on its strategy, from another one.
    std::optional<scatter_record> scatter(const ray& /*r_in*/, const hit_record& rec) const override
    {
        return diffuse_scatter{tex_->value(rec.u, rec.v, rec.p), cosine_pdf{rec.normal}};
    }

    double scattering_pdf(const ray& /*r_in*/, const hit_record& rec, const ray& scattered) const override
    {
        return cosine_density(rec.normal, scattered.direction);
    }

    // For code that copies a scene into its own format (the GPU program); the same for the other materials below
    inline const texture& albedo_texture() const { return *tex_; }

private:
    // Fraction of light reflected per channel: the surface's "color". A texture lets it vary
    // from point to point; a single color for the whole surface is solid_color.
    std::shared_ptr<const texture> tex_;
};

// Metal: mirror reflection blurred by fuzz (0 is a perfect mirror, 1 is a very matte metal).
export class metal : public material
{
public:
    metal(const color& albedo, double fuzz) : albedo_(albedo), fuzz_(std::min(fuzz, 1.0)) {}

    std::optional<scatter_record> scatter(const ray& r_in, const hit_record& rec) const override
    {
        // Blur: move the tip of the reflected unit vector to a random point on a sphere of radius fuzz
        const vec3 reflected = unit_vector(reflect(r_in.direction, rec.normal)) + fuzz_ * random_unit_vector();

        // Blurring can send the ray below the surface; such a ray counts as absorbed
        if (dot(reflected, rec.normal) <= 0)
        {
            return std::nullopt;
        }
        return specular_scatter{albedo_, ray{rec.p, reflected, r_in.time}};
    }

    inline const color& albedo() const { return albedo_; }
    inline double fuzz() const { return fuzz_; }

private:
    color albedo_;
    double fuzz_;
};

// Transparent material (glass, water): the ray is refracted or reflected, no light is absorbed.
export class dielectric : public material
{
public:
    // refraction_index is the refractive index relative to the surrounding medium:
    // glass in air ≈ 1.5, and an air bubble inside glass is 1 / 1.5.
    explicit dielectric(double refraction_index) : refraction_index_(refraction_index) {}

    std::optional<scatter_record> scatter(const ray& r_in, const hit_record& rec) const override
    {
        // Entering the glass, the ray goes from medium 1 to medium n, ratio 1/n; leaving, the other way round: n
        const double ri = rec.front_face ? (1.0 / refraction_index_) : refraction_index_;

        const vec3 unit_direction = unit_vector(r_in.direction);
        const double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
        const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

        // Total internal reflection: by Snell's law the sine of the refraction angle equals ri·sin θ,
        // and a sine can't exceed 1, so the ray can't refract and can only reflect.
        const bool cannot_refract = ri * sin_theta > 1.0;

        // Even when refraction is possible, some light is reflected (glare on glass), the more so
        // the more grazing the angle. Choose randomly with the reflection probability from Schlick's formula.
        const vec3 direction = cannot_refract || reflectance(cos_theta, ri) > random_double()
                                   ? reflect(unit_direction, rec.normal)
                                   : refract(unit_direction, rec.normal, ri);

        return specular_scatter{color{1.0, 1.0, 1.0}, ray{rec.p, direction, r_in.time}};
    }

    inline double refraction_index() const { return refraction_index_; }

private:
    double refraction_index_;

    // Schlick's approximation: what fraction of light is reflected at a given angle of incidence
    static double reflectance(double cosine, double refraction_index)
    {
        double r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * std::pow(1 - cosine, 5);
    }
};

// Glowing surface: a light source. It emits light but reflects nothing.
// Brightness can exceed 1: a lamp shines brighter than a white wall reflects. A small lamp must be very bright to
// light a room: seen from a point in the room, it covers only a small part of the directions light can come from.
// (Aiming more rays at it, as the lights and mixture strategies do, changes the noise, not the brightness: each
// such ray gets the small weight pScatter / p.)
// It glows on the front side only, the side its outward normal points to (for a quad, u × v): a ceiling lamp
// shines down into the room. Glowing on both sides, as in the second book, the Cornell box's lamp also lit the
// ceiling 1 unit above it very brightly, and the rare paths that found that gap spread its light as fireflies
// (book 3, section 9.3).
export class diffuse_light : public material
{
public:
    explicit diffuse_light(const color& emit) : diffuse_light(std::make_shared<solid_color>(emit)) {}
    explicit diffuse_light(std::shared_ptr<const texture> tex) : tex_(std::move(tex)) {}

    std::optional<scatter_record> scatter(const ray&, const hit_record&) const override { return std::nullopt; }

    color emitted(const hit_record& rec) const override
    {
        return rec.front_face ? tex_->value(rec.u, rec.v, rec.p) : color{0, 0, 0};
    }

    inline const texture& emission_texture() const { return *tex_; }

private:
    std::shared_ptr<const texture> tex_;
};

// Material of smoke or fog particles: scatters light equally in all directions (isotropically).
// A cloud of particles has no surface, so the normal doesn't matter: the new direction is fully random.
export class isotropic : public material
{
public:
    explicit isotropic(const color& albedo) : isotropic(std::make_shared<solid_color>(albedo)) {}
    explicit isotropic(std::shared_ptr<const texture> tex) : tex_(std::move(tex)) {}

    // The same density in every direction, as a pdf: the camera draws the direction (book 3, section 12.1)
    std::optional<scatter_record> scatter(const ray& /*r_in*/, const hit_record& rec) const override
    {
        return diffuse_scatter{tex_->value(rec.u, rec.v, rec.p), sphere_pdf{}};
    }

    // In smoke the phase function (the volume's pScatter) is the same in every direction
    double scattering_pdf(const ray& /*r_in*/, const hit_record& /*rec*/, const ray& /*scattered*/) const override
    {
        return uniform_sphere_density;
    }

    // Beyond the book: under the mixture strategy smoke and fog keep sampling with their own density, unmixed.
    // A particle deep in a volume gets most of its light from other particles, not straight from a lamp (inside the
    // final scene's glass ball, rays aimed at the lamp are even bent away by the glass). Mixing the lamps in would
    // still weigh the directions the particle's own half draws by up to 2, and in a volume that scatters light
    // again and again, losing little of it, these weights multiply along a path: rare paths get huge weights and
    // show up as bright specks. In the final scene's blue fog that more than doubled the error; with its own
    // density every weight is exactly 1. The price: a thin haze, where light mostly scatters once, straight from
    // the lamp, gets no help from aiming at it, so the final scene's hazy back wall is a little noisier (but evenly
    // so). Telling thin and dense volumes apart, or sending separate rays to the lamps, is left for later.
    bool mixes_in_lights() const override { return false; }

private:
    std::shared_ptr<const texture> tex_;
};
