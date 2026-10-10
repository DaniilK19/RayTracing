// Module material: surface materials, i.e. how they reflect and refract light.
export module material;

import std;
import hittable;
import onb;
import random;
import ray;
import texture;
import vec3;

// Matte (diffuse) surface: scatters light in all directions, more often close to the normal.
export class lambertian : public material
{
public:
    explicit lambertian(const color& albedo) : lambertian(std::make_shared<solid_color>(albedo)) {}
    explicit lambertian(std::shared_ptr<const texture> tex) : tex_(std::move(tex)) {}

    // A direction with the Lambertian density cos θ / π (book 3, chapter 8): random_cosine_direction() draws one
    // around the z axis, and the orthonormal basis with w = normal turns the z axis into the normal. The first two
    // books' normal + random_unit_vector() has the same density (a uniformly random point of the unit sphere that
    // touches the surface at p), but it needs the rejection loop (1.9 tries of three random numbers on average),
    // and the sum can be close to zero, which needed a guard. The inversion method takes two random numbers, and
    // its direction has length 1, up to rounding.
    std::optional<scatter_record> scatter(const ray& r_in, const hit_record& rec) const override
    {
        const onb uvw{rec.normal};

        // Unlike the book, we don't normalize the direction: it is already unit length, and cosine_density doesn't
        // need it. The scattered ray exists at the same moment in time as the incoming one. Its density is
        // scattering_pdf itself: the same function on the same values, so the camera's pScatter is the same number
        // to the last bit.
        const ray scattered{rec.p, uvw.transform(random_cosine_direction()), r_in.time};
        return diffuse_scatter{tex_->value(rec.u, rec.v, rec.p), scattered, scattering_pdf(r_in, rec, scattered)};
    }

    double scattering_pdf(const ray& /*r_in*/, const hit_record& rec, const ray& scattered) const override
    {
        return cosine_density(rec.normal, scattered.direction);
    }

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
// Brightness can exceed 1: a lamp shines brighter than a white wall reflects. A small lamp
// must be very bright to light a room: only a small fraction of the scattered rays reach it.
export class diffuse_light : public material
{
public:
    explicit diffuse_light(const color& emit) : diffuse_light(std::make_shared<solid_color>(emit)) {}
    explicit diffuse_light(std::shared_ptr<const texture> tex) : tex_(std::move(tex)) {}

    std::optional<scatter_record> scatter(const ray&, const hit_record&) const override { return std::nullopt; }

    color emitted(double u, double v, const point3& p) const override { return tex_->value(u, v, p); }

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

    std::optional<scatter_record> scatter(const ray& r_in, const hit_record& rec) const override
    {
        const ray scattered{rec.p, random_unit_vector(), r_in.time};
        return diffuse_scatter{tex_->value(rec.u, rec.v, rec.p), scattered, uniform_sphere_density};
    }

    // In smoke the phase function (the volume's pScatter) is the same in every direction
    double scattering_pdf(const ray& /*r_in*/, const hit_record& /*rec*/, const ray& /*scattered*/) const override
    {
        return uniform_sphere_density;
    }

private:
    std::shared_ptr<const texture> tex_;
};
