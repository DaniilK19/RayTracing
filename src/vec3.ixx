// Module vec3: a 3D vector for points, directions and colors.
export module vec3;

import std;
import random;

// All functions here are constexpr or inline. This matters: in a named module a function
// defined inside a class does NOT become inline by itself (unlike in headers), and without this
// the compiler couldn't inline vec3 calls into other modules' code, and a ray tracer makes billions of them.
// constexpr functions are implicitly inline; where std::sqrt is needed (not yet constexpr in MSVC), inline is used.
export struct vec3
{
    double x = 0;
    double y = 0;
    double z = 0;

    constexpr vec3 operator-() const { return {-x, -y, -z}; }

    constexpr vec3& operator+=(const vec3& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    constexpr vec3& operator*=(double t)
    {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    constexpr vec3& operator/=(double t) { return *this *= 1 / t; }

    constexpr double length_squared() const { return x * x + y * y + z * z; }
    inline double length() const { return std::sqrt(length_squared()); }

    // Is the vector close to zero in all coordinates?
    inline bool near_zero() const
    {
        constexpr double s = 1e-8;
        return std::fabs(x) < s && std::fabs(y) < s && std::fabs(z) < s;
    }

    // Vector with random components in [0, 1) or in [min, max)
    inline static vec3 random() { return {random_double(), random_double(), random_double()}; }
    inline static vec3 random(double min, double max)
    {
        return {random_double(min, max), random_double(min, max), random_double(min, max)};
    }
};

export using point3 = vec3; // a point in space
export using color = vec3;  // a color: x, y, z are red, green, blue; 0.0 is darkness, 1.0 is full brightness

export constexpr vec3 operator+(const vec3& u, const vec3& v) { return {u.x + v.x, u.y + v.y, u.z + v.z}; }
export constexpr vec3 operator-(const vec3& u, const vec3& v) { return {u.x - v.x, u.y - v.y, u.z - v.z}; }

// Component-wise product: this is how colors mix (e.g. light color × surface color).
export constexpr vec3 operator*(const vec3& u, const vec3& v) { return {u.x * v.x, u.y * v.y, u.z * v.z}; }

export constexpr vec3 operator*(double t, const vec3& v) { return {t * v.x, t * v.y, t * v.z}; }
export constexpr vec3 operator*(const vec3& v, double t) { return t * v; }
export constexpr vec3 operator/(const vec3& v, double t) { return (1 / t) * v; }

export constexpr double dot(const vec3& u, const vec3& v) { return u.x * v.x + u.y * v.y + u.z * v.z; }

export constexpr vec3 cross(const vec3& u, const vec3& v)
{
    return {u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
}

export inline vec3 unit_vector(const vec3& v) { return v / v.length(); }

// Degrees → radians: people think in degrees, while std::sin, std::cos and std::tan expect radians.
export constexpr double degrees_to_radians(double degrees) { return degrees * std::numbers::pi / 180.0; }

// Random point inside the unit disk in the z = 0 plane (also by rejection).
export inline vec3 random_in_unit_disk()
{
    while (true)
    {
        const vec3 p{random_double(-1, 1), random_double(-1, 1), 0};
        if (p.length_squared() < 1)
        {
            return p;
        }
    }
}

// Mirror reflection of v off a surface with unit normal n:
// the component along the normal, dot(v, n) * n, changes sign, so we subtract it twice.
export constexpr vec3 reflect(const vec3& v, const vec3& n) { return v - 2 * dot(v, n) * n; }

// Refraction of the unit vector uv at a surface with unit normal n (Snell's law).
// etai_over_etat is the ratio of refractive indices: of the medium the ray comes from to the one it enters.
// The refracted ray is assembled from two parts: perpendicular to the normal and parallel to it.
export inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat)
{
    const double cos_theta = std::fmin(dot(-uv, n), 1.0);
    const vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    const vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

// Random direction: a unit-length vector, uniformly distributed over all directions.
// Rejection method: take random points in the cube [-1, 1]³ until one falls inside the unit ball,
// and normalize it. Normalizing any cube point instead would favor directions toward the cube's corners.
// Very short vectors are rejected too: dividing by almost zero gives a huge error.
export inline vec3 random_unit_vector()
{
    while (true)
    {
        const vec3 p = vec3::random(-1, 1);
        const double length_squared = p.length_squared();
        if (1e-160 < length_squared && length_squared <= 1)
        {
            return p / std::sqrt(length_squared);
        }
    }
}

// Densities of directions (book 3, chapters 4 and 5), used by the materials' scattering_pdf and by the camera.
// A density of directions is per steradian (unit of solid angle): it integrates to 1 over the whole sphere.

// The density of random_unit_vector(): the same in every direction, 1 / (area of the unit sphere).
export constexpr double uniform_sphere_density = 1 / (4 * std::numbers::pi);

// The Lambertian density around unit_normal (length 1): cos θ / π above the surface, θ being the angle between
// the direction and the normal, and 0 below it. The 1 / π normalizes it: ∫ cos θ dω over the hemisphere is π.
// The direction may have any length.
export inline double cosine_density(const vec3& unit_normal, const vec3& direction)
{
    const double cosine = dot(unit_normal, unit_vector(direction));
    return cosine > 0 ? cosine / std::numbers::pi : 0.0;
}

// Random direction with the density cos θ / π around the z axis, i.e. cosine_density((0, 0, 1), d) (book 3,
// chapter 7). The inversion method: two random numbers and no loop. When a density depends only on θ, the angle
// from the z axis, φ is uniform: φ = 2π·r1. θ gets the density 2π·sin θ·(cos θ / π) = 2·sin θ·cos θ (the band
// between θ and θ + dθ covers 2π·sin θ·dθ steradians); its cumulative distribution is sin²θ, and setting it to r2
// gives sin θ = √r2, cos θ = √(1 - r2). So x² + y² + z² = r2 + (1 - r2): unit length, up to rounding, and z > 0.
export inline vec3 random_cosine_direction()
{
    const double r1 = random_double();
    const double r2 = random_double();
    const double phi = 2 * std::numbers::pi * r1;
    const double x = std::cos(phi) * std::sqrt(r2);
    const double y = std::sin(phi) * std::sqrt(r2);
    const double z = std::sqrt(1 - r2);
    return {x, y, z};
}
