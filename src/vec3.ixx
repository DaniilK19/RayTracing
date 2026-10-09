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
