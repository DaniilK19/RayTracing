// Module pdf: densities of directions, and pdfs, which also draw random directions with them.
export module pdf;

import std;
import onb;
import random;
import vec3;

// Densities of directions (book 3, chapters 4 and 5), used by the pdfs below, by the materials' scattering_pdf, by a
// sphere lamp seen from inside and by MonteCarlo. A density of directions is per steradian (unit of solid angle): it
// integrates to 1 over the whole sphere.

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

// A pdf (book 3, chapter 10) is a density of directions together with a way to draw directions with it:
//   value(direction) - the density, per steradian, of that direction;
//   generate()       - a random direction with exactly that density, of length 1 up to rounding (the camera
//                      looks for hits from t = 0.001, measured in lengths of the direction).
// The book makes pdf an abstract base class with virtual functions and allocates new pdfs on the heap (make_shared)
// at every bounce. Ours are a small, fixed set of short-lived values: each lives on the stack for one bounce, and
// wherever one is used, the compiler knows its type, or at least the few types it can have (a material's pdf is a
// cosine_pdf or a sphere_pdf, chosen at run time: see material_pdf). So instead of a base class, a concept (C++20)
// states what a pdf must offer: a named set of requirements on a type. A template constrained by it accepts any type
// that meets them and is compiled for each such type separately, with direct calls that the compiler can inline. The
// pdfs below meet the concept, and so does hittable_pdf (directions toward the lamps), which is in the hittable module:
// it needs the class hittable, and that module imports this one. Virtual functions stay where they fit better: hittable
// and material, open families of objects that live as long as the scene and that anyone can extend with a new kind.
export template <typename T>
concept direction_pdf = requires(const T& pdf, const vec3& direction) {
    { pdf.value(direction) } -> std::convertible_to<double>;
    { pdf.generate() } -> std::convertible_to<vec3>;
};

// The same density in every direction: smoke particles scatter like this, and the uniform strategy samples with it.
export class sphere_pdf
{
public:
    inline double value(const vec3& /*direction*/) const { return uniform_sphere_density; }
    inline vec3 generate() const { return random_unit_vector(); }
};

// The Lambertian density cos θ / π around a normal: how a matte surface scatters light.
export class cosine_pdf
{
public:
    // unit_normal must have length 1 (hit normals have): the basis keeps it as its w, bit for bit (see onb)
    inline explicit cosine_pdf(const vec3& unit_normal) : uvw_(unit_normal) {}

    // The same function on the same numbers as lambertian's scattering_pdf: for a matte surface sampled with
    // its own pdf, the camera's ratio pScatter / p is exactly 1.0.
    inline double value(const vec3& direction) const { return cosine_density(uvw_.w(), direction); }

    // random_cosine_direction() draws around the z axis by the inversion method (book 3, chapter 7): two random
    // numbers and no rejection loop, unlike the first two books' normal + random_unit_vector(), which has the same
    // density. The basis turns the z axis into the normal (chapter 8). The result isn't normalized again: its length
    // is already 1 up to rounding, and cosine_density doesn't need it.
    inline vec3 generate() const { return uvw_.transform(random_cosine_direction()); }

private:
    onb uvw_;
};

// The pdf a material scatters with: a cosine_pdf (matte surfaces) or a sphere_pdf (smoke). Which one is known only
// at run time, from the material a ray hits, so it is kept in a std::variant (C++17): a value that holds one of the
// listed types, stored in place, without a heap allocation. std::visit calls the lambda with the value the variant
// holds, so each call reaches the value() or generate() of its real type. Wrapped in a class, the variant is a pdf
// itself: it meets direction_pdf.
export class material_pdf
{
public:
    // Not explicit: a material can return cosine_pdf{normal} or sphere_pdf{} where a material_pdf is expected
    inline material_pdf(const cosine_pdf& pdf) : pdf_(pdf) {}
    inline material_pdf(const sphere_pdf& pdf) : pdf_(pdf) {}

    inline double value(const vec3& direction) const
    {
        return std::visit([&direction](const auto& pdf) { return pdf.value(direction); }, pdf_);
    }

    inline vec3 generate() const
    {
        return std::visit([](const auto& pdf) { return pdf.generate(); }, pdf_);
    }

private:
    std::variant<cosine_pdf, sphere_pdf> pdf_;
};

// Half of one pdf, half of another: a mixture (book 3, section 10.3). An average of two densities is a density too
// (it integrates to ½ + ½ = 1). To draw from it, one random number picks which of the two pdfs draws the direction.
// Its density is NOT that of the pdf that happened to draw it: the other one could have drawn the same direction
// (a direction toward the lamp is also one the material scatters into), so value() averages both densities.
// A class template, where the book has two shared_ptr<pdf>: A and B are the types of the two pdfs, any that meet
// direction_pdf. With mixture_pdf{a, b} the compiler deduces A and B from the constructor's arguments (class template
// argument deduction, CTAD, C++17). The pdfs are kept by value: copies are cheap, and a copy can't dangle, while a
// reference member would silently refer to a dead temporary after `const mixture_pdf m{hittable_pdf{lights, p}, x};`.
// The members need no `inline` (see vec3.ixx): a template is compiled where it is used, for the types it is used with.
export template <direction_pdf A, direction_pdf B>
class mixture_pdf
{
public:
    mixture_pdf(const A& a, const B& b) : a_(a), b_(b) {}

    double value(const vec3& direction) const { return 0.5 * a_.value(direction) + 0.5 * b_.value(direction); }

    vec3 generate() const { return random_double() < 0.5 ? a_.generate() : b_.generate(); }

private:
    A a_;
    B b_;
};
