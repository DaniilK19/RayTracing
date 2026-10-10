// Module texture: textures, i.e. a color that changes from point to point of a surface.
export module texture;

import std;
import vec3;

// A texture answers the question "what color is the surface at this point". The point can be given
// by texture coordinates (u, v) on the surface (that is how an image is wrapped around an object) or directly
// by a point p in space: that is how solid textures like the checker below work.
export class texture
{
public:
    virtual ~texture() = default;

    virtual color value(double u, double v, const point3& p) const = 0;
};

// One color for the whole surface: this makes an ordinary color a special case of a texture.
export class solid_color : public texture
{
public:
    explicit solid_color(const color& albedo) : albedo_(albedo) {}

    color value(double, double, const point3&) const override { return albedo_; }

private:
    color albedo_;
};

// A solid checkerboard: space is divided into cubes with side scale, and neighboring cubes
// are painted with different textures. An object's surface passing through the cubes becomes "checkered".
export class checker_texture : public texture
{
public:
    checker_texture(double scale, std::shared_ptr<const texture> even, std::shared_ptr<const texture> odd)
        : inv_scale_(1.0 / scale), even_(std::move(even)), odd_(std::move(odd))
    {
    }

    checker_texture(double scale, const color& even, const color& odd)
        : checker_texture(scale, std::make_shared<solid_color>(even), std::make_shared<solid_color>(odd))
    {
    }

    color value(double u, double v, const point3& p) const override
    {
        // Cube index along each axis. It has to be floor, not static_cast<int>(...) right away: the cast
        // drops the fractional part and would give 0 instead of -1 for -0.5, so the square at zero would be twice as wide.
        const int x = static_cast<int>(std::floor(inv_scale_ * p.x));
        const int y = static_cast<int>(std::floor(inv_scale_ * p.y));
        const int z = static_cast<int>(std::floor(inv_scale_ * p.z));

        // A step to a neighboring cube along any axis flips the parity of the sum, and so the color
        const bool is_even = (x + y + z) % 2 == 0;
        return is_even ? even_->value(u, v, p) : odd_->value(u, v, p);
    }

private:
    double inv_scale_; // 1 / scale: multiplying is faster than dividing
    std::shared_ptr<const texture> even_;
    std::shared_ptr<const texture> odd_;
};
