// Module ray: a ray is an origin point and a direction.
export module ray;

import vec3;

// The ray's points are given by P(t) = origin + t * direction:
// t = 0 is the ray's origin, t > 0 are points ahead, t < 0 are behind.
export struct ray
{
    point3 origin;
    vec3 direction;

    constexpr point3 at(double t) const { return origin + t * direction; }
};
