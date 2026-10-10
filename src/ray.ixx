// Module ray: a ray is an origin point, a direction and a moment in time.
export module ray;

import vec3;

// The ray's points are given by P(t) = origin + t * direction:
// t = 0 is the ray's origin, t > 0 are points ahead, t < 0 are behind.
// time is when the ray was emitted (0 to 1, while the camera shutter is open). Moving objects
// are in different places at different moments; that is how motion blur appears.
export struct ray
{
    point3 origin;
    vec3 direction;
    double time = 0;

    constexpr point3 at(double t) const { return origin + t * direction; }
};
