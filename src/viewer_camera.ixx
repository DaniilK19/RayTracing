// Module viewer_camera: the window's camera moves, as pure functions of the camera's settings. The mouse and the keys
// call them (main_viewer.cpp), and so do the camera scripts' verbs (viewer_script), so a script moves the camera
// exactly as the hand does. Every move works on lookfrom, lookat, vfov and focus_dist in double, and a move by nothing
// returns the settings unchanged, bit for bit: a view the user hasn't touched is the scene's own.
export module viewer_camera;

import std;
import camera;
import vec3;

// What a notch of the wheel does: the distance to lookat times wheel_dolly, or with Ctrl the field of view times
// wheel_zoom (the - and = keys too). A script's `notches` takes the same powers of the same numbers.
export constexpr double wheel_dolly = 0.85;
export constexpr double wheel_zoom = 0.9;

namespace
{
// Degrees to radians, the angle first brought into (-360, 360): exact for any finite angle (as fmod is), so a script's
// yaw of 1e308 degrees turns the camera instead of overflowing
double radians(double degrees)
{
    return degrees_to_radians(std::fmod(degrees, 360.0));
}

// v turned by `angle` radians about the unit axis k (Rodrigues' rotation formula)
vec3 rotate(const vec3& v, const vec3& k, double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return c * v + s * cross(k, v) + (1 - c) * dot(k, v) * k;
}

// The angle between a and b, in radians
double angle_between(const vec3& a, const vec3& b)
{
    return std::acos(std::clamp(dot(unit_vector(a), unit_vector(b)), -1.0, 1.0));
}

// How close a view may come to looking straight up or down: within a degree, the camera's frame (built from vup and
// the view direction, see camera) would turn unstable
constexpr double min_tilt = 1 * std::numbers::pi / 180;

// `offset` (a vector from the point the camera turns about) turned by yaw about vup, and by pitch toward vup (away
// from it if negative), the pitch limited so that the offset stays at least min_tilt from vup and from -vup (an offset
// already nearer, from a view given on the command line, comes no nearer: a limit never turns against the pitch). The
// pitch turns about offset × vup: by the right-hand rule, a positive angle about that axis brings the offset toward
// vup.
vec3 turn(const vec3& offset, const vec3& vup, double yaw, double pitch)
{
    const vec3 up = unit_vector(vup);
    vec3 turned = yaw != 0 ? rotate(offset, up, yaw) : offset;
    if (pitch != 0)
    {
        const double tilt = angle_between(turned, up); // from vup: 0 to pi
        const double lowest = std::min(min_tilt, tilt);
        const double highest = std::max(std::numbers::pi - min_tilt, tilt);
        const double wanted = std::clamp(tilt - pitch, lowest, highest);
        if (wanted != tilt)
        {
            turned = rotate(turned, unit_vector(cross(turned, up)), tilt - wanted);
        }
    }
    return turned;
}

// The camera's right axis: as the camera computes it, unit(vup × w), w pointing back from lookat to lookfrom
vec3 right_axis(const camera_settings& s)
{
    return unit_vector(cross(s.vup, unit_vector(s.lookfrom - s.lookat)));
}

// The focus distance where check_view wants it: from a thousandth to a thousand times the distance to lookat. A dolly
// can carry it out of that range, and any move changes the distance by a rounding error at least, which a focus just
// at a bound wouldn't survive; outside, it goes to the bound, a hair inside. A focus inside isn't touched.
camera_settings focus_in_range(camera_settings s)
{
    const double distance = (s.lookfrom - s.lookat).length(); // as check_view computes it
    if (!(s.focus_dist >= distance / 1000))
    {
        s.focus_dist = distance / 1000 * (1 + 1e-12);
    }
    else if (!(s.focus_dist <= distance * 1000))
    {
        s.focus_dist = distance * 1000 * (1 - 1e-12);
    }
    return s;
}
}

// Orbit: lookfrom turns about lookat, by yaw degrees about vup and pitch degrees up (the camera rises) or down
export camera_settings orbit(camera_settings s, double yaw_degrees, double pitch_degrees)
{
    if (yaw_degrees == 0 && pitch_degrees == 0)
    {
        return s;
    }
    s.lookfrom = s.lookat + turn(s.lookfrom - s.lookat, s.vup, radians(yaw_degrees), radians(pitch_degrees));
    return focus_in_range(s);
}

// Look: lookat turns about lookfrom, the way a head turns: by yaw degrees about vup and pitch degrees up or down
export camera_settings look(camera_settings s, double yaw_degrees, double pitch_degrees)
{
    if (yaw_degrees == 0 && pitch_degrees == 0)
    {
        return s;
    }
    s.lookat = s.lookfrom + turn(s.lookat - s.lookfrom, s.vup, radians(yaw_degrees), radians(pitch_degrees));
    return focus_in_range(s);
}

// Pan: lookfrom and lookat move together across the image plane by dx, dy pixels of an image `height` pixels high,
// measured at the distance of lookat, so the point under the cursor at that distance follows the cursor (a pan is in
// pixels: the same dx moves the camera less in a taller image)
export camera_settings pan(camera_settings s, double dx, double dy, int height)
{
    if (dx == 0 && dy == 0)
    {
        return s;
    }
    const vec3 back = s.lookfrom - s.lookat;
    const vec3 u = right_axis(s);
    const vec3 v = cross(unit_vector(back), u);
    const double per_pixel = 2 * back.length() * std::tan(radians(s.vfov) / 2) / height;
    const vec3 shift = (-dx * u + dy * v) * per_pixel;
    s.lookfrom = s.lookfrom + shift;
    s.lookat = s.lookat + shift;
    return focus_in_range(s);
}

// Dolly: lookfrom moves toward lookat (factor < 1) or away from it, and the focus distance by the same amount, so
// the plane in focus stays where it was in the scene. Neither comes nearer than `min_distance`, `min_focus` (or than
// it already is: a limit never moves the camera against the factor); the focus stays within check_view's range.
export camera_settings dolly(camera_settings s, double factor, double min_distance, double min_focus)
{
    if (factor == 1)
    {
        return s;
    }
    const vec3 back = s.lookfrom - s.lookat;
    const double distance = back.length();
    const double moved = std::max(distance * factor, std::min(min_distance, distance));
    s.lookfrom = s.lookat + back * (moved / distance);
    s.focus_dist = std::max(s.focus_dist + (moved - distance), std::min(min_focus, s.focus_dist));
    return focus_in_range(s);
}

// Zoom: the field of view times factor, from 1 to 160 degrees (a view given wider or narrower than that, on the
// command line, only comes back toward the range)
export camera_settings zoom(camera_settings s, double factor)
{
    if (factor == 1)
    {
        return s;
    }
    s.vfov = std::clamp(s.vfov * factor, std::min(1.0, s.vfov), std::max(160.0, s.vfov));
    return s;
}

// Fly: lookfrom and lookat move together, `forward` along the view, `right` to the side and `up` along vup
export camera_settings fly(camera_settings s, double forward, double right, double up)
{
    if (forward == 0 && right == 0 && up == 0)
    {
        return s;
    }
    const vec3 ahead = unit_vector(s.lookat - s.lookfrom);
    const vec3 shift = forward * ahead + right * right_axis(s) + up * unit_vector(s.vup);
    s.lookfrom = s.lookfrom + shift;
    s.lookat = s.lookat + shift;
    return focus_in_range(s);
}
