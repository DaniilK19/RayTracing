// Module onb: an orthonormal basis, a frame of three mutually perpendicular unit vectors.
export module onb;

import std;
import vec3;

// An orthonormal basis (book 3, chapter 8): unit vectors u, v, w, each perpendicular to the other two, like the
// x, y, z axes but turned. The coordinates (x, y, z) in it mean the vector x·u + y·v + z·w. random_cosine_direction()
// draws directions around the z axis; a basis whose w is a surface normal turns them around that normal.
// A direction has no starting point, so the basis needs no origin.
// All members are inline: see the comment at the top of vec3.ixx.
export class onb
{
public:
    // unit_w must already have length 1 (hit normals have). It is taken as is: normalizing it again would cost
    // a square root and could change its last bits, while this way w() is exactly the given vector.
    inline explicit onb(const vec3& unit_w) : w_(unit_w)
    {
        // For any vector a not parallel to w, w × a is perpendicular to both. The x axis will do unless w is close
        // to it (|w.x| > 0.9); then the y axis. Either way |w × a| = √(1 - (w·a)²) ≥ √(1 - 0.9²) ≈ 0.44, far from 0.
        const vec3 a = std::fabs(w_.x) > 0.9 ? vec3{0, 1, 0} : vec3{1, 0, 0};
        v_ = unit_vector(cross(w_, a));
        // v and w are perpendicular unit vectors, so v × w has length 1 too. This order makes the basis
        // right-handed like x, y, z (u × v = w). The book takes w × v, the mirror image: just as good for
        // distributions that are symmetric around w, as all of ours are.
        u_ = cross(v_, w_);
    }

    inline const vec3& u() const { return u_; }
    inline const vec3& v() const { return v_; }
    inline const vec3& w() const { return w_; }

    // From coordinates in this basis to the world's: (x, y, z) → x·u + y·v + z·w
    inline vec3 transform(const vec3& a) const { return a.x * u_ + a.y * v_ + a.z * w_; }

private:
    vec3 u_;
    vec3 v_;
    vec3 w_;
};
