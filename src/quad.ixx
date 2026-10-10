// Module quad: a quadrilateral, a flat shape for walls, floors and light sources.
export module quad;

import std;
import aabb;
import hittable;
import interval;
import ray;
import vec3;

// A parallelogram: corner q and two side vectors u and v. Its points are q + α·u + β·v with α and β from 0 to 1.
export class quad : public hittable
{
public:
    quad(const point3& q, const vec3& u, const vec3& v, std::shared_ptr<const material> mat)
        : q_(q), u_(u), v_(v), mat_(std::move(mat))
    {
        // The plane the shape lies in: dot(normal, P) = d for all its points P
        const vec3 n = cross(u, v);
        normal_ = unit_vector(n);
        d_ = dot(normal_, q);
        w_ = n / dot(n, n);

        // Box around all four corners: the union of the boxes of the two diagonals
        bbox_ = enclose(aabb::from_corners(q, q + u + v), aabb::from_corners(q + u, q + v));
    }

    aabb bounding_box() const override { return bbox_; }

    std::optional<hit_record> hit(const ray& r, interval ray_t) const override
    {
        // Where the ray crosses the plane: substitute P = origin + t·direction into dot(normal, P) = d
        const double denom = dot(normal_, r.direction);
        if (std::fabs(denom) < 1e-8)
        {
            return std::nullopt; // the ray is parallel to the plane
        }
        const double t = (d_ - dot(normal_, r.origin)) / denom;
        if (!ray_t.contains(t))
        {
            return std::nullopt;
        }

        // Decompose the intersection point along the sides: p = P − q = α·u + β·v.
        // Cross with v: p × v = α·(u × v) = α·n, because v × v = 0.
        // Now dot with n: dot(n, p × v) = α·|n|², i.e. α = dot(w, p × v), where w = n / |n|².
        // Likewise β = dot(w, u × p).
        const point3 intersection = r.at(t);
        const vec3 p = intersection - q_;
        const double alpha = dot(w_, cross(p, v_));
        const double beta = dot(w_, cross(u_, p));

        // The point is inside the shape if both coordinates are between 0 and 1
        constexpr interval unit{0, 1};
        if (!unit.contains(alpha) || !unit.contains(beta))
        {
            return std::nullopt;
        }

        hit_record rec;
        rec.t = t;
        rec.p = intersection;
        rec.mat = mat_.get();
        rec.set_face_normal(r, normal_);
        rec.u = alpha; // α and β are ready-made texture coordinates
        rec.v = beta;
        return rec;
    }

private:
    point3 q_;
    vec3 u_;
    vec3 v_;
    std::shared_ptr<const material> mat_;
    vec3 normal_; // unit normal of the plane
    double d_;    // dot(normal, P) for any point P of the plane
    vec3 w_;      // n / |n|², where n = u × v: needed to find α and β
    aabb bbox_;
};
