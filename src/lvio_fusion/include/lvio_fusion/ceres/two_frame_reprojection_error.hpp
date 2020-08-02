#ifndef lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H
#define lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H

#include "lvio_fusion/ceres/pose_only_reprojection_error.hpp"
#include "lvio_fusion/sensors/camera.hpp"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

template <typename T>
inline void Projection(const T *p_c, const T *Tcw, T *result)
{
    T Tcw_inverse[7];
    ceres::SE3Inverse(Tcw, Tcw_inverse);
    ceres::SE3TransformPoint(Tcw_inverse, p_c, result);
}

class TwoFrameReprojectionError
{
public:
    TwoFrameReprojectionError(Vector3d p_r, Vector2d ob, Camera::Ptr camera)
        : p_r_(p_r), ob_(ob), camera_(camera) {}

    template <typename T>
    bool operator()(const T *Tcw1, const T *Tcw2, T *residuals) const
    {
        T p_p[2], p_w[3];
        T p_r[3] = {T(p_r_.x()), T(p_r_.y()), T(p_r_.z())};
        T ob2[2] = {T(ob_.x()), T(ob_.y())};
        Projection(p_r, Tcw1, p_w);
        Reprojection(p_w, Tcw2, camera_, p_p);
        residuals[0] = T(sqrt_info(0, 0)) * (p_p[0] - ob2[0]);
        residuals[1] = T(sqrt_info(1, 1)) * (p_p[1] - ob2[1]);
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p_r, Vector2d ob, Camera::Ptr camera)
    {
        return (new ceres::AutoDiffCostFunction<TwoFrameReprojectionError, 2, 7, 7>(
            new TwoFrameReprojectionError(p_r, ob, camera)));
    }

    static Matrix2d sqrt_info;

private:
    Vector3d p_r_;
    Vector2d ob_;
    Camera::Ptr camera_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H
