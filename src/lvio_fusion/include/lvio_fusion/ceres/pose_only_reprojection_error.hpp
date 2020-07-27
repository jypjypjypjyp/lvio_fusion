#ifndef lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
#define lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/sensors/camera.hpp"
#include <ceres/ceres.h>

namespace lvio_fusion
{

template <typename T>
inline void Reprojection(const T *p_w, const T *Tcw, Camera::Ptr camera, T *result)
{
    T p_c[3], p_c_[3];
    ceres::SE3TransformPoint(Tcw, p_w, p_c_);
    T extrinsic[7];
    ceres::Cast(camera->extrinsic.data(), SE3d::num_parameters, extrinsic);
    ceres::SE3TransformPoint(extrinsic, p_c_, p_c);
    T xp = p_c[0] / p_c[2];
    T yp = p_c[1] / p_c[2];
    result[0] = camera->fx * xp + camera->cx;
    result[1] = camera->fy * yp + camera->cy;
}

class PoseOnlyReprojectionError
{
public:
    PoseOnlyReprojectionError(Vector2d ob, Camera::Ptr camera, Vector3d point)
        : ob_x_(ob.x()), ob_y_(ob.y()), camera_(camera), x_(point.x()), y_(point.y()), z_(point.z()) {}

    template <typename T>
    bool operator()(const T *T_c_w, T *residuals) const
    {
        T p_p[2];
        T p_w[3] = {T(x_), T(y_), T(z_)};
        T ob[2] = {T(ob_x_), T(ob_y_)};
        Reprojection(p_w, T_c_w, camera_, p_p);
        residuals[0] = T(sqrt_information(0, 0)) * (p_p[0] - ob[0]);
        residuals[1] = T(sqrt_information(1, 1)) * (p_p[1] - ob[1]);
        return true;
    }

    static ceres::CostFunction *Create(Vector2d ob, Camera::Ptr camera, Vector3d point)
    {
        return (new ceres::AutoDiffCostFunction<PoseOnlyReprojectionError, 2, 7>(
            new PoseOnlyReprojectionError(ob, camera, point)));
    }

    static Matrix2d sqrt_information;

private:
    double ob_x_, ob_y_;
    double x_, y_, z_;
    Camera::Ptr camera_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
