#ifndef lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
#define lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H

#include "lvio_fusion/mappoint.h"
#include "lvio_fusion/sensors/camera.hpp"
#include "lvio_fusion/utility.h"
#include <ceres/ceres.h>

namespace lvio_fusion
{

class PoseOnlyReprojectionError
{
public:
    PoseOnlyReprojectionError(Vector2d ob, Camerad::Ptr camera, Vector3d point)
        : ob_(ob), camera_(camera), point_(point) {}

    template <typename T>
    bool operator()(const T *pose_, T *residuals_) const
    {
        Eigen::Map<Sophus::SE3<T> const> pose(pose_);
        Eigen::Map<Matrix<T, 2, 1>> residuals(residuals_);
        Camera<T> camera = camera_->cast<T>();
        Matrix<T, 3, 1> P_c = camera.World2Sensor(point_.cast<T>(), pose);
        Matrix<T, 2, 1> pixel = camera.Sensor2Pixel(P_c);
        residuals = ob_ - pixel;
        residuals.applyOnTheLeft(sqrt_information);
        return true;
    }

    static ceres::CostFunction *Create(Vector2d ob, Camerad::Ptr camera, Vector3d point)
    {
        return (new ceres::AutoDiffCostFunction<PoseOnlyReprojectionError, 2, 7>(
            new PoseOnlyReprojectionError(ob, camera, point)));
    }

    static Matrix2d sqrt_information;

private:
    Vector2d ob_;
    Camerad::Ptr camera_;
    Vector3d point_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
