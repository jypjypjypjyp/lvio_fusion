#ifndef lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
#define lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H

#include "lvio_fusion/mappoint.h"
#include "lvio_fusion/sensor.h"
#include "lvio_fusion/utility.h"
#include <ceres/ceres.h>

namespace lvio_fusion
{

class PoseOnlyReprojectionError
{
public:
    PoseOnlyReprojectionError(Vector2d ob, Sensor::Ptr sensor, MapPoint::Ptr mappoint)
        : ob_(ob), sensor_(sensor)
    {
        point_ = mappoint->Position();
    }

    template <typename T>
    bool operator()(const T *pose_, T *residuals_) const
    {
        Eigen::Map<Sophus::SE3<T> const> pose(pose_);
        Eigen::Map<Matrix<T, 2, 1>> residuals(residuals_);
        Matrix<T, 3, 1> P_c = sensor_->World2Sensor<T>(point_.template cast<T>(), pose);
        Matrix<T, 2, 1> pixel = sensor_->Sensor2Pixel<T>(P_c);
        residuals = ob_ - pixel;
        residuals.applyOnTheLeft(sqrt_information.template cast<T>());
        return true;
    }

    static ceres::CostFunction *Create(Vector2d ob, Sensor::Ptr sensor, MapPoint::Ptr mappoint)
    {
        return (new ceres::AutoDiffCostFunction<PoseOnlyReprojectionError, 2, 7>(
            new PoseOnlyReprojectionError(ob, sensor, mappoint)));
    }

    static Matrix2d sqrt_information;

private:
    Vector2d ob_;
    Sensor::Ptr sensor_;
    Vector3d point_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
