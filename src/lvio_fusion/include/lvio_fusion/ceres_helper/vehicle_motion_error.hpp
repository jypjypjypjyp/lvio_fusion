#ifndef lvio_fusion_VEHICLE_MOTION_ERROR_H
#define lvio_fusion_VEHICLE_MOTION_ERROR_H

#include <ceres/ceres.h>

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

class VehicleMotionError
{
public:
    VehicleMotionError() {}

    template <typename T>
    bool operator()(const T *pose_, T *residuals) const
    {
        Eigen::Map<Sophus::SE3<T> const> pose(pose_);
        Matrix<T, 3, 1> angle = pose.inverse().rotationMatrix().eulerAngles(1, 0, 2);
        residuals[0] = T(100) * pose.inverse().translation().y();
        residuals[1] = T(1000) * sin(angle[1]);
        residuals[2] = T(1000) * sin(angle[2]);
        return true;
    }

    static ceres::CostFunction *Create()
    {
        return new ceres::AutoDiffCostFunction<VehicleMotionError, 3, 7>(new VehicleMotionError());
    }

    static Matrix3d sqrt_information;
};

} // namespace lvio_fusion

#endif // lvio_fusion_VEHICLE_MOTION_ERROR_H
