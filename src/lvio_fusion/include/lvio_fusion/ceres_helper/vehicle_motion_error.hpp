#ifndef lvio_fusion_VEHICLE_MOTION_ERROR_H
#define lvio_fusion_VEHICLE_MOTION_ERROR_H

#include <ceres/ceres.h>

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

class VehicleMotionError
{
public:
    VehicleMotionError(Matrix4d covariance = covariance)
    {
        LLT<Matrix4d> llt(covariance.inverse());
        sqrt_information_ = llt.matrixU();
    }

    template <typename T>
    bool operator()(const T *pose_, const T *last_pose_, T *residuals_) const
    {
        Eigen::Map<Sophus::SE3<T> const> pose(pose_);
        Eigen::Map<Sophus::SE3<T> const> last_pose(last_pose_);
        Eigen::Map<Matrix<T, 2, 1>> residuals(residuals_);
        Sophus::SE3<T> relative_motion = pose * last_pose.inverse();
        // Matrix<T, 3, 1> euler_angles = relative_motion.rotationMatrix().eulerAngles(0, 1, 2);
        residuals[0] = T(0.1) * pose.inverse().translation().y();
        residuals[1] = T(0.01) * relative_motion.translation().norm();
        // residuals.applyOnTheLeft(sqrt_information_);
        // LOG(INFO)<<residuals[0];//<<","<<residuals[1]<<","<<residuals[2];
        return true;
    }

    static ceres::CostFunction *Create()
    {
        return new ceres::AutoDiffCostFunction<VehicleMotionError, 2, 7, 7>(new VehicleMotionError());
    }

    static Matrix4d covariance;

private:
    Matrix4d sqrt_information_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_VEHICLE_MOTION_ERROR_H
