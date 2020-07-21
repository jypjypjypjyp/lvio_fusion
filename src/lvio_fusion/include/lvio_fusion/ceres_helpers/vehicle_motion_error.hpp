#ifndef lvio_fusion_VEHICLE_MOTION_ERROR_H
#define lvio_fusion_VEHICLE_MOTION_ERROR_H

#include <ceres/ceres.h>

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

class VehicleMotionErrorA
{
public:
    VehicleMotionErrorA(Vector3d v) : v_(v) {}

    template <typename T>
    bool operator()(const T *pose_, T *residuals) const
    {
        Eigen::Map<Sophus::SE3<T> const> pose(pose_);

        return true;
    }

    static ceres::CostFunction *Create(Vector3d v)
    {
        return new ceres::AutoDiffCostFunction<VehicleMotionErrorA, 2, 7>(new VehicleMotionErrorA(v));
    }

private:
    Vector3d v_;
};

class VehicleMotionErrorB
{
public:
    VehicleMotionErrorB(double dt) : dt_(dt) {}

    template <typename T>
    bool operator()(const T *pose1_, const T *pose2_, T *residuals) const
    {
        Eigen::Map<Sophus::SE3<T> const> pose1(pose1_);
        Eigen::Map<Sophus::SE3<T> const> pose2(pose2_);
        Sophus::SE3<T> relative_motion = pose2 * pose1.inverse();
        Matrix<T, 3, 1> angle = relative_motion.rotationMatrix().eulerAngles(1, 0, 2);
        residuals[0] = T(100) * tan(angle[1]);
        residuals[1] = T(100) * tan(angle[2]);
        return true;
    }

    static ceres::CostFunction *Create(double dt)
    {
        return new ceres::AutoDiffCostFunction<VehicleMotionErrorB, 2, 7, 7>(new VehicleMotionErrorB(dt));
    }

private:
    double dt_;
};

// class VehicleMotionErrorC
// {
// public:
//     VehicleMotionErrorC() {}

//     template <typename T>
//     bool operator()(const T *pose_, T *residuals) const
//     {
//         Eigen::Map<Sophus::SE3<T> const> pose(pose_);
//         Matrix<T, 3, 1> angle = pose.inverse().rotationMatrix().eulerAngles(1, 0, 2);
//         residuals[0] = T(100) * pose.inverse().translation().y();
//         residuals[1] = T(1000) * sin(angle[1]);
//         residuals[2] = T(1000) * sin(angle[2]);
//         return true;
//     }

//     static ceres::CostFunction *Create()
//     {
//         return new ceres::AutoDiffCostFunction<VehicleMotionErrorC, 3, 7, 7, 7>(new VehicleMotionErrorC());
//     }
// };

} // namespace lvio_fusion

#endif // lvio_fusion_VEHICLE_MOTION_ERROR_H
