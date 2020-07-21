// #ifndef lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
// #define lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H

// #include "lvio_fusion/mappoint.h"
// #include "lvio_fusion/sensors/camera.hpp"
// #include "lvio_fusion/utility.h"
// #include <ceres/ceres.h>

// namespace lvio_fusion
// {

// class PoseOnlyReprojectionError
// {
// public:
//     PoseOnlyReprojectionError(Vector2d ob, Camerad::Ptr camera, Vector3d point)
//         : ob_(ob), camera_(camera), point_(point) {}

//     template <typename T>
//     bool operator()(const T *pose_, T *residuals_) const
//     {
//         Eigen::Map<Sophus::SE3<T> const> pose(pose_);
//         Eigen::Map<Matrix<T, 2, 1>> residuals(residuals_);
//         Camera<T> camera = camera_->cast<T>();
//         Matrix<T, 3, 1> P_c = camera.World2Sensor(point_.template cast<T>(), pose);
//         Matrix<T, 2, 1> pixel = camera.Sensor2Pixel(P_c);
//         residuals = ob_ - pixel;
//         residuals.applyOnTheLeft(sqrt_information);
//         return true;
//     }

//     static ceres::CostFunction *Create(Vector2d ob, Camerad::Ptr camera, Vector3d point)
//     {
//         return (new ceres::AutoDiffCostFunction<PoseOnlyReprojectionError, 2, 7>(
//             new PoseOnlyReprojectionError(ob, camera, point)));
//     }

//     static Matrix2d sqrt_information;

// private:
//     Vector2d ob_;
//     Camerad::Ptr camera_;
//     Vector3d point_;
// };

// } // namespace lvio_fusion

// #endif // lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H

#ifndef lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
#define lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H

#include <ceres/ceres.h>
#include "lvio_fusion/utility.h"
#include "lvio_fusion/camera.h"
#include "lvio_fusion/sensors/camera.hpp"

namespace lvio_fusion
{

class PoseOnlyReprojectionError : public ceres::SizedCostFunction<2, 7>
{
public:
    PoseOnlyReprojectionError(Vector2d observation, Camerad::Ptr camera, Vector3d point)
        : observation_(observation), camera_(camera), point_(point) {}

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        Eigen::Map<SE3d const> Tcw(parameters[0]);
        Eigen::Map<Vector2d> residual(residuals);
        Vector3d P_c = camera_->World2Sensor(point_, Tcw);

        residual = sqrt_information * (observation_ - camera_->Sensor2Pixel(P_c));

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Matrix<double, 2, 7, RowMajor>> jacobian(jacobians[0]);
                Matrix<double, 2, 3> jaco_res_2_Pc;
                jaco_res_2_Pc << -camera_->fx / P_c(2), 0, camera_->fx * P_c(0) / (P_c(2) * P_c(2)),
                    0, -camera_->fy / P_c(2), camera_->fy * P_c(1) / (P_c(2) * P_c(2));
                Matrix<double, 3, 7> jaco_Pc_2_Pose;
                jaco_Pc_2_Pose.setZero();
                jaco_Pc_2_Pose.block<3, 3>(0, 0) = -Sophus::SO3d::hat(P_c);
                jaco_Pc_2_Pose.block<3, 3>(0, 4) = Matrix3d::Identity();

                jacobian = sqrt_information * jaco_res_2_Pc * jaco_Pc_2_Pose;
            }
        }

        return true;
    }

    static Matrix2d sqrt_information;

private:
    Vector2d observation_;
    Camerad::Ptr camera_;
    Vector3d point_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
