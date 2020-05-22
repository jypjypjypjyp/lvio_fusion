#ifndef lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
#define lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H

#include <ceres/ceres.h>
#include "lvio_fusion/utility.h"
#include "lvio_fusion/camera.h"

namespace lvio_fusion
{

class PoseOnlyReprojectionError : public ceres::SizedCostFunction<2, 7>
{
public:
    PoseOnlyReprojectionError(Vector2d observation, Camera::Ptr camera, Vector3d point)
        : observation_(observation), camera_(camera), point_(point)
    {
        LLT<Matrix2d> llt(Matrix2d::Identity().inverse());
        sqrt_information_ = llt.matrixU();
    }

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        Eigen::Map<SE3 const> Tcw(parameters[0]);
        Eigen::Map<Vector2d> residual(residuals);
        Vector3d P_c = camera_->world2camera(point_, Tcw);

        residual = sqrt_information_ * (observation_ - camera_->camera2pixel(P_c));

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

                jacobian = sqrt_information_ * jaco_res_2_Pc * jaco_Pc_2_Pose;
            }
        }

        return true;
    }

private:
    Vector2d observation_;
    Camera::Ptr camera_;
    Vector3d point_;
    Matrix2d sqrt_information_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
