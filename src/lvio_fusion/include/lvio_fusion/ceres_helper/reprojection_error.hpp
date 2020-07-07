#ifndef lvio_fusion_REPROJECTION_ERROR_H
#define lvio_fusion_REPROJECTION_ERROR_H

#include "lvio_fusion/camera.h"
#include "lvio_fusion/utility.h"
#include <ceres/ceres.h>

namespace lvio_fusion
{

class ReprojectionError : public ceres::SizedCostFunction<2, 7, 3>
{
public:
    ReprojectionError(Vector2d observation, Camera::Ptr camera, Matrix2d covariance = covariance)
        : observation_(observation), camera_(camera)
    {
        LLT<Matrix2d> llt(covariance.inverse());
        sqrt_information_ = llt.matrixU();
    }

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        Eigen::Map<SE3d const> Tcw(parameters[0]);
        Eigen::Map<Vector3d const> point(parameters[1]);
        Eigen::Map<Vector2d> residual(residuals);
        Vector3d P_c = camera_->world2camera(point, Tcw);

        residual = sqrt_information_ * (observation_ - camera_->camera2pixel(P_c));

        if (jacobians)
        {
            Eigen::Matrix<double, 2, 3> jaco_res_2_Pc;
            jaco_res_2_Pc << -camera_->fx / P_c(2), 0, camera_->fx * P_c(0) / (P_c(2) * P_c(2)),
                0, -camera_->fy / P_c(2), camera_->fy * P_c(1) / (P_c(2) * P_c(2));
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian(jacobians[0]);
                Eigen::Matrix<double, 3, 7> jaco_Pc_2_Pose;
                jaco_Pc_2_Pose.setZero();
                jaco_Pc_2_Pose.block<3, 3>(0, 0) = -Sophus::SO3d::hat(P_c);
                jaco_Pc_2_Pose.block<3, 3>(0, 4) = Eigen::Matrix3d::Identity();
                jacobian = sqrt_information_ * jaco_res_2_Pc * jaco_Pc_2_Pose;
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian(jacobians[1]);
                Eigen::Matrix3d R = camera_->pose.rotationMatrix() * Tcw.rotationMatrix();
                jacobian = sqrt_information_ * jaco_res_2_Pc * R;
            }
        }

        return true;
    }

    static Matrix2d covariance;

private:
    Vector2d observation_;
    Camera::Ptr camera_;
    Matrix2d sqrt_information_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
