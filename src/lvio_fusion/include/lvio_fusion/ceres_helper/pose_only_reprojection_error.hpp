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
    PoseOnlyReprojectionError(Vec2 observation, Camera::Ptr camera, Vec3 point)
        : observation_(observation), camera_(camera), point_(point)
    {
        Eigen::LLT<Mat22> llt(Mat22::Identity().inverse());
        sqrt_information_ = llt.matrixU();
    }

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        Eigen::Map<SE3 const> Tcw(parameters[0]);
        Eigen::Map<Vec2> residual(residuals);
        Vec3 P_c = camera_->world2camera(point_, Tcw);

        residual = sqrt_information_ * (observation_ - camera_->camera2pixel(P_c));

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian(jacobians[0]);
                Eigen::Matrix<double, 2, 3> jaco_res_2_Pc;
                jaco_res_2_Pc << -camera_->fx_ / P_c(2), 0, camera_->fx_ * P_c(0) / (P_c(2) * P_c(2)),
                    0, -camera_->fy_ / P_c(2), camera_->fy_ * P_c(1) / (P_c(2) * P_c(2));
                Eigen::Matrix<double, 3, 7> jaco_Pc_2_Pose;
                jaco_Pc_2_Pose.setZero();
                jaco_Pc_2_Pose.block<3, 3>(0, 0) = -Sophus::SO3d::hat(P_c);
                jaco_Pc_2_Pose.block<3, 3>(0, 4) = Eigen::Matrix3d::Identity();

                jacobian = sqrt_information_ * jaco_res_2_Pc * jaco_Pc_2_Pose;
            }
        }

        return true;
    }

private:
    Vec2 observation_;
    Camera::Ptr camera_;
    Vec3 point_;
    Mat22 sqrt_information_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
