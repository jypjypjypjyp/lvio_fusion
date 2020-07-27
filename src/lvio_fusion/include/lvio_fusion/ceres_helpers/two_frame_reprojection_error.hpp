#ifndef lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H
#define lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H

#include "lvio_fusion/ceres_helpers/pose_only_reprojection_error.hpp"
#include "lvio_fusion/sensors/camera.hpp"
#include "lvio_fusion/utility.h"
#include <ceres/ceres.h>

namespace lvio_fusion
{

template <typename T>
inline void Projection(const T *p_p, const T depth, const T *T_c_w, Camera::Ptr camera, T *result)
{
    //     T p_c[3];
    //     p_c[0] += (p_p[0] - camera->cx) * depth / camera->fx;
    //     p_c[1] += (p_p[1] - camera->cy) * depth / camera->fy;
    //     p_c[2] += depth;

    // T_c_w.inverse() * extrinsic.inverse() * p_c;
    //     ceres::EigenQuaternionRotatePoint(T_c_w, p_w, p_c);

    //     T xp = p_c[0] / p_c[2];
    //     T yp = p_c[1] / p_c[2];
    //     result[0] = camera->fx * xp + camera->cx;
    //     result[1] = camera->fy * yp + camera->cy;
}

class TwoFrameReprojectionError
{
public:
    TwoFrameReprojectionError(Vector2d ob1, Vector2d ob2, double depth, Camera::Ptr camera)
        : ob1_x_(ob1.x()), ob1_y_(ob1.y()), ob2_x_(ob2.x()), ob2_y_(ob2.y()), camera_(camera) {}

    template <typename T>
    bool operator()(const T *pose1_, const T *pose2_, T *residuals_) const
    {
        // T pixel[2];
        // T p_w[3] = {T(x_), T(y_), T(z_)};
        // T p_w[3] = {T(x_), T(y_), T(z_)};
        // Reprojection(p_w, T_c_w, camera_, pixel);
        // residuals[0] = T(sqrt_information(0, 0)) * (pixel[0] - ob_x_);
        // residuals[1] = T(sqrt_information(1, 1)) * (pixel[1] - ob_y_);
        return true;
    }

    static ceres::CostFunction *Create(Vector2d ob1, Vector2d ob2, double depth, Camera::Ptr camera)
    {
        return (new ceres::AutoDiffCostFunction<TwoFrameReprojectionError, 2, 7, 7>(
            new TwoFrameReprojectionError(ob1, ob2, depth, camera)));
    }

    static Matrix2d sqrt_information;

private:
    double ob1_x_, ob1_y_;
    double ob2_x_, ob2_y_;
    double depth_;
    Camera::Ptr camera_;
};

// class TwoFrameReprojectionError : public ceres::SizedCostFunction<2, 7, 7>
// {
// public:
//     TwoFrameReprojectionError(Vector2d observation, Camera::Ptr camera, Vector3d point)
//         : observation_(observation), camera_(camera), point_(point) {}

//     virtual bool Evaluate(double const *const *parameters,
//                           double *residuals,
//                           double **jacobians) const
//     {
//         Eigen::Map<SE3d const> Tcw(parameters[0]);
//         Eigen::Map<Vector2d> residual(residuals);
//         Vector3d P_c = camera_->World2Sensor(point_, Tcw);

//         residual = sqrt_information * (observation_ - camera_->Sensor2Pixel(P_c));

//         if (jacobians)
//         {
//             if (jacobians[0])
//             {
//                 Eigen::Map<Matrix<double, 2, 7, RowMajor>> jacobian(jacobians[0]);
//                 Matrix<double, 2, 3> jaco_res_2_Pc;
//                 jaco_res_2_Pc << -camera_->fx / P_c(2), 0, camera_->fx * P_c(0) / (P_c(2) * P_c(2)),
//                     0, -camera_->fy / P_c(2), camera_->fy * P_c(1) / (P_c(2) * P_c(2));
//                 Matrix<double, 3, 7> jaco_Pc_2_Pose;
//                 jaco_Pc_2_Pose.setZero();
//                 jaco_Pc_2_Pose.block<3, 3>(0, 0) = -Sophus::SO3d::hat(P_c);
//                 jaco_Pc_2_Pose.block<3, 3>(0, 4) = Matrix3d::Identity();
//                 jacobian = sqrt_information * jaco_res_2_Pc * jaco_Pc_2_Pose;
//             }
//         }

//         return true;
//     }

//     static Matrix2d sqrt_information;

// private:
//     Vector2d observation_;
//     Camera::Ptr camera_;
//     Vector3d point_;
// };

} // namespace lvio_fusion

#endif // lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H
