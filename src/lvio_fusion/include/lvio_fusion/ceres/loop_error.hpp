#ifndef lvio_fusion_LOOP_ERROR_H
#define lvio_fusion_LOOP_ERROR_H

#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"

namespace lvio_fusion
{

class PoseGraphError
{
public:
    PoseGraphError(SE3d last_frame, SE3d frame)
    {
        relative_pose_ = frame * last_frame.inverse();
        angle_x_ = normalize_angle(relative_pose_.angleX());
        angle_y_ = normalize_angle(relative_pose_.angleY());
        angle_z_ = normalize_angle(relative_pose_.angleZ());
        Vector3d angle = {angle_x_, angle_y_, angle_z_};
        if (angle.norm() > 0.25)
        {
            weight_ = 0.1;
        }
        else
        {
            weight_ = 1;
        }
    }

    template <typename T>
    bool operator()(const T *Tcw1, const T *Tcw2, T *residuals) const
    {
        T relative_pose[7], Tcw1_inverse[7];
        ceres::SE3Inverse(Tcw1, Tcw1_inverse);
        ceres::SE3Product(Tcw2, Tcw1_inverse, relative_pose);

        // Eigen::Map<Sophus::SE3<T>> relative_pose_se3(relative_pose);
        // T angle_x = normalize_angle<T>(relative_pose_se3.angleX());
        // T angle_y = normalize_angle<T>(relative_pose_se3.angleY());
        // T angle_z = normalize_angle<T>(relative_pose_se3.angleZ());

        residuals[0] = relative_pose[4] - T(relative_pose_.translation().x());
        residuals[1] = relative_pose[5] - T(relative_pose_.translation().y());
        residuals[2] = relative_pose[6] - T(relative_pose_.translation().z());
        // residuals[3] = T(weight_) * (angle_x - T(angle_x_));
        // residuals[4] = T(weight_) * (angle_y - T(angle_y_));
        // residuals[5] = T(weight_) * (angle_z - T(angle_z_));
        return true;
    }

    static ceres::CostFunction *Create(SE3d last_frame, SE3d frame)
    {
        return (new ceres::AutoDiffCostFunction<PoseGraphError, 3, 7, 7>(
            new PoseGraphError(last_frame, frame)));
    }

private:
    SE3d relative_pose_;
    double angle_x_, angle_y_, angle_z_;
    double weight_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_LOOP_ERROR_H
