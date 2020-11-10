#ifndef lvio_fusion_LOOP_ERROR_H
#define lvio_fusion_LOOP_ERROR_H

#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"

namespace lvio_fusion
{

class PoseGraphError
{
public:
    PoseGraphError() = default;
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

// class PoseGraphErrorBasedLoop
// {
// public:
//     PoseGraphErrorBasedLoop(SE3d last_frame, SE3d frame, SE3d relative_pose)
//     {
//         origin_error_ = PoseGraphError(last_frame, frame);
//         relative_pose_ = relative_pose;
//     }

//     template <typename T>
//     bool operator()(const T *Tcw_old, const T *Tcw2, T *residuals) const
//     {
//         T relative_pose[7], Tcw1[7];
//         ceres::Cast(relative_pose_.data(), SE3d::num_parameters, relative_pose);
//         ceres::SE3Product(relative_pose, Tcw_old, Tcw1);
//         return origin_error_(Tcw1, Tcw2, residuals);
//     }

//     static ceres::CostFunction *Create(SE3d last_frame, SE3d frame, SE3d relative_pose)
//     {
//         return (new ceres::AutoDiffCostFunction<PoseGraphErrorBasedLoop, 6, 7, 7>(
//             new PoseGraphErrorBasedLoop(last_frame, frame, relative_pose)));
//     }

// private:
//     PoseGraphError origin_error_;
//     SE3d relative_pose_;
// };

// class TwoFrameReprojectionErrorBasedLoop
// {
// public:
//     TwoFrameReprojectionErrorBasedLoop(Vector3d pr, Vector2d ob, Camera::Ptr camera, SE3d relative_pose)
//     {
//         origin_error_ = TwoFrameReprojectionError(pr, ob, camera);
//         relative_pose_ = relative_pose;
//     }

//     template <typename T>
//     bool operator()(const T *Tcw_old, const T *Tcw2, T *residuals) const
//     {
//         T relative_pose[7], Tcw1[7];
//         ceres::Cast(relative_pose_.data(), SE3d::num_parameters, relative_pose);
//         ceres::SE3Product(relative_pose, Tcw_old, Tcw1);
//         return origin_error_(Tcw1, Tcw2, residuals);
//     }

//     static ceres::CostFunction *Create(Vector3d pr, Vector2d ob, Camera::Ptr camera, SE3d relative_pose)
//     {
//         return (new ceres::AutoDiffCostFunction<TwoFrameReprojectionErrorBasedLoop, 2, 7, 7>(
//             new TwoFrameReprojectionErrorBasedLoop(pr, ob, camera, relative_pose)));
//     }

// private:
//     TwoFrameReprojectionError origin_error_;
//     SE3d relative_pose_;
// };

// class LidarEdgeErrorBasedLoop
// {
// public:
//     LidarEdgeErrorBasedLoop(Vector3d p, Vector3d pa, Vector3d pb, Lidar::Ptr lidar, SE3d relative_pose)
//     {
//         origin_error_ = LidarEdgeError(p, pa, pb, lidar);
//         relative_pose_ = relative_pose;
//     }

//     template <typename T>
//     bool operator()(const T *Tcw_old, const T *Tcw2, T *residuals) const
//     {
//         T relative_pose[7], Tcw1[7];
//         ceres::Cast(relative_pose_.data(), SE3d::num_parameters, relative_pose);
//         ceres::SE3Product(relative_pose, Tcw_old, Tcw1);
//         return origin_error_(Tcw1, Tcw2, residuals);
//     }

//     static ceres::CostFunction *Create(Vector3d p, Vector3d pa, Vector3d pb, Lidar::Ptr lidar, SE3d relative_pose)
//     {
//         return (new ceres::AutoDiffCostFunction<LidarEdgeErrorBasedLoop, 2, 7, 7>(
//             new LidarEdgeErrorBasedLoop(p, pa, pb, lidar, relative_pose)));
//     }

// private:
//     LidarEdgeError origin_error_;
//     SE3d relative_pose_;
// };

class LidarPlaneErrorBasedLoop
{
public:
    LidarPlaneErrorBasedLoop(const Vector3d p, const Vector3d pa, const Vector3d pb, const Vector3d pc, Lidar::Ptr lidar, SE3d relative_pose)
    {
        origin_error_ = LidarPlaneError(p, pa, pb, pc, lidar);
        relative_pose_ = relative_pose;
    }

    template <typename T>
    bool operator()(const T *Tcw_old, const T *Tcw2, T *residuals) const
    {
        T relative_pose[7], Tcw1[7];
        ceres::Cast(relative_pose_.data(), SE3d::num_parameters, relative_pose);
        ceres::SE3Product(relative_pose, Tcw_old, Tcw1);
        return origin_error_(Tcw1, Tcw2, residuals);
    }

    static ceres::CostFunction *Create(const Vector3d p, const Vector3d pa, const Vector3d pb, const Vector3d pc, Lidar::Ptr lidar, SE3d relative_pose)
    {
        return (new ceres::AutoDiffCostFunction<LidarPlaneErrorBasedLoop, 2, 7, 7>(
            new LidarPlaneErrorBasedLoop(p, pa, pb, pc, lidar, relative_pose)));
    }

private:
    LidarPlaneError origin_error_;
    SE3d relative_pose_;
};

class LidarPlaneRErrorBasedLoop
{
public:
    LidarPlaneRErrorBasedLoop(Vector3d p, Vector3d pa, Vector3d pb, Vector3d pc, Vector3d last_kf_t, Vector3d kf_t, Lidar::Ptr lidar, SE3d relative_pose)
    {
        origin_error_ = LidarPlaneRError(p, pa, pb, pc, last_kf_t, kf_t, lidar);
        relative_pose_ = relative_pose;
    }

    template <typename T>
    bool operator()(const T *q_old, const T *q2, T *residuals) const
    {
        T relative_pose[7], q1[4];
        ceres::Cast(relative_pose_.data(), SO3d::num_parameters, relative_pose);
        ceres::EigenQuaternionProduct(relative_pose, q_old, q1);
        return origin_error_(q1, q2, residuals);
    }

    static ceres::CostFunction *Create(Vector3d p, Vector3d pa, Vector3d pb, Vector3d pc, Vector3d last_kf_t, Vector3d kf_t, Lidar::Ptr lidar, SE3d relative_pose)
    {
        return (new ceres::AutoDiffCostFunction<LidarPlaneRErrorBasedLoop, 1, 4, 4>(
            new LidarPlaneRErrorBasedLoop(p, pa, pb, pc, last_kf_t, kf_t, lidar, relative_pose)));
    }

private:
    LidarPlaneRError origin_error_;
    SE3d relative_pose_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_LOOP_ERROR_H
