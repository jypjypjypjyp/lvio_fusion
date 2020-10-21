#ifndef lvio_fusion_LOOP_ERROR_H
#define lvio_fusion_LOOP_ERROR_H

#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"

namespace lvio_fusion
{

class TwoFrameReprojectionErrorBasedLoop
{
public:
    TwoFrameReprojectionErrorBasedLoop(Vector3d pr, Vector2d ob, Camera::Ptr camera, SE3d relative_pose)
    {
        origin_error_ = TwoFrameReprojectionError(pr, ob, camera);
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

    static ceres::CostFunction *Create(Vector3d pr, Vector2d ob, Camera::Ptr camera, SE3d relative_pose)
    {
        return (new ceres::AutoDiffCostFunction<TwoFrameReprojectionErrorBasedLoop, 2, 7, 7>(
            new TwoFrameReprojectionErrorBasedLoop(pr, ob, camera, relative_pose)));
    }

private:
    TwoFrameReprojectionError origin_error_;
    SE3d relative_pose_;
};

class LidarEdgeErrorBasedLoop
{
public:
    LidarEdgeErrorBasedLoop(Vector3d p, Vector3d pa, Vector3d pb, Lidar::Ptr lidar, SE3d relative_pose)
    {
        origin_error_ = LidarEdgeError(p, pa, pb, lidar);
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

    static ceres::CostFunction *Create(Vector3d p, Vector3d pa, Vector3d pb, Lidar::Ptr lidar, SE3d relative_pose)
    {
        return (new ceres::AutoDiffCostFunction<LidarEdgeErrorBasedLoop, 2, 7, 7>(
            new LidarEdgeErrorBasedLoop(p, pa, pb, lidar, relative_pose)));
    }

private:
    LidarEdgeError origin_error_;
    SE3d relative_pose_;
};

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

} // namespace lvio_fusion

#endif // lvio_fusion_LOOP_ERROR_H
