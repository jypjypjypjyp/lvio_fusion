#ifndef lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H
#define lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H

#include "lvio_fusion/ceres/pose_only_reprojection_error.hpp"
#include "lvio_fusion/sensors/camera.hpp"
#include "lvio_fusion/utility.h"
#include <ceres/ceres.h>

namespace lvio_fusion
{

template <typename T>
inline void Projection(const T *p_p, const T &depth, const T *Tcw, Camera::Ptr camera, T *result)
{
    T p_c[3], p_c_[3];
    p_c[0] = (p_p[0] - camera->cx) * depth / camera->fx;
    p_c[1] = (p_p[1] - camera->cy) * depth / camera->fy;
    p_c[2] = depth;
    T extrinsic[7];
    ceres::Cast(camera->extrinsic.data(), 7, extrinsic);
    T Tcw_inverse[7], extrinsic_inverse[7];
    ceres::SE3Inverse(Tcw, Tcw_inverse);
    ceres::SE3Inverse(extrinsic, extrinsic_inverse);
    ceres::SE3TransformPoint(extrinsic_inverse, p_c, p_c_);
    ceres::SE3TransformPoint(Tcw_inverse, p_c_, result);
}

class TwoFrameReprojectionError
{
public:
    TwoFrameReprojectionError(Vector2d ob1, Vector2d ob2, double depth, Camera::Ptr camera)
        : ob1_x_(ob1.x()), ob1_y_(ob1.y()), ob2_x_(ob2.x()), ob2_y_(ob2.y()), depth_(depth), camera_(camera) {}

    template <typename T>
    bool operator()(const T *Tcw1, const T *Tcw2, T *residuals) const
    {
        T p_p[2], p_w[3];
        T ob1[2] = {T(ob1_x_), T(ob1_y_)};
        T ob2[2] = {T(ob2_x_), T(ob2_y_)};
        Projection(ob1, T(depth_), Tcw1, camera_, p_w);
        Reprojection(p_w, Tcw2, camera_, p_p);
        residuals[0] = T(sqrt_information(0, 0)) * (p_p[0] - ob2[0]);
        residuals[1] = T(sqrt_information(1, 1)) * (p_p[1] - ob2[1]);
        // LOG(INFO) << "pw" << p_w[0];
        // LOG(INFO) << "pw" << p_w[1];
        // LOG(INFO) << "pw" << p_w[2];
        // Eigen::Map<Sophus::SE3<T> const> Tcw1_(Tcw1);
        // Eigen::Map<Sophus::SE3<T> const> Tcw2_(Tcw2);
        // Matrix<T, 3, 1> p_c(T((ob1_x_ - camera_->cx) * depth_ / camera_->fx),
        //                     T((ob1_y_ - camera_->cy) * depth_ / camera_->fy),
        //                     T(depth_));
        // Matrix<T,3,1> p_w_ = Tcw1_.inverse() * camera_->extrinsic.inverse().template cast<T>() * p_c;
        // LOG(INFO) << p_w_[0];
        // LOG(INFO) << p_w_[1];
        // LOG(INFO) << p_w_[2];
        // LOG(INFO) << "********************************************";
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

} // namespace lvio_fusion

#endif // lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H
