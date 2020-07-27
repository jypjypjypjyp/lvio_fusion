#ifndef lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
#define lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H

#include "lvio_fusion/ceres_helpers/base.hpp"
#include "lvio_fusion/sensors/camera.hpp"
#include <ceres/ceres.h>

namespace lvio_fusion
{

template <typename T>
inline void Reprojection(const T *p_w, const T *T_c_w, Camera::Ptr camera, T *result)
{
    T p_c[3], p_c_[3];
    ceres::EigenQuaternionRotatePoint(T_c_w, p_w, p_c_);
    p_c_[0] += T_c_w[4];
    p_c_[1] += T_c_w[5];
    p_c_[2] += T_c_w[6];
    T *extrinsic = camera->extrinsic.template cast<T>().data();
    ceres::EigenQuaternionRotatePoint(extrinsic, p_c_, p_c);
    p_c_[0] += extrinsic[4];
    p_c_[1] += extrinsic[5];
    p_c_[2] += extrinsic[6];
    T xp = p_c[0] / p_c[2];
    T yp = p_c[1] / p_c[2];
    result[0] = camera->fx * xp + camera->cx;
    result[1] = camera->fy * yp + camera->cy;
}

class PoseOnlyReprojectionError
{
public:
    PoseOnlyReprojectionError(Vector2d ob, Camera::Ptr camera, Vector3d point)
        : ob_x_(ob.x()), ob_y_(ob.y()), camera_(camera), x_(point.x()), y_(point.y()), z_(point.z()) {}

    template <typename T>
    bool operator()(const T *T_c_w_, T *residuals_) const
    {
        Eigen::Map<Sophus::SE3<T> const> T_c_w(T_c_w_);
        Eigen::Map<Matrix<T, 2, 1>> residuals(residuals_);
        double *ext_ptr = camera_->extrinsic.data();
        Sophus::SE3<T> ext(Quaternion<T>{T(ext_ptr[3]), T(ext_ptr[0]), T(ext_ptr[1]), T(ext_ptr[2])}, 
                            Matrix<T, 3, 1>{T(ext_ptr[4]), T(ext_ptr[5]), T(ext_ptr[6])});
        Matrix<T, 3, 1> p_w{T(x_), T(y_), T(z_)};
        Matrix<T, 2, 1> ob{T(ob_x_), T(ob_y_)};
        Matrix<T, 3, 1> p_c = ext * T_c_w * p_w;
        Matrix<T, 2, 1> p_p(camera_->fx * p_c(0, 0) / p_c(2, 0) + camera_->cx,
                            camera_->fy * p_c(1, 0) / p_c(2, 0) + camera_->cy);
        residuals = p_p - ob;
        // T p_p[2];
        // T p_w[3] = {T(x_), T(y_), T(z_)};
        // Reprojection(p_w, T_c_w, camera_, p_p);
        // residuals[0] = T(sqrt_information(0, 0)) * (p_p[0] - ob_x_);
        // residuals[1] = T(sqrt_information(1, 1)) * (p_p[1] - ob_y_);
        return true;
    }

    static ceres::CostFunction *Create(Vector2d ob, Camera::Ptr camera, Vector3d point)
    {
        return (new ceres::AutoDiffCostFunction<PoseOnlyReprojectionError, 2, 7>(
            new PoseOnlyReprojectionError(ob, camera, point)));
    }

    static Matrix2d sqrt_information;

private:
    double ob_x_, ob_y_;
    double x_, y_, z_;
    Camera::Ptr camera_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_ONLY_REPROJECTION_ERROR_H
