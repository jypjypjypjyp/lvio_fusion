#ifndef lvio_fusion_NAVSAT_ERROR_H
#define lvio_fusion_NAVSAT_ERROR_H

#include "lvio_fusion/ceres/base.hpp"

namespace lvio_fusion
{

Vector3d cov2sqrt_info(const Vector3d &cov)
{
    Matrix3d cov_m;
    cov_m << cov[0], 0, 0, 0, cov[1], 0, 0, 0, cov[2];
    Matrix3d sqrt_info_m = Eigen::LLT<Matrix3d>(cov_m.inverse()).matrixL().transpose();
    return Vector3d(sqrt_info_m(0), sqrt_info_m(4), sqrt_info_m(8));
}

class NavsatInitError
{
public:
    NavsatInitError(Vector3d p0, Vector3d p1, Vector3d cov)
        : x0_(p0.x()), y0_(p0.y()), z0_(p0.z()),
          x1_(p1.x()), y1_(p1.y()), z1_(p1.z())
    {
        sqrt_info_ = cov2sqrt_info(cov);
    }

    template <typename T>
    bool operator()(const T *yaw, const T *x, const T *y, T *residuals) const
    {
        T tf[7];
        T rpyxyz[6] = {yaw[0], T(0), T(0), x[0], y[0], T(0)};
        ceres::RpyxyzToSE3(rpyxyz, tf);
        T p1[3] = {T(x1_), T(y1_), T(z1_)};
        T tf_p1[3];
        ceres::SE3TransformPoint(tf, p1, tf_p1);
        residuals[0] = T(sqrt_info_[0]) * (T(x0_) - tf_p1[0]);
        residuals[1] = T(sqrt_info_[1]) * (T(y0_) - tf_p1[1]);
        residuals[2] = T(sqrt_info_[2]) * (T(z0_) - tf_p1[2]);
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p0, Vector3d p1, Vector3d cov)
    {
        return (new ceres::AutoDiffCostFunction<NavsatInitError, 3, 1, 1, 1>(new NavsatInitError(p0, p1, cov)));
    }

private:
    double x0_, y0_, z0_;
    double x1_, y1_, z1_;
    Vector3d sqrt_info_;
};

class NavsatRXError
{
public:
    NavsatRXError(Vector3d p0, Vector3d p1, SE3d pose, Vector3d cov)
        : x0_(p0.x()), y0_(p0.y()), z0_(p0.z()),
          x1_(p1.x()), y1_(p1.y()), z1_(p1.z()),
          pose_(pose)
    {
        sqrt_info_ = cov2sqrt_info(cov);
    }

    template <typename T>
    bool operator()(const T *yaw, const T *pitch, const T *roll, const T *x, const T *y, const T *z, T *residuals) const
    {
        T pose[7], tf[7], relative_pose[7];
        T rpyxyz[6] = {*yaw, *pitch, *roll, *x, *y, *z};
        ceres::RpyxyzToSE3(rpyxyz, relative_pose);
        ceres::Cast(pose_.data(), SE3d::num_parameters, pose);
        ceres::SE3Product(pose, relative_pose, tf);
        T p1[3] = {T(x1_), T(y1_), T(z1_)};
        T tf_p1[3];
        ceres::SE3TransformPoint(tf, p1, tf_p1);
        residuals[0] = T(sqrt_info_[0]) * (T(x0_) - tf_p1[0]);
        residuals[1] = T(sqrt_info_[1]) * (T(y0_) - tf_p1[1]);
        residuals[2] = T(sqrt_info_[2]) * (T(z0_) - tf_p1[2]);
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p0, Vector3d p1, SE3d pose, Vector3d cov)
    {
        return (new ceres::AutoDiffCostFunction<NavsatRXError, 3, 1, 1, 1, 1, 1, 1>(new NavsatRXError(p0, p1, pose, cov)));
    }

private:
    double x0_, y0_, z0_;
    double x1_, y1_, z1_;
    SE3d pose_;
    Vector3d sqrt_info_;
};

class NavsatRError
{
public:
    NavsatRError(Vector3d y, SE3d pose) : y_(y), pose_(pose) {}

    template <typename T>
    bool operator()(const T *roll, T *residual) const
    {
        T pose[4], relative_pose[4], pb[4], y[3], tf_y[3];
        T rpy[3] = {T(0), T(0), roll[0]};
        ceres::RPYToEigenQuaternion(rpy, relative_pose);
        ceres::Cast(pose_.data(), SO3d::num_parameters, pose);
        ceres::Cast(y_.data(), 3, y);
        ceres::EigenQuaternionProduct(pose, relative_pose, pb);
        ceres::EigenQuaternionRotatePoint(pb, y, tf_y);
        residual[0] = tf_y[2];
        return true;
    }

    static ceres::CostFunction *Create(Vector3d y, SE3d pose)
    {
        return (new ceres::AutoDiffCostFunction<NavsatRError, 1, 1>(new NavsatRError(y, pose)));
    }

private:
    SE3d pose_;
    Vector3d y_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_ERROR_H
