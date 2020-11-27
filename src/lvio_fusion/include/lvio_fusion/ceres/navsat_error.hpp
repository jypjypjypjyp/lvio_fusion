#ifndef lvio_fusion_NAVSAT_ERROR_H
#define lvio_fusion_NAVSAT_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/common.h"

namespace lvio_fusion
{

class NavsatError
{
public:
    NavsatError(Vector3d p, double* weights) : x_(p.x()), y_(p.y()), z_(p.z()), weights_(weights) {}

    template <typename T>
    bool operator()(const T *pose, T *residuals) const
    {
        T pose_inverse[7];
        ceres::SE3Inverse(pose, pose_inverse);
        residuals[0] = T(weights_[0]) * (pose_inverse[4] - T(x_));
        residuals[1] = T(weights_[1]) * (pose_inverse[5] - T(y_));
        residuals[2] = T(weights_[2]) * (pose_inverse[6] - T(z_));
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p, double* weights)
    {
        return (new ceres::AutoDiffCostFunction<NavsatError, 3, 7>(new NavsatError(p, weights)));
    }

private:
    double x_, y_, z_;
    const double *weights_;
};

class NavsatInitError
{
public:
    NavsatInitError(Vector3d p0, Vector3d p1)
        : x0_(p0.x()), y0_(p0.y()), z0_(p0.z()),
          x1_(p1.x()), y1_(p1.y()), z1_(p1.z()) {}

    template <typename T>
    bool operator()(const T *tf, T *residuals) const
    {
        T p0[3] = {T(x0_), T(y0_), T(z0_)};
        T p1_[3] = {T(x1_), T(y1_), T(z1_)};
        T p1[3];
        ceres::SE3TransformPoint(tf, p1_, p1);
        residuals[0] = p0[0] - p1[0];
        residuals[1] = p0[1] - p1[1];
        residuals[2] = p0[2] - p1[2];
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p0, Vector3d p1)
    {
        return (new ceres::AutoDiffCostFunction<NavsatInitError, 3, 7>(new NavsatInitError(p0, p1)));
    }

private:
    double x0_, y0_, z0_;
    double x1_, y1_, z1_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_ERROR_H
