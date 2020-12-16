#ifndef lvio_fusion_NAVSAT_ERROR_H
#define lvio_fusion_NAVSAT_ERROR_H

#include "lvio_fusion/ceres/base.hpp"

namespace lvio_fusion
{

class NavsatError
{
public:
    NavsatError(Vector3d p, double *weights) : p_(p)
    {
        weights_[0] = weights[0];
        weights_[1] = weights[1];
        weights_[2] = weights[2];
    }

    template <typename T>
    bool operator()(const T *Twc, T *residuals) const
    {
        residuals[0] = T(weights_[0]) * (Twc[4] - T(p_[0]));
        residuals[1] = T(weights_[1]) * (Twc[5] - T(p_[1]));
        residuals[2] = T(weights_[2]) * (Twc[6] - T(p_[2]));
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p, double *weights)
    {
        return (new ceres::AutoDiffCostFunction<NavsatError, 3, 7>(new NavsatError(p, weights)));
    }

private:
    Vector3d p_; //position
    double weights_[3];
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
        T p1[3] = {T(x1_), T(y1_), T(z1_)};
        T tf_p1[3];
        ceres::SE3TransformPoint(tf, p1, tf_p1);
        residuals[0] = (T(x0_) - tf_p1[0]);
        residuals[1] = (T(y0_) - tf_p1[1]);
        residuals[2] = (T(z0_) - tf_p1[2]);
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
