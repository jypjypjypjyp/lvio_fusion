#ifndef lvio_fusion_NAVSAT_ERROR_H
#define lvio_fusion_NAVSAT_ERROR_H

#include "lvio_fusion/ceres/base.hpp"

namespace lvio_fusion
{

class NavsatError
{
public:
    NavsatError(Vector3d A, Vector3d B, Vector3d C, double *weights)
        : A_(A)
    {
        abc_norm_ = (A_ - B).cross(A_ - C);
        abc_norm_.normalize();
        weights_[0] = weights[0];
    }

    template <typename T>
    bool operator()(const T *Twc, T *residuals) const
    {
        T unit_x[3] = {T(100), T(100), T(0)}, AP[3], axis_x[3];
        T A[3] = {T(A_.x()), T(A_.y()), T(A_.z())};
        T adc_norm[3] = {T(abc_norm_.x()), T(abc_norm_.y()), T(abc_norm_.z())};
        ceres::SE3TransformPoint(Twc, unit_x, axis_x);
        ceres::Minus(axis_x, A, AP);

        residuals[0] = T(weights_[0]) * ceres::DotProduct(AP, adc_norm);
        return true;
    }

    static ceres::CostFunction *Create(Vector3d A, Vector3d B, Vector3d C, double *weights)
    {
        return (new ceres::AutoDiffCostFunction<NavsatError, 1, 7>(new NavsatError(A, B, C, weights)));
    }

private:
    Vector3d A_, abc_norm_; // ground level (ABC)
    double weights_[1];
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
