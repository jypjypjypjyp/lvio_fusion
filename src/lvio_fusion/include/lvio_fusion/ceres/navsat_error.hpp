#ifndef lvio_fusion_NAVSAT_ERROR_H
#define lvio_fusion_NAVSAT_ERROR_H

#include "lvio_fusion/ceres/base.hpp"

namespace lvio_fusion
{

class NavsatError
{
public:
    NavsatError(Vector3d p, Vector3d last, Vector3d A, Vector3d B, Vector3d C, double *weights)
        : p_(p), A_(A), B_(B), C_(C)
    {
        heading_ = (p - last);
        heading_.normalize();
        abc_norm_ = (A_ - B_).cross(A_ - C_);
        abc_norm_.normalize();
        weights_[0] = weights[0];
        weights_[1] = weights[1];
        weights_[2] = weights[2];
        weights_[3] = weights[3];
        weights_[4] = weights[4];
        weights_[5] = weights[5];
        weights_[6] = weights[6];
    }

    template <typename T>
    bool operator()(const T *Twc, T *residuals) const
    {
        T unit_x[3] = {T(1), T(0), T(0)}, axis_x[3];
        T heading[3] = {T(heading_.x()), T(heading_.y()), T(heading_.z())};
        ceres::EigenQuaternionRotatePoint(Twc, unit_x, axis_x);
        ceres::Cast(heading_.data(), 3, heading);

        T unit_y[3] = {T(0), T(1), T(0)}, AP[3], axis_y[3];
        T A[3] = {T(A_.x()), T(A_.y()), T(A_.z())};
        T adc_norm[3] = {T(abc_norm_.x()), T(abc_norm_.y()), T(abc_norm_.z())};
        ceres::SE3TransformPoint(Twc, unit_y, axis_y);
        ceres::Minus(axis_y, A, AP);

        T p[3];
        ceres::Cast(p_.data(), 3, p);

        residuals[0] = T(weights_[0]) * (heading_[0] - heading[0]);
        residuals[1] = T(weights_[1]) * (heading_[1] - heading[1]);
        residuals[2] = T(weights_[2]) * (heading_[2] - heading[2]);
        residuals[3] = T(weights_[3]) * ceres::DotProduct(AP, adc_norm);
        residuals[4] = T(weights_[4]) * (Twc[4] - p[0]);
        residuals[5] = T(weights_[5]) * (Twc[5] - p[1]);
        residuals[6] = T(weights_[6]) * (Twc[6] - p[2]);
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d p, const Vector3d last, const Vector3d A, const Vector3d B, const Vector3d C, double *weights)
    {
        return (new ceres::AutoDiffCostFunction<NavsatError, 7, 7>(new NavsatError(p, last, A, B, C, weights)));
    }

private:
    Vector3d heading_;   // heading,
    Vector3d A_, B_, C_; //ground plane(ABC), position
    Vector3d p_;         //position
    Vector3d abc_norm_;
    double weights_[7];
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
