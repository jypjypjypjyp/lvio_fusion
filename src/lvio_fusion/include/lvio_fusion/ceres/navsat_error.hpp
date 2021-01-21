#ifndef lvio_fusion_NAVSAT_ERROR_H
#define lvio_fusion_NAVSAT_ERROR_H

#include "lvio_fusion/ceres/base.hpp"

namespace lvio_fusion
{

class NavsatInitError
{
public:
    NavsatInitError(Vector3d p0, Vector3d p1)
        : x0_(p0.x()), y0_(p0.y()), z0_(p0.z()),
          x1_(p1.x()), y1_(p1.y()), z1_(p1.z())
    {
    }

    template <typename T>
    bool operator()(const T *tf, T *residuals) const
    {
        T p1[3] = {T(x1_), T(y1_), T(z1_)};
        T tf_p1[3];
        ceres::SE3TransformPoint(tf, p1, tf_p1);
        residuals[0] = T(x0_) - tf_p1[0];
        residuals[1] = T(y0_) - tf_p1[1];
        residuals[2] = T(z0_) - tf_p1[2];
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

class NavsatXError
{
public:
    NavsatXError(Vector3d p0, Vector3d p1)
        : x0_(p0.x()), y0_(p0.y()), z0_(p0.z()),
          x1_(p1.x()), y1_(p1.y()), z1_(p1.z())
    {
    }

    template <typename T>
    bool operator()(const T *t, T *residuals) const
    {
        T p1[3] = {T(x1_), T(y1_), T(z1_)};
        T tf_p1[3];
        ceres::Add(t, p1, tf_p1);
        residuals[0] = T(x0_) - tf_p1[0];
        residuals[1] = T(y0_) - tf_p1[1];
        // residuals[2] = T(z0_) - tf_p1[2];
        T ljm[3] = {T(BC.x()), T(BC.y()), T(BC.z())};
        T a[3] = {T(A.x()), T(A.y()), T(A.z())};
        T lp_lpj[3];
        ceres::Minus(tf_p1, a, lp_lpj);
        residuals[2] = ceres::DotProduct(lp_lpj, ljm);
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p0, Vector3d p1)
    {
        return (new ceres::AutoDiffCostFunction<NavsatXError, 3, 3>(new NavsatXError(p0, p1)));
    }
    
    static Vector3d A, BC;

private:
    double x0_, y0_, z0_;
    double x1_, y1_, z1_;
};

class NavsatRError
{
public:
    NavsatRError(Vector3d p0, Vector3d p1)
        : x0_(p0.x()), y0_(p0.y()), z0_(p0.z()),
          x1_(p1.x()), y1_(p1.y()), z1_(p1.z())
    {
    }

    template <typename T>
    bool operator()(const T *q, T *residuals) const
    {
        T p1[3] = {T(x1_), T(y1_), T(z1_)};
        T tf_p1[3];
        ceres::EigenQuaternionRotatePoint(q, p1, tf_p1);
        residuals[0] = T(x0_) - tf_p1[0];
        residuals[1] = T(y0_) - tf_p1[1];
        // residuals[2] = T(z0_) - tf_p1[2];
        T ljm[3] = {T(BC.x()), T(BC.y()), T(BC.z())};
        T a[3] = {T(A.x()), T(A.y()), T(A.z())};
        T lp_lpj[3];
        ceres::Minus(tf_p1, a, lp_lpj);
        residuals[2] = ceres::DotProduct(lp_lpj, ljm);
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p0, Vector3d p1)
    {
        return (new ceres::AutoDiffCostFunction<NavsatRError, 3, 4>(new NavsatRError(p0, p1)));
    }

    static Vector3d A, BC;

private:
    double x0_, y0_, z0_;
    double x1_, y1_, z1_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_ERROR_H
