#ifndef lvio_fusion_NAVSAT_ERROR_H
#define lvio_fusion_NAVSAT_ERROR_H

#include "lvio_fusion/ceres/base.hpp"

namespace lvio_fusion
{

// class NavsatInitRError
// {
// public:
//     NavsatInitRError(Vector3d p0, Vector3d p1)
//         : x0_(p0.x()), y0_(p0.y()), z0_(p0.z()),
//           x1_(p1.x()), y1_(p1.y()), z1_(p1.z())
//     {
//     }

//     template <typename T>
//     bool operator()(const T *yaw, T *residuals) const
//     {
//         T tf[7];
//         T rpyxyz[6] = {yaw[0], T(0), T(0), T(0), T(0), T(0)};
//         ceres::RpyxyzToSE3(rpyxyz, tf);
//         T p1[3] = {T(x1_), T(y1_), T(z1_)};
//         T tf_p1[3];
//         ceres::SE3TransformPoint(tf, p1, tf_p1);
//         residuals[0] = T(x0_) - tf_p1[0];
//         residuals[1] = T(y0_) - tf_p1[1];
//         residuals[2] = T(z0_) - tf_p1[2];
//         return true;
//     }

//     static ceres::CostFunction *Create(Vector3d p0, Vector3d p1)
//     {
//         return (new ceres::AutoDiffCostFunction<NavsatInitRError, 3, 1>(new NavsatInitRError(p0, p1)));
//     }

// private:
//     double x0_, y0_, z0_;
//     double x1_, y1_, z1_;
// };

class NavsatInitError
{
public:
    NavsatInitError(Vector3d p0, Vector3d p1)
        : x0_(p0.x()), y0_(p0.y()), z0_(p0.z()),
          x1_(p1.x()), y1_(p1.y()), z1_(p1.z())
    {
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
        residuals[0] = T(x0_) - tf_p1[0];
        residuals[1] = T(y0_) - tf_p1[1];
        residuals[2] = T(z0_) - tf_p1[2];
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p0, Vector3d p1)
    {
        return (new ceres::AutoDiffCostFunction<NavsatInitError, 3, 1, 1, 1>(new NavsatInitError(p0, p1)));
    }

private:
    double x0_, y0_, z0_;
    double x1_, y1_, z1_;
};

class NavsatRError
{
public:
    NavsatRError(SE3d origin, SE3d pose) : pose_(pose), origin_(origin)
    {
        pi_o_ = pose_.inverse() * origin_;
    }

    template <typename T>
    bool operator()(const T *yaw, const T *pitch, const T *roll, T *residual) const
    {
        T pose[4], new_pose[4], relative_pose[4], pi_o[4], pr[4];
        T rpy[3] = {yaw[0], pitch[0], roll[0]};
        ceres::RPYToEigenQuaternion(rpy, relative_pose);
        ceres::Cast(pose_.data(), SO3d::num_parameters, pose);
        ceres::Cast(pi_o_.data(), SO3d::num_parameters, pi_o);
        ceres::EigenQuaternionProduct(pose, relative_pose, pr);
        ceres::EigenQuaternionProduct(pr, pi_o, new_pose);
        T y[3] = {T(0), T(1), T(0)};
        T tf_y[3];
        ceres::EigenQuaternionRotatePoint(new_pose, y, tf_y);
        residual[0] = T(100) * tf_y[2] * tf_y[2];
        return true;
    }

    static ceres::CostFunction *Create(SE3d origin, SE3d pose)
    {
        return (new ceres::AutoDiffCostFunction<NavsatRError, 1, 1, 1, 1>(new NavsatRError(origin, pose)));
    }

private:
    SE3d pose_, origin_, pi_o_;
};

class NavsatR2Error
{
public:
    NavsatR2Error(SE3d pose) : pose_(pose)
    {
    }

    template <typename T>
    bool operator()(const T *yaw, const T *pitch, const T *roll, T *residual) const
    {
        T pose[4], relative_pose[4], pr[4];
        T rpy[3] = {yaw[0], pitch[0], roll[0]};
        ceres::RPYToEigenQuaternion(rpy, relative_pose);
        ceres::Cast(pose_.data(), SO3d::num_parameters, pose);
        ceres::EigenQuaternionProduct(pose, relative_pose, pr);
        T z[3] = {T(0), T(0), T(1)};
        T tf_z[3];
        ceres::EigenQuaternionRotatePoint(pr, z, tf_z);
        if (tf_z[2] > T(0))
        {
            residual[0] = T(0);
        }
        else
        {
            residual[0] = T(1000) * tf_z[2];
        }
        return true;
    }

    static ceres::CostFunction *Create(SE3d pose)
    {
        return (new ceres::AutoDiffCostFunction<NavsatR2Error, 1, 1, 1, 1>(new NavsatR2Error(pose)));
    }

private:
    SE3d pose_;
};

class NavsatRXError
{
public:
    NavsatRXError(Vector3d p0, Vector3d p1, SE3d pose)
        : x0_(p0.x()), y0_(p0.y()), z0_(p0.z()),
          x1_(p1.x()), y1_(p1.y()), z1_(p1.z()),
          pose_(pose)
    {
    }

    template <typename T>
    bool operator()(const T *yaw, const T *pitch, const T *roll, const T *x, T *residuals) const
    {
        T pose[7], tf[7], relative_pose[7];
        T rpyxyz[6] = {yaw[0], pitch[0], roll[0], x[0], T(0), T(0)};
        ceres::RpyxyzToSE3(rpyxyz, relative_pose);
        ceres::Cast(pose_.data(), SE3d::num_parameters, pose);
        ceres::SE3Product(pose, relative_pose, tf);
        T p1[3] = {T(x1_), T(y1_), T(z1_)};
        T tf_p1[3];
        ceres::SE3TransformPoint(tf, p1, tf_p1);
        residuals[0] = T(x0_) - tf_p1[0];
        residuals[1] = T(y0_) - tf_p1[1];
        residuals[2] = T(z0_) - tf_p1[2];
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p0, Vector3d p1, SE3d pose)
    {
        return (new ceres::AutoDiffCostFunction<NavsatRXError, 3, 1, 1, 1, 1>(new NavsatRXError(p0, p1, pose)));
    }

private:
    double x0_, y0_, z0_;
    double x1_, y1_, z1_;
    SE3d pose_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_ERROR_H
