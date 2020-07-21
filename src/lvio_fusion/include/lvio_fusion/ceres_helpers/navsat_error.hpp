#ifndef lvio_fusion_NAVSAT_ERROR_H
#define lvio_fusion_NAVSAT_ERROR_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <lvio_fusion/common.h>

namespace lvio_fusion
{

class NavsatError
{
public:
    NavsatError(Vector3d p) : p_(p) {}

    template <typename T>
    bool operator()(const T *pose_, T *residuals_) const
    {
        Eigen::Map<Sophus::SE3<T> const> pose(pose_);
        Eigen::Map<Matrix<T, 3, 1>> residuals(residuals_);
        residuals = pose.inverse().translation() - p_;
        residuals.applyOnTheLeft(sqrt_information.template cast<T>());
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p)
    {
        return (new ceres::AutoDiffCostFunction<NavsatError, 3, 7>(new NavsatError(p)));
    }

    static Matrix3d sqrt_information;

private:
    Vector3d p_;
};

class NavsatInitError
{
public:
    NavsatInitError(Vector3d p0, Vector3d p1)
        : p0_(p0), p1_(p1) {}

    template <typename T>
    bool operator()(const T *tf_, T *residuals_) const
    {
        Eigen::Map<Sophus::SE3<T> const> tf(tf_);
        Eigen::Map<Matrix<T, 3, 1>> residuals(residuals_);
        Matrix<T, 3, 1> p1 = tf * p1_.template cast<T>();
        Matrix<T, 3, 1> p0 = p0_.template cast<T>();
        residuals = p0 - p1;
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p0, Vector3d p1)
    {
        return (new ceres::AutoDiffCostFunction<NavsatInitError, 3, 7>(new NavsatInitError(p0, p1)));
    }

private:
    Vector3d p0_, p1_;
};

// class NavsatPoseError
// {
// public:
//     NavsatPoseError(Vector3d p1, Vector3d p2)
//         : x1_(p1.x()), y1_(p1.y()), z1_(p1.z()),
//           x2_(p2.x()), y2_(p2.y()), z2_(p2.z()) {}

//     template <typename T>
//     bool operator()(const T *R, const T *t, T *residuals) const
//     {
//         T q_inv[4];
//         EigenQuaternionInverse(R, q_inv);
//         T p0[3];
//         ceres::QuaternionRotatePoint(q_inv, t, p0);

//         T x1 = T(x1_), y1 = T(y1_), z1 = T(z1_),
//           x2 = T(x2_), y2 = T(y2_), z2 = T(z2_),
//           x0 = p0[0], y0 = p0[1], z0 = p0[2];
//         T x2_x1 = x2 - x1, y2_y1 = y2 - y1, z2_z1 = z2 - z1;
//         T Ax = x0 - x1, Ay = y0 - y1, Az = z0 - z1;
//         T Bx = x0 - x2, By = y0 - y2, Bz = z0 - z2;
//         T AyBz_AzBy = Ay * Bz - Az * By;
//         T AxBz_AzBx = Ax * Bz - Az * Bx;
//         T AxBy_AyBx = Ax * By - Ay * Bx;
//         residuals[0] = T(100.0) * (AyBz_AzBy * AyBz_AzBy + AxBy_AyBx * AxBy_AyBx + AxBz_AzBx * AxBz_AzBx) / (x2_x1 * x2_x1 + y2_y1 * y2_y1 + z2_z1 * z2_z1);
//         return true;
//     }

//     static ceres::CostFunction *Create(Vector3d p1, Vector3d p2)
//     {
//         return (new ceres::AutoDiffCostFunction<NavsatPoseError, 1, 4, 3>(
//             new NavsatPoseError(p1, p2)));
//     }

// private:
//     double x1_, y1_, z1_;
//     double x2_, y2_, z2_;
// };

// class Navsat1PointError
// {
// public:
//     Navsat1PointError(Vector3d p)
//         : x_(p.x()), y_(p.y()), z_(p.z()) {}

//     template <typename T>
//     bool operator()(const T *p1, T *residuals) const
//     {
//         T x1 = p1[0], y1 = p1[1], z1 = p1[2],
//           x0 = T(x_), y0 = T(y_), z0 = T(z_);
//         T x1_x0 = x1 - x0, y1_y0 = y1 - y0, z1_z0 = z1 - z0;
//         residuals[0] = T(10000.0) * (x1_x0 * x1_x0 + y1_y0 * y1_y0 + z1_z0 * z1_z0);
//         return true;
//     }

//     static ceres::CostFunction *Create(Vector3d p)
//     {
//         return (new ceres::AutoDiffCostFunction<Navsat1PointError, 1, 3>(
//             new Navsat1PointError(p)));
//     }

// private:
//     double x_, y_, z_;
// };

// class NavsatRT1PointError
// {
// public:
//     NavsatRT1PointError(Vector3d p0, Vector3d p1)
//         : x0_(p0.x()), y0_(p0.y()), z0_(p0.z()),
//           x1_(p1.x()), y1_(p1.y()), z1_(p1.z()) {}

//     template <typename T>
//     bool operator()(const T *R, T *residuals) const
//     {
//         T tmp[3], p1[3];
//         // clang-format off
//         tmp[0] = T(x0_); tmp[1] = T(y0_); tmp[2] = T(z0_);
//         p1[0] = T(x1_); p1[1] = T(y1_); p1[2] = T(z1_);
//         // clang-format on
//         T p0[3];
//         ceres::QuaternionRotatePoint(R, tmp, p0);

//         T Ax = p0[0] - p1[0], Ay = p0[1] - p1[1], Az = p0[2] - p1[2];
//         residuals[0] = T(1.0) * (Ax * Ax + Ay * Ay + Az * Az);
//         return true;
//     }

//     static ceres::CostFunction *Create(Vector3d p0, Vector3d p1)
//     {
//         return (new ceres::AutoDiffCostFunction<NavsatRT1PointError, 1, 4>(
//             new NavsatRT1PointError(p0, p1)));
//     }

// private:
//     double x0_, y0_, z0_;
//     double x1_, y1_, z1_;
// };

} // namespace lvio_fusion

#endif // lvio_fusion_NAVSAT_ERROR_H
