#ifndef lvio_fusion_NAVSAT_ERROR_H
#define lvio_fusion_NAVSAT_ERROR_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <lvio_fusion/common.h>

namespace lvio_fusion
{

template <typename T> inline
void EigenQuaternionInverse(const T q[4], T q_inverse[4])
{
	q_inverse[3] = q[0];
	q_inverse[0] = -q[1];
	q_inverse[1] = -q[2];
	q_inverse[2] = -q[3];
};

class Navsat1PointError
{
public:
    Navsat1PointError(Vector3d p, double *R)
        : x_(p.x()), y_(p.y()), z_(p.z()), R_(R) {}

    template <typename T>
    bool operator()(const T *t, T *residuals) const
    {
        T q_inv[4];
        T R[4];
        R[0] = T(R_[0]);
        R[1] = T(R_[1]);
        R[2] = T(R_[2]);
        R[3] = T(R_[3]);
        EigenQuaternionInverse(R, q_inv);
        T tmp[3];
        tmp[0] = -t[0];
        tmp[1] = -t[1];
        tmp[2] = -t[2];
        T p0[3];
        ceres::QuaternionRotatePoint(q_inv, tmp, p0);
        T x1 = p0[0], y1 = p0[1], z1 = p0[2],
          x0 = T(x_), y0 = T(y_), z0 = T(z_);
        T x1_x0 = x1 - x0, y1_y0 = y1 - y0, z1_z0 = z1 - z0;
        residuals[0] = T(10000) * x1_x0;
        residuals[1] = T(10000) *y1_y0;
        residuals[2] = T(10000) *z1_z0;
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p, double *R)
    {
        return (new ceres::AutoDiffCostFunction<Navsat1PointError, 3, 3>(
            new Navsat1PointError(p, R)));
    }

private:
    double x_, y_, z_;
    double *R_;
};

class NavsatPoseError
{
public:
    NavsatPoseError(Vector3d p1, Vector3d p2)
        : x1_(p1.x()), y1_(p1.y()), z1_(p1.z()),
          x2_(p2.x()), y2_(p2.y()), z2_(p2.z()) {}

    template <typename T>
    bool operator()(const T *R, const T *t, T *residuals) const
    {
        T q_inv[4];
        EigenQuaternionInverse(R, q_inv);
        T p0[3];
        ceres::QuaternionRotatePoint(q_inv, t, p0);
        
        T x1 = T(x1_), y1 = T(y1_), z1 = T(z1_),
          x2 = T(x2_), y2 = T(y2_), z2 = T(z2_),
          x0 = p0[0], y0 = p0[1], z0 = p0[2];
        T x2_x1 = x2 - x1, y2_y1 = y2 - y1, z2_z1 = z2 - z1;
        T Ax = x0 - x1, Ay = y0 - y1, Az = z0 - z1;
        T Bx = x0 - x2, By = y0 - y2, Bz = z0 - z2;
        T AyBz_AzBy = Ay * Bz - Az * By;
        T AxBz_AzBx = Ax * Bz - Az * Bx;
        T AxBy_AyBx = Ax * By - Ay * Bx;
        residuals[0] = T(100.0) * (AyBz_AzBy * AyBz_AzBy + AxBy_AyBx * AxBy_AyBx + AxBz_AzBx * AxBz_AzBx) / (x2_x1 * x2_x1 + y2_y1 * y2_y1 + z2_z1 * z2_z1);
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p1, Vector3d p2)
    {
        return (new ceres::AutoDiffCostFunction<NavsatPoseError, 1, 4, 3>(
            new NavsatPoseError(p1, p2)));
    }

private:
    double x1_, y1_, z1_;
    double x2_, y2_, z2_;
};

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

class NavsatRError
{
public:
    NavsatRError(Vector3d p0, Vector3d p1, Vector3d p2)
        : x0_(p0.x()), y0_(p0.y()), z0_(p0.z()),
          x1_(p1.x()), y1_(p1.y()), z1_(p1.z()),
          x2_(p2.x()), y2_(p2.y()), z2_(p2.z()) {}

template <typename T>
    bool operator()(const T *R, const T* t, T *residuals) const
    {
        T p0[3], tmp1[3], tmp2[3];
        // clang-format off
        p0[0] = T(x0_); p0[1] = T(y0_); p0[2] = T(z0_);
        tmp1[0] = T(x1_); tmp1[1] = T(y1_); tmp1[2] = T(z1_);
        tmp2[0] = T(x2_); tmp2[1] = T(y2_); tmp2[2] = T(z2_);
        // clang-format on
        T p1[3], p2[3];
        ceres::QuaternionRotatePoint(R, tmp1, p1);
        ceres::QuaternionRotatePoint(R, tmp2, p2);
        // p0[0] -= t[0];
        // p0[1] -= t[1];
        p0[2] -= t[2];

        T x2_x1 = p2[0] - p1[0], y2_y1 = p2[1] - p1[1], z2_z1 = p2[2] - p1[2];
        T Ax = p0[0] - p1[0], Ay = p0[1] - p1[1], Az = p0[2] - p1[2];
        T Bx = p0[0] - p2[0], By = p0[1] - p2[1], Bz = p0[2] - p2[2];
        T AyBz_AzBy = Ay * Bz - Az * By;
        T AxBz_AzBx = Ax * Bz - Az * Bx;
        T AxBy_AyBx = Ax * By - Ay * Bx;
        residuals[0] = (AyBz_AzBy * AyBz_AzBy + AxBy_AyBx * AxBy_AyBx + AxBz_AzBx * AxBz_AzBx) / (x2_x1 * x2_x1 + y2_y1 * y2_y1 + z2_z1 * z2_z1);
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p0, Vector3d p1, Vector3d p2)
    {
        return (new ceres::AutoDiffCostFunction<NavsatRError, 1, 4, 3>(
            new NavsatRError(p0, p1, p2)));
    }

private:
    double x0_, y0_, z0_;
    double x1_, y1_, z1_;
    double x2_, y2_, z2_;
};

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
