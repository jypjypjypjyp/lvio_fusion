#ifndef lvio_fusion_LOOP_ERROR_H
#define lvio_fusion_LOOP_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/ceres/lidar_error.hpp"

namespace lvio_fusion
{

class PoseGraphError
{
public:
    PoseGraphError(SE3d last_frame, SE3d frame, double *weights)
    {
        se32rpyxyz(frame * last_frame.inverse(), rpyxyz_);
        weights_[0] = weights[0];
        weights_[1] = weights[1];
        weights_[2] = weights[2];
        weights_[3] = weights[3];
        weights_[4] = weights[4];
        weights_[5] = weights[5];
    }

    template <typename T>
    bool operator()(const T *Twc1, const T *Twc2, T *residuals) const
    {
        T relative_i_j[7], Twc1_inverse[7], rpyxyz[6];
        ceres::SE3Inverse(Twc1, Twc1_inverse);
        ceres::SE3Product(Twc2, Twc1_inverse, relative_i_j);
        ceres::SE3ToRpyxyz(relative_i_j, rpyxyz);
        residuals[0] = T(weights_[0]) * (rpyxyz[0] - rpyxyz_[0]);
        residuals[1] = T(weights_[1]) * (rpyxyz[1] - rpyxyz_[1]);
        residuals[2] = T(weights_[2]) * (rpyxyz[2] - rpyxyz_[2]);
        residuals[3] = T(weights_[3]) * (rpyxyz[3] - rpyxyz_[3]);
        residuals[4] = T(weights_[4]) * (rpyxyz[4] - rpyxyz_[4]);
        residuals[5] = T(weights_[5]) * (rpyxyz[5] - rpyxyz_[5]);
        return true;
    }

    static ceres::CostFunction *Create(SE3d last_frame, SE3d frame, double *weights)
    {
        return (new ceres::AutoDiffCostFunction<PoseGraphError, 6, 7, 7>(
            new PoseGraphError(last_frame, frame, weights)));
    }

private:
    double rpyxyz_[6];
    double weights_[6];
};

class PoseErrorRPZ
{
public:
    PoseErrorRPZ(const double *rpyxyz, double *weights)
    {
        r_ = rpyxyz[1];
        p_ = rpyxyz[2];
        z_ = rpyxyz[5];
        weights_[0] = weights[1];
        weights_[1] = weights[2];
        weights_[2] = weights[5];
    }

    template <typename T>
    bool operator()(const T *r, const T *p, const T *z, T *residuals) const
    {
        residuals[0] = T(weights_[0]) * (r[0] - T(r_));
        residuals[1] = T(weights_[1]) * (p[0] - T(p_));
        residuals[2] = T(weights_[2]) * (z[0] - T(z_));
        return true;
    }

    static ceres::CostFunction *Create(const double *rpyxyz, double *weights)
    {
        return (new ceres::AutoDiffCostFunction<PoseErrorRPZ, 3, 1, 1, 1>(
            new PoseErrorRPZ(rpyxyz, weights)));
    }

private:
    double r_, p_, z_;
    double weights_[3];
};

class PoseErrorYXY
{
public:
    PoseErrorYXY(const double *rpyxyz, double *weights)
    {
        Y_ = rpyxyz[0];
        x_ = rpyxyz[3];
        y_ = rpyxyz[4];
        weights_[0] = weights[0];
        weights_[1] = weights[3];
        weights_[2] = weights[4];
    }

    template <typename T>
    bool operator()(const T *Y, const T *x, const T *y, T *residuals) const
    {
        residuals[0] = T(weights_[0]) * (Y[0] - T(Y_));
        residuals[1] = T(weights_[1]) * (x[0] - T(x_));
        residuals[2] = T(weights_[2]) * (y[0] - T(y_));
        return true;
    }

    static ceres::CostFunction *Create(const double *rpyxyz, double *weights)
    {
        return (new ceres::AutoDiffCostFunction<PoseErrorYXY, 3, 1, 1, 1>(
            new PoseErrorYXY(rpyxyz, weights)));
    }

private:
    double Y_, x_, y_;
    double weights_[3];
};

} // namespace lvio_fusion

#endif // lvio_fusion_LOOP_ERROR_H
