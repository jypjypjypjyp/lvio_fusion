#ifndef lvio_fusion_POSE_ERROR_H
#define lvio_fusion_POSE_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/common.h"

namespace lvio_fusion
{

class PoseGraphError : public ceres::Error
{
public:
    PoseGraphError(SE3d last_pose, SE3d pose, double weight, double v) : v_(v), Error(weight)
    {
        SE3d relative_i_j = last_pose.inverse() * pose;
        ceres::SE3ToRpyxyz(relative_i_j.data(), rpyxyz_);
    }

    PoseGraphError(SE3d relative_i_j, double weight, double v) : v_(v), Error(weight)
    {
        ceres::SE3ToRpyxyz(relative_i_j.data(), rpyxyz_);
    }

    template <typename T>
    bool operator()(const T *Twc1, const T *Twc2, T *residuals) const
    {
        T Twc1_inverse[7], relative_i_j[7], rpyxyz[6];
        ceres::SE3Inverse(Twc1, Twc1_inverse);
        ceres::SE3Product(Twc1_inverse, Twc2, relative_i_j);
        ceres::SE3ToRpyxyz(relative_i_j, rpyxyz);
        residuals[0] = T(v_ * weight_) * (T(rpyxyz_[0] - rpyxyz[0]));
        residuals[1] = T(v_ * weight_) * (T(rpyxyz_[1] - rpyxyz[1]));
        residuals[2] = T(v_ * weight_) * (T(rpyxyz_[2] - rpyxyz[2]));
        residuals[3] = T(weight_) * (T(rpyxyz_[3] - rpyxyz[3]));
        residuals[4] = T(10 * weight_) * (T(rpyxyz_[4] - rpyxyz[4]));
        residuals[5] = T(10 * weight_) * (T(rpyxyz_[5] - rpyxyz[5]));
        return true;
    }

    static ceres::CostFunction *Create(SE3d last_pose, SE3d pose, double weight = 1, double v = 1)
    {
        return (new ceres::AutoDiffCostFunction<PoseGraphError, 6, 7, 7>(new PoseGraphError(last_pose, pose, weight, v)));
    }

    static ceres::CostFunction *Create(SE3d relative_i_j, double weight = 1, double v = 1)
    {
        return (new ceres::AutoDiffCostFunction<PoseGraphError, 6, 7, 7>(new PoseGraphError(relative_i_j, weight, v)));
    }

private:
    double rpyxyz_[6];
    double v_;
};

class PoseError : public ceres::Error
{
public:
    PoseError(SE3d pose, double weight, double v) : pose_(pose), v_(v), Error(weight) {}

    template <typename T>
    bool operator()(const T *pose, T *residuals) const
    {
        T origin[7], origin_inverse[7], relative[7], rpyxyz[6];
        ceres::Cast(pose_.data(), 7, origin);
        ceres::SE3Inverse(origin, origin_inverse);
        ceres::SE3Product(origin_inverse, pose, relative);
        ceres::SE3ToRpyxyz(relative, rpyxyz);

        residuals[0] = T(v_ * weight_) * rpyxyz[0];
        residuals[1] = T(v_ * weight_) * rpyxyz[1];
        residuals[2] = T(v_ * weight_) * rpyxyz[2];
        residuals[3] = T(weight_) * rpyxyz[3];
        residuals[4] = T(weight_) * rpyxyz[4];
        residuals[5] = T(weight_) * rpyxyz[5];
        return true;
    }

    static ceres::CostFunction *Create(SE3d pose, double weight = 1, double v = 1)
    {
        return (new ceres::AutoDiffCostFunction<PoseError, 6, 7>(new PoseError(pose, weight, v)));
    }

private:
    SE3d pose_;
    double v_;
};

class RError : public ceres::Error
{
public:
    RError(SE3d pose, double weight) : pose_(pose), Error(weight) {}

    template <typename T>
    bool operator()(const T *pose, T *residuals) const
    {
        residuals[0] = T(weight_) * (pose[0] - T(pose_.data()[0]));
        residuals[1] = T(weight_) * (pose[1] - T(pose_.data()[1]));
        residuals[2] = T(weight_) * (pose[2] - T(pose_.data()[2]));
        residuals[3] = T(weight_) * (pose[3] - T(pose_.data()[3]));
        return true;
    }

    static ceres::CostFunction *Create(SE3d pose, double weight = 1)
    {
        return (new ceres::AutoDiffCostFunction<RError, 4, 7>(new RError(pose, weight)));
    }

private:
    SE3d pose_;
};

class TError : public ceres::Error
{
public:
    TError(Vector3d p, double weight) : p_(p), Error(weight) {}

    template <typename T>
    bool operator()(const T *pose, T *residuals) const
    {
        residuals[0] = T(weight_) * (pose[4] - T(p_.data()[0]));
        residuals[1] = T(weight_) * (pose[5] - T(p_.data()[1]));
        residuals[2] = T(weight_) * (pose[6] - T(p_.data()[2]));
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p, double weight = 1)
    {
        return (new ceres::AutoDiffCostFunction<TError, 3, 7>(new TError(p, weight)));
    }

private:
    Vector3d p_;
};

class PoseErrorRPZ : public ceres::Error
{
public:
    PoseErrorRPZ(double *rpyxyz, double weight)
        : Error(weight)
    {
        p_ = rpyxyz[1];
        r_ = rpyxyz[2];
        z_ = rpyxyz[5];
    }

    template <typename T>
    bool operator()(const T *p, const T *r, const T *z, T *residuals) const
    {
        residuals[0] = T(weight_) * (r[0] - T(r_));
        residuals[1] = T(weight_) * (p[0] - T(p_));
        residuals[2] = T(weight_) * (z[0] - T(z_));
        return true;
    }

    static ceres::CostFunction *Create(double *rpyxyz, double weight = 1)
    {
        return (new ceres::AutoDiffCostFunction<PoseErrorRPZ, 3, 1, 1, 1>(new PoseErrorRPZ(rpyxyz, weight)));
    }

private:
    double r_, p_, z_;
};

class PoseErrorYXY : public ceres::Error
{
public:
    PoseErrorYXY(double *rpyxyz, double weight = 1) : Error(weight)
    {
        Y_ = rpyxyz[0];
        x_ = rpyxyz[3];
        y_ = rpyxyz[4];
    }

    template <typename T>
    bool operator()(const T *Y, const T *x, const T *y, T *residuals) const
    {
        residuals[0] = T(weight_) * (Y[0] - T(Y_));
        residuals[1] = T(weight_) * (x[0] - T(x_));
        residuals[2] = T(weight_) * (y[0] - T(y_));
        return true;
    }

    static ceres::CostFunction *Create(double *rpyxyz, double weight = 1)
    {
        return (new ceres::AutoDiffCostFunction<PoseErrorYXY, 3, 1, 1, 1>(new PoseErrorYXY(rpyxyz, weight)));
    }

private:
    double Y_, x_, y_;
};

class RelocateRError
{
public:
    RelocateRError(SE3d relocated, SE3d unrelocated) : relocated_(relocated), unrelocated_(unrelocated) {}

    template <typename T>
    bool operator()(const T *r, T *residuals) const
    {
        T R[7] = {r[0], r[1], r[2], r[3], T(0), T(0), T(0)};
        T unrelocated[7];
        ceres::Cast(unrelocated_.data(), SE3d::num_parameters, unrelocated);
        T R_unrelocated[7];
        ceres::SE3Product(R, unrelocated, R_unrelocated);
        residuals[0] = T(relocated_.data()[0]) - R_unrelocated[0];
        residuals[1] = T(relocated_.data()[1]) - R_unrelocated[1];
        residuals[2] = T(relocated_.data()[2]) - R_unrelocated[2];
        residuals[3] = T(relocated_.data()[3]) - R_unrelocated[3];
        residuals[4] = T(relocated_.data()[4]) - R_unrelocated[4];
        residuals[5] = T(relocated_.data()[5]) - R_unrelocated[5];
        residuals[6] = T(relocated_.data()[6]) - R_unrelocated[6];
        return true;
    }

    static ceres::CostFunction *Create(SE3d relocated, SE3d unrelocated)
    {
        return (new ceres::AutoDiffCostFunction<RelocateRError, 7, 4>(new RelocateRError(relocated, unrelocated)));
    }

private:
    SE3d relocated_, unrelocated_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_POSE_ERROR_H
