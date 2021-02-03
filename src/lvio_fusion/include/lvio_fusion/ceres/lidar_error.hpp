#ifndef lvio_fusion_LIDAR_ERROR_H
#define lvio_fusion_LIDAR_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/common.h"

namespace lvio_fusion
{
class LidarPlaneError
{
public:
    LidarPlaneError(Vector3d p, Vector3d pa, Vector3d pb, Vector3d pc)
        : p_(p), pa_(pa)
    {
        abc_norm_ = (pa_ - pb).cross(pa_ - pc);
        abc_norm_.normalize();
    }

    template <typename T>
    bool operator()(const T *Twc2, T *residual) const
    {
        T cp[3] = {T(p_.x()), T(p_.y()), T(p_.z())};
        T lpj[3] = {T(pa_.x()), T(pa_.y()), T(pa_.z())};
        T ljm[3] = {T(abc_norm_.x()), T(abc_norm_.y()), T(abc_norm_.z())};
        T lp[3], lp_lpj[3];
        ceres::SE3TransformPoint(Twc2, cp, lp);
        ceres::Minus(lp, lpj, lp_lpj);
        residual[0] = ceres::DotProduct(lp_lpj, ljm);
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p, Vector3d pa, Vector3d pb, Vector3d pc)
    {
        return (new ceres::AutoDiffCostFunction<LidarPlaneError, 1, 7>(new LidarPlaneError(p, pa, pb, pc)));
    }

private:
    Vector3d p_, pa_, abc_norm_;
};

inline void se32rpyxyz(const SE3d relatice_i_j, double *rpyxyz)
{
    ceres::EigenQuaternionToRPY(relatice_i_j.data(), rpyxyz);
    rpyxyz[3] = relatice_i_j.data()[4];
    rpyxyz[4] = relatice_i_j.data()[5];
    rpyxyz[5] = relatice_i_j.data()[6];
}

inline SE3d rpyxyz2se3(const double *rpyxyz)
{
    double e_q[4];
    ceres::RPYToEigenQuaternion(rpyxyz, e_q);
    return SE3d(Quaterniond(e_q), Vector3d(rpyxyz[3], rpyxyz[4], rpyxyz[5]));
}

class LidarPlaneErrorRPZ
{
public:
    LidarPlaneErrorRPZ(LidarPlaneError origin_error, SE3d Twc1, double *rpyxyz, double weight)
        : origin_error_(origin_error), Twc1_(Twc1), rpyxyz_(rpyxyz), weight_(weight) {}

    template <typename T>
    bool operator()(const T *pitch, const T *roll, const T *z, T *residual) const
    {
        T Twc1[7], Twc2[7], relative_i_j[7], rpyxyz[6];
        ceres::Cast(rpyxyz_, 6, rpyxyz);
        //NOTE: the real order of rpy is y p r
        rpyxyz[1] = *pitch;
        rpyxyz[2] = *roll;
        rpyxyz[5] = *z;
        ceres::RpyxyzToSE3(rpyxyz, relative_i_j);
        ceres::Cast(Twc1_.data(), SE3d::num_parameters, Twc1);
        ceres::SE3Product(Twc1, relative_i_j, Twc2);
        origin_error_(Twc2, residual);
        residual[0] = T(weight_) * residual[0];
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p, Vector3d pa, Vector3d pb, Vector3d pc, SE3d Twc1, double *rpyxyz, double weight)
    {
        LidarPlaneError origin_error(p, pa, pb, pc);
        return (new ceres::AutoDiffCostFunction<LidarPlaneErrorRPZ, 1, 1, 1, 1>(new LidarPlaneErrorRPZ(origin_error, Twc1, rpyxyz, weight)));
    }

private:
    LidarPlaneError origin_error_;
    SE3d Twc1_;
    double *rpyxyz_;
    double weight_;
};

class LidarPlaneErrorYXY
{
public:
    LidarPlaneErrorYXY(LidarPlaneError origin_error, SE3d Twc1, double *rpyxyz, double weight)
        : origin_error_(origin_error), Twc1_(Twc1), rpyxyz_(rpyxyz), weight_(weight) {}

    template <typename T>
    bool operator()(const T *yaw, const T *x, const T *y, T *residual) const
    {
        T Twc1[7], Twc2[7], relative_i_j[7], rpyxyz[6];
        ceres::Cast(rpyxyz_, 6, rpyxyz);
        //NOTE: the real order of rpy is y p r
        rpyxyz[0] = *yaw;
        rpyxyz[3] = *x;
        rpyxyz[4] = *y;
        ceres::RpyxyzToSE3(rpyxyz, relative_i_j);
        ceres::Cast(Twc1_.data(), SE3d::num_parameters, Twc1);
        ceres::SE3Product(Twc1, relative_i_j, Twc2);
        origin_error_(Twc2, residual);
        residual[0] = T(weight_) * residual[0];
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p, Vector3d pa, Vector3d pb, Vector3d pc, SE3d Twc1, double *rpyxyz, double weight)
    {
        LidarPlaneError origin_error(p, pa, pb, pc);
        return (new ceres::AutoDiffCostFunction<LidarPlaneErrorYXY, 1, 1, 1, 1>(new LidarPlaneErrorYXY(origin_error, Twc1, rpyxyz, weight)));
    }

private:
    LidarPlaneError origin_error_;
    SE3d Twc1_;
    double *rpyxyz_;
    double weight_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_LIDAR_ERROR_H
