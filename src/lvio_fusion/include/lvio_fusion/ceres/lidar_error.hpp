#ifndef lvio_fusion_LIDAR_ERROR_H
#define lvio_fusion_LIDAR_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/common.h"
#include "lvio_fusion/lidar/lidar.hpp"

namespace lvio_fusion
{

// class LidarEdgeError
// {
// public:
//     LidarEdgeError(Vector3d p, Vector3d pa, Vector3d pb, Lidar::Ptr lidar)
//         : p_(p), pa_(pa), pb_(pb), lidar_(lidar) {}

//     template <typename T>
//     bool operator()(const T *Tcw2, T *residual) const
//     {
//         T cp[3] = {T(p_.x()), T(p_.y()), T(p_.z())};
//         T lpa[3] = {T(pa_.x()), T(pa_.y()), T(pa_.z())};
//         T lpb[3] = {T(pb_.x()), T(pb_.y()), T(pb_.z())};
//         T Twc2_inverse[7], extrinsic[7], extrinsic_inverse[7], se3[7];
//         T lp[3], nu[3], de[3], lp_lpa[3], lp_lpb[3], de_norm;
//         ceres::Cast(lidar_->extrinsic.data(), SE3d::num_parameters, extrinsic);
//         ceres::SE3Inverse(extrinsic, extrinsic_inverse);
//         ceres::SE3Inverse(Tcw2, Twc2_inverse);
//         ceres::SE3Product(Twc2_inverse, extrinsic_inverse, se3);
//         ceres::SE3TransformPoint(se3, cp, lp);
//         ceres::Minus(lp, lpa, lp_lpa);
//         ceres::Minus(lp, lpb, lp_lpb);
//         ceres::CrossProduct(lp_lpa, lp_lpb, nu);
//         ceres::Minus(lpa, lpb, de);
//         ceres::Norm(de, &de_norm);
//         residual[0] = nu[0] / de_norm;
//         residual[1] = nu[1] / de_norm;
//         residual[2] = nu[2] / de_norm;
//         return true;
//     }

//     static ceres::CostFunction *Create(const Vector3d p, const Vector3d pa, const Vector3d pb, Lidar::Ptr lidar)
//     {
//         return (new ceres::AutoDiffCostFunction<LidarEdgeError, 3, 7>(new LidarEdgeError(p, pa, pb, lidar)));
//     }

// private:
//     Vector3d p_, pa_, pb_;
//     Lidar::Ptr lidar_;
// };

class LidarPlaneError
{
public:
    LidarPlaneError(Vector3d p, Vector3d pa, Vector3d pb, Vector3d pc, Lidar::Ptr lidar)
        : p_(p), pa_(pa), pb_(pb), pc_(pc), lidar_(lidar)
    {
        abc_norm_ = (pa_ - pb_).cross(pa_ - pc_);
        abc_norm_.normalize();
    }

    template <typename T>
    bool operator()(const T *Tcw2, T *residual) const
    {
        T cp[3] = {T(p_.x()), T(p_.y()), T(p_.z())};
        T lpj[3] = {T(pa_.x()), T(pa_.y()), T(pa_.z())};
        T ljm[3] = {T(abc_norm_.x()), T(abc_norm_.y()), T(abc_norm_.z())};
        T Twc2_inverse[7], extrinsic[7], extrinsic_inverse[7], se3[7];
        T lp[3], lp_lpj[3];
        ceres::Cast(lidar_->extrinsic.data(), SE3d::num_parameters, extrinsic);
        ceres::SE3Inverse(extrinsic, extrinsic_inverse);
        ceres::SE3Inverse(Tcw2, Twc2_inverse);
        ceres::SE3Product(Twc2_inverse, extrinsic_inverse, se3);
        ceres::SE3TransformPoint(se3, cp, lp);
        ceres::Minus(lp, lpj, lp_lpj);
        residual[0] = ceres::DotProduct(lp_lpj, ljm);
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d p, const Vector3d pa, const Vector3d pb, const Vector3d pc, Lidar::Ptr lidar)
    {
        return (new ceres::AutoDiffCostFunction<LidarPlaneError, 1, 7>(new LidarPlaneError(p, pa, pb, pc, lidar)));
    }

private:
    Vector3d p_, pa_, pb_, pc_;
    Vector3d abc_norm_;
    Lidar::Ptr lidar_;
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
    LidarPlaneErrorRPZ(LidarPlaneError origin_error, SE3d Tcw1, double *rpyxyz, double* weights)
        : origin_error_(origin_error), Tcw1_(Tcw1), rpyxyz_(rpyxyz), weights_(weights) {}

    template <typename T>
    bool operator()(const T *pitch, const T *roll, const T *z, T *residual) const
    {
        T Tcw1[7], Tcw2[7], relative_i_j[7], relative_j_i[7], rpyxyz[6];
        ceres::Cast(rpyxyz_, 6, rpyxyz);
        //NOTE: the real order of rpy is y p r
        rpyxyz[1] = *pitch;
        rpyxyz[2] = *roll;
        rpyxyz[5] = *z;
        ceres::RpyxyzToSE3(rpyxyz, relative_i_j);
        ceres::SE3Inverse(relative_i_j, relative_j_i);
        ceres::Cast(Tcw1_.data(), SE3d::num_parameters, Tcw1);
        ceres::SE3Product(relative_j_i, Tcw1, Tcw2);
        origin_error_(Tcw2, residual);
        residual[0] = T(weights_[0]) * residual[0];
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d p, const Vector3d pa, const Vector3d pb, const Vector3d pc, const Lidar::Ptr lidar, const SE3d Tcw1, double *rpyxyz, double* weights)
    {
        LidarPlaneError origin_error(p, pa, pb, pc, lidar);
        return (new ceres::AutoDiffCostFunction<LidarPlaneErrorRPZ, 1, 1, 1, 1>(new LidarPlaneErrorRPZ(origin_error, Tcw1, rpyxyz, weights)));
    }

private:
    LidarPlaneError origin_error_;
    SE3d Tcw1_;
    double *rpyxyz_;
    const double *weights_;
};

class LidarPlaneErrorYXY
{
public:
    LidarPlaneErrorYXY(LidarPlaneError origin_error, SE3d Tcw1, double *rpyxyz, double* weights)
        : origin_error_(origin_error), Tcw1_(Tcw1), rpyxyz_(rpyxyz), weights_(weights) {}

    template <typename T>
    bool operator()(const T *yaw, const T *x, const T *y, T *residual) const
    {
        T Tcw1[7], Tcw2[7], relative_i_j[7], relative_j_i[7], rpyxyz[6];
        ceres::Cast(rpyxyz_, 6, rpyxyz);
        //NOTE: the real order of rpy is y p r
        rpyxyz[0] = *yaw;
        rpyxyz[3] = *x;
        rpyxyz[4] = *y;
        ceres::RpyxyzToSE3(rpyxyz, relative_i_j);
        ceres::SE3Inverse(relative_i_j, relative_j_i);
        ceres::Cast(Tcw1_.data(), SE3d::num_parameters, Tcw1);
        ceres::SE3Product(relative_j_i, Tcw1, Tcw2);
        origin_error_(Tcw2, residual);
        residual[0] = T(weights_[0]) * residual[0];
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d p, const Vector3d pa, const Vector3d pb, const Vector3d pc, const Lidar::Ptr lidar, const SE3d Tcw1, double *rpyxyz, double* weights)
    {
        LidarPlaneError origin_error(p, pa, pb, pc, lidar);
        return (new ceres::AutoDiffCostFunction<LidarPlaneErrorYXY, 1, 1, 1, 1>(new LidarPlaneErrorYXY(origin_error, Tcw1, rpyxyz, weights)));
    }

private:
    LidarPlaneError origin_error_;
    SE3d Tcw1_;
    double *rpyxyz_;
    const double *weights_;
};

// class LidarEdgeErrorYXY
// {
// public:
//     LidarEdgeErrorYXY(LidarEdgeError origin_error, SE3d Tcw1, double *rpyxyz)
//         : origin_error_(origin_error), Tcw1_(Tcw1), rpyxyz_(rpyxyz) {}

//     template <typename T>
//     bool operator()(const T *yaw, const T *x, const T *y, T *residual) const
//     {
//         T Tcw1[7], Tcw2[7], relative_i_j[7], relative_j_i[7], rpyxyz[6];
//         ceres::Cast(rpyxyz_, 6, rpyxyz);
//         //NOTE: the real order of rpy is y p r
//         rpyxyz[0] = *yaw;
//         rpyxyz[3] = *x;
//         rpyxyz[4] = *y;
//         ceres::RpyxyzToSE3(rpyxyz, relative_i_j);
//         ceres::SE3Inverse(relative_i_j, relative_j_i);
//         ceres::Cast(Tcw1_.data(), SE3d::num_parameters, Tcw1);
//         ceres::SE3Product(relative_j_i, Tcw1, Tcw2);
//         origin_error_(Tcw2, residual);
//         return true;
//     }

//     static ceres::CostFunction *Create(const Vector3d p, const Vector3d pa, const Vector3d pb, const Lidar::Ptr lidar, const SE3d Tcw1, double *rpyxyz)
//     {
//         LidarEdgeError origin_error(p, pa, pb, lidar);
//         return (new ceres::AutoDiffCostFunction<LidarEdgeErrorYXY, 3, 1, 1, 1>(new LidarEdgeErrorYXY(origin_error, Tcw1, rpyxyz)));
//     }

// private:
//     LidarEdgeError origin_error_;
//     SE3d Tcw1_;
//     double *rpyxyz_;
// }

} // namespace lvio_fusion

#endif // lvio_fusion_LIDAR_ERROR_H
