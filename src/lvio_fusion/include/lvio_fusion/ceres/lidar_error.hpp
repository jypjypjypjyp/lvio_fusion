#ifndef lvio_fusion_LIDAR_ERROR_H
#define lvio_fusion_LIDAR_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/common.h"
#include "lvio_fusion/lidar/lidar.hpp"

namespace lvio_fusion
{

//TODO
// class LidarEdgeRError
// {
// public:
//     LidarEdgeRError() = default;
//     LidarEdgeRError(Vector3d p, Vector3d pa, Vector3d pb, Vector3d last_kf_t, Vector3d kf_t, Lidar::Ptr lidar)
//         : p_(p), pa_(pa), pb_(pb), lidar_(lidar) {}

//     template <typename T>
//     bool operator()(const T *q1, const T *q2, T *residual) const
//     {
//         T cp[3] = {T(p_.x()), T(p_.y()), T(p_.z())};
//         T lpa[3] = {T(pa_.x()), T(pa_.y()), T(pa_.z())};
//         T lpb[3] = {T(pb_.x()), T(pb_.y()), T(pb_.z())};
//         T Twc2_inverse[7], extrinsic[7], extrinsic_inverse[7], se3[7], se3_[7];
//         T lp[3], nu[3], de[3], lp_lpa[3], lp_lpb[3], de_norm;
//         ceres::Cast(lidar_->extrinsic.data(), SE3d::num_parameters, extrinsic);
//         ceres::SE3Inverse(extrinsic, extrinsic_inverse);
//         ceres::SE3Inverse(Tcw2, Twc2_inverse);
//         ceres::SE3Product(extrinsic, Tcw1, se3);
//         ceres::SE3Product(se3, Twc2_inverse, se3_);
//         ceres::SE3Product(se3_, extrinsic_inverse, se3);
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

//     static ceres::CostFunction *Create(const Vector3d p, const Vector3d pa, const Vector3d pb,const Vector3d last_kf_t, const Vector3d kf_t, Lidar::Ptr lidar)
//     {
//         return (new ceres::AutoDiffCostFunction<LidarEdgeRError, 3, 4, 4>(new LidarEdgeRError(p, pa, pb,last_kf_t, kf_t, lidar)));
//     }

// private:
//     Vector3d p_, pa_, pb_;
//     Vector3d last_kf_t_, kf_t_;
//     Lidar::Ptr lidar_;
// };
    

class LidarPlaneRError
{
public:
    LidarPlaneRError() = default;
    LidarPlaneRError(Vector3d p, Vector3d pa, Vector3d pb, Vector3d pc, Vector3d last_kf_t, Vector3d kf_t, Lidar::Ptr lidar)
        : p_(p), pa_(pa), pb_(pb), pc_(pc), last_kf_t_(last_kf_t), kf_t_(kf_t), lidar_(lidar)
    {
        abc_norm_ = (pa_ - pb_).cross(pa_ - pc_);
        abc_norm_.normalize();
    }

    template <typename T>
    bool operator()(const T *q1, const T *q2, T *residual) const
    {
        T cp[3] = {T(p_.x()), T(p_.y()), T(p_.z())};
        T lpj[3] = {T(pa_.x()), T(pa_.y()), T(pa_.z())};
        T ljm[3] = {T(abc_norm_.x()), T(abc_norm_.y()), T(abc_norm_.z())};
        T Tcw1[7] = {q1[0], q1[0], q1[0], q1[0], T(last_kf_t_.x()), T(last_kf_t_.y()), T(last_kf_t_.z())};
        T Tcw2[7] = {q2[0], q2[0], q2[0], q2[0], T(kf_t_.x()), T(kf_t_.y()), T(kf_t_.z())};
        T Tcw2_inverse[7], extrinsic[7], extrinsic_inverse[7], se3[7], se3_[7];
        T lp[3], lp_lpj[3];
        ceres::Cast(lidar_->extrinsic.data(), SE3d::num_parameters, extrinsic);
        ceres::SE3Inverse(extrinsic, extrinsic_inverse);
        ceres::SE3Inverse(Tcw2, Tcw2_inverse);
        ceres::SE3Product(extrinsic, Tcw1, se3);
        ceres::SE3Product(se3, Tcw2_inverse, se3_);
        ceres::SE3Product(se3_, extrinsic_inverse, se3);
        ceres::SE3TransformPoint(se3, cp, lp);
        ceres::Minus(lp, lpj, lp_lpj);
        residual[0] = ceres::DotProduct(lp_lpj, ljm);
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d p, const Vector3d pa, const Vector3d pb, const Vector3d pc, const Vector3d last_kf_t, const Vector3d kf_t, Lidar::Ptr lidar)
    {
        return (new ceres::AutoDiffCostFunction<LidarPlaneRError, 1, 4, 4>(new LidarPlaneRError(p, pa, pb, pc, last_kf_t, kf_t, lidar)));
    }

private:
    Vector3d p_, pa_, pb_, pc_;
    Vector3d abc_norm_;
    Vector3d last_kf_t_, kf_t_;
    Lidar::Ptr lidar_;
};

class LidarEdgeError
{
public:
    LidarEdgeError() = default;
    LidarEdgeError(Vector3d p, Vector3d pa, Vector3d pb, Lidar::Ptr lidar)
        : p_(p), pa_(pa), pb_(pb), lidar_(lidar) {}

    template <typename T>
    bool operator()(const T *Tcw1, const T *Tcw2, T *residual) const
    {
        T cp[3] = {T(p_.x()), T(p_.y()), T(p_.z())};
        T lpa[3] = {T(pa_.x()), T(pa_.y()), T(pa_.z())};
        T lpb[3] = {T(pb_.x()), T(pb_.y()), T(pb_.z())};
        T Twc2_inverse[7], extrinsic[7], extrinsic_inverse[7], se3[7], se3_[7];
        T lp[3], nu[3], de[3], lp_lpa[3], lp_lpb[3], de_norm;
        ceres::Cast(lidar_->extrinsic.data(), SE3d::num_parameters, extrinsic);
        ceres::SE3Inverse(extrinsic, extrinsic_inverse);
        ceres::SE3Inverse(Tcw2, Twc2_inverse);
        ceres::SE3Product(extrinsic, Tcw1, se3);
        ceres::SE3Product(se3, Twc2_inverse, se3_);
        ceres::SE3Product(se3_, extrinsic_inverse, se3);
        ceres::SE3TransformPoint(se3, cp, lp);
        ceres::Minus(lp, lpa, lp_lpa);
        ceres::Minus(lp, lpb, lp_lpb);
        ceres::CrossProduct(lp_lpa, lp_lpb, nu);
        ceres::Minus(lpa, lpb, de);
        ceres::Norm(de, &de_norm);
        residual[0] = nu[0] / de_norm;
        residual[1] = nu[1] / de_norm;
        residual[2] = nu[2] / de_norm;
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d p, const Vector3d pa, const Vector3d pb, Lidar::Ptr lidar)
    {
        return (new ceres::AutoDiffCostFunction<LidarEdgeError, 3, 7, 7>(new LidarEdgeError(p, pa, pb, lidar)));
    }

private:
    Vector3d p_, pa_, pb_;
    Lidar::Ptr lidar_;
};

class LidarPlaneError
{
public:
    LidarPlaneError() = default;
    LidarPlaneError(Vector3d p, Vector3d pa, Vector3d pb, Vector3d pc, Lidar::Ptr lidar)
        : p_(p), pa_(pa), pb_(pb), pc_(pc), lidar_(lidar)
    {
        abc_norm_ = (pa_ - pb_).cross(pa_ - pc_);
        abc_norm_.normalize();
    }

    template <typename T>
    bool operator()(const T *Tcw1, const T *Tcw2, T *residual) const
    {
        T cp[3] = {T(p_.x()), T(p_.y()), T(p_.z())};
        T lpj[3] = {T(pa_.x()), T(pa_.y()), T(pa_.z())};
        T ljm[3] = {T(abc_norm_.x()), T(abc_norm_.y()), T(abc_norm_.z())};
        T Twc2_inverse[7], extrinsic[7], extrinsic_inverse[7], se3[7], se3_[7];
        T lp[3], lp_lpj[3];
        ceres::Cast(lidar_->extrinsic.data(), SE3d::num_parameters, extrinsic);
        ceres::SE3Inverse(extrinsic, extrinsic_inverse);
        ceres::SE3Inverse(Tcw2, Twc2_inverse);
        ceres::SE3Product(extrinsic, Tcw1, se3);
        ceres::SE3Product(se3, Twc2_inverse, se3_);
        ceres::SE3Product(se3_, extrinsic_inverse, se3);
        ceres::SE3TransformPoint(se3, cp, lp);
        ceres::Minus(lp, lpj, lp_lpj);
        residual[0] = ceres::DotProduct(lp_lpj, ljm);
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d p, const Vector3d pa, const Vector3d pb, const Vector3d pc, Lidar::Ptr lidar)
    {
        return (new ceres::AutoDiffCostFunction<LidarPlaneError, 1, 7, 7>(new LidarPlaneError(p, pa, pb, pc, lidar)));
    }

private:
    Vector3d p_, pa_, pb_, pc_;
    Vector3d abc_norm_;
    Lidar::Ptr lidar_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_LIDAR_ERROR_H
