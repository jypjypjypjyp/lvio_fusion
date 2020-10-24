#ifndef lvio_fusion_LIDAR_ERROR_H
#define lvio_fusion_LIDAR_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/common.h"
#include "lvio_fusion/lidar/lidar.hpp"

namespace lvio_fusion
{

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
        T Twc1_inverse[7], extrinsic[7], extrinsic_inverse[7], se3[7], se3_[7];
        T lp[3], nu[3], de[3], lp_lpa[3], lp_lpb[3], de_norm;
        ceres::Cast(lidar_->extrinsic.data(), SE3d::num_parameters, extrinsic);
        ceres::SE3Inverse(extrinsic, extrinsic_inverse);
        ceres::SE3Inverse(Tcw1, Twc1_inverse);
        ceres::SE3Product(extrinsic, Tcw2, se3);
        ceres::SE3Product(se3, Twc1_inverse, se3_);
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
        T Twc1_inverse[7], extrinsic[7], extrinsic_inverse[7], se3[7], se3_[7];
        T lp[3], lp_lpj[3], de_norm;
        ceres::Cast(lidar_->extrinsic.data(), SE3d::num_parameters, extrinsic);
        ceres::SE3Inverse(extrinsic, extrinsic_inverse);
        ceres::SE3Inverse(Tcw1, Twc1_inverse);
        ceres::SE3Product(extrinsic, Tcw2, se3);
        ceres::SE3Product(se3, Twc1_inverse, se3_);
        ceres::SE3Product(se3_, extrinsic_inverse, se3);
        ceres::SE3TransformPoint(se3, cp, lp);
        ceres::Minus(lp, lpj, lp_lpj);
        residual[0] = ceres::DotProduct(lp_lpj, ljm);
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d p, const Vector3d pa, const Vector3d pb, const Vector3d pc, Lidar::Ptr lidar)
    {
        return (new ceres::AutoDiffCostFunction<LidarPlaneError, 1, 7, 7>(
            new LidarPlaneError(p, pa, pb, pc, lidar)));
    }

private:
    Vector3d p_, pa_, pb_, pc_;
    Vector3d abc_norm_;
    Lidar::Ptr lidar_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_LIDAR_ERROR_H
