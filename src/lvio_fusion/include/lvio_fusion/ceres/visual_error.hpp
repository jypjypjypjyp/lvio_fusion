#ifndef lvio_fusion_VISUAL_ERROR_H
#define lvio_fusion_VISUAL_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/visual/camera.h"

namespace lvio_fusion
{

template <typename T>
inline void Reprojection(const T *pw, const T *Twc, Camera::Ptr camera, T *result)
{
    T e[7], e_i[7], Twc_i[7], pc[3], pc_[3];
    ceres::SE3Inverse(Twc, Twc_i);
    ceres::SE3TransformPoint(Twc_i, pw, pc_);
    ceres::Cast(camera->extrinsic.data(), SE3d::num_parameters, e);
    ceres::SE3Inverse(e, e_i);
    ceres::SE3TransformPoint(e_i, pc_, pc);
    T xp = pc[0] / pc[2];
    T yp = pc[1] / pc[2];
    result[0] = camera->fx * xp + camera->cx;
    result[1] = camera->fy * yp + camera->cy;
}

template <typename T>
inline void Pixel2Robot(const T *ob, const T *inv_d, Camera::Ptr camera, T *result)
{
    T d = T(1) / inv_d[0];
    T ps[3] = {T((ob[0] - camera->cx) / camera->fx) * d, T((ob[1] - camera->cy) / camera->fy) * d, d};
    T e[7];
    ceres::Cast(camera->extrinsic.data(), SE3d::num_parameters, e);
    ceres::SE3TransformPoint(e, ps, result);
}

template <typename T>
inline void Robot2Pixel(const T *pb, Camera::Ptr camera, T *result)
{
    T e[7], e_i[7], pc[3];
    ceres::Cast(camera->extrinsic.data(), SE3d::num_parameters, e);
    ceres::SE3Inverse(e, e_i);
    ceres::SE3TransformPoint(e_i, pb, pc);
    T xp = pc[0] / pc[2];
    T yp = pc[1] / pc[2];
    result[0] = camera->fx * xp + camera->cx;
    result[1] = camera->fy * yp + camera->cy;
}

class PoseOnlyReprojectionError : public ceres::Error
{
public:
    PoseOnlyReprojectionError(Vector2d ob, Vector3d pw, Camera::Ptr camera, double weight)
        : ob_(ob), pw_(pw), camera_(camera), Error(weight) {}

    template <typename T>
    bool operator()(const T *Twc, T *residuals) const
    {
        T p_p[2];
        T pw[3] = {T(pw_.x()), T(pw_.y()), T(pw_.z())};
        T ob[2] = {T(ob_.x()), T(ob_.y())};
        Reprojection(pw, Twc, camera_, p_p);
        residuals[0] = T(weight_) * (p_p[0] - ob[0]);
        residuals[1] = T(weight_) * (p_p[1] - ob[1]);
        return true;
    }

    static ceres::CostFunction *Create(Vector2d ob, Vector3d pw, Camera::Ptr camera, double weight)
    {
        return (new ceres::AutoDiffCostFunction<PoseOnlyReprojectionError, 2, 7>(
            new PoseOnlyReprojectionError(ob, pw, camera, weight)));
    }

private:
    Vector2d ob_;
    Vector3d pw_;
    Camera::Ptr camera_;
};

class TwoFrameReprojectionError: public ceres::Error
{
public:
    TwoFrameReprojectionError(Vector2d first_ob, Vector2d ob, Camera::Ptr left, Camera::Ptr right, double weight)
        : first_ob_(first_ob), ob_(ob), left_(left), right_(right), Error(weight) {}

    template <typename T>
    bool operator()(const T *inv_d, const T *Twc1, const T *Twc2, T *residuals) const
    {
        T pixel[2], pw[3], pb[3];
        T first_ob[2] = {T(first_ob_.x()), T(first_ob_.y())};
        T ob2[2] = {T(ob_.x()), T(ob_.y())};
        Pixel2Robot(first_ob, inv_d, right_, pb);
        ceres::SE3TransformPoint(Twc1, pb, pw);
        Reprojection(pw, Twc2, left_, pixel);
        residuals[0] = T(weight_) * (pixel[0] - ob2[0]);
        residuals[1] = T(weight_) * (pixel[1] - ob2[1]);
        return true;
    }

    static ceres::CostFunction *Create(Vector2d first_ob, Vector2d ob, Camera::Ptr left, Camera::Ptr right, double weight)
    {
        return (new ceres::AutoDiffCostFunction<TwoFrameReprojectionError, 2, 1, 7, 7>(
            new TwoFrameReprojectionError(first_ob, ob, left, right, weight)));
    }

private:
    Vector2d first_ob_, ob_;
    Camera::Ptr left_, right_;
};

class TwoCameraReprojectionError: public ceres::Error
{
public:
    TwoCameraReprojectionError(Vector2d left_ob, Vector2d right_ob, Camera::Ptr left, Camera::Ptr right, double weight)
        : left_ob_(left_ob), right_ob_(right_ob), left_(left), right_(right), Error(weight) {}

    template <typename T>
    bool operator()(const T *inv_d, T *residuals) const
    {
        T pixel[2], pb[3];
        T right_ob[2] = {T(right_ob_.x()), T(right_ob_.y())};
        T left_ob[2] = {T(left_ob_.x()), T(left_ob_.y())};
        Pixel2Robot(right_ob, inv_d, right_, pb);
        Robot2Pixel(pb, left_, pixel);
        residuals[0] = T(weight_) * (pixel[0] - left_ob[0]);
        residuals[1] = T(weight_) * (pixel[1] - left_ob[1]);
        return true;
    }

    static ceres::CostFunction *Create(Vector2d left_ob, Vector2d right_ob, Camera::Ptr left, Camera::Ptr right, double weight)
    {
        return (new ceres::AutoDiffCostFunction<TwoCameraReprojectionError, 2, 1>(
            new TwoCameraReprojectionError(left_ob, right_ob, left, right, weight)));
    }

private:
    Vector2d left_ob_, right_ob_;
    Camera::Ptr left_, right_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_VISUAL_ERROR_H
