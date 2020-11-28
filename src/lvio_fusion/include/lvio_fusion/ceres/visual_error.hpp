#ifndef lvio_fusion_VISUAL_ERROR_H
#define lvio_fusion_VISUAL_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/visual/camera.hpp"

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

class PoseOnlyReprojectionError
{
public:
    PoseOnlyReprojectionError(Vector2d ob, Vector3d pw, Camera::Ptr camera, double* weights)
        : ob_(ob), pw_(pw), camera_(camera), weights_(weights) {}

    template <typename T>
    bool operator()(const T *Twc, T *residuals) const
    {
        T p_p[2];
        T pw[3] = {T(pw_.x()), T(pw_.y()), T(pw_.z())};
        T ob[2] = {T(ob_.x()), T(ob_.y())};
        Reprojection(pw, Twc, camera_, p_p);
        residuals[0] = T(weights_[0]) * (p_p[0] - ob[0]);
        residuals[1] = T(weights_[1]) * (p_p[1] - ob[1]);
        return true;
    }

    static ceres::CostFunction *Create(Vector2d ob, Vector3d pw, Camera::Ptr camera, double* weights)
    {
        return (new ceres::AutoDiffCostFunction<PoseOnlyReprojectionError, 2, 7>(
            new PoseOnlyReprojectionError(ob, pw, camera, weights)));
    }

private:
    Vector2d ob_;
    Vector3d pw_;
    Camera::Ptr camera_;
    const double *weights_;
};

class TwoFrameReprojectionError
{
public:
    TwoFrameReprojectionError(Vector3d pr, Vector2d ob, Camera::Ptr camera, double* weights)
        : pr_(pr), ob_(ob), camera_(camera), weights_(weights) {}

    template <typename T>
    bool operator()(const T *Twc1, const T *Twc2, T *residuals) const
    {
        T pixel[2], pw[3];
        T pr[3] = {T(pr_.x()), T(pr_.y()), T(pr_.z())};
        T ob2[2] = {T(ob_.x()), T(ob_.y())};
        ceres::SE3TransformPoint(Twc1, pr, pw);
        Reprojection(pw, Twc2, camera_, pixel);
        residuals[0] = T(weights_[0]) * (pixel[0] - ob2[0]);
        residuals[1] = T(weights_[1]) * (pixel[1] - ob2[1]);
        return true;
    }

    static ceres::CostFunction *Create(Vector3d pr, Vector2d ob, Camera::Ptr camera, double* weights)
    {
        return (new ceres::AutoDiffCostFunction<TwoFrameReprojectionError, 2, 7, 7>(
            new TwoFrameReprojectionError(pr, ob, camera, weights)));
    }

private:
    Vector3d pr_;
    Vector2d ob_;
    Camera::Ptr camera_;
    const double* weights_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_VISUAL_ERROR_H
