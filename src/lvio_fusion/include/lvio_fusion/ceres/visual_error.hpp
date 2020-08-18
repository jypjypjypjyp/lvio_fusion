#ifndef lvio_fusion_VISUAL_ERROR_H
#define lvio_fusion_VISUAL_ERROR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/visual/camera.hpp"

namespace lvio_fusion
{

template <typename T>
inline void Reprojection(const T *p_w, const T *Tcw, Camera::Ptr camera, T *result)
{
    T p_c[3], p_c_[3];
    ceres::SE3TransformPoint(Tcw, p_w, p_c_);
    T extrinsic[7];
    ceres::Cast(camera->extrinsic.data(), SE3d::num_parameters, extrinsic);
    ceres::SE3TransformPoint(extrinsic, p_c_, p_c);
    T xp = p_c[0] / p_c[2];
    T yp = p_c[1] / p_c[2];
    result[0] = camera->fx * xp + camera->cx;
    result[1] = camera->fy * yp + camera->cy;
}

class PoseOnlyReprojectionError
{
public:
    PoseOnlyReprojectionError(Vector2d ob, Vector3d p_w, Camera::Ptr camera)
        : ob_(ob), p_w_(p_w), camera_(camera) {}

    template <typename T>
    bool operator()(const T *Tcw, T *residuals) const
    {
        T p_p[2];
        T p_w[3] = {T(p_w_.x()), T(p_w_.y()), T(p_w_.z())};
        T ob[2] = {T(ob_.x()), T(ob_.y())};
        Reprojection(p_w, Tcw, camera_, p_p);
        residuals[0] = T(sqrt_info(0, 0)) * (p_p[0] - ob[0]);
        residuals[1] = T(sqrt_info(1, 1)) * (p_p[1] - ob[1]);
        return true;
    }

    static ceres::CostFunction *Create(Vector2d ob, Vector3d p_w, Camera::Ptr camera)
    {
        return (new ceres::AutoDiffCostFunction<PoseOnlyReprojectionError, 2, 7>(
            new PoseOnlyReprojectionError(ob, p_w, camera)));
    }

    static Matrix2d sqrt_info;

private:
    Vector2d ob_;
    Vector3d p_w_;
    Camera::Ptr camera_;
};

template <typename T>
inline void Projection(const T *p_c, const T *Tcw, T *result)
{
    T Tcw_inverse[7];
    ceres::SE3Inverse(Tcw, Tcw_inverse);
    ceres::SE3TransformPoint(Tcw_inverse, p_c, result);
}

class TwoFrameReprojectionError
{
public:
    TwoFrameReprojectionError(Vector3d p_r, Vector2d ob, Camera::Ptr camera)
        : p_r_(p_r), ob_(ob), camera_(camera) {}

    template <typename T>
    bool operator()(const T *Tcw1, const T *Tcw2, T *residuals) const
    {
        T p_p[2], p_w[3];
        T p_r[3] = {T(p_r_.x()), T(p_r_.y()), T(p_r_.z())};
        T ob2[2] = {T(ob_.x()), T(ob_.y())};
        Projection(p_r, Tcw1, p_w);
        Reprojection(p_w, Tcw2, camera_, p_p);
        residuals[0] = T(sqrt_info(0, 0)) * (p_p[0] - ob2[0]);
        residuals[1] = T(sqrt_info(1, 1)) * (p_p[1] - ob2[1]);
        return true;
    }

    static ceres::CostFunction *Create(Vector3d p_r, Vector2d ob, Camera::Ptr camera)
    {
        return (new ceres::AutoDiffCostFunction<TwoFrameReprojectionError, 2, 7, 7>(
            new TwoFrameReprojectionError(p_r, ob, camera)));
    }

    static Matrix2d sqrt_info;

private:
    Vector3d p_r_;
    Vector2d ob_;
    Camera::Ptr camera_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_VISUAL_ERROR_H
