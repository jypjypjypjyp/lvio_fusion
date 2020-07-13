#ifndef lvio_fusion_TWO_CAMERA_REPROJECTION_ERROR_H
#define lvio_fusion_TWO_CAMERA_REPROJECTION_ERROR_Hv

#include "lvio_fusion/sensors/camera.hpp"
#include "lvio_fusion/utility.h"
#include <ceres/ceres.h>

namespace lvio_fusion
{

class TwoCameraReprojectionError
{
public:
    TwoCameraReprojectionError(Vector2d ob1, Vector2d ob2, Camerad::Ptr camera1, Camerad::Ptr camera2)
        : ob1_(ob1), ob2_(ob2), camera1_(camera1), camera2_(camera2) {}

    template <typename T>
    bool operator()(const T *depth_, T *residuals_) const
    {
        T depth = depth_[0];
        Eigen::Map<Matrix<T, 2, 1>> residuals(residuals_);
        Camera<T> camera1 = camera1_->cast<T>();
        Camera<T> camera2 = camera2_->cast<T>();
        Matrix<T, 3, 1> p_c = camera1.Pixel2Sensor(ob1_.cast<T>(), depth);
        Matrix<T, 2, 1> pixel = camera2.Sensor2Pixel(p_c);
        residuals = ob2_ - pixel;
        residuals.applyOnTheLeft(sqrt_information);
        return true;
    }

    static ceres::CostFunction *Create(Vector2d ob1, Vector2d ob2, Camerad::Ptr camera1, Camerad::Ptr camera2)
    {
        return (new ceres::AutoDiffCostFunction<TwoCameraReprojectionError, 2, 1>(
            new TwoCameraReprojectionError(ob1, ob2, camera1, camera2)));
    }

    static Matrix2d sqrt_information;

private:
    Vector2d ob1_;
    Vector2d ob2_;
    Camerad::Ptr camera1_;
    Camerad::Ptr camera2_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_TWO_CAMERA_REPROJECTION_ERROR_H
