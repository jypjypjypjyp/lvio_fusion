#ifndef lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H
#define lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H

#include "lvio_fusion/sensor.h"
#include "lvio_fusion/utility.h"
#include <ceres/ceres.h>

namespace lvio_fusion
{

class TwoFrameReprojectionError
{
public:
    TwoFrameReprojectionError(Vector2d ob1, Vector2d ob2, Sensor::Ptr sensor)
        : ob1_(ob1), ob2_(ob2), sensor_(sensor) {}

    template <typename T>
    bool operator()(const T *pose1_, const T *pose2_, const T *depth_, T *residuals_) const
    {
        Eigen::Map<Sophus::SE3<T> const> pose1(pose1_);
        Eigen::Map<Sophus::SE3<T> const> pose2(pose2_);
        T depth = depth_[0];
        Eigen::Map<Matrix<T, 2, 1>> residuals(residuals);
        Matrix<T, 3, 1> p_w = sensor_->Pixel2World<T>(ob1_.template cast<T>(), pose1, depth);
        Matrix<T, 2, 1> pixel = sensor_->World2Pixel<T>(p_w, pose2);
        residuals = ob2_.template cast<T>() - pixel;
        residuals.applyOnTheLeft(sqrt_information.template cast<T>());
        return true;
    }

    static ceres::CostFunction *Create(Vector2d ob1, Vector2d ob2, Sensor::Ptr sensor)
    {
        return (new ceres::AutoDiffCostFunction<TwoFrameReprojectionError, 2, 7, 7, 1>(
            new TwoFrameReprojectionError(ob1, ob2, sensor)));
    }

    static Matrix2d sqrt_information;

private:
    Vector2d ob1_;
    Vector2d ob2_;
    Sensor::Ptr sensor_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H
