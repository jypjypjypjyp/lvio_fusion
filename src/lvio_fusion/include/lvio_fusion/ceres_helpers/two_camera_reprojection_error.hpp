#ifndef lvio_fusion_TWO_CAMERA_REPROJECTION_ERROR_H
#define lvio_fusion_TWO_CAMERA_REPROJECTION_ERROR_Hv

#include "lvio_fusion/sensor.h"
#include "lvio_fusion/utility.h"
#include <ceres/ceres.h>

namespace lvio_fusion
{

class TwoCameraReprojectionError
{
public:
    TwoCameraReprojectionError(Vector2d ob1, Vector2d ob2, Sensor::Ptr sensor1, Sensor::Ptr sensor2)
        : ob1_(ob1), ob2_(ob2), sensor1_(sensor1), sensor2_(sensor2) {}

    template <typename T>
    bool operator()(const T *depth_, T *residuals_) const
    {
        T depth = depth_[0];
        Eigen::Map<Matrix<T,2,1>> residuals(residuals_);
        Matrix<T,3,1> p_c = sensor1_->Pixel2Sensor<T>(ob1_.template cast<T>(), depth);
        Matrix<T,2,1> pixel = sensor2_->Sensor2Pixel<T>(p_c);
        residuals = ob2_.template cast<T>() - pixel;
        residuals.applyOnTheLeft(sqrt_information.template cast<T>());
        return true;
    }

    static ceres::CostFunction *Create(Vector2d ob1, Vector2d ob2, Sensor::Ptr sensor1, Sensor::Ptr sensor2)
    {
        return (new ceres::AutoDiffCostFunction<TwoCameraReprojectionError, 2, 1>(
            new TwoCameraReprojectionError(ob1, ob2, sensor1, sensor2)));
    }

    static Matrix2d sqrt_information;

private:
    Vector2d ob1_;
    Vector2d ob2_;
    Sensor::Ptr sensor1_;
    Sensor::Ptr sensor2_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_TWO_CAMERA_REPROJECTION_ERROR_H
