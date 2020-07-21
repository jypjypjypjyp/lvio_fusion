// #ifndef lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H
// #define lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H

// #include "lvio_fusion/sensors/camera.hpp"
// #include "lvio_fusion/utility.h"
// #include <ceres/ceres.h>

// namespace lvio_fusion
// {

// class TwoFrameReprojectionError
// {
// public:
//     TwoFrameReprojectionError(Vector2d ob1, Vector2d ob2, Camerad::Ptr camera)
//         : ob1_(ob1), ob2_(ob2), camera_(camera) {}

//     template <typename T>
//     bool operator()(const T *pose1_, const T *pose2_, const T *depth_, T *residuals_) const
//     {
//         Eigen::Map<Sophus::SE3<T> const> pose1(pose1_);
//         Eigen::Map<Sophus::SE3<T> const> pose2(pose2_);
//         T depth = depth_[0];
//         Eigen::Map<Matrix<T, 2, 1>> residuals(residuals_);
//         Camera<T> camera = camera_->cast<T>();
//         Matrix<T, 3, 1> p_w = camera.Pixel2World(ob1_.cast<T>(), pose1, depth);
//         Matrix<T, 2, 1> pixel = camera.World2Pixel(p_w, pose2);
//         residuals = ob2_ - pixel;
//         residuals.applyOnTheLeft(sqrt_information);
//         return true;
//     }

//     static ceres::CostFunction *Create(Vector2d ob1, Vector2d ob2, Camerad::Ptr camera)
//     {
//         return (new ceres::AutoDiffCostFunction<TwoFrameReprojectionError, 2, 7, 7, 1>(
//             new TwoFrameReprojectionError(ob1, ob2, camera)));
//     }

//     static Matrix2d sqrt_information;

// private:
//     Vector2d ob1_;
//     Vector2d ob2_;
//     Camerad::Ptr camera_;
// };

// } // namespace lvio_fusion

// #endif // lvio_fusion_TWO_FRAME_REPROJECTION_ERROR_H
