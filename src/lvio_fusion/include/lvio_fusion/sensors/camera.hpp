

#ifndef lvio_fusion_CAMERA_H
#define lvio_fusion_CAMERA_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/sensor.h"

namespace lvio_fusion
{

// Pinhole stereo camera model
class Camera : public Sensor
{
public:
    typedef std::shared_ptr<Camera> Ptr;

    Camera(double fx, double fy, double cx, double cy, const SE3d &extrinsic)
        : fx(fx), fy(fy), cx(cx), cy(cy), Sensor(extrinsic) {}

    // return intrinsic matrix
    Matrix3d K() const
    {
        Matrix3d k;
        k << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        return k;
    }

    // coordinate transform: world, sensor, pixel
    template <typename T>
    Matrix<T, 3, 1> World2Sensor(const Matrix<T, 3, 1> &p_w, const Sophus::SE3<T> &T_c_w)
    {
        return extrinsic.template cast<T>() * T_c_w * p_w;
    }

    template <typename T>
    Matrix<T, 3, 1> Sensor2World(const Matrix<T, 3, 1> &p_c, const Sophus::SE3<T> &T_c_w)
    {
        return T_c_w.inverse() * extrinsic.inverse().template cast<T>() * p_c;
    }

    template <typename T>
    Matrix<T, 2, 1> Camera2Pixel(const Matrix<T, 3, 1> &p_c)
    {
        return Matrix<T, 2, 1>(
            T(fx) * p_c(0, 0) / p_c(2, 0) + T(cx),
            T(fy) * p_c(1, 0) / p_c(2, 0) + T(cy));
    }

    template <typename T>
    Matrix<T, 3, 1> Pixel2Camera(const Matrix<T, 2, 1> &p_p, T depth = T(1))
    {
        return Matrix<T, 3, 1>(
            (p_p(0, 0) - T(cx)) * depth / T(fx),
            (p_p(1, 0) - T(cy)) * depth / T(fy),
            depth);
    }

    double fx = 0, fy = 0, cx = 0, cy = 0; // Camera intrinsics
};

} // namespace lvio_fusion
#endif // lvio_fusion_CAMERA_H
