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
    virtual Vector2d Sensor2Pixel(const Vector3d &p_c)
    {
        return Vector2d(
            fx * p_c(0, 0) / p_c(2, 0) + cx,
            fy * p_c(1, 0) / p_c(2, 0) + cy);
    }

    virtual Vector3d Pixel2Sensor(const Vector2d &p_p, double depth = double(1))
    {
        return Vector3d(
            (p_p(0, 0) - cx) * depth / fx,
            (p_p(1, 0) - cy) * depth / fy,
            depth);
    }

    virtual Vector3d Pixel2World(const Vector2d &p_p, const SE3d &T_c_w, double depth = 1)
    {
        return Sensor2World(Pixel2Sensor(p_p, depth), T_c_w);
    }

    virtual Vector2d World2Pixel(const Vector3d &p_w, const SE3d &T_c_w)
    {
        return Sensor2Pixel(World2Sensor(p_w, T_c_w));
    }

    virtual Vector3d Pixel2Robot(const Vector2d &p_p, double depth = 1)
    {
        return Sensor2Robot(Pixel2Sensor(p_p, depth));
    }

    virtual Vector2d Robot2Pixel(const Vector3d &p_w)
    {
        return Sensor2Pixel(Robot2Sensor(p_w));
    }

    double fx = 0, fy = 0, cx = 0, cy = 0; // Camera intrinsics
};

} // namespace lvio_fusion
#endif // lvio_fusion_CAMERA_H
