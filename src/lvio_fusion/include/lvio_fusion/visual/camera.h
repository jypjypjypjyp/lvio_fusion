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

    static int Create(double fx, double fy, double cx, double cy, const SE3d &extrinsic)
    {
        devices_.push_back(Camera::Ptr(new Camera(fx, fy, cx, cy, extrinsic)));
        return devices_.size() - 1;
    }

    static int Num()
    {
        return devices_.size();
    }

    static Camera::Ptr &Get(int id = 0)
    {
        return devices_[id];
    }

    // return intrinsic matrix
    Matrix3d K() const
    {
        Matrix3d k;
        k << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        return k;
    }

    // coordinate transform: world, sensor, pixel
    Vector2d Sensor2Pixel(const Vector3d &pc)
    {
        return Vector2d(
            fx * pc(0, 0) / pc(2, 0) + cx,
            fy * pc(1, 0) / pc(2, 0) + cy);
    }

    Vector3d Pixel2Sensor(const Vector2d &pp, double depth = double(1))
    {
        return Vector3d(
            (pp(0, 0) - cx) * depth / fx,
            (pp(1, 0) - cy) * depth / fy,
            depth);
    }

    Vector3d Pixel2World(const Vector2d &pp, const SE3d &Tcw, double depth = 1)
    {
        return Sensor2World(Pixel2Sensor(pp, depth), Tcw);
    }

    Vector2d World2Pixel(const Vector3d &pw, const SE3d &Tcw)
    {
        return Sensor2Pixel(World2Sensor(pw, Tcw));
    }

    Vector3d Pixel2Robot(const Vector2d &pp, double depth = 1)
    {
        return Sensor2Robot(Pixel2Sensor(pp, depth));
    }

    Vector2d Robot2Pixel(const Vector3d &pw)
    {
        return Sensor2Pixel(Robot2Sensor(pw));
    }

    double fx = 0, fy = 0, cx = 0, cy = 0; // Camera intrinsics

private:
    Camera(double fx, double fy, double cx, double cy, const SE3d &extrinsic)
        : fx(fx), fy(fy), cx(cx), cy(cy), Sensor(extrinsic) {}
    Camera(const Camera &);
    Camera &operator=(const Camera &);

    static std::vector<Camera::Ptr> devices_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_CAMERA_H
