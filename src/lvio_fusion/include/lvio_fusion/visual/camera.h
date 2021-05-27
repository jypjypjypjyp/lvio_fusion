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
        devices_.push_back(Camera::Ptr(new Camera(fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, extrinsic)));
        return devices_.size() - 1;
    }

    static int Create(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, const SE3d &extrinsic)
    {
        devices_.push_back(Camera::Ptr(new Camera(fx, fy, cx, cy, k1, k2, p1, p2, extrinsic)));
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

    bool Far(const Vector3d &pw, const SE3d &Tcw)
    {
        return World2Sensor(pw, Tcw).z() > baseline * 50;
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

    static double baseline;
    static double sqrt_info;
    double fx = 0, fy = 0, cx = 0, cy = 0; // Camera intrinsics
    double k1 = 0, k2 = 0, p1 = 0, p2 = 0; // Camera intrinsics
    cv::Mat K, D;

private:
    Camera(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, const SE3d &extrinsic)
        : fx(fx), fy(fy), cx(cx), cy(cy), k1(k1), k2(k2), p1(p1), p2(p2), Sensor(extrinsic)
    {
        K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
        D = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, 0);
    }
    Camera(const Camera &);
    Camera &operator=(const Camera &);

    static std::vector<Camera::Ptr> devices_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_CAMERA_H
