#ifndef lvio_fusion_SENSOR_H
#define lvio_fusion_SENSOR_H

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

class Sensor
{
public:
    typedef std::shared_ptr<Sensor> Ptr;

    Sensor(const SE3d &extrinsic) : extrinsic(extrinsic) {}

    // coordinate transform: world, sensor, pixel
    virtual Vector3d World2Sensor(const Vector3d &p_w, const SE3d &T_c_w)
    {
        throw NotImplemented();
    }

    virtual Vector3d Sensor2World(const Vector3d &p_c, const SE3d &T_c_w)
    {
        throw NotImplemented();
    }

    virtual Vector2d Sensor2Pixel(const Vector3d &p_c)
    {
        throw NotImplemented();
    }

    virtual Vector3d Pixel2Sensor(const Vector2d &p_p, double depth = 1)
    {
        throw NotImplemented();
    }

    virtual Vector3d Pixel2World(const Vector2d &p_p, const SE3d &T_c_w, double depth = 1)
    {
        return Sensor2World(Pixel2Sensor(p_p, depth), T_c_w);
    }

    virtual Vector2d World2Pixel(const Vector3d &p_w, const SE3d &T_c_w)
    {
        return Sensor2Pixel(World2Sensor(p_w, T_c_w));
    }

    SE3d extrinsic;
};

} // namespace lvio_fusion
#endif // lvio_fusion_SENSOR_H
