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

    // coordinate transform: world, sensor
    virtual Vector3d World2Sensor(const Vector3d &p_w, const SE3d &T_c_w)
    {
        return extrinsic * T_c_w * p_w;
    }

    virtual Vector3d Sensor2World(const Vector3d &p_c, const SE3d &T_c_w)
    {
        return T_c_w.inverse() * extrinsic.inverse() * p_c;
    }

    virtual Vector3d World2Robot(const Vector3d &p_w, const SE3d &T_c_w)
    {
        return T_c_w * p_w;
    }

    virtual Vector3d Robot2World(const Vector3d &p_c, const SE3d &T_c_w)
    {
        return T_c_w.inverse() * p_c;
    }

    virtual Vector3d Robot2Sensor(const Vector3d &p_w)
    {
        return extrinsic * p_w;
    }

    virtual Vector3d Sensor2Robot(const Vector3d &p_c)
    {
        return extrinsic.inverse() * p_c;
    }

    SE3d extrinsic;
};

} // namespace lvio_fusion
#endif // lvio_fusion_SENSOR_H
