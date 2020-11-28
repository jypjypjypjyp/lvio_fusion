#ifndef lvio_fusion_SENSOR_H
#define lvio_fusion_SENSOR_H

#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/common.h"

namespace lvio_fusion
{

class Sensor
{
public:
    typedef std::shared_ptr<Sensor> Ptr;

    Sensor(const SE3d &extrinsic) : extrinsic(extrinsic) {}

    // coordinate transform: world, sensor
    virtual Vector3d World2Sensor(const Vector3d &pw, const SE3d &Twc)
    {
        return extrinsic.inverse() * Twc.inverse() * pw;
    }

    virtual Vector3d Sensor2World(const Vector3d &pc, const SE3d &Twc)
    {
        return Twc * extrinsic * pc;
    }

    virtual Vector3d World2Robot(const Vector3d &pw, const SE3d &Twc)
    {
        return Twc.inverse() * pw;
    }

    virtual Vector3d Robot2World(const Vector3d &pc, const SE3d &Twc)
    {
        return Twc * pc;
    }

    virtual Vector3d Robot2Sensor(const Vector3d &pb)
    {
        return extrinsic.inverse() * pb;
    }

    virtual Vector3d Sensor2Robot(const Vector3d &pc)
    {
        return extrinsic * pc;
    }

    SE3d extrinsic;
};

} // namespace lvio_fusion
#endif // lvio_fusion_SENSOR_H
