#ifndef lvio_fusion_LIDAR_H
#define lvio_fusion_LIDAR_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/sensor.h"

namespace lvio_fusion
{

class Lidar : public Sensor
{
public:
    typedef std::shared_ptr<Lidar> Ptr;

    Lidar(const SE3d &extrinsic) : Sensor(extrinsic) {}

    PointI Transform(const PointI &p, Frame::Ptr from, Frame::Ptr to)
    {
        auto p1 = Sensor2World(Vector3d(p.x, p.y, p.z), from->pose);
        auto p2 = World2Sensor(p1, to->pose);
        PointI r;
        r.x = p2.x();
        r.y = p2.y();
        r.z = p2.z();
        r.intensity = p.intensity;
        return r;
    }
};

} // namespace lvio_fusion
#endif // lvio_fusion_LIDAR_H
