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

    Lidar(const SE3d &extrinsic, double resolution) : Sensor(extrinsic), resolution(resolution) {}

    inline void Transform(const PointI &in, SE3d from_pose, SE3d to_pose, PointI &out)
    {
        auto p1 = Sensor2World(Vector3d(in.x, in.y, in.z), from_pose);
        auto p2 = World2Sensor(p1, to_pose);
        out.x = p2.x();
        out.y = p2.y();
        out.z = p2.z();
        out.intensity = in.intensity;
    }

    inline SE3d TransformMatrix(SE3d from_pose)
    {
        return from_pose * extrinsic;
    }

    double resolution;
};

} // namespace lvio_fusion
#endif // lvio_fusion_LIDAR_H
