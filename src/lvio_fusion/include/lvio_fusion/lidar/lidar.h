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

    static int Create(double resolution, const SE3d &extrinsic)
    {
        devices_.push_back(Lidar::Ptr(new Lidar(resolution, extrinsic)));
        return devices_.size() - 1;
    }

    static int Num()
    {
        return devices_.size();
    }

    static Lidar::Ptr &Get(int id = 0)
    {
        return devices_[id];
    }

    double resolution;

private:
    Lidar(double resolution, const SE3d &extrinsic) : resolution(resolution), Sensor(extrinsic) {}
    Lidar(const Lidar &);
    Lidar &operator=(const Lidar &);

    static std::vector<Lidar::Ptr> devices_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_LIDAR_H
