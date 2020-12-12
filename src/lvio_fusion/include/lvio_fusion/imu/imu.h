#ifndef lvio_fusion_IMU_H
#define lvio_fusion_IMU_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/sensor.h"

namespace lvio_fusion
{

class Imu : public Sensor
{
public:
    typedef std::shared_ptr<Imu> Ptr;

    static int Create(const SE3d &extrinsic)
    {
        devices_.push_back(Imu::Ptr(new Imu(extrinsic)));
        return devices_.size() - 1;
    }

    static int Num()
    {
        return devices_.size();
    }

    static Imu::Ptr Get(int id = 0)
    {
        return devices_[id];
    }

    double ACC_N, ACC_W;
    double GYR_N, GYR_W;
    bool initialized = false;

private:
    Imu(const SE3d &extrinsic) : Sensor(extrinsic) {}
    Imu(const Imu &);
    Imu &operator=(const Imu &);

    static std::vector<Imu::Ptr> devices_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_IMU_H
