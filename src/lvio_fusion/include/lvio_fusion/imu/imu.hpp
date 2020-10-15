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

    Imu(const SE3d &extrinsic) : Sensor(extrinsic) {}

    double ACC_N, ACC_W;
    double GYR_N, GYR_W;
    bool initialized =false;
};

} // namespace lvio_fusion
#endif // lvio_fusion_IMU_H
