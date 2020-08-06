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

};

} // namespace lvio_fusion
#endif // lvio_fusion_LIDAR_H
