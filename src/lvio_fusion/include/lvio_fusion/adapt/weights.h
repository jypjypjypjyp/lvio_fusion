#ifndef lvio_fusion_WEIGHTS_H
#define lvio_fusion_WEIGHTS_H

namespace lvio_fusion
{

struct Weights
{
    // imu is a constant 1
    float visual = 0;
    float lidar_ground = 1;
    float lidar_surf = 0.01;
    bool updated = false;
};

} // namespace lvio_fusion

#endif // lvio_fusion_WEIGHTS_H
