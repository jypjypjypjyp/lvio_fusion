#ifndef lvio_fusion_WEIGHTS_H
#define lvio_fusion_WEIGHTS_H

namespace lvio_fusion
{

struct Weights
{
    // imu is a constant 1
    float visual;
    float lidar_ground;
    float lidar_surf;
    bool updated = false;
};

} // namespace lvio_fusion

#endif // lvio_fusion_WEIGHTS_H
