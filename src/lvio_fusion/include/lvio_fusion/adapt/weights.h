#ifndef lvio_fusion_WEIGHTS_H
#define lvio_fusion_WEIGHTS_H

namespace lvio_fusion
{

struct Weights
{
    double visual[2] = {1, 1};
    double navsat[3] = {10, 10, 10};
    double lidar_ground[1] = {0.1};
    double lidar_surf[1] = {0.1};
    double imu[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
};

} // namespace lvio_fusion

#endif // lvio_fusion_WEIGHTS_H
