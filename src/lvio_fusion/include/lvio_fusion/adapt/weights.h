#ifndef lvio_fusion_WEIGHTS_H
#define lvio_fusion_WEIGHTS_H

namespace lvio_fusion
{

struct Weights
{
    double visual[2] = {1, 1};
    double navsat[7] = {10, 10, 10, 10, 10, 10, 1};
    double lidar_ground[3] = {1, 1, 0};
    double lidar_surf[3] = {0, 0, 0};
    double imu[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    double pose_graph[6] = {1, 1, 1, 1, 1, 1};
};

} // namespace lvio_fusion

#endif // lvio_fusion_WEIGHTS_H
