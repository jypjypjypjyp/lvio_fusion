#ifndef lvio_fusion_WEIGHTS_H
#define lvio_fusion_WEIGHTS_H

namespace lvio_fusion
{

struct Weights
{
    double visual[2] = {1, 1};
    double navsat[7] = {1, 1, 1, 10, 50, 50, 1};
    double lidar_ground[3] = {1, 1, 1};
    double lidar_surf[3] = {2, 2, 2};
    double imu[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    double pose_graph[6] = {1, 1, 1, 1, 1, 1};
};

} // namespace lvio_fusion

#endif // lvio_fusion_WEIGHTS_H
