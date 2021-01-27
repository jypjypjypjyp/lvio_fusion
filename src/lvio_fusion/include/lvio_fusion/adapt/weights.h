#ifndef lvio_fusion_WEIGHTS_H
#define lvio_fusion_WEIGHTS_H

namespace lvio_fusion
{

struct Weights
{
    double visual = 1;
    double lidar_ground = 1;
    double lidar_surf = 0.01;
    double pose_graph = 1;
    bool updated = false;
};

} // namespace lvio_fusion

#endif // lvio_fusion_WEIGHTS_H
