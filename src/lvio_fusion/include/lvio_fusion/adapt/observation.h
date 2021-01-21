#ifndef lvio_fusion_OBSERVATION_H
#define lvio_fusion_OBSERVATION_H

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

struct Observation
{
    cv::Mat image;
    PointICloud points_ground;
    PointICloud points_surf;
};

} // namespace lvio_fusion

#endif // lvio_fusion_OBSERVATION_H
