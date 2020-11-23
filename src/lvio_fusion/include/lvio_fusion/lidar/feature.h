#ifndef lvio_fusion_LIDAR_FEATURE_H
#define lvio_fusion_LIDAR_FEATURE_H

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

class Frame;

namespace lidar
{
class Feature
{
public:
    typedef std::shared_ptr<Feature> Ptr;

    static Feature::Ptr Create()
    {
        return Feature::Ptr(new Feature);
    }

    // PointICloud points_sharp;
    // PointICloud points_less_sharp;
    PointICloud points_surf;
    PointICloud points_ground;
    PointICloud points_full;
};

} // namespace lidar

} // namespace lvio_fusion

#endif // lvio_fusion_FEATURE_H
