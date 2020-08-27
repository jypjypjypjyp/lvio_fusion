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

    static Feature::Ptr Create(
        const PointICloud &points_sharp,
        const PointICloud &points_less_sharp,
        const PointICloud &points_flat,
        const PointICloud &points_less_flat)
    {
        Feature::Ptr new_feature(new Feature);
        new_feature->points_sharp = points_sharp;
        new_feature->points_less_sharp = points_less_sharp;
        new_feature->points_flat = points_flat;
        new_feature->points_less_flat = points_less_flat;
        return new_feature;
    }

    PointICloud points_sharp;
    PointICloud points_less_sharp;
    PointICloud points_flat;
    PointICloud points_less_flat;
};

} // namespace lidar

} // namespace lvio_fusion

#endif // lvio_fusion_FEATURE_H
