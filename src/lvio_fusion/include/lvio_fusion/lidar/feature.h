#ifndef lvio_fusion_LIDAR_FEATURE_H
#define lvio_fusion_LIDAR_FEATURE_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/lidar/lidar.hpp"
#include <ceres/ceres.h>

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
        const PointICloud &cornerPointsSharp,
        const PointICloud &cornerPointsLessSharp,
        const PointICloud &surfPointsFlat,
        const PointICloud &surfPointsLessFlat)
    {
        Feature::Ptr new_feature(new Feature);
        new_feature->cornerPointsSharp = cornerPointsSharp;
        new_feature->cornerPointsLessSharp = cornerPointsLessSharp;
        new_feature->surfPointsFlat = surfPointsFlat;
        new_feature->surfPointsLessFlat = surfPointsLessFlat;
        return new_feature;
    }

    PointICloud cornerPointsSharp;
    PointICloud cornerPointsLessSharp;
    PointICloud surfPointsFlat;
    PointICloud surfPointsLessFlat;

    static void Feature::Associate(Lidar::Ptr lidar, std::shared_ptr<Frame> current_frame, std::shared_ptr<Frame> last_frame, ceres::Problem &problem);
};

} // namespace lidar

} // namespace lvio_fusion

#endif // lvio_fusion_FEATURE_H
