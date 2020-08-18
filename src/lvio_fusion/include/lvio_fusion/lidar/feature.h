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
        new_feature->cornerPointsSharpDeskew = cornerPointsSharp;
        new_feature->cornerPointsLessSharpDeskew = cornerPointsLessSharp;
        new_feature->surfPointsFlatDeskew = surfPointsFlat;
        new_feature->surfPointsLessFlatDeskew = surfPointsLessFlat;
        return new_feature;
    }

    // raw point cloud
    PointICloud cornerPointsSharp;
    PointICloud cornerPointsLessSharp;
    PointICloud surfPointsFlat;
    PointICloud surfPointsLessFlat;

    // undistorted point cloud
    PointICloud cornerPointsSharpDeskew;
    PointICloud cornerPointsLessSharpDeskew;
    PointICloud surfPointsFlatDeskew;
    PointICloud surfPointsLessFlatDeskew;

    int iterations = 0;
};

} // namespace lidar

} // namespace lvio_fusion

#endif // lvio_fusion_FEATURE_H
