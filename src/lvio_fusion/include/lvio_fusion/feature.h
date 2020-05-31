#ifndef lvio_fusion_FEATURE_H
#define lvio_fusion_FEATURE_H

#include <memory>
#include <opencv2/features2d.hpp>
#include "lvio_fusion/common.h"

namespace lvio_fusion
{

class Frame;
class MapPoint;

class Feature
{
public:
    typedef std::shared_ptr<Feature> Ptr;

    std::weak_ptr<Frame> frame;
    cv::KeyPoint pos;
    std::weak_ptr<MapPoint> map_point;

    bool is_outlier = false;
    bool is_on_left_image = true;

public:
    Feature() {}

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
        : frame(frame), pos(kp) {}
};
} // namespace lvio_fusion

#endif // lvio_fusion_FEATURE_H
