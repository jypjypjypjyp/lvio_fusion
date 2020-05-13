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

    std::weak_ptr<Frame> frame_;
    cv::KeyPoint position_;
    std::weak_ptr<MapPoint> map_point_;

    bool is_outlier_ = false;
    bool is_on_left_image_ = true;

public:
    Feature() {}

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
        : frame_(frame), position_(kp) {}
};
} // namespace lvio_fusion

#endif // lvio_fusion_FEATURE_H
