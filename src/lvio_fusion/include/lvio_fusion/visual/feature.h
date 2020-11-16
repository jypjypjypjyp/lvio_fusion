#ifndef lvio_fusion_VISUAL_FEATURE_H
#define lvio_fusion_VISUAL_FEATURE_H

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

class Frame;

namespace visual
{

class Landmark;

class Feature
{
public:
    typedef std::shared_ptr<Feature> Ptr;

    Feature() {}

    static Feature::Ptr Create(std::shared_ptr<Frame> frame, const cv::Point2f &kp, std::shared_ptr<Landmark> landmark)
    {
        Feature::Ptr new_feature(new Feature);
        new_feature->frame = frame;
        new_feature->keypoint = kp;
        new_feature->landmark = landmark;
        return new_feature;
    }

    std::weak_ptr<Frame> frame;
    cv::Point2f keypoint;
    std::weak_ptr<Landmark> landmark;
    bool is_on_left_image = true;
};

typedef std::map<unsigned long, Feature::Ptr> Features;
} // namespace visual

} // namespace lvio_fusion

#endif // lvio_fusion_VISUAL_FEATURE_H
