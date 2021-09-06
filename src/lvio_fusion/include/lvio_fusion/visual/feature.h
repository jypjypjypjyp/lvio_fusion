#ifndef lvio_fusion_VISUAL_FEATURE_H
#define lvio_fusion_VISUAL_FEATURE_H

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

typedef std::bitset<256> BRIEF;

class Frame;

namespace visual
{

class Landmark;

class Feature
{
public:
    typedef std::shared_ptr<Feature> Ptr;

    Feature() {}

    static Feature::Ptr Create(std::shared_ptr<Frame> frame, const cv::KeyPoint &keypoint, std::shared_ptr<Landmark> landmark = nullptr)
    {
        Feature::Ptr new_feature(new Feature);
        new_feature->frame = frame;
        new_feature->keypoint = keypoint;
        if (landmark)
        {
            new_feature->landmark = landmark;
        }
        return new_feature;
    }

    cv::KeyPoint keypoint;
    std::weak_ptr<Frame> frame;
    std::weak_ptr<Landmark> landmark;
    BRIEF brief;
    bool match = false;
    bool insert = false;
    bool is_on_left_image = true;
};

typedef std::map<unsigned long, Feature::Ptr> Features;
} // namespace visual

} // namespace lvio_fusion

#endif // lvio_fusion_VISUAL_FEATURE_H
