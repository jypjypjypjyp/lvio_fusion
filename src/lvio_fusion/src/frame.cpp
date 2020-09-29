#include "lvio_fusion/frame.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

unsigned long Frame::current_frame_id = 0;

Frame::Ptr Frame::Create()
{
    Frame::Ptr new_frame(new Frame);
    new_frame->id = current_frame_id + 1;
    return new_frame;
}

void Frame::AddFeature(visual::Feature::Ptr feature)
{
    assert(feature->frame.lock()->id == id);
    if (feature->is_on_left_image)
    {
        features_left.insert(std::make_pair(feature->landmark.lock()->id, feature));
    }
    else
    {
        features_right.insert(std::make_pair(feature->landmark.lock()->id, feature));
    }
}

void Frame::RemoveFeature(visual::Feature::Ptr feature)
{
    assert(feature->is_on_left_image && id != feature->landmark.lock()->FirstFrame().lock()->id);
    features_left.erase(feature->landmark.lock()->id);
}

//NOTE:semantic map
LabelType Frame::GetLabelType(int x, int y)
{
    for (auto obj : objects)
    {
        if (obj.xmin < x && obj.xmax > x && obj.ymin < y && obj.ymax > y)
        {
            return obj.label;
        }
    }
    return LabelType::None;
}

void Frame::UpdateLabel()
{
    for (auto feature_pair : features_left)
    {
        auto camera_point = feature_pair.second->landmark.lock();
        camera_point->label = GetLabelType(feature_pair.second->keypoint.x, feature_pair.second->keypoint.y);
    }
}

} // namespace lvio_fusion
