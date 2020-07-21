#include "lvio_fusion/frame.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/mappoint.h"

namespace lvio_fusion
{

Frame::Ptr Frame::Create()
{
    Frame::Ptr new_frame(new Frame);
    new_frame->id = Map::current_frame_id + 1;
    return new_frame;
}

void Frame::AddFeature(Feature::Ptr feature)
{
    assert(feature->frame.lock()->id == id);
    if (feature->is_on_left_image)
    {
        left_features.insert(std::make_pair(feature->mappoint.lock()->id, feature));
    }
    else
    {
        right_features.insert(std::make_pair(feature->mappoint.lock()->id, feature));
    }
}

void Frame::RemoveFeature(Feature::Ptr feature)
{
    assert(feature->is_on_left_image && id != feature->mappoint.lock()->FindFirstFrame()->id);
    left_features.erase(feature->mappoint.lock()->id);
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
    for (auto feature_pair : left_features)
    {
        auto mappoint = feature_pair.second->mappoint.lock();
        mappoint->label = GetLabelType(feature_pair.second->keypoint.x(), feature_pair.second->keypoint.y());
    }
}

} // namespace lvio_fusion
