#include "lvio_fusion/frame.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/mappoint.h"

namespace lvio_fusion
{

Frame::Frame(long id, double time, const SE3 &pose, const cv::Mat &left_image, const cv::Mat &right_image)
    : time(time), pose_(pose), left_image(left_image), right_image(right_image) {}

Frame::Ptr Frame::CreateFrame()
{
    return Frame::Ptr(new Frame);
}

void Frame::SetKeyFrame()
{
    static long keyframe_factory_id = 0;
    id = keyframe_factory_id++;
}

void Frame::AddFeature(Feature::Ptr feature)
{
    if (feature->is_on_left_image)
    {
        left_features.push_back(feature);
    }
    else
    {
        right_features.push_back(feature);
    }
}

void Frame::RemoveFeature(Feature::Ptr feature)
{
    Features &features = left_features;
    if (!feature->is_on_left_image)
    {
        features = right_features;
    }
    for (auto it = features.begin(); it != features.end();)
    {
        if (*it == feature)
        {
            it = features.erase(it);
            return;
        }
        else
        {
            ++it;
        }
    }
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
    for (auto feature : left_features)
    {
        auto map_point = feature->mappoint.lock();
        map_point->label = GetLabelType(feature->keypoint.x, feature->keypoint.y);
    }
}

} // namespace lvio_fusion
