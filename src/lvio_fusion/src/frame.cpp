#include "lvio_fusion/frame.h"

namespace lvio_fusion
{

Frame::Frame(long id, double time, const SE3 &pose, const cv::Mat &left_image, const cv::Mat &right_image)
    : id(id), time(time), pose(pose), left_image(left_image), right_image(right_image) {}

Frame::Ptr Frame::CreateFrame()
{
    static long factory_id = 0;
    Frame::Ptr new_frame(new Frame);
    new_frame->id = factory_id++;
    return new_frame;
}

void Frame::SetKeyFrame()
{
    static long keyframe_factory_id = 0;
    is_keyframe = true;
    keyframe_id = keyframe_factory_id++;
}

//NOTE:semantic map
LabelType Frame::GetLabelType(int x, int y)
{
    for(auto obj: objects)
    {
        if(obj.xmin<x&&obj.xmax>x&&obj.ymin<y&&obj.ymax>y)
        {
            return obj.label;
        }
    }
    return LabelType::None;
}

void Frame::UpdateLabel()
{
    for(auto feature: features_left)
    {
        auto map_point = feature->map_point.lock();
        if (map_point)
        {
            map_point->label = GetLabelType(feature->pos.pt.x, feature->pos.pt.y);
        }
    }
}

} // namespace lvio_fusion
