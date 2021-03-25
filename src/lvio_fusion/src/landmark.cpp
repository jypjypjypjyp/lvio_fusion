
#include "lvio_fusion/visual/landmark.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/camera.h"

namespace lvio_fusion
{

namespace visual
{
unsigned long Landmark::current_landmark_id = 0;

Vector3d Landmark::ToWorld()
{
    Vector3d pb = Camera::Get(1)->Pixel2Robot(cv2eigen(first_observation->keypoint), depth);
    return Camera::Get()->Robot2World(pb, FirstFrame().lock()->pose);
}

visual::Landmark::Ptr Landmark::Create(double depth)
{
    visual::Landmark::Ptr new_point(new Landmark);
    new_point->depth = depth;
    return new_point;
}

void Landmark::Clear()
{
    for (auto &pair_feature : observations)
    {
        auto a = pair_feature.second->frame.lock();
        pair_feature.second->frame.lock()->features_left.erase(id);
    }
    auto right_feature = first_observation;
    right_feature->frame.lock()->features_right.erase(id);

    int num = 0;
    Frames a = Map::Instance().GetKeyFrames(FirstFrame().lock()->time);
    for (auto &i : a)
    {
        if (i.second->features_left.find(id) != i.second->features_left.end())
        {
            num++;
        }
    }
    assert(num == 0);
}

std::weak_ptr<Frame> Landmark::FirstFrame()
{
    auto frame = first_observation->frame;
    assert(!frame.expired());
    return frame;
}

std::weak_ptr<Frame> Landmark::LastFrame()
{
    auto frame = (--observations.end())->second->frame;
    assert(!frame.expired());
    return frame;
}

void Landmark::AddObservation(visual::Feature::Ptr feature)
{
    assert(feature->landmark.lock()->id == id);
    if (feature->is_on_left_image)
    {
        observations[feature->frame.lock()->id] = feature;
    }
    else
    {
        assert(feature->frame.lock()->id == observations.begin()->first);
        first_observation = feature;
    }
}

void Landmark::RemoveObservation(visual::Feature::Ptr feature)
{
    assert(feature->is_on_left_image && feature != observations.begin()->second);
    observations.erase(feature->frame.lock()->id);
}
} // namespace visual

} // namespace lvio_fusion
