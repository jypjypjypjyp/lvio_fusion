#include "lvio_fusion/visual/landmark.h"
#include "lvio_fusion/frame.h"

namespace lvio_fusion
{

namespace visual
{
unsigned long Landmark::current_landmark_id = 0;

Vector3d Landmark::ToWorld()
{
    return camera->Robot2World(position, FirstFrame().lock()->pose);
}

visual::Landmark::Ptr Landmark::Create(Vector3d position, Camera::Ptr camera)
{
    visual::Landmark::Ptr new_point(new Landmark);
    new_point->position = position;
    new_point->camera = camera;
    return new_point;
}

void Landmark::Clear()
{
    for (auto pair_feature : observations)
    {
        auto feature = pair_feature.second;
        feature->frame.lock()->features_left.erase(id);
    }
    auto right_feature = first_observation;
    right_feature->frame.lock()->features_right.erase(id);
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
