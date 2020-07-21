#include "lvio_fusion/mappoint.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{

MapPoint::Ptr MapPoint::Create(double depth, Sensord::Ptr sensor)
{
    MapPoint::Ptr new_mappoint(new MapPoint);
    new_mappoint->id = Map::current_mappoint_id + 1;
    new_mappoint->depth = depth;
    new_mappoint->sensor = sensor;
    return new_mappoint;
}

Vector3d MapPoint::Position()
{
    Vector2d kp = observations.begin()->second->keypoint;
    return sensor->Pixel2World(kp, FindFirstFrame()->pose, depth);
}

Frame::Ptr MapPoint::FindFirstFrame()
{
    return observations.begin()->second->frame.lock();
}

Frame::Ptr MapPoint::FindLastFrame()
{
    return (--observations.end())->second->frame.lock();
}

void MapPoint::AddObservation(Feature::Ptr feature)
{
    assert(feature->mappoint.lock()->id == id);
    if (feature->is_on_left_image)
    {
        observations.insert(std::make_pair(feature->frame.lock()->id, feature));
    }
    else
    {
        assert(feature->frame.lock()->id == observations.begin()->first);
        init_observation = feature;
    }
}

void MapPoint::RemoveObservation(Feature::Ptr feature)
{
    assert(feature->is_on_left_image && feature != observations.begin()->second);
    observations.erase(feature->frame.lock()->id);
}

} // namespace lvio_fusion
