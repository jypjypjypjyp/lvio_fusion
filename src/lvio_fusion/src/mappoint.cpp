#include "lvio_fusion/mappoint.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{

MapPoint::Ptr MapPoint::CreateNewMappoint(Vector3d position)
{
    MapPoint::Ptr new_mappoint(new MapPoint);
    new_mappoint->id = Map::current_mappoint_id + 1;
    new_mappoint->position = position;
    return new_mappoint;
}

Frame::Ptr MapPoint::FindFirstFrame()
{
    return right_observation->frame.lock();
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
        right_observation = feature;
    }
}

void MapPoint::RemoveObservation(Feature::Ptr feature)
{
    assert(feature->is_on_left_image && feature != observations.begin()->second);
    observations.erase(feature->frame.lock()->id);
}

} // namespace lvio_fusion
