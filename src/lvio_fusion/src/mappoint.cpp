#include "lvio_fusion/mappoint.h"
#include "lvio_fusion/feature.h"

namespace lvio_fusion
{

MapPoint::Ptr MapPoint::CreateNewMappoint(Vector3d position)
{
    static long factory_id = 0;
    MapPoint::Ptr new_mappoint(new MapPoint);
    new_mappoint->id = factory_id++;
    new_mappoint->position_ = position;
    return new_mappoint;
}

Frame::Ptr MapPoint::FindFirstFrame()
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    if (observations.empty())
        return nullptr;
    return observations.front()->frame.lock();
}

Frame::Ptr MapPoint::FindLastFrame()
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    if (observations.empty())
        return nullptr;
    return observations.back()->frame.lock();
}

void MapPoint::AddObservation(Feature::Ptr feature)
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    observations.push_back(feature);
}

void MapPoint::RemoveObservation(Feature::Ptr feature)
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    for (auto it = observations.begin(); it != observations.end();)
    {
        if (*it == feature)
        {
            it = observations.erase(it);
            return;
        }
        else
        {
            ++it;
        }
    }
}

} // namespace lvio_fusion
