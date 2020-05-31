#include "lvio_fusion/mappoint.h"
#include "lvio_fusion/feature.h"

namespace lvio_fusion
{

MapPoint::MapPoint(long id, Vector3d position) : id(id), pos_(position) {}

MapPoint::Ptr MapPoint::CreateNewMappoint()
{
    static long factory_id = 0;
    MapPoint::Ptr new_mappoint(new MapPoint);
    new_mappoint->id = factory_id++;
    return new_mappoint;
}

void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat)
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    for (auto iter = observations.begin(); iter != observations.end();
         iter++)
    {
        if (iter->lock() == feat)
        {
            observations.erase(iter);
            feat->map_point.reset();
            observed_times--;
            break;
        }
    }
}

} // namespace lvio_fusion
