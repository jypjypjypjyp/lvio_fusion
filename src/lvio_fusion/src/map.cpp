#include "lvio_fusion/map.h"
#include "lvio_fusion/feature.h"

namespace lvio_fusion
{

unsigned long Map::current_frame_id = 0;
unsigned long Map::current_mappoint_id = 0;

void Map::InsertKeyFrame(Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    current_frame_id++;
    current_frame = frame;
    keyframes_.insert(make_pair(frame->time, frame));
}

void Map::InsertMapPoint(MapPoint::Ptr mappoint)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    current_mappoint_id++;
    landmarks_.insert(make_pair(mappoint->id, mappoint));
}

Map::Keyframes Map::GetActiveKeyFrames(double time)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    return Keyframes(keyframes_.upper_bound(time), keyframes_.end());
} 

void Map::RemoveMapPoint(MapPoint::Ptr mappoint)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    for (auto feature_pair : mappoint->observations)
    {
        auto feature = feature_pair.second;
        feature->frame.lock()->features_left.erase(mappoint->id);
    }
    auto right_feature = mappoint->first_observation;
    right_feature->frame.lock()->features_right.erase(mappoint->id);
    landmarks_.erase(mappoint->id);
}

} // namespace lvio_fusion
