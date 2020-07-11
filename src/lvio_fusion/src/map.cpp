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
    active_keyframes_.insert(make_pair(frame->time, frame));
    if (active_keyframes_.size() > WINDOW_SIZE)
    {
        RemoveOldKeyframe();
    }
}

void Map::InsertMapPoint(MapPoint::Ptr mappoint)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    current_mappoint_id++;
    landmarks_.insert(make_pair(mappoint->id, mappoint));
}

void Map::RemoveOldKeyframe()
{
    LOG(INFO) << "remove keyframe " << active_keyframes_.begin()->second->id;
    active_keyframes_.erase(active_keyframes_.begin());
}

// freeze the current frame
Map::Keyframes Map::GetActiveKeyFrames(bool full)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    Keyframes keyframes = full ? keyframes_ : active_keyframes_;
    keyframes.erase(current_frame->time);
    return keyframes;
}

void Map::RemoveMapPoint(MapPoint::Ptr mappoint)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    for (auto feature_pair : mappoint->observations)
    {
        auto feature = feature_pair.second;
        feature->frame.lock()->left_features.erase(mappoint->id);
    }
    auto right_feature = mappoint->right_observation;
    right_feature->frame.lock()->right_features.erase(mappoint->id);
    landmarks_.erase(mappoint->id);
}

} // namespace lvio_fusion
