#include "lvio_fusion/map.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

void Map::InsertKeyFrame(Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(map_mutex_);
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
    std::unique_lock<std::mutex> lock(map_mutex_);
    landmarks_.insert(make_pair(mappoint->id, mappoint));
    active_landmarks_.insert(make_pair(mappoint->id, mappoint));
}

void Map::RemoveOldKeyframe()
{
    if (current_frame == nullptr)
        return;
    // find the oldest frame of the current frame
    Frame::Ptr frame_to_remove = active_keyframes_.begin()->second;
    LOG(INFO) << "remove keyframe " << frame_to_remove->id;

    int num_landmark_removed = 0;
    for (auto it = active_landmarks_.begin(); it != active_landmarks_.end();)
    {
        auto last_frame = it->second->FindLastFrame();
        if (last_frame)
        {
            if (last_frame->time <= frame_to_remove->time)
            {
                it = active_landmarks_.erase(it);
                num_landmark_removed++;
            }
            else
                ++it;
        }
        else
        {
            landmarks_.erase(it->first);
            it = active_landmarks_.erase(it);
            num_landmark_removed++;
        }
    }
    active_keyframes_.erase(frame_to_remove->time);

    LOG(INFO) << "Removed " << num_landmark_removed << " active landmarks";
}

// freeze related mappoints of the current frame
Map::Landmarks Map::GetActiveMapPoints(bool full)
{
    std::unique_lock<std::mutex> lock(map_mutex_);
    Landmarks landmarks = full ? landmarks_ : active_landmarks_;
    int current_frame_time = current_frame->time;
    for (auto it = landmarks.begin(); it != landmarks.end();)
    {
        Frame::Ptr last_frame = it->second->FindLastFrame();
        if (last_frame == nullptr || last_frame->time >= current_frame_time)
            it = landmarks.erase(it);
        else
            ++it;
    }
    return landmarks;
}

// freeze the current frame
Map::Keyframes Map::GetActiveKeyFrames(bool full)
{
    std::unique_lock<std::mutex> lock(map_mutex_);
    Keyframes keyframes = full ? keyframes_ : active_keyframes_;
    keyframes.erase(current_frame->time);
    return keyframes;
}

void Map::RemoveMapPoint(MapPoint::Ptr mappoint)
{
    std::unique_lock<std::mutex> lock(map_mutex_);
    for (auto feature: mappoint->observations)
    {
        feature->frame.lock()->RemoveFeature(feature);
    }
    landmarks_.erase(mappoint->id);
    active_landmarks_.erase(mappoint->id);
}

} // namespace lvio_fusion
