#include "lvio_fusion/map.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

void Map::InsertKeyFrame(Frame::Ptr frame)
{
    current_frame = frame;
    if (empty)
    {
        empty = false;
    }
    if (keyframes_.find(frame->time) == keyframes_.end())
    {
        keyframes_.insert(make_pair(frame->time, frame));
        active_keyframes_.insert(make_pair(frame->time, frame));
    }
    else
    {
        keyframes_[frame->time] = frame;
        active_keyframes_[frame->time] = frame;
    }

    if (active_keyframes_.size() > WINDOW_SIZE)
    {
        RemoveOldKeyframe();
    }
}

void Map::InsertMapPoint(MapPoint::Ptr map_point)
{
    if (landmarks_.find(map_point->id) == landmarks_.end())
    {
        landmarks_.insert(make_pair(map_point->id, map_point));
        active_landmarks_.insert(make_pair(map_point->id, map_point));
    }
    else
    {
        landmarks_[map_point->id] = map_point;
        active_landmarks_[map_point->id] = map_point;
    }
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

// freeze the current frame
Map::Landmarks Map::GetActiveMapPoints(bool full)
{
    Landmarks landmarks = full ? landmarks_ : active_landmarks_;
    int last_frame_id = current_frame->id - 1;
    for (auto it = landmarks.begin(); it != landmarks.end();)
    {
        Frame::Ptr last_frame = it->second->FindLastFrame();
        if (last_frame == nullptr || last_frame->id > last_frame_id)
            it = landmarks.erase(it);
        else
            ++it;
    }
    return landmarks;
}

// freeze related mappoints of the current frame
Map::Keyframes Map::GetActiveKeyFrames(bool full)
{
    Keyframes keyframes = full ? keyframes_ : active_keyframes_;
    int last_frame_id = current_frame->id - 1;
    for (auto it = keyframes.begin(); it != keyframes.end();)
        if (it->second->id > last_frame_id)
            it = keyframes.erase(it);
        else
            ++it;
    return keyframes;
}

Map::Params Map::GetPoseParams(bool full)
{
    Params para_Pose;
    Keyframes& keyframes = full ? keyframes_ : active_keyframes_;
    for (auto keyframe : keyframes)
    {
        para_Pose[keyframe.second->id] = keyframe.second->pose.data();
    }
    return para_Pose;
}

Map::Params Map::GetPointParams(bool full)
{
    Params para_Point;
    Landmarks& landmarks = full ? landmarks_ : active_landmarks_;
    for (auto landmark : landmarks)
    {
        para_Point[landmark.first] = landmark.second->position.data();
    }
    return para_Point;
}

} // namespace lvio_fusion
