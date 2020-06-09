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
        first_frame = frame;
        empty = false;
    }
    if (keyframes_.find(frame->id) == keyframes_.end())
    {
        keyframes_.insert(make_pair(frame->id, frame));
        active_keyframes_.insert(make_pair(frame->id, frame));
    }
    else
    {
        keyframes_[frame->id] = frame;
        active_keyframes_[frame->id] = frame;
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
    // find the nearest and furthest frame of the current frame
    double max_dis = 0, min_dis = 9999;
    double max_kf_id = 0, min_kf_id = 0;
    auto Twc = current_frame->Pose().inverse();
    for (auto &kf : active_keyframes_)
    {
        if (kf.second == current_frame)
            continue;
        auto dis = (kf.second->Pose() * Twc).log().norm();
        if (dis > max_dis)
        {
            max_dis = dis;
            max_kf_id = kf.first;
        }
        if (dis < min_dis)
        {
            min_dis = dis;
            min_kf_id = kf.first;
        }
    }

    const double min_dis_th = 0.2;
    Frame::Ptr frame_to_remove = nullptr;
    if (min_dis < min_dis_th)
    {
        // if there is a near frame, remove the nearest first
        frame_to_remove = keyframes_.at(min_kf_id);
    }
    else
    {
        // remove the furthest
        frame_to_remove = keyframes_.at(max_kf_id);
    }

    LOG(INFO) << "remove keyframe " << frame_to_remove->id;

    int num_landmark_removed = 0;
    for (auto it = active_landmarks_.begin(); it != active_landmarks_.end();)
    {
        auto last_frame = it->second->FindLastFrame();
        if (last_frame)
        {
            if (last_frame->id <= frame_to_remove->id)
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
    active_keyframes_.erase(frame_to_remove->id);

    LOG(INFO) << "Removed " << num_landmark_removed << " active landmarks";
}

// freeze the current frame
Map::Landmarks Map::GetActiveMapPoints()
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    Landmarks landmarks = active_landmarks_;
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
Map::Keyframes Map::GetActiveKeyFrames()
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    Keyframes keyframes = active_keyframes_;
    int last_frame_id = current_frame->id - 1;
    for (auto it = keyframes.begin(); it != keyframes.end();)
        if (it->second->id > last_frame_id)
            it = keyframes.erase(it);
        else
            ++it;
    return keyframes;
}

Map::Params Map::GetPoseParams()
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    std::unordered_map<unsigned long, double *> para_Pose;
    for (auto keyframe : active_keyframes_)
    {
        para_Pose[keyframe.first] = keyframe.second->Pose().data();
    }
    return para_Pose;
}

Map::Params Map::GetPointParams()
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    std::unordered_map<unsigned long, double *> para_Point;
    for (auto landmark : active_landmarks_)
    {
        para_Point[landmark.first] = landmark.second->Position().data();
    }
    return para_Point;
}

} // namespace lvio_fusion
