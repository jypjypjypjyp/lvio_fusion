#include "lvio_fusion/map.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

void Map::InsertKeyFrame(Frame::Ptr frame)
{
    current_frame_ = frame;
    if (keyframes_.find(frame->keyframe_id_) == keyframes_.end())
    {
        keyframes_.insert(make_pair(frame->keyframe_id_, frame));
        active_keyframes_.insert(make_pair(frame->keyframe_id_, frame));
    }
    else
    {
        keyframes_[frame->keyframe_id_] = frame;
        active_keyframes_[frame->keyframe_id_] = frame;
    }

    if (active_keyframes_.size() > num_active_keyframes_)
    {
        RemoveOldKeyframe();
    }
}

void Map::InsertMapPoint(MapPoint::Ptr map_point)
{
    if (landmarks_.find(map_point->id_) == landmarks_.end())
    {
        landmarks_.insert(make_pair(map_point->id_, map_point));
        active_landmarks_.insert(make_pair(map_point->id_, map_point));
    }
    else
    {
        landmarks_[map_point->id_] = map_point;
        active_landmarks_[map_point->id_] = map_point;
    }
}

void Map::RemoveOldKeyframe()
{
    if (current_frame_ == nullptr)
        return;
    // find the nearest and furthest frame of the current frame
    double max_dis = 0, min_dis = 9999;
    double max_kf_id = 0, min_kf_id = 0;
    auto Twc = current_frame_->Pose().inverse();
    for (auto &kf : active_keyframes_)
    {
        if (kf.second == current_frame_)
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

LOG(INFO) << "remove keyframe " << frame_to_remove->keyframe_id_;
    // remove keyframe and landmark observation
    active_keyframes_.erase(frame_to_remove->keyframe_id_);
    for (auto feat : frame_to_remove->features_left_)
    {
        auto mp = feat->map_point_.lock();
        if (mp)
        {
            mp->RemoveObservation(feat);
        }
    }
    for (auto feat : frame_to_remove->features_right_)
    {
        if (feat == nullptr)
            continue;
        auto mp = feat->map_point_.lock();
        if (mp)
        {
            mp->RemoveObservation(feat);
        }
    }

    CleanMap();
}

Map::ParamsType Map::GetPoseParams()
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    for (auto keyframe : active_keyframes_)
    {
        para_Pose[keyframe.first] = keyframe.second->Pose().data();
    }
    return para_Pose;
}

Map::ParamsType Map::GetPointParams()
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    for (auto landmark : active_landmarks_)
    {
        para_Point[landmark.first] = landmark.second->Pos().data();
    }
    return para_Point;
}

void Map::CleanMap()
{
    int cnt_landmark_removed = 0;
    for (auto iter = active_landmarks_.begin();
         iter != active_landmarks_.end();)
    {
        if (iter->second->observed_times_ == 0)
        {
            iter = active_landmarks_.erase(iter);
            cnt_landmark_removed++;
        }
        else
        {
            ++iter;
        }
    }
    LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
}

} // namespace lvio_fusion
