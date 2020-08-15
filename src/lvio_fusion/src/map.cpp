#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/feature.h"

namespace lvio_fusion
{

void Map::InsertKeyFrame(Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    Frame::current_frame_id++;
    current_frame = frame;
    keyframes_.insert(make_pair(frame->time, frame));
}

void Map::InsertLandmark(visual::Landmark::Ptr landmark)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    visual::Landmark::current_landmark_id++;
    landmarks_.insert(make_pair(landmark->id, landmark));
}

Keyframes Map::GetActiveKeyFrames(double time)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    return Keyframes(keyframes_.upper_bound(time), keyframes_.end());
}

void Map::RemoveLandmark(visual::Landmark::Ptr landmark)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    landmark->Clear();
    landmarks_.erase(landmark->id);
}

SE3d Map::ComputePose(double time)
{
    auto frame1 = keyframes_.lower_bound(time)->second;
    auto frame2 = keyframes_.upper_bound(time)->second;
    double d_t = time - frame1->time;
    double t_t = frame2->time - frame1->time;
    double s = d_t / t_t;
    Quaterniond q = frame1->pose.unit_quaternion().slerp(s, frame2->pose.unit_quaternion());
    Vector3d t = (1 - s) * frame1->pose.translation() + s * frame2->pose.translation();
    return SE3d(q, t);
}

} // namespace lvio_fusion
