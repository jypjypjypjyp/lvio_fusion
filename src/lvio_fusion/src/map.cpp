#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/feature.h"

namespace lvio_fusion
{

void Map::InsertKeyFrame(Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(mutex_data_);
    Frame::current_frame_id++;
    keyframes_[frame->time] = frame;
}

void Map::InsertLandmark(visual::Landmark::Ptr landmark)
{
    std::unique_lock<std::mutex> lock(mutex_data_);
    visual::Landmark::current_landmark_id++;
    landmarks_[landmark->id] = landmark;
}

// 1: (start]
// 2: (start -> end]
// 2: (start -> num]
// 3: (num -> end]
Frames Map::GetKeyFrames(double start, double end, int num)
{
    std::unique_lock<std::mutex> lock(mutex_data_);
    if (end == 0 && num == 0)
    {
        return Frames(keyframes_.upper_bound(start), keyframes_.end());
    }
    else if (num == 0)
    {
        return Frames(keyframes_.upper_bound(start), keyframes_.lower_bound(end));
    }
    else if (end == 0)
    {
        auto iter = keyframes_.upper_bound(start);
        Frames frames;
        for (size_t i = 0; i < num; i++)
        {
            frames.insert(*iter);
            iter++;
        }
        return frames;
    }
    else if (start == 0)
    {
        auto iter = keyframes_.lower_bound(end);
        Frames frames;
        for (size_t i = 0; i < num && iter != --keyframes_.begin(); i++)
        {
            frames.insert(*iter);
            iter--;
        }
        return frames;
    }
    return Frames();
}

void Map::RemoveLandmark(visual::Landmark::Ptr landmark)
{
    std::unique_lock<std::mutex> lock(mutex_data_);
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
