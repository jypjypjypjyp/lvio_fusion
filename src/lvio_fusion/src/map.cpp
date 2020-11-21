#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/feature.h"

namespace lvio_fusion
{

void Map::InsertKeyFrame(Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(mutex_local_kfs);
    Frame::current_frame_id++;
     current_frame = frame;
    keyframes_[frame->time] = frame;
}

void Map::InsertLandmark(visual::Landmark::Ptr landmark)
{
    std::unique_lock<std::mutex> lock(mutex_local_kfs);
    visual::Landmark::current_landmark_id++;
    landmarks_[landmark->id] = landmark;
}

// 1: [start]
// 2: [start -> end]
// 3: (start -> num]
// 4: [num -> end)
Frames Map::GetKeyFrames(double start, double end, int num)
{
    if (end == 0 && num == 0)
    {
        auto start_iter = keyframes_.lower_bound(start);
        return start_iter == keyframes_.end() ? Frames() : Frames(start_iter, keyframes_.end());
    }
    else if (num == 0)
    {
        auto start_iter = keyframes_.lower_bound(start);
        auto end_iter = keyframes_.upper_bound(end);
        return start >= end ? Frames() : Frames(start_iter, end_iter);
    }
    else if (end == 0)
    {
        auto iter = keyframes_.upper_bound(start);
        Frames frames;
        for (size_t i = 0; i < num && iter != keyframes_.end(); i++)
        {
            frames.insert(*(iter++));
        }
        return frames;
    }
    else if (start == 0)
    {
        auto iter = keyframes_.lower_bound(end);
        Frames frames;
        for (size_t i = 0; i < num && iter != keyframes_.begin(); i++)
        {
            frames.insert(*(--iter));
        }
        return frames;
    }
    return Frames();
}

void Map::RemoveLandmark(visual::Landmark::Ptr landmark)
{
    std::unique_lock<std::mutex> lock(mutex_local_kfs);
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

// 对地图数据应用相似变换
// R Rgw
void Map::ApplyScaledRotation(const Matrix3d &R, const double s)
{
    // Step 1: 求解变换矩阵
    // Body position (IMU) of first keyframe is fixed to (0,0,0)
    // T world 2 gravity
    Matrix4d Txw = Matrix4d::Identity();
    Txw.block<3,3>(0,0)=R;

    Matrix4d Tyx = Matrix4d::Identity();

    // Tyw:world 2 correct 不带s的变换矩阵，将原始位姿转换到gravity矫正后的坐标系下
    Matrix4d Tyw = Tyx*Txw;
    Tyw.block<3,1>(0,3)= Tyw.block<3,1>(0,3);
    Matrix3d Ryw = Tyw.block<3,3>(0,0);
    Vector3d tyw = Tyw.block<3,1>(0,3);

    // Step 2: 变换KF的位姿到s，gravity矫正的坐标系下
    for(std::map<double, Frame::Ptr>::iterator sit=keyframes_.begin(); sit!=keyframes_.end(); sit++)
    {
        Frame::Ptr pKF = (*sit).second;
        Matrix4d Twc = pKF->GetPoseInverse();
        //对位姿的translation进行尺度放缩
        Twc.block<3,1>(0,3)*=s;
        // 对位姿进行相似变换
       Matrix4d Tyc = Tyw*Twc;  //此处Tyw = Tgw
        Matrix3d Rcy = Tyc.block<3,3>(0,0).transpose();
        Vector3d tcy = -Rcy*Tyc.block<3,1>(0,3);
        pKF->SetPose(Rcy,tcy);
        Vector3d Vw = pKF->GetVelocity();
        pKF->SetVelocity(R*Vw*s);

    }
    // Step 2: 对MapPoints进行相似变换
    // | sR t |
    // |  0 1 | x MapPoints
    for(visual::Landmarks::iterator sit=landmarks_.begin(); sit!=landmarks_.end(); sit++)
    {
       visual::Landmark::Ptr pMP = (*sit).second;
      Vector3d pos = pMP->position;
      pMP->position= s*Ryw*pos+tyw;
    }
}
} // namespace lvio_fusion
