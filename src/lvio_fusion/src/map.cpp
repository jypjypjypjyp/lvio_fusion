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

// 1: start
// 2: start -> end
// 2: start -> num
// 3: num -> end
Frames Map::GetKeyFrames(double start, double end, int num)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
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

//NEWADD   TODO
// 对地图数据应用相似变换
// R Rgw
void Map::ApplyScaledRotation(const cv::Mat &R, const float s, const bool bScaledVel, const cv::Mat t)
{
    // Step 1: 求解变换矩阵
    // Body position (IMU) of first keyframe is fixed to (0,0,0)
    // T world 2 gravity
    cv::Mat Txw = cv::Mat::eye(4,4,CV_32F);
    R.copyTo(Txw.rowRange(0,3).colRange(0,3));

    cv::Mat Tyx = cv::Mat::eye(4,4,CV_32F);

    // Tyw:world 2 correct 不带s的变换矩阵，将原始位姿转换到gravity矫正后的坐标系下
    cv::Mat Tyw = Tyx*Txw;
    Tyw.rowRange(0,3).col(3) = Tyw.rowRange(0,3).col(3)+t;
    cv::Mat Ryw = Tyw.rowRange(0,3).colRange(0,3);
    cv::Mat tyw = Tyw.rowRange(0,3).col(3);

    // Step 2: 变换KF的位姿到s，gravity矫正的坐标系下
    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(); sit!=mspKeyFrames.end(); sit++)
    {
        KeyFrame* pKF = *sit;
        cv::Mat Twc = pKF->GetPoseInverse();
        //对位姿的translation进行尺度放缩
        Twc.rowRange(0,3).col(3)*=s;
        // 对位姿进行相似变换
        cv::Mat Tyc = Tyw*Twc;  //此处Tyw = Tgw
        cv::Mat Tcy = cv::Mat::eye(4,4,CV_32F);
        Tcy.rowRange(0,3).colRange(0,3) = Tyc.rowRange(0,3).colRange(0,3).t();
        Tcy.rowRange(0,3).col(3) = -Tcy.rowRange(0,3).colRange(0,3)*Tyc.rowRange(0,3).col(3);
        pKF->SetPose(Tcy);
        cv::Mat Vw = pKF->GetVelocity();
        if(!bScaledVel)
            pKF->SetVelocity(Ryw*Vw);
        else
            pKF->SetVelocity(Ryw*Vw*s);

    }
    // Step 2: 对MapPoints进行相似变换
    // | sR t |
    // |  0 1 | x MapPoints
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(); sit!=mspMapPoints.end(); sit++)
    {
        MapPoint* pMP = *sit;
        pMP->SetWorldPos(s*Ryw*pMP->GetWorldPos()+tyw);
        pMP->UpdateNormalAndDepth();
    }
    mnMapChange++;
}
//NEWADDEND



} // namespace lvio_fusion
