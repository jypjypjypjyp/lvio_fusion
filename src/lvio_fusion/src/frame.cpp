#include "lvio_fusion/frame.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

unsigned long Frame::current_frame_id = 0;

Frame::Ptr Frame::Create()
{
    Frame::Ptr new_frame(new Frame());
    new_frame->id = current_frame_id + 1;
    return new_frame;
}

void Frame::AddFeature(visual::Feature::Ptr feature)
{
    assert(feature->frame.lock()->id == id);
    if (feature->is_on_left_image)
    {
        features_left[feature->landmark.lock()->id] = feature;
    }
    else
    {
        features_right[feature->landmark.lock()->id] = feature;
    }
}

void Frame::RemoveFeature(visual::Feature::Ptr feature)
{
    assert(feature->is_on_left_image && id != feature->landmark.lock()->FirstFrame().lock()->id);
    features_left.erase(feature->landmark.lock()->id);
}

//NOTE:semantic map
LabelType Frame::GetLabelType(int x, int y)
{
    for (auto obj : objects)
    {
        if (obj.xmin < x && obj.xmax > x && obj.ymin < y && obj.ymax > y)
        {
            return obj.label;
        }
    }
    return LabelType::None;
}

void Frame::UpdateLabel()
{
    for (auto pair_feature : features_left)
    {
        auto camera_point = pair_feature.second->landmark.lock();
        camera_point->label = GetLabelType(pair_feature.second->keypoint.x, pair_feature.second->keypoint.y);
    }
}
void Frame::SetVelocity(const cv::Mat &Vw_)
{
    cv::cv2eigen(Vw_,mVw);
}

void Frame::SetNewBias(const Bias &b)
{
    mImuBias = b;
    if(preintegration)
        preintegration->SetNewBias(b);
}

void Frame::SetPose(const cv::Mat &Tcw_)
{
     cv::Mat Rcw = Tcw_.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw_.rowRange(0,3).col(3);
    Eigen::Matrix3d R ;
    cv::cv2eigen(Rcw,R);
    Eigen::Vector3d t(tcw.at<double>(0,0),tcw.at<double>(1,0),tcw.at<double>(2,0));  
    pose=SE3d(R,t);
}

cv::Mat Frame::GetVelocity()
{
    cv::Mat Vw_;
    cv::eigen2cv(mVw,Vw_);
    return Vw_;
}

cv::Mat   Frame::GetImuRotation()
{
     cv::Mat Rwc;
     cv::eigen2cv(pose.rotationMatrix(),Rwc);
    return Rwc*calib_.Tcb.rowRange(0,3).colRange(0,3);
}

cv::Mat Frame::GetImuPosition()
{
    cv::Mat Tcw_;
    cv::eigen2cv(pose.matrix(),Tcw_);  
cv::Mat Rcw = Tcw_.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw_.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    cv::Mat Ow=Rwc*tcw;
cv::Mat TCB=calib_.Tcb;
     if (!TCB.empty())
        Owb = Rwc*TCB.rowRange(0,3).col(3)+Ow; //imu position
    return  Owb.clone();
}

cv::Mat Frame::GetGyroBias()
{
    return (cv::Mat_<float>(3,1) << mImuBias.bwx, mImuBias.bwy, mImuBias.bwz);
}

cv::Mat Frame::GetAccBias()
{
    
      return (cv::Mat_<float>(3,1) << mImuBias.bax, mImuBias.bay, mImuBias.baz);
}
Bias Frame::GetImuBias()
{
    return mImuBias;
}
cv::Mat Frame::GetPoseInverse()
{
     cv::Mat Tcw_;
    cv::eigen2cv(pose.matrix(),Tcw_);  
    cv::Mat Rcw = Tcw_.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw_.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    cv::Mat Ow=Rwc*tcw;


    cv::Mat Twc = cv::Mat::eye(4,4,Tcw_.type());
   Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    return Twc.clone();
}

void Frame::SetImuPoseVelocity(const cv::Mat &Rwb, const cv::Mat &twb, const cv::Mat &Vwb)
{
    cv::cv2eigen(Vwb,mVw);
    cv::Mat Rbw = Rwb.t();
    cv::Mat tbw = -Rbw*twb;
    cv::Mat Tbw = cv::Mat::eye(4,4,CV_32F);
    Rbw.copyTo(Tbw.rowRange(0,3).colRange(0,3));
    tbw.copyTo(Tbw.rowRange(0,3).col(3));
    mTcw = calib_.Tcb*Tbw;
    UpdatePoseMatrices();
}
void Frame::UpdatePoseMatrices()
{
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}
} // namespace lvio_fusion
