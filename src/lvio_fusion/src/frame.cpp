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
void Frame::SetVelocity(const Vector3d  &Vw_)
{
    mVw=Vw_;
}

void Frame::SetNewBias(const Bias &b)
{
    mImuBias = b;
    if(preintegration)
        preintegration->SetNewBias(b);
}

void Frame::SetPose(const Matrix3d Rcw,const Vector3d tcw)
{
    pose=SE3d(Rcw,tcw);
}

Vector3d Frame::GetVelocity()
{
    return mVw;
}

Matrix3d  Frame::GetImuRotation()
{
    Matrix4d Tcw_=pose.matrix();  
    Matrix3d Rcw = Tcw_.block<3,3>(0,0);
    Matrix3d Rwc = Rcw.transpose();
    return Rwc;
}

Vector3d Frame::GetImuPosition()
{
    Matrix4d Tcw_=pose.matrix();  
    Matrix3d Rcw = Tcw_.block<3,3>(0,0);
    Vector3d tcw = Tcw_.block<3,1>(0,3);
    Matrix3d Rwc = Rcw.transpose();
    Vector3d Ow=Rwc*tcw;
    Owb =Ow; //imu position
    return  Owb;
}

Vector3d Frame::GetGyroBias()
{
    return mImuBias.linearized_bg;
}

Vector3d Frame::GetAccBias()
{
    return mImuBias.linearized_ba;
}
Bias Frame::GetImuBias()
{
    return mImuBias;
}
Matrix4d Frame::GetPoseInverse()
{
    Matrix4d Tcw_=pose.matrix();  
    Matrix3d Rcw = Tcw_.block<3,3>(0,0);
    Vector3d tcw = Tcw_.block<3,1>(0,3);
    Matrix3d Rwc = Rcw.transpose();
    Vector3d Ow=Rwc*tcw;

    Matrix4d  Twc =  Matrix4d ::Identity();
    Twc.block<3,3>(0,0)=Rwc;
    Twc.block<3,1>(0,3)=Ow;
    return Twc;
}

void Frame::SetImuPoseVelocity(const Matrix3d &Rwb,const Vector3d &twb, const Vector3d &Vwb)
{
    mVw=Vwb;
    Matrix3d Rbw = Rwb.transpose();
    Vector3d tbw = -Rbw*twb;
    Matrix4d Tbw = Matrix4d::Identity();
    Tbw.block<3,3>(0,0)=Rbw;
    Tbw.block<3,1>(0,3)=tbw;
    //mTcw = calib_.Tcb*Tbw;
   // UpdatePoseMatrices();
}
void Frame::UpdatePoseMatrices()
{
    mRcw = mTcw.block<3,3>(0,0);
    mRwc = mRcw.transpose();
    mtcw = mTcw.block<3,1>(0,3);
    mOw = -mRcw.transpose()*mtcw;
}
} // namespace lvio_fusion
