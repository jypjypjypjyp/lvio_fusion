#include "lvio_fusion/frame.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/landmark.h"
#include <opencv2/core/eigen.hpp>
namespace lvio_fusion
{

unsigned long Frame::current_frame_id = 0;

Frame::Ptr Frame::Create()
{
    Frame::Ptr new_frame(new Frame);
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
//NEWADD
void Frame::SetVelocity(const Vector3d  &Vw_)
{
    Vw=Vw_;
}
void Frame::SetPose(const Matrix3d &Rwb_,const Vector3d  &twb_)
{
    Quaterniond R(Rwb_);
    pose=SE3d(R,twb_);
}

void Frame::SetNewBias(const Bias &bias_)
{
    ImuBias = bias_;
    if(preintegration)
        preintegration->SetNewBias(bias_);
}


Vector3d Frame::GetVelocity()
{
    return Vw;
}

Matrix3d  Frame::GetImuRotation()
{
    Matrix4d Twb_=pose.matrix();  
    Matrix3d Rwb = Twb_.block<3,3>(0,0);
    return Rwb;
}

Vector3d Frame::GetImuPosition()
{
    Matrix4d Twb_=pose.matrix();  
    Vector3d Owb =Twb_.block<3,1>(0,3); //imu position
    return  Owb;
}

Vector3d Frame::GetGyroBias()
{
    return ImuBias.linearized_bg;
}

Vector3d Frame::GetAccBias()
{
    return ImuBias.linearized_ba;
}
Bias Frame::GetImuBias()
{
    return ImuBias;
}

//NEWADDEND
} // namespace lvio_fusion
