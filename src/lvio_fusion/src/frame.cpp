#include "lvio_fusion/frame.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/landmark.h"
namespace lvio_fusion
{

unsigned long Frame::current_frame_id = 0;

Frame::Ptr Frame::Create()
{
    Frame::Ptr new_frame(new Frame);
    new_frame->id = current_frame_id + 1;
    new_frame->bImu=false;
    return new_frame;
}

void Frame::AddFeature(visual::Feature::Ptr feature)
{
    assert(feature->frame.lock()->id == id);
    if (feature->is_on_left_image)
    {
        features_left.insert(std::make_pair(feature->landmark.lock()->id, feature));
    }
    else
    {
        features_right.insert(std::make_pair(feature->landmark.lock()->id, feature));
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
    for (auto feature_pair : features_left)
    {
        auto camera_point = feature_pair.second->landmark.lock();
        camera_point->label = GetLabelType(feature_pair.second->keypoint.x, feature_pair.second->keypoint.y);
    }
}

//NEWADD
cv::Mat   Frame::GetImuRotation(){

     cv::Mat Rwc;
     cv::eigen2cv(pose.rotationMatrix(),Rwc);
     return Rwc*preintegration->calib.Tcb;

}

cv::Mat Frame::GetImuPosition()
{
    cv::Mat Tcw_;
    cv::eigen2cv(pose.matrix(),Tcw_);  
cv::Mat Rcw = Tcw_.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw_.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    cv::Mat Ow=Rwc*tcw;
cv::Mat TCB=preintegration->calib.Tcb;
     if (!TCB.empty())
        Owb = Rwc*TCB.rowRange(0,3).col(3)+Ow; //imu position
    return  Owb.clone();
}

void Frame::SetVelocity(const cv::Mat &Vw_)
{
    Vw_.copyTo(Vw);
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
void Frame::SetNewBias(const Bias &b)
{
    mImuBias = b;
    if(preintegration)
        preintegration->SetNewBias(b);
}
//NEWADDEND

} // namespace lvio_fusion
