#include "lvio_fusion/frame.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/camera.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

unsigned long Frame::current_frame_id = 0;

Frame::Frame()
{
    weights.visual = Camera::Get()->fx / 1.5;
    weights.lidar_ground = 1;
    weights.lidar_surf = 0.01;
}

Frame::Ptr Frame::Create()
{
    Frame::Ptr new_frame(new Frame);
    new_frame->id = current_frame_id + 1;
    return new_frame;
}

void Frame::AddFeature(visual::Feature::Ptr feature)
{
    auto landmark = feature->landmark.lock();
    assert(feature->frame.lock()->id == id && landmark);
    if (feature->is_on_left_image)
    {
        features_left[landmark->id] = feature;
    }
    else
    {
        features_right[landmark->id] = feature;
    }
}

void Frame::RemoveFeature(visual::Feature::Ptr feature)
{
    assert(feature->is_on_left_image && id != feature->landmark.lock()->FirstFrame().lock()->id);
    int a = features_left.erase(feature->landmark.lock()->id);
}

Observation Frame::GetObservation()
{
    assert(last_keyframe);
    static int obs_rows = 4, obs_cols = 12;
    cv::Mat obs = cv::Mat::zeros(obs_rows, obs_cols, CV_32FC3);
    int height = image_left.rows, width = image_left.cols;
    for (auto &pair_feature : features_left)
    {
        auto observations = pair_feature.second->landmark.lock()->observations;
        if (observations.find(id - 1) != observations.end())
        {
            auto pt = pair_feature.second->keypoint.pt;
            auto prev_pt = observations[id - 1]->keypoint.pt;
            int row = (int)(pt.y / (height / obs_rows));
            int col = (int)(pt.x / (width / obs_cols));
            obs.at<cv::Vec3f>(row, col)[0] += 1;
            obs.at<cv::Vec3f>(row, col)[1] += pt.x - prev_pt.x;
            obs.at<cv::Vec3f>(row, col)[2] += pt.y - prev_pt.y;
        }
    }

    for (auto iter = obs.begin<cv::Vec3f>(); iter != obs.end<cv::Vec3f>(); iter++)
    {
        float n = std::max(1.f, (*iter)[0]);
        (*iter)[1] = (*iter)[1] / n;
        (*iter)[2] = (*iter)[2] / n;
    }

    return obs.reshape(1, 1);
}

void Frame::SetVelocity(const Vector3d &Vw_)
{
    Vw = Vw_;
}

void Frame::SetPose(const Matrix3d &Rwb_, const Vector3d &twb_)
{
    Quaterniond R(Rwb_);
    pose = SE3d(R, twb_);
}

void Frame::SetImuBias(const Bias &bias_)
{
    ImuBias = bias_;
    if (preintegration)
        preintegration->SetNewBias(bias_);
}

Vector3d Frame::GetVelocity()
{
    return Vw;
}

Matrix3d Frame::GetImuRotation()
{
    Matrix4d Twb_ = pose.matrix();
    Matrix3d Rwb = Twb_.block<3, 3>(0, 0);
    return Rwb;
}

Vector3d Frame::GetImuPosition()
{
    Matrix4d Twb_ = pose.matrix();
    Vector3d Owb = Twb_.block<3, 1>(0, 3); //imu position
    return Owb;
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

} // namespace lvio_fusion
