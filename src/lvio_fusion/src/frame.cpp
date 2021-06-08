#include "lvio_fusion/frame.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/camera.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

unsigned long Frame::current_frame_id = 0;

Frame::Frame()
{
    weights.visual = 1;
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

void Frame::SetVelocity(const Vector3d &_Vw)
{
    Vw = _Vw;
}

void Frame::SetPose(const Matrix3d &_Rwb, const Vector3d &_twb)
{
    pose = SE3d(Quaterniond(_Rwb), _twb);
}

void Frame::SetBias(const Bias &_bias)
{
    bias = _bias;
    if (preintegration)
        preintegration->UpdateBias(bias);
}

Matrix3d Frame::R()
{
    return pose.rotationMatrix();
}

Vector3d Frame::t()
{
    return pose.translation();
}

} // namespace lvio_fusion
