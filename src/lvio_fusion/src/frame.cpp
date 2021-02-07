#include "lvio_fusion/frame.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

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
    int a = features_left.erase(feature->landmark.lock()->id);
}

//NOTE:semantic map
LabelType Frame::GetLabelType(int x, int y)
{
    for (auto &obj : objects)
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
    for (auto &pair_feature : features_left)
    {
        auto landmark = pair_feature.second->landmark.lock();
        landmark->label = GetLabelType(pair_feature.second->keypoint.x, pair_feature.second->keypoint.y);
    }
}

Observation Frame::GetObservation()
{
    if (Map::Instance().keyframes.find(id - 1) == Map::Instance().keyframes.end())
        return Observation();
        
    static int obs_rows = 4, obs_cols = 12;
    cv::Mat obs = cv::Mat::zeros(obs_rows, obs_cols, CV_32FC3);
    int height = image_left.rows, width = image_left.cols;
    for (auto &pair_feature : features_left)
    {
        auto observations = pair_feature.second->landmark.lock()->observations;
        if (observations.find(id - 1) != observations.end())
        {
            auto pt = pair_feature.second->keypoint;
            auto next_pt = observations[id + 1]->keypoint;
            int row = (int)(pt.y / (height / obs_rows));
            int col = (int)(pt.x / (width / obs_cols));
            obs.at<cv::Vec3f>(row, col)[0] += 1;
            obs.at<cv::Vec3f>(row, col)[1] += next_pt.x - pt.x;
            obs.at<cv::Vec3f>(row, col)[2] += next_pt.y - pt.y;
        }
    }

    for (auto iter = obs.begin<cv::Vec3f>(); iter != obs.end<cv::Vec3f>(); iter++)
    {
        float n = std::max(1.f, (*iter)[0]);
        (*iter)[1] = (*iter)[1] / n;
        (*iter)[2] = (*iter)[2] / n;
    }

    return obs;
}

} // namespace lvio_fusion
