#include "lvio_fusion/visual/matcher.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

cv::Mat ORBMatcher::ComputeBRIEF(std::vector<cv::KeyPoint>)
{
    // compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    for (auto &pair_feature : frame->features_left)
    {
        keypoints.push_back(cv::KeyPoint(pair_feature.second->keypoint, 1));
    }
    cv::Mat descriptors;
    detector_->compute(frame->image_left, keypoints, descriptors);

    // NOTE: detector_->compute maybe remove some row because its descriptor cannot be computed
    int j = 0, i = 0;
    frame->descriptors = cv::Mat::zeros(frame->features_left.size(), 32, CV_8U);
    for (auto &pair_feature : frame->features_left)
    {
        if (pair_feature.second->keypoint == keypoints[j].pt && j < descriptors.rows)
        {
            descriptors.row(j).copyTo(frame->descriptors.row(i));
            j++;
        }
        i++;
    }
}

int ORBMatcher::Search(Frame::Ptr current_frame, Frame::Ptr last_frame, const float radius)
{
    ComputeBRIEF(last_frame);
    detector_->detectAndCompute(current_frame->image_left, cv::Mat(), );

    // search by BRIEFDes
    auto descriptors = mat2briefs(current_frame);
    auto descriptors_old = mat2briefs(last_frame);
    for (auto &pair_desciptor : descriptors)
    {
        unsigned long best_id = 0;
        if (SearchInAera(pair_desciptor.second, descriptors_old, best_id))
        {
            cv::Point2f point_2d = last_frame->features_left[best_id]->keypoint;
            visual::Landmark::Ptr landmark = last_frame->features_left[best_id]->landmark.lock();
            visual::Feature::Ptr new_left_feature = visual::Feature::Create(current_frame, point_2d, landmark);
            points_2d.push_back(point_2d);
            points_3d.push_back(eigen2cv(landmark->position));
        }
    }

    return false;
}

bool ORBMatcher::SearchInAera(const BRIEF descriptor, const std::map<unsigned long, BRIEF> &descriptors_old, unsigned long &best_id)
{
    cv::Point2f best_pt;
    int best_distance = 256;
    for (auto &pair_desciptor : descriptors_old)
    {
        int distance = Hamming(descriptor, pair_desciptor.second);
        if (distance < best_distance)
        {
            best_distance = distance;
            best_id = pair_desciptor.first;
        }
    }
    return best_distance < 160;
}

int ORBMatcher::Hamming(const BRIEF &a, const BRIEF &b)
{
    BRIEF xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

} // namespace lvio_fuison
