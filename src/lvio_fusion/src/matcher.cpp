#include "lvio_fusion/visual/matcher.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

// void ORBMatcher::ComputeBRIEF(Frame::Ptr frame)
// {
//     // compute descriptors
//     std::vector<cv::KeyPoint> keypoints;
//     for (auto &pair_feature : frame->features_left)
//     {
//         keypoints.push_back(cv::KeyPoint(pair_feature.second->keypoint, 1));
//     }
//     cv::Mat descriptors;
//     detector_->compute(frame->image_left, keypoints, descriptors);

//     // NOTE: detector_->compute maybe remove some row because its descriptor cannot be computed
//     int j = 0, i = 0;
//     frame->descriptors = cv::Mat::zeros(frame->features_left.size(), 32, CV_8U);
//     for (auto &pair_feature : frame->features_left)
//     {
//         if (pair_feature.second->keypoint == keypoints[j].pt && j < descriptors.rows)
//         {
//             descriptors.row(j).copyTo(frame->descriptors.row(i));
//             j++;
//         }
//         i++;
//     }
// }

int ORBMatcher::Search(Frame::Ptr current_frame, Frame::Ptr last_frame, std::vector<cv::Point2f> &kps_current, std::vector<cv::Point2f> &kps_last, std::vector<uchar> &status, float thershold)
{
    const static int max_dist = 50;
    std::vector<cv::KeyPoint> kps_mismatch;
    cv::Mat mask = cv::Mat::zeros(current_frame->image_left.size(), CV_8UC1);
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (!status[i])
        {
            kps_mismatch.push_back(cv::KeyPoint(kps_last[i], 1));
            cv::circle(mask, kps_last[i], max_dist, 255, cv::FILLED);
        }
    }

    cv::Mat descriptors_mismatch;
    detector_->compute(last_frame->image_left, kps_mismatch, descriptors_mismatch);

    // for (auto kp : kps_mismatch)
    // {
    //     cv::Mat mask = cv::Mat::zeros(current_frame->image_left.size(), CV_8UC1);
    //     cv::circle(mask, kp.pt, thershold, 255, cv::FILLED);
    //     std::vector<cv::KeyPoint> kps_add;
    //     cv::Mat descriptors_add;
    //     detector_->detectAndCompute(current_frame->image_left, mask, kps_add, descriptors_add);
    // }

    std::vector<cv::KeyPoint> kps_add;
    cv::Mat descriptors_add;
    detector_->detectAndCompute(current_frame->image_left, mask, kps_add, descriptors_add);
    std::vector<cv::DMatch> matches;
    matcher_->match(descriptors_mismatch, descriptors_add, matches);

    int i = 0, num_good_matches = 0;
    cv::Mat img_track = current_frame->image_left;
    cv::cvtColor(img_track, img_track, cv::COLOR_GRAY2RGB);
    for (auto &match : matches)
    {
        int index_mismatch = match.queryIdx;
        int index_add = match.trainIdx;
        // for (; i < kps_last.size(); i++)
        // {
        //     if (kps_mismatch[index_mismatch].pt == kps_last[i])
        //         break;
        // }
        if (distance(kps_mismatch[index_mismatch].pt, kps_add[index_add].pt) < thershold)
        {
            LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!" << distance(kps_mismatch[index_mismatch].pt, kps_add[index_add].pt);
            // kps_current[i] = kps_add[index_add].pt;
            // status[i] = 2;
            num_good_matches++;
            cv::Scalar color = cv::Scalar(255, 0, 0);
            cv::arrowedLine(img_track, kps_mismatch[index_mismatch].pt, kps_add[index_add].pt, color, 1, 8, 0, 0.2);
            cv::circle(img_track, kps_mismatch[index_mismatch].pt, 2, color, cv::FILLED);
        }
    }
    cv::imshow("debug", img_track);
    cv::waitKey(1);
    return num_good_matches;
}

// bool ORBMatcher::SearchInAera(const BRIEF descriptor, const std::map<unsigned long, BRIEF> &descriptors_old, unsigned long &best_id)
// {
//     cv::Point2f best_pt;
//     int best_distance = 256;
//     for (auto &pair_desciptor : descriptors_old)
//     {
//         int distance = Hamming(descriptor, pair_desciptor.second);
//         if (distance < best_distance)
//         {
//             best_distance = distance;
//             best_id = pair_desciptor.first;
//         }
//     }
//     return best_distance < 160;
// }

// int ORBMatcher::Hamming(const BRIEF &a, const BRIEF &b)
// {
//     BRIEF xor_of_bitset = a ^ b;
//     int dis = xor_of_bitset.count();
//     return dis;
// }

} // namespace lvio_fusion
