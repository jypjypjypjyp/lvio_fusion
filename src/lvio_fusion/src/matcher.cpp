#include "lvio_fusion/visual/matcher.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/camera.h"

#include <opencv2/flann.hpp>

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

double get_search_area(double depth)
{
    return std::max(30.0, Camera::Get()->fx / depth);
}

void convert(const std::vector<cv::KeyPoint> &kps, std::vector<cv::Point2f> &ps)
{
    ps.resize(kps.size());
    for (int i = 0; i < kps.size(); i++)
    {
        ps[i] = kps[i].pt;
    }
}

void convert(const std::vector<cv::Point2f> &ps, std::vector<cv::KeyPoint> &kps)
{
    kps.resize(ps.size());
    for (int i = 0; i < ps.size(); i++)
    {
        kps[i] = cv::KeyPoint(ps[i], 1);
    }
}

int ORBMatcher::Search(Frame::Ptr current_frame, Frame::Ptr last_frame, std::vector<cv::Point2f> &kps_current, std::vector<cv::Point2f> &kps_last, std::vector<uchar> &status, std::vector<double> &depths, float thershold)
{
    const static int max_dist = 50;
    std::vector<cv::Point2f> kps_mismatch, kps_add;
    std::vector<cv::KeyPoint> kps_mismatch_, kps_add_;
    cv::Mat descriptors_mismatch, descriptors_add;
    cv::Mat mask = cv::Mat::zeros(current_frame->image_left.size(), CV_8UC1);
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (!status[i])
        {
            kps_mismatch.push_back(kps_last[i]);
            cv::circle(mask, kps_last[i], max_dist, 255, cv::FILLED);
        }
    }
    if (kps_mismatch.empty())
        return 0;

    // compute descriptors of mismatch keypoints
    convert(kps_mismatch, kps_mismatch_);
    detector_->compute(last_frame->image_left, kps_mismatch_, descriptors_mismatch);
    convert(kps_mismatch_, kps_mismatch);
    // NOTE: detector_->compute maybe remove some row because its descriptor cannot be computed
    // remap
    std::unordered_map<int, int> map;
    for (int j = 0, i = 0; j < kps_mismatch.size(); j++)
    {
        for (; i < kps_last.size(); i++)
        {
            if (kps_mismatch[j] == kps_last[i])
            {
                map[j] = i;
            }
        }
    }

    // detect and compute descriptors of supplementary keypoints
    detector_->detectAndCompute(current_frame->image_left, mask, kps_add_, descriptors_add);
    LOG(INFO) << ")))))))))))))))))))))))))))))))" << kps_add_.size();
    if (kps_add_.empty())
        return 0;
    convert(kps_add_, kps_add);
    for(auto p:kps_add)
    {
        cv::circle(mask, p, 2, cv::Scalar(0, 0, 255), cv::FILLED);
    }
    cv::imshow("mask", mask);

    int num_good_matches = 0;
    cv::Mat img_debug = current_frame->image_left;
    cv::cvtColor(img_debug, img_debug, cv::COLOR_GRAY2RGB);
    cv::Mat img_debug2 = current_frame->image_left;
    cv::cvtColor(img_debug2, img_debug2, cv::COLOR_GRAY2RGB);
    // match by distance and descriptors
    cv::flann::KDTreeIndexParams indexParams(1);
    cv::Mat_<float> features(0, 2);

    for (auto &&point : kps_add)
    {
        //Fill matrix
        cv::Mat row = (cv::Mat_<float>(1, 2) << point.x, point.y);
        features.push_back(row);
    }
    cv::flann::Index kdtree(features.reshape(1), indexParams);
    std::vector<int> indices;
    std::vector<float> dists;
    for (int j = 0; j < kps_mismatch.size(); j++)
    {
        std::vector<float> query = {kps_mismatch[j].x, kps_mismatch[j].y};
        int n = kdtree.radiusSearch(query, indices, dists, max_dist, 100);
        LOG(INFO) << n;
        for (int k = 0; k < n; k++)
        {
            int i = indices[k];
            if (i != 0)
            {
                cv::arrowedLine(img_debug2, kps_mismatch[j], kps_add[i], cv::Scalar(0, 0, 255), 1, 8, 0, 0.2);
            }
        }
        // cv::Mat descriptors = cv::Mat::zeros(indices.size(), descriptors_add.cols, descriptors_add.type());
        // for (int i = 0; i < indices.size(); i++)
        // {
        //     descriptors_add.row(indices[i]).copyTo(descriptors.row(i));
        // }
        // std::vector<cv::DMatch> matches;
        // matcher_->match(descriptors_mismatch.row(j), descriptors, matches);
        // if (matches.empty())
        //     continue;
        // int index_mismatch = matches[0].queryIdx, index_add = matches[0].trainIdx;
        // cv::arrowedLine(img_debug, kps_mismatch[index_mismatch], kps_add[index_add], cv::Scalar(0, 0, 255), 1, 8, 0, 0.2);
        // cv::circle(img_debug, kps_mismatch[index_mismatch], 2, cv::Scalar(0, 0, 255), cv::FILLED);
        // kps_current[map[index_mismatch]] = kps_add[index_add];
        // status[map[index_mismatch]] = 2;
        // num_good_matches++;
    }
    cv::imshow("debug", img_debug2);
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
