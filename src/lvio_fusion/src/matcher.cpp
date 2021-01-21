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

// int ORBMatcher::Search(Frame::Ptr current_frame, Frame::Ptr last_frame, std::vector<cv::Point2f> &kps_current, std::vector<cv::Point2f> &kps_last, std::vector<uchar> &status, std::vector<double> &depths, float thershold)
// {
//     const static int max_dist = 50;
//     std::vector<cv::Point2f> kps_mismatch, kps_add;
//     std::vector<cv::KeyPoint> kps_mismatch_, kps_add_;
//     cv::Mat descriptors_mismatch, descriptors_add;
//     cv::Mat mask = cv::Mat::zeros(current_frame->image_left.size(), CV_8UC1);
//     for (size_t i = 0; i < status.size(); ++i)
//     {
//         if (!status[i])
//         {
//             kps_mismatch.push_back(kps_last[i]);
//             cv::circle(mask, kps_last[i], max_dist, 255, cv::FILLED);
//         }
//     }
//     if (kps_mismatch.empty())
//         return 0;

//     // compute descriptors of mismatch keypoints
//     convert_points(kps_mismatch, kps_mismatch_);
//     detector_->compute(last_frame->image_left, kps_mismatch_, descriptors_mismatch);
//     convert_points(kps_mismatch_, kps_mismatch);
//     // NOTE: detector_->compute maybe remove some row because its descriptor cannot be computed
//     // remap
//     std::unordered_map<int, int> map;
//     for (int j = 0, i = 0; j < kps_mismatch.size(); j++)
//     {
//         for (; i < kps_last.size(); i++)
//         {
//             if (kps_mismatch[j] == kps_last[i])
//             {
//                 map[j] = i;
//             }
//         }
//     }

//     // detect and compute descriptors of supplementary keypoints
//     detector_->detectAndCompute(current_frame->image_left, mask, kps_add_, descriptors_add);
//     LOG(INFO) << ")))))))))))))))))))))))))))))))" << kps_add_.size();
//     if (kps_add_.empty())
//         return 0;
//     convert_points(kps_add_, kps_add);

//     cv::cvtColor(mask, mask, cv::COLOR_GRAY2RGB);
//     for (auto p : kps_add)
//     {
//         cv::circle(mask, p, 2, cv::Scalar(0, 0, 255), cv::FILLED);
//     }
//     cv::imshow("mask", mask);

//     int num_good_matches = 0;
//     cv::Mat img_debug = current_frame->image_left;
//     cv::cvtColor(img_debug, img_debug, cv::COLOR_GRAY2RGB);
//     cv::Mat img_debug2 = current_frame->image_left;
//     cv::cvtColor(img_debug2, img_debug2, cv::COLOR_GRAY2RGB);
//     // match by distance and descriptors
//     cv::flann::KDTreeIndexParams indexParams(1);
//     cv::Mat_<float> features(0, 2);

//     for (auto &&point : kps_add)
//     {
//         //Fill matrix
//         cv::Mat row = (cv::Mat_<float>(1, 2) << point.x, point.y);
//         features.push_back(row);
//     }
//     cv::flann::Index kdtree(features.reshape(1), indexParams);
//     std::vector<int> indices;
//     std::vector<float> dists;
//     for (int j = 0; j < kps_mismatch.size(); j++)
//     {
//         std::vector<float> query = {kps_mismatch[j].x, kps_mismatch[j].y};
//         int n = std::min(100, kdtree.radiusSearch(query, indices, dists, max_dist * max_dist, 100));
//         cv::Mat descriptors = cv::Mat::zeros(n, descriptors_add.cols, descriptors_add.type());
//         for (int i = 0; i < n; i++)
//         {
//             descriptors_add.row(indices[i]).copyTo(descriptors.row(i));
//         }
//         std::vector<cv::DMatch> matches;
//         matcher_->match(descriptors_mismatch.row(j), descriptors, matches);
//         if (matches.empty())
//             continue;
//         int index_mismatch = matches[0].queryIdx, index_add = indices[matches[0].trainIdx];
//         cv::arrowedLine(img_debug, kps_mismatch[index_mismatch], kps_add[index_add], cv::Scalar(0, 0, 255), 1, 8, 0, 0.2);
//         cv::circle(img_debug, kps_mismatch[index_mismatch], 2, cv::Scalar(0, 0, 255), cv::FILLED);
//         kps_current[map[index_mismatch]] = kps_add[index_add];
//         status[map[index_mismatch]] = 2;
//         cv::arrowedLine(img_debug2, kps_mismatch[j], kps_add[index_add], cv::Scalar(0, 0, 255), 1, 8, 0, 0.2);
//         num_good_matches++;
//     }
//     cv::imshow("debug", img_debug2);
//     cv::waitKey(1);
//     return num_good_matches;
// }

// kps_left: last keypoints, pbs: new landmark in last frame
int ORBMatcher::Relocate(Frame::Ptr last_frame, Frame::Ptr current_frame,
                         std::vector<cv::Point2f> &kps_left, std::vector<cv::Point2f> &kps_right, std::vector<cv::Point2f> &kps_current, std::vector<Vector3d> &pbs)
{
    // detect and match by ORB
    cv::Mat img_debug;
    cv::cvtColor(current_frame->image_left, img_debug, cv::COLOR_GRAY2RGB);
    std::vector<cv::KeyPoint> kps_fast_last, kps_fast_current;
    cv::Mat descriptors_last, descriptors_current;
    cv::FAST(last_frame->image_left, kps_fast_last, 20);
    detector_->compute(last_frame->image_left, kps_fast_last, descriptors_last);
    cv::FAST(current_frame->image_left, kps_fast_current, 20);
    detector_->compute(current_frame->image_left, kps_fast_current, descriptors_current);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(descriptors_last, descriptors_current, knn_matches, 2);
    const float ratio_thresh = 0.8;
    std::vector<cv::Point2f> kps_match_left, kps_match_right, kps_match_current;
    cv::Mat mask = cv::Mat::zeros(current_frame->image_left.size(), CV_8U);
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            auto pt_fast_last = kps_fast_last[knn_matches[i][0].queryIdx].pt;
            auto pt_fast_current = kps_fast_current[knn_matches[i][0].trainIdx].pt;
            if (distance(pt_fast_last, pt_fast_current) < 200 && mask.at<uchar>(pt_fast_current) != 255)
            {
                cv::circle(mask, pt_fast_current, 10, 255, cv::FILLED);
                cv::Point2f kp_last = kps_fast_last[knn_matches[i][0].queryIdx].pt,
                            kp_current = kps_fast_current[knn_matches[i][0].trainIdx].pt;
                kps_match_left.push_back(kp_last);
                kps_match_current.push_back(kp_current);
                cv::arrowedLine(img_debug, kp_current, kp_last, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            }
        }
    }

    // triangulate points
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    std::vector<Vector3d> points_pb;
    std::vector<int> map;
    if (kps_match_left.size() > num_features_threshold_)
    {
        kps_match_right = kps_match_left;
        std::vector<uchar> status;
        optical_flow(last_frame->image_left, last_frame->image_right, kps_match_left, kps_match_right, status);

        for (size_t i = 0; i < kps_match_left.size(); ++i)
        {
            if (status[i])
            {
                // triangulation
                Vector2d kp_left = cv2eigen(kps_match_left[i]);
                Vector2d kp_right = cv2eigen(kps_match_right[i]);
                Vector3d pb = Vector3d::Zero();
                triangulate(Camera::Get()->extrinsic.inverse(), Camera::Get(1)->extrinsic.inverse(),
                            Camera::Get()->Pixel2Sensor(kp_left), Camera::Get(1)->Pixel2Sensor(kp_right), pb);
                if ((Camera::Get()->Robot2Pixel(pb) - kp_left).norm() < 0.5 && (Camera::Get(1)->Robot2Pixel(pb) - kp_right).norm() < 0.5)
                {
                    map.push_back(i);
                    points_2d.push_back(kps_match_current[i]);
                    Vector3d p = Camera::Get()->Robot2World(pb, last_frame->pose);
                    points_3d.push_back(cv::Point3f(p.x(), p.y(), p.z()));
                    points_pb.push_back(pb);
                }
            }
        }
    }

    // solve pnp
    int num_good_pts = 0;
    cv::Mat rvec, tvec, inliers, cv_R;
    if (points_2d.size() > num_features_threshold_ &&
        cv::solvePnPRansac(points_3d, points_2d, Camera::Get()->K, Camera::Get()->D, rvec, tvec, false, 100, 8.0F, 0.98, inliers, cv::SOLVEPNP_EPNP))
    {
        cv::Rodrigues(rvec, cv_R);
        Matrix3d R;
        cv::cv2eigen(cv_R, R);
        current_frame->pose = (Camera::Get()->extrinsic * SE3d(SO3d(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)))).inverse();

        for (int r = 0; r < inliers.rows; r++)
        {
            int index_points = inliers.at<int>(r);
            int index_kps = map[index_points];
            kps_left.push_back(kps_match_left[index_kps]);
            kps_right.push_back(kps_match_right[index_kps]);
            kps_current.push_back(kps_match_current[index_kps]);
            pbs.push_back(points_pb[index_points]);
            num_good_pts++;
        }
    }

    cv::imshow("matcher", img_debug);
    cv::waitKey(1);
    LOG(INFO) << "Matcher relocate by " << num_good_pts << " points.";
    return num_good_pts < num_features_threshold_ ? 0 : num_good_pts;
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
