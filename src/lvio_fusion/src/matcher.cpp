#include "lvio_fusion/visual/matcher.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/camera.h"

#include <opencv2/flann.hpp>

namespace lvio_fusion
{

void ORBMatcher::FastFeatureToTrack(cv::Mat &image, std::vector<cv::Point2f> &corners, double minDistance, cv::Mat mask)
{
    if (mask.empty())
    {
        mask = cv::Mat(image.size(), CV_8UC1, 255);
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(image, keypoints, 10);
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        if (mask.at<uchar>(keypoints[i].pt) != 0)
        {
            cv::circle(mask, keypoints[i].pt, minDistance, 0, cv::FILLED);
            corners.push_back(keypoints[i].pt);
        }
    }
}

void ORBMatcher::Match(cv::Mat &prevImg, cv::Mat &nextImg, std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts, std::vector<uchar> &status)
{
    std::vector<cv::KeyPoint> kps_prev, kps_next;
    cv::Mat descriptors_prev, descriptors_next;
    convert_points(prevPts, kps_prev);
    detector_->compute(prevImg, kps_prev, descriptors_prev);
    convert_points(kps_prev, prevPts);
    cv::FAST(nextImg, kps_next, 20);
    detector_->compute(nextImg, kps_next, descriptors_next);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(descriptors_prev, descriptors_next, knn_matches, 2);
    const float ratio_thresh = 0.8;
    nextPts.resize(prevPts.size(), cv::Point2f(0, 0));
    status.resize(prevPts.size(), 0);
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance && distance(kps_prev[knn_matches[i][0].queryIdx].pt, kps_next[knn_matches[i][0].trainIdx].pt) < 200)
        {
            nextPts[knn_matches[i][0].queryIdx] = kps_next[knn_matches[i][0].trainIdx].pt;
            status[knn_matches[i][0].queryIdx] = 1;
        }
    }
}

// kps_left: last keypoints, pbs: new landmark in last frame
int ORBMatcher::Relocate(Frame::Ptr last_frame, Frame::Ptr current_frame,
                         std::vector<cv::Point2f> &kps_left, std::vector<cv::Point2f> &kps_right, std::vector<cv::Point2f> &kps_current, std::vector<Vector3d> &pbs)
{
    // detect and match by ORB
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
    cv::Mat mask = cv::Mat(current_frame->image_left.size(), CV_8UC1, 255);
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            auto pt_fast_last = kps_fast_last[knn_matches[i][0].queryIdx].pt;
            auto pt_fast_current = kps_fast_current[knn_matches[i][0].trainIdx].pt;
            if (distance(pt_fast_last, pt_fast_current) < 200 && mask.at<uchar>(pt_fast_current) != 0)
            {
                cv::circle(mask, pt_fast_current, 10, 0, cv::FILLED);
                kps_match_left.push_back(pt_fast_last);
                kps_match_current.push_back(pt_fast_current);
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
        LOG(INFO) << "Matcher relocate by " << num_good_pts << " points.";
    }
    return num_good_pts;
}

int ORBMatcher::Relocate(Frame::Ptr last_frame, Frame::Ptr current_frame, SE3d &relative_o_c)
{
    // detect and match by ORB
    std::vector<cv::KeyPoint> kps_fast_last, kps_fast_current;
    std::vector<visual::Landmark::Ptr> landmarks;
    for (auto pair_feature : last_frame->features_left)
    {
        kps_fast_last.push_back(cv::KeyPoint(pair_feature.second->keypoint, 1));
        landmarks.push_back(pair_feature.second->landmark.lock());
    }
    cv::Mat descriptors_last, descriptors_current;
    detector_->compute(last_frame->image_left, kps_fast_last, descriptors_last);
    detector_->detectAndCompute(current_frame->image_left, cv::noArray(), kps_fast_current, descriptors_current);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(descriptors_last, descriptors_current, knn_matches, 2);
    const float ratio_thresh = 0.8;
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            points_3d.push_back(eigen2cv(landmarks[knn_matches[i][0].queryIdx]->ToWorld()));
            points_2d.push_back(kps_fast_current[knn_matches[i][0].trainIdx].pt);
        }
    }

    // Solve PnP
    int num_good_pts = 0;
    cv::Mat rvec, tvec, inliers, cv_R;
    if (points_2d.size() > num_features_threshold_ &&
        cv::solvePnPRansac(points_3d, points_2d, Camera::Get()->K, Camera::Get()->D, rvec, tvec, false, 100, 8.0F, 0.98, inliers, cv::SOLVEPNP_EPNP))
    {
        cv::Rodrigues(rvec, cv_R);
        Matrix3d R;
        cv::cv2eigen(cv_R, R);
        relative_o_c = (last_frame->pose * Camera::Get()->extrinsic * SE3d(SO3d(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)))).inverse();
        num_good_pts = inliers.rows;
    }
    return std::min(num_good_pts / 2, 50);
}

} // namespace lvio_fusion
