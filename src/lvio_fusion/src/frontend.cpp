#include "lvio_fusion/frontend.h"
#include "lvio_fusion/backend.h"
#include "lvio_fusion/config.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

#include <opencv2/core/eigen.hpp>

namespace lvio_fusion
{

Frontend::Frontend(int num_features, int init, int tracking, int tracking_bad, int need_for_keyframe)
    : num_features_(num_features), num_features_init_(init), num_features_tracking_(tracking),
      num_features_tracking_bad_(tracking_bad), num_features_needed_for_keyframe_(need_for_keyframe)
{
}

bool Frontend::AddFrame(lvio_fusion::Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(mutex);
    current_frame = frame;

    switch (status)
    {
    case FrontendStatus::BUILDING:
        InitMap();
        break;
    case FrontendStatus::INITIALIZING:
    case FrontendStatus::TRACKING_GOOD:
    case FrontendStatus::TRACKING_BAD:
    case FrontendStatus::TRACKING_TRY:
        if (Track())
        {
            //NOTE: semantic map
            if (!current_frame->objects.empty())
            {
                current_frame->UpdateLabel();
            }
            break;
        }
        else
        {
            return false;
        }
    case FrontendStatus::LOST:
        Reset();
        InitMap();
        break;
    }
    last_frame = current_frame;
    last_frame_pose_cache_ = last_frame->pose;
    return true;
}

void Frontend::AddImu(double time, Vector3d acc, Vector3d gyr)
{
    static double current_imu_time = 0;
    static bool first = true;
    static Vector3d acc0(0, 0, 0), gyr0(0, 0, 0), R(0, 0, 0), T(0, 0, 0), V(0, 0, 0);
    double dt = time - current_imu_time;
    current_imu_time = time;
    if (current_frame)
    {
        if (first)
        {
            first = false;
            acc0 = acc;
            gyr0 = gyr;
        }
        if (!current_frame->preintegration)
        {
            Vector3d ba = Vector3d::Zero(), bg = Vector3d::Zero(), v0 = Vector3d::Zero();
            if (last_frame && last_frame->preintegration)
            {
                ba = last_frame->preintegration->linearized_ba;
                bg = last_frame->preintegration->linearized_bg;
                v0 = last_frame->preintegration->v0 + last_frame->preintegration->delta_v;
            }
            current_frame->preintegration = imu::Preintegration::Create(acc0, gyr0, v0, ba, bg, imu_);
        }
        current_frame->preintegration->Append(dt, acc, gyr);
        if (current_key_frame && current_key_frame->preintegration && current_key_frame != current_frame)
        {
            current_key_frame->preintegration->Append(dt, acc, gyr);
        }
        acc0 = acc;
        gyr0 = acc;
    }
}

bool Frontend::Track()
{
    current_frame->pose = relative_i_j * last_frame_pose_cache_;
    TrackLastFrame();
    InitFramePoseByPnP();
    int inliers = current_frame->features_left.size();

    static int num_tries = 0;
    if (status == FrontendStatus::INITIALIZING)
    {
        if (inliers <= num_features_tracking_bad_)
        {
            status = FrontendStatus::BUILDING;
        }
    }
    else
    {
        if (inliers > num_features_tracking_)
        {
            // tracking good
            status = FrontendStatus::TRACKING_GOOD;
            num_tries = 0;
        }
        else if (inliers > num_features_tracking_bad_)
        {
            // tracking bad
            status = FrontendStatus::TRACKING_BAD;
            num_tries = 0;
        }
        else
        {
            // lost, but give a chance
            num_tries++;
            status = num_tries >= 4 ? FrontendStatus::LOST : FrontendStatus::TRACKING_TRY;
            num_tries %= 4;
            return false;
        }
    }

    // Add every frame during initializing
    if (inliers < num_features_needed_for_keyframe_)
    {
        CreateKeyframe();
    }
    else if (status == FrontendStatus::INITIALIZING)
    {
        CreateKeyframe(false);
    }
    relative_i_j = current_frame->pose * last_frame_pose_cache_.inverse();
    return true;
}

void Frontend::CreateKeyframe(bool need_new_features)
{
    // first, add new observations of old points
    for (auto pair_feature : current_frame->features_left)
    {
        auto feature = pair_feature.second;
        auto mp = feature->landmark.lock();
        mp->AddObservation(feature);
    }
    // detect new features, track in right image and triangulate map points
    if (need_new_features)
    {
        DetectNewFeatures();
    }
    // insert!
    Map::Instance().InsertKeyFrame(current_frame);
    current_key_frame = current_frame;
    LOG(INFO) << "Add a keyframe " << current_frame->id;
    // update backend because we have a new keyframe
    backend_.lock()->UpdateMap();
}

bool Frontend::InitFramePoseByPnP()
{
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    for (auto pair_feature : current_frame->features_left)
    {
        auto feature = pair_feature.second;
        auto camera_point = feature->landmark.lock();
        points_2d.push_back(feature->keypoint);
        Vector3d p = position_cache_[camera_point->id];
        points_3d.push_back(cv::Point3f(p.x(), p.y(), p.z()));
    }

    cv::Mat K;
    cv::eigen2cv(camera_left_->K(), K);
    cv::Mat rvec, tvec, inliers, D, cv_R;
    if (cv::solvePnPRansac(points_3d, points_2d, K, D, rvec, tvec, false, 100, 8.0F, 0.98, cv::noArray(), cv::SOLVEPNP_EPNP))
    {
        cv::Rodrigues(rvec, cv_R);
        Matrix3d R;
        cv::cv2eigen(cv_R, R);
        current_frame->pose = (camera_left_->extrinsic * SE3d(SO3d(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)))).inverse();
        return true;
    }
    return false;
}

inline double distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

inline void calcOpticalFlowPyrLK(cv::Mat &prevImg, cv::Mat &nextImg,
                                 std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                                 std::vector<uchar> &status, cv::Mat &err)
{
    cv::calcOpticalFlowPyrLK(
        prevImg, nextImg, prevPts, nextPts, status, err, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    std::vector<uchar> reverse_status;
    std::vector<cv::Point2f> reverse_pts = prevPts;
    cv::calcOpticalFlowPyrLK(
        nextImg, prevImg, nextPts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    for (size_t i = 0; i < status.size(); i++)
    {
        // clang-format off
        if (status[i] && reverse_status[i] && distance(prevPts[i], reverse_pts[i]) <= 0.5 
        && nextPts[i].x >= 0 && nextPts[i].x < prevImg.cols 
        && nextPts[i].y >= 0 && nextPts[i].y < prevImg.rows)
        // clang-format on
        {
            status[i] = 1;
        }
        else
            status[i] = 0;
    }
}

int Frontend::TrackLastFrame()
{
    // use LK flow to estimate points in the last image
    std::vector<cv::Point2f> kps_last, kps_current;
    std::vector<visual::Landmark::Ptr> landmarks;
    for (auto pair_feature : last_frame->features_left)
    {
        // use project point
        auto feature = pair_feature.second;
        auto camera_point = feature->landmark.lock();
        auto px = camera_left_->World2Pixel(position_cache_[camera_point->id], current_frame->pose);
        landmarks.push_back(camera_point);
        kps_last.push_back(feature->keypoint);
        kps_current.push_back(cv::Point2f(px[0], px[1]));
    }

    std::vector<uchar> status;
    cv::Mat error;
    calcOpticalFlowPyrLK(last_frame->image_left, current_frame->image_left, kps_last, kps_current, status, error);

    // NOTE: don‘t use, accuracy is very low
    // cv::findFundamentalMat(kps_current, kps_last, cv::FM_RANSAC, 3.0, 0.9, status);

    //DEBUG
    int num_good_pts = 0;
    mask_ = cv::Mat(current_frame->image_left.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat img_track = current_frame->image_left;
    cv::cvtColor(img_track, img_track, cv::COLOR_GRAY2RGB);
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i] && mask_.at<uchar>(kps_current[i]) == 255)
        {
            cv::arrowedLine(img_track, kps_current[i], kps_last[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            auto feature = visual::Feature::Create(current_frame, kps_current[i], landmarks[i]);
            current_frame->AddFeature(feature);
            cv::circle(mask_, feature->keypoint, 30, 0, cv::FILLED);
            num_good_pts++;
        }
    }
    cv::imshow("tracking", img_track);
    cv::waitKey(1);
    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::InitMap()
{
    int num_new_features = DetectNewFeatures();
    if (num_new_features < num_features_init_)
    {
        return false;
    }
    if (imu_)
    {
        status = FrontendStatus::INITIALIZING;
    }
    else
    {
        status = FrontendStatus::TRACKING_GOOD;
    }

    // the first frame is a keyframe
    Map::Instance().InsertKeyFrame(current_frame);
    LOG(INFO) << "Initial map created with " << num_new_features << " map points";

    // update backend and loop because we have a new keyframe
    backend_.lock()->UpdateMap();
    return true;
}

int Frontend::DetectNewFeatures()
{
    cv::Mat mask(current_frame->image_left.size(), CV_8UC1, 255);
    for (auto pair_feature : current_frame->features_left)
    {
        auto feature = pair_feature.second;
        cv::rectangle(mask, feature->keypoint - cv::Point2f(10, 10), feature->keypoint + cv::Point2f(10, 10), 0, cv::FILLED);
    }

    std::vector<cv::Point2f> kps_left, kps_right; // must be point2f
    cv::goodFeaturesToTrack(current_frame->image_left, kps_left, num_features_ - current_frame->features_left.size(), 0.01, 30, mask);

    // use LK flow to estimate points in the right image
    kps_right = kps_left;
    std::vector<uchar> status;
    cv::Mat error;
    calcOpticalFlowPyrLK(current_frame->image_left, current_frame->image_right, kps_left, kps_right, status, error);

    // triangulate new points
    int num_triangulated_pts = 0;
    int num_good_pts = 0;
    for (size_t i = 0; i < kps_left.size(); ++i)
    {
        if (status[i])
        {
            num_good_pts++;
            // triangulation
            Vector2d kp_left = cv2eigen(kps_left[i]);
            Vector2d kp_right = cv2eigen(kps_right[i]);
            Vector3d pb = Vector3d::Zero();
            triangulate(camera_left_->extrinsic.inverse(), camera_right_->extrinsic.inverse(),
                        camera_left_->Pixel2Sensor(kp_left), camera_right_->Pixel2Sensor(kp_right), pb);
            if ((camera_left_->Robot2Pixel(pb) - kp_left).norm() < 0.5 && (camera_right_->Robot2Pixel(pb) - kp_right).norm() < 0.5)
            {
                auto new_landmark = visual::Landmark::Create(pb, camera_left_);
                auto new_left_feature = visual::Feature::Create(current_frame, eigen2cv(kp_left), new_landmark);
                auto new_right_feature = visual::Feature::Create(current_frame, eigen2cv(kp_right), new_landmark);
                new_right_feature->is_on_left_image = false;
                new_landmark->AddObservation(new_left_feature);
                new_landmark->AddObservation(new_right_feature);
                current_frame->AddFeature(new_left_feature);
                current_frame->AddFeature(new_right_feature);
                Map::Instance().InsertLandmark(new_landmark);
                position_cache_[new_landmark->id] = new_landmark->ToWorld();
                num_triangulated_pts++;
            }
        }
    }

    LOG(INFO) << "Detect " << kps_left.size() << " new features";
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    LOG(INFO) << "new landmarks: " << num_triangulated_pts;
    return num_triangulated_pts;
}

// TODO
bool Frontend::Reset()
{
    backend_.lock()->Pause();
    Map::Instance().Reset();
    backend_.lock()->Continue();
    status = FrontendStatus::BUILDING;
    LOG(INFO) << "Reset Succeed";
    return true;
}

void Frontend::UpdateCache()
{
    position_cache_.clear();
    for (auto pair_feature : last_frame->features_left)
    {
        auto feature = pair_feature.second;
        auto camera_point = feature->landmark.lock();
        position_cache_[camera_point->id] = camera_point->ToWorld();
    }
    last_frame_pose_cache_ = last_frame->pose;
}

} // namespace lvio_fusion