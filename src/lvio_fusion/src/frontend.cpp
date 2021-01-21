#include "lvio_fusion/frontend.h"
#include "lvio_fusion/backend.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/camera.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

#include <opencv2/core/eigen.hpp>

namespace lvio_fusion
{

Frontend::Frontend(int num_features, int init, int tracking, int tracking_bad, int need_for_keyframe)
    : num_features_(num_features), num_features_init_(init), num_features_tracking_bad_(tracking_bad),
      num_features_needed_for_keyframe_(need_for_keyframe),
      matcher_(ORBMatcher(tracking_bad))
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
            return false;
    case FrontendStatus::LOST:
        //TODO
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
            current_frame->preintegration = imu::Preintegration::Create(acc0, gyr0, v0, ba, bg);
        }
        current_frame->preintegration->Append(dt, acc, gyr);
        if (last_key_frame && last_key_frame->preintegration && last_key_frame != current_frame)
        {
            last_key_frame->preintegration->Append(dt, acc, gyr);
        }
        acc0 = acc;
        gyr0 = acc;
    }
}

bool Frontend::Track()
{
    current_frame->pose = relative_i_j * last_frame_pose_cache_;
    Frame::Ptr base_frame = status == FrontendStatus::TRACKING_TRY ? last_key_frame : last_frame;
    int num_inliers = (num_inliers = TrackLastFrame(base_frame)) == 0 ? Relocate(base_frame) : num_inliers;

    if ((status != FrontendStatus::INITIALIZING && num_inliers < num_features_needed_for_keyframe_) ||
        (status == FrontendStatus::INITIALIZING && current_frame->time - last_key_frame->time > 0.25))
    {
        CreateKeyframe();
    }

    if (status == FrontendStatus::INITIALIZING)
    {
        if (!num_inliers)
        {
            status = FrontendStatus::BUILDING;
        }
    }
    else
    {
        if (num_inliers && (current_frame->pose.translation() - last_frame->pose.translation()).norm() < 5)
        {
            // tracking good
            status = FrontendStatus::TRACKING_GOOD;
        }
        else
        {
            // tracking bad, but give a chance
            status = FrontendStatus::TRACKING_TRY;
            LOG(INFO) << "Lost, try again!";
            return false;
        }
    }

    // smooth trajectory
    current_frame->pose = se3_slerp(current_frame->pose, relative_i_j * last_frame_pose_cache_, 0.5);
    relative_i_j = current_frame->pose * last_frame_pose_cache_.inverse();
    return true;
}

int Frontend::TrackLastFrame(Frame::Ptr base_frame)
{
    std::vector<cv::Point2f> kps_last, kps_current;
    std::vector<visual::Landmark::Ptr> landmarks;
    std::vector<uchar> status;
    // use LK flow to estimate points in the last image
    for (auto &pair_feature : base_frame->features_left)
    {
        // use project point
        auto feature = pair_feature.second;
        auto landmark = feature->landmark.lock();
        auto px = Camera::Get()->World2Pixel(position_cache_[landmark->id], current_frame->pose);
        kps_last.push_back(feature->keypoint);
        kps_current.push_back(cv::Point2f(px[0], px[1]));
        landmarks.push_back(landmark);
    }
    optical_flow(base_frame->image_left, current_frame->image_left, kps_last, kps_current, status);

    // Solve PnP
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    std::vector<int> map;
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i])
        {
            map.push_back(i);
            points_2d.push_back(kps_current[i]);
            Vector3d p = position_cache_[landmarks[i]->id];
            points_3d.push_back(cv::Point3f(p.x(), p.y(), p.z()));
        }
    }

    int num_good_pts = 0;
    cv::Mat rvec, tvec, inliers, cv_R;
    if ((2 * (int)points_2d.size() - (int)kps_last.size() > num_features_tracking_bad_) &&
        cv::solvePnPRansac(points_3d, points_2d, Camera::Get()->K, Camera::Get()->D, rvec, tvec, false, 100, 8.0F, 0.98, inliers, cv::SOLVEPNP_EPNP))
    {
        cv::Rodrigues(rvec, cv_R);
        Matrix3d R;
        cv::cv2eigen(cv_R, R);
        current_frame->pose = (Camera::Get()->extrinsic * SE3d(SO3d(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)))).inverse();

        cv::Mat img_track = current_frame->image_left;
        cv::cvtColor(img_track, img_track, cv::COLOR_GRAY2RGB);
        for (int r = 0; r < inliers.rows; r++)
        {
            int i = map[inliers.at<int>(r)];
            cv::arrowedLine(img_track, kps_current[i], kps_last[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            cv::circle(img_track, kps_current[i], 2, cv::Scalar(0, 255, 0), cv::FILLED);
            auto feature = visual::Feature::Create(current_frame, kps_current[i], landmarks[i]);
            current_frame->AddFeature(feature);
            num_good_pts++;
        }
        cv::imshow("tracking", img_track);
        cv::waitKey(1);
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return 2 * num_good_pts - kps_last.size() < num_features_tracking_bad_ ? 0 : num_good_pts;
}

int Frontend::Relocate(Frame::Ptr base_frame)
{
    std::vector<cv::Point2f> kps_left, kps_right, kps_current;
    std::vector<Vector3d> pbs;
    int num_good_pts = matcher_.Relocate(base_frame, current_frame, kps_left, kps_right, kps_current, pbs);
    if (num_good_pts)
    {
        for (int i = 0; i < kps_left.size(); i++)
        {
            auto new_landmark = visual::Landmark::Create(pbs[i]);
            auto new_left_feature = visual::Feature::Create(base_frame, kps_left[i], new_landmark);
            auto new_right_feature = visual::Feature::Create(base_frame, kps_right[i], new_landmark);
            new_left_feature->a = 2;
            new_right_feature->a = 2;
            new_right_feature->is_on_left_image = false;
            new_landmark->AddObservation(new_left_feature);
            new_landmark->AddObservation(new_right_feature);
            base_frame->AddFeature(new_left_feature);
            base_frame->AddFeature(new_right_feature);
            Map::Instance().InsertLandmark(new_landmark);
            position_cache_[new_landmark->id] = new_landmark->ToWorld();

            auto feature = visual::Feature::Create(current_frame, kps_current[i], new_landmark);
            feature->a = 3;
            current_frame->AddFeature(feature);
            new_landmark->AddObservation(feature);
        }
    }
    if (base_frame != last_key_frame)
    {
        // first, add new observations of old points
        for (auto &pair_feature : base_frame->features_left)
        {
            auto feature = pair_feature.second;
            auto landmark = feature->landmark.lock();
            landmark->AddObservation(feature);
        }

        // insert!
        current_frame->id++;
        Map::Instance().InsertKeyFrame(base_frame);
        last_key_frame = base_frame;
        LOG(INFO) << "Make last frame a keyframe " << base_frame->id;

        // update backend because we have a new keyframe
        backend_.lock()->UpdateMap();
    }
    return num_good_pts;
}

bool Frontend::InitMap()
{
    int num_new_features = DetectNewFeatures();
    if (num_new_features < num_features_init_)
    {
        return false;
    }
    if (Imu::Num())
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
    int num_tries = 0;
    int num_triangulated_pts = 0;
    int num_good_pts = 0;
    while (num_tries++ < 2 && current_frame->features_left.size() < 0.8 * num_features_)
    {
        cv::Mat mask(current_frame->image_left.size(), CV_8UC1, 255);
        for (auto &pair_feature : current_frame->features_left)
        {
            auto feature = pair_feature.second;
            cv::circle(mask, feature->keypoint, 20, 0, cv::FILLED);
        }

        std::vector<cv::Point2f> kps_left, kps_right; // must be point2f
        cv::goodFeaturesToTrack(current_frame->image_left, kps_left, num_features_ - current_frame->features_left.size(), 0.01, 20, mask);

        // use LK flow to estimate points in the right image
        kps_right = kps_left;
        std::vector<uchar> status;
        optical_flow(current_frame->image_left, current_frame->image_right, kps_left, kps_right, status);

        // triangulate new points
        for (size_t i = 0; i < kps_left.size(); ++i)
        {
            if (status[i])
            {
                num_good_pts++;
                // triangulation
                Vector2d kp_left = cv2eigen(kps_left[i]);
                Vector2d kp_right = cv2eigen(kps_right[i]);
                Vector3d pb = Vector3d::Zero();
                triangulate(Camera::Get()->extrinsic.inverse(), Camera::Get(1)->extrinsic.inverse(),
                            Camera::Get()->Pixel2Sensor(kp_left), Camera::Get(1)->Pixel2Sensor(kp_right), pb);
                if ((Camera::Get()->Robot2Pixel(pb) - kp_left).norm() < 0.5 && (Camera::Get(1)->Robot2Pixel(pb) - kp_right).norm() < 0.5)
                {
                    auto new_landmark = visual::Landmark::Create(pb);
                    auto new_left_feature = visual::Feature::Create(current_frame, kps_left[i], new_landmark);
                    auto new_right_feature = visual::Feature::Create(current_frame, kps_right[i], new_landmark);
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
    }

    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    LOG(INFO) << "new landmarks: " << num_triangulated_pts;
    return num_triangulated_pts;
}

void Frontend::CreateKeyframe()
{
    // first, add new observations of old points
    for (auto &pair_feature : current_frame->features_left)
    {
        auto feature = pair_feature.second;
        auto landmark = feature->landmark.lock();
        landmark->AddObservation(feature);
    }

    // detect new features, track in right image and triangulate map points
    DetectNewFeatures();

    // insert!
    Map::Instance().InsertKeyFrame(current_frame);
    last_key_frame = current_frame;
    LOG(INFO) << "Add a keyframe " << current_frame->id;

    // update backend because we have a new keyframe
    backend_.lock()->UpdateMap();
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
    for (auto &pair_feature : last_frame->features_left)
    {
        auto feature = pair_feature.second;
        auto camera_point = feature->landmark.lock();
        position_cache_[camera_point->id] = camera_point->ToWorld();
    }
    last_frame_pose_cache_ = last_frame->pose;
}

} // namespace lvio_fusion