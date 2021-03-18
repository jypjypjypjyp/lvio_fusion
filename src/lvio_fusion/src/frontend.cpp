#include "lvio_fusion/frontend.h"
#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres/base.hpp"
#include "lvio_fusion/ceres/imu_error.hpp"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/navsat/navsat.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/camera.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

namespace lvio_fusion
{

Frontend::Frontend(int num_features, int init, int tracking, int tracking_bad, int need_for_keyframe)
    : num_features_(num_features), num_features_init_(init), num_features_tracking_bad_(tracking_bad),
      num_features_needed_for_keyframe_(need_for_keyframe),
      matcher_(ORBMatcher(tracking_bad))
{
}

bool Frontend::AddFrame(Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(mutex);

    current_frame = frame;
    switch (status)
    {
    case FrontendStatus::BUILDING:
    case FrontendStatus::LOST:
        // Reset();
        InitMap();
        break;
    case FrontendStatus::INITIALIZING:
    case FrontendStatus::TRACKING_GOOD:
    case FrontendStatus::TRACKING_TRY:
        InitFrame();
        Track();
        //NOTE: semantic map
        if (!current_frame->objects.empty())
        {
            current_frame->UpdateLabel();
        }
        break;
    }
    last_frame = current_frame;
    last_frame_pose_cache_ = last_frame->pose;
    return true;
}

void Frontend::AddImu(double time, Vector3d acc, Vector3d gyr)
{
    imu_buf.push_back(ImuData(acc, gyr, time));
}

void Frontend::PreintegrateIMU()
{
    if (!last_frame)
        return;
    if (Imu::Get()->initialized)
    {
        current_frame->bImu = true;
    }
    std::vector<ImuData> imu_from_last_frame;
    while (true)
    {
        if (imu_buf.empty())
        {
            usleep(500);
            continue;
        }
        ImuData imu_data = imu_buf.front();
        if (imu_data.t < last_frame->time - 0.001)
        {
            imu_buf.pop_front();
        }
        else if (imu_data.t < current_frame->time - 0.001)
        {
            imu_from_last_frame.push_back(imu_data);
            imu_buf.pop_front();
        }
        else
        {
            imu_from_last_frame.push_back(imu_data);
            break;
        }
    }
    const int n = imu_from_last_frame.size() - 1;
    imu::Preintegration::Ptr imu_preintegrated_from_last_frame = imu::Preintegration::Create(last_frame->GetImuBias());
    if (imu_preintegrated_from_last_kf == nullptr)
        imu_preintegrated_from_last_kf = imu::Preintegration::Create(last_frame->GetImuBias());

    if (imu_from_last_frame[0].t - last_frame->time > 0.015) //freq*1.5
    {
        valid_imu_time = imu_from_last_frame[0].t;
        if (Imu::Get()->initialized)
        {
            Imu::Get()->initialized = false;
            status = FrontendStatus::INITIALIZING;
        }
        imu_preintegrated_from_last_kf->bad = true;
        imu_preintegrated_from_last_frame->bad = true;
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            if (imu_from_last_frame[i + 1].t - imu_from_last_frame[i].t > 0.015)
            {
                valid_imu_time = imu_from_last_frame[i + 1].t;
                if (Imu::Get()->initialized)
                {
                    Imu::Get()->initialized = false;
                    status = FrontendStatus::INITIALIZING;
                }
                imu_preintegrated_from_last_kf->bad = true;
                imu_preintegrated_from_last_frame->bad = true;
                break;
            }
            double tstep;
            double tab;
            Vector3d acc0, ang_vel0; //第一帧
            Vector3d acc, ang_vel;   //第二帧
            if ((i == 0) && (i < (n - 1)))
            {
                tab = imu_from_last_frame[i + 1].t - imu_from_last_frame[i].t;
                double tini = imu_from_last_frame[i].t - last_frame->time;
                acc = imu_from_last_frame[i + 1].a;
                acc0 = (imu_from_last_frame[i].a -
                        (imu_from_last_frame[i + 1].a - imu_from_last_frame[i].a) * (tini / tab));
                ang_vel = imu_from_last_frame[i + 1].w;
                ang_vel0 = (imu_from_last_frame[i].w -
                            (imu_from_last_frame[i + 1].w - imu_from_last_frame[i].w) * (tini / tab));
                tstep = imu_from_last_frame[i + 1].t - last_frame->time;
            }
            else if (i < (n - 1))
            {
                acc = (imu_from_last_frame[i + 1].a);
                acc0 = (imu_from_last_frame[i].a);
                ang_vel = (imu_from_last_frame[i + 1].w);
                ang_vel0 = (imu_from_last_frame[i].w);
                tstep = imu_from_last_frame[i + 1].t - imu_from_last_frame[i].t;
            }
            else if ((i > 0) && (i == (n - 1)))
            {
                tab = imu_from_last_frame[i + 1].t - imu_from_last_frame[i].t;
                double tend = imu_from_last_frame[i + 1].t - current_frame->time;

                acc = (imu_from_last_frame[i + 1].a -
                       (imu_from_last_frame[i + 1].a - imu_from_last_frame[i].a) * (tend / tab));
                acc0 = imu_from_last_frame[i].a;
                ang_vel = (imu_from_last_frame[i + 1].w -
                           (imu_from_last_frame[i + 1].w - imu_from_last_frame[i].w) * (tend / tab));
                ang_vel0 = imu_from_last_frame[i].w;
                tstep = current_frame->time - imu_from_last_frame[i].t;
            }
            else if ((i == 0) && (i == (n - 1)))
            {
                tab = imu_from_last_frame[i + 1].t - imu_from_last_frame[i].t;
                double tini = imu_from_last_frame[i].t - last_frame->time;
                double tend = imu_from_last_frame[i + 1].t - current_frame->time;
                acc = (imu_from_last_frame[i + 1].a -
                       (imu_from_last_frame[i + 1].a - imu_from_last_frame[i].a) * (tend / tab));
                acc0 = (imu_from_last_frame[i].a -
                        (imu_from_last_frame[i + 1].a - imu_from_last_frame[i].a) * (tini / tab));
                ang_vel = (imu_from_last_frame[i + 1].w -
                           (imu_from_last_frame[i + 1].w - imu_from_last_frame[i].w) * (tend / tab));
                ang_vel0 = (imu_from_last_frame[i].w -
                            (imu_from_last_frame[i + 1].w - imu_from_last_frame[i].w) * (tini / tab));
                tstep = current_frame->time - last_frame->time;
            }
            if (tab == 0)
                continue;
            imu_preintegrated_from_last_kf->Append(tstep, acc, ang_vel, acc0, ang_vel0);
            imu_preintegrated_from_last_frame->Append(tstep, acc, ang_vel, acc0, ang_vel0);
        }
        if ((n == 0))
        {
            valid_imu_time = imu_from_last_frame[0].t;
            if (Imu::Get()->initialized)
            {
                Imu::Get()->initialized = false;
                status = FrontendStatus::INITIALIZING;
            }
            imu_preintegrated_from_last_kf->bad = true;
            imu_preintegrated_from_last_frame->bad = true;
        }
    }

    if (!imu_preintegrated_from_last_kf->bad) //如果imu帧没坏，就赋给当前帧
        current_frame->preintegration = imu_preintegrated_from_last_kf;
    else
    {
        current_frame->preintegration = nullptr;
        current_frame->bImu = false;
    }

    if (!imu_preintegrated_from_last_frame->bad)
        current_frame->preintegration_last = imu_preintegrated_from_last_frame;
}

void Frontend::InitFrame()
{
    current_frame->last_keyframe = last_keyframe;
    current_frame->pose = last_frame_pose_cache_ * relative_i_j;
    if (Imu::Num())
    {
        current_frame->SetNewBias(last_frame->GetImuBias());
        PreintegrateIMU();
        if (Imu::Get()->initialized)
        {
            PredictStateIMU();
        }
    }
}

bool check_pose(SE3d current_pose, SE3d last_pose, SE3d init_pose)
{
    double current_relative[6], relative[6];
    ceres::SE3ToRpyxyz((last_pose.inverse() * current_pose).data(), current_relative);
    ceres::SE3ToRpyxyz((last_pose.inverse() * init_pose).data(), relative);
    return std::fabs(current_relative[0] - relative[0]) < 0.5 &&
           std::fabs(current_relative[1] - relative[1]) < 0.2 &&
           std::fabs(current_relative[2] - relative[2]) < 0.2 &&
           std::fabs(current_relative[3] - relative[3]) < 2 &&
           std::fabs(current_relative[4] - relative[4]) < 1 &&
           std::fabs(current_relative[5] - relative[5]) < 1;
}

bool Frontend::Track()
{
    SE3d init_pose = current_frame->pose;
    int num_inliers = TrackLastFrame(last_frame);
    bool success = num_inliers > num_features_tracking_bad_ &&
                   check_pose(current_frame->pose, last_frame->pose, init_pose);

    if (!success)
    {
        num_inliers = Relocate(last_frame);
        success = num_inliers > num_features_tracking_bad_ &&
                  check_pose(current_frame->pose, last_frame->pose, init_pose);
    }

    if (status == FrontendStatus::INITIALIZING)
    {
        if (!success)
        {
            status = FrontendStatus::BUILDING;
        }
    }
    else
    {
        if (true || success)
        {
            // tracking good
            status = FrontendStatus::TRACKING_GOOD;
        }
        else
        {
            // tracking bad, but give a chance
            status = FrontendStatus::TRACKING_GOOD;
            current_frame->features_left.clear();
            InitMap();
            if (Navsat::Num() && Navsat::Get()->initialized)
            {
                current_frame->pose = init_pose;
                current_frame->pose.translation() = Navsat::Get()->GetAroundPoint(current_frame->time);
            }
            else
            {
                current_frame->pose = last_frame_pose_cache_ * relative_i_j;
            }
            LOG(INFO) << "Lost, disable cameras!";
            return false;
        }
    }

    if ((status == FrontendStatus::TRACKING_GOOD && num_inliers < num_features_needed_for_keyframe_) ||
        (status == FrontendStatus::INITIALIZING && current_frame->time - last_keyframe->time > 0.25))
    {
        CreateKeyframe();
    }

    // smooth trajectory
    relative_i_j = se3_slerp(relative_i_j, last_frame_pose_cache_.inverse() * current_frame->pose, 0.5);
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
    if ((int)points_2d.size() > num_features_tracking_bad_ &&
        cv::solvePnPRansac(points_3d, points_2d, Camera::Get()->K, Camera::Get()->D, rvec, tvec, false, 100, 8.0F, 0.98, inliers, cv::SOLVEPNP_EPNP))
    {
        cv::Rodrigues(rvec, cv_R);
        Matrix3d R;
        cv::cv2eigen(cv_R, R);
        if (!Imu::Num() || !Imu::Get()->initialized ||
            (Imu::Get()->initialized && current_frame->preintegration_last == nullptr)) //IMU
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
    return num_good_pts;
}

int Frontend::Relocate(Frame::Ptr base_frame)
{
    std::vector<cv::Point2f> kps_left, kps_right, kps_current;
    std::vector<Vector3d> pbs;
    int num_good_pts = matcher_.Relocate(base_frame, current_frame, kps_left, kps_right, kps_current, pbs);
    if (num_good_pts > num_features_tracking_bad_)
    {
        for (int i = 0; i < kps_left.size(); i++)
        {
            auto new_landmark = visual::Landmark::Create(pbs[i]);
            auto new_left_feature = visual::Feature::Create(base_frame, kps_left[i], new_landmark);
            auto new_right_feature = visual::Feature::Create(base_frame, kps_right[i], new_landmark);
            new_right_feature->is_on_left_image = false;
            new_landmark->AddObservation(new_left_feature);
            new_landmark->AddObservation(new_right_feature);
            base_frame->AddFeature(new_left_feature);
            base_frame->AddFeature(new_right_feature);
            Map::Instance().InsertLandmark(new_landmark);
            position_cache_[new_landmark->id] = new_landmark->ToWorld();

            auto feature = visual::Feature::Create(current_frame, kps_current[i], new_landmark);
            current_frame->AddFeature(feature);
        }
    }
    if (base_frame != last_keyframe)
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
        last_keyframe = base_frame;
        current_frame->last_keyframe = base_frame;
        current_frame->preintegration = current_frame->preintegration_last;
        imu_preintegrated_from_last_kf = current_frame->preintegration_last;
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
        return false;

    if (Imu::Num())
    {
        status = FrontendStatus::INITIALIZING;
        Imu::Get()->initialized = false;
        imu_preintegrated_from_last_kf = nullptr;
        valid_imu_time = current_frame->time;
    }
    else
    {
        status = FrontendStatus::TRACKING_GOOD;
    }

    // the first frame is a keyframe
    Map::Instance().InsertKeyFrame(current_frame);
    last_keyframe = current_frame;

    LOG(INFO) << "Initial map created with " << num_new_features << " map points";

    // update backend because we have a new keyframe
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
    if (Imu::Num())
    {
        imu_preintegrated_from_last_kf = nullptr;
    }

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
    last_keyframe = current_frame;
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

//IMU
void Frontend::UpdateIMU(const Bias &bias_)
{
    if (last_keyframe->preintegration != nullptr)
        last_keyframe->bImu = true;
    last_frame->SetNewBias(bias_);
    current_frame->SetNewBias(bias_);
    Vector3d Gz;
    Gz << 0, 0, -Imu::Get()->G;
    Gz = Imu::Get()->Rwg * Gz;
    double dt; // 时间间隔
    Vector3d twb1;
    Matrix3d Rwb1;
    Vector3d Vwb1;
    Vector3d twb2;
    Matrix3d Rwb2;
    Vector3d Vwb2;
    if (last_frame->last_keyframe && last_frame->preintegration)
    {
        if (std::fabs(last_frame->time - last_keyframe->time) > 0.001)
        {
            twb1 = last_frame->last_keyframe->GetImuPosition();
            Rwb1 = last_frame->last_keyframe->GetImuRotation();
            Vwb1 = last_frame->last_keyframe->GetVelocity();
            dt = last_frame->preintegration->sum_dt;
            Rwb2 = Rwb1 * last_frame->preintegration->GetUpdatedDeltaRotation();
            twb2 = twb1 + Vwb1 * dt + 0.5f * dt * dt * Gz + Rwb1 * last_frame->preintegration->GetUpdatedDeltaPosition();
            Vwb2 = Vwb1 + Gz * dt + Rwb1 * last_frame->preintegration->GetUpdatedDeltaVelocity();
            last_frame->SetPose(Rwb2, twb2);
            last_frame->SetVelocity(Vwb2);
        }
    }
    if (std::fabs(current_frame->time - last_keyframe->time) > 0.001 &&
        current_frame->preintegration && current_frame->last_keyframe)
    {
        twb1 = current_frame->last_keyframe->GetImuPosition();
        Rwb1 = current_frame->last_keyframe->GetImuRotation();
        Vwb1 = current_frame->last_keyframe->GetVelocity();
        dt = current_frame->preintegration->sum_dt;
        Rwb2 = Rwb1 * current_frame->preintegration->GetUpdatedDeltaRotation();
        twb2 = twb1 + Vwb1 * dt + 0.5f * dt * dt * Gz + Rwb1 * current_frame->preintegration->GetUpdatedDeltaPosition();
        Vwb2 = Vwb1 + Gz * dt + Rwb1 * current_frame->preintegration->GetUpdatedDeltaVelocity();
        current_frame->SetPose(Rwb2, twb2);
        current_frame->SetVelocity(Vwb2);
    }
}

void Frontend::PredictStateIMU()
{
    if (last_keyframe_updated && last_keyframe)
    {
        Vector3d Gz;
        Gz << 0, 0, -Imu::Get()->G;
        Gz = Imu::Get()->Rwg * Gz;
        double dt = current_frame->preintegration->sum_dt;
        Vector3d twb1 = last_keyframe->GetImuPosition();
        Matrix3d Rwb1 = last_keyframe->GetImuRotation();
        Vector3d Vwb1 = last_keyframe->Vw;

        Matrix3d Rwb2 = NormalizeRotation(Rwb1 * current_frame->preintegration->GetDeltaRotation(last_keyframe->GetImuBias()).toRotationMatrix());
        Vector3d twb2 = twb1 + Vwb1 * dt + 0.5f * dt * dt * Gz + Rwb1 * current_frame->preintegration->GetDeltaPosition(last_keyframe->GetImuBias());
        Vector3d Vwb2 = Vwb1 + dt * Gz + Rwb1 * current_frame->preintegration->GetDeltaVelocity(last_keyframe->GetImuBias());
        current_frame->SetVelocity(Vwb2);
        current_frame->SetPose(Rwb2, twb2);
        current_frame->SetNewBias(last_keyframe->GetImuBias());
        last_keyframe_updated = false; //IMU
    }
    else if (!last_keyframe_updated)
    {
        Vector3d Gz;
        Gz << 0, 0, -Imu::Get()->G;
        Gz = Imu::Get()->Rwg * Gz;
        double dt = current_frame->preintegration_last->sum_dt;
        Vector3d twb1 = last_frame->GetImuPosition();
        Matrix3d Rwb1 = last_frame->GetImuRotation();
        Vector3d Vwb1 = last_frame->Vw;
        Matrix3d Rwb2 = NormalizeRotation(Rwb1 * current_frame->preintegration_last->GetDeltaRotation(last_frame->GetImuBias()).toRotationMatrix());
        Vector3d twb2 = twb1 + Vwb1 * dt + 0.5f * dt * dt * Gz + Rwb1 * current_frame->preintegration_last->GetDeltaPosition(last_frame->GetImuBias());
        Vector3d Vwb2 = Vwb1 + dt * Gz + Rwb1 * current_frame->preintegration_last->GetDeltaVelocity(last_frame->GetImuBias());

        current_frame->SetVelocity(Vwb2);
        current_frame->SetPose(Rwb2, twb2);
        current_frame->SetNewBias(last_frame->GetImuBias());
    }
}

} // namespace lvio_fusion