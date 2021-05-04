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
    : num_features_init_(init), num_features_tracking_bad_(tracking_bad), num_features_needed_for_keyframe_(need_for_keyframe), local_map(num_features)
{
}

cv::Mat img_track;
bool Frontend::AddFrame(Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(mutex);
    current_frame = frame;
    cv::cvtColor(current_frame->image_left, img_track, cv::COLOR_GRAY2RGB);
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
        break;
    }
    cv::imshow("tracking", img_track);
    cv::waitKey(1);
    last_frame = current_frame;
    last_frame_pose_cache_ = last_frame->pose;
    return true;
}

void Frontend::AddImu(double time, Vector3d acc, Vector3d gyr)
{
    imu_buf_.push(ImuData(acc, gyr, time));
}

void Frontend::PreintegrateImu()
{
    if (!last_frame)
        return;

    std::vector<ImuData> imu_from_last_frame;
    int timeout = 100; // ms
    while (timeout)
    {
        if (imu_buf_.empty())
        {
            usleep(1e3);
            timeout--;
            continue;
        }
        ImuData imu_data = imu_buf_.front();
        if (imu_data.t < last_frame->time - epsilon)
        {
            imu_buf_.pop();
        }
        else if (imu_data.t < current_frame->time - epsilon)
        {
            imu_from_last_frame.push_back(imu_data);
            imu_buf_.pop();
        }
        else
        {
            imu_from_last_frame.push_back(imu_data);
            break;
        }
    }

    const int n = imu_from_last_frame.size() - 1;
    imu::Preintegration::Ptr imu_preintegrated_from_last_frame = imu::Preintegration::Create(last_frame->GetImuBias());
    if (imu_preintegrated_from_last_kf_ == nullptr)
        imu_preintegrated_from_last_kf_ = imu::Preintegration::Create(last_frame->GetImuBias());

    bool last_kf_bad = false, last_frame_bad = false;
    if (n < 5)
    {
        valid_imu_time = last_frame->time + epsilon;
        if (Imu::Get()->initialized)
        {
            Imu::Get()->initialized = false;
            status = FrontendStatus::INITIALIZING;
        }
        last_kf_bad = true;
        last_frame_bad = true;
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            double tstep;
            double tab;
            Vector3d acc0, ang_vel0;
            Vector3d acc, ang_vel;
            if (i == 0)
            {
                tab = imu_from_last_frame[i + 1].t - imu_from_last_frame[i].t;
                double tini = imu_from_last_frame[i].t - last_frame->time;
                acc = imu_from_last_frame[i + 1].a;
                acc0 = imu_from_last_frame[i].a - (imu_from_last_frame[i + 1].a - imu_from_last_frame[i].a) * (tini / tab);
                ang_vel = imu_from_last_frame[i + 1].w;
                ang_vel0 = imu_from_last_frame[i].w - (imu_from_last_frame[i + 1].w - imu_from_last_frame[i].w) * (tini / tab);
                tstep = imu_from_last_frame[i + 1].t - last_frame->time;
            }
            else if (i < n - 1)
            {
                acc = imu_from_last_frame[i + 1].a;
                acc0 = imu_from_last_frame[i].a;
                ang_vel = imu_from_last_frame[i + 1].w;
                ang_vel0 = imu_from_last_frame[i].w;
                tstep = imu_from_last_frame[i + 1].t - imu_from_last_frame[i].t;
            }
            else if (i == n - 1)
            {
                tab = imu_from_last_frame[i + 1].t - imu_from_last_frame[i].t;
                double tend = imu_from_last_frame[i + 1].t - current_frame->time;
                acc = imu_from_last_frame[i + 1].a - (imu_from_last_frame[i + 1].a - imu_from_last_frame[i].a) * (tend / tab);
                acc0 = imu_from_last_frame[i].a;
                ang_vel = imu_from_last_frame[i + 1].w - (imu_from_last_frame[i + 1].w - imu_from_last_frame[i].w) * (tend / tab);
                ang_vel0 = imu_from_last_frame[i].w;
                tstep = current_frame->time - imu_from_last_frame[i].t;
            }
            if (tab == 0)
                continue;
            imu_preintegrated_from_last_kf_->Append(tstep, acc, ang_vel, acc0, ang_vel0);
            imu_preintegrated_from_last_frame->Append(tstep, acc, ang_vel, acc0, ang_vel0);
        }
    }

    if (!last_kf_bad)
    {
        if (Imu::Get()->initialized)
        {
            current_frame->is_imu_good = true;
        }
        current_frame->preintegration = imu_preintegrated_from_last_kf_;
    }
    else
    {
        current_frame->preintegration = nullptr;
        current_frame->is_imu_good = false;
    }

    if (!last_frame_bad)
        current_frame->preintegration_last = imu_preintegrated_from_last_frame;
}

bool check_pose(SE3d current_pose, SE3d last_pose, SE3d init_pose)
{
    double current_relative[6], relative[6];
    ceres::SE3ToRpyxyz((last_pose.inverse() * current_pose).data(), current_relative);
    ceres::SE3ToRpyxyz((last_pose.inverse() * init_pose).data(), relative);
    // TODO： stat and change
    return std::fabs(current_relative[0] - relative[0]) < 0.5 &&
           std::fabs(current_relative[1] - relative[1]) < 0.2 &&
           std::fabs(current_relative[2] - relative[2]) < 0.2 &&
           std::fabs(current_relative[3] - relative[3]) < 2 &&
           std::fabs(current_relative[4] - relative[4]) < 1 &&
           std::fabs(current_relative[5] - relative[5]) < 1;
}

void Frontend::InitFrame()
{
    current_frame->last_keyframe = last_keyframe;
    if (current_frame->pose.translation() == Vector3d::Zero())
    {
        current_frame->pose = last_frame_pose_cache_ * relative_i_j;
        if (Imu::Num())
        {
            current_frame->SetNewBias(last_frame->GetImuBias());
            PreintegrateImu();
            if (Imu::Get()->initialized)
            {
                PredictStateImu();
            }
        }
        if (Navsat::Num() && Navsat::Get()->initialized)
        {
            static bool has_last_pose = false;
            static SE3d last_navsat_pose;
            SE3d navsat_pose = Navsat::Get()->GetAroundPose(current_frame->time);
            if (has_last_pose)
            {
                // relative navsat pose
                SE3d current_pose = last_frame_pose_cache_ * last_navsat_pose.inverse() * navsat_pose;
                if (check_pose(current_pose, last_frame_pose_cache_, current_frame->pose))
                {
                    current_frame->pose = current_pose;
                }
            }
            last_navsat_pose = navsat_pose;
            has_last_pose = true;
        }
    }
}

bool Frontend::Track()
{
    SE3d init_pose = current_frame->pose;
    int num_inliers = TrackLastFrame();

    if (status == FrontendStatus::INITIALIZING)
    {
        if (Imu::Num() && Imu::Get()->initialized)
        {
            status = FrontendStatus::TRACKING_GOOD;
        }
        else if (!num_inliers)
        {
            status = FrontendStatus::BUILDING;
        }
    }
    if (status != FrontendStatus::INITIALIZING)
    {
        if (num_inliers)
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
            current_frame->pose = init_pose;
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

int Frontend::TrackLastFrame()
{
    std::vector<cv::Point2f> kps_last, kps_current;
    std::vector<visual::Landmark::Ptr> landmarks;
    std::vector<uchar> status;
    // use LK flow to estimate points in the last image
    kps_last.reserve(last_frame->features_left.size());
    kps_current.reserve(last_frame->features_left.size());
    for (auto &pair : last_frame->features_left)
    {
        // use project point
        auto feature = pair.second;
        auto landmark = feature->landmark.lock();
        auto px = Camera::Get()->World2Pixel(local_map.position_cache[landmark->id], current_frame->pose);
        kps_last.push_back(feature->keypoint.pt);
        kps_current.push_back(cv::Point2f(px[0], px[1]));
        landmarks.push_back(landmark);
    }
    // if last frame is a key frame, use new landmarks
    if (last_frame == last_keyframe)
    {
        auto features = local_map.GetFeatures(last_frame->time);
        kps_last.reserve(kps_last.size() + features.size());
        kps_current.reserve(kps_current.size() + features.size());
        for (auto &feature : features)
        {
            auto landmark = feature->landmark.lock();
            assert(landmark);
            auto px = Camera::Get()->World2Pixel(local_map.position_cache[landmark->id], current_frame->pose);
            kps_last.push_back(feature->keypoint.pt);
            kps_current.push_back(cv::Point2f(px[0], px[1]));
            landmarks.push_back(landmark);
        }
    }
    optical_flow(last_frame->image_left, current_frame->image_left, kps_last, kps_current, status);

    // Solve PnP
    std::vector<cv::Point3f> points_3d_far, points_3d_near;
    std::vector<cv::Point2f> points_2d_far, points_2d_near;
    std::vector<int> map_far, map_near;
    for (int i = 0; i < status.size(); ++i)
    {
        if (status[i])
        {
            if (Camera::Get()->Far(local_map.position_cache[landmarks[i]->id], current_frame->pose))
            {
                map_far.push_back(i);
                points_2d_far.push_back(kps_current[i]);
                Vector3d pw = local_map.position_cache[landmarks[i]->id];
                points_3d_far.push_back(cv::Point3f(pw.x(), pw.y(), pw.z()));
            }
            else
            {
                map_near.push_back(i);
                points_2d_near.push_back(kps_current[i]);
                Vector3d pw = local_map.position_cache[landmarks[i]->id];
                points_3d_near.push_back(cv::Point3f(pw.x(), pw.y(), pw.z()));
            }
        }
    }

    bool use_imu = Imu::Num() && Imu::Get()->initialized && current_frame->preintegration_last;
    bool use_pnp = (int)points_2d_near.size() > num_features_tracking_bad_;
    bool use_far = (int)points_2d_far.size() > num_features_tracking_bad_;
    int num_good_pts = 0;
    if (use_far || use_pnp)
    {
        // near
        bool success = false;
        if (use_pnp && !use_imu)
        {
            cv::Mat rvec, tvec, inliers, cv_R;
            success = cv::solvePnPRansac(points_3d_near, points_2d_near, Camera::Get()->K, Camera::Get()->D, rvec, tvec, false, 100, 8.0F, 0.98, inliers, cv::SOLVEPNP_EPNP);
            SE3d current_pose;
            if (success)
            {
                cv::Rodrigues(rvec, cv_R);
                Matrix3d R;
                cv::cv2eigen(cv_R, R);
                current_pose = (Camera::Get()->extrinsic * SE3d(SO3d(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)))).inverse();
                success = inliers.rows > num_features_tracking_bad_ && check_pose(current_pose, last_frame_pose_cache_, current_frame->pose);
            }
            if (success)
            {
                current_frame->pose = current_pose;
                for (int r = 0; r < inliers.rows; r++)
                {
                    int i = map_near[inliers.at<int>(r)];
                    cv::arrowedLine(img_track, kps_current[i], kps_last[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
                    cv::circle(img_track, kps_current[i], 2, cv::Scalar(0, 255, 0), cv::FILLED);
                    auto feature = visual::Feature::Create(current_frame, cv::KeyPoint(kps_current[i], 1), landmarks[i]);
                    current_frame->AddFeature(feature);
                    num_good_pts++;
                }
            }
        }
        if (use_imu || !success)
        {
            for (auto &i : map_near)
            {
                cv::arrowedLine(img_track, kps_current[i], kps_last[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
                cv::circle(img_track, kps_current[i], 2, cv::Scalar(0, 255, 0), cv::FILLED);
                auto feature = visual::Feature::Create(current_frame, cv::KeyPoint(kps_current[i], 1), landmarks[i]);
                current_frame->AddFeature(feature);
                num_good_pts++;
            }
            success = true;
        }

        // far
        for (auto &i : map_far)
        {
            cv::arrowedLine(img_track, kps_current[i], kps_last[i], cv::Scalar(0, 0, 255), 1, 8, 0, 0.2);
            cv::circle(img_track, kps_current[i], 2, cv::Scalar(0, 0, 255), cv::FILLED);
            auto feature = visual::Feature::Create(current_frame, cv::KeyPoint(kps_current[i], 1), landmarks[i]);
            current_frame->AddFeature(feature);
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::InitMap()
{
    int num_new_features = local_map.Init(current_frame);
    if (num_new_features < num_features_init_)
        return false;

    if (Imu::Num())
    {
        status = FrontendStatus::INITIALIZING;
        Imu::Get()->initialized = false;
        imu_preintegrated_from_last_kf_ = nullptr;
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

void Frontend::CreateKeyframe()
{
    if (Imu::Num())
    {
        imu_preintegrated_from_last_kf_ = nullptr;
    }

    // first, add new observations of old points
    for (auto &pair_feature : current_frame->features_left)
    {
        auto feature = pair_feature.second;
        auto landmark = feature->landmark.lock();
        landmark->AddObservation(feature);
    }

    // detect new features, track in right image and triangulate map points
    local_map.AddKeyFrame(current_frame);

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
    local_map.UpdateCache();
    for (auto &pair_feature : last_frame->features_left)
    {
        auto feature = pair_feature.second;
        auto landmark = feature->landmark.lock();
        if (local_map.position_cache.find(landmark->id) == local_map.position_cache.end())
        {
            local_map.position_cache[landmark->id] = landmark->ToWorld();
        }
    }
    last_frame_pose_cache_ = last_frame->pose;
}

void Frontend::UpdateImu(const Bias &bias_)
{
    if (last_keyframe->preintegration != nullptr)
        last_keyframe->is_imu_good = true;

    last_frame->SetNewBias(bias_);
    current_frame->SetNewBias(bias_);
    Vector3d Gz;
    Gz << 0, 0, -Imu::Get()->G;
    Gz = Imu::Get()->Rwg * Gz;
    double dt;
    Vector3d twb1, Vwb1, twb2, Vwb2;
    Matrix3d Rwb1, Rwb2;
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

void Frontend::PredictStateImu()
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
        Matrix3d Rwb2 = normalize_R(Rwb1 * current_frame->preintegration->GetDeltaRotation(last_keyframe->GetImuBias()).toRotationMatrix());
        Vector3d twb2 = twb1 + Vwb1 * dt + 0.5f * dt * dt * Gz + Rwb1 * current_frame->preintegration->GetDeltaPosition(last_keyframe->GetImuBias());
        Vector3d Vwb2 = Vwb1 + dt * Gz + Rwb1 * current_frame->preintegration->GetDeltaVelocity(last_keyframe->GetImuBias());
        current_frame->SetVelocity(Vwb2);
        current_frame->SetPose(Rwb2, twb2);
        current_frame->SetNewBias(last_keyframe->GetImuBias());
        last_keyframe_updated = false;
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
        Matrix3d Rwb2 = normalize_R(Rwb1 * current_frame->preintegration_last->GetDeltaRotation(last_frame->GetImuBias()).toRotationMatrix());
        Vector3d twb2 = twb1 + Vwb1 * dt + 0.5f * dt * dt * Gz + Rwb1 * current_frame->preintegration_last->GetDeltaPosition(last_frame->GetImuBias());
        Vector3d Vwb2 = Vwb1 + dt * Gz + Rwb1 * current_frame->preintegration_last->GetDeltaVelocity(last_frame->GetImuBias());
        current_frame->SetVelocity(Vwb2);
        current_frame->SetPose(Rwb2, twb2);
        current_frame->SetNewBias(last_frame->GetImuBias());
    }
}

} // namespace lvio_fusion