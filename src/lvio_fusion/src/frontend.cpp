#include "lvio_fusion/frontend.h"
#include "lvio_fusion/backend.h"
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
        InitMap();
        break;
    case FrontendStatus::INITIALIZING:
    case FrontendStatus::TRACKING:
    case FrontendStatus::LOST:
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

bool check_velocity(SE3d &current_pose, SE3d last_pose, double dt)
{
    double relative[6], abs[6];
    ceres::SE3ToRpyxyz((last_pose.inverse() * current_pose).data(), relative);
    for (int i = 0; i < 6; i++)
    {
        abs[i] = std::fabs(relative[i]);
    }
    // speed is lower than 40m/s, and vy,vz change slowly.
    if (relative[3] / dt < max_speed)
    {
        relative[0] = (relative[0] / abs[0]) * std::min(abs[0], 0.2);
        relative[1] = (relative[1] / abs[1]) * std::min(abs[1], 0.1);
        relative[2] = (relative[2] / abs[2]) * std::min(abs[2], 0.01);
        relative[4] = (relative[0] / abs[0]) * std::min(tan(abs[0]) * relative[3], abs[4]);
        relative[5] = (relative[1] / abs[1]) * std::min(tan(abs[1]) * relative[3], abs[5]);
        SE3d relative_i_j;
        ceres::RpyxyzToSE3(relative, relative_i_j.data());
        current_pose = last_pose * relative_i_j;
        return true;
    }
    else
    {
        return false;
    }
}

void Frontend::InitFrame()
{
    dt_ = current_frame->time - last_frame->time;

    current_frame->last_keyframe = last_keyframe;
    SE3d init_pose = last_frame_pose_cache_ * relative_i_j_;
    bool success = false;
    if (Imu::Num())
    {
        current_frame->SetBias(last_frame->bias);
        Preintegrate();
        if (Imu::Get()->initialized)
        {
            PredictState();
            success = check_velocity(current_frame->pose, last_frame_pose_cache_, dt_);
        }
    }
    else if (Navsat::Num() && Navsat::Get()->initialized)
    {
        static bool has_last_pose = false;
        static SE3d last_navsat_pose;
        SE3d navsat_pose = Navsat::Get()->GetAroundPose(current_frame->time);
        if (has_last_pose)
        {
            // relative navsat pose
            current_frame->pose = last_frame_pose_cache_ * last_navsat_pose.inverse() * navsat_pose;
            success = check_velocity(current_frame->pose, last_frame_pose_cache_, dt_);
            LOG(INFO) << (success ? "success" : "failed");
        }
        last_navsat_pose = navsat_pose;
        has_last_pose = true;
    }
    if (!success)
    {
        current_frame->pose = init_pose;
    }
}

bool Frontend::Track()
{
    SE3d init_pose = current_frame->pose;
    int num_inliers = TrackLastFrame();

    if (num_inliers)
    {
        // tracking good
        status = FrontendStatus::TRACKING;
    }
    else if (Imu::Num() && Imu::Get()->initialized)
    {
        // tracking bad, disable imu and try again
        ResetImu();
        current_frame->pose = last_frame_pose_cache_ * relative_i_j_;
        return Track();
    }
    else
    {
        // tracking bad, init map again
        status = FrontendStatus::LOST;
        current_frame->features_left.clear();
        InitMap();
        current_frame->pose = init_pose;
        LOG(INFO) << "Lost, init map again!";
        return false;
    }

    if (num_inliers < num_features_needed_for_keyframe_)
    {
        CreateKeyframe();
    }
    return true;
}

void Frontend::ResetImu()
{
    Imu::Get()->initialized = false;
    init_time = current_frame->time + epsilon;
    status = FrontendStatus::INITIALIZING;
    current_frame->good_imu = false;
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
                success = inliers.rows > num_features_tracking_bad_ && check_velocity(current_pose, last_frame_pose_cache_, dt_);
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
        ResetImu();
    }
    else
    {
        status = FrontendStatus::TRACKING;
    }

    // the first frame is a keyframe
    Map::Instance().InsertKeyFrame(current_frame);
    last_keyframe = current_frame;
    preintegration_last_kf_ = nullptr;
    LOG(INFO) << "Initial map created with " << num_new_features << " map points";

    // update backend because we have a new keyframe
    backend_.lock()->UpdateMap();
    return true;
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
    local_map.AddKeyFrame(current_frame);

    // insert!
    Map::Instance().InsertKeyFrame(current_frame);
    last_keyframe = current_frame;
    preintegration_last_kf_ = nullptr;
    LOG(INFO) << "Add a keyframe " << current_frame->id;
    if(globalplanner_)//NAVI
    {
        Vector3d pose3d =last_frame->pose.translation();
        Vector2d pose2d(pose3d[0],pose3d[1]);
        globalplanner_->SetRobotPose(pose2d);
    }
    // update backend because we have a new keyframe
    backend_.lock()->UpdateMap();
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
    auto last_keyframe2 = last_keyframe->last_keyframe;
    if (last_keyframe2)
    {
        relative_i_j_ = se3_slerp(SE3d(), last_keyframe2->pose.inverse() * last_keyframe->pose, dt_ / (last_keyframe->time - last_keyframe2->time));
    }
}

void Frontend::UpdateImu(const Bias &bias_)
{
    if (last_keyframe->preintegration)
        last_keyframe->good_imu = true;

    last_frame->SetBias(bias_);
    current_frame->SetBias(bias_);
    Vector3d G(0, 0, -Imu::Get()->G);
    if (last_frame != last_keyframe && last_frame->preintegration)
    {
        double sum_dt = last_frame->preintegration->sum_dt;
        Vector3d twb1 = last_frame->last_keyframe->GetPosition();
        Matrix3d Rwb1 = last_frame->last_keyframe->GetRotation();
        Vector3d Vwb1 = last_frame->last_keyframe->Vw;
        Matrix3d Rwb2 = Rwb1 * last_frame->preintegration->GetUpdatedDeltaRotation();
        Vector3d twb2 = twb1 + Vwb1 * sum_dt + 0.5f * sum_dt * sum_dt * G + Rwb1 * last_frame->preintegration->GetUpdatedDeltaPosition();
        Vector3d Vwb2 = Vwb1 + G * sum_dt + Rwb1 * last_frame->preintegration->GetUpdatedDeltaVelocity();
        last_frame->SetPose(Rwb2, twb2);
        last_frame->SetVelocity(Vwb2);
    }
}

void Frontend::Preintegrate()
{
    // get imu data fron last frame
    std::vector<ImuData> imu_from_last_frame;
    int timeout = dt_ * 1e3; // 100ms
    while (timeout)
    {
        if (imu_buf_.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

    // preintegrate
    int n = imu_from_last_frame.size();
    auto preintegration_last_frame = imu::Preintegration::Create(last_frame->bias);
    preintegration_last_kf_ = preintegration_last_kf_ ? preintegration_last_kf_ : imu::Preintegration::Create(last_frame->bias);
    if (n < 4)
    {
        // If n is smaller than 4, initialize again.
        ResetImu();
    }
    else
    {
        for (int i = 0; i < n - 1; i++)
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
            else if (i < n - 2)
            {
                acc = imu_from_last_frame[i + 1].a;
                acc0 = imu_from_last_frame[i].a;
                ang_vel = imu_from_last_frame[i + 1].w;
                ang_vel0 = imu_from_last_frame[i].w;
                tstep = imu_from_last_frame[i + 1].t - imu_from_last_frame[i].t;
            }
            else if (i == n - 2)
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
            preintegration_last_kf_->Append(tstep, acc, ang_vel, acc0, ang_vel0);
            preintegration_last_frame->Append(tstep, acc, ang_vel, acc0, ang_vel0);
        }

        if (Imu::Get()->initialized)
        {
            current_frame->good_imu = true;
        }
        current_frame->preintegration = preintegration_last_kf_;
        current_frame->preintegration_last = preintegration_last_frame;
    }
}

void Frontend::PredictState()
{
    Vector3d G(0, 0, -Imu::Get()->G);
    double sum_dt = current_frame->preintegration_last->sum_dt;
    Vector3d twb1 = last_frame->GetPosition();
    Matrix3d Rwb1 = last_frame->GetRotation();
    Vector3d Vwb1 = last_frame->Vw;
    Matrix3d Rwb2 = normalize_R(Rwb1 * current_frame->preintegration_last->GetDeltaRotation(last_frame->bias).toRotationMatrix());
    Vector3d twb2 = twb1 + Vwb1 * sum_dt + 0.5f * sum_dt * sum_dt * G + Rwb1 * current_frame->preintegration_last->GetDeltaPosition(last_frame->bias);
    Vector3d Vwb2 = Vwb1 + sum_dt * G + Rwb1 * current_frame->preintegration_last->GetDeltaVelocity(last_frame->bias);

    current_frame->SetVelocity(Vwb2);
    current_frame->SetPose(Rwb2, twb2);
    current_frame->SetBias(last_frame->bias);
}

} // namespace lvio_fusion