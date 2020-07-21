#include "lvio_fusion/frontend.h"
#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres_helpers/pose_only_reprojection_error.hpp"
#include "lvio_fusion/ceres_helpers/se3_parameterization.hpp"
#include "lvio_fusion/config.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"

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
    std::unique_lock<std::mutex> lock(last_frame_mutex);
    current_frame = frame;

    switch (status)
    {
    case FrontendStatus::INITING:
        StereoInit();
        break;
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
        StereoInit();
        break;
    }

    last_frame = current_frame;
    last_frame_pose_cache_ = last_frame->pose;
    return true;
}

bool Frontend::Track()
{
    current_frame->pose = relative_motion * last_frame_pose_cache_;
    TrackLastFrame();
    InitFramePoseByPnP();
    int tracking_inliers_ = Optimize();

    static int num_tries = 0;
    if (tracking_inliers_ > num_features_tracking_)
    {
        // tracking good
        status = FrontendStatus::TRACKING_GOOD;
        num_tries = 0;
    }
    else if (tracking_inliers_ > num_features_tracking_bad_)
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

    if (tracking_inliers_ < num_features_needed_for_keyframe_)
    {
        CreateKeyframe();
    }
    relative_motion = current_frame->pose * last_frame_pose_cache_.inverse();
    return true;
}

void Frontend::CreateKeyframe()
{
    // first, add new observations of old points
    for (auto feature_pair : current_frame->left_features)
    {
        auto feature = feature_pair.second;
        auto mp = feature->mappoint.lock();
        mp->AddObservation(feature);
    }
    // detect new features, track in right image and triangulate map points
    DetectNewFeatures();
    // insert!
    map_->InsertKeyFrame(current_frame);
    LOG(INFO) << "Add a keyframe " << current_frame->id;
    // update backend because we have a new keyframe
    backend_->UpdateMap();
}

bool Frontend::InitFramePoseByPnP()
{
    std::vector<cv::Point3d> points_3d;
    std::vector<cv::Point2f> points_2d;
    for (auto feature_pair : current_frame->left_features)
    {
        auto feature = feature_pair.second;
        auto mappoint = feature->mappoint.lock();
        points_2d.push_back(eigen2cv(feature->keypoint));
        Vector3d p = position_cache_[mappoint->id];
        points_3d.push_back(cv::Point3f(p.x(), p.y(), p.z()));
    }

    cv::Mat K;
    cv::eigen2cv(left_camera_->K(), K);
    cv::Mat rvec, tvec, inliers, D, cv_R;
    if (cv::solvePnPRansac(points_3d, points_2d, K, D, rvec, tvec, false, 100, 8.0F, 0.98, cv::noArray(), cv::SOLVEPNP_EPNP))
    {
        cv::Rodrigues(rvec, cv_R);
        Matrix3d R;
        cv::cv2eigen(cv_R, R);
        current_frame->pose = left_camera_->extrinsic.inverse() *
                              SE3d(SO3d(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));
        return true;
    }
    return false;
}

int Frontend::Optimize()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    double *para = current_frame->pose.data();
    ceres::LocalParameterization *local_parameterization = new SE3Parameterization();
    problem.AddParameterBlock(para, SE3d::num_parameters, local_parameterization);

    for (auto feature_pair : current_frame->left_features)
    {
        auto feature = feature_pair.second;
        auto mappoint = feature->mappoint.lock();
        // ceres::CostFunction *cost_function = PoseOnlyReprojectionError::Create(feature->keypoint, left_camera_, position_cache_[mappoint->id]);
        ceres::CostFunction *cost_function;
        cost_function = new PoseOnlyReprojectionError(feature->keypoint, left_camera_, position_cache_[mappoint->id]);
        problem.AddResidualBlock(cost_function, loss_function, para);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 5;
    options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // reject outliers
    // Features left_features = current_frame->left_features;
    // for (auto feature_pair : left_features)
    // {
    //     auto feature = feature_pair.second;
    //     auto mappoint = feature->mappoint.lock();
    //     Vector2d error(0, 0);
    //     PoseOnlyReprojectionError(feature->keypoint, left_camera_, position_cache_[mappoint->id])(current_frame->pose.data(), error.data());
    //     if (error.norm() > 3)
    //     {
    //         current_frame->RemoveFeature(feature);
    //     }
    // }
    Features inliners;
    for (auto feature_pair : current_frame->left_features)
    {
        auto feature = feature_pair.second;
        auto mappoint = feature->mappoint.lock();
        Vector2d error = feature->keypoint - left_camera_->World2Pixel(position_cache_[mappoint->id], current_frame->pose);
        if (error[0] * error[0] + error[1] * error[1] < 9)
        {
            inliners.insert(std::make_pair(feature_pair.first, feature_pair.second));
        }
    }
    current_frame->left_features = inliners;
    // LOG(INFO) << "Current Pose = \n"
    //           << current_frame->pose.matrix();
    return inliners.size();

    // return current_frame->left_features.size();
}

int Frontend::TrackLastFrame()
{
    // use LK flow to estimate points in the last image
    std::vector<cv::Point2f> kps_last, kps_current;
    std::vector<MapPoint::Ptr> mappoints;
    for (auto feature_pair : last_frame->left_features)
    {
        // use project point
        auto feature = feature_pair.second;
        auto mappoint = feature->mappoint.lock();
        auto px = left_camera_->World2Pixel(position_cache_[mappoint->id], current_frame->pose);
        mappoints.push_back(mappoint);
        kps_last.push_back(eigen2cv(feature->keypoint));
        kps_current.push_back(cv::Point2f(px[0], px[1]));
    }

    std::vector<uchar> status;
    cv::Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame->left_image, current_frame->left_image, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // NOTE: don‘t use, accuracy is very low
    // cv::findFundamentalMat(kps_current, kps_last, cv::FM_RANSAC, 3.0, 0.9, status);

    int num_good_pts = 0;
    //DEBUG
    cv::Mat img_track = current_frame->left_image;
    cv::cvtColor(img_track, img_track, CV_GRAY2RGB);
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i])
        {
            cv::arrowedLine(img_track, kps_current[i], kps_last[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            auto feature = Feature::Create(current_frame, cv2eigen(kps_current[i]), mappoints[i]);
            current_frame->AddFeature(feature);
            num_good_pts++;
        }
    }
    cv::imshow("tracking", img_track);
    cv::waitKey(1);
    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::StereoInit()
{
    int num_new_features = DetectNewFeatures();
    if (num_new_features < num_features_init_)
    {
        return false;
    }
    status = FrontendStatus::TRACKING_GOOD;

    // the first frame is a keyframe
    map_->InsertKeyFrame(current_frame);
    LOG(INFO) << "Initial map created with " << num_new_features << " map points";

    // update backend because we have a new keyframe
    backend_->UpdateMap();
    return true;
}

int Frontend::DetectNewFeatures()
{
    if (current_frame->left_features.size() >= num_features_)
        return -1;

    cv::Mat mask(current_frame->left_image.size(), CV_8UC1, 255);
    for (auto feature_pair : current_frame->left_features)
    {
        auto feature = feature_pair.second;
        cv::rectangle(mask, eigen2cv(feature->keypoint - Vector2d(10, 10)), eigen2cv(feature->keypoint + Vector2d(10, 10)), 0, CV_FILLED);
    }

    std::vector<cv::Point2f> kps_left, kps_right;

    // std::vector<cv::KeyPoint> keypoints;
    // fast_->detect(current_frame->left_image, keypoints, mask);
    // cv::KeyPointsFilter::retainBest(keypoints, num_features_ - current_frame->left_features.size());
    // cv::KeyPoint::convert(keypoints, kps_left);

    cv::goodFeaturesToTrack(current_frame->left_image, kps_left, num_features_ - current_frame->left_features.size(), 0.01, 30, mask);

    // use LK flow to estimate points in the right image
    kps_right = kps_left;
    std::vector<uchar> status;
    cv::Mat error;
    cv::calcOpticalFlowPyrLK(
        current_frame->left_image, current_frame->right_image, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // triangulate new points
    std::vector<SE3d> poses{left_camera_->extrinsic, right_camera_->extrinsic};
    SE3d current_pose_Twc = current_frame->pose.inverse();
    int num_triangulated_pts = 0;
    int num_good_pts = 0;
    for (size_t i = 0; i < kps_left.size(); ++i)
    {
        if (status[i])
        {
            num_good_pts++;
            // triangulation
            std::vector<Vector3d> points{
                left_camera_->Pixel2Sensor(cv2eigen(kps_left[i])),
                right_camera_->Pixel2Sensor(cv2eigen(kps_right[i]))};
            Vector3d pworld = Vector3d::Zero();

            // clang-format off
            if (triangulation(poses, points, pworld))
                // && (left_camera_->World2Pixel(pworld, SE3d()) - cv2eigen(kps_left[i])).norm() < 0.2
                // && (right_camera_->World2Pixel(pworld, SE3d()) - cv2eigen(kps_right[i])).norm() < 0.2)
            // clang-format on
            {
                auto new_mappoint = MapPoint::Create(pworld.z(), nullptr);
                auto new_left_feature = Feature::Create(current_frame, cv2eigen(kps_left[i]), new_mappoint);
                auto new_right_feature = Feature::Create(current_frame, cv2eigen(kps_right[i]), new_mappoint);
                new_right_feature->is_on_left_image = false;
                new_mappoint->AddObservation(new_left_feature);
                new_mappoint->AddObservation(new_right_feature);
                current_frame->AddFeature(new_left_feature);
                current_frame->AddFeature(new_right_feature);
                map_->InsertMapPoint(new_mappoint);
                position_cache_[new_mappoint->id] = current_pose_Twc * pworld;
                num_triangulated_pts++;
            }
        }
    }

    LOG(INFO) << "Detect " << kps_left.size() << " new features";
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    LOG(INFO) << "new landmarks: " << num_triangulated_pts;
    return num_triangulated_pts;
}

bool Frontend::Reset()
{
    backend_->Pause();
    map_->Reset();
    backend_->Continue();
    status = FrontendStatus::INITING;
    LOG(INFO) << "Reset Succeed";
    return true;
}

void Frontend::UpdateCache()
{
    // position_cache_.clear();
    // for (auto feature_pair : last_frame->left_features)
    // {
    //     auto feature = feature_pair.second;
    //     auto mappoint = feature->mappoint.lock();
    //     position_cache_.insert(std::make_pair(mappoint->id, mappoint->Position()));
    // }
    // last_frame_pose_cache_ = last_frame->pose;
}

std::unordered_map<unsigned long, Vector3d> Frontend::GetPositionCache()
{
    std::unique_lock<std::mutex> lock(last_frame_mutex);
    return position_cache_;
}

} // namespace lvio_fusion