#include <opencv2/opencv.hpp>

#include "lvio_fusion/utility.h"
#include "lvio_fusion/backend.h"
#include "lvio_fusion/config.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/ceres_helper/pose_only_reprojection_error.hpp"
#include "lvio_fusion/ceres_helper/se3_parameterization.hpp"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{

Frontend::Frontend() 
{
    gftt_ =
        cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(lvio_fusion::Frame::Ptr frame)
{
    current_frame = frame;

    switch (status)
    {
    case FrontendStatus::INITING:
        StereoInit();
        break;
    case FrontendStatus::TRACKING_GOOD:
    case FrontendStatus::TRACKING_BAD:
        Track();
        break;
    case FrontendStatus::LOST:
        Reset();
        return false;
    }
    if(!current_frame->objects.empty())
    {
        current_frame->UpdateLabel();
    }
    last_frame = current_frame;
    return true;
}

bool Frontend::Track()
{
    if (last_frame)
    {
        current_frame->SetPose(relative_motion * last_frame->Pose());
    }

    int num_track_last = TrackLastFrame();
    tracking_inliers_ = EstimateCurrentPose();

    if (tracking_inliers_ > num_features_tracking_)
    {
        // tracking good
        status = FrontendStatus::TRACKING_GOOD;
    }
    else if (tracking_inliers_ > num_features_tracking_bad_)
    {
        // tracking bad
        status = FrontendStatus::TRACKING_BAD;
    }
    else
    {
        // lost
        status = FrontendStatus::LOST;
    }

    InsertKeyframe();
    relative_motion = current_frame->Pose() * last_frame->Pose().inverse();
    double dt = current_frame->time - last_frame->time;
    current_frame->velocity = relative_motion.translation() / dt;
    return true;
}

bool Frontend::InsertKeyframe()
{
    if (tracking_inliers_ >= num_features_needed_for_keyframe_)
    {
        // still have enough features, don't insert keyframe
        return false;
    }
    // current frame is a new keyframe
    current_frame->SetKeyFrame();
    map_->InsertKeyFrame(current_frame);

    LOG(INFO) << "Set frame " << current_frame->id << " as keyframe "
              << current_frame->keyframe_id;

    SetObservationsForKeyFrame();
    DetectFeatures(); // detect new features

    // track in right image
    FindFeaturesInRight();
    // triangulate map points
    TriangulateNewPoints();
    // update backend because we have a new keyframe
    backend_->UpdateMap();

    return true;
}

void Frontend::SetObservationsForKeyFrame()
{
    for (auto &feat : current_frame->features_left)
    {
        auto mp = feat->map_point.lock();
        if (mp)
            mp->AddObservation(feat);
    }
}

int Frontend::TriangulateNewPoints()
{
    std::vector<SE3> poses{camera_left->Pose(), camera_right->Pose()};
    SE3 current_pose_Twc = current_frame->Pose().inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < current_frame->features_left.size(); ++i)
    {
        if (current_frame->features_left[i]->map_point.expired() &&
            current_frame->features_right[i] != nullptr)
        {
            // triangulation
            std::vector<Vector3d> points{
                camera_left->pixel2camera(
                    Vector2d(current_frame->features_left[i]->pos.pt.x,
                         current_frame->features_left[i]->pos.pt.y)),
                camera_right->pixel2camera(
                    Vector2d(current_frame->features_right[i]->pos.pt.x,
                         current_frame->features_right[i]->pos.pt.y))};
            Vector3d pworld = Vector3d::Zero();

            if (triangulation(poses, points, pworld))
            {
                auto new_map_point = MapPoint::CreateNewMappoint();
                pworld = current_pose_Twc * pworld;
                new_map_point->SetPos(pworld);
                new_map_point->AddObservation(
                    current_frame->features_left[i]);
                new_map_point->AddObservation(
                    current_frame->features_right[i]);

                current_frame->features_left[i]->map_point = new_map_point;
                current_frame->features_right[i]->map_point = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

int Frontend::EstimateCurrentPose()
{
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 5;
    ceres::Solver::Summary summary;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

    double *para = current_frame->Pose().data();
    ceres::LocalParameterization *local_parameterization = new SE3Parameterization();
    problem.AddParameterBlock(para, SE3::num_parameters, local_parameterization);

    int feature_count = 0;
    for (auto &feature : current_frame->features_left)
    {
        auto map_point = feature->map_point.lock();
        if (map_point)
        {
            feature_count++;
            ceres::CostFunction *cost_function;
            cost_function = new PoseOnlyReprojectionError(toVector2d(feature->pos.pt), camera_left, map_point->Pos());
            problem.AddResidualBlock(cost_function, loss_function, para);
        }
    }

    ceres::Solve(options, &problem, &summary);

    // reject outliers
    for (auto &feature : current_frame->features_left)
    {
        auto map_point = feature->map_point.lock();
        if (map_point)
        {
            Vector2d error = toVector2d(feature->pos.pt) - camera_left->world2pixel(map_point->Pos(),current_frame->Pose());
            if (error[0] * error[0] + error[1] * error[1] > 9)
            {
                feature->map_point.reset();
                //NOTE: maybe we can still use it in future
                feature->is_outlier = false;
                feature_count--;
            }
        }
    }

    LOG(INFO) << "Current Pose = \n"
              << current_frame->Pose().matrix();

    return feature_count;
}

int Frontend::TrackLastFrame()
{
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_last, kps_current;
    for (auto &kp : last_frame->features_left)
    {
        if (kp->map_point.lock())
        {
            // use project point
            auto mp = kp->map_point.lock();
            auto px =
                camera_left->world2pixel(mp->Pos(), current_frame->Pose());
            kps_last.push_back(kp->pos.pt);
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        }
        else
        {
            kps_last.push_back(kp->pos.pt);
            kps_current.push_back(kp->pos.pt);
        }
    }

    std::vector<uchar> status;
    cv::Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame->left_image, current_frame->left_image, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;

    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i])
        {
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame, kp));
            feature->map_point = last_frame->features_left[i]->map_point;
            current_frame->features_left.push_back(feature);
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::StereoInit()
{
    int num_features_left = DetectFeatures();
    int num_coor_features = FindFeaturesInRight();
    if (num_coor_features < num_features_init_)
    {
        return false;
    }

    bool build_map_success = BuildInitMap();
    if (build_map_success)
    {
        status = FrontendStatus::TRACKING_GOOD;
        return true;
    }
    return false;
}

int Frontend::DetectFeatures()
{
    cv::Mat mask(current_frame->left_image.size(), CV_8UC1, 255);
    for (auto &feat : current_frame->features_left)
    {
        cv::rectangle(mask, feat->pos.pt - cv::Point2f(10, 10),
                      feat->pos.pt + cv::Point2f(10, 10), 0, CV_FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    gftt_->detect(current_frame->left_image, keypoints, mask);
    int cnt_detected = 0;
    for (auto &kp : keypoints)
    {
        current_frame->features_left.push_back(
            Feature::Ptr(new Feature(current_frame, kp)));
        cnt_detected++;
    }
    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

int Frontend::FindFeaturesInRight()
{
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : current_frame->features_left)
    {
        kps_left.push_back(kp->pos.pt);
        auto mp = kp->map_point.lock();
        if (mp)
        {
            // use projected points as initial guess
            auto px =
                camera_right->world2pixel(mp->Pos(), current_frame->Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        }
        else
        {
            // use same pixel in left iamge
            kps_right.push_back(kp->pos.pt);
        }
    }

    std::vector<uchar> status;
    cv::Mat error;
    cv::calcOpticalFlowPyrLK(
        current_frame->left_image, current_frame->right_image, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i])
        {
            cv::KeyPoint kp(kps_right[i], 7);
            Feature::Ptr feat(new Feature(current_frame, kp));
            feat->is_on_left_image = false;
            current_frame->features_right.push_back(feat);
            num_good_pts++;
        }
        else
        {
            current_frame->features_right.push_back(nullptr);
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

bool Frontend::BuildInitMap()
{
    std::vector<SE3> poses{camera_left->Pose(), camera_right->Pose()};
    size_t cnt_init_landmarks = 0;
    for (size_t i = 0; i < current_frame->features_left.size(); ++i)
    {
        if (current_frame->features_right[i] == nullptr)
            continue;
        // create map point from triangulation
        std::vector<Vector3d> points{
            camera_left->pixel2camera(
                Vector2d(current_frame->features_left[i]->pos.pt.x,
                     current_frame->features_left[i]->pos.pt.y)),
            camera_right->pixel2camera(
                Vector2d(current_frame->features_right[i]->pos.pt.x,
                     current_frame->features_right[i]->pos.pt.y))};
        Vector3d pworld = Vector3d::Zero();

        if (triangulation(poses, points, pworld))
        {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(current_frame->features_left[i]);
            new_map_point->AddObservation(current_frame->features_right[i]);
            current_frame->features_left[i]->map_point = new_map_point;
            current_frame->features_right[i]->map_point = new_map_point;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    current_frame->SetKeyFrame();
    map_->InsertKeyFrame(current_frame);
    backend_->UpdateMap();

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Reset()
{
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}

} // namespace lvio_fusion