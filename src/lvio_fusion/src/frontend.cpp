#include "lvio_fusion/frontend.h"
#include "lvio_fusion/backend.h"
#include "lvio_fusion/ceres_helper/pose_only_reprojection_error.hpp"
#include "lvio_fusion/ceres_helper/se3_parameterization.hpp"
#include "lvio_fusion/config.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"

#include <opencv2/core/eigen.hpp>

namespace lvio_fusion
{

Frontend::Frontend()
{
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
    return true;
}

bool Frontend::Track()
{
    current_frame->SetPose(relative_motion * last_frame->Pose());
    int num_track_last = TrackLastFrame();
    //InitFramePoseByPnP();
    tracking_inliers_ = Optimization();

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

bool Frontend::InitFramePoseByPnP()
{
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    for (auto feature : current_frame->features_left)
    {
        auto map_point = feature->map_point.lock();
        if (map_point)
        {
            points_2d.push_back(feature->pos.pt);
            Vector3d p = map_point->Pos();
            points_3d.push_back(cv::Point3f(p.x(), p.y(), p.z()));
        }
    }

    cv::Mat K;
    cv::eigen2cv(camera_left->K(), K);
    cv::Mat rvec, tvec, inliers, D, cv_R;
    if (cv::solvePnP(points_3d, points_2d, K, D, rvec, tvec, false, cv::SOLVEPNP_EPNP))
    {
        cv::Rodrigues(rvec, cv_R);
        Matrix3d R;
        cv::cv2eigen(cv_R, R);
        current_frame->SetPose(camera_left->Pose().inverse() *
                               SE3(SO3(R), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))));
        return true;
    }
    return false;
}

int Frontend::Optimization()
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
            cost_function = new PoseOnlyReprojectionError(to_vector2d(feature->pos.pt), camera_left, map_point->Pos());
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
            Vector2d error = to_vector2d(feature->pos.pt) - camera_left->world2pixel(map_point->Pos(), current_frame->Pose());
            if (error[0] * error[0] + error[1] * error[1] > 9)
            {
                feature->map_point.reset();
                //NOTE: maybe we can still use it in future
                feature->is_outlier = false;
                feature_count--;
            }
        }
    }

    // LOG(INFO) << "Current Pose = \n"
    //           << current_frame->Pose().matrix();

    return feature_count;
}

int Frontend::TrackLastFrame()
{
    // use LK flow to estimate points in the last image
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

    //NOTE: Ransac
    //cv::findFundamentalMat(kps_current, kps_last, cv::FM_RANSAC, 3.0, 0.9, status);

    int num_good_pts = 0;
    //DEBUG
    cv::Mat img_track = current_frame->left_image;
    cv::cvtColor(img_track, img_track, CV_GRAY2RGB);
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i])
        {
            cv::arrowedLine(img_track, kps_current[i], kps_last[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame, kp));
            feature->map_point = last_frame->features_left[i]->map_point;
            current_frame->features_left.push_back(feature);
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
    if (current_frame->features_left.size() >= num_features_)
    {
        return -1;
    }
    cv::Mat mask(current_frame->left_image.size(), CV_8UC1, 255);
    for (auto &feat : current_frame->features_left)
    {
        cv::rectangle(mask, feat->pos.pt - cv::Point2f(10, 10),
                      feat->pos.pt + cv::Point2f(10, 10), 0, CV_FILLED);
    }

    std::vector<cv::Point2f> keypoints;
    // gftt_->detect(current_frame->left_image, keypoints, mask);
    cv::goodFeaturesToTrack(current_frame->left_image, keypoints, num_features_ - current_frame->features_left.size(), 0.01, 30, mask);
    int cnt_detected = 0;
    for (auto &kp : keypoints)
    {
        current_frame->features_left.push_back(
            Feature::Ptr(new Feature(current_frame, cv::KeyPoint(kp, 1))));
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
    map_->Reset();
    status = FrontendStatus::INITING;
    LOG(ERROR) << "Reset Succeed";
    return true;
}

} // namespace lvio_fusion