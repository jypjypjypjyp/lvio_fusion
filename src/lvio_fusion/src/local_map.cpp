#include "lvio_fusion/visual/local_map.h"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/camera.h"

#include <fstream>

namespace lvio_fusion
{

inline cv::Mat brief2mat(BRIEF &brief)
{
    return cv::Mat(1, 32, CV_8U, reinterpret_cast<uchar *>(&brief));
}

inline BRIEF mat2brief(const cv::Mat &mat)
{
    BRIEF brief;
    memcpy(&brief, mat.data, 32);
    return brief;
}

inline std::vector<BRIEF> mat2briefs(cv::Mat descriptors)
{
    std::vector<BRIEF> briefs;
    for (int i = 0; i < descriptors.rows; i++)
    {
        briefs.push_back(mat2brief(descriptors.row(i)));
    }
    return briefs;
}

inline cv::Mat briefs2mat(std::vector<BRIEF> briefs)
{
    cv::Mat descriptors(briefs.size(), 32, CV_8U);
    for (int i = 0; i < briefs.size(); i++)
    {
        brief2mat(briefs[i]).copyTo(descriptors.row(i));
    }
    return descriptors;
}

inline Vector3d LocalMap::ToWorld(visual::Feature::Ptr feature)
{
    Vector3d pb = Camera::Get(1)->Pixel2Robot(
        cv2eigen(feature->landmark.lock()->first_observation->keypoint.pt),
        1 / feature->landmark.lock()->inv_depth);
    return Camera::Get()->Robot2World(pb, pose_cache[feature->frame.lock()->time]);
}

int LocalMap::Init(Frame::Ptr new_kf)
{
    // reset
    Reset();
    // get feature pyramid
    {
        std::unique_lock<std::mutex> lock(mutex_);
        local_features_[new_kf->time] = Pyramid();
        GetFeaturePyramid(new_kf, local_features_[new_kf->time]);
        GetNewLandmarks(new_kf, local_features_[new_kf->time]);
    }
    return GetFeatures(new_kf->time).size();
}

void LocalMap::Reset()
{
    std::unique_lock<std::mutex> lock(mutex_);
    local_features_.clear();
    landmarks.clear();
    position_cache.clear();
    pose_cache.clear();
}

void LocalMap::AddKeyFrame(Frame::Ptr new_kf)
{
    std::unique_lock<std::mutex> lock(mutex_);
    // insert new landmarks
    for (auto &pair_feature : new_kf->features_left)
    {
        auto landmark = pair_feature.second->landmark.lock();
        if (!landmark->first_observation->insert)
        {
            auto last_frame = landmark->FirstFrame().lock();
            landmark->observations.begin()->second->insert = true;
            landmark->first_observation->insert = true;
            last_frame->AddFeature(landmark->observations.begin()->second);
            last_frame->AddFeature(landmark->first_observation);
            Map::Instance().InsertLandmark(landmark);
        }
    }
    // local BA
    LocalBA(new_kf);
    // get feature pyramid
    local_features_[new_kf->time] = Pyramid();
    GetFeaturePyramid(new_kf, local_features_[new_kf->time]);
    // search
    pose_cache[new_kf->time] = new_kf->pose;
    std::vector<double> kfs = GetCovisibilityKeyFrames(new_kf);
    GetNewLandmarks(new_kf, local_features_[new_kf->time]);
    Search(kfs, new_kf);
    // remove old key frame and old landmarks
    if (local_features_.size() > windows_size_)
    {
        for (auto &level : local_features_.begin()->second)
        {
            for (auto &feature : level)
            {
                if (!feature->landmark.expired())
                {
                    landmarks.erase(feature->landmark.lock()->id);
                }
            }
        }
        local_features_.erase(local_features_.begin());
    }
}

void LocalMap::UpdateCache()
{
    std::unique_lock<std::mutex> lock(mutex_);
    pose_cache.clear();
    position_cache.clear();

    for (auto &pair : local_features_)
    {
        pose_cache[pair.first] = Map::Instance().GetKeyFrame(pair.first)->pose;
    }

    for (auto &pair : landmarks)
    {
        position_cache[pair.first] = pair.second->ToWorld();
    }
}

void LocalMap::GetFeaturePyramid(Frame::Ptr frame, Pyramid &pyramid)
{
    cv::Mat mask = cv::Mat(frame->image_left.size(), CV_8UC1, 255);
    for (auto &pair_feature : frame->features_left)
    {
        cv::circle(mask, pair_feature.second->keypoint.pt, extractor_.patch_size, 0, cv::FILLED);
    }

    std::vector<std::vector<cv::KeyPoint>> kps;
    extractor_.Detect(frame->image_left, mask, kps);

    pyramid.clear();
    pyramid.resize(num_levels_);
    for (int i = 0; i < num_levels_; i++)
    {
        pyramid[i].reserve(kps[i].size());
        for (auto &kp : kps[i])
        {
            pyramid[i].push_back(visual::Feature::Create(frame, kp));
        }
    }
}

void LocalMap::LocalBA(Frame::Ptr frame)
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));
    double *para_kf = frame->pose.data();
    problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
    for (auto &pair_feature : frame->features_left)
    {
        auto feature = pair_feature.second;
        auto landmark = feature->landmark.lock();
        auto first_frame = landmark->FirstFrame().lock();
        ceres::CostFunction *cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint.pt), landmark->ToWorld(), Camera::Get(), frame->weights.visual);
        problem.AddResidualBlock(cost_function, loss_function, para_kf);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 1;
    options.num_threads = num_threads;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

void LocalMap::GetNewLandmarks(Frame::Ptr frame, Pyramid &pyramid)
{
    // put all features into one level
    Level features;
    for (auto &level : pyramid)
    {
        for (auto &feature : level)
        {
            features.push_back(feature);
        }
    }
    // need to check if is a moving point
    for (auto &pair_feature : frame->features_left)
    {
        auto feature = pair_feature.second;
        auto landmark = feature->landmark.lock();
        if (!Camera::Get()->Far(position_cache[landmark->id], frame->pose))
        {
            features.push_back(feature);
        }
    }

    Triangulate(frame, features);

    // remove all failed features
    for (auto &level : pyramid)
    {
        for (auto iter = level.begin(); iter != level.end();)
        {
            if ((*iter)->landmark.expired())
            {
                iter = level.erase(iter);
            }
            else
            {
                ++iter;
            }
        }
    }

    // compute descriptors
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    keypoints.resize(num_levels_);
    for (int i = 0; i < num_levels_; i++)
    {
        keypoints[i].reserve(pyramid[i].size());
        for (auto &feature : pyramid[i])
        {
            keypoints[i].push_back(feature->keypoint);
        }
    }
    cv::Mat descriptors = extractor_.Compute(keypoints);

    for (int i = 0, j = 0; i < num_levels_; i++)
    {
        for (auto &feature : pyramid[i])
        {
            feature->brief = mat2brief(descriptors.row(j++));
        }
    }
}

void LocalMap::Triangulate(Frame::Ptr frame, Level &features)
{
    std::vector<cv::Point2f> kps_left, kps_right;
    kps_left.resize(features.size());
    for (int i = 0; i < features.size(); i++)
    {
        kps_left[i] = features[i]->keypoint.pt;
    }
    kps_right = kps_left;
    std::vector<uchar> status;
    optical_flow(frame->image_left, frame->image_right, kps_left, kps_right, status);
    // triangulate new points
    for (int i = 0; i < kps_left.size(); ++i)
    {
        if (status[i])
        {
            // triangulation
            Vector2d kp_left = cv2eigen(kps_left[i]);
            Vector2d kp_right = cv2eigen(kps_right[i]);
            Vector3d pb = Vector3d::Zero();
            triangulate(Camera::Get()->extrinsic.inverse(), Camera::Get(1)->extrinsic.inverse(), Camera::Get()->Pixel2Sensor(kp_left), Camera::Get(1)->Pixel2Sensor(kp_right), pb);
            if (pb.z() > 0 && (Camera::Get()->Robot2Pixel(pb) - kp_left).norm() < 0.5 && (Camera::Get(1)->Robot2Pixel(pb) - kp_right).norm() < 0.5)
            {
                if (features[i]->landmark.expired())
                {
                    auto new_landmark = visual::Landmark::Create(1 / Camera::Get(1)->Robot2Sensor(pb).z());
                    features[i]->landmark = new_landmark; // new left feature
                    auto new_right_feature = visual::Feature::Create(frame, cv::KeyPoint(kps_right[i], 1), new_landmark);
                    new_right_feature->is_on_left_image = false;
                    new_landmark->AddObservation(features[i]);
                    new_landmark->AddObservation(new_right_feature);
                    position_cache[new_landmark->id] = ToWorld(features[i]);
                    landmarks[new_landmark->id] = new_landmark;
                    cv::circle(img_track, kps_left[i], 2, cv::Scalar(255, 0, 0), cv::FILLED);
                }
                else
                {
                    // check if is a moving point
                    Vector3d pw = Camera::Get(1)->Robot2World(pb, frame->pose);
                    double dt = frame->time - features[i]->landmark.lock()->FirstFrame().lock()->time;
                    double e = (pw - position_cache[features[i]->landmark.lock()->id]).norm();
                    if (e / dt > 4 || e > 4)
                    {
                        frame->RemoveFeature(features[i]);
                        cv::putText(img_track, "X", kps_left[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
                    }
                }
            }
        }
    }
}

std::vector<double> LocalMap::GetCovisibilityKeyFrames(Frame::Ptr frame)
{
    std::vector<double> kfs;
    for (auto &pair : local_features_)
    {
        if (pair.first == frame->time)
            continue;
        Vector3d last_heading = pose_cache[pair.first].so3() * Vector3d::UnitX();
        Vector3d heading = frame->pose.so3() * Vector3d::UnitX();
        double degree = vectors_degree_angle(last_heading, heading);
        if (degree < 30)
        {
            kfs.push_back(pair.first);
        }
    }
    std::sort(kfs.begin(), kfs.end(), std::greater<double>());
    return kfs;
}

void LocalMap::Search(std::vector<double> kfs, Frame::Ptr frame)
{
    for (int i = 0; i < kfs.size(); i++)
    {
        Search(local_features_[kfs[i]], pose_cache[kfs[i]], local_features_[frame->time], frame);
    }
}

void LocalMap::Search(Pyramid &last_pyramid, SE3d last_pose, Pyramid &current_pyramid, Frame::Ptr frame)
{
    for (auto &features : current_pyramid)
    {
        for (auto &feature : features)
        {
            if (!feature->match)
            {
                Search(last_pyramid, last_pose, feature, frame);
            }
        }
    }
}

void LocalMap::Search(Pyramid &last_pyramid, SE3d last_pose, visual::Feature::Ptr feature, Frame::Ptr frame)
{
    auto pc = Camera::Get()->World2Sensor(position_cache[feature->landmark.lock()->id], last_pose);
    if (pc.z() < 0)
        return;
    cv::Point2f p_in_last_left = eigen2cv(Camera::Get()->Sensor2Pixel(pc));
    Level features_in_radius;
    std::vector<BRIEF> briefs;
    //TODO: check if forward or backward
    int min_level = feature->keypoint.octave, max_level = feature->keypoint.octave + 1;
    for (int i = min_level; i <= max_level && i < num_levels_; i++)
    {
        double radius = extractor_.patch_size * scale_factors_[i];
        for (auto &last_feature : last_pyramid[i])
        {
            double rotate = std::abs(last_feature->keypoint.angle - feature->keypoint.angle);
            if (rotate < 15)
            {
                if (cv_distance(p_in_last_left, last_feature->keypoint.pt) < radius)
                {
                    features_in_radius.push_back(last_feature);
                    briefs.push_back(last_feature->brief);
                }
            }
        }
    }

    cv::Mat descriptors_last = briefs2mat(briefs), descriptors_current = brief2mat(feature->brief);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(descriptors_current, descriptors_last, knn_matches, 2);
    const float ratio_threshold = 0.8;
    const float low_threshold = 50;
    if (!features_in_radius.empty() && !knn_matches.empty() && knn_matches[0].size() == 2 &&
        knn_matches[0][0].distance < low_threshold &&
        knn_matches[0][0].distance < ratio_threshold * knn_matches[0][1].distance)
    {
        auto last_feature = features_in_radius[knn_matches[0][0].trainIdx];

        // add feature
        feature->match = true;
        feature->landmark = last_feature->landmark;
        feature->frame.lock()->AddFeature(feature);
        last_feature->landmark.lock()->AddObservation(feature);
        // add last feature
        if (!last_feature->match && !last_feature->insert)
        {
            auto last_frame = last_feature->frame.lock();
            auto last_landmark = last_feature->landmark.lock();
            last_feature->insert = true;
            last_landmark->first_observation->insert = true;
            last_frame->AddFeature(last_feature);
            last_frame->AddFeature(last_landmark->first_observation);
            Map::Instance().InsertLandmark(last_landmark);
        }
    }
}

Level LocalMap::GetFeatures(double time)
{
    std::unique_lock<std::mutex> lock(mutex_);
    Level result;
    for (auto &features : local_features_[time])
    {
        for (auto &feature : features)
        {
            result.push_back(feature);
        }
    }
    return result;
}

PointRGBCloud LocalMap::GetLocalLandmarks()
{
    std::unique_lock<std::mutex> lock(mutex_);
    PointRGBCloud out;
    for (auto &pair : landmarks)
    {
        PointRGB point_color;
        Vector3d pw = position_cache[pair.second->id];
        point_color.x = pw.x();
        point_color.y = pw.y();
        point_color.z = pw.z();
        point_color.r = 255;
        point_color.g = 255;
        point_color.b = 255;
        out.push_back(point_color);
    }
    return out;
}

} // namespace lvio_fusion
