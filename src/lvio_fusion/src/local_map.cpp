#include "lvio_fusion/visual/local_map.h"
#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/ceres/visual_error.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/camera.h"

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

inline Vector3d LocalMap::ToWorld(Feature::Ptr feature)
{
    Vector3d pb = Camera::Get(1)->Pixel2Robot(
        cv2eigen(feature->landmark->first_observation->keypoint),
        feature->landmark->depth);
    return Camera::Get()->Robot2World(pb, feature->frame->pose);
}

int LocalMap::Init(Frame::Ptr new_kf)
{
    // get feature pyramid
    local_features_[new_kf->time] = FeaturePyramid();
    GetFeaturePyramid(new_kf, local_features_[new_kf->time]);
    GetNewLandmarks(new_kf, local_features_[new_kf->time]);
    return GetLandmarks(new_kf).size();
}

void LocalMap::Reset()
{
    local_features_.clear();
}

void LocalMap::AddKeyFrame(Frame::Ptr new_kf)
{
    CheckNewLandmarks(new_kf);
    // get feature pyramid
    local_features_[new_kf->time] = FeaturePyramid();
    GetFeaturePyramid(new_kf, local_features_[new_kf->time]);
    // search
    std::vector<double> kfs = GetCovisibilityKeyFrames(new_kf);
    GetNewLandmarks(new_kf, local_features_[new_kf->time]);
    Search(kfs, new_kf);
    // local BA
    LocalBA(new_kf);
    // remove old key frames
    if (local_features_.size() > windows_size_)
    {
        local_features_.erase(local_features_.begin());
        oldest = local_features_.begin()->first;
    }
}

void LocalMap::UpdateCache()
{
    for (auto &pair : local_features_)
    {
        pose_cache[pair.first] = Map::Instance().GetKeyFrame(pair.first)->pose;
        for (auto &features : pair.second)
        {
            for (auto &feature : features)
            {
                if (!feature->match && feature->landmark)
                {
                    position_cache[feature->landmark->id] = ToWorld(feature);
                    map_[feature->landmark->id] = feature;
                }
            }
        }
    }
}

void LocalMap::CheckNewLandmarks(Frame::Ptr frame)
{
    for (auto &pair : frame->features_left)
    {
        auto landmark = pair.second->landmark.lock();
        auto iter = map_.find(landmark->id);
        if (iter != map_.end() && !iter->second->insert)
        {
            iter->second->insert = true;
            iter->second->frame->AddFeature(landmark->first_observation);
            iter->second->frame->AddFeature(landmark->observations.begin()->second);
            Map::Instance().InsertLandmark(landmark);
        }
    }
}

void LocalMap::GetFeaturePyramid(Frame::Ptr frame, FeaturePyramid &pyramid)
{
    cv::Mat mask = cv::Mat(frame->image_left.size(), CV_8UC1, 255);
    for (auto &pair_feature : frame->features_left)
    {
        cv::circle(mask, pair_feature.second->keypoint, 20, 0, cv::FILLED);
    }
    std::vector<cv::KeyPoint> kps;
    cv::Mat descriptors;
    detector_->detectAndCompute(frame->image_left, mask, kps, descriptors);
    std::vector<std::vector<int>> index_levels;
    index_levels.resize(num_levels_);
    for (int i = 0; i < kps.size(); i++)
    {
        index_levels[kps[i].octave].push_back(i);
    }
    pyramid.clear();
    for (int i = 0; i < num_levels_; i++)
    {
        Features level_features;
        for (int index : index_levels[i])
        {
            level_features.push_back(Feature::Ptr(new Feature(frame, kps[index], mat2brief(descriptors.row(index)))));
        }
        pyramid.push_back(level_features);
    }
}

void LocalMap::GetNewLandmarks(Frame::Ptr frame, FeaturePyramid &pyramid)
{
    Features featrues;
    for (int i = 0; i < num_levels_; i++)
    {
        for (auto &feature : pyramid[i])
        {
            if (!feature->landmark)
            {
                featrues.push_back(feature);
            }
        }
    }
    Triangulate(frame, featrues);
}

void LocalMap::Triangulate(Frame::Ptr frame, Features &featrues)
{
    std::vector<cv::Point2f> kps_left, kps_right;
    kps_left.resize(featrues.size());
    for (int i = 0; i < featrues.size(); i++)
    {
        kps_left[i] = featrues[i]->kp.pt;
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
            if ((Camera::Get()->Robot2Pixel(pb) - kp_left).norm() < 0.5 && (Camera::Get(1)->Robot2Pixel(pb) - kp_right).norm() < 0.5)
            {
                auto new_landmark = visual::Landmark::Create(Camera::Get(1)->Robot2Sensor(pb).z());
                auto new_left_feature = visual::Feature::Create(frame, kps_left[i], new_landmark);
                auto new_right_feature = visual::Feature::Create(frame, kps_right[i], new_landmark);
                new_right_feature->is_on_left_image = false;

                new_landmark->AddObservation(new_left_feature);
                new_landmark->AddObservation(new_right_feature);
                featrues[i]->landmark = new_landmark;
                position_cache[new_landmark->id] = ToWorld(featrues[i]);
                map_[new_landmark->id] = featrues[i];
            }
        }
    }
}

std::vector<double> LocalMap::GetCovisibilityKeyFrames(Frame::Ptr frame)
{
    std::vector<double> kfs;
    for (auto &pair : local_features_)
    {
        kfs.push_back(pair.first);
    }
    kfs.pop_back();
    return kfs;
}

void LocalMap::Search(std::vector<double> kfs, Frame::Ptr frame)
{
    for (int i = 0; i < kfs.size(); i++)
    {
        Search(local_features_[kfs[i]], pose_cache[kfs[i]], local_features_[frame->time], frame);
    }
}

void LocalMap::Search(FeaturePyramid &last_pyramid, SE3d last_pose, FeaturePyramid &current_pyramid, Frame::Ptr frame)
{
    for (auto &features : current_pyramid)
    {
        for (auto &feature : features)
        {
            if (feature->landmark && !feature->match)
            {
                Search(last_pyramid, last_pose, feature, frame);
            }
        }
    }
}

void LocalMap::Search(FeaturePyramid &last_pyramid, SE3d last_pose, Feature::Ptr feature, Frame::Ptr frame)
{
    if (!feature->landmark)
        return;

    int min_level = feature->kp.octave, max_level = feature->kp.octave + 1;
    cv::Point2f p_in_last_left = eigen2cv(Camera::Get()->World2Pixel(position_cache[feature->landmark->id], last_pose));
    cv::Point2f p_in_last_right = eigen2cv(Camera::Get(1)->World2Pixel(position_cache[feature->landmark->id], last_pose));
    Features features_in_radius;
    std::vector<BRIEF> briefs;
    double radius = 200 * scale_factors_[feature->kp.octave];
    for (int i = min_level; i < max_level; i++)
    {
        for (auto &last_feature : last_pyramid[i])
        {
            if (last_feature->landmark &&
                distance(p_in_last_left, last_feature->kp.pt) < radius)
            {
                features_in_radius.push_back(last_feature);
                briefs.push_back(last_feature->brief);
            }
        }
    }

    cv::Mat descriptors_last = briefs2mat(briefs), descriptors_current = brief2mat(feature->brief);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(descriptors_current, descriptors_last, knn_matches, 2);
    const float ratio_thresh = 0.8;
    if (!features_in_radius.empty() && !knn_matches.empty() &&
        knn_matches[0][0].distance < ratio_thresh * knn_matches[0][1].distance)
    {
        auto last_feature = features_in_radius[knn_matches[0][0].trainIdx];
        feature->landmark = last_feature->landmark;
        feature->match = true;
        if (!last_feature->insert && !last_feature->match)
        {
            // insert!
            last_feature->insert = true;
            last_feature->frame->AddFeature(last_feature->landmark->first_observation);
            last_feature->frame->AddFeature(last_feature->landmark->observations.begin()->second);
            Map::Instance().InsertLandmark(last_feature->landmark);
        }

        auto new_left_feature = visual::Feature::Create(frame, feature->kp.pt, feature->landmark);
        assert(feature->landmark->FirstFrame().lock());
        feature->landmark->AddObservation(new_left_feature);
        frame->AddFeature(new_left_feature);
    }
}

void LocalMap::LocalBA(Frame::Ptr frame)
{
    SE3d old_pose = frame->pose;
    adapt::Problem problem;
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
        if (landmark->FirstFrame().lock() != frame)
        {
            ceres::CostFunction *cost_function = PoseOnlyReprojectionError::Create(cv2eigen(feature->keypoint), position_cache[landmark->id], Camera::Get(), frame->weights.visual);
            problem.AddResidualBlock(ProblemType::VisualError, cost_function, loss_function, para_kf);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 1;
    options.num_threads = num_threads;
    ceres::Solver::Summary summary;
    adapt::Solve(options, &problem, &summary);

    SE3d new_pose = frame->pose;
    SE3d transform = new_pose * old_pose.inverse();
    for (auto &features : local_features_[frame->time])
    {
        for (auto &feature : features)
        {
            if (feature->landmark && !feature->insert)
            {
                position_cache[feature->landmark->id] = transform * position_cache[feature->landmark->id];
            }
        }
    }
}

LocalMap::Features LocalMap::GetLandmarks(Frame::Ptr frame)
{
    Features result;
    for (auto &features : local_features_[frame->time])
    {
        for (auto &feature : features)
        {
            if (feature->landmark)
            {
                result.push_back(feature);
            }
        }
    }
    return result;
}

} // namespace lvio_fusion
