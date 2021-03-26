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
    return Camera::Get()->Robot2World(pb, pose_cache[feature->frame->time]);
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
    // get feature pyramid
    local_features_[new_kf->time] = FeaturePyramid();
    GetFeaturePyramid(new_kf, local_features_[new_kf->time]);
    // search
    pose_cache[new_kf->time] = new_kf->pose;
    std::vector<double> kfs = GetCovisibilityKeyFrames(new_kf);
    GetNewLandmarks(new_kf, local_features_[new_kf->time]);
    Search(kfs, new_kf);
    InsertNewLandmarks(new_kf);
    // remove old key frames
    if (local_features_.size() > windows_size_)
    {
        local_features_.erase(local_features_.begin());
        oldest = local_features_.begin()->first;
    }
}

void LocalMap::UpdateCache()
{
    std::unique_lock<std::mutex> lock(mutex_);
    map_.clear();
    pose_cache.clear();
    position_cache.clear();
    for (auto &pair : local_features_)
    {
        pose_cache[pair.first] = Map::Instance().GetKeyFrame(pair.first)->pose;
        for (auto &features : pair.second)
        {
            for (auto &feature : features)
            {
                if (feature->landmark && position_cache.find(feature->landmark->id) == position_cache.end())
                {
                    position_cache[feature->landmark->id] = ToWorld(feature);
                    if (!feature->match)
                    {
                        map_[feature->landmark->id] = feature;
                    }
                }
            }
        }
    }
}

void LocalMap::InsertNewLandmarks(Frame::Ptr frame)
{
    for (auto &pair : frame->features_left)
    {
        auto landmark = pair.second->landmark.lock();
        if (Map::Instance().landmarks.find(landmark->id) == Map::Instance().landmarks.end())
        {
            auto iter = map_.find(landmark->id);
            if (iter != map_.end() && !iter->second->match && !iter->second->insert)
            {
                iter->second->insert = true;
            }
            auto first_frame = landmark->FirstFrame().lock();
            first_frame->AddFeature(landmark->first_observation);
            first_frame->AddFeature(landmark->observations.begin()->second);
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
        Vector3d last_heading = pose_cache[pair.first].so3() * Vector3d::UnitX();
        Vector3d heading = frame->pose.so3() * Vector3d::UnitX();
        double degree = vectors_degree_angle(last_heading, heading);
        if (degree < 30)
        {
            kfs.push_back(pair.first);
        }
    }
    kfs.pop_back();
    return kfs;
}

cv::Mat img_track;
void LocalMap::Search(std::vector<double> kfs, Frame::Ptr frame)
{
    img_track = frame->image_left;
    cv::cvtColor(img_track, img_track, cv::COLOR_GRAY2RGB);
    for (int i = 0; i < kfs.size(); i++)
    {
        Search(local_features_[kfs[i]], pose_cache[kfs[i]], local_features_[frame->time], frame);
    }
    cv::imshow("tracking2", img_track);
    cv::waitKey(1);
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
    auto pc = Camera::Get()->World2Sensor(position_cache[feature->landmark->id], last_pose);
    if (pc.z() < 0)
        return;
    cv::Point2f p_in_last_left = eigen2cv(Camera::Get()->Sensor2Pixel(pc));
    Features features_in_radius;
    std::vector<BRIEF> briefs;
    int min_level = feature->kp.octave, max_level = feature->kp.octave + 1;
    for (int i = min_level; i <= max_level && i < num_levels_; i++)
    {
        double radius = 20 * scale_factors_[i];
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
    const float ratio_thresh = 0.6;
    if (!features_in_radius.empty() && !knn_matches.empty() && knn_matches[0].size() == 2 &&
        knn_matches[0][0].distance < ratio_thresh * knn_matches[0][1].distance)
    {
        auto last_feature = features_in_radius[knn_matches[0][0].trainIdx];
        feature->landmark = last_feature->landmark;
        feature->match = true;

        cv::arrowedLine(img_track, last_feature->kp.pt, feature->kp.pt, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        cv::circle(img_track, last_feature->kp.pt, 2, cv::Scalar(0, 255, 0), cv::FILLED);
        auto new_left_feature = visual::Feature::Create(frame, feature->kp.pt, feature->landmark);
        feature->landmark->AddObservation(new_left_feature);
        frame->AddFeature(new_left_feature);
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

PointRGBCloud LocalMap::GetLocalLandmarks()
{
    std::unique_lock<std::mutex> lock(mutex_);
    PointRGBCloud out;
    for (auto &pyramid : local_features_)
    {
        for (auto &features : pyramid.second)
        {
            for (auto &feature : features)
            {
                if (feature->insert)
                {
                    PointRGB point_color;
                    point_color.x = position_cache[feature->landmark->id].x();
                    point_color.y = position_cache[feature->landmark->id].y();
                    point_color.z = position_cache[feature->landmark->id].z();
                    point_color.r = 255;
                    point_color.g = 255;
                    point_color.b = 255;
                    out.push_back(point_color);
                }
            }
        }
    }
    return out;
}

} // namespace lvio_fusion
