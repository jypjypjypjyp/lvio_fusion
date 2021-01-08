#include "lvio_fusion/lidar/association.h"
#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/loop_error.hpp"
#include "lvio_fusion/lidar/feature.h"
#include "lvio_fusion/lidar/lidar.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace lvio_fusion
{

void FeatureAssociation::AddScan(double time, Point3Cloud::Ptr new_scan)
{
    static double finished = 0;
    raw_point_clouds_[time] = new_scan;

    Frames new_kfs = Map::Instance().GetKeyFrames(finished, time);
    for (auto &pair_kf : new_kfs)
    {
        PointICloud point_cloud;
        if (AlignScan(pair_kf.first, point_cloud))
        {
            Process(point_cloud, pair_kf.second);
            finished = pair_kf.first + epsilon;
        }
    }
}

bool FeatureAssociation::AlignScan(double time, PointICloud &out)
{
    auto iter = raw_point_clouds_.upper_bound(time);
    if (iter == raw_point_clouds_.begin())
        return false;
    Point3Cloud &pc2 = *(iter->second);
    double end_time = iter->first + cycle_time_ / 2;
    Point3Cloud &pc1 = *((--iter)->second);
    double start_time = iter->first - cycle_time_ / 2;
    Point3Cloud pc = pc1 + pc2;
    int size = pc.size();
    if (time - cycle_time_ / 2 < start_time || time + cycle_time_ / 2 > end_time)
    {
        return false;
    }
    auto start_iter = pc.begin() + size * (time - start_time - cycle_time_ / 2) / (end_time - start_time);
    auto end_iter = pc.begin() + size * (time - start_time + cycle_time_ / 2) / (end_time - start_time);
    Point3Cloud out3;
    out3.clear();
    out3.insert(out3.begin(), start_iter, end_iter);
    pcl::copyPointCloud(out3, out);
    raw_point_clouds_.erase(raw_point_clouds_.begin(), iter);
    return true;
}

void FeatureAssociation::UndistortPoint(PointI &point, Frame::Ptr frame)
{
    double time_delta = (point.intensity - int(point.intensity));
    double time = frame->time - cycle_time_ * 0.5 + time_delta;
    SE3d pose = Map::Instance().ComputePose(time);
    auto p1 = Lidar::Get()->Sensor2World(Vector3d(point.x, point.y, point.z), pose);
    auto p2 = Lidar::Get()->World2Sensor(p1, frame->pose);
    point.x = p2.x();
    point.y = p2.y();
    point.z = p2.z();
}

inline void FeatureAssociation::UndistortPointCloud(PointICloud &points, Frame::Ptr frame)
{
    for (auto &p : points)
    {
        UndistortPoint(p, frame);
    }
}

void FeatureAssociation::Process(PointICloud &points, Frame::Ptr frame)
{
    Preprocess(points);

    PointICloud points_segmented;
    auto segmented_info = projection_->Process(points, points_segmented);

    Extract(points_segmented, segmented_info, frame);
}

void FeatureAssociation::Preprocess(PointICloud &points)
{
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(points, points, indices);
    filter_points_by_distance(points, points, min_range_, max_range_);
}

void FeatureAssociation::Extract(PointICloud &points_segmented, SegmentedInfo &segemented_info, Frame::Ptr frame)
{
    AdjustDistortion(points_segmented, segemented_info);

    CalculateSmoothness(points_segmented, segemented_info);

    ExtractFeatures(points_segmented, segemented_info, frame);
}

void FeatureAssociation::AdjustDistortion(PointICloud &points_segmented, SegmentedInfo &segemented_info)
{
    bool half_passed = false;
    int size = points_segmented.size();
    PointI point;
    for (int i = 0; i < size; i++)
    {
        point.x = points_segmented[i].x;
        point.y = points_segmented[i].y;
        point.z = points_segmented[i].z;

        float ori = -atan2(point.y, point.x);
        if (!half_passed)
        {
            if (ori < segemented_info.start_orientation - M_PI / 2)
                ori += 2 * M_PI;
            else if (ori > segemented_info.start_orientation + M_PI * 3 / 2)
                ori -= 2 * M_PI;

            if (ori - segemented_info.start_orientation > M_PI)
                half_passed = true;
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < segemented_info.end_orientation - M_PI * 3 / 2)
                ori += 2 * M_PI;
            else if (ori > segemented_info.end_orientation + M_PI / 2)
                ori -= 2 * M_PI;
        }

        float rel_time = (ori - segemented_info.start_orientation) / segemented_info.orientation_diff;
        point.intensity = int(points_segmented[i].intensity) + cycle_time_ * rel_time;
        //TODO:deskew
        points_segmented[i] = point;
    }
}

void FeatureAssociation::CalculateSmoothness(PointICloud &points_segmented, SegmentedInfo &segemented_info)
{
    int size = points_segmented.size();
    for (int i = 5; i < size - 5; i++)
    {
        float dr = (segemented_info.range[i + 5] - segemented_info.range[i - 5]) / 10;
        float r1 = segemented_info.range[i + 4] - segemented_info.range[i - 5] - 9 * dr;
        float r2 = segemented_info.range[i + 3] - segemented_info.range[i - 5] - 8 * dr;
        float r3 = segemented_info.range[i + 2] - segemented_info.range[i - 5] - 7 * dr;
        float r4 = segemented_info.range[i + 1] - segemented_info.range[i - 5] - 6 * dr;
        float r5 = segemented_info.range[i] - segemented_info.range[i - 5] - 5 * dr;
        float r6 = segemented_info.range[i - 1] - segemented_info.range[i - 5] - 4 * dr;
        float r7 = segemented_info.range[i - 2] - segemented_info.range[i - 5] - 3 * dr;
        float r8 = segemented_info.range[i - 3] - segemented_info.range[i - 5] - 2 * dr;
        float r9 = segemented_info.range[i - 4] - segemented_info.range[i - 5] - 1 * dr;
        float cov = (r1 * r1 + r2 * r2 + r3 * r3 + r4 * r4 + r5 * r5 + r6 * r6 + r7 * r7 + r8 * r8 + r9 * r9) / 9;
        curvatures[i] = cov * 10 / segemented_info.range[i];
    }
}

void FeatureAssociation::ExtractFeatures(PointICloud &points_segmented, SegmentedInfo &segemented_info, Frame::Ptr frame)
{
    PointICloud points_ground, points_surf; //, points_full;
    static const float threshold = 1;
    for (int i = 0; i < num_scans_; i++)
    {
        // divide one scan into six segments
        for (int j = 0; j < 6; j++)
        {
            int sp = (segemented_info.start_ring_index[i] * (6 - j) + segemented_info.end_ring_index[i] * j) / 6;
            int ep = (segemented_info.start_ring_index[i] * (5 - j) + segemented_info.end_ring_index[i] * (j + 1)) / 6 - 1;
            if (sp >= ep)
                continue;

            for (int k = sp; k <= ep; k++)
            {
                if (segemented_info.ground_flag[k] == true)
                {
                    points_ground.push_back(points_segmented[k]);
                }
                else if (curvatures[k] < threshold)
                {
                    points_surf.push_back(points_segmented[k]);
                }
            }
        }
    }

    PointICloud::Ptr temp(new PointICloud());
    pcl::VoxelGrid<PointI> voxel_filter;
    voxel_filter.setLeafSize(2 * Lidar::Get()->resolution, 2 * Lidar::Get()->resolution, 2 * Lidar::Get()->resolution);
    pcl::copyPointCloud(points_surf, *temp);
    voxel_filter.setInputCloud(temp);
    voxel_filter.filter(points_surf);

    pcl::RadiusOutlierRemoval<PointI> ror_filter;
    ror_filter.setRadiusSearch(4 * Lidar::Get()->resolution);
    ror_filter.setMinNeighborsInRadius(4);
    pcl::copyPointCloud(points_surf, *temp);
    ror_filter.setInputCloud(temp);
    ror_filter.filter(points_surf);

    pcl::copyPointCloud(points_ground, *temp);
    voxel_filter.setInputCloud(temp);
    voxel_filter.filter(points_ground);

    SegmentGround(points_ground);

    lidar::Feature::Ptr feature = lidar::Feature::Create();
    Sensor2Robot(points_ground, feature->points_ground);
    Sensor2Robot(points_surf, feature->points_surf);
    frame->feature_lidar = feature;
}

inline void FeatureAssociation::Sensor2Robot(PointICloud &in, PointICloud &out)
{
    Sophus::SE3f tf_se3 = Lidar::Get()->extrinsic.cast<float>();
    float *tf = tf_se3.data();
    for (auto &point_in : in)
    {
        PointI point_out;
        ceres::SE3TransformPoint(tf, point_in.data, point_out.data);
        point_out.intensity = point_in.intensity;
        out.push_back(point_out);
    }
}

inline void FeatureAssociation::SegmentGround(PointICloud &points_ground)
{
    PointICloud::Ptr pointcloud_seg(new PointICloud());
    pcl::copyPointCloud(points_ground, *pointcloud_seg);
    pcl::ModelCoefficients coefficients;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<PointI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.1 * Lidar::Get()->resolution);
    seg.setInputCloud(pointcloud_seg);
    seg.segment(*inliers, coefficients);
    pcl::ExtractIndices<PointI> extract;
    extract.setInputCloud(pointcloud_seg);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(points_ground);
}

void FeatureAssociation::ScanToMapWithGround(Frame::Ptr frame, Frame::Ptr map_frame, double *para, adapt::Problem &problem)
{
    ceres::LossFunction *loss_function = new ceres::TrivialLoss();
    PointICloud &points_ground_last = map_frame->feature_lidar->points_ground;
    problem.AddParameterBlock(para + 1, 1);
    problem.AddParameterBlock(para + 2, 1);
    problem.AddParameterBlock(para + 5, 1);

    pcl::KdTreeFLANN<PointI> kdtree_last;
    kdtree_last.setInputCloud(boost::make_shared<PointICloud>(points_ground_last));

    PointI point;
    std::vector<int> points_index;
    std::vector<float> points_distance;

    static const double distance_threshold = Lidar::Get()->resolution * Lidar::Get()->resolution * 16; // squared
    int num_points_flat = frame->feature_lidar->points_ground.size();
    Sophus::SE3f tf_se3 = frame->pose.cast<float>();
    float *tf = tf_se3.data();

    // find correspondence for ground features
    for (int i = 0; i < num_points_flat; ++i)
    {
        //NOTE: Sophus is too slow
        ceres::SE3TransformPoint(tf, frame->feature_lidar->points_ground[i].data, point.data);
        point.intensity = frame->feature_lidar->points_ground[i].intensity;
        kdtree_last.nearestKSearch(point, 3, points_index, points_distance);
        // clang-format off
        if (points_index[0] < points_ground_last.size() && points_distance[0] < distance_threshold 
         && points_index[1] < points_ground_last.size() && points_distance[1] < distance_threshold 
         && points_index[2] < points_ground_last.size() && points_distance[2] < distance_threshold)
        // clang-format on
        {
            Vector3d curr_point(frame->feature_lidar->points_ground[i].x,
                                frame->feature_lidar->points_ground[i].y,
                                frame->feature_lidar->points_ground[i].z);
            Vector3d last_point_a(points_ground_last[points_index[0]].x,
                                  points_ground_last[points_index[0]].y,
                                  points_ground_last[points_index[0]].z);
            Vector3d last_point_b(points_ground_last[points_index[1]].x,
                                  points_ground_last[points_index[1]].y,
                                  points_ground_last[points_index[1]].z);
            Vector3d last_point_c(points_ground_last[points_index[2]].x,
                                  points_ground_last[points_index[2]].y,
                                  points_ground_last[points_index[2]].z);
            ceres::CostFunction *cost_function;
            cost_function = LidarPlaneErrorRPZ::Create(curr_point, last_point_a, last_point_b, last_point_c, map_frame->pose, para, frame->weights.lidar_ground);
            problem.AddResidualBlock(ProblemType::LidarPlaneErrorRPZ, cost_function, loss_function, para + 1, para + 2, para + 5);
        }
    }

    if (frame->id == map_frame->id + 1)
    {
        ceres::CostFunction *cost_function = PoseErrorRPZ::Create(para, frame->weights.visual);
        problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para + 1, para + 2, para + 5);
    }
}

void FeatureAssociation::ScanToMapWithSegmented(Frame::Ptr frame, Frame::Ptr map_frame, double *para, adapt::Problem &problem)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    PointICloud &points_surf_last = map_frame->feature_lidar->points_surf;
    problem.AddParameterBlock(para + 0, 1);
    problem.AddParameterBlock(para + 3, 1);
    problem.AddParameterBlock(para + 4, 1);

    pcl::KdTreeFLANN<PointI> kdtree_last;
    kdtree_last.setInputCloud(boost::make_shared<PointICloud>(points_surf_last));

    PointI point;
    std::vector<int> points_index;
    std::vector<float> points_distance;

    static const double distance_threshold = Lidar::Get()->resolution * Lidar::Get()->resolution * 16; // squared
    int num_points_flat = frame->feature_lidar->points_surf.size();
    Sophus::SE3f tf_se3 = frame->pose.cast<float>();
    float *tf = tf_se3.data();

    // find correspondence for plane features
    for (int i = 0; i < num_points_flat; ++i)
    {
        //NOTE: Sophus is too slow
        ceres::SE3TransformPoint(tf, frame->feature_lidar->points_surf[i].data, point.data);
        point.intensity = frame->feature_lidar->points_surf[i].intensity;
        kdtree_last.nearestKSearch(point, 3, points_index, points_distance);
        // clang-format off
        if (points_index[0] < points_surf_last.size() && points_distance[0] < distance_threshold 
         && points_index[1] < points_surf_last.size() && points_distance[1] < distance_threshold 
         && points_index[2] < points_surf_last.size() && points_distance[2] < distance_threshold)
        // clang-format on
        {
            Vector3d curr_point(frame->feature_lidar->points_surf[i].x,
                                frame->feature_lidar->points_surf[i].y,
                                frame->feature_lidar->points_surf[i].z);
            Vector3d last_point_a(points_surf_last[points_index[0]].x,
                                  points_surf_last[points_index[0]].y,
                                  points_surf_last[points_index[0]].z);
            Vector3d last_point_b(points_surf_last[points_index[1]].x,
                                  points_surf_last[points_index[1]].y,
                                  points_surf_last[points_index[1]].z);
            Vector3d last_point_c(points_surf_last[points_index[2]].x,
                                  points_surf_last[points_index[2]].y,
                                  points_surf_last[points_index[2]].z);
            ceres::CostFunction *cost_function;
            cost_function = LidarPlaneErrorYXY::Create(curr_point, last_point_a, last_point_b, last_point_c, map_frame->pose, para, frame->weights.lidar_surf);
            problem.AddResidualBlock(ProblemType::LidarPlaneErrorYXY, cost_function, loss_function, para, para + 3, para + 4);
        }
    }

    if (frame->id == map_frame->id + 1)
    {
        ceres::CostFunction *cost_function = PoseErrorYXY::Create(para, frame->weights.visual);
        problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para, para + 3, para + 4);
    }
}

} // namespace lvio_fusion
