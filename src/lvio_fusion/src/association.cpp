#include "lvio_fusion/lidar/association.h"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/loop_error.hpp"
#include "lvio_fusion/lidar/feature.h"
#include "lvio_fusion/utility.h"

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace lvio_fusion
{

void FeatureAssociation::AddScan(double time, Point3Cloud::Ptr new_scan)
{
    static double head = 0;
    raw_point_clouds_[time] = new_scan;

    Frames new_kfs = map_->GetKeyFrames(head, time);
    for (auto pair_kf : new_kfs)
    {
        PointICloud point_cloud;
        if (AlignScan(pair_kf.first, point_cloud))
        {
            Process(point_cloud, pair_kf.second);
            head = pair_kf.first;
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
    SE3d pose = map_->ComputePose(time);
    auto p1 = lidar_->Sensor2World(Vector3d(point.x, point.y, point.z), pose);
    auto p2 = lidar_->World2Sensor(p1, frame->pose);
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
    static const float threshold = 1;
    lidar::Feature::Ptr feature = lidar::Feature::Create();
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
                    feature->points_ground.push_back(points_segmented[k]);
                }
                else if (curvatures[k] < threshold)
                {
                    feature->points_surf.push_back(points_segmented[k]);
                }
                feature->points_full.push_back(points_segmented[k]);
            }
        }
    }

    PointICloud::Ptr temp(new PointICloud());
    pcl::VoxelGrid<PointI> voxel_filter;
    pcl::copyPointCloud(feature->points_full, *temp);
    voxel_filter.setLeafSize(lidar_->resolution, lidar_->resolution, lidar_->resolution);
    voxel_filter.setInputCloud(temp);
    voxel_filter.filter(feature->points_full);

    pcl::copyPointCloud(feature->points_surf, *temp);
    voxel_filter.setLeafSize(2 * lidar_->resolution, 2 * lidar_->resolution, 2 * lidar_->resolution);
    voxel_filter.setInputCloud(temp);
    voxel_filter.filter(feature->points_surf);

    pcl::copyPointCloud(feature->points_ground, *temp);
    voxel_filter.setLeafSize(2 * lidar_->resolution, 2 * lidar_->resolution, 2 * lidar_->resolution);
    voxel_filter.setInputCloud(temp);
    voxel_filter.filter(feature->points_ground);

    SegmentGround(feature->points_ground, feature->points_surf);

    frame->feature_lidar = feature;
}

inline void FeatureAssociation::SegmentGround(PointICloud &points_ground, PointICloud &points_surf)
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
    seg.setDistanceThreshold(0.5 * lidar_->resolution);
    seg.setInputCloud(pointcloud_seg);
    seg.segment(*inliers, coefficients);
    pcl::ExtractIndices<PointI> extract;
    extract.setInputCloud(pointcloud_seg);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(points_ground);
}

void FeatureAssociation::ScanToMapWithGround(Frame::Ptr frame, Frame::Ptr map_frame, double *para, ceres::Problem &problem)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    PointICloud &points_less_flat_last = map_frame->feature_lidar->points_full;
    problem.AddParameterBlock(para + 1, 1);
    problem.AddParameterBlock(para + 2, 1);
    problem.AddParameterBlock(para + 5, 1);

    pcl::KdTreeFLANN<PointI> kdtree_flat_last;
    kdtree_flat_last.setInputCloud(boost::make_shared<PointICloud>(points_less_flat_last));

    PointI point;
    std::vector<int> points_index;
    std::vector<float> points_distance;

    static const double distance_threshold = lidar_->resolution * lidar_->resolution * 100; // squared
    static const double nearby_scan = 2;
    int num_points_flat = frame->feature_lidar->points_ground.size();
    Sophus::SE3f tf_se3 = lidar_->TransformMatrix(frame->pose).cast<float>();
    float *tf = tf_se3.data();

    double weights[10] = {10000, 10000, 10000, 1000, 1000, 1000};
    ceres::CostFunction *cost_function = PoseErrorRPZ::Create(para, weights);
    problem.AddResidualBlock(cost_function, loss_function, para + 1, para + 2, para + 5);

    // find correspondence for ground features
    for (int i = 0; i < num_points_flat; ++i)
    {
        //NOTE: Sophus is too slow
        // lidar_->Transform(points_flat[i], current_frame->pose, last_frame->pose, point);
        ceres::SE3TransformPoint(tf, frame->feature_lidar->points_ground[i].data, point.data);
        point.intensity = frame->feature_lidar->points_ground[i].intensity;
        kdtree_flat_last.nearestKSearch(point, 1, points_index, points_distance);
        int closest_index = -1, closest_index2 = -1, closest_index3 = -1;
        if (points_index[0] < points_less_flat_last.size() && points_distance[0] < distance_threshold)
        {
            closest_index = points_index[0];

            // get closest point's scan ID
            int scan_id = int(points_less_flat_last[closest_index].intensity);
            double min_distance2 = distance_threshold, min_distance3 = distance_threshold;
            // search in the direction of increasing scan line
            for (int j = closest_index + 1; j < (int)points_less_flat_last.size(); ++j)
            {
                // if not in nearby scans, end the loop
                if (int(points_less_flat_last[j].intensity) > (scan_id + nearby_scan))
                    break;

                double point_distance = (points_less_flat_last[j].x - point.x) *
                                            (points_less_flat_last[j].x - point.x) +
                                        (points_less_flat_last[j].y - point.y) *
                                            (points_less_flat_last[j].y - point.y) +
                                        (points_less_flat_last[j].z - point.z) *
                                            (points_less_flat_last[j].z - point.z);

                // if in the same or lower scan line
                if (int(points_less_flat_last[j].intensity) <= scan_id && point_distance < min_distance2)
                {
                    min_distance2 = point_distance;
                    closest_index2 = j;
                }
                // if in the higher scan line
                else if (int(points_less_flat_last[j].intensity) > scan_id && point_distance < min_distance3)
                {
                    min_distance3 = point_distance;
                    closest_index3 = j;
                }
            }
            // search in the direction of decreasing scan line
            for (int j = closest_index - 1; j >= 0; --j)
            {
                // if not in nearby scans, end the loop
                if (int(points_less_flat_last[j].intensity) < (scan_id - nearby_scan))
                    break;

                double point_distance = (points_less_flat_last[j].x - point.x) *
                                            (points_less_flat_last[j].x - point.x) +
                                        (points_less_flat_last[j].y - point.y) *
                                            (points_less_flat_last[j].y - point.y) +
                                        (points_less_flat_last[j].z - point.z) *
                                            (points_less_flat_last[j].z - point.z);

                // if in the same or higher scan line
                if (int(points_less_flat_last[j].intensity) >= scan_id && point_distance < min_distance2)
                {
                    min_distance2 = point_distance;
                    closest_index2 = j;
                }
                else if (int(points_less_flat_last[j].intensity) < scan_id && point_distance < min_distance3)
                {
                    // find nearer point
                    min_distance3 = point_distance;
                    closest_index3 = j;
                }
            }
            if (closest_index2 >= 0 && closest_index3 >= 0)
            {

                Vector3d curr_point(frame->feature_lidar->points_ground[i].x,
                                    frame->feature_lidar->points_ground[i].y,
                                    frame->feature_lidar->points_ground[i].z);
                Vector3d last_point_a(points_less_flat_last[closest_index].x,
                                      points_less_flat_last[closest_index].y,
                                      points_less_flat_last[closest_index].z);
                Vector3d last_point_b(points_less_flat_last[closest_index2].x,
                                      points_less_flat_last[closest_index2].y,
                                      points_less_flat_last[closest_index2].z);
                Vector3d last_point_c(points_less_flat_last[closest_index3].x,
                                      points_less_flat_last[closest_index3].y,
                                      points_less_flat_last[closest_index3].z);

                ceres::CostFunction *cost_function;
                cost_function = LidarPlaneErrorRPZ::Create(curr_point, last_point_a, last_point_b, last_point_c, lidar_, map_frame->pose, para, frame->weights.lidar_ground);
                problem.AddResidualBlock(cost_function, loss_function, para + 1, para + 2, para + 5);
            }
        }
    }
}

void FeatureAssociation::ScanToMapWithSegmented(Frame::Ptr frame, Frame::Ptr map_frame, double *para, ceres::Problem &problem)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1);
    // PointICloud &points_less_sharp_last = map_frame->feature_lidar->points_less_sharp;
    PointICloud &points_less_flat_last = map_frame->feature_lidar->points_full;
    problem.AddParameterBlock(para + 0, 1);
    problem.AddParameterBlock(para + 3, 1);
    problem.AddParameterBlock(para + 4, 1);

    // pcl::KdTreeFLANN<PointI> kdtree_sharp_last;
    pcl::KdTreeFLANN<PointI> kdtree_flat_last;
    // kdtree_sharp_last.setInputCloud(boost::make_shared<PointICloud>(points_less_sharp_last));
    kdtree_flat_last.setInputCloud(boost::make_shared<PointICloud>(points_less_flat_last));

    PointI point;
    std::vector<int> points_index;
    std::vector<float> points_distance;

    static const double distance_threshold = lidar_->resolution * lidar_->resolution * 25; // squared
    static const double nearby_scan = 2;
    // int num_points_sharp = frame->feature_lidar->points_sharp.size();
    int num_points_flat = frame->feature_lidar->points_surf.size();
    Sophus::SE3f tf_se3 = lidar_->TransformMatrix(frame->pose).cast<float>();
    float *tf = tf_se3.data();

    double weights[10] = {10000, 10000, 10000, 1000, 1000, 1000};
    ceres::CostFunction *cost_function = PoseErrorRPZ::Create(para, weights);
    problem.AddResidualBlock(cost_function, loss_function, para, para + 3, para + 4);

    // find correspondence for plane features
    for (int i = 0; i < num_points_flat; ++i)
    {
        //NOTE: Sophus is too slow
        // lidar_->Transform(frame->feature_lidar->points_flat[i], current_frame->pose, last_frame->pose, point);
        ceres::SE3TransformPoint(tf, frame->feature_lidar->points_surf[i].data, point.data);
        point.intensity = frame->feature_lidar->points_surf[i].intensity;
        kdtree_flat_last.nearestKSearch(point, 1, points_index, points_distance);
        int closest_index = -1, closest_index2 = -1, closest_index3 = -1;
        if (points_index[0] < points_less_flat_last.size() && points_distance[0] < distance_threshold)
        {
            closest_index = points_index[0];

            // get closest point's scan ID
            int scan_id = int(points_less_flat_last[closest_index].intensity);
            double min_distance2 = distance_threshold, min_distance3 = distance_threshold;

            // search in the direction of increasing scan line
            for (int j = closest_index + 1; j < (int)points_less_flat_last.size(); ++j)
            {
                // if not in nearby scans, end the loop
                if (int(points_less_flat_last[j].intensity) > (scan_id + nearby_scan))
                    break;

                double point_distance = (points_less_flat_last[j].x - point.x) *
                                            (points_less_flat_last[j].x - point.x) +
                                        (points_less_flat_last[j].y - point.y) *
                                            (points_less_flat_last[j].y - point.y) +
                                        (points_less_flat_last[j].z - point.z) *
                                            (points_less_flat_last[j].z - point.z);

                // if in the same or lower scan line
                if (int(points_less_flat_last[j].intensity) <= scan_id && point_distance < min_distance2)
                {
                    min_distance2 = point_distance;
                    closest_index2 = j;
                }
                // if in the higher scan line
                else if (int(points_less_flat_last[j].intensity) > scan_id && point_distance < min_distance3)
                {
                    min_distance3 = point_distance;
                    closest_index3 = j;
                }
            }

            // search in the direction of decreasing scan line
            for (int j = closest_index - 1; j >= 0; --j)
            {
                // if not in nearby scans, end the loop
                if (int(points_less_flat_last[j].intensity) < (scan_id - nearby_scan))
                    break;

                double point_distance = (points_less_flat_last[j].x - point.x) *
                                            (points_less_flat_last[j].x - point.x) +
                                        (points_less_flat_last[j].y - point.y) *
                                            (points_less_flat_last[j].y - point.y) +
                                        (points_less_flat_last[j].z - point.z) *
                                            (points_less_flat_last[j].z - point.z);

                // if in the same or higher scan line
                if (int(points_less_flat_last[j].intensity) >= scan_id && point_distance < min_distance2)
                {
                    min_distance2 = point_distance;
                    closest_index2 = j;
                }
                else if (int(points_less_flat_last[j].intensity) < scan_id && point_distance < min_distance3)
                {
                    // find nearer point
                    min_distance3 = point_distance;
                    closest_index3 = j;
                }
            }

            if (closest_index2 >= 0 && closest_index3 >= 0)
            {

                Vector3d curr_point(frame->feature_lidar->points_surf[i].x,
                                    frame->feature_lidar->points_surf[i].y,
                                    frame->feature_lidar->points_surf[i].z);
                Vector3d last_point_a(points_less_flat_last[closest_index].x,
                                      points_less_flat_last[closest_index].y,
                                      points_less_flat_last[closest_index].z);
                Vector3d last_point_b(points_less_flat_last[closest_index2].x,
                                      points_less_flat_last[closest_index2].y,
                                      points_less_flat_last[closest_index2].z);
                Vector3d last_point_c(points_less_flat_last[closest_index3].x,
                                      points_less_flat_last[closest_index3].y,
                                      points_less_flat_last[closest_index3].z);
                ceres::CostFunction *cost_function;
                cost_function = LidarPlaneErrorYXY::Create(curr_point, last_point_a, last_point_b, last_point_c, lidar_, map_frame->pose, para, frame->weights.lidar_surf);
                problem.AddResidualBlock(cost_function, loss_function, para, para + 3, para + 4);
            }
        }
    }
}
} // namespace lvio_fusion
