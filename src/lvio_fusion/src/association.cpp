#include "lvio_fusion/lidar/association.h"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/loop_error.hpp"
#include "lvio_fusion/lidar/feature.h"
#include "lvio_fusion/utility.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

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

void FeatureAssociation::MergeScan(const PointICloud &in, SE3d from_pose, SE3d to_pose, PointICloud &out)
{
    Sophus::SE3f tf_se3 = lidar_->TransformMatrix(from_pose, to_pose).cast<float>();
    float *tf = tf_se3.data();
    for (auto point_in : in)
    {
        PointI point_out;
        ceres::SE3TransformPoint(tf, point_in.data, point_out.data);
        point_out.intensity = point_in.intensity;
        out.push_back(point_out);
    }
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

    PointICloud points_segmented, points_outlier;
    auto segmented_info = projection_->Process(points, points_segmented, points_outlier);

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

    MarkOccludedPoints(points_segmented, segemented_info);

    ExtractFeatures(points_segmented, segemented_info, frame);
}

void FeatureAssociation::AdjustDistortion(PointICloud &points_segmented, SegmentedInfo &segemented_info)
{
    bool halfPassed = false;
    int cloudSize = points_segmented.size();
    PointI point;
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = points_segmented[i].x;
        point.y = points_segmented[i].y;
        point.z = points_segmented[i].z;

        float ori = -atan2(point.y, point.x);
        if (!halfPassed)
        {
            if (ori < segemented_info.startOrientation - M_PI / 2)
                ori += 2 * M_PI;
            else if (ori > segemented_info.startOrientation + M_PI * 3 / 2)
                ori -= 2 * M_PI;

            if (ori - segemented_info.startOrientation > M_PI)
                halfPassed = true;
        }
        else
        {
            ori += 2 * M_PI;

            if (ori < segemented_info.endOrientation - M_PI * 3 / 2)
                ori += 2 * M_PI;
            else if (ori > segemented_info.endOrientation + M_PI / 2)
                ori -= 2 * M_PI;
        }

        float relTime = (ori - segemented_info.startOrientation) / segemented_info.orientationDiff;
        point.intensity = int(points_segmented[i].intensity) + cycle_time_ * relTime;
        //TODO:deskew
        points_segmented[i] = point;
    }
}

void FeatureAssociation::CalculateSmoothness(PointICloud &points_segmented, SegmentedInfo &segemented_info)
{
    int cloudSize = points_segmented.size();
    for (int i = 5; i < cloudSize - 5; i++)
    {
        // clang-format off
        float diffRange = segemented_info.segmentedCloudRange[i - 5] 
                        + segemented_info.segmentedCloudRange[i - 4] 
                        + segemented_info.segmentedCloudRange[i - 3] 
                        + segemented_info.segmentedCloudRange[i - 2] 
                        + segemented_info.segmentedCloudRange[i - 1] 
                        - segemented_info.segmentedCloudRange[i] * 10 
                        + segemented_info.segmentedCloudRange[i + 1] 
                        + segemented_info.segmentedCloudRange[i + 2] 
                        + segemented_info.segmentedCloudRange[i + 3] 
                        + segemented_info.segmentedCloudRange[i + 4] 
                        + segemented_info.segmentedCloudRange[i + 5];
        // clang-format on
        cloudCurvature[i] = diffRange * diffRange;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
        cloudSmoothness[i].value = cloudCurvature[i];
        cloudSmoothness[i].ind = i;
    }
}

void FeatureAssociation::MarkOccludedPoints(PointICloud &points_segmented, SegmentedInfo &segemented_info)
{
    int cloudSize = points_segmented.size();

    for (int i = 5; i < cloudSize - 6; ++i)
    {

        float depth1 = segemented_info.segmentedCloudRange[i];
        float depth2 = segemented_info.segmentedCloudRange[i + 1];
        int columnDiff = std::abs(int(segemented_info.segmentedCloudColInd[i + 1] - segemented_info.segmentedCloudColInd[i]));

        if (columnDiff < 10)
        {

            if (depth1 - depth2 > 0.3)
            {
                cloudNeighborPicked[i - 5] = 1;
                cloudNeighborPicked[i - 4] = 1;
                cloudNeighborPicked[i - 3] = 1;
                cloudNeighborPicked[i - 2] = 1;
                cloudNeighborPicked[i - 1] = 1;
                cloudNeighborPicked[i] = 1;
            }
            else if (depth2 - depth1 > 0.3)
            {
                cloudNeighborPicked[i + 1] = 1;
                cloudNeighborPicked[i + 2] = 1;
                cloudNeighborPicked[i + 3] = 1;
                cloudNeighborPicked[i + 4] = 1;
                cloudNeighborPicked[i + 5] = 1;
                cloudNeighborPicked[i + 6] = 1;
            }
        }

        float diff1 = std::abs(float(segemented_info.segmentedCloudRange[i - 1] - segemented_info.segmentedCloudRange[i]));
        float diff2 = std::abs(float(segemented_info.segmentedCloudRange[i + 1] - segemented_info.segmentedCloudRange[i]));

        if (diff1 > 0.02 * segemented_info.segmentedCloudRange[i] && diff2 > 0.02 * segemented_info.segmentedCloudRange[i])
            cloudNeighborPicked[i] = 1;
    }
}

void FeatureAssociation::ExtractFeatures(PointICloud &points_segmented, SegmentedInfo &segemented_info, Frame::Ptr frame)
{
    lidar::Feature::Ptr feature = lidar::Feature::Create();

    pcl::VoxelGrid<PointI> voxel_filter;
    voxel_filter.setLeafSize(lidar_->resolution, lidar_->resolution, lidar_->resolution);
    for (int i = 0; i < num_scans_; i++)
    {
        PointICloud::Ptr scan_less_flat(new PointICloud());
        // divide one scan into six segments
        for (int j = 0; j < 6; j++)
        {

            int sp = (segemented_info.startRingIndex[i] * (6 - j) + segemented_info.endRingIndex[i] * j) / 6;
            int ep = (segemented_info.startRingIndex[i] * (5 - j) + segemented_info.endRingIndex[i] * (j + 1)) / 6 - 1;

            if (sp >= ep)
                continue;

            std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, smoothness_t());

            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSmoothness[k].ind;
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > edgeThreshold &&
                    segemented_info.segmentedCloudGroundFlag[ind] == false)
                {

                    largestPickedNum++;
                    if (largestPickedNum <= edgeFeatureNum)
                    {
                        cloudLabel[ind] = 2;
                        feature->points_sharp.push_back(points_segmented[ind]);
                        feature->points_less_sharp.push_back(points_segmented[ind]);
                    }
                    else if (largestPickedNum <= 20)
                    {
                        cloudLabel[ind] = 1;
                        feature->points_less_sharp.push_back(points_segmented[ind]);
                    }
                    else
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    {
                        int columnDiff = std::abs(int(segemented_info.segmentedCloudColInd[ind + l] - segemented_info.segmentedCloudColInd[ind + l - 1]));
                        if (columnDiff > 10)
                            break;
                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        int columnDiff = std::abs(int(segemented_info.segmentedCloudColInd[ind + l] - segemented_info.segmentedCloudColInd[ind + l + 1]));
                        if (columnDiff > 10)
                            break;
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSmoothness[k].ind;
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < surfThreshold)
                {
                    cloudLabel[ind] = -1;
                    if (segemented_info.segmentedCloudGroundFlag[ind] == true)
                    {
                        feature->points_ground.push_back(points_segmented[ind]);
                    }
                    else
                    {
                        feature->points_flat.push_back(points_segmented[ind]);
                    }

                    smallestPickedNum++;
                    if (smallestPickedNum >= surfFeatureNum)
                        break;

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    {

                        int columnDiff = std::abs(int(segemented_info.segmentedCloudColInd[ind + l] - segemented_info.segmentedCloudColInd[ind + l - 1]));
                        if (columnDiff > 10)
                            break;

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {

                        int columnDiff = std::abs(int(segemented_info.segmentedCloudColInd[ind + l] - segemented_info.segmentedCloudColInd[ind + l + 1]));
                        if (columnDiff > 10)
                            break;

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    feature->points_less_flat.push_back(points_segmented[k]);
                }
            }
        }
        PointICloud scan_less_flat_ds;
        voxel_filter.setInputCloud(scan_less_flat);
        voxel_filter.filter(scan_less_flat_ds);
        feature->points_less_flat += scan_less_flat_ds;
    }
    frame->feature_lidar = feature;
}

void FeatureAssociation::ScanToMapWithGround(Frame::Ptr frame, Frame::Ptr map_frame, double *para, ceres::Problem &problem)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    PointICloud &points_ground = frame->feature_lidar->points_ground;
    PointICloud &points_less_flat_last = map_frame->feature_lidar->points_less_flat;

    pcl::KdTreeFLANN<PointI> kdtree_flat_last;
    kdtree_flat_last.setInputCloud(boost::make_shared<PointICloud>(points_less_flat_last));

    PointI point;
    std::vector<int> points_index;
    std::vector<float> points_distance;

    static const double distance_threshold = lidar_->resolution * lidar_->resolution * 100; // squared
    static const double nearby_scan = 2;
    int num_points_flat = points_ground.size();
    Sophus::SE3f tf_se3 = lidar_->TransformMatrix(frame->pose, map_frame->pose).cast<float>();
    float *tf = tf_se3.data();

    // find correspondence for ground features
    for (int i = 0; i < num_points_flat; ++i)
    {
        if(&(frame->feature_lidar->points_ground) != &points_ground)
        {
            LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
        }
        //NOTE: Sophus is too slow
        // lidar_->Transform(points_flat[i], current_frame->pose, last_frame->pose, point);
        ceres::SE3TransformPoint(tf, points_ground[i].data, point.data);
        point.intensity = points_ground[i].intensity;
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

                Vector3d curr_point(points_ground[i].x,
                                    points_ground[i].y,
                                    points_ground[i].z);
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
                cost_function = LidarPlaneErrorRPZ::Create(curr_point, last_point_a, last_point_b, last_point_c, lidar_, map_frame->pose, para);
                problem.AddResidualBlock(cost_function, loss_function, para + 2, para + 1, para + 5);
            }
        }
    }
}

void FeatureAssociation::ScanToMapWithSegmented(Frame::Ptr frame, Frame::Ptr map_frame, double *para, ceres::Problem &problem)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    PointICloud &points_sharp = frame->feature_lidar->points_sharp;
    PointICloud &points_flat = frame->feature_lidar->points_flat;
    PointICloud &points_less_sharp_last = map_frame->feature_lidar->points_less_sharp;
    PointICloud &points_less_flat_last = map_frame->feature_lidar->points_less_flat;

    pcl::KdTreeFLANN<PointI> kdtree_sharp_last;
    pcl::KdTreeFLANN<PointI> kdtree_flat_last;
    kdtree_sharp_last.setInputCloud(boost::make_shared<PointICloud>(points_less_sharp_last));
    kdtree_flat_last.setInputCloud(boost::make_shared<PointICloud>(points_less_flat_last));

    PointI point;
    std::vector<int> points_index;
    std::vector<float> points_distance;

    static const double distance_threshold = lidar_->resolution * lidar_->resolution * 25; // squared
    static const double nearby_scan = 2;
    int num_points_sharp = points_sharp.size();
    int num_points_flat = points_flat.size();
    Sophus::SE3f tf_se3 = lidar_->TransformMatrix(frame->pose, map_frame->pose).cast<float>();
    float *tf = tf_se3.data();

    //TODO: bad
    // find correspondence for corner features
    for (int i = 0; i < num_points_sharp; ++i)
    {
        //NOTE: Sophus is too slow
        // lidar_->Transform(points_sharp[i], current_frame->pose, last_frame->pose, point);  //  too slow
        ceres::SE3TransformPoint(tf, points_sharp[i].data, point.data);
        point.intensity = points_sharp[i].intensity;
        kdtree_sharp_last.nearestKSearch(point, 1, points_index, points_distance);
        int closest_index = -1, closest_index2 = -1;
        if (points_index[0] < points_less_sharp_last.size() && points_distance[0] < distance_threshold)
        {
            closest_index = points_index[0];
            int scan_id = int(points_less_sharp_last[closest_index].intensity);
            double min_distance = distance_threshold;
            // point b in the direction of increasing scan line
            for (int j = closest_index + 1; j < (int)points_less_sharp_last.size(); ++j)
            {
                // if in the same scan line, continue
                if (int(points_less_sharp_last[j].intensity) <= scan_id)
                    continue;
                // if not in nearby scans, end the loop
                if (int(points_less_sharp_last[j].intensity) > (scan_id + nearby_scan))
                    break;
                double point_distance = (points_less_sharp_last[j].x - point.x) *
                                            (points_less_sharp_last[j].x - point.x) +
                                        (points_less_sharp_last[j].y - point.y) *
                                            (points_less_sharp_last[j].y - point.y) +
                                        (points_less_sharp_last[j].z - point.z) *
                                            (points_less_sharp_last[j].z - point.z);
                if (point_distance < min_distance)
                {
                    // find nearer point
                    min_distance = point_distance;
                    closest_index2 = j;
                }
            }
            // point b in the direction of decreasing scan line
            for (int j = closest_index - 1; j >= 0; --j)
            {
                // if in the same scan line, continue
                if (int(points_less_sharp_last[j].intensity) >= scan_id)
                    continue;
                // if not in nearby scans, end the loop
                if (int(points_less_sharp_last[j].intensity) < (scan_id - nearby_scan))
                    break;
                double point_distance = (points_less_sharp_last[j].x - point.x) *
                                            (points_less_sharp_last[j].x - point.x) +
                                        (points_less_sharp_last[j].y - point.y) *
                                            (points_less_sharp_last[j].y - point.y) +
                                        (points_less_sharp_last[j].z - point.z) *
                                            (points_less_sharp_last[j].z - point.z);
                if (point_distance < min_distance)
                {
                    // find nearer point
                    min_distance = point_distance;
                    closest_index2 = j;
                }
            }
        }
        if (closest_index2 >= 0) // both A and B is valid
        {
            Vector3d curr_point(points_sharp[i].x,
                                points_sharp[i].y,
                                points_sharp[i].z);
            Vector3d last_point_a(points_less_sharp_last[closest_index].x,
                                  points_less_sharp_last[closest_index].y,
                                  points_less_sharp_last[closest_index].z);
            Vector3d last_point_b(points_less_sharp_last[closest_index2].x,
                                  points_less_sharp_last[closest_index2].y,
                                  points_less_sharp_last[closest_index2].z);
            ceres::CostFunction *cost_function;
            cost_function = LidarEdgeErrorYXY::Create(curr_point, last_point_a, last_point_b, lidar_, map_frame->pose, para);
            problem.AddResidualBlock(cost_function, loss_function, para, para + 3, para + 4);
        }
    }

    // find correspondence for plane features
    for (int i = 0; i < num_points_flat; ++i)
    {
        if(&(frame->feature_lidar->points_flat) != &points_flat)
        {
            LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
        }
        //NOTE: Sophus is too slow
        // lidar_->Transform(points_flat[i], current_frame->pose, last_frame->pose, point);
        ceres::SE3TransformPoint(tf, points_flat[i].data, point.data);
        point.intensity = points_flat[i].intensity;
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

                Vector3d curr_point(points_flat[i].x,
                                    points_flat[i].y,
                                    points_flat[i].z);
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
                cost_function = LidarPlaneErrorYXY::Create(curr_point, last_point_a, last_point_b, last_point_c, lidar_, map_frame->pose, para);
                problem.AddResidualBlock(cost_function, loss_function, para, para + 3, para + 4);
            }
        }
    }
}
} // namespace lvio_fusion
