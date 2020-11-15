#include "lvio_fusion/lidar/association.h"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/loop_error.hpp"
#include "lvio_fusion/lidar/feature.h"
#include "lvio_fusion/utility.h"

#include <pcl/filters/statistical_outlier_removal.h>
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
        if (TimeAlign(pair_kf.first, point_cloud))
        {
            Process(point_cloud, pair_kf.second);
            head = pair_kf.first;
        }
    }
}

bool FeatureAssociation::TimeAlign(double time, PointICloud &out)
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

void remove_outliers(PointICloud &in, PointICloud &out)
{
    pcl::StatisticalOutlierRemoval<PointI> sor;
    sor.setInputCloud(boost::make_shared<PointICloud>(in));
    sor.setMeanK(10);
    sor.setStddevMulThresh(1);
    sor.filter(out);
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
    int cloudSize = points_segmented.points.size();
    PointI point;
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = points_segmented.points[i].x;
        point.y = points_segmented.points[i].y;
        point.z = points_segmented.points[i].z;

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
        point.intensity = int(points_segmented.points[i].intensity) + cycle_time_ * relTime;
        //TODO:deskew
        points_segmented.points[i] = point;
    }
}

void FeatureAssociation::CalculateSmoothness(PointICloud &points_segmented, SegmentedInfo &segemented_info)
{
    int cloudSize = points_segmented.points.size();
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
    int cloudSize = points_segmented.points.size();

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
    PointICloud points_sharp;
    PointICloud points_less_sharp;
    PointICloud points_flat;
    PointICloud point_less_flat;

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
                        points_sharp.push_back(points_segmented.points[ind]);
                        points_less_sharp.push_back(points_segmented.points[ind]);
                    }
                    else if (largestPickedNum <= 20)
                    {
                        cloudLabel[ind] = 1;
                        points_less_sharp.push_back(points_segmented.points[ind]);
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
                    cloudCurvature[ind] < surfThreshold &&
                    segemented_info.segmentedCloudGroundFlag[ind] == true)
                {

                    cloudLabel[ind] = -1;
                    points_flat.push_back(points_segmented.points[ind]);

                    smallestPickedNum++;
                    if (smallestPickedNum >= surfFeatureNum)
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

            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    point_less_flat.push_back(points_segmented.points[k]);
                }
            }
        }
        PointICloud scan_less_flat_ds;
        voxel_filter.setInputCloud(scan_less_flat);
        voxel_filter.filter(scan_less_flat_ds);
        point_less_flat += scan_less_flat_ds;
    }
    lidar::Feature::Ptr feature = lidar::Feature::Create(points_sharp, points_less_sharp, points_flat, point_less_flat);
    frame->feature_lidar = feature;
}

void FeatureAssociation::Associate(Frame::Ptr current_frame, Frame::Ptr last_frame, ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    PointICloud &points_sharp = current_frame->feature_lidar->points_sharp;
    PointICloud &points_less_sharp = current_frame->feature_lidar->points_less_sharp;
    PointICloud &points_flat = current_frame->feature_lidar->points_flat;
    PointICloud &points_less_flat = current_frame->feature_lidar->points_less_flat;
    PointICloud &points_less_sharp_last = last_frame->feature_lidar->points_less_sharp;
    PointICloud &points_less_flat_last = last_frame->feature_lidar->points_less_flat;

    pcl::KdTreeFLANN<PointI> kdtree_sharp_last;
    pcl::KdTreeFLANN<PointI> kdtree_flat_last;
    kdtree_sharp_last.setInputCloud(boost::make_shared<PointICloud>(points_less_sharp_last));
    kdtree_flat_last.setInputCloud(boost::make_shared<PointICloud>(points_less_flat_last));

    double *para_kf = current_frame->pose.data();
    double *para_last_kf = last_frame->pose.data();

    PointI point;
    std::vector<int> points_index;
    std::vector<float> points_distance;

    static const double distance_threshold = lidar_->resolution * lidar_->resolution * 100; // squared
    static const double nearby_scan = 2;
    int num_points_sharp = points_sharp.points.size();
    int num_points_flat = points_flat.points.size();
    Sophus::SE3f tf_se3 = lidar_->TransformMatrix(current_frame->pose, last_frame->pose).cast<float>();
    float *tf = tf_se3.data();

    //TODO: bad
    // find correspondence for corner features
    for (int i = 0; i < num_points_sharp; ++i)
    {
        //NOTE: Sophus is too slow
        // lidar_->Transform(points_sharp.points[i], current_frame->pose, last_frame->pose, point);  //  too slow
        ceres::SE3TransformPoint(tf, points_sharp.points[i].data, point.data);
        point.intensity = points_sharp.points[i].intensity;
        kdtree_sharp_last.nearestKSearch(point, 1, points_index, points_distance);
        int closest_index = -1, closest_index2 = -1;
        if (points_distance[0] < distance_threshold)
        {
            closest_index = points_index[0];
            int scan_id = int(points_less_sharp_last.points[closest_index].intensity);
            double min_distance = distance_threshold;
            // point b in the direction of increasing scan line
            for (int j = closest_index + 1; j < (int)points_less_sharp_last.points.size(); ++j)
            {
                // if in the same scan line, continue
                if (int(points_less_sharp_last.points[j].intensity) <= scan_id)
                    continue;
                // if not in nearby scans, end the loop
                if (int(points_less_sharp_last.points[j].intensity) > (scan_id + nearby_scan))
                    break;
                double point_distance = (points_less_sharp_last.points[j].x - point.x) *
                                            (points_less_sharp_last.points[j].x - point.x) +
                                        (points_less_sharp_last.points[j].y - point.y) *
                                            (points_less_sharp_last.points[j].y - point.y) +
                                        (points_less_sharp_last.points[j].z - point.z) *
                                            (points_less_sharp_last.points[j].z - point.z);
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
                if (int(points_less_sharp_last.points[j].intensity) >= scan_id)
                    continue;
                // if not in nearby scans, end the loop
                if (int(points_less_sharp_last.points[j].intensity) < (scan_id - nearby_scan))
                    break;
                double point_distance = (points_less_sharp_last.points[j].x - point.x) *
                                            (points_less_sharp_last.points[j].x - point.x) +
                                        (points_less_sharp_last.points[j].y - point.y) *
                                            (points_less_sharp_last.points[j].y - point.y) +
                                        (points_less_sharp_last.points[j].z - point.z) *
                                            (points_less_sharp_last.points[j].z - point.z);
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
            Vector3d curr_point(points_sharp.points[i].x,
                                points_sharp.points[i].y,
                                points_sharp.points[i].z);
            Vector3d last_point_a(points_less_sharp_last.points[closest_index].x,
                                  points_less_sharp_last.points[closest_index].y,
                                  points_less_sharp_last.points[closest_index].z);
            Vector3d last_point_b(points_less_sharp_last.points[closest_index2].x,
                                  points_less_sharp_last.points[closest_index2].y,
                                  points_less_sharp_last.points[closest_index2].z);
            ceres::CostFunction *cost_function;
            cost_function = LidarEdgeError::Create(curr_point, last_point_a, last_point_b, lidar_);
            problem.AddResidualBlock(cost_function, loss_function, para_last_kf, para_kf);
        }
    }

    // find correspondence for plane features
    for (int i = 0; i < num_points_flat; ++i)
    {
        //NOTE: Sophus is too slow
        // lidar_->Transform(points_flat.points[i], current_frame->pose, last_frame->pose, point);
        ceres::SE3TransformPoint(tf, points_flat.points[i].data, point.data);
        point.intensity = points_flat.points[i].intensity;
        try
        {
            kdtree_flat_last.nearestKSearch(point, 1, points_index, points_distance);
        }
        catch (const std::exception &e)
        {
            continue;
        }

        int closest_index = -1, closest_index2 = -1, closest_index3 = -1;
        if (points_distance[0] < distance_threshold && points_index[0] < points_less_flat_last.size())
        {
            closest_index = points_index[0];

            // get closest point's scan ID
            int scan_id = int(points_less_flat_last.points[closest_index].intensity);
            double min_distance2 = distance_threshold, min_distance3 = distance_threshold;

            // search in the direction of increasing scan line
            for (int j = closest_index + 1; j < (int)points_less_flat_last.points.size(); ++j)
            {
                // if not in nearby scans, end the loop
                if (int(points_less_flat_last.points[j].intensity) > (scan_id + nearby_scan))
                    break;

                double point_distance = (points_less_flat_last.points[j].x - point.x) *
                                            (points_less_flat_last.points[j].x - point.x) +
                                        (points_less_flat_last.points[j].y - point.y) *
                                            (points_less_flat_last.points[j].y - point.y) +
                                        (points_less_flat_last.points[j].z - point.z) *
                                            (points_less_flat_last.points[j].z - point.z);

                // if in the same or lower scan line
                if (int(points_less_flat_last.points[j].intensity) <= scan_id && point_distance < min_distance2)
                {
                    min_distance2 = point_distance;
                    closest_index2 = j;
                }
                // if in the higher scan line
                else if (int(points_less_flat_last.points[j].intensity) > scan_id && point_distance < min_distance3)
                {
                    min_distance3 = point_distance;
                    closest_index3 = j;
                }
            }

            // search in the direction of decreasing scan line
            for (int j = closest_index - 1; j >= 0; --j)
            {
                // if not in nearby scans, end the loop
                if (int(points_less_flat_last.points[j].intensity) < (scan_id - nearby_scan))
                    break;

                double point_distance = (points_less_flat_last.points[j].x - point.x) *
                                            (points_less_flat_last.points[j].x - point.x) +
                                        (points_less_flat_last.points[j].y - point.y) *
                                            (points_less_flat_last.points[j].y - point.y) +
                                        (points_less_flat_last.points[j].z - point.z) *
                                            (points_less_flat_last.points[j].z - point.z);

                // if in the same or higher scan line
                if (int(points_less_flat_last.points[j].intensity) >= scan_id && point_distance < min_distance2)
                {
                    min_distance2 = point_distance;
                    closest_index2 = j;
                }
                else if (int(points_less_flat_last.points[j].intensity) < scan_id && point_distance < min_distance3)
                {
                    // find nearer point
                    min_distance3 = point_distance;
                    closest_index3 = j;
                }
            }

            if (closest_index2 >= 0 && closest_index3 >= 0)
            {

                Vector3d curr_point(points_flat.points[i].x,
                                    points_flat.points[i].y,
                                    points_flat.points[i].z);
                Vector3d last_point_a(points_less_flat_last.points[closest_index].x,
                                      points_less_flat_last.points[closest_index].y,
                                      points_less_flat_last.points[closest_index].z);
                Vector3d last_point_b(points_less_flat_last.points[closest_index2].x,
                                      points_less_flat_last.points[closest_index2].y,
                                      points_less_flat_last.points[closest_index2].z);
                Vector3d last_point_c(points_less_flat_last.points[closest_index3].x,
                                      points_less_flat_last.points[closest_index3].y,
                                      points_less_flat_last.points[closest_index3].z);
                Vector3d last_kf_t(para_last_kf[4], para_last_kf[5], para_last_kf[6]);
                Vector3d kf_t(para_kf[4], para_kf[5], para_kf[6]);

                ceres::CostFunction *cost_function;
                cost_function = LidarPlaneError::Create(curr_point, last_point_a, last_point_b, last_point_c, lidar_);
                problem.AddResidualBlock(cost_function, loss_function, para_last_kf, para_kf);
            }
        }
    }
}
} // namespace lvio_fusion
