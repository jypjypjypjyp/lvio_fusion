#include "lvio_fusion/lidar/mapping.h"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/utility.h"

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

namespace lvio_fusion
{

inline void Mapping::AddToWorld(const PointICloud &in, Frame::Ptr frame, PointRGBCloud &out)
{
    double lti[7], lei[7], ltiei[7], cet[7];
    ceres::SE3Inverse(frame->pose.data(), lti);
    ceres::SE3Inverse(lidar_->extrinsic.data(), lei);
    ceres::SE3Product(lti, lei, ltiei);
    ceres::SE3Product(camera_->extrinsic.data(), frame->pose.data(), cet);

    for (int i = 0; i < in.size(); i++)
    {
        //NOTE: colorful pointcloud
        // if (in[i].x <= 0)
        //     continue;
        // if (in[i].y >= 0)
        // continue;
        //NOTE: Sophus is too slow
        // auto p_w = lidar_->Sensor2World(Vector3d(in[i].x, in[i].y, in[i].z), frame->pose);
        // auto pixel = camera_->World2Pixel(p_w, frame->pose);
        double pin[3] = {in[i].x, in[i].y, in[i].z}, pw[3], pc[3];
        ceres::SE3TransformPoint(ltiei, pin, pw);
        ceres::SE3TransformPoint(cet, pw, pc);
        // auto pixel = camera_->Sensor2Pixel(Vector3d(pc[0], pc[1], pc[2]));

        // auto &image = frame->image_left;
        // if (0 < pixel.x() && pixel.x() < image.cols && 0 < pixel.y() && pixel.y() < image.rows)
        // {
        //     unsigned char gray = image.at<uchar>((int)pixel.y(), (int)pixel.x());
        //     PointRGB point_world;
        //     point_world.x = pw[0];
        //     point_world.y = pw[1];
        //     point_world.z = pw[2];
        //     point_world.r = gray;
        //     point_world.g = gray;
        //     point_world.b = gray;
        //     out.push_back(point_world);
        // }

        PointRGB point_world;
        point_world.x = pw[0];
        point_world.y = pw[1];
        point_world.z = pw[2];
        point_world.r = 0;
        point_world.g = 0;
        point_world.b = 0;
        out.push_back(point_world);
    }
}

void Mapping::BuildMapFrame(Frame::Ptr frame, Frame::Ptr map_frame)
{
    double start_time = frame->time;
    static int num_last_frames = 3;
    Frames last_frames = map_->GetKeyFrames(0, start_time, num_last_frames);
    if (last_frames.empty())
        return;
    map_frame->time = (--last_frames.end())->second->time;
    map_frame->pose = (--last_frames.end())->second->pose;
    PointICloud points_less_sharp_merged;
    PointICloud points_less_flat_merged;
    for (auto pair_kf : last_frames)
    {
        if (pair_kf.second->feature_lidar)
        {
            if (pair_kf.first == map_frame->time)
            {
                points_less_sharp_merged += pair_kf.second->feature_lidar->points_less_sharp;
                points_less_flat_merged += pair_kf.second->feature_lidar->points_less_flat;
            }
            else
            {
                association_->MergeScan(pair_kf.second->feature_lidar->points_less_sharp, pair_kf.second->pose, map_frame->pose, points_less_sharp_merged);
                association_->MergeScan(pair_kf.second->feature_lidar->points_less_flat, pair_kf.second->pose, map_frame->pose, points_less_flat_merged);
            }
        }
    }
    {
        PointICloud::Ptr temp(new PointICloud());
        pcl::RadiusOutlierRemoval<PointI> radius_filter;
        pcl::copyPointCloud(points_less_sharp_merged, *temp);
        radius_filter.setInputCloud(temp);
        radius_filter.setRadiusSearch(2 * lidar_->resolution);
        radius_filter.setMinNeighborsInRadius(8);
        radius_filter.filter(points_less_sharp_merged);
        temp->clear();
        pcl::VoxelGrid<PointI> voxel_filter;
        pcl::copyPointCloud(points_less_sharp_merged, *temp);
        voxel_filter.setInputCloud(temp);
        voxel_filter.setLeafSize(lidar_->resolution, lidar_->resolution, lidar_->resolution);
        voxel_filter.filter(points_less_sharp_merged);
    }
    {
        PointICloud::Ptr temp(new PointICloud());
        pcl::RadiusOutlierRemoval<PointI> radius_filter;
        pcl::copyPointCloud(points_less_sharp_merged, *temp);
        radius_filter.setInputCloud(temp);
        radius_filter.setRadiusSearch(2 * lidar_->resolution);
        radius_filter.setMinNeighborsInRadius(4);
        radius_filter.filter(points_less_sharp_merged);
        temp->clear();
        pcl::VoxelGrid<PointI> voxel_filter;
        pcl::copyPointCloud(points_less_sharp_merged, *temp);
        voxel_filter.setInputCloud(temp);
        voxel_filter.setLeafSize(lidar_->resolution, lidar_->resolution, lidar_->resolution);
        voxel_filter.filter(points_less_sharp_merged);
    }

    map_frame->feature_lidar = lidar::Feature::Create(PointICloud(), points_less_sharp_merged, PointICloud(), points_less_flat_merged, PointICloud());
}

void Mapping::BuildProblem(Frame::Ptr frame, Frame::Ptr map_frame, double *para, ceres::Problem &problem, Mapping::ProblemType type)
{
    if (type == Mapping::ProblemType::Ground)
    {
        problem.AddParameterBlock(para + 2, 1);
        problem.AddParameterBlock(para + 1, 1);
        problem.AddParameterBlock(para + 5, 1);
        association_->ScanToMapWithGround(frame, map_frame, para, problem);
    }
    else if (type == Mapping::ProblemType::Segmented)
    {
        problem.AddParameterBlock(para + 0, 1);
        problem.AddParameterBlock(para + 3, 1);
        problem.AddParameterBlock(para + 4, 1);
        association_->AssociateWithSegmented(frame, map_frame, para, problem);
    }
}

void Mapping::Optimize(Frames &active_kfs)
{
    // NOTE: some place is good, don't need optimize too much.
    for (auto pair_kf : active_kfs)
    {
        Frame::Ptr map_frame = Frame::Ptr(new Frame());
        BuildMapFrame(pair_kf.second, map_frame);
        if (!map_frame->feature_lidar || !pair_kf.second->feature_lidar || map_frame->feature_lidar->points_less_flat.empty() || map_frame->feature_lidar->points_less_sharp.empty())
            continue;

        double rpyxyz[6];
        se32rpyxyz(map_frame->pose * pair_kf.second->pose.inverse(), rpyxyz); // relative_i_j
        for (int i = 0; i < 4; i++)
        {
            {
                ceres::Problem problem;
                BuildProblem(pair_kf.second, map_frame, rpyxyz, problem, Mapping::ProblemType::Ground);
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_SCHUR;
                options.function_tolerance = DBL_MIN;
                options.gradient_tolerance = DBL_MIN;
                options.max_num_iterations = 1;
                options.num_threads = 4;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                pair_kf.second->pose = rpyxyz2se3(rpyxyz).inverse() * map_frame->pose;
            }
            {
                ceres::Problem problem;
                BuildProblem(pair_kf.second, map_frame, rpyxyz, problem, Mapping::ProblemType::Segmented);
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_SCHUR;
                options.function_tolerance = DBL_MIN;
                options.gradient_tolerance = DBL_MIN;
                options.max_num_iterations = 1;
                options.num_threads = 4;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                pair_kf.second->pose = rpyxyz2se3(rpyxyz).inverse() * map_frame->pose;
                if (summary.final_cost / summary.initial_cost > 0.9)
                    break;
            }
        }
    }

    // Build global map
    BuildGlobalMap(active_kfs);
}

void Mapping::BuildGlobalMap(Frames &active_kfs)
{
    for (auto pair_kf : active_kfs)
    {
        Frame::Ptr frame = pair_kf.second;
        if (frame->feature_lidar)
        {
            PointRGBCloud pointcloud;
            AddToWorld(frame->feature_lidar->points_less_flat, frame, pointcloud);
            pointclouds_[frame->time] = pointcloud;
        }
    }
}

PointRGBCloud Mapping::GetGlobalMap()
{

    PointRGBCloud global_map;
    for (auto pair_pc : pointclouds_)
    {
        auto &pointcloud = pair_pc.second;
        global_map.insert(global_map.end(), pointcloud.begin(), pointcloud.end());
    }
    if (global_map.size() > 0)
    {
        PointRGBCloud::Ptr temp(new PointRGBCloud());

        pcl::RadiusOutlierRemoval<PointRGB> radius_filter;
        pcl::copyPointCloud(global_map, *temp);
        radius_filter.setInputCloud(temp);
        radius_filter.setRadiusSearch(2 * lidar_->resolution);
        radius_filter.setMinNeighborsInRadius(4);
        radius_filter.filter(global_map);
        temp->clear();

        pcl::VoxelGrid<PointRGB> voxel_filter;
        pcl::copyPointCloud(global_map, *temp);
        voxel_filter.setInputCloud(temp);
        voxel_filter.setLeafSize(lidar_->resolution * 2, lidar_->resolution * 2, lidar_->resolution * 2);
        voxel_filter.filter(global_map);
    }
    return global_map;
}

} // namespace lvio_fusion
