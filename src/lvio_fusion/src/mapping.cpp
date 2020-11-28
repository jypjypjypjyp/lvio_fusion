#include "lvio_fusion/lidar/mapping.h"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/utility.h"

#include <pcl/filters/voxel_grid.h>

namespace lvio_fusion
{

inline void Mapping::Color(const PointICloud &in, Frame::Ptr frame, PointRGBCloud &out)
{
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

        PointRGB point_color;
        point_color.x = in[i].x;
        point_color.y = in[i].y;
        point_color.z = in[i].z;
        point_color.r = 0;
        point_color.g = 0;
        point_color.b = 0;
        out.push_back(point_color);
    }
}

void Mapping::BuildOldMapFrame(Frames old_frames, Frame::Ptr map_frame)
{
    PointICloud points_full_merged;
    for (auto pair_kf : old_frames)
    {
        points_full_merged += pointclouds_full[pair_kf.first];
    }
    PointICloud::Ptr temp(new PointICloud());
    pcl::VoxelGrid<PointI> voxel_filter;
    pcl::copyPointCloud(points_full_merged, *temp);
    voxel_filter.setInputCloud(temp);
    voxel_filter.setLeafSize(lidar_->resolution, lidar_->resolution, lidar_->resolution);
    voxel_filter.filter(points_full_merged);

    map_frame->time = old_frames.begin()->second->time;
    map_frame->pose = old_frames.begin()->second->pose;
    map_frame->feature_lidar = lidar::Feature::Create();
    map_frame->feature_lidar->points_full = points_full_merged;
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
    PointICloud points_full_merged;
    for (auto pair_kf : last_frames)
    {
        points_full_merged += pointclouds_full[pair_kf.first];
    }
    PointICloud::Ptr temp(new PointICloud());
    pcl::VoxelGrid<PointI> voxel_filter;
    pcl::copyPointCloud(points_full_merged, *temp);
    voxel_filter.setInputCloud(temp);
    voxel_filter.setLeafSize(lidar_->resolution, lidar_->resolution, lidar_->resolution);
    voxel_filter.filter(points_full_merged);

    map_frame->feature_lidar = lidar::Feature::Create();
    map_frame->feature_lidar->points_full = points_full_merged;
}

void Mapping::Optimize(Frames &active_kfs)
{
    // NOTE: some place is good, don't need optimize too much.
    for (auto pair_kf : active_kfs)
    {
        auto t1 = std::chrono::steady_clock::now();
        Frame::Ptr map_frame = Frame::Ptr(new Frame());
        BuildMapFrame(pair_kf.second, map_frame);
        if (map_frame->feature_lidar && pair_kf.second->feature_lidar && !map_frame->feature_lidar->points_full.empty())
        {
            double rpyxyz[6];
            se32rpyxyz(map_frame->pose * pair_kf.second->pose.inverse(), rpyxyz); // relative_i_j
            {
                ceres::Problem problem;
                association_->ScanToMapWithGround(pair_kf.second, map_frame, rpyxyz, problem);
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 4;
                options.num_threads = 4;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                pair_kf.second->pose = rpyxyz2se3(rpyxyz).inverse() * map_frame->pose;
            }
            {
                ceres::Problem problem;
                association_->ScanToMapWithSegmented(pair_kf.second, map_frame, rpyxyz, problem);
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 1;
                options.num_threads = 4;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                pair_kf.second->pose = rpyxyz2se3(rpyxyz).inverse() * map_frame->pose;
            }
        }
        AddToWorld(pair_kf.second);

        auto t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "Mapping cost time: " << time_used.count() << " seconds.";
    }
}

void Mapping::MergeScan(const PointICloud &in, SE3d from_pose, PointICloud &out)
{
    Sophus::SE3f tf_se3 = lidar_->TransformMatrix(from_pose).cast<float>();
    float *tf = tf_se3.data();
    for (auto point_in : in)
    {
        PointI point_out;
        ceres::SE3TransformPoint(tf, point_in.data, point_out.data);
        point_out.intensity = point_in.intensity;
        out.push_back(point_out);
    }
}

void Mapping::AddToWorld(Frame::Ptr frame)
{
    PointICloud pointcloud_full;
    PointICloud pointcloud_temp;
    PointRGBCloud pointcloud_color;
    if (frame->feature_lidar)
    {
        MergeScan(frame->feature_lidar->points_full, frame->pose, pointcloud_full);
        MergeScan(frame->feature_lidar->points_surf + frame->feature_lidar->points_ground, frame->pose, pointcloud_temp);
        Color(pointcloud_temp, frame, pointcloud_color);
    }
    pointclouds_full[frame->time] = pointcloud_full;
    pointclouds_color[frame->time] = pointcloud_color;
}

PointRGBCloud Mapping::GetGlobalMap()
{
    PointRGBCloud global_map;
    for (auto pair_pc : pointclouds_color)
    {
        auto &pointcloud = pair_pc.second;
        global_map.insert(global_map.end(), pointcloud.begin(), pointcloud.end());
    }
    if (global_map.size() > 0)
    {
        PointRGBCloud::Ptr temp(new PointRGBCloud());
        pcl::VoxelGrid<PointRGB> voxel_filter;
        pcl::copyPointCloud(global_map, *temp);
        voxel_filter.setInputCloud(temp);
        voxel_filter.setLeafSize(lidar_->resolution * 2, lidar_->resolution * 2, lidar_->resolution * 2);
        voxel_filter.filter(global_map);
    }
    return global_map;
}

} // namespace lvio_fusion
