#include "lvio_fusion/lidar/mapping.h"
#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/lidar/lidar.h"
#include "lvio_fusion/loop/pose_graph.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"

#include <pcl/filters/voxel_grid.h>

namespace lvio_fusion
{

inline void Mapping::Color(const PointICloud &points_ground, const PointICloud &points_surf, Frame::Ptr frame, PointRGBCloud &out)
{
    for (int i = 0; i < points_ground.size(); i++)
    {
        PointRGB point_color;
        point_color.x = points_ground[i].x;
        point_color.y = points_ground[i].y;
        point_color.z = points_ground[i].z;
        point_color.r = 255;
        point_color.g = 0;
        point_color.b = 255;
        out.push_back(point_color);
    }
    for (int i = 0; i < points_surf.size(); i++)
    {
        PointRGB point_color;
        point_color.x = points_surf[i].x;
        point_color.y = points_surf[i].y;
        point_color.z = points_surf[i].z;
        point_color.r = 0;
        point_color.g = 255;
        point_color.b = 0;
        out.push_back(point_color);
    }
}

void Mapping::BuildOldMapFrame(Frames old_frames, Frame::Ptr map_frame)
{
    PointICloud points_surf_merged;
    PointICloud points_ground_merged;
    for (auto &pair_kf : old_frames)
    {
        points_surf_merged += pointclouds_surf[pair_kf.first];
        points_ground_merged += pointclouds_ground[pair_kf.first];
    }

    association_->SegmentGround(points_ground_merged);

    map_frame->id = old_frames.begin()->second->id;
    map_frame->time = old_frames.begin()->second->time;
    map_frame->pose = old_frames.begin()->second->pose;
    map_frame->feature_lidar = lidar::Feature::Create();
    map_frame->feature_lidar->points_surf = points_surf_merged;
    map_frame->feature_lidar->points_ground = points_ground_merged;
}

void Mapping::BuildMapFrame(Frame::Ptr frame, Frame::Ptr map_frame)
{
    double start_time = frame->time;
    static int num_last_frames = 3;
    Frames last_frames = Map::Instance().GetKeyFrames(0, start_time, num_last_frames);
    if (last_frames.empty())
        return;
    PointICloud points_surf_merged;
    PointICloud points_ground_merged;
    for (auto &pair_kf : last_frames)
    {
        points_surf_merged += pointclouds_surf[pair_kf.first];
        points_ground_merged += pointclouds_ground[pair_kf.first];
    }

    association_->SegmentGround(points_ground_merged);

    map_frame->id = (--last_frames.end())->second->id;
    map_frame->time = (--last_frames.end())->second->time;
    map_frame->pose = (--last_frames.end())->second->pose;
    map_frame->feature_lidar = lidar::Feature::Create();
    map_frame->feature_lidar->points_surf = points_surf_merged;
    map_frame->feature_lidar->points_ground = points_ground_merged;
}

void Mapping::Optimize(Frames &active_kfs)
{
    // NOTE: some place is good, don't need optimize too much.
    for (auto &pair_kf : active_kfs)
    {
        auto t1 = std::chrono::steady_clock::now();
        SE3d old_pose = pair_kf.second->pose;
        {
            auto map_frame = Frame::Ptr(new Frame());
            BuildMapFrame(pair_kf.second, map_frame);
            if (map_frame->feature_lidar && pair_kf.second->feature_lidar)
            {
                double rpyxyz[6];
                se32rpyxyz(map_frame->pose.inverse() * pair_kf.second->pose, rpyxyz); // relative_i_j
                if (!map_frame->feature_lidar->points_ground.empty())
                {
                    adapt::Problem problem;
                    association_->ScanToMapWithGround(pair_kf.second, map_frame, rpyxyz, problem);
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 2;
                    options.num_threads = num_threads;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    pair_kf.second->pose = map_frame->pose * rpyxyz2se3(rpyxyz);
                }
                if (!map_frame->feature_lidar->points_surf.empty())
                {
                    adapt::Problem problem;
                    association_->ScanToMapWithSegmented(pair_kf.second, map_frame, rpyxyz, problem);
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 2;
                    options.num_threads = num_threads;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    pair_kf.second->pose = map_frame->pose * rpyxyz2se3(rpyxyz);
                }
            }
        }
        SE3d new_pose = pair_kf.second->pose;
        SE3d transform = new_pose * old_pose.inverse();
        PoseGraph::Instance().Propagate(transform, Frames(++active_kfs.find(pair_kf.first), active_kfs.end()));

        ToWorld(pair_kf.second);

        auto t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "Mapping cost time: " << time_used.count() << " seconds.";
    }
}

void Mapping::MergeScan(const PointICloud &in, SE3d Twc, PointICloud &out)
{
    Sophus::SE3f tf_se3 = Twc.cast<float>();
    float *tf = tf_se3.data();
    for (auto &point_in : in)
    {
        PointI point_out;
        ceres::SE3TransformPoint(tf, point_in.data, point_out.data);
        point_out.intensity = point_in.intensity;
        out.push_back(point_out);
    }
}

void Mapping::ToWorld(Frame::Ptr frame)
{
    PointICloud pointcloud_surf;
    PointICloud pointcloud_ground;
    PointRGBCloud pointcloud_color;
    if (frame->feature_lidar)
    {
        MergeScan(frame->feature_lidar->points_surf, frame->pose, pointcloud_surf);
        MergeScan(frame->feature_lidar->points_ground, frame->pose, pointcloud_ground);
        Color(pointcloud_ground, pointcloud_surf, frame, pointcloud_color);
    }
    pointclouds_surf[frame->time] = pointcloud_surf;
    pointclouds_ground[frame->time] = pointcloud_ground;
    pointclouds_color[frame->time] = pointcloud_color;
}

PointRGBCloud Mapping::GetGlobalMap()
{
    PointRGBCloud global_map;
    for (auto &pair_pc : pointclouds_color)
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
        voxel_filter.setLeafSize(Lidar::Get()->resolution * 2, Lidar::Get()->resolution * 2, Lidar::Get()->resolution * 2);
        voxel_filter.filter(global_map);
    }
    return global_map;
}

int Mapping::Relocate(Frame::Ptr last_frame, Frame::Ptr current_frame, SE3d &relative_o_c)
{
    // init relative pose
    Frame::Ptr clone_frame = Frame::Ptr(new Frame());
    *clone_frame = *current_frame;
    clone_frame->pose = last_frame->pose * clone_frame->loop_closure->relative_o_c;

    // build two pointclouds
    Frame::Ptr old_frame_prev = Map::Instance().GetKeyFrames(0, last_frame->time, 1).begin()->second;
    Frame::Ptr old_frame_subs = Map::Instance().GetKeyFrames(last_frame->time, 0, 1).begin()->second;
    Frames old_frames = {{last_frame->time, last_frame}, {old_frame_prev->time, old_frame_prev}, {old_frame_subs->time, old_frame_subs}};
    Frame::Ptr map_frame = Frame::Ptr(new Frame());
    BuildOldMapFrame(old_frames, map_frame);

    // optimize
    double score_ground, score_surf;
    for (int i = 0; i < 4; i++)
    {
        double rpyxyz[6];
        se32rpyxyz(map_frame->pose.inverse() * clone_frame->pose, rpyxyz); // relative_i_j
        if (!map_frame->feature_lidar->points_ground.empty())
        {
            adapt::Problem problem;
            association_->ScanToMapWithGround(clone_frame, map_frame, rpyxyz, problem);
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 1;
            options.num_threads = num_threads;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            clone_frame->pose = map_frame->pose * rpyxyz2se3(rpyxyz) ;
            score_ground = std::min((double)summary.num_residual_blocks_reduced / 10, 20.0);
            score_ground -= 2 * summary.final_cost / summary.num_residual_blocks_reduced;
        }
        if (!map_frame->feature_lidar->points_surf.empty())
        {
            adapt::Problem problem;
            association_->ScanToMapWithSegmented(clone_frame, map_frame, rpyxyz, problem);
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 1;
            options.num_threads = num_threads;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            clone_frame->pose = map_frame->pose * rpyxyz2se3(rpyxyz) ;
            score_surf = std::min((double)summary.num_residual_blocks_reduced / 10, 30.0);
            score_surf -= 2 * summary.final_cost / summary.num_residual_blocks_reduced;
        }
    }

    relative_o_c = last_frame->pose.inverse() * clone_frame->pose;
    return score_ground + score_surf;
}

} // namespace lvio_fusion
