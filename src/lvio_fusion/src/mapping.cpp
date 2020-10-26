#include "lvio_fusion/lidar/mapping.h"
#include "lvio_fusion/utility.h"
#include <lvio_fusion/ceres/base.hpp>
#include <pcl/filters/voxel_grid.h>

namespace lvio_fusion
{
Mapping::Mapping()
{
    thread_ = std::thread(std::bind(&Mapping::MappingLoop, this));
}

void Mapping::Pause()
{
    if (status == MappingStatus::RUNNING)
    {
        std::unique_lock<std::mutex> lock(pausing_mutex_);
        status = MappingStatus::TO_PAUSE;
        pausing_.wait(lock);
    }
}

void Mapping::Continue()
{
    if (status == MappingStatus::PAUSING)
    {
        status = MappingStatus::RUNNING;
        running_.notify_one();
    }
}

inline void Mapping::AddToWorld(const PointICloud &in, Frame::Ptr frame, PointRGBCloud &out)
{
    double lti[7], lei[7], ltiei[7], cet[7];
    ceres::SE3Inverse(frame->pose.data(), lti);
    ceres::SE3Inverse(lidar_->extrinsic.data(), lei);
    ceres::SE3Product(lti, lei, ltiei);
    ceres::SE3Product(camera_->extrinsic.data(), frame->pose.data(), cet);

    for (int i = 0; i < in.size(); i++)
    {
        if (in[i].x <= 0)
            continue;
        //NOTE: Sophus is too slow
        // auto p_w = lidar_->Sensor2World(Vector3d(in[i].x, in[i].y, in[i].z), frame->pose);
        // auto pixel = camera_->World2Pixel(p_w, frame->pose);
        double pin[3] = {in[i].x, in[i].y, in[i].z}, pw[3], pc[3];
        ceres::SE3TransformPoint(ltiei, pin, pw);
        ceres::SE3TransformPoint(cet, pw, pc);
        auto pixel = camera_->Sensor2Pixel(Vector3d(pc[0], pc[1], pc[2]));

        auto &image = frame->image_left;
        if (0 < pixel.x() && pixel.x() < image.cols && 0 < pixel.y() && pixel.y() < image.rows)
        {
            unsigned char gray = image.at<uchar>((int)pixel.y(), (int)pixel.x());
            PointRGB point_world;
            point_world.x = pw[0];
            point_world.y = pw[1];
            point_world.z = pw[2];
            point_world.r = gray;
            point_world.g = gray;
            point_world.b = gray;
            out.push_back(point_world);
        }
    }
}

void Mapping::MappingLoop()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(running_mutex_);
        if (status == MappingStatus::TO_PAUSE)
        {
            status = MappingStatus::PAUSING;
            pausing_.notify_one();
            running_.wait(lock);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto t1 = std::chrono::steady_clock::now();
        {
            Frames active_kfs = map_->GetKeyFrames(head_, map_->local_map_head);
            if (active_kfs.empty())
            {
                continue;
            }
            SE3d old_pose = (--active_kfs.end())->second->pose;
            Optimize(active_kfs);
            SE3d new_pose = (--active_kfs.end())->second->pose;

            // forward propogate
            {
                std::unique_lock<std::mutex> lock1(backend_->mutex);
                std::unique_lock<std::mutex> lock2(frontend_->mutex);

                Frame::Ptr last_frame = frontend_->last_frame;
                Frames forward_kfs = map_->GetKeyFrames(map_->local_map_head);
                if (forward_kfs.find(last_frame->time) == forward_kfs.end())
                {
                    forward_kfs[last_frame->time] = last_frame;
                }
                SE3d transform = old_pose.inverse() * new_pose;
                for (auto pair_kf : forward_kfs)
                {
                    pair_kf.second->pose = pair_kf.second->pose * transform;
                    // TODO: Repropagate
                    // if(pair_kf.second->preintegration)
                    // {
                    //     pair_kf.second->preintegration->Repropagate();
                    // }
                }
                frontend_->UpdateCache();
            }
        }
        auto t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "Mapping cost time: " << time_used.count() << " seconds.";
    }
}

void Mapping::BuildProblem(Frames &active_kfs, ceres::Problem &problem)
{
    double start_time = active_kfs.begin()->first;
    Frames base_kfs = map_->GetKeyFrames(0, start_time, 2);

    ceres::LossFunction *lidar_loss_function = new ceres::HuberLoss(0.1);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    Frame::Ptr last_frame;
    Frame::Ptr last_frame2;
    unsigned int start_id = active_kfs.begin()->second->id;
    for (auto pair_kf : base_kfs)
    {
        double *para_kf_base = pair_kf.second->pose.data();
        problem.AddParameterBlock(para_kf_base, SE3d::num_parameters, local_parameterization);
        problem.SetParameterBlockConstant(para_kf_base);
        if (pair_kf.second->id + 1 == start_id)
        {
            last_frame = pair_kf.second;
        }
        else if (pair_kf.second->id + 2 == start_id)
        {
            last_frame2 = pair_kf.second;
        }
    }

    Frame::Ptr current_frame;
    for (auto pair_kf : active_kfs)
    {
        current_frame = pair_kf.second;
        double *para_kf = current_frame->pose.data();
        problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
        if (current_frame->feature_lidar)
        {
            if (last_frame && last_frame->feature_lidar)
            {
                if (last_frame->id + 1 != current_frame->id)
                {
                    // last_frame is in a inner submap
                    auto real_last_frame = (--map_->GetAllKeyFrames().find(current_frame->time))->second;
                    scan_registration_->Associate(current_frame, real_last_frame, problem, lidar_loss_function, last_frame);
                    continue;
                }
                else
                {
                    scan_registration_->Associate(current_frame, last_frame, problem, lidar_loss_function);
                }
            }
            if (last_frame2 && last_frame2->feature_lidar)
            {
                if (last_frame2->id + 2 != current_frame->id)
                {
                    // last_frame2 is in a inner submap
                    auto real_last_frame2 = (----map_->GetAllKeyFrames().find(current_frame->time))->second;
                    scan_registration_->Associate(current_frame, real_last_frame2, problem, lidar_loss_function, last_frame2);
                }
                else
                {
                    scan_registration_->Associate(current_frame, last_frame2, problem, lidar_loss_function);
                }
            }
        }
        last_frame2 = last_frame;
        last_frame = current_frame;
    }

    // loop constraint
    for (auto pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        if (frame->loop_constraint)
        {
            double *para_kf = frame->pose.data();
            double *para_old_kf = frame->loop_constraint->frame_old->pose.data();
            problem.SetParameterBlockConstant(para_kf);
            problem.SetParameterBlockConstant(para_old_kf);
        }
    }

    // initial point
    if (active_kfs.begin()->second->id == 1)
    {
        auto &pose = active_kfs.begin()->second->pose;
        problem.SetParameterBlockConstant(pose.data());
    }
}

void Mapping::Optimize(Frames &active_kfs)
{
    ceres::Problem problem;
    BuildProblem(active_kfs, problem);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.function_tolerance = 1e-9;
    options.max_num_iterations = 3;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    LOG(INFO) << summary.FullReport();

    // Build global map
    BuildGlobalMap(active_kfs);
    head_ = (--active_kfs.end())->first;
}

void Mapping::BuildGlobalMap(Frames &active_kfs)
{
    for (auto pair_kf : active_kfs)
    {
        Frame::Ptr frame = pair_kf.second;
        if (frame->feature_lidar)
        {
            PointRGBCloud pointcloud;
            AddToWorld(frame->feature_lidar->points_less_sharp, frame, pointcloud);
            AddToWorld(frame->feature_lidar->points_less_flat, frame, pointcloud);
            pointclouds_[frame->time] = pointcloud;
        }
    }
}

PointRGBCloud Mapping::GetGlobalMap()
{
    static pcl::VoxelGrid<PointRGB> downSizeFilter;
    PointRGBCloud global_map;
    for (auto pair_pc : pointclouds_)
    {
        auto &pointcloud = pair_pc.second;
        global_map.insert(global_map.end(), pointcloud.begin(), pointcloud.end());
    }
    if (global_map.size() > 0)
    {
        PointRGBCloud::Ptr mapDS(new PointRGBCloud());
        pcl::copyPointCloud(global_map, *mapDS);
        downSizeFilter.setInputCloud(mapDS);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(global_map);
    }
    return global_map;
}

} // namespace lvio_fusion
