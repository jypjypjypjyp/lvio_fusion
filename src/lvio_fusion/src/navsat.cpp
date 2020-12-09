#include "lvio_fusion/navsat/navsat.h"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/map.h"

#include <ceres/ceres.h>

namespace lvio_fusion
{

void NavsatMap::AddPoint(double time, double x, double y, double z)
{
    raw[time] = Vector3d(x, y, z);

    if (!initialized && UpdateLevel(time, raw[time]))
    {
        Initialize();
    }

    if (initialized)
    {
        static double head = 0;
        Frames new_kfs = Map::Instance().GetKeyFrames(head);
        for (auto pair_kf : new_kfs)
        {
            auto this_iter = raw.lower_bound(pair_kf.first);
            auto last_iter = --this_iter;
            if (this_iter == raw.begin() || std::fabs(this_iter->first - pair_kf.first) > 1e-1)
                continue;

            pair_kf.second->feature_navsat = navsat::Feature::Ptr(new navsat::Feature(this_iter->first, last_iter->first, A_.first, B_.first, C_.first, this));
            head = pair_kf.first + epsilon;
        }
    }
}

Vector3d NavsatMap::GetPoint(double time)
{
    return extrinsic * raw[time];
}

bool NavsatMap::UpdateLevel(double time, Vector3d position)
{
    auto iter = raw.lower_bound(0);

    double max_height = 0, B_time = 0;
    while ((++iter)->first < time)
    {
        auto AC = position - C_.second;
        auto AB = iter->second - C_.second;
        double height = AC.cross(AB).norm() / AC.norm();
        if (height > max_height)
        {
            max_height = height;
            B_time = iter->first;
        }
    }
    if (max_height > 20)
    {
        A_ = C_;
        B_ = std::make_pair(B_time, raw[B_time]);
        C_ = std::make_pair(time, position);
        return true;
    }
    return false;
}

void NavsatMap::Initialize()
{
    Frames keyframes = Map::Instance().GetAllKeyFrames();

    ceres::Problem problem;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    problem.AddParameterBlock(extrinsic.data(), SE3d::num_parameters, local_parameterization);

    for (auto pair_kf : keyframes)
    {
        auto position_kf = pair_kf.second->pose.translation();
        auto pair_np = raw.lower_bound(pair_kf.first);
        if (std::fabs(pair_np->first - pair_kf.first) < 1e-1)
        {
            ceres::CostFunction *cost_function = NavsatInitError::Create(position_kf, pair_np->second);
            problem.AddResidualBlock(cost_function, NULL, extrinsic.data());
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    initialized = true;
}

void NavsatMap::Optimize(Frames active_kfs)
{
    // // find Atlas

    // // relocate Turning point

    // // relocate straight route

    // // build the pose graph and submaps
    // Frames active_kfs = Map::Instance().GetKeyFrames(old_time, end_time);
    // Frames new_submap_kfs = Map::Instance().GetKeyFrames(start_time, end_time);
    // Frames all_kfs = active_kfs;
    // // std::map<double, SE3d> inner_submap_old_frames = atlas_.GetActiveSubMaps(active_kfs, old_time, start_time);
    // // atlas_.AddSubMap(old_time, start_time, end_time);
    // // adapt::Problem problem;
    // // BuildProblem(active_kfs, problem);

    // // update new submap frams
    // SE3d old_pose = (--new_submap_kfs.end())->second->pose;
    // {
    //     // build new submap pose graph
    //     adapt::Problem problem;
    //     BuildProblem(new_submap_kfs, problem);

    //     // relocate new submaps
    //     std::map<double, double> score_table;
    //     for (auto pair_kf : new_submap_kfs)
    //     {
    //         Relocate(pair_kf.second, pair_kf.second->loop_constraint->frame_old);
    //         score_table[-pair_kf.second->loop_constraint->score] = pair_kf.first;
    //     }
    //     int max_num_relocated = 1;
    //     for (auto pair : score_table)
    //     {
    //         if (max_num_relocated-- == 0)
    //             break;
    //         auto frame = new_submap_kfs[pair.second];
    //         frame->loop_constraint->relocated = true;
    //         frame->pose = frame->loop_constraint->relative_o_c * frame->loop_constraint->frame_old->pose;
    //     }

    //     BuildProblemWithLoop(new_submap_kfs, problem);
    //     ceres::Solver::Options options;
    //     options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    //     options.num_threads = 1;
    //     ceres::Solver::Summary summary;
    //     ceres::Solve(options, &problem, &summary);

    //     for (auto pair_kf : new_submap_kfs)
    //     {
    //         Relocate(pair_kf.second, pair_kf.second->loop_constraint->frame_old);
    //         pair_kf.second->loop_constraint->relocated = true;
    //         pair_kf.second->pose = pair_kf.second->loop_constraint->relative_o_c * pair_kf.second->loop_constraint->frame_old->pose;
    //     }
    // }
    // SE3d new_pose = (--new_submap_kfs.end())->second->pose;

    // // forward propogate
    // {
    //     std::unique_lock<std::mutex> lock1(backend_->mutex);
    //     std::unique_lock<std::mutex> lock2(frontend_->mutex);

    //     Frame::Ptr last_frame = frontend_->last_frame;
    //     Frames forward_kfs = Map::Instance().GetKeyFrames(end_time + epsilon);
    //     if (forward_kfs.find(last_frame->time) == forward_kfs.end())
    //     {
    //         forward_kfs[last_frame->time] = last_frame;
    //     }
    //     SE3d transform = old_pose.inverse() * new_pose;
    //     for (auto pair_kf : forward_kfs)
    //     {
    //         pair_kf.second->pose = pair_kf.second->pose * transform;
    //         // TODO: Repropagate

    //         if (lidar_)
    //         {
    //             mapping_->AddToWorld(pair_kf.second);
    //         }
    //     }
    //     frontend_->UpdateCache();
    // }

    // // BuildProblemWithLoop(active_kfs, problem);
    // // ceres::Solver::Options options;
    // // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // // options.num_threads = 1;
    // // ceres::Solver::Summary summary;
    // // ceres::Solve(options, &problem, &summary);

    // // // update pose of inner submaps
    // // for (auto pair_of : inner_submap_old_frames)
    // // {
    // //     auto old_frame = active_kfs[pair_of.first];
    // //     // T2_new = T2 * T1.inverse() * T1_new
    // //     SE3d transform = pair_of.second.inverse() * old_frame->pose;
    // //     for (auto iter = ++all_kfs.find(pair_of.first); active_kfs.find(iter->first) == active_kfs.end(); iter++)
    // //     {
    // //         auto frame = iter->second;
    // //         frame->pose = frame->pose * transform;
    // //     }
    // // }

    // // if (lidar_)
    // // {
    // //     for (auto pair_kf : all_kfs)
    // //     {
    // //         mapping_->AddToWorld(pair_kf.second);
    // //     }
    // // }
}

} // namespace lvio_fusion
