#include "lvio_fusion/ceres/loop_error.hpp"
#include "lvio_fusion/loop/pose_graph.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

void PoseGraph::AddSubMap(double old_time, double start_time, double end_time)
{
    Section new_submap;
    new_submap.C = end_time;
    new_submap.B = start_time;
    new_submap.A = old_time;
    atlas_[end_time] = new_submap;
}

/**
 * build active submaps and inner submaps
 * @param active_kfs
 * @param old_time      time of the first frame
 * @param start_time    time of the first loop frame
 * @return old frame of inner submaps; key is the first frame's time; value is the pose of the first frame
 */
std::map<double, SE3d> PoseGraph::GetActiveSubMaps(Frames &active_kfs, double &old_time, double start_time)
{

    auto start_iter = atlas_.lower_bound(old_time);
    auto end_iter = atlas_.upper_bound(start_time);
    if (start_iter != atlas_.end())
    {
        for (auto iter = start_iter; iter != end_iter; iter++)
        {
            if (iter->second.A <= old_time)
            {
                // remove outer submap
                auto new_old_iter = ++active_kfs.find(iter->first);
                old_time = new_old_iter->first;
                active_kfs.erase(active_kfs.begin(), new_old_iter);
            }
            else
            {
                // remove inner submap
                active_kfs.erase(++active_kfs.find(iter->second.A), ++active_kfs.find(iter->first));
            }
        }
    }

    std::map<double, SE3d> inner_old_frames;
    Frame::Ptr last_frame;
    for (auto pair_kf : active_kfs)
    {
        if (last_frame && last_frame->id + 1 != pair_kf.second->id)
        {
            inner_old_frames[last_frame->time] = last_frame->pose;
        }
        last_frame = pair_kf.second;
    }
    return inner_old_frames;
}

void PoseGraph::UpdateSections(double time)
{
    static double head = 0;
    Frames active_kfs = Map::Instance().GetKeyFrames(head, time);
    head = time + epsilon;

    static Frame::Ptr last_frame;
    static Vector3d last_heading(1, 0, 0);
    static bool turning = false;
    static Section current_section;
    for (auto pair_kf : active_kfs)
    {
        Vector3d heading = pair_kf.second->pose.so3() * Vector3d::UnitX();
        if (last_frame && pair_kf.second->feature_navsat)
        {
            double degree = vectors_degree_angle(last_heading, heading);
            if (!turning && degree >= 5)
            {
                if (current_section.A != 0)
                {
                    current_section.C = pair_kf.first;
                    sections_[current_section.C] = current_section;
                }
                current_section.A = pair_kf.first;
                turning = true;
            }
            else if (turning && degree < 5)
            {
                current_section.B = pair_kf.first;
                turning = false;
            }
        }
        last_frame = pair_kf.second;
        last_heading = heading;
    }
}

Atlas PoseGraph::GetSections(double start, double end)
{
    UpdateSections(end);

    auto start_iter = sections_.upper_bound(start);
    auto end_iter = sections_.upper_bound(end);
    return Atlas(start_iter, end_iter);
}

void PoseGraph::BuildProblem(Atlas &sections, adapt::Problem &problem)
{
    for (auto &pair : sections)
    {
        Frames active_kfs = Map::Instance().GetKeyFrames(pair.second.A, pair.second.B);
        ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
            new ceres::EigenQuaternionParameterization(),
            new ceres::IdentityParameterization(3));

        Frame::Ptr last_frame;
        for (auto pair_kf : active_kfs)
        {
            auto frame = pair_kf.second;
            double *para_kf = frame->pose.data();
            problem.AddParameterBlock(para_kf, SE3d::num_parameters, local_parameterization);
            if (last_frame)
            {
                double *para_last_kf = last_frame->pose.data();
                ceres::CostFunction *cost_function;
                cost_function = PoseGraphError::Create(last_frame->pose, frame->pose, frame->weights.pose_graph);
                problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para_last_kf, para_kf);
            }
            else
            {
                problem.SetParameterBlockConstant(frame->pose.data());
            }

            last_frame = frame;
        }
        problem.SetParameterBlockConstant(last_frame->pose.data());
    }
}

void PoseGraph::Optimize(Atlas &sections, adapt::Problem &problem)
{
    // ceres::Solver::Options options;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // options.num_threads = 1;
    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &problem, &summary);
    // LOG(INFO) << summary.FullReport();
}

// end_time = 0 means full forward propagate
void PoseGraph::ForwardPropagate(SE3d transfrom, double start_time)
{
    std::unique_lock<std::mutex> lock(frontend_->mutex);
    Frames forward_kfs = Map::Instance().GetKeyFrames(start_time + epsilon);
    Frame::Ptr last_frame = frontend_->last_frame;
    if (forward_kfs.find(last_frame->time) == forward_kfs.end())
    {
        forward_kfs[last_frame->time] = last_frame;
    }

    for (auto pair_kf : forward_kfs)
    {
        pair_kf.second->pose = transfrom * pair_kf.second->pose;
    }

    frontend_->UpdateCache();
}

void PoseGraph::ForwardPropagate(SE3d transfrom, const Frames &forward_kfs)
{
    for (auto pair_kf : forward_kfs)
    {
        pair_kf.second->pose = transfrom * pair_kf.second->pose;
    }
}

} // namespace lvio_fusion