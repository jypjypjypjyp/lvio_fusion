#include "lvio_fusion/ceres/loop_error.hpp"
#include "lvio_fusion/loop/pose_graph.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"

namespace lvio_fusion
{

Section &PoseGraph::AddSubMap(double old_time, double start_time, double end_time)
{
    Section new_submap;
    new_submap.C = end_time;
    new_submap.B = start_time;
    new_submap.A = old_time;
    submaps_[end_time] = new_submap;
    return submaps_[end_time];
}

/**
 * build active submaps and inner submaps
 * @param active_kfs
 * @param old_time      time of the first frame
 * @param start_time    time of the first loop frame
 * @return old frame of inner submaps; key is the first frame's time; value is the pose of the first frame
 */
Atlas PoseGraph::GetActiveSections(Frames &active_kfs, double &old_time, double start_time)
{
    auto start_iter = submaps_.lower_bound(old_time);
    auto end_iter = submaps_.upper_bound(start_time);
    if (start_iter != submaps_.end())
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

    Atlas active_sections;
    Frame::Ptr last_frame;
    double start = 0;
    for (auto &pair_kf : active_kfs)
    {
        if (last_frame)
        {
            if (last_frame->id + 1 != pair_kf.second->id)
            {
                auto sections = GetSections(start, last_frame->time);
                for (auto &pair : sections)
                {
                    if (pair.second.C > last_frame->time)
                        break;
                    active_sections.insert(pair);
                }
                start = pair_kf.first;
            }
        }
        else
        {
            start = pair_kf.first;
        }
        last_frame = pair_kf.second;
    }
    Atlas sections = GetSections(start, start_time);
    active_sections.insert(sections.begin(), sections.end());
    return active_sections;
}

void PoseGraph::UpdateSections(double time)
{
    static double finished = 0;
    if (time <= finished)
        return;
    Frames active_kfs = Map::Instance().GetKeyFrames(finished, time);
    finished = time + epsilon;

    static Frame::Ptr last_frame;
    static Vector3d last_heading(1, 0, 0);
    static bool turning = false;
    static int A_id = 0;
    static Section current_section;
    for (auto &pair_kf : active_kfs)
    {
        Vector3d heading = pair_kf.second->pose.so3() * Vector3d::UnitX();
        if (last_frame && pair_kf.second->feature_navsat)
        {
            double degree = vectors_degree_angle(last_heading, heading);
            if (!turning && degree >= 5)
            {
                if (current_section.A != 0 && pair_kf.second->id - A_id > 10)
                {
                    current_section.C = pair_kf.first;
                    sections_[current_section.A] = current_section;
                }
                current_section.A = pair_kf.first;
                A_id = pair_kf.second->id;
                turning = true;
            }
            else if (turning)
            {
                if (degree < 1)
                {
                    current_section.B = pair_kf.first;
                    turning = false;
                }
                else if (pair_kf.second->id - A_id > 20)
                {
                    current_section.B = pair_kf.first;
                    current_section.C = pair_kf.first;
                    sections_[current_section.A] = current_section;
                    current_section.A = pair_kf.first;
                }
            }
        }
        last_frame = pair_kf.second;
        last_heading = heading;
    }
}

// [start, end]
// end = 0 -> all sections from start
Atlas PoseGraph::GetSections(double start, double end)
{
    UpdateSections(end);

    auto start_iter = sections_.upper_bound(start);
    auto end_iter = end == 0 ? sections_.end() : sections_.upper_bound(end);
    return Atlas(start_iter, end_iter);
}

void PoseGraph::BuildProblem(Atlas &sections, Section &submap, adapt::Problem &problem)
{
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    Frame::Ptr old_frame = Map::Instance().keyframes[submap.A];
    Frame::Ptr start_frame = Map::Instance().keyframes[submap.B];
    double *para_old = old_frame->pose.data(), *para_start = start_frame->pose.data();
    problem.AddParameterBlock(para_old, SE3d::num_parameters, local_parameterization);
    problem.SetParameterBlockConstant(para_old);
    problem.AddParameterBlock(para_start, SE3d::num_parameters, local_parameterization);
    problem.SetParameterBlockConstant(para_start);

    Frame::Ptr last_frame = old_frame;
    for (auto &pair : sections)
    {
        auto frame_A = Map::Instance().keyframes[pair.second.A];
        double *para = frame_A->pose.data();
        problem.AddParameterBlock(para, SE3d::num_parameters, local_parameterization);
        double *para_last_kf = last_frame->pose.data();
        ceres::CostFunction *cost_function;
        cost_function = PoseGraphError::Create(last_frame->pose, frame_A->pose, frame_A->weights.pose_graph);
        problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para_last_kf, para);
        pair.second.pose = frame_A->pose;
        last_frame = frame_A;
    }
}

void PoseGraph::Optimize(Atlas &sections, Section &submap, adapt::Problem &problem)
{
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << summary.FullReport();

    Section last_section;
    double last_time = 0;
    for (auto &pair : sections)
    {
        if (last_time)
        {
            SE3d transfrom = Map::Instance().keyframes[last_time]->pose * last_section.pose.inverse();
            Frames forward_kfs = Map::Instance().GetKeyFrames(last_time + epsilon, pair.first);
            ForwardPropagate(transfrom, forward_kfs);
        }
        last_time = pair.first;
        last_section = pair.second;
    }
    SE3d transfrom = Map::Instance().keyframes[last_time]->pose * last_section.pose.inverse();
    Frames forward_kfs = Map::Instance().GetKeyFrames(last_time + epsilon, submap.B - epsilon);
    ForwardPropagate(transfrom, forward_kfs);
}

// end_time = 0 means full forward propagate
void PoseGraph::ForwardPropagate(SE3d transfrom, double start_time)
{
    std::unique_lock<std::mutex> lock(frontend_->mutex);
    Frames forward_kfs = Map::Instance().GetKeyFrames(start_time);
    Frame::Ptr last_frame = frontend_->last_frame;
    if (forward_kfs.find(last_frame->time) == forward_kfs.end())
    {
        forward_kfs[last_frame->time] = last_frame;
    }
    ForwardPropagate(transfrom, forward_kfs);
    frontend_->UpdateCache();
}

void PoseGraph::ForwardPropagate(SE3d transfrom, const Frames &forward_kfs)
{
    for (auto &pair_kf : forward_kfs)
    {
        pair_kf.second->pose = transfrom * pair_kf.second->pose;
    }
}

} // namespace lvio_fusion