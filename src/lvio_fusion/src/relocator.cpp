#include "lvio_fusion/loop/relocator.h"
#include "lvio_fusion/ceres/lidar_error.hpp"
#include "lvio_fusion/ceres/loop_error.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"
#include "lvio_fusion/visual/camera.h"
#include "lvio_fusion/visual/feature.h"
#include "lvio_fusion/visual/landmark.h"

#include <opencv2/core/eigen.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>

namespace lvio_fusion
{

Relocator::Relocator(int mode) : mode_((Mode)mode), matcher_(ORBMatcher(20))
{
    thread_ = std::thread(std::bind(&Relocator::DetectorLoop, this));
}

void Relocator::DetectorLoop()
{
    static double finished = 0;
    static double old_time = DBL_MAX;
    static double start_time = DBL_MAX;
    static Frame::Ptr last_frame;
    static Frame::Ptr last_old_frame;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto new_kfs = Map::Instance().GetKeyFrames(finished, backend_->finished);
        if (new_kfs.empty())
            continue;
        for (auto &pair_kf : new_kfs)
        {
            Frame::Ptr frame = pair_kf.second, old_frame;
            // AddKeyFrameIntoVoc(frame);
            // if last is loop and this is not loop, then correct all new loops
            if (DetectLoop(frame, old_frame))
            {
                if (!last_old_frame)
                {
                    start_time = pair_kf.first;
                }
                old_time = std::min(old_time, old_frame->time);
                last_frame = frame;
                last_old_frame = old_frame;
            }
            else if (start_time != DBL_MAX)
            {
                LOG(INFO) << "Detected new loop, and correct it now. old_time:" << old_time << ";start_time:" << start_time << ";end_time:" << last_frame->time;
                auto t1 = std::chrono::steady_clock::now();
                CorrectLoop(old_time, start_time, last_frame->time);
                auto t2 = std::chrono::steady_clock::now();
                auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
                LOG(INFO) << "Correct Loop cost time: " << time_used.count() << " seconds.";
                start_time = old_time = DBL_MAX;
                last_old_frame = last_frame = nullptr;
            }
        }
        finished = (--new_kfs.end())->first + epsilon;
    }
}

bool Relocator::DetectLoop(Frame::Ptr frame, Frame::Ptr &old_frame)
{
    Frames candidate_kfs = Map::Instance().GetKeyFrames(0, backend_->finished - 30);
    double min_distance = 10;
    for (auto &pair_kf : candidate_kfs)
    {
        Vector3d vec = (pair_kf.second->pose.translation() - frame->pose.translation());
        vec.z() = 0;
        double distance = vec.norm();
        if (distance < min_distance)
        {
            Frame::Ptr prev_frame = Map::Instance().GetKeyFrames(0, frame->time, 1).begin()->second;
            Frame::Ptr subs_frame = Map::Instance().GetKeyFrames(frame->time, 0, 1).begin()->second;
            Vector3d prev_vec = (pair_kf.second->pose.translation() - prev_frame->pose.translation());
            Vector3d subs_vec = (pair_kf.second->pose.translation() - subs_frame->pose.translation());
            prev_vec.z() = 0;
            subs_vec.z() = 0;
            double prev_distance = prev_vec.norm();
            double subs_distance = subs_vec.norm();
            if (prev_distance < min_distance && subs_distance < min_distance)
            {
                min_distance = distance;
                old_frame = pair_kf.second;
            }
        }
    }
    if (old_frame)
    {
        loop::LoopClosure::Ptr loop_constraint = loop::LoopClosure::Ptr(new loop::LoopClosure());
        loop_constraint->frame_old = old_frame;
        loop_constraint->relocated = false;
        frame->loop_closure = loop_constraint;
        return true;
    }
    return false;
}

bool Relocator::Relocate(Frame::Ptr frame, Frame::Ptr old_frame)
{
    frame->loop_closure->score = 0;
    // put it on the same level
    SE3d pose = frame->pose;
    pose.translation().z() = old_frame->pose.translation().z();
    frame->loop_closure->relative_o_c = pose * old_frame->pose.inverse();
    // check its orientation
    double rpyxyz_o[6], rpyxyz_i[6], rpy_o_i[3];
    se32rpyxyz(frame->pose, rpyxyz_i);
    se32rpyxyz(old_frame->pose, rpyxyz_o);
    rpy_o_i[0] = rpyxyz_i[0] - rpyxyz_o[0];
    rpy_o_i[1] = rpyxyz_i[1] - rpyxyz_o[1];
    rpy_o_i[2] = rpyxyz_i[2] - rpyxyz_o[2];
    if ((mode_ == Mode::Visual || mode_ == Mode::VisualAndLidar) && Vector3d(rpy_o_i[0], rpy_o_i[1], rpy_o_i[2]).norm() < 0.1)
    {
        RelocateByImage(frame, old_frame);
    }
    if ((mode_ == Mode::Lidar || mode_ == Mode::VisualAndLidar) && mapping_ && frame->feature_lidar && old_frame->feature_lidar)
    {
        RelocateByPoints(frame, old_frame);
    }
    if (mode_ == Mode::None || frame->loop_closure->score > 20)
    {
        return true;
    }
    frame->loop_closure.reset();
    return false;
}

bool Relocator::RelocateByImage(Frame::Ptr frame, Frame::Ptr old_frame)
{
    int score = matcher_.Relocate(old_frame, frame, frame->loop_closure->relative_o_c);
    if (score > 10)
    {
        frame->loop_closure->score += score;
        return true;
    }
    return false;
}

bool Relocator::RelocateByPoints(Frame::Ptr frame, Frame::Ptr old_frame)
{
    int score = mapping_->Relocate(old_frame, frame, frame->loop_closure->relative_o_c);
    if (score > 10)
    {
        frame->loop_closure->score += score;
        return true;
    }
    return false;
}

void Relocator::BuildProblem(Frames &active_kfs, adapt::Problem &problem)
{
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    Frame::Ptr last_frame;
    for (auto &pair_kf : active_kfs)
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
        last_frame = frame;
    }
}

void Relocator::BuildProblemWithRelocated(Frames &active_kfs, adapt::Problem &problem)
{
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    double start_time = active_kfs.begin()->first;

    // loop constraint
    for (auto &pair_kf : active_kfs)
    {
        auto frame = pair_kf.second;
        if (frame->loop_closure && frame->loop_closure->relocated)
        {
            double *para_kf = frame->pose.data();
            problem.SetParameterBlockConstant(para_kf);

            auto old_frame = frame->loop_closure->frame_old;
            if (old_frame->time >= start_time)
            {
                double *para_old_kf = old_frame->pose.data();
                problem.SetParameterBlockConstant(para_old_kf);
            }
        }
    }
}

void Relocator::CorrectLoop(double old_time, double start_time, double end_time)
{
    // build the pose graph and submaps
    Frames active_kfs = Map::Instance().GetKeyFrames(old_time, end_time);
    Frames new_submap_kfs = Map::Instance().GetKeyFrames(start_time, end_time);
    Frames all_kfs = active_kfs;
    Atlas active_sections = PoseGraph::Instance().GetActiveSections(active_kfs, old_time, start_time);
    Section &new_submap = PoseGraph::Instance().AddSubMap(old_time, start_time, end_time);
    adapt::Problem problem;
    PoseGraph::Instance().BuildProblem(active_sections, new_submap, problem);

    // update new submap frames
    SE3d old_pose = (--new_submap_kfs.end())->second->pose;
    {
        // build new submap pose graph
        adapt::Problem problem;
        BuildProblem(new_submap_kfs, problem);

        // relocate new submaps
        std::map<double, double> score_table;
        for (auto &pair_kf : new_submap_kfs)
        {
            Relocate(pair_kf.second, pair_kf.second->loop_closure->frame_old);
            score_table[-pair_kf.second->loop_closure->score] = pair_kf.first;
        }
        int max_num_relocated = 1;
        for (auto &pair : score_table)
        {
            if (max_num_relocated-- == 0)
                break;
            auto frame = new_submap_kfs[pair.second];
            frame->loop_closure->relocated = true;
            frame->pose = frame->loop_closure->relative_o_c * frame->loop_closure->frame_old->pose;
        }

        BuildProblemWithRelocated(new_submap_kfs, problem);
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = 1;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        for (auto &pair_kf : new_submap_kfs)
        {
            Relocate(pair_kf.second, pair_kf.second->loop_closure->frame_old);
            pair_kf.second->loop_closure->relocated = true;
            pair_kf.second->pose = pair_kf.second->loop_closure->relative_o_c * pair_kf.second->loop_closure->frame_old->pose;
        }
    }
    SE3d new_pose = (--new_submap_kfs.end())->second->pose;

    // forward propogate
    {
        std::unique_lock<std::mutex> lock1(backend_->mutex);
        SE3d transform = old_pose.inverse() * new_pose;
        PoseGraph::Instance().ForwardPropagate(transform, end_time + epsilon);
        if (mapping_)
        {
            Frames mapping_kfs = Map::Instance().GetKeyFrames(end_time + epsilon);
            for (auto &pair : mapping_kfs)
            {
                mapping_->ToWorld(pair.second);
            }
        }
    }

    PoseGraph::Instance().Optimize(active_sections, new_submap, problem);
}

} // namespace lvio_fusion