#include "lvio_fusion/loop/relocator.h"
#include "lvio_fusion/ceres/loop_error.hpp"
#include "lvio_fusion/manager.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"

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
        double end = Navsat::Num() ? Navsat::Get()->finished : backend_->finished;
        auto new_kfs = Map::Instance().GetKeyFrames(finished, end);
        if (new_kfs.empty())
            continue;
        for (auto &pair_kf : new_kfs)
        {
            Frame::Ptr frame = pair_kf.second, old_frame;
            // if last is loop and this is not loop, then correct all new loops
            if (DetectLoop(frame, old_frame, end - 30))
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

bool Relocator::DetectLoop(Frame::Ptr frame, Frame::Ptr &old_frame, double end)
{
    // the distances of 3 closest old frames is smaller than threshold
    Frames candidate_kfs = Map::Instance().GetKeyFrames(0, end);
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
        loop::LoopClosure::Ptr loop_closure = loop::LoopClosure::Ptr(new loop::LoopClosure());
        loop_closure->frame_old = old_frame;
        loop_closure->relocated = false;
        frame->loop_closure = loop_closure;
        return true;
    }
    return false;
}

bool Relocator::Relocate(Frame::Ptr frame, Frame::Ptr old_frame)
{
    frame->loop_closure->score = 0;
    // put it on the same level
    SE3d init_pose = frame->pose;
    init_pose.translation().z() = old_frame->pose.translation().z();
    frame->loop_closure->relative_o_c = old_frame->pose.inverse() * init_pose;
    // check its orientation
    double rpyxyz_o[6], rpyxyz_i[6], rpy_o_i[3];
    se32rpyxyz(frame->pose, rpyxyz_i);
    se32rpyxyz(old_frame->pose, rpyxyz_o);
    rpy_o_i[0] = rpyxyz_i[0] - rpyxyz_o[0];
    rpy_o_i[1] = rpyxyz_i[1] - rpyxyz_o[1];
    rpy_o_i[2] = rpyxyz_i[2] - rpyxyz_o[2];
    if ((mode_ == Mode::VisualOnly || mode_ == Mode::VisualAndLidar) && Vector3d(rpy_o_i[0], rpy_o_i[1], rpy_o_i[2]).norm() < 0.1)
    {
        RelocateByImage(frame, old_frame);
    }
    if ((mode_ == Mode::LidarOnly || mode_ == Mode::VisualAndLidar) && Lidar::Num() && mapping_ && frame->feature_lidar && old_frame->feature_lidar)
    {
        RelocateByPoints(frame, old_frame);
    }
    if (mode_ == Mode::None || frame->loop_closure->score > 0) // 0 is the base score
    {
        return true;
    }
    return false;
}

bool Relocator::RelocateByImage(Frame::Ptr frame, Frame::Ptr old_frame)
{
    int score = matcher_.Relocate(old_frame, frame, frame->loop_closure->relative_o_c);
    frame->loop_closure->score += score - 20;
    if (score > 0)
    {
        return true;
    }
    return false;
}

bool Relocator::RelocateByPoints(Frame::Ptr frame, Frame::Ptr old_frame)
{
    int score = mapping_->Relocate(old_frame, frame, frame->loop_closure->relative_o_c);
    frame->loop_closure->score += score - 20;
    if (score > 0)
    {
        return true;
    }
    return false;
}

void Relocator::CorrectLoop(double old_time, double start_time, double end_time)
{
    std::unique_lock<std::mutex> lock(backend_->mutex, std::defer_lock);
    // build the pose graph and submaps
    Frames active_kfs = Map::Instance().GetKeyFrames(old_time, end_time);
    Frames new_submap_kfs = Map::Instance().GetKeyFrames(start_time, end_time);
    Frames all_kfs = active_kfs;

    // update new submap frames
    SE3d old_pose = (--new_submap_kfs.end())->second->pose;
    {
        // update frames in the new submap
        double max_score = -1;
        Frame::Ptr best_frame;
        for (auto &pair_kf : new_submap_kfs)
        {
            if (Relocate(pair_kf.second, pair_kf.second->loop_closure->frame_old))
            {
                if (pair_kf.second->loop_closure->score > max_score)
                {
                    max_score = pair_kf.second->loop_closure->score;
                    best_frame = pair_kf.second;
                }
            }
        }

        if (best_frame)
        {
            lock.lock();
            Atlas active_sections = PoseGraph::Instance().GetActiveSections(active_kfs, old_time, start_time);
            Section &new_submap = PoseGraph::Instance().AddSubMap(old_time, start_time, end_time);
            adapt::Problem problem;
            PoseGraph::Instance().BuildProblem(active_sections, new_submap, problem);
            UpdateNewSubmap(best_frame, new_submap_kfs);
            PoseGraph::Instance().Optimize(active_sections, new_submap, problem);
        }
        else
        {
            for (auto &pair_kf : new_submap_kfs)
            {
                pair_kf.second->loop_closure.reset();
            }
            return;
        }
    }
    SE3d new_pose = (--new_submap_kfs.end())->second->pose;
    // forward propogate
    SE3d transform = new_pose * old_pose.inverse();
    PoseGraph::Instance().ForwardPropagate(transform, end_time + epsilon);
    // fix navsat
    if (Navsat::Num())
    {
        Navsat::Get()->fix = new_pose.translation() - old_pose.translation();
    }
    // update pointscloud
    if (Lidar::Num() && mapping_)
    {
        Frames mapping_kfs = Map::Instance().GetKeyFrames(old_time);
        for (auto &pair : mapping_kfs)
        {
            mapping_->ToWorld(pair.second);
        }
    }
}

void Relocator::UpdateNewSubmap(Frame::Ptr best_frame, Frames &new_submap_kfs)
{
    // optimize the best frame's rotation
    SE3d old_pose = best_frame->pose;
    {
        adapt::Problem problem;
        SE3d base = best_frame->pose;
        best_frame->pose = best_frame->loop_closure->frame_old->pose * best_frame->loop_closure->relative_o_c;
        SO3d r;
        double *para = r.data();
        problem.AddParameterBlock(para, 4, new ceres::EigenQuaternionParameterization());

        for (auto &pair_kf : new_submap_kfs)
        {
            ceres::CostFunction *cost_function = RelocateRError::Create(
                best_frame->pose.inverse() * pair_kf.second->loop_closure->frame_old->pose * pair_kf.second->loop_closure->relative_o_c,
                base.inverse() * pair_kf.second->pose);
            problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        best_frame->pose = best_frame->pose * SE3d(r, Vector3d::Zero());
    }
    SE3d new_pose = best_frame->pose;
    SE3d transform = new_pose * old_pose.inverse();
    for (auto &pair_kf : new_submap_kfs)
    {
        if (pair_kf.second != best_frame)
        {
            pair_kf.second->pose = transform * pair_kf.second->pose;
        }
    }
}

} // namespace lvio_fusion