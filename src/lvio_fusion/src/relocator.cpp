#include "lvio_fusion/loop/relocator.h"
#include "lvio_fusion/ceres/loop_error.hpp"
#include "lvio_fusion/manager.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"

#include <iomanip>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>

namespace lvio_fusion
{

Relocator::Relocator(int mode, double threshold)
    : mode_((Mode)mode), threshold_(threshold), matcher_(ORBMatcher(20))
{
    thread_ = std::thread(std::bind(&Relocator::DetectorLoop, this));
}

void Relocator::DetectorLoop()
{
    static double finished = 0;
    static double old_time = DBL_MAX;
    static double start_time = DBL_MAX;
    static double loop_section = DBL_MAX;
    static Frame::Ptr last_frame;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        double end = Navsat::Num() ? Navsat::Get()->finished - epsilon : backend_->finished;
        auto new_kfs = Map::Instance().GetKeyFrames(finished, end);
        if (new_kfs.empty())
            continue;
        for (auto &pair_kf : new_kfs)
        {
            Frame::Ptr frame = pair_kf.second, old_frame;
            // if last is loop and this is not loop, then correct all new loops
            if (DetectLoop(frame, old_frame))
            {
                double section = PoseGraph::Instance().GetSection(old_frame->time).A;
                if (!last_frame)
                {
                    loop_section = section;
                    start_time = pair_kf.first;
                }
                if (section == loop_section)
                {
                    old_time = std::min(old_time, old_frame->time);
                    last_frame = frame;
                }
                if (section != loop_section ||
                    (Map::Instance().end && frame == (--Map::Instance().keyframes.end())->second))
                {
                    // new old section, new loop
                    LOG(INFO) << std::setiosflags(std::ios::fixed) << std::setprecision(5) << "1Detected new loop, and correct it now. old_time:" << old_time << ";start_time:" << start_time << ";end_time:" << last_frame->time;
                    auto t1 = std::chrono::steady_clock::now();
                    CorrectLoop(old_time, start_time, last_frame->time);
                    auto t2 = std::chrono::steady_clock::now();
                    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
                    LOG(INFO) << "Correct Loop cost time: " << time_used.count() << " seconds.";
                    start_time = pair_kf.first;
                    loop_section = section;
                    old_time = old_frame->time;
                    last_frame = frame;
                }
            }
            else if (start_time != DBL_MAX)
            {
                LOG(INFO) << std::setiosflags(std::ios::fixed) << std::setprecision(5) << "2Detected new loop, and correct it now. old_time:" << old_time << ";start_time:" << start_time << ";end_time:" << last_frame->time;
                auto t1 = std::chrono::steady_clock::now();
                CorrectLoop(old_time, start_time, last_frame->time);
                auto t2 = std::chrono::steady_clock::now();
                auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
                LOG(INFO) << "Correct Loop cost time: " << time_used.count() << " seconds.";
                loop_section = start_time = old_time = DBL_MAX;
                last_frame = nullptr;
            }
        }
        finished = (--new_kfs.end())->first + epsilon;
    }
}

bool Relocator::DetectLoop(Frame::Ptr frame, Frame::Ptr &old_frame)
{
    static double finished = 0;
    static PointICloud points;
    static std::unordered_map<int, double> map;
    Frames active_kfs = Map::Instance().GetKeyFrames(finished, frame->time - 30);
    finished = frame->time - 30 + epsilon;
    for (auto pair : active_kfs)
    {
        PointI p;
        p.x = pair.second->pose.translation().x();
        p.y = pair.second->pose.translation().y();
        p.z = 0;
        p.intensity = pair.second->id;
        map[p.intensity] = pair.first;
        points.push_back(p);
    }
    if (points.empty())
        return false;
    PointI p;
    p.x = frame->pose.translation().x();
    p.y = frame->pose.translation().y();
    p.z = 0;
    std::vector<int> points_index;
    std::vector<float> points_distance;
    pcl::KdTreeFLANN<PointI> kdtree;
    kdtree.setInputCloud(boost::make_shared<PointICloud>(points));
    kdtree.nearestKSearch(p, 3, points_index, points_distance);
    // clang-format off
    double threshold = threshold_ * threshold_;
    if (points_index[0] < points.size() && points_distance[0] < threshold 
        && points_index[1] < points.size() && points_distance[1] < threshold 
        && points_index[2] < points.size() && points_distance[2] < threshold)
    // clang-format on
    {
        double time = map[points[points_index[0]].intensity];
        old_frame = Map::Instance().GetKeyFrame(time);
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
    Frames new_submap_kfs = Map::Instance().GetKeyFrames(start_time, end_time);

    // update frames
    SE3d old_pose = (--new_submap_kfs.end())->second->pose;
    {
        double max_score = -1;
        Frame::Ptr best_frame;
        for (auto &pair_kf : new_submap_kfs)
        {
            if (Relocate(pair_kf.second, pair_kf.second->loop_closure->frame_old))
            {
                if (pair_kf.second->loop_closure->score >= max_score)
                {
                    max_score = pair_kf.second->loop_closure->score;
                    best_frame = pair_kf.second;
                }
            }
        }

        if (best_frame)
        {
            lock.lock();
            Atlas active_sections = PoseGraph::Instance().FilterOldSubmaps(old_time + epsilon, start_time - 5);
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
        // Navsat::Get()->fix.z() = (new_pose.translation() - Navsat::Get()->GetAroundPoint((--new_submap_kfs.end())->first)).z();
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
    for (auto &pair_kf : new_submap_kfs)
    {
        pair_kf.second->pose.translation().z() = pair_kf.second->loop_closure->frame_old->pose.translation().z();
    }
    // optimize the best frame's rotation
    // SE3d old_pose = best_frame->pose;
    // {
    //     adapt::Problem problem;
    //     SE3d base = best_frame->pose;
    //     best_frame->pose = best_frame->loop_closure->frame_old->pose * best_frame->loop_closure->relative_o_c;
    //     SO3d r;
    //     double *para = r.data();
    //     problem.AddParameterBlock(para, 4, new ceres::EigenQuaternionParameterization());

    //     for (auto &pair_kf : new_submap_kfs)
    //     {
    //         ceres::CostFunction *cost_function = RelocateRError::Create(
    //             best_frame->pose.inverse() * pair_kf.second->loop_closure->frame_old->pose * pair_kf.second->loop_closure->relative_o_c,
    //             base.inverse() * pair_kf.second->pose);
    //         problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para);
    //     }

    //     ceres::Solver::Options options;
    //     options.linear_solver_type = ceres::DENSE_QR;
    //     ceres::Solver::Summary summary;
    //     ceres::Solve(options, &problem, &summary);
    //     best_frame->pose = best_frame->pose * SE3d(r, Vector3d::Zero());
    // }
    // SE3d new_pose = best_frame->pose;
    // SE3d transform = new_pose * old_pose.inverse();
    // for (auto &pair_kf : new_submap_kfs)
    // {
    //     if (pair_kf.second != best_frame)
    //     {
    //         pair_kf.second->pose = transform * pair_kf.second->pose;
    //     }
    // }
}

} // namespace lvio_fusion