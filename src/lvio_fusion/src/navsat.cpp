#include "lvio_fusion/navsat/navsat.h"
#include "lvio_fusion/ceres/loop_error.hpp"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/utility.h"

#include <ceres/ceres.h>

namespace lvio_fusion
{

void Navsat::AddPoint(double time, double x, double y, double z)
{
    raw[time] = Vector3d(x, y, z);

    static double finished = 0;
    Frames new_kfs = Map::Instance().GetKeyFrames(finished);
    for (auto &pair_kf : new_kfs)
    {
        auto this_iter = raw.lower_bound(pair_kf.first);
        auto last_iter = this_iter;
        last_iter--;
        if (this_iter == raw.begin() || this_iter == raw.end() || std::fabs(this_iter->first - pair_kf.first) > 1)
            continue;

        double t1 = pair_kf.first - last_iter->first,
               t2 = this_iter->first - pair_kf.first;
        auto p = (this_iter->second * t1 + last_iter->second * t2) / (t1 + t2);
        raw[pair_kf.first] = p;
        pair_kf.second->feature_navsat = navsat::Feature::Ptr(new navsat::Feature(pair_kf.first));
        // check
        pair_kf.second->feature_navsat->trust = (p.z() > -3 && p.z() < 3);
        // pair_kf.second->feature_navsat->trust = true;
        finished = pair_kf.first + epsilon;
    }
    if (!initialized && !Map::Instance().keyframes.empty() && frames_distance(0, -1) > 40)
    {
        Initialize();
    }
}

Vector3d Navsat::GetRawPoint(double time)
{
    assert(raw.find(time) != raw.end());
    return raw[time];
}

Vector3d Navsat::GetFixPoint(Frame::Ptr frame)
{
    if (frame->loop_closure)
    {
        return (frame->loop_closure->frame_old->pose * frame->loop_closure->relative_o_c).translation();
    }
    if (frame->feature_navsat)
    {
        return fix + GetPoint(frame->feature_navsat->time);
    }
    else
    {
        return fix + GetAroundPoint(frame->time);
    }
}

Vector3d Navsat::GetPoint(double time)
{
    return extrinsic * GetRawPoint(time);
}

Vector3d Navsat::GetAroundPoint(double time)
{
    auto iter = raw.lower_bound(time);
    if (iter == raw.end())
        iter--;
    return extrinsic * iter->second;
}

SE3d Navsat::GetAroundPose(double time)
{
    auto iter1 = raw.lower_bound(time);
    auto iter2 = iter1--;
    if (iter2 == raw.begin())
    {
        iter1--;
        iter2--;
    }
    Vector3d p1 = extrinsic * iter1->second;
    Vector3d p2 = extrinsic * iter2->second;
    while (iter1 != raw.begin())
    {
        if ((p1 - p2).norm() > 2)
        {
            return get_pose_from_two_points(p1, p2);
        }
        else
        {
            iter1--;
            p1 = extrinsic * iter1->second;
        }
    }
    return SE3d();
}

void Navsat::Initialize()
{
    Frames keyframes = Map::Instance().keyframes;

    ceres::Problem problem;
    double para[6] = {0, 0, 0, 0, 0, 0};
    problem.AddParameterBlock(para, 1);
    problem.AddParameterBlock(para + 3, 1);
    problem.AddParameterBlock(para + 4, 1);
    problem.SetParameterBlockConstant(para + 3);
    problem.SetParameterBlockConstant(para + 4);

    for (auto &pair_kf : keyframes)
    {
        auto position = pair_kf.second->pose.translation();
        if (pair_kf.second->feature_navsat)
        {
            ceres::CostFunction *cost_function = NavsatInitError::Create(position, GetRawPoint(pair_kf.second->feature_navsat->time));
            problem.AddResidualBlock(cost_function, NULL, para, para + 3, para + 4);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    problem.SetParameterBlockVariable(para + 3);
    problem.SetParameterBlockVariable(para + 4);
    ceres::Solve(options, &problem, &summary);

    extrinsic = rpyxyz2se3(para);
    initialized = true;
}

double Navsat::Optimize(double time)
{
    // get secions
    auto sections = PoseGraph::Instance().GetSections(finished, time);

    SE3d transform;
    for (auto &pair : sections)
    {
        auto frame_A = Map::Instance().GetKeyFrame(pair.second.A);
        auto frame_B = Map::Instance().GetKeyFrame(pair.second.B);
        auto frame_C = Map::Instance().GetKeyFrame(pair.second.C);
        OptimizeZ(frame_A, time);
        // optimize A - B
        for (int i = 0; i < 2; i++)
        {
            OptimizeRX(frame_A, pair.second.C, time, 8);
            Frames AB_kfs = Map::Instance().GetKeyFrames(pair.second.A + epsilon, pair.second.B - epsilon);
            for (auto &pair_kf : AB_kfs)
            {
                auto frame = pair_kf.second;
                // OptimizeRX(frame, std::min(pair.second.B - epsilon, frame->time + 1), time, 1 + 2 + 4);
                OptimizeRX(frame, std::min(frame->time + 3, pair.second.B - epsilon), time, 2 + 4);
            }
        }
        // optimize B - C
        OptimizeZ(frame_B, time);
        OptimizeRX(frame_B, pair.second.C, time, 8);
        Frames BC_kfs = Map::Instance().GetKeyFrames(pair.second.B + epsilon, pair.second.C - epsilon);
        for (auto &pair_kf : BC_kfs)
        {
            auto frame = pair_kf.second;
            OptimizeRX(frame, std::min(frame->time + 1, time), time, 1 + 2 + 4);
        }
        finished = pair.second.C;
    }

    return sections.empty() ? 0 : sections.begin()->second.A;
}

double Navsat::QuickFix(double current_time, double end_time)
{
    Frame::Ptr current_frame = Map::Instance().GetKeyFrame(current_time);
    Frame::Ptr finished_frame = Map::Instance().GetKeyFrame(finished);
    Vector3d t = current_frame->pose.translation() - GetFixPoint(current_frame);
    current_time = current_frame->time;
    double B = PoseGraph::Instance().current_section.B;
    if (t.norm() > 2 &&
        !PoseGraph::Instance().turning &&
        frames_distance(current_time, B) > 60)
    {
        // optimize A - B
        OptimizeZ(finished_frame, end_time);
        OptimizeRX(finished_frame, current_time, end_time, 8);
        Frames AB_kfs = Map::Instance().GetKeyFrames(finished + epsilon, B - epsilon);
        for (auto &pair_kf : AB_kfs)
        {
            auto frame = pair_kf.second;
            // OptimizeRX(frame, std::min(B - epsilon, frame->time + 1), end_time, 1 + 2 + 4);
            OptimizeRX(frame, std::min(frame->time + 3, B - epsilon), end_time, 8);
        }
        // optimize B - C
        OptimizeRX(finished_frame, current_time, end_time, 8);
        Frames BC_kfs = Map::Instance().GetKeyFrames(B + epsilon, current_time - epsilon);
        for (auto &pair_kf : BC_kfs)
        {
            auto frame = pair_kf.second;
            OptimizeRX(frame, std::min(frame->time + 1, end_time), end_time, 1 + 2 + 4);
        }

        t = current_frame->pose.translation() - GetFixPoint(current_frame);
        t.z() = 0;
        if (t.norm() > 2)
        {
            PoseGraph::Instance().AddSection(current_time);
        }
        return finished;
    }
    return 0;
}

void Navsat::OptimizeRX(Frame::Ptr frame, double end, double time, int mode)
{
    if (!(mode & (1 << 1) && mode & (1 << 2)) && frames_distance(frame->time, end) < 10)
        return;
    SE3d old_pose = frame->pose;
    adapt::Problem problem;
    Frames active_kfs = Map::Instance().GetKeyFrames(frame->time, end);
    double para[6] = {0, 0, 0, 0, 0, 0};
    //NOTE: the real order of rpy is y p r
    problem.AddParameterBlock(para + 3, 1); //x
    problem.AddParameterBlock(para + 2, 1); //r
    problem.AddParameterBlock(para + 1, 1); //p
    problem.AddParameterBlock(para + 0, 1); //y
    if (frame->last_keyframe)
    {
        problem.SetParameterUpperBound(para + 3, 0, (frame->last_keyframe->pose.inverse() * frame->pose).translation().x());
        problem.SetParameterLowerBound(para + 3, 0, -(frame->last_keyframe->pose.inverse() * frame->pose).translation().x());
    }
    problem.SetParameterUpperBound(para + 2, 0, 0.1);
    problem.SetParameterLowerBound(para + 2, 0, -0.1);
    problem.SetParameterUpperBound(para + 1, 0, 0.1);
    problem.SetParameterLowerBound(para + 1, 0, -0.1);
    problem.SetParameterUpperBound(para + 0, 0, 0.5);
    problem.SetParameterLowerBound(para + 0, 0, -0.5);

    if (mode & (1 << 0))
        problem.SetParameterBlockConstant(para);
    if (mode & (1 << 1))
        problem.SetParameterBlockConstant(para + 1);
    if (mode & (1 << 2))
        problem.SetParameterBlockConstant(para + 2);
    if (mode & (1 << 3))
        problem.SetParameterBlockConstant(para + 3);

    for (auto &pair_kf : active_kfs)
    {
        auto origin = pair_kf.second->pose;
        if (pair_kf.second->feature_navsat)
        {
            Vector3d point = GetFixPoint(pair_kf.second);
            if (!pair_kf.second->feature_navsat->trust)
            {
                point.z() = frame->pose.translation().z();
            }
            ceres::CostFunction *cost_function = NavsatRXError::Create(point, frame->pose.inverse() * origin.translation(), frame->pose);
            problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para, para + 1, para + 2, para + 3);
        }
    }

    if (!(mode & (1 << 0) && mode & (1 << 1) && mode & (1 << 2)))
    {
        for (auto &pair_kf : active_kfs)
        {
            auto origin = pair_kf.second->pose;
            ceres::CostFunction *cost_function = NavsatRError::Create(origin, frame->pose);
            problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para, para + 1, para + 2);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    frame->pose = frame->pose * rpyxyz2se3(para);
    SE3d new_pose = frame->pose;
    SE3d transform = new_pose * old_pose.inverse();
    PoseGraph::Instance().Propagate(transform, Map::Instance().GetKeyFrames(frame->time + epsilon, time));
}

void Navsat::OptimizeZ(Frame::Ptr frame, double time)
{
    if (frame->feature_navsat && frame->feature_navsat->trust)
    {
        SE3d old_pose = frame->pose;
        frame->pose.translation() = GetFixPoint(frame);
        SE3d new_pose = frame->pose;
        SE3d transform = new_pose * old_pose.inverse();
        PoseGraph::Instance().Propagate(transform, Map::Instance().GetKeyFrames(frame->time + epsilon, time));
    }
}

} // namespace lvio_fusion
