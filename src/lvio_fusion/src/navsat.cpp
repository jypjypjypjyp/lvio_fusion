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
        finished = pair_kf.first + epsilon;
    }

    if (!initialized && PoseGraph::Instance().GetSections(0, finished).size() >= 2)
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
    return extrinsic * raw.lower_bound(time)->second;
}

void Navsat::Initialize()
{
    Frames keyframes = Map::Instance().keyframes;
    {
        ceres::Problem problem;
        problem.AddParameterBlock(extrinsic.data(), 4, new ceres::EigenQuaternionParameterization());

        for (auto &pair_kf : keyframes)
        {
            auto position = pair_kf.second->pose.translation();
            if (pair_kf.second->feature_navsat)
            {
                ceres::CostFunction *cost_function = NavsatInitRError::Create(position, GetRawPoint(pair_kf.second->feature_navsat->time));
                problem.AddResidualBlock(cost_function, NULL, extrinsic.data());
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }
    {
        ceres::Problem problem;
        ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
            new ceres::EigenQuaternionParameterization(),
            new ceres::IdentityParameterization(1));

        problem.AddParameterBlock(extrinsic.data(), 5, local_parameterization);

        for (auto &pair_kf : keyframes)
        {
            auto position = pair_kf.second->pose.translation();
            if (pair_kf.second->feature_navsat)
            {
                ceres::CostFunction *cost_function = NavsatInitRXError::Create(position, GetRawPoint(pair_kf.second->feature_navsat->time));
                problem.AddResidualBlock(cost_function, NULL, extrinsic.data());
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    Optimize((--keyframes.end())->first);
    auto sections = PoseGraph::Instance().GetSections(0, 0);
    A = GetAroundPoint(sections.begin()->second.A);
    B = GetAroundPoint((--sections.end())->second.A);
    C = GetAroundPoint((--sections.end())->second.C);
    initialized = true;
}

double Navsat::Optimize(double time)
{
    // get secions
    auto sections = PoseGraph::Instance().GetSections(finished, time);
    if (sections.empty())
        return 0;

    SE3d transform;
    auto current_pose = Map::Instance().keyframes[time]->pose.translation();
    for (auto &pair : sections)
    {
        auto frame_A = Map::Instance().keyframes[pair.second.A];
        auto frame_B = Map::Instance().keyframes[pair.second.B];
        auto frame_C = Map::Instance().keyframes[pair.second.C];
        // get navsat's level
        Vector3d point_A = GetFixPoint(frame_A), point_C = GetFixPoint(frame_C);
        if (panel_height(B, point_A, point_C) > 30)
        {
            A = B;
            B = point_A;
            C = point_C;
        }
        // optimize point A and B
        OptimizeRX(frame_A, pair.second.B, time);
        OptimizeRX(frame_B, pair.second.C, time);
        // optimize (B-C)'s X
        Frames BC_kfs = Map::Instance().GetKeyFrames(pair.second.B + epsilon, pair.second.C - epsilon);
        for (auto &pair_kf : BC_kfs)
        {
            auto frame = pair_kf.second;
            OptimizeX(frame, time);
        }
        finished = pair.second.C;
    }

    auto frame_finished = Map::Instance().keyframes[finished];
    OptimizeRX(frame_finished, time, time);

    if (Map::Instance().end)
    {
        finished = (--Map::Instance().keyframes.end())->first;
    }

    return sections.begin()->second.A;
}

void Navsat::OptimizeRX(Frame::Ptr frame, double end, double time)
{
    SE3d old_pose = frame->pose;
    adapt::Problem problem;
    Frames active_kfs = Map::Instance().GetKeyFrames(frame->time, end);
    double para[6] = {0, 0, 0, 0, 0, 0};
    //NOTE: the real order of rpy is y p r
    problem.AddParameterBlock(para + 3, 1);
    problem.AddParameterBlock(para + 2, 1);
    problem.AddParameterBlock(para + 1, 1);
    problem.AddParameterBlock(para + 0, 1);

    for (auto &pair_kf : active_kfs)
    {
        auto origin = pair_kf.second->pose;
        ceres::CostFunction *cost_function = NavsatRError::Create(
            origin, frame->pose, A, B, C);
        problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para + 2, para + 1, para);

        if (pair_kf.second->feature_navsat)
        {
            ceres::CostFunction *cost_function = NavsatRXError::Create(
                GetFixPoint(pair_kf.second), frame->pose.inverse() * origin.translation(), frame->pose);
            problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para + 2, para + 1, para, para + 3);
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

void Navsat::OptimizeX(Frame::Ptr frame, double time)
{
    SE3d old_pose = frame->pose;
    adapt::Problem problem;
    Frames active_kfs = Map::Instance().GetKeyFrames(frame->time, frame->time + 1);
    double para[6] = {0, 0, 0, 0, 0, 0};
    problem.AddParameterBlock(para + 3, 1);

    for (auto &pair_kf : active_kfs)
    {
        auto position = pair_kf.second->pose.translation();
        if (pair_kf.second->feature_navsat)
        {
            ceres::CostFunction *cost_function = NavsatXError::Create(
                GetFixPoint(pair_kf.second), frame->pose.inverse() * position, frame->pose);
            problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para + 3);
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

} // namespace lvio_fusion
