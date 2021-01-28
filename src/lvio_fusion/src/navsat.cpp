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
        if (this_iter == raw.begin() || std::fabs(this_iter->first - pair_kf.first) > 1e-1)
            continue;

        pair_kf.second->feature_navsat = navsat::Feature::Ptr(new navsat::Feature(this_iter->first));
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
    return fix + GetPoint(frame->feature_navsat->time);
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
            ceres::CostFunction *cost_function = NavsatInitError::Create(position, GetRawPoint(pair_kf.second->feature_navsat->time));
            problem.AddResidualBlock(cost_function, NULL, extrinsic.data());
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Optimize((--keyframes.end())->first);

    // Vector3d A = Map::Instance().keyframes.begin()->second->pose.translation(),
    //          B = Map::Instance().keyframes[PoseGraph::Instance().GetSections(0, finished).begin()->second.C]->pose.translation(),
    //          C = (--Map::Instance().keyframes.end())->second->pose.translation();
    // ground_norm = (A - B).cross(A - C);
    // ground_point = A;

    initialized = true;
}

double Navsat::Optimize(double time)
{
    // get secions
    auto sections = PoseGraph::Instance().GetSections(finished, time);
    if (sections.empty())
        return 0;

    SE3d transform;
    for (auto &pair : sections)
    {
        // optimize point A's rotation
        auto frame_A = Map::Instance().keyframes[pair.second.A];
        if (time > pair.second.C + 3)
        {
            OptimizeRPY(frame_A, time);
        }
        OptimizeY(frame_A, pair.second.C, time);

        // optimize (A-C)'s X
        Frames AC_kfs = Map::Instance().GetKeyFrames(pair.second.A + epsilon, pair.second.C);
        for (auto &pair_kf : AC_kfs)
        {
            auto frame = pair_kf.second;
            if (frame->feature_navsat && (frame->pose.translation() - GetFixPoint(frame)).norm() > 0.3)
            {
                OptimizeX(frame, time);
            }
        }
        finished = pair.second.A;
    }

    return sections.begin()->second.A;
}

void Navsat::OptimizeRPY(Frame::Ptr frame, double time)
{
    SE3d old_pose = frame->pose;
    adapt::Problem problem;
    Frames active_kfs = Map::Instance().GetKeyFrames(frame->time, time);
    double para[6] = {0, 0, 0, 0, 0, 0};
    //NOTE: the real order of rpy is y p r
    problem.AddParameterBlock(para + 2, 1);
    problem.AddParameterBlock(para + 1, 1);
    problem.AddParameterBlock(para + 0, 1);

    for (auto &pair_kf : active_kfs)
    {
        auto position = pair_kf.second->pose.translation();
        if (pair_kf.second->feature_navsat)
        {
            ceres::CostFunction *cost_function = NavsatRPYError::Create(
                GetFixPoint(pair_kf.second), frame->pose.inverse() * position, frame->pose);
            problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para + 2, para + 1, para);
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

void Navsat::OptimizeY(Frame::Ptr frame, double end, double time)
{
    SE3d old_pose = frame->pose;
    adapt::Problem problem;
    Frames active_kfs = Map::Instance().GetKeyFrames(frame->time, end);
    double para[6] = {0, 0, 0, 0, 0, 0};
    problem.AddParameterBlock(para, 1);

    for (auto &pair_kf : active_kfs)
    {
        auto position = pair_kf.second->pose.translation();
        if (pair_kf.second->feature_navsat)
        {
            ceres::CostFunction *cost_function = NavsatYError::Create(
                GetFixPoint(pair_kf.second), frame->pose.inverse() * position, frame->pose);
            problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para);
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
    Frames active_kfs = Map::Instance().GetKeyFrames(frame->time, frame->time + 5);
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
