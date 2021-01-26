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
        return (frame->loop_closure->relative_o_c * frame->loop_closure->frame_old->pose).translation();
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
    options.num_threads = 1;
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
    auto t1 = std::chrono::steady_clock::now();
    static double finished = 0;
    // get secions
    auto sections = PoseGraph::Instance().GetSections(finished, time);
    if (sections.empty())
        return 0;

    SE3d transform;
    for (auto &pair : sections)
    {
        SE3d old_pose, new_pose;
        // optimize point A's rotation
        auto frame_A = Map::Instance().keyframes[pair.second.A];
        old_pose = frame_A->pose;
        OptimizeRP(frame_A, time);
        new_pose = frame_A->pose;
        // section propagate
        transform = new_pose * old_pose.inverse();
        PoseGraph::Instance().Propagate(transform, Map::Instance().GetKeyFrames(pair.second.A + epsilon, time));
        old_pose = frame_A->pose;
        OptimizeYaw(frame_A, pair.second.C);
        new_pose = frame_A->pose;
        // section propagate
        transform = new_pose * old_pose.inverse();
        PoseGraph::Instance().Propagate(transform, Map::Instance().GetKeyFrames(pair.second.A + epsilon, time));

        // optimize (A-C)'s X
        Frames AC_kfs = Map::Instance().GetKeyFrames(pair.second.A, pair.second.C);
        for (auto &pair_kf : AC_kfs)
        {
            auto frame = pair_kf.second;
            if (frame->feature_navsat && (frame->pose.translation() - GetFixPoint(frame)).norm() > 0.2)
            {
                old_pose = frame->pose;
                OptimizeX(frame, time);
                new_pose = frame->pose;
                // section propagate
                transform = new_pose * old_pose.inverse();
                PoseGraph::Instance().Propagate(transform, Map::Instance().GetKeyFrames(frame->time + epsilon, time));
            }
        }
        finished = pair.second.A;
    }
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "Navsat cost time: " << time_used.count() << " seconds.";

    // forward propagate
    PoseGraph::Instance().ForwardPropagate(transform, time + epsilon);
    return sections.begin()->second.A;
}

void Navsat::OptimizeRP(Frame::Ptr frame, double time)
{
    adapt::Problem problem;
    Frames active_kfs = Map::Instance().GetKeyFrames(frame->time, time);
    double rpyxyz[6] = {0};
    double *para_roll = rpyxyz;
    double *para_pitch = rpyxyz + 1;
    problem.AddParameterBlock(para_roll, 1);
    problem.AddParameterBlock(para_pitch, 1);

    for (auto &pair_kf : active_kfs)
    {
        auto position = pair_kf.second->pose.translation();
        if (pair_kf.second->feature_navsat)
        {
            ceres::CostFunction *cost_function = NavsatRPError::Create(
                GetFixPoint(pair_kf.second), frame->pose.inverse() * position, frame->pose);
            problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para_roll, para_pitch);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    frame->pose = frame->pose * rpyxyz2se3(rpyxyz);
}

void Navsat::OptimizeYaw(Frame::Ptr frame, double time)
{
    adapt::Problem problem;
    Frames active_kfs = Map::Instance().GetKeyFrames(frame->time, time);
    double rpyxyz[6] = {0};
    double *para = rpyxyz + 2;
    problem.AddParameterBlock(para, 1);

    for (auto &pair_kf : active_kfs)
    {
        auto position = pair_kf.second->pose.translation();
        if (pair_kf.second->feature_navsat)
        {
            ceres::CostFunction *cost_function = NavsatYawError::Create(
                GetFixPoint(pair_kf.second), frame->pose.inverse() * position, frame->pose);
            problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    frame->pose = frame->pose * rpyxyz2se3(rpyxyz);
}

void Navsat::OptimizeX(Frame::Ptr frame, double time)
{
    adapt::Problem problem;
    Frames active_kfs = Map::Instance().GetKeyFrames(frame->time, frame->time + 5);
    SE3d relative_pose;
    double *para = relative_pose.data() + 4;
    problem.AddParameterBlock(para, 1);

    for (auto &pair_kf : active_kfs)
    {
        auto position = pair_kf.second->pose.translation();
        if (pair_kf.second->feature_navsat)
        {
            ceres::CostFunction *cost_function = NavsatXError::Create(
                GetFixPoint(pair_kf.second), frame->pose.inverse() * position, frame->pose);
            problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    frame->pose = frame->pose * relative_pose;
}

} // namespace lvio_fusion
