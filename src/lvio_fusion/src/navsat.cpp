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

    if (!initialized && !pose_graph_->GetSections(0, finished).empty())
    {
        Initialize();
    }
}

Vector3d Navsat::GetPoint(double time)
{
    assert(raw.find(time) != raw.end());
    return extrinsic * raw[time];
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
        new ceres::IdentityParameterization(3));

    problem.AddParameterBlock(extrinsic.data(), SE3d::num_parameters, local_parameterization);

    for (auto &pair_kf : keyframes)
    {
        auto position = pair_kf.second->pose.translation();
        if (pair_kf.second->feature_navsat)
        {
            ceres::CostFunction *cost_function = NavsatInitError::Create(position, GetPoint(pair_kf.second->feature_navsat->time));
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

double Navsat::Optimize(double time)
{
    static double finished = 0;
    // get secions
    auto sections = pose_graph_->GetSections(finished, time);
    if (sections.empty())
        return 0;

    SE3d transform;
    for (auto &pair : sections)
    {
        Frames active_kfs = Map::Instance().GetKeyFrames(pair.second.A, time);
        auto frame_A = Map::Instance().keyframes[pair.second.A];

        SE3d old_pose = frame_A->pose;
        {
            // optimize point A of sections
            adapt::Problem problem;
            ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
                new ceres::EigenQuaternionParameterization(),
                new ceres::IdentityParameterization(3));

            double *para = frame_A->pose.data();
            problem.AddParameterBlock(para, SE3d::num_parameters, local_parameterization);

            for (auto &pair_kf : active_kfs)
            {
                auto frame = pair_kf.second;
                auto position = frame->pose.translation();
                if (frame->feature_navsat)
                {
                    ceres::CostFunction *cost_function = NavsatInitError::Create(GetPoint(frame->feature_navsat->time), frame_A->pose.inverse() * position);
                    problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para);
                }
            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.num_threads = 1;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            frame_A->loop_closure = loop::LoopClosure::Ptr(new loop::LoopClosure());
            frame_A->loop_closure->frame_old = frame_A;
            frame_A->loop_closure->relocated = true;
        }
        SE3d new_pose = frame_A->pose;

        // forward propagate
        transform = new_pose * old_pose.inverse();
        pose_graph_->ForwardPropagate(transform, Map::Instance().GetKeyFrames(pair.second.A + epsilon, pair.second.C));
        finished = pair.second.A;
    }

    // forward propagate
    pose_graph_->ForwardPropagate(transform, (--sections.end())->second.C + epsilon);
    return sections.begin()->second.A;
}

} // namespace lvio_fusion
