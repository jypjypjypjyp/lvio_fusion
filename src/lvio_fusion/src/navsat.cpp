#include "lvio_fusion/navsat/navsat.h"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/map.h"

#include <ceres/ceres.h>

namespace lvio_fusion
{

void Navsat::AddPoint(double time, double x, double y, double z)
{
    raw[time] = Vector3d(x, y, z);

    if (!initialized && UpdateLevel(time, raw[time]))
    {
        Initialize();
    }

    if (initialized)
    {
        static double head = 0;
        Frames new_kfs = Map::Instance().GetKeyFrames(head);
        for (auto pair_kf : new_kfs)
        {
            auto this_iter = raw.lower_bound(pair_kf.first);
            auto last_iter = --this_iter;
            if (this_iter == raw.begin() || std::fabs(this_iter->first - pair_kf.first) > 1e-1)
                continue;

            pair_kf.second->feature_navsat = navsat::Feature::Ptr(new navsat::Feature(this_iter->first, last_iter->first, A_.first, B_.first, C_.first));
            head = pair_kf.first + epsilon;
        }
    }
}

Vector3d Navsat::GetPoint(double time)
{
    return extrinsic * raw[time];
}

bool Navsat::UpdateLevel(double time, Vector3d position)
{
    auto iter = raw.lower_bound(0);

    double max_height = 0, B_time = 0;
    while ((++iter)->first < time)
    {
        auto AC = position - C_.second;
        auto AB = iter->second - C_.second;
        double height = AC.cross(AB).norm() / AC.norm();
        if (height > max_height)
        {
            max_height = height;
            B_time = iter->first;
        }
    }
    if (max_height > 20)
    {
        A_ = C_;
        B_ = std::make_pair(B_time, raw[B_time]);
        C_ = std::make_pair(time, position);
        return true;
    }
    return false;
}

void Navsat::Initialize()
{
    Frames keyframes = Map::Instance().keyframes;

    ceres::Problem problem;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    problem.AddParameterBlock(extrinsic.data(), SE3d::num_parameters, local_parameterization);

    for (auto pair_kf : keyframes)
    {
        auto position_kf = pair_kf.second->pose.translation();
        auto pair_np = raw.lower_bound(pair_kf.first);
        if (std::fabs(pair_np->first - pair_kf.first) < 1e-1)
        {
            ceres::CostFunction *cost_function = NavsatInitError::Create(position_kf, pair_np->second);
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

Atlas Navsat::Optimize(double time)
{
    static double head = 0;

    // find Atlas
    auto sections = pose_graph_->GetSections(head, time);
    head = time;
    if (sections.empty())
        return sections;

    adapt::Problem problem;
    pose_graph_->BuildProblem(sections, problem);

    // relocate turning sections
    {
        adapt::Problem problem;
        ceres::LossFunction *navsat_loss_function = new ceres::TrivialLoss();
        ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
            new ceres::EigenQuaternionParameterization(),
            new ceres::IdentityParameterization(3));

        for (auto pair : sections)
        {
            Frames section_kfs = Map::Instance().GetKeyFrames(pair.second.A, pair.second.C);

            {
                Frame::Ptr A = section_kfs[pair.second.A];
                double *para = A->pose.data();
                auto feature = A->feature_navsat;
                problem.AddParameterBlock(para, SE3d::num_parameters, local_parameterization);
                ceres::CostFunction *cost_function = NavsatError::Create(GetPoint(feature->time), GetPoint(feature->last), GetPoint(feature->A), GetPoint(feature->B), GetPoint(feature->C), A->weights.navsat);
                problem.AddResidualBlock(ProblemType::NavsatError, cost_function, navsat_loss_function, para);
                A->loop_closure = loop::LoopClosure::Ptr(new loop::LoopClosure());
                A->loop_closure->frame_old = A;
                A->loop_closure->relocated = true;
            }

            {
                Frame::Ptr B = section_kfs[pair.second.B];
                Frames BC = Frames(section_kfs.upper_bound(pair.second.B), section_kfs.find(pair.second.C));
                if (BC.size() < 5)
                    continue;
    
                double *para = B->pose.data();
                auto feature = B->feature_navsat;
                problem.AddParameterBlock(para, SE3d::num_parameters, local_parameterization);
                ceres::CostFunction *cost_function = NavsatError::Create(GetPoint(feature->time), GetPoint(feature->last), GetPoint(feature->A), GetPoint(feature->B), GetPoint(feature->C), B->weights.navsat);
                problem.AddResidualBlock(ProblemType::NavsatError, cost_function, navsat_loss_function, para);

                for (auto pair_kf : BC)
                {
                    auto frame = pair_kf.second;
                    auto position = frame->pose.translation();
                    auto feature = frame->feature_navsat;
                    ceres::CostFunction *cost_function = NavsatInitError::Create(GetPoint(feature->time), frame->pose.inverse() * position);
                    problem.AddResidualBlock(ProblemType::Other, cost_function, NULL, para);
                }

                B->loop_closure = loop::LoopClosure::Ptr(new loop::LoopClosure());
                B->loop_closure->frame_old = B;
                B->loop_closure->relocated = true;
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.num_threads = 1;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    // optimize pose graph
    pose_graph_->Optimize(sections, problem);

    return sections;
}

} // namespace lvio_fusion
