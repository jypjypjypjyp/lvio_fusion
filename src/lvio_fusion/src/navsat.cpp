#include "lvio_fusion/navsat/navsat.h"
#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/map.h"

#include <ceres/ceres.h>

namespace lvio_fusion
{

void NavsatMap::AddPoint(double time, double x, double y, double z)
{
    raw[time] = Vector3d(x, y, z);

    if (!initialized && Check(time, raw[time]))
    {
        Initialize();
    }

    if (initialized)
    {
        static double head = 0;
        Frames new_kfs = map_.lock()->GetKeyFrames(head);
        for (auto pair_kf : new_kfs)
        {
            auto this_iter = raw.lower_bound(pair_kf.first);
            auto last_iter = --this_iter;
            if (this_iter == raw.begin() || std::fabs(this_iter->first - pair_kf.first) > 1e-1)
                continue;

            pair_kf.second->feature_navsat = navsat::Feature::Ptr(new navsat::Feature(this_iter->first, last_iter->first, A_.first, B_.first, C_.first, this));
            head = pair_kf.first + epsilon;
        }
    }
}

Vector3d NavsatMap::GetPoint(double time)
{
    return tf * raw[time];
}

bool NavsatMap::Check(double time, Vector3d position)
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

void NavsatMap::Initialize()
{
    Frames keyframes = map_.lock()->GetAllKeyFrames();

    ceres::Problem problem;
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    problem.AddParameterBlock(tf.data(), SE3d::num_parameters, local_parameterization);

    for (auto pair_kf : keyframes)
    {
        auto position_kf = pair_kf.second->pose.translation();
        auto pair_np = raw.lower_bound(pair_kf.first);
        if (std::fabs(pair_np->first - pair_kf.first) < 1e-1)
        {
            ceres::CostFunction *cost_function = NavsatInitError::Create(position_kf, pair_np->second);
            problem.AddResidualBlock(cost_function, NULL, tf.data());
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    initialized = true;
}

} // namespace lvio_fusion
