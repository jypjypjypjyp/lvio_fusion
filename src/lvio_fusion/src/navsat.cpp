#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/navsat/navsat.h"

#include <ceres/ceres.h>


namespace lvio_fusion
{

void NavsatMap::Initialize()
{
    Frames keyframes = map_.lock()->GetAllKeyFrames();

    // initialized = false;
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    problem.AddParameterBlock(tf.data(), SE3d::num_parameters, local_parameterization);

    for (auto pair_kf : keyframes)
    {
        auto kf_point = pair_kf.second->pose.inverse().translation();
        auto pair_np = navsat_points.lower_bound(pair_kf.first);
        if (std::fabs(pair_np->first - pair_kf.first) < 1e-1)
        {
            ceres::CostFunction *cost_function = NavsatInitError::Create(kf_point, pair_np->second.position);
            problem.AddResidualBlock(cost_function, loss_function, tf.data());
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    initialized = true;
}

} // namespace lvio_fusion
