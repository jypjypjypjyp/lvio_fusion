#include "lvio_fusion/ceres/navsat_error.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/sensors/navsat.h"

#include <ceres/ceres.h>


namespace lvio_fusion
{

void NavsatMap::Initialize()
{
    Map::Keyframes keyframes = map_.lock()->GetAllKeyFrames();

    // initialized = false;
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::ProductParameterization(
        new ceres::EigenQuaternionParameterization(),
        new ceres::IdentityParameterization(3));

    problem.AddParameterBlock(tf.data(), SE3d::num_parameters, local_parameterization);

    for (auto kf_pair : keyframes)
    {
        auto kf_point = kf_pair.second->pose.inverse().translation();
        auto np_pair = navsat_points.lower_bound(kf_pair.first);
        if (std::fabs(np_pair->first - kf_pair.first) < 1e-1)
        {
            ceres::CostFunction *cost_function = NavsatInitError::Create(kf_point, np_pair->second.position);
            problem.AddResidualBlock(cost_function, loss_function, tf.data());
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    initialized = true;
}

} // namespace lvio_fusion
