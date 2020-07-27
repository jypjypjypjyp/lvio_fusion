#include "lvio_fusion/ceres_helpers/navsat_error.hpp"
#include "lvio_fusion/map.h"
#include "lvio_fusion/sensors/navsat.h"

#include <ceres/ceres.h>


namespace lvio_fusion
{

void NavsatMap::Initialize()
{
    Map::Keyframes keyframes = map_.lock()->GetAllKeyFrames();

    // Init RT
    if (!initialized)
    {
        std::vector<Vector3d> pts1, pts2;
        for (auto kf_pair : keyframes)
        {
            auto kf_point = kf_pair.second->pose.inverse().translation();
            auto navsat_point = navsat_points.lower_bound(kf_pair.first)->second.position;
            pts1.push_back(kf_point);
            pts2.push_back(navsat_point);
        }
        Vector3d p1(0, 0, 0), p2(0, 0, 0);
        int N = pts1.size();
        for (int i = 0; i < N; i++)
        {
            p1 += pts1[i];
            p2 += pts2[i];
        }
        p1 = p1 / N;
        p2 = p2 / N;
        double a = atan2(p1[2], p1[0]);
        double b = atan2(p2[1], p2[0]);
        AngleAxisd v1(M_PI / 2, Eigen::Vector3d(1, 0, 0));
        Matrix3d R1 = v1.toRotationMatrix();
        AngleAxisd v2(b - a, Eigen::Vector3d(0, 1, 0));
        Matrix3d R2 = v2.toRotationMatrix();
        tf.setRotationMatrix(R2 * R1);
    }

    initialized = false;
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
    options.max_num_iterations = 10;
    options.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    initialized = true;
}

} // namespace lvio_fusion
