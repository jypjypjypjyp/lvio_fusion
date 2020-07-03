#include <ceres/ceres.h>

#include "lvio_fusion/ceres_helper/navsat_error.hpp"
#include "lvio_fusion/components/navsat.h"
#include "lvio_fusion/map.h"

#include <fstream>

namespace lvio_fusion
{

void NavsatMap::Initialize()
{
    Map::Keyframes keyframes = map_.lock()->GetAllKeyFrames();

    // Init RT
    if (!initialized)
    {
        // std::ofstream o1("/home/jyp/Projects/test-jyp/kps.csv");
        // std::ofstream o2("/home/jyp/Projects/test-jyp/nps.csv");
        std::vector<Vector3d> pts1, pts2;
        for (auto kf_pair : keyframes)
        {
            auto kf_point = kf_pair.second->pose.inverse().translation();
            auto navsat_point = navsat_points_.lower_bound(kf_pair.first)->second.position;
            // o1 << kf_point[0] << "," << kf_point[1] << "," << kf_point[2] << std::endl;
            // o2 << neareast_point[0] << "," << neareast_point[1] << "," << neareast_point[2] << std::endl;
            pts1.push_back(kf_point);
            pts2.push_back(navsat_point);
        }
        // o1.close();
        // o2.close();
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
        Quaterniond q(R2 * R1);
        R[0] = q.w();
        R[1] = q.x();
        R[2] = q.y();
        R[3] = q.z();
    }

    initialized = false;
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();

    problem.AddParameterBlock(R, 4, local_parameterization);
    problem.AddParameterBlock(t, 3);

    // if (epoch == 3)
    // {
    //     std::ofstream o1("/home/jyp/Projects/test-jyp/kps.csv");
    //     std::ofstream o2("/home/jyp/Projects/test-jyp/As.csv");
    //     std::ofstream o3("/home/jyp/Projects/test-jyp/Bs.csv");
    //     double t1 = -1, t2 = -1;
    //     for (auto kf_pair : keyframes)
    //     {
    //         t2 = kf_pair.first;
    //         if (t1 != -1)
    //         {
    //             auto kf_point = kf_pair.second->pose.inverse().translation();
    //             auto navsat_frame = GetFrame(t1, t2);
    //             o1 << kf_point[0] << "," << kf_point[1] << "," << kf_point[2] << std::endl;
    //             o2 << navsat_frame.A[0] << "," << navsat_frame.A[1] << "," << navsat_frame.A[2] << std::endl;
    //             o3 << navsat_frame.B[0] << "," << navsat_frame.B[1] << "," << navsat_frame.B[2] << std::endl;
    //         }
    //         t1 = t2;
    //     }
    //     o1.close();
    //     o2.close();
    //     o3.close();
    // }

    double t1 = -1, t2 = -1;
    for (auto kf_pair : keyframes)
    {
        t2 = kf_pair.first;
        if (t1 != -1)
        {
            auto kf_point = kf_pair.second->pose.inverse().translation();
            auto navsat_frame = GetFrame(t1, t2);
            ceres::CostFunction *cost_function = NavsatRError::Create(kf_point, navsat_frame.A, navsat_frame.B);
            problem.AddResidualBlock(cost_function, loss_function, R, t);
        }
        t1 = t2;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 5;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    LOG(INFO) << summary.FullReport();
    LOG(INFO) << t[0] << "," << t[1] << "," << t[2];
    initialized = true;
}

} // namespace lvio_fusion
