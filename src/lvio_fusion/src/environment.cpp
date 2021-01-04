#include "lvio_fusion/adapt/environment.h"

namespace lvio_fusion
{

std::mutex Environment::mutex;
std::map<double, SE3d> Environment::ground_truths;
std::default_random_engine Environment::e_;
std::uniform_real_distribution<double> Environment::u_;
std::vector<Environment::Ptr> Environment::environments_;
int Environment::num_virtual_frames_ = 10;
bool Environment::initialized_=false;

void Environment::BuildProblem(adapt::Problem &problem)
{
    
}

void Environment::Step(Weights *weights, Observation* obs, double *reward, bool *done)
{
    adapt::Problem problem;
    BuildProblem(problem);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 1;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

} // namespace lvio_fusion
