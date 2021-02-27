#ifndef lvio_fusion_PROBLEM_H
#define lvio_fusion_PROBLEM_H

#include "lvio_fusion/common.h"

#include <ceres/ceres.h>

namespace lvio_fusion
{

enum class ProblemType
{
    VisualError,
    LidarError,
    NavsatError,
    PoseError,
    IMUError,
    Other
};

namespace adapt
{

class Problem : public ceres::Problem
{
public:
    template <typename... Ts>
    void AddResidualBlock(
        ProblemType type,
        ceres::CostFunction *cost_function,
        ceres::LossFunction *loss_function,
        double *x0, Ts *...xs)
    {
        ceres::ResidualBlockId id = ceres::Problem::AddResidualBlock(cost_function, loss_function, x0, xs...);
        types[id] = type;
        num_types[type]++;
    }

    std::unordered_map<ceres::ResidualBlockId, ProblemType> types;
    std::map<ProblemType, int> num_types = {
        {ProblemType::VisualError, 0},
        {ProblemType::LidarError, 0},
        {ProblemType::NavsatError, 0},
        {ProblemType::PoseError, 0},
        {ProblemType::IMUError, 0},
        {ProblemType::Other, 0}};
};

inline void Solve(const ceres::Solver::Options &options,
           adapt::Problem *problem,
           ceres::Solver::Summary *summary)
{
    if (problem->num_types[ProblemType::VisualError] > 20 ||
        problem->num_types[ProblemType::LidarError] > 100)
    {
        ceres::Solve(options, problem, summary);
    }
}

} // namespace adapt
} // namespace lvio_fusion

#endif // lvio_fusion_PROBLEM_H
