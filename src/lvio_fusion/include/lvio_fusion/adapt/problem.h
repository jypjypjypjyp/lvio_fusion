#ifndef lvio_fusion_PROBLEM_H
#define lvio_fusion_PROBLEM_H

#include "lvio_fusion/common.h"

#include <ceres/ceres.h>

namespace lvio_fusion
{

enum class ProblemType
{
    VisualError,
    FarVisualError,
    LidarError,
    NavsatError,
    PoseError,
    ImuError,
    Other
};

const std::map<ProblemType, int> init_num_types = {
        {ProblemType::VisualError, 0},
        {ProblemType::FarVisualError, 0},
        {ProblemType::LidarError, 0},
        {ProblemType::NavsatError, 0},
        {ProblemType::PoseError, 0},
        {ProblemType::ImuError, 0},
        {ProblemType::Other, 0}};

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

    void AddParameterBlock(double *values, int size)
    {
        ceres::Problem::AddParameterBlock(values, size);
    }

    void AddParameterBlock(double *values,
                           int size,
                           ceres::LocalParameterization *local_parameterization)
    {
        if (size == SE3d::num_parameters)
        {
            num_frames++;
        }
        ceres::Problem::AddParameterBlock(values, size, local_parameterization);
    }

    std::map<ProblemType, int> GetTypes(double *para)
    {
        std::vector<ceres::ResidualBlockId> residual_blocks;
        GetResidualBlocksForParameterBlock(para, &residual_blocks);

        std::map<ProblemType, int> result = init_num_types;
        for (auto i : residual_blocks)
        {
            result[types[i]]++;
        }
        return result;
    }

    int num_frames = 0;
    std::unordered_map<ceres::ResidualBlockId, ProblemType> types;
    std::map<ProblemType, int> num_types = init_num_types;
};

inline void Solve(const ceres::Solver::Options &options,
                  adapt::Problem *problem,
                  ceres::Solver::Summary *summary)
{
    ceres::Solve(options, problem, summary);
}

} // namespace adapt
} // namespace lvio_fusion

#endif // lvio_fusion_PROBLEM_H
