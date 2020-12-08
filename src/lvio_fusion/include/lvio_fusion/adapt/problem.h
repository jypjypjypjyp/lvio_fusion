#ifndef lvio_fusion_PROBLEM_H
#define lvio_fusion_PROBLEM_H

#include "lvio_fusion/common.h"

#include <ceres/ceres.h>

namespace lvio_fusion
{

enum class ProblemType
{
    PoseOnlyReprojectionError,
    TwoFrameReprojectionError,
    LidarPlaneErrorRPZ,
    LidarPlaneErrorYXY,
    NavsatError,
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
        double *x0, Ts *... xs)
    {
        ceres::ResidualBlockId id = ceres::Problem::AddResidualBlock(cost_function, loss_function, x0, xs...);
        types[id] = type;
        num_types[type]++;
    }

    std::unordered_map<ceres::ResidualBlockId, ProblemType> types;
    std::map<ProblemType, int> num_types = {
        {ProblemType::PoseOnlyReprojectionError, 0},
        {ProblemType::TwoFrameReprojectionError, 0},
        {ProblemType::LidarPlaneErrorRPZ, 0},
        {ProblemType::LidarPlaneErrorYXY, 0},
        {ProblemType::NavsatError, 0},
        {ProblemType::IMUError, 0},
        {ProblemType::Other, 0}};
};

} // namespace adapt
} // namespace lvio_fusion

#endif // lvio_fusion_PROBLEM_H
