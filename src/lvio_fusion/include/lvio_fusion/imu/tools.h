#ifndef lvio_fusion_IMU_TOOLS_H
#define lvio_fusion_IMU_TOOLS_H
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
namespace lvio_fusion
{

namespace imu
{

void RePredictVel(Frames &frames, Frame::Ptr &prior_frame);

bool InertialOptimization(Frames &frames, Matrix3d &Rwg, double prior_a, double prior_g, bool isOptRwg);

void FullBA(Frames &frames, double prior_a, double prior_g);

void RecoverBias(Frames &frames);

} // namespace imu

} // namespace lvio_fusion
#endif // lvio_fusion_IMU_TOOLS_H
