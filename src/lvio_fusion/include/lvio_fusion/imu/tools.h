#ifndef lvio_fusion_IMU_TOOLS_H
#define lvio_fusion_IMU_TOOLS_H
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
namespace lvio_fusion
{

namespace imu
{

void ReComputeBiasVel(Frames &frames, Frame::Ptr &prior_frame);

void ReComputeBiasVel(Frames &frames);

void RePredictVel(Frames &frames, Frame::Ptr &prior_frame);

bool InertialOptimization(Frames &frames, Matrix3d &Rwg, double priorG, double priorA);

void FullInertialBA(Frames &frames, double priorG, double priorA);

void RecoverData(Frames &frames, SE3d old_pose, bool set_bias);

} // namespace imu

} // namespace lvio_fusion
#endif // lvio_fusion_IMU_TOOLS_H
