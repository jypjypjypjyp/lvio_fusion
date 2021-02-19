#ifndef lvio_fusion_IMU_TOOLS_H
#define lvio_fusion_IMU_TOOLS_H
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
namespace lvio_fusion
{

namespace imu
{
void ComputeGyroBias(const Frames &frames);

void ComputeVelocitiesAccBias(const Frames &frames);

void ReComputeBiasVel(Frames &frames, Frame::Ptr &prior_frame);

void ReComputeBiasVel(Frames &frames);

void RePredictVel(Frames &frames, Frame::Ptr &prior_frame);

bool InertialOptimization(Frames &key_frames, Matrix3d &Rwg, double priorG, double priorA, bool isOptRwg);

void FullInertialBA(Frames &key_frames, double priorG, double priorA);

void FullInertialBA(Frames &key_frames);

void RecoverData(Frames active_kfs, SE3d old_pose, bool set_bias);

} // namespace imu

} // namespace lvio_fusion
#endif // lvio_fusion_IMU_TOOLS_H
