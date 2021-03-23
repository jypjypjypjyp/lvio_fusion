#ifndef lvio_fusion_IMU_TOOLS_H
#define lvio_fusion_IMU_TOOLS_H
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
namespace lvio_fusion
{

namespace imu
{

void ReComputeBiasVel(Frames &frames, Frame::Ptr &prior_frame);// recompute bias and velocity with prior

void ReComputeBiasVel(Frames &frames);// recompute bias and velocity

void RePredictVel(Frames &frames, Frame::Ptr &prior_frame); //recompute velocity

bool InertialOptimization(Frames &key_frames, Matrix3d &Rwg, double priorG, double priorA, bool isOptRwg);//only imu optimization

void FullInertialBA(Frames &key_frames, double priorG, double priorA);//imu and visual optimization

void RecoverData(Frames active_kfs, SE3d old_pose, bool set_bias);

} // namespace imu

} // namespace lvio_fusion
#endif // lvio_fusion_IMU_TOOLS_H
