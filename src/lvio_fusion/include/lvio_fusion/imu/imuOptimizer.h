#ifndef lvio_fusion_IMUOPTIMIZER_H
#define lvio_fusion_IMUOPTIMIZER_H
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
namespace lvio_fusion
{

class Frame;
class ImuOptimizer
{
public:
    typedef std::shared_ptr<ImuOptimizer> Ptr;
    void static ComputeGyroBias(const Frames &frames);
    void static ComputeVelocitiesAccBias(const Frames &frames);
    void static ReComputeBiasVel( Frames &frames,Frame::Ptr &prior_frame );
    void static ReComputeBiasVel( Frames &frames );
    void static  RePredictVel(Frames &frames,Frame::Ptr &prior_frame );

    bool static InertialOptimization(Frames &key_frames, Eigen::Matrix3d &Rwg,double priorG, double priorA,bool isOptRwg);
    void static FullInertialBA(Frames &key_frames, double priorG, double priorA);
    void static FullInertialBA(Frames &key_frames);
    void static recoverData(Frames active_kfs,SE3d old_pose,bool bias=true);
    void static showIMUError(const double*  parameters0, const double*  parameters1, const double*  parameters2, const double*  parameters3, const double*  parameters4, const double*  parameters5, const double*  parameters6, const double*  parameters7,  imu::Preintegration::Ptr preintegration_, double time) ;
};

} // namespace lvio_fusion
#endif // lvio_fusion_IMUOPTIMIZER_H
