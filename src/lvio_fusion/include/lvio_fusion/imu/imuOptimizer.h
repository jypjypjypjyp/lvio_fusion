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
    void static  RePredictVel(Frames &frames,Frame::Ptr &prior_frame );

    void static InertialOptimization(Frames &key_frames, Eigen::Matrix3d &Rwg,double priorG, double priorA);
    void static FullInertialBA(Frames &key_frames, double priorG, double priorA);
    void static FullInertialBA(Frames &key_frames);
    void static recoverData(Frames active_kfs,SE3d old_pose);
};

} // namespace lvio_fusion
#endif // lvio_fusion_IMUOPTIMIZER_H
