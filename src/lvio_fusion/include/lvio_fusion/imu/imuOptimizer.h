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

};

} // namespace lvio_fusion
#endif // lvio_fusion_IMUOPTIMIZER_H
