#ifndef lvio_fusion_INITIALIZER_H
#define lvio_fusion_INITIALIZER_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/imu.h"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{

class Initializer
{
public:
    typedef std::shared_ptr<Initializer> Ptr;

    bool Initialize(Frames frames, double prior_a, double prior_g);

    bool finished_first_init = false;
    const int num_frames = 10;

private:
    bool EstimateVelAndRwg(Frames keyframes);
    Matrix3d Rwg_; // R of gravity in world frame
};

} // namespace lvio_fusion
#endif // lvio_fusion_INITIALIZER_H
