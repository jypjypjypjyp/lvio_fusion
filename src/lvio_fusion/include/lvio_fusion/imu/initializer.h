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

    bool EstimateVelAndRwg(std::vector<Frame::Ptr> keyframes);

    bool Initialize(Frames keyframes, double priorA = 1e6, double priorG = 1e2);

    bool bimu = false;        //first init finished
    bool reinit = false;        //need re-init

    const int num_frames = 10;
private:
    Matrix3d Rwg_; //Gravity direction
};

} // namespace lvio_fusion
#endif // lvio_fusion_INITIALIZER_H
