#ifndef lvio_fusion_INITIALIZER_H
#define lvio_fusion_INITIALIZER_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/imu.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/navigation/gridmap.h"//NAVI

namespace lvio_fusion
{

class Initializer
{
public:
    typedef std::shared_ptr<Initializer> Ptr;

    bool EstimateVelAndRwg(std::vector<Frame::Ptr> keyframes);

    bool Initialize(Frames keyframes, double priorA = 1e6, double priorG = 1e2);
    //NAVI
    void SetGridmap(Gridmap::Ptr gridmap){gridmap_ = gridmap;}

    bool bimu = false;        //是否经过imu尺度优化
    bool reinit = false;

    const int num_frames = 10;
private:
    Gridmap::Ptr gridmap_;//NAVI
    Matrix3d Rwg_; //重力方向
};

} // namespace lvio_fusion
#endif // lvio_fusion_INITIALIZER_H
