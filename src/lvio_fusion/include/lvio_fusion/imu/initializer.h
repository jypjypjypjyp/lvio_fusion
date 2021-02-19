#ifndef lvio_fusion_INITIALIZER_H
#define lvio_fusion_INITIALIZER_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/imu/imu.h"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{

class Initializer
{
public:
    typedef std::shared_ptr<Initializer> Ptr;

    bool EstimateVelAndRwg(std::vector<Frame::Ptr> Key_frames);
    bool Initialize(Frames keyframes, double priorA = 1e6, double priorG = 1e2);
    void ApplyScaledRotation(const Matrix3d &R, Frames keyframes);
    void SetFrontend(std::shared_ptr<Frontend> frontend) { frontend_ = frontend; }
    std::weak_ptr<Frontend> frontend_;
    bool initialized = false; //是否初始化完成
    bool bimu = false;        //是否经过imu尺度优化
    bool reinit = false;
    int num_frames = 10;

    Eigen::Matrix3d Rwg; //重力方向

private:
    Vector3d g_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_INITIALIZER_H
