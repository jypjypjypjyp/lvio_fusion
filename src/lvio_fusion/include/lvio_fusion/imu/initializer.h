#ifndef lvio_fusion_INITIALIZER_H
#define lvio_fusion_INITIALIZER_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/imu.hpp"
#include "lvio_fusion/frontend.h"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{
class Frontend;

class Initializer
{  
public:
    typedef std::shared_ptr<Initializer> Ptr;

    void  estimate_Vel_Rwg(std::vector< Frame::Ptr > Key_frames);
    void InitializeIMU( bool bFIBA);
    void SetFrontend(std::shared_ptr<Frontend>  frontend) { frontend_ = frontend; }
    std::weak_ptr<Frontend> frontend_;
    bool initialized = false;//是否初始化完成
    bool bimu=false;//是否经过imu尺度优化

    int num_frames = 10;

    Eigen::Matrix3d Rwg;       //重力方向
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    double Scale;                  //尺度
    double FirstTs;   /// 用于imu初始化第一个可用关键帧的时间

private:
    Vector3d g_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_INITIALIZER_H
