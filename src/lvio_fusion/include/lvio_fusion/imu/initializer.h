#ifndef lvio_fusion_INITIALIZER_H
#define lvio_fusion_INITIALIZER_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/imu/imu.hpp"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{

class Initializer
{
public:
    typedef std::shared_ptr<Initializer> Ptr;

   // bool Initialize(Frames kfs);
    void InitializeIMU(float priorG, float priorA, bool bFIBA);
    void SetMap(Map::Ptr map) { map_ = map; }

    bool initialized = false;
    
    bool bInitializing=false;

    int num_frames = 10;
    double mFirstTs;   /// 用于imu初始化第一个可用关键帧的时间

   /* class Frame
    {
    public:
        imu::Preintegration::Ptr preintegration;
        Matrix3d R;
        Vector3d T;
        Vector3d Ba, Bg;
    };
*/
private:
   // bool VisualInitialAlign();

   // void SolveGyroscopeBias(std::vector<Initializer::Frame> &frames);

    // void RefineGravity(VectorXd &x);

    // bool LinearAlignment(VectorXd &x);

    // bool VisualIMUAlignment(VectorXd &x);
    Map::Ptr map_;
    Vector3d g_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_INITIALIZER_H
