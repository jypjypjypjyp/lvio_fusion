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

    bool Initialize(Frames kfs);
    
    bool initialized = false;
    int num_frames = 10;

    class Frame
    {
    public:
        imu::Preintegration::Ptr preintegration;
        Matrix3d R;
        Vector3d T;
        Vector3d Ba, Bg;
    };

private:
    bool VisualInitialAlign();

    void SolveGyroscopeBias(std::vector<Initializer::Frame> &frames);

    // void RefineGravity(VectorXd &x);

    // bool LinearAlignment(VectorXd &x);

    // bool VisualIMUAlignment(VectorXd &x);

    Vector3d g_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_INITIALIZER_H
