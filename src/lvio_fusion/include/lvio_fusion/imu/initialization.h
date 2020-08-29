#ifndef lvio_fusion_INITIALIZATION_H
#define lvio_fusion_INITIALIZATION_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/sensor.h"

namespace lvio_fusion
{

class Initialization
{
public:
    typedef std::shared_ptr<Initialization> Ptr;

    void AddFrame(Frame::Ptr frame)
    {
        frames_.insert(std::make_pair(frame->time, frame));
    }

    bool Initialize();

    bool Align();

    bool VisualInitialAlign();

private:
    Frames frames_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_INITIALIZATION_H
