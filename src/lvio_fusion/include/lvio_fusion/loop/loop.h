#ifndef lvio_fusion_LOOP_CONSTRAINT_H
#define lvio_fusion_LOOP_CONSTRAINT_H

#include "lvio_fusion/common.h"

namespace lvio_fusion
{

class Frame;

namespace loop
{

class LoopClosure
{
public:
    typedef std::shared_ptr<LoopClosure> Ptr;

    bool relocated = false;
    double score = 0;
    std::shared_ptr<Frame> frame_old;
    SE3d relative_o_c;
};

} // namespace loop
} // namespace lvio_fusion

#endif // lvio_fusion_LOOP_CONSTRAINT_H
