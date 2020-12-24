#ifndef lvio_fusion_ENVIRONMENT_H
#define lvio_fusion_ENVIRONMENT_H

#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/adapt/weights.h"
#include "lvio_fusion/frame.h"

namespace lvio_fusion
{

class Environment
{
public:
    void Interact(Frame::Ptr frame, Weights *weights, double *reward);

private:
};

} // namespace lvio_fusion

#endif // lvio_fusion_ENVIRONMENT_H
