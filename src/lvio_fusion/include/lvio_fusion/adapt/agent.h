#ifndef lvio_fusion_AGENT_H
#define lvio_fusion_AGENT_H

#include "lvio_fusion/adapt/observation.h"
#include "lvio_fusion/adapt/weights.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/lidar/lidar.h"

namespace lvio_fusion
{

class Core
{
public:
    virtual void UpdateWeights(Observation &obs, Weights &weights){};
};

class Agent
{
public:
    static void SetCore(Core *core)
    {
        Agent::instance_ = new Agent(core);
    }

    static Agent *Instance()
    {
        return instance_;
    }

    void UpdateWeights(Frame::Ptr frame, Weights &weights)
    {
        Observation obs = frame->GetObservation();
        core_->UpdateWeights(obs, weights);
    }

private:
    Agent(Core *core)
    {
        core_ = core;
    }
    Agent(const Agent &);
    Agent &operator=(const Agent &);

    Core *core_;
    static Agent *instance_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_AGENT_H
