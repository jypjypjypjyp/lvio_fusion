#include "lvio_fusion/adapt/agent.h"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{
Agent *Agent::instance_ = nullptr;

Agent::Agent(Core *core) : core_(core)
{
    thread_ = std::thread(std::bind(&Agent::AgentLoop, this));
}

void Agent::AgentLoop()
{
    static double finished = 0;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto new_kfs = Map::Instance().GetKeyFrames(finished);
        if (!new_kfs.empty())
        {
            for (auto &pair_kf : new_kfs)
            {
                UpdateWeights(pair_kf.second);
            }
            finished = (--new_kfs.end())->first + epsilon;
        }
    }
}

void Agent::UpdateWeights(Frame::Ptr frame)
{
    Observation obs = frame->GetObservation();
    if (!obs.empty())
    {
        core_->UpdateWeights(obs, frame->weights);
        frame->weights.updated = true;
    }
}

} // namespace lvio_fusion
