#ifndef lvio_fusion_BACKEND_H
#define lvio_fusion_BACKEND_H

#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/imu/initializer.h"
#include "lvio_fusion/lidar/mapping.h"

#include <ceres/ceres.h>

namespace lvio_fusion
{

class Frontend;

enum class BackendStatus
{
    RUNNING,
    TO_PAUSE,
    PAUSING
};

class Backend
{
public:
    typedef std::shared_ptr<Backend> Ptr;

    Backend(double window_size, bool update_weights);

    void SetFrontend(std::shared_ptr<Frontend> frontend) { frontend_ = frontend; }

    void SetMapping(Mapping::Ptr mapping) { mapping_ = mapping; }

    void SetInitializer(Initializer::Ptr initializer) { initializer_ = initializer; }

    void UpdateMap();

    void Pause();

    void Continue();

    BackendStatus status = BackendStatus::RUNNING;
    std::mutex mutex;
    double finished = 0;

private:
    void BackendLoop();

    void GlobalLoop();

    void Optimize();

    void ForwardPropagate(double time);

    void BuildProblem(Frames &active_kfs, adapt::Problem &problem);

    std::weak_ptr<Frontend> frontend_;
    Mapping::Ptr mapping_;
    Initializer::Ptr initializer_;

    std::thread thread_;
    std::mutex running_mutex_, pausing_mutex_;
    std::condition_variable running_;
    std::condition_variable pausing_;
    std::condition_variable map_update_;
    const double window_size_;
    const bool update_weights_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_BACKEND_H