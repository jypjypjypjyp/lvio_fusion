#ifndef lvio_fusion_BACKEND_H
#define lvio_fusion_BACKEND_H

#include "lvio_fusion/adapt/problem.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/imu/initializer.h"
#include "lvio_fusion/lidar/mapping.h"

namespace lvio_fusion
{

class Frontend;

class Backend
{
public:
    typedef std::shared_ptr<Backend> Ptr;

    Backend(double window_size, bool update_weights);

    void SetFrontend(std::shared_ptr<Frontend> frontend) { frontend_ = frontend; }

    void SetMapping(Mapping::Ptr mapping) { mapping_ = mapping; }

    void SetInitializer(Initializer::Ptr initializer) { initializer_ = initializer; }

    void UpdateMap();

    std::mutex mutex;
    double finished = 0;

private:
    void BackendLoop();

    void GlobalLoop();

    void Optimize();

    void UpdateFrontend(SE3d transform, double time);

    double BuildProblem(Frames &active_kfs, adapt::Problem &problem);

    std::weak_ptr<Frontend> frontend_;
    Mapping::Ptr mapping_;
    Initializer::Ptr initializer_;

    std::thread thread_, thread_global_;
    std::mutex mutex_optimize_;
    std::condition_variable map_update_;
    double global_end_ = 0;
    const double window_size_;
    const bool update_weights_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_BACKEND_H