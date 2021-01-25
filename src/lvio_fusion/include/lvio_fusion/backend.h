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

    Backend(double range);

    void SetFrontend(std::shared_ptr<Frontend> frontend) { frontend_ = frontend; }

    void SetMapping(Mapping::Ptr mapping) { mapping_ = mapping; }

    void SetInitializer(Initializer::Ptr initializer) { initializer_ = initializer; }

    void UpdateMap();

    void Pause();

    void Continue();
//NEWADD
// Vector3d Backend::ComputeVelocitiesAccBias(const Frames &frames);
// Vector3d Backend::ComputeGyroBias(const Frames &frames);
//NEWADDEND
    BackendStatus status = BackendStatus::RUNNING;
    std::mutex mutex;
    double head = 0;
    Initializer::Ptr initializer_;//NEWADD
    bool isInitliazing=false;//NEWADD
    bool initA=false;
    bool initB=false;
    Frame::Ptr new_frame;
    SE3d old_pose;
private:
    void BackendLoop();

    void GlobalLoop();

    void Optimize();

    void ForwardPropagate(double time);

    void BuildProblem(Frames &active_kfs, adapt::Problem &problem,bool isimu=true);

    std::weak_ptr<Frontend> frontend_;
    Mapping::Ptr mapping_;

    std::thread thread_;
    std::mutex running_mutex_, pausing_mutex_;
    std::condition_variable running_;
    std::condition_variable pausing_;
    std::condition_variable map_update_;
    const double delay_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_BACKEND_H