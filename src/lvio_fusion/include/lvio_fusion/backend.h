#ifndef lvio_fusion_BACKEND_H
#define lvio_fusion_BACKEND_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/imu/imu.hpp"
#include "lvio_fusion/imu/initializer.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/camera.hpp"

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

    void SetCameras(Camera::Ptr left, Camera::Ptr right)
    {
        camera_left_ = left;
        camera_right_ = right;
    }

    void SetImu(Imu::Ptr imu) { imu_ = imu; }

    void SetMap(Map::Ptr map) { map_ = map; }

    void SetFrontend(std::shared_ptr<Frontend> frontend) { frontend_ = frontend; }

    void SetInitializer(Initializer::Ptr initializer) { initializer_ = initializer; }

    void UpdateMap();

    void Pause();

    void Continue();

    BackendStatus status = BackendStatus::RUNNING;
    std::mutex mutex;
    double head = 0;
    Initializer::Ptr initializer_;
private:
    void BackendLoop();

    void GlobalLoop();

    void Optimize(bool full = false);

    void ForwardPropagate(double time);

    void BuildProblem(Frames &active_kfs, ceres::Problem &problem,std::vector<double *> &para_gbs,std::vector<double *> &para_abs);

    Map::Ptr map_;
    std::weak_ptr<Frontend> frontend_;


    std::thread thread_;
    std::mutex running_mutex_, pausing_mutex_;
    std::condition_variable running_;
    std::condition_variable pausing_;
    std::condition_variable map_update_;
    const double delay_;

    Camera::Ptr camera_left_;
    Camera::Ptr camera_right_;
    Imu::Ptr imu_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_BACKEND_H