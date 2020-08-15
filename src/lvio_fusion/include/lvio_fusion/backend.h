#ifndef lvio_fusion_BACKEND_H
#define lvio_fusion_BACKEND_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/camera.hpp"
#include "lvio_fusion/lidar/lidar.hpp"

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

    void SetLidar(Lidar::Ptr lidar)
    {
        lidar_ = lidar;
    }

    void SetMap(Map::Ptr map) { map_ = map; }

    void SetFrontend(std::shared_ptr<Frontend> frontend) { frontend_ = frontend; }

    void UpdateMap();

    void Pause();

    void Continue();

    double ActiveTime()
    {
        return head_ - range_;
    }

    BackendStatus status = BackendStatus::RUNNING;
    
private:
    void BackendLoop();

    void Optimize(bool full = false);

    void Propagate(double time);

    void BuildProblem(Keyframes &active_kfs, ceres::Problem &problem);

    Map::Ptr map_;
    std::weak_ptr<Frontend> frontend_;
    std::thread thread_;
    std::mutex running_mutex_, pausing_mutex_;
    std::condition_variable running_;
    std::condition_variable pausing_;
    std::condition_variable map_update_;
    double head_ = 0;
    const double range_;

    Camera::Ptr camera_left_ = nullptr;
    Camera::Ptr camera_right_ = nullptr;
    Lidar::Ptr lidar_ = nullptr;
};

} // namespace lvio_fusion

#endif // lvio_fusion_BACKEND_H