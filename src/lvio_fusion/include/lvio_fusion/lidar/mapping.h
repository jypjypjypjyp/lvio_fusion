#ifndef lvio_fusion_MAPPING_H
#define lvio_fusion_MAPPING_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/lidar/lidar.hpp"
#include "lvio_fusion/lidar/scan_registration.h"
#include "lvio_fusion/map.h"
#include "lvio_fusion/visual/camera.hpp"

namespace lvio_fusion
{

class Mapping
{
public:
    typedef std::shared_ptr<Mapping> Ptr;
    typedef std::weak_ptr<Mapping> WeakPtr;

    Mapping();

    void SetLidar(Lidar::Ptr lidar)
    {
        lidar_ = lidar;
    }

    void SetCamera(Camera::Ptr camera)
    {
        camera_ = camera;
    }

    void SetMap(Map::Ptr map)
    {
        map_ = map;
    }

    void SetScanRegistration(ScanRegistration::Ptr scan_registration)
    {
        scan_registration_ = scan_registration;
    }

private:
    void MappingLoop();

    void AddToWorld(const PointICloud &in, Frame::Ptr frame, PointRGBCloud &out);

    void Optimize();

    void BuildGlobalMap(Frames& active_kfs);

    std::thread thread_;
    std::mutex running_mutex_, pausing_mutex_;
    std::condition_variable running_;
    std::condition_variable pausing_;
    std::condition_variable map_update_;
    Map::Ptr map_;
    ScanRegistration::Ptr scan_registration_;
    Lidar::Ptr lidar_;
    Camera::Ptr camera_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_MAPPING_H
