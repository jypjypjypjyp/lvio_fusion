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

private:
    void MappingLoop();

    void AddToWorld(const PointICloud &in, Frame::Ptr frame, PointRGBCloud &out);

    void Optimize();

    std::thread thread_;
    Map::Ptr map_;
    ScanRegistration::Ptr scan_registration_;
    double head_ = 0;
    Lidar::Ptr lidar_;
    Camera::Ptr camera_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_MAPPING_H
