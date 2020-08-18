#ifndef lvio_fusion_MAPPING_H
#define lvio_fusion_MAPPING_H

#include "lvio_fusion/backend.h"
#include "lvio_fusion/common.h"
#include "lvio_fusion/lidar/lidar.hpp"
#include "lvio_fusion/map.h"

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

    void SetMap(Map::Ptr map)
    {
        map_ = map;
    }

    void SetBackend(Backend::Ptr backend)
    {
        backend_ = backend;
    }

private:
    void MappingLoop();
    
    void AddToWorld(const PointICloud &in, Frame::Ptr frame, Point3Cloud &out);

    void Optimize();

    std::thread thread_;
    Map::Ptr map_ = nullptr;
    Backend::Ptr backend_ = nullptr;
    Lidar::Ptr lidar_ = nullptr;
    double head_ = 0;
};

} // namespace lvio_fusion

#endif // lvio_fusion_MAPPING_H
