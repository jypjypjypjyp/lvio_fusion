#ifndef lvio_fusion_MAPPING_H
#define lvio_fusion_MAPPING_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/lidar/lidar.hpp"
#include "lvio_fusion/map.h"

namespace lvio_fusion
{

class Mapping
{
public:
    typedef std::shared_ptr<Mapping> Ptr;

    void SetLidar(Lidar::Ptr lidar)
    {
        lidar_ = lidar;
    }

    void SetMap(Map::Ptr map)
    {
        map_ = map;
    }

private:

    Map::Ptr map_;
    Lidar::Ptr lidar_;
};

} // namespace lvio_fusion

#endif // lvio_fusion_MAPPING_H
