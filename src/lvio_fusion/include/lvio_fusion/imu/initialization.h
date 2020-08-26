#ifndef lvio_fusion_INITIALIZATION_H
#define lvio_fusion_INITIALIZATION_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/sensor.h"

namespace lvio_fusion
{

class Map;

class Initialization
{
public:
    typedef std::shared_ptr<Initialization> Ptr;

    void SetMap(std::shared_ptr<Map> map)
    {
        map_ = map;
    }

    // bool InitialStructure();

    // bool VisualInitialAlign();

    // bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x);

private:
    std::weak_ptr<Map> map_;
};

} // namespace lvio_fusion
#endif // lvio_fusion_INITIALIZATION_H
