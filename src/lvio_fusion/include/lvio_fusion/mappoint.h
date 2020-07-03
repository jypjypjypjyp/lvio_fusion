#ifndef lvio_fusion_MAPPOINT_H
#define lvio_fusion_MAPPOINT_H

#include "lvio_fusion/common.h"
#include "lvio_fusion/feature.h"
#include "lvio_fusion/frame.h"
#include "lvio_fusion/semantic/detected_object.h"

namespace lvio_fusion
{

class MapPoint
{
public:
    typedef std::shared_ptr<MapPoint> Ptr;

    MapPoint() {}

    Features GetObservations()
    {
        return observations;
    }

    Frame::Ptr FindFirstFrame();

    Frame::Ptr FindLastFrame();

    void AddObservation(Feature::Ptr feature);

    void RemoveObservation(Feature::Ptr feature);

    // factory function
    static MapPoint::Ptr CreateNewMappoint(Vector3d position);

    unsigned long id = 0; // ID
    Features observations;
    Vector3d position; // Position in world
    LabelType label = LabelType::None; // Sematic Label

};
} // namespace lvio_fusion

#endif // lvio_fusion_MAPPOINT_H
