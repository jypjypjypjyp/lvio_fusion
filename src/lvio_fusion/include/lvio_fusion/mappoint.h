#ifndef lvio_fusion_MAPPOINT_H
#define lvio_fusion_MAPPOINT_H

#include "lvio_fusion/sensor.h"
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

    Vector3d ToWorld();

    Frame::Ptr FirstFrame();
    Frame::Ptr LastFrame();

    void AddObservation(Feature::Ptr feature);

    void RemoveObservation(Feature::Ptr feature);

    static MapPoint::Ptr Create(Vector3d position, Sensor::Ptr sensor);

    unsigned long id = 0;               // ID
    Sensor::Ptr sensor = nullptr;       // observed by which sensor
    Features observations;              // only for left feature
    Feature::Ptr first_observation;     // the first observation
    Vector3d position;                  // position in the first robot coordinate
    LabelType label = LabelType::None;  // Sematic Label

};

} // namespace lvio_fusion

#endif // lvio_fusion_MAPPOINT_H
