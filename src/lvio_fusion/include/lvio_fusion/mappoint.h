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

    Vector3d Position();

    Frame::Ptr FindFirstFrame();

    Frame::Ptr FindLastFrame();

    void AddObservation(Feature::Ptr feature);

    void RemoveObservation(Feature::Ptr feature);

    // factory function
    static MapPoint::Ptr CreateNewMappoint(double depth, Sensord::Ptr sensor);

    unsigned long id = 0;               // ID
    Sensord::Ptr sensor = nullptr;       // observed by which sensor
    Features observations;              // only for left feature
    Feature::Ptr init_observation;      // only one initial observation
    double depth;                       // depth in the first frame
    LabelType label = LabelType::None;  // Sematic Label

};

} // namespace lvio_fusion

#endif // lvio_fusion_MAPPOINT_H
