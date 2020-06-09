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

    Vector3d &Position()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return position_;
    }

    void SetPosition(const Vector3d &position)
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        position_ = position;
    };

    Features GetObservations()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
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
    LabelType label = LabelType::None; // Sematic Label

private:
    std::mutex data_mutex_;
    Vector3d position_; // Position in world
};
} // namespace lvio_fusion

#endif // lvio_fusion_MAPPOINT_H
